"""Read-only prepared-video frame access for behavioral inspection.

Reuses verified S1 ``read_raw_frame_at_index``. Suitable for asynchronous GUI
workers: decoding is explicit and never implied on the Qt main thread.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from behavioral_review.contracts import VideoAccessError
from preprocess.exceptions import VideoProbeError
from preprocess.video_probe import read_raw_frame_at_index

FrameReader = Callable[[Path, int], np.ndarray]


@dataclass(frozen=True, slots=True)
class DecodedFrame:
    """One prepared-video frame with provenance for the playhead."""

    prepared_frame_idx: int
    video_path: Path
    image_bgr: np.ndarray
    from_cache: bool


class PreparedFrameAccessor:
    """Frame-index seek API with a small LRU cache (no full-video load)."""

    def __init__(
        self,
        video_path: Path | str,
        *,
        cache_size: int = 8,
        frame_reader: FrameReader | None = None,
    ) -> None:
        path = Path(video_path).expanduser().resolve()
        if not path.is_file():
            raise VideoAccessError(f"Prepared video unavailable: {path}")
        if (
            isinstance(cache_size, bool)
            or not isinstance(cache_size, int)
            or cache_size < 0
        ):
            raise VideoAccessError("cache_size must be a non-negative integer.")
        self._video_path = path
        self._cache_size = int(cache_size)
        self._frame_reader: FrameReader = frame_reader or _default_frame_reader
        self._cache: OrderedDict[int, np.ndarray] = OrderedDict()
        self._lock = threading.RLock()
        self._last_prepared_frame_idx: int | None = None

    @property
    def video_path(self) -> Path:
        return self._video_path

    @property
    def cache_size(self) -> int:
        return self._cache_size

    @property
    def last_prepared_frame_idx(self) -> int | None:
        with self._lock:
            return self._last_prepared_frame_idx

    def read_frame(self, prepared_frame_idx: int) -> DecodedFrame:
        """Seek to ``prepared_frame_idx`` (0-based prepared decode order)."""

        idx = _require_frame_index(prepared_frame_idx)
        with self._lock:
            cached = self._cache.get(idx)
            if cached is not None:
                self._cache.move_to_end(idx)
                self._last_prepared_frame_idx = idx
                return DecodedFrame(
                    prepared_frame_idx=idx,
                    video_path=self._video_path,
                    image_bgr=cached,
                    from_cache=True,
                )

        try:
            frame = self._frame_reader(self._video_path, idx)
        except VideoProbeError as exc:
            raise VideoAccessError(
                f"Failed to decode prepared_frame_idx={idx} from "
                f"{self._video_path}: {exc}"
            ) from exc
        except Exception as exc:  # noqa: BLE001 - wrap decoder failures
            raise VideoAccessError(
                f"Decoder failure at prepared_frame_idx={idx} from "
                f"{self._video_path}: {exc}"
            ) from exc

        if not isinstance(frame, np.ndarray) or frame.ndim < 2:
            raise VideoAccessError(
                f"Decoder returned invalid frame for prepared_frame_idx={idx}."
            )
        # Store a defensive copy so callers mutating the array cannot poison cache.
        stored = np.asarray(frame).copy()
        with self._lock:
            if self._cache_size > 0:
                self._cache[idx] = stored
                self._cache.move_to_end(idx)
                while len(self._cache) > self._cache_size:
                    self._cache.popitem(last=False)
            self._last_prepared_frame_idx = idx
            return DecodedFrame(
                prepared_frame_idx=idx,
                video_path=self._video_path,
                image_bgr=stored,
                from_cache=False,
            )

    def read_frame_if_changed(
        self,
        prepared_frame_idx: int,
        *,
        previous_prepared_frame_idx: int | None,
    ) -> DecodedFrame | None:
        """Return a frame only when the prepared index changes (plateau policy)."""

        idx = _require_frame_index(prepared_frame_idx)
        if (
            previous_prepared_frame_idx is not None
            and int(previous_prepared_frame_idx) == idx
        ):
            return None
        return self.read_frame(idx)

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    def cache_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "cache_size": self._cache_size,
                "cached_frames": len(self._cache),
                "cached_indices": list(self._cache.keys()),
                "last_prepared_frame_idx": self._last_prepared_frame_idx,
            }


def _default_frame_reader(video_path: Path, prepared_frame_idx: int) -> np.ndarray:
    return read_raw_frame_at_index(video_path, prepared_frame_idx)


def _require_frame_index(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise VideoAccessError(
            f"prepared_frame_idx must be a non-negative integer; got {value!r}."
        )
    idx = int(value)
    if idx < 0:
        raise VideoAccessError(
            f"prepared_frame_idx must be a non-negative integer; got {idx}."
        )
    return idx
