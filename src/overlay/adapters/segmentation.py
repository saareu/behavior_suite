"""S5 blob-segmentation replay adapter (C3-3).

Masks/contours are not persisted. This adapter reuses
``blob_features.segmentation.segment_frame`` with the pinned run settings and
static background. It does not invent a second segmentation algorithm, blob
identities, or contours from parquet scalars.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from blob_features.config import BlobSegmentationSettings
from blob_features.preview import (
    draw_contour_overlay,
    load_cropped_background,
    read_prepared_video_frame,
)
from blob_features.segmentation import segment_frame
from overlay.adapters.base import SourceIdentity
from overlay.context import OverlayObservation
from overlay.contracts import (
    S5_SEGMENTATION_DRAW_IMPLEMENTATION_ID,
    S5_SEGMENTATION_REPLAY_IMPLEMENTATION_ID,
    LayerId,
)
from overlay.source_s5 import segmentation_settings_fingerprint
from overlay.timing import PreparedFrameTiming
from project.file_digest import sha256_file


@dataclass
class PreparedFrameReplayCache:
    """Optional export-loop bridge so drawn pixels match the encoded frame.

    When set to the just-decoded prepared frame, the adapter segments that
    array instead of re-seeking the video (avoids seek-vs-sequential mismatch).
    """

    prepared_frame_idx: int | None = None
    frame_bgr: np.ndarray | None = None

    def set(self, prepared_frame_idx: int, frame_bgr: np.ndarray) -> None:
        self.prepared_frame_idx = int(prepared_frame_idx)
        self.frame_bgr = frame_bgr

    def clear(self) -> None:
        self.prepared_frame_idx = None
        self.frame_bgr = None

    def get(self, prepared_frame_idx: int) -> np.ndarray | None:
        if (
            self.prepared_frame_idx is None
            or self.frame_bgr is None
            or int(self.prepared_frame_idx) != int(prepared_frame_idx)
        ):
            return None
        return self.frame_bgr


class S5SegmentationReplayAdapter:
    """Replay S5 segmentation for one prepared frame at a time."""

    def __init__(
        self,
        *,
        prepared_video_path: Path | str,
        cropped_background_path: Path | str,
        segmentation_settings: BlobSegmentationSettings,
        run_id: str | None = None,
        run_dir: Path | str | None = None,
        settings_fingerprint: str | None = None,
        background_sha256: str | None = None,
        frame_cache: PreparedFrameReplayCache | None = None,
        expected_settings_fingerprint: str | None = None,
    ) -> None:
        self._layer_id = LayerId.S5_SEGMENTATION
        self._video_path = Path(prepared_video_path).expanduser().resolve()
        self._background_path = Path(cropped_background_path).expanduser().resolve()
        self._run_dir = (
            Path(run_dir).expanduser().resolve() if run_dir is not None else None
        )
        self._settings = segmentation_settings.validated()
        self._fingerprint = settings_fingerprint or segmentation_settings_fingerprint(
            self._settings
        )
        self._expected_fingerprint = expected_settings_fingerprint or self._fingerprint
        self._frame_cache = frame_cache
        self._load_error: str | None = None
        self._background: np.ndarray | None = None

        if self._fingerprint != self._expected_fingerprint:
            self._load_error = (
                "S5 settings fingerprint mismatch; refusing approximate contour "
                f"overlay (pinned={self._expected_fingerprint!r} "
                f"live={self._fingerprint!r})."
            )
        elif not self._video_path.is_file():
            self._load_error = f"Prepared video missing for S5 replay: {self._video_path}"
        elif not self._background_path.is_file():
            self._load_error = (
                f"Cropped background missing for S5 replay: {self._background_path}"
            )
        else:
            try:
                self._background = load_cropped_background(self._background_path)
            except Exception as exc:  # noqa: BLE001
                self._load_error = f"Failed to load cropped background: {exc}"
                self._background = None

        self._identity = SourceIdentity(
            subsystem="blob_features",
            run_id=run_id,
            run_dir=self._run_dir,
            artifact_path=self._background_path,
            digest_sha256=background_sha256
            or (
                sha256_file(self._background_path)
                if self._background_path.is_file()
                else None
            ),
            extra={
                "legend": "S5 segmentation replay QC",
                "domain": "prepared",
                "geometry_source": "s5_segmentation_replay",
                "segmentation_implementation_id": S5_SEGMENTATION_REPLAY_IMPLEMENTATION_ID,
                "draw_implementation_id": S5_SEGMENTATION_DRAW_IMPLEMENTATION_ID,
                "settings_fingerprint": self._fingerprint,
                "no_invented_blob_identities": True,
            },
        )

    @property
    def layer_id(self) -> LayerId:
        return self._layer_id

    @property
    def source_identity(self) -> SourceIdentity:
        return self._identity

    @property
    def available(self) -> bool:
        return self._load_error is None and self._background is not None

    def observe(
        self,
        timing: PreparedFrameTiming,
        *,
        prepared_frame_idx: int,
    ) -> OverlayObservation:
        _ = timing
        prep = int(prepared_frame_idx)
        if not self.available or self._background is None:
            return OverlayObservation(
                layer_id=self._layer_id,
                available=False,
                reason=self._load_error or "S5 segmentation replay unavailable.",
                payload={
                    "legend": "S5 segmentation replay QC (unavailable)",
                    "settings_fingerprint": self._fingerprint,
                },
            )

        try:
            frame = None
            if self._frame_cache is not None:
                frame = self._frame_cache.get(prep)
            if frame is None:
                frame = read_prepared_video_frame(self._video_path, prep)
            result = segment_frame(frame, self._background, self._settings)
        except Exception as exc:  # noqa: BLE001
            return OverlayObservation(
                layer_id=self._layer_id,
                available=False,
                reason=f"S5 segmentation replay failed at prepared_frame_idx={prep}: {exc}",
                payload={
                    "legend": "S5 segmentation replay QC (unavailable)",
                    "prepared_frame_idx": prep,
                    "settings_fingerprint": self._fingerprint,
                },
            )

        # Contours are OpenCV arrays; store copies for drawer safety.
        contours = tuple(np.asarray(c).copy() for c in result.contours)
        mask = np.asarray(result.mask).copy()
        return OverlayObservation(
            layer_id=self._layer_id,
            available=True,
            reason=None,
            payload={
                "legend": "S5 segmentation replay QC",
                "prepared_frame_idx": prep,
                "contours": contours,
                "mask": mask,
                "cc_count_raw": int(result.cc_count_raw),
                "cc_count": float(result.cc_count),
                "min_inter_contour_dist": float(result.min_inter_contour_dist),
                "settings_fingerprint": self._fingerprint,
                "segmentation_implementation_id": S5_SEGMENTATION_REPLAY_IMPLEMENTATION_ID,
                "draw_implementation_id": S5_SEGMENTATION_DRAW_IMPLEMENTATION_ID,
                "no_invented_blob_identities": True,
                "geometry_source": "s5_segmentation_replay",
            },
        )


def make_s5_segmentation_adapter(
    *,
    prepared_video_path: Path | str,
    cropped_background_path: Path | str,
    segmentation_settings: BlobSegmentationSettings,
    run_id: str | None = None,
    run_dir: Path | str | None = None,
    settings_fingerprint: str | None = None,
    background_sha256: str | None = None,
    frame_cache: PreparedFrameReplayCache | None = None,
    expected_settings_fingerprint: str | None = None,
) -> S5SegmentationReplayAdapter:
    """Factory for the C3-3 S5 segmentation replay adapter."""

    return S5SegmentationReplayAdapter(
        prepared_video_path=prepared_video_path,
        cropped_background_path=cropped_background_path,
        segmentation_settings=segmentation_settings,
        run_id=run_id,
        run_dir=run_dir,
        settings_fingerprint=settings_fingerprint,
        background_sha256=background_sha256,
        frame_cache=frame_cache,
        expected_settings_fingerprint=expected_settings_fingerprint,
    )


def contours_match_preview(
    *,
    frame_bgr: np.ndarray,
    background: np.ndarray,
    settings: BlobSegmentationSettings,
    prepared_frame_idx: int,
    contours: tuple[np.ndarray, ...],
) -> bool:
    """True when adapter contours match ``segment_frame`` for the same inputs."""

    _ = prepared_frame_idx
    expected = segment_frame(frame_bgr, background, settings.validated())
    if len(expected.contours) != len(contours):
        return False
    for left, right in zip(expected.contours, contours, strict=True):
        if not np.array_equal(np.asarray(left), np.asarray(right)):
            return False
    return True


def preview_overlay_bgr(
    frame_bgr: np.ndarray,
    contours: tuple[np.ndarray, ...],
    *,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
) -> np.ndarray:
    """Thin wrapper so tests can compare against S5 preview draw helper."""

    return draw_contour_overlay(frame_bgr, contours, color=color, thickness=thickness)


__all__ = [
    "PreparedFrameReplayCache",
    "S5SegmentationReplayAdapter",
    "contours_match_preview",
    "make_s5_segmentation_adapter",
    "preview_overlay_bgr",
]
