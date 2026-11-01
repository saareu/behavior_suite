"""Bounded lazy gallery preview decode (S10 Gallery A1).

Uses SyncMap + PreparedFrameAccessor. Playback is visibility-driven: only
visible cards decode, concurrency is capped, queues are bounded, and source
switches invalidate in-flight work via generation tokens. Never writes science
or journal artifacts and never eagerly encodes GIF/MP4.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np

from behavioral_review.contracts import (
    GALLERY_DEFAULT_MAX_CONCURRENT_PREVIEWS,
    GALLERY_DEFAULT_MAX_QUEUED_PREVIEW_REQUESTS,
    InspectionError,
    VideoAccessError,
)
from behavioral_review.gallery import GalleryCard, GalleryPreviewPlan
from behavioral_review.video import FrameReader, PreparedFrameAccessor

PreviewCallback = Callable[["GalleryPreviewFrame"], None]


@dataclass(frozen=True, slots=True)
class GalleryPreviewRequest:
    """One decode request with stale-rejection identity."""

    request_id: str
    generation: int
    card_id: str
    bout_id: str
    prepared_frame_idx: int
    source_token: str


@dataclass(frozen=True, slots=True)
class GalleryPreviewFrame:
    """Decoded preview frame delivered to GUI / tests."""

    request: GalleryPreviewRequest
    image_bgr: np.ndarray | None
    from_cache: bool
    skipped_plateau: bool
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.image_bgr is not None


@dataclass
class _CardPlaybackState:
    card_id: str
    bout_id: str
    plan: GalleryPreviewPlan
    visible: bool = False
    cursor_prepared_idx: int | None = None
    last_emitted_prepared_idx: int | None = None
    in_flight: bool = False


class GalleryPreviewScheduler:
    """Visibility-driven looping preview decode with bounded workers/queues."""

    def __init__(
        self,
        video_path: Path | str,
        *,
        max_concurrent: int = GALLERY_DEFAULT_MAX_CONCURRENT_PREVIEWS,
        max_queued: int = GALLERY_DEFAULT_MAX_QUEUED_PREVIEW_REQUESTS,
        cache_size: int = 32,
        frame_reader: FrameReader | None = None,
        source_token: str | None = None,
    ) -> None:
        if (
            isinstance(max_concurrent, bool)
            or not isinstance(max_concurrent, int)
            or max_concurrent < 1
        ):
            raise InspectionError("max_concurrent must be a positive integer.")
        if (
            isinstance(max_queued, bool)
            or not isinstance(max_queued, int)
            or max_queued < 1
        ):
            raise InspectionError("max_queued must be a positive integer.")
        self._video_path = Path(video_path).expanduser().resolve()
        self._max_concurrent = int(max_concurrent)
        self._max_queued = int(max_queued)
        self._accessor = PreparedFrameAccessor(
            self._video_path,
            cache_size=int(cache_size),
            frame_reader=frame_reader,
        )
        self._source_token = source_token or str(uuid4())
        self._generation = 1
        self._lock = threading.RLock()
        self._cards: dict[str, _CardPlaybackState] = {}
        self._visible_order: list[str] = []
        self._pending: deque[str] = deque()
        self._pending_set: set[str] = set()
        self._dropped_queue_count = 0
        self._completed_count = 0
        self._rejected_stale_count = 0
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_concurrent,
            thread_name_prefix="gallery-preview",
        )
        self._futures: dict[str, Future[GalleryPreviewFrame]] = {}
        self._callback: PreviewCallback | None = None
        self._closed = False

    @property
    def video_path(self) -> Path:
        return self._video_path

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    @property
    def source_token(self) -> str:
        return self._source_token

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    @property
    def max_queued(self) -> int:
        return self._max_queued

    def set_callback(self, callback: PreviewCallback | None) -> None:
        with self._lock:
            self._callback = callback

    def register_cards(self, cards: Sequence[GalleryCard]) -> None:
        """Install preview plans for cards (lazy — no decode yet)."""

        with self._lock:
            self._assert_open()
            keep = {c.card_id for c in cards}
            for card_id in list(self._cards):
                if card_id not in keep:
                    self._drop_card_locked(card_id)
            for card in cards:
                existing = self._cards.get(card.card_id)
                if existing is None:
                    self._cards[card.card_id] = _CardPlaybackState(
                        card_id=card.card_id,
                        bout_id=card.bout_id,
                        plan=card.preview_plan,
                        cursor_prepared_idx=card.preview_plan.first_visible_prepared_frame_idx,
                    )
                else:
                    existing.plan = card.preview_plan
                    existing.bout_id = card.bout_id
                    if existing.cursor_prepared_idx is None:
                        existing.cursor_prepared_idx = (
                            card.preview_plan.first_visible_prepared_frame_idx
                        )

    def set_visible_card_ids(self, card_ids: Sequence[str]) -> None:
        """Mark visible cards; release resources for cards that leave view."""

        with self._lock:
            self._assert_open()
            wanted = [str(cid) for cid in card_ids if str(cid) in self._cards]
            wanted_set = set(wanted)
            self._visible_order = wanted
            for card_id, state in self._cards.items():
                was_visible = state.visible
                state.visible = card_id in wanted_set
                if was_visible and not state.visible:
                    state.in_flight = False
                    state.last_emitted_prepared_idx = None
                    fut = self._futures.pop(card_id, None)
                    if fut is not None:
                        fut.cancel()
                    self._pending_set.discard(card_id)
            self._pending.clear()
            self._pending_set.clear()
            for card_id in wanted:
                if not self._cards[card_id].in_flight:
                    self._enqueue_locked(card_id)
            self._pump_locked()

    def tick(self) -> list[GalleryPreviewFrame]:
        """Advance visible looping cursors and schedule bounded decode work."""

        with self._lock:
            self._assert_open()
            for card_id in self._visible_order:
                state = self._cards.get(card_id)
                if state is None or not state.visible:
                    continue
                if state.in_flight:
                    continue
                state.cursor_prepared_idx = self._next_prepared_idx(state)
                self._enqueue_locked(card_id)
            self._pump_locked()
        return self.drain_completed()

    def drain_completed(self) -> list[GalleryPreviewFrame]:
        """Collect finished futures; reject stale generation/source tokens."""

        done: list[tuple[str, Future[GalleryPreviewFrame]]] = []
        with self._lock:
            for card_id, fut in list(self._futures.items()):
                if fut.done():
                    done.append((card_id, fut))
                    self._futures.pop(card_id, None)
        frames: list[GalleryPreviewFrame] = []
        with self._lock:
            callback = self._callback
            generation = self._generation
            source_token = self._source_token
        for card_id, fut in done:
            try:
                frame = fut.result()
            except Exception as exc:  # noqa: BLE001 - surface as preview error
                with self._lock:
                    state = self._cards.get(card_id)
                    if state is not None:
                        state.in_flight = False
                    self._rejected_stale_count += 1
                frame = GalleryPreviewFrame(
                    request=GalleryPreviewRequest(
                        request_id=str(uuid4()),
                        generation=generation,
                        card_id=card_id,
                        bout_id=card_id,
                        prepared_frame_idx=-1,
                        source_token=source_token,
                    ),
                    image_bgr=None,
                    from_cache=False,
                    skipped_plateau=False,
                    error=str(exc),
                )
            accepted = self._accept_frame(frame)
            if accepted is not None:
                frames.append(accepted)
                if callback is not None:
                    callback(accepted)
            with self._lock:
                state = self._cards.get(card_id)
                if state is not None:
                    state.in_flight = False
                    if state.visible:
                        self._enqueue_locked(card_id)
                self._pump_locked()
        return frames

    def invalidate(self, *, new_source_token: str | None = None) -> int:
        """Bump generation, clear queues/caches, and drop in-flight work."""

        with self._lock:
            self._assert_open()
            self._generation += 1
            self._source_token = (
                str(new_source_token) if new_source_token is not None else str(uuid4())
            )
            for fut in self._futures.values():
                fut.cancel()
            self._futures.clear()
            self._pending.clear()
            self._pending_set.clear()
            for state in self._cards.values():
                state.in_flight = False
                state.visible = False
                state.last_emitted_prepared_idx = None
                state.cursor_prepared_idx = state.plan.first_visible_prepared_frame_idx
            self._visible_order = []
            self._accessor.clear_cache()
            return self._generation

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for fut in self._futures.values():
                fut.cancel()
            self._futures.clear()
            self._pending.clear()
            self._pending_set.clear()
            self._cards.clear()
            self._visible_order = []
            self._callback = None
            self._accessor.clear_cache()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "generation": self._generation,
                "source_token": self._source_token,
                "max_concurrent": self._max_concurrent,
                "max_queued": self._max_queued,
                "n_registered_cards": len(self._cards),
                "n_visible": sum(1 for s in self._cards.values() if s.visible),
                "n_pending": len(self._pending),
                "n_in_flight": len(self._futures),
                "dropped_queue_count": self._dropped_queue_count,
                "completed_count": self._completed_count,
                "rejected_stale_count": self._rejected_stale_count,
                "cache": self._accessor.cache_stats(),
            }

    def _accept_frame(self, frame: GalleryPreviewFrame) -> GalleryPreviewFrame | None:
        with self._lock:
            if frame.request.generation != self._generation:
                self._rejected_stale_count += 1
                return None
            if frame.request.source_token != self._source_token:
                self._rejected_stale_count += 1
                return None
            state = self._cards.get(frame.request.card_id)
            if state is None or not state.visible:
                self._rejected_stale_count += 1
                return None
            if frame.skipped_plateau:
                self._completed_count += 1
                return frame
            if frame.image_bgr is None and frame.error is None:
                self._rejected_stale_count += 1
                return None
            state.last_emitted_prepared_idx = frame.request.prepared_frame_idx
            self._completed_count += 1
            return frame

    def _enqueue_locked(self, card_id: str) -> None:
        if card_id in self._pending_set or card_id in self._futures:
            return
        state = self._cards.get(card_id)
        if state is None or not state.visible:
            return
        if len(self._pending) >= self._max_queued:
            dropped = self._pending.popleft()
            self._pending_set.discard(dropped)
            self._dropped_queue_count += 1
        self._pending.append(card_id)
        self._pending_set.add(card_id)

    def _pump_locked(self) -> None:
        while len(self._futures) < self._max_concurrent and self._pending:
            card_id = self._pending.popleft()
            self._pending_set.discard(card_id)
            state = self._cards.get(card_id)
            if state is None or not state.visible or state.in_flight:
                continue
            prepared_idx = int(
                state.cursor_prepared_idx
                if state.cursor_prepared_idx is not None
                else state.plan.first_visible_prepared_frame_idx
            )
            request = GalleryPreviewRequest(
                request_id=str(uuid4()),
                generation=self._generation,
                card_id=card_id,
                bout_id=state.bout_id,
                prepared_frame_idx=prepared_idx,
                source_token=self._source_token,
            )
            previous = state.last_emitted_prepared_idx
            state.in_flight = True
            fut = self._executor.submit(self._decode_one, request, previous)
            self._futures[card_id] = fut

    def _decode_one(
        self,
        request: GalleryPreviewRequest,
        previous_prepared_frame_idx: int | None,
    ) -> GalleryPreviewFrame:
        with self._lock:
            if self._closed:
                return GalleryPreviewFrame(
                    request=request,
                    image_bgr=None,
                    from_cache=False,
                    skipped_plateau=False,
                    error="scheduler_closed",
                )
            generation = self._generation
            source_token = self._source_token
        if (
            request.generation != generation
            or request.source_token != source_token
        ):
            return GalleryPreviewFrame(
                request=request,
                image_bgr=None,
                from_cache=False,
                skipped_plateau=False,
                error="stale_request",
            )
        try:
            changed = self._accessor.read_frame_if_changed(
                request.prepared_frame_idx,
                previous_prepared_frame_idx=previous_prepared_frame_idx,
            )
            if changed is None:
                return GalleryPreviewFrame(
                    request=request,
                    image_bgr=None,
                    from_cache=True,
                    skipped_plateau=True,
                    error=None,
                )
            return GalleryPreviewFrame(
                request=request,
                image_bgr=changed.image_bgr,
                from_cache=changed.from_cache,
                skipped_plateau=False,
                error=None,
            )
        except (VideoAccessError, Exception) as exc:  # noqa: BLE001
            return GalleryPreviewFrame(
                request=request,
                image_bgr=None,
                from_cache=False,
                skipped_plateau=False,
                error=str(exc),
            )

    def _next_prepared_idx(self, state: _CardPlaybackState) -> int:
        plan = state.plan
        cur = state.cursor_prepared_idx
        if cur is None:
            return plan.first_visible_prepared_frame_idx
        nxt = int(cur) + 1
        if nxt > plan.last_visible_prepared_frame_idx:
            return plan.first_visible_prepared_frame_idx
        return nxt

    def _drop_card_locked(self, card_id: str) -> None:
        fut = self._futures.pop(card_id, None)
        if fut is not None:
            fut.cancel()
        self._pending_set.discard(card_id)
        self._cards.pop(card_id, None)
        self._visible_order = [c for c in self._visible_order if c != card_id]

    def _assert_open(self) -> None:
        if self._closed:
            raise InspectionError("GalleryPreviewScheduler is closed.")


def measure_preview_concurrency_capacity(
    *,
    video_path: Path | str,
    prepared_indices: Sequence[int],
    candidates: Sequence[int] = (2, 4, 6, 8),
    repeats: int = 3,
    frame_reader: FrameReader | None = None,
) -> dict[str, Any]:
    """Microbench helper to choose a practical concurrency cap.

    Measures wall time to decode ``prepared_indices`` under each candidate
    worker count. Returns the smallest candidate within 15% of the best mean
    throughput (avoids oversized pools that add contention).
    """

    indices = [int(i) for i in prepared_indices]
    if not indices:
        raise InspectionError("prepared_indices required for capacity measurement.")
    results: list[dict[str, Any]] = []
    for workers in candidates:
        times: list[float] = []
        for _ in range(int(repeats)):
            accessor = PreparedFrameAccessor(
                video_path,
                cache_size=0,
                frame_reader=frame_reader,
            )
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=int(workers)) as pool:
                list(pool.map(accessor.read_frame, indices))
            times.append(time.perf_counter() - started)
        mean_t = float(sum(times) / len(times))
        results.append(
            {
                "workers": int(workers),
                "mean_sec": mean_t,
                "throughput_fps": float(len(indices) / mean_t) if mean_t > 0 else 0.0,
            }
        )
    best = max(results, key=lambda r: r["throughput_fps"])
    threshold = float(best["throughput_fps"]) * 0.85
    chosen = next(
        r["workers"] for r in results if float(r["throughput_fps"]) >= threshold
    )
    return {
        "candidates": results,
        "best_workers": int(best["workers"]),
        "recommended_workers": int(chosen),
        "threshold_throughput_fps": threshold,
    }
