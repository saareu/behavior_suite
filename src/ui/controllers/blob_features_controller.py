"""Headless coordination for Subsystem 05 interactive preview and extraction.

Owns session eligibility, working vs preview/extraction settings snapshots,
frame navigation, bookmarks, preview generation, completed-run discovery, and
dispatch of the headless blob-feature runner. Scientific work always runs
through ``blob_features`` modules — never duplicated in the GUI.
"""

from __future__ import annotations

import contextlib
import json
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from blob_features import (
    BlobExecutionSettings,
    BlobFeaturesConfig,
    BlobFeaturesError,
    BlobFeaturesRequest,
    BlobFeaturesResult,
    BlobSegmentationSettings,
    BlobTemporalSettings,
    EligibleS1Input,
    FrameSegmentationPreview,
    SegmentationBookmark,
    SegmentationBookmarkStore,
    add_bookmark,
    draw_contour_overlay,
    load_blob_features_config,
    load_bookmarks_or_empty,
    preview_prepared_frame,
    read_prepared_video_frame,
    remove_bookmark,
    run_blob_feature_extraction,
    save_bookmarks,
    validate_s1_for_blob_features,
)
from blob_features.gui_persistence import (
    S1SourceIdentity,
    load_working_settings_for_source,
    save_working_settings_for_source,
)
from blob_features.preview import load_cropped_background
from blob_features.run_discovery import (
    BlobFeaturesProjectSummary,
    BlobFeaturesRunSummary,
    summarize_blob_features_project,
)
from preprocess.exceptions import VideoProbeError
from preprocess.models import OperationProgress
from ui.controllers.run_preprocess_controller import open_folder_in_platform_explorer

ProgressCallback = Callable[[OperationProgress], None]

FolderOpener = Callable[[Path], bool]
BlobFeaturesRunner = Callable[..., BlobFeaturesResult]


class BlobFeaturesUiError(ValueError):
    """Expected, actionable Subsystem 05 UI validation error."""


@dataclass(frozen=True, slots=True)
class BlobFeaturesUiSettings:
    """GUI-exposed working settings (scientific defaults from suite YAML)."""

    segmentation: BlobSegmentationSettings = field(
        default_factory=BlobSegmentationSettings
    )
    temporal: BlobTemporalSettings = field(default_factory=BlobTemporalSettings)
    execution: BlobExecutionSettings = field(default_factory=BlobExecutionSettings)

    @classmethod
    def from_config(cls, config: BlobFeaturesConfig) -> BlobFeaturesUiSettings:
        validated = config.validated()
        return cls(
            segmentation=validated.segmentation,
            temporal=validated.temporal,
            execution=validated.execution,
        )

    def to_config(self) -> BlobFeaturesConfig:
        return BlobFeaturesConfig(
            segmentation=self.segmentation,
            temporal=self.temporal,
            execution=self.execution,
        ).validated()

    def validated(self) -> BlobFeaturesUiSettings:
        return self.from_config(self.to_config())

    def segmentation_fingerprint(self) -> tuple[Any, ...]:
        seg = self.segmentation.validated()
        return (
            seg.blob_threshold,
            seg.blob_blur_kernel_size,
            seg.blob_kernel_size,
            seg.blob_tail_open_kernel_size,
            seg.blob_min_area,
            seg.blob_max_area,
            seg.blob_prune_min_radius,
            seg.blob_prune_regrow_iters,
            seg.blob_prune_max_regrow_dist,
            seg.blob_post_prune_min_area,
            seg.blob_post_prune_max_area,
            seg.blob_final_smooth_kernel_size,
            seg.abs_dark_threshold,
        )


@dataclass(frozen=True, slots=True)
class PreviewRequestIdentity:
    """Identity used to reject stale asynchronous preview results."""

    request_id: str
    session_root: Path
    prepared_video: Path
    cropped_background: Path
    prepared_frame_indices: tuple[int, ...]
    settings: BlobFeaturesUiSettings
    settings_fingerprint: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class ExtractionRequestIdentity:
    """Identity used to reject stale asynchronous extraction results."""

    request_id: str
    session_root: Path
    prepared_sync: Path
    prepared_sync_sha256: str
    settings: BlobFeaturesUiSettings


@dataclass(frozen=True, slots=True)
class SingleFramePreviewResult:
    identity: PreviewRequestIdentity
    frame_bgr: np.ndarray
    background_bgr: np.ndarray
    preview: FrameSegmentationPreview
    contour_overlay_bgr: np.ndarray


@dataclass(frozen=True, slots=True)
class MultiFramePreviewItem:
    prepared_frame_idx: int
    preview: FrameSegmentationPreview
    contour_overlay_bgr: np.ndarray
    frame_bgr: np.ndarray


@dataclass(frozen=True, slots=True)
class MultiFramePreviewResult:
    identity: PreviewRequestIdentity
    items: tuple[MultiFramePreviewItem, ...]
    errors: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True, slots=True)
class CachedFramePreview:
    """One successfully computed preview retained for frame navigation."""

    source_key: str
    prepared_frame_idx: int
    settings_fingerprint: tuple[Any, ...]
    frame_bgr: np.ndarray
    background_bgr: np.ndarray
    preview: FrameSegmentationPreview
    contour_overlay_bgr: np.ndarray


class PreviewResultCache:
    """Bounded LRU cache keyed by S1 identity, frame index, and settings."""

    def __init__(self, *, max_entries: int = 64) -> None:
        self._max_entries = max(1, int(max_entries))
        self._entries: OrderedDict[
            tuple[str, int, tuple[Any, ...]], CachedFramePreview
        ] = OrderedDict()

    def clear(self) -> None:
        self._entries.clear()

    def put(self, entry: CachedFramePreview) -> None:
        key = (entry.source_key, entry.prepared_frame_idx, entry.settings_fingerprint)
        if key in self._entries:
            del self._entries[key]
        self._entries[key] = entry
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)

    def get(
        self,
        *,
        source_key: str,
        prepared_frame_idx: int,
        settings_fingerprint: tuple[Any, ...],
    ) -> CachedFramePreview | None:
        key = (source_key, int(prepared_frame_idx), settings_fingerprint)
        entry = self._entries.get(key)
        if entry is None:
            return None
        self._entries.move_to_end(key)
        return entry

    def __len__(self) -> int:
        return len(self._entries)


@dataclass(slots=True)
class BlobFeaturesUiState:
    """Mutable S5 workspace state (working settings vs preview snapshots)."""

    session_root: Path | None = None
    eligible: EligibleS1Input | None = None
    source_identity: S1SourceIdentity | None = None
    eligibility_error: str | None = None
    summary: BlobFeaturesProjectSummary | None = None
    selected_s5_run_id: str | None = None
    current_frame_idx: int = 0
    current_frame_bgr: np.ndarray | None = None
    background_bgr: np.ndarray | None = None
    frame_load_error: str | None = None
    working_settings: BlobFeaturesUiSettings = field(
        default_factory=BlobFeaturesUiSettings
    )
    last_preview_settings: BlobFeaturesUiSettings | None = None
    last_extraction_settings: BlobFeaturesUiSettings | None = None
    preview_valid: bool = False
    last_single_preview: SingleFramePreviewResult | None = None
    last_multi_preview: MultiFramePreviewResult | None = None
    preview_cache: PreviewResultCache = field(default_factory=PreviewResultCache)
    bookmarks: SegmentationBookmarkStore | None = None
    task_running: bool = False
    active_preview_request_id: str | None = None
    active_extraction_identity: ExtractionRequestIdentity | None = None
    preview_error: str | None = None
    run_stage: str = "Not started"
    run_error: str | None = None
    last_result: BlobFeaturesResult | None = None
    unexpected_error_detail: str | None = None


FrameReader = Callable[[Path, int], np.ndarray]
BackgroundLoader = Callable[[Path], np.ndarray]


class BlobFeaturesController:
    """Session eligibility, frame access, settings, bookmarks, preview, extraction."""

    def __init__(
        self,
        *,
        config_loader: Callable[[], BlobFeaturesConfig] | None = None,
        frame_reader: FrameReader = read_prepared_video_frame,
        background_loader: BackgroundLoader = load_cropped_background,
        folder_opener: FolderOpener = open_folder_in_platform_explorer,
        runner: BlobFeaturesRunner = run_blob_feature_extraction,
    ) -> None:
        self.state = BlobFeaturesUiState()
        self._config_loader = config_loader or (lambda: load_blob_features_config())
        self._frame_reader = frame_reader
        self._background_loader = background_loader
        self._folder_opener = folder_opener
        self._runner = runner
        self.reset_working_settings_to_defaults()

    def reset_working_settings_to_defaults(self) -> BlobFeaturesUiSettings:
        try:
            config = self._config_loader()
            settings = BlobFeaturesUiSettings.from_config(config)
        except BlobFeaturesError as exc:
            raise BlobFeaturesUiError(str(exc)) from exc
        self.state.working_settings = settings
        self._invalidate_preview(reason="working settings reset to defaults")
        self._persist_working_settings()
        return settings

    def set_session(self, session_root: Path | str) -> EligibleS1Input:
        """Open a session/project and validate eligible S1 independently of S2–S4."""

        root = Path(session_root).expanduser().resolve(strict=False)
        if not root.is_dir():
            raise BlobFeaturesUiError(f"Session directory does not exist: {root}")
        previous = self.state.session_root
        if (
            self.state.task_running
            and previous is not None
            and previous != root
        ):
            raise BlobFeaturesUiError(
                "Cannot change S5 session while a preview or extraction task is active."
            )
        self.state.session_root = root
        self.state.eligibility_error = None
        self.state.preview_error = None
        self.state.frame_load_error = None
        self.state.last_single_preview = None
        self.state.last_multi_preview = None
        self.state.preview_valid = False
        self.state.last_preview_settings = None
        self.state.preview_cache.clear()
        # Changing S1 source invalidates prior preview; completed extraction
        # snapshots are session-local and cleared with the session.
        if previous is None or previous != root:
            self.state.last_extraction_settings = None
            self.state.last_result = None
            self.state.selected_s5_run_id = None
            self.state.run_error = None
            self.state.run_stage = "Not started"
        try:
            eligible = validate_s1_for_blob_features(root)
        except BlobFeaturesError as exc:
            self.state.eligible = None
            self.state.source_identity = None
            self.state.background_bgr = None
            self.state.current_frame_bgr = None
            self.state.bookmarks = None
            self.state.summary = None
            self.state.eligibility_error = str(exc)
            raise BlobFeaturesUiError(str(exc)) from exc

        self.state.eligible = eligible
        identity = S1SourceIdentity.from_eligible(eligible)
        self.state.source_identity = identity
        self.state.current_frame_idx = 0
        try:
            self.state.background_bgr = self._background_loader(eligible.cropped_background)
        except BlobFeaturesError as exc:
            self.state.background_bgr = None
            self.state.eligibility_error = str(exc)
            raise BlobFeaturesUiError(str(exc)) from exc

        persisted = load_working_settings_for_source(identity)
        if persisted is not None:
            # Preserve the user's explicit saved working values for this source.
            self.state.working_settings = BlobFeaturesUiSettings.from_config(persisted)
        else:
            # Do not carry scientific settings across distinct S1 identities.
            try:
                self.state.working_settings = BlobFeaturesUiSettings.from_config(
                    self._config_loader()
                )
            except BlobFeaturesError as exc:
                raise BlobFeaturesUiError(str(exc)) from exc

        self.state.bookmarks = load_bookmarks_or_empty(
            session_root=eligible.session_root,
            preprocess_dir=eligible.preprocess_dir,
            prepared_video=eligible.prepared_video,
            prepared_sync=eligible.prepared_sync,
            prepared_video_sha256=identity.prepared_video_sha256,
            prepared_sync_sha256=identity.prepared_sync_sha256,
        )
        self.load_current_frame()
        self.refresh_discovery()
        self._persist_working_settings()
        return eligible

    def clear_session(self) -> None:
        if self.state.task_running:
            raise BlobFeaturesUiError(
                "Cannot clear the S5 session while a task is active."
            )
        self.state = BlobFeaturesUiState(working_settings=self.state.working_settings)

    def refresh_discovery(self) -> BlobFeaturesProjectSummary:
        """Re-scan completed S5 outputs for the current session."""

        root = self.state.session_root
        if root is None:
            raise BlobFeaturesUiError("Select a project/session first.")
        try:
            summary = summarize_blob_features_project(root)
        except BlobFeaturesError as exc:
            raise BlobFeaturesUiError(str(exc)) from exc
        self.state.summary = summary
        self._reconcile_s5_selection()
        return summary

    def require_eligible(self) -> EligibleS1Input:
        if self.state.eligible is None:
            detail = self.state.eligibility_error or "Select an eligible S1 session first."
            raise BlobFeaturesUiError(detail)
        return self.state.eligible

    def set_working_segmentation(
        self, segmentation: BlobSegmentationSettings
    ) -> BlobFeaturesUiSettings:
        try:
            validated = replace(
                self.state.working_settings,
                segmentation=segmentation,
            ).validated()
        except BlobFeaturesError as exc:
            raise BlobFeaturesUiError(str(exc)) from exc
        if (
            validated.segmentation_fingerprint()
            != self.state.working_settings.segmentation_fingerprint()
        ):
            self.state.working_settings = validated
            self._invalidate_preview(reason="segmentation parameter changed")
        else:
            self.state.working_settings = validated
        self._persist_working_settings()
        return validated

    def set_working_settings(self, settings: BlobFeaturesUiSettings) -> BlobFeaturesUiSettings:
        try:
            validated = settings.validated()
        except BlobFeaturesError as exc:
            raise BlobFeaturesUiError(str(exc)) from exc
        fingerprint_changed = (
            validated.segmentation_fingerprint()
            != self.state.working_settings.segmentation_fingerprint()
        )
        self.state.working_settings = validated
        if fingerprint_changed:
            self._invalidate_preview(reason="segmentation parameter changed")
        self._persist_working_settings()
        return validated

    def working_settings_validated(self) -> BlobFeaturesUiSettings:
        try:
            return self.state.working_settings.validated()
        except BlobFeaturesError as exc:
            raise BlobFeaturesUiError(str(exc)) from exc

    def set_frame_index(self, prepared_frame_idx: int) -> int:
        eligible = self.require_eligible()
        idx = int(prepared_frame_idx)
        if idx < 0 or idx >= eligible.frame_count:
            raise BlobFeaturesUiError(
                f"Prepared frame index out of range: {idx} "
                f"(valid 0..{eligible.frame_count - 1})."
            )
        if idx != self.state.current_frame_idx:
            self.state.current_frame_idx = idx
            self.load_current_frame()
        self._sync_current_frame_preview_from_cache()
        return idx

    def step_frame(self, delta: int) -> int:
        eligible = self.require_eligible()
        nxt = int(self.state.current_frame_idx) + int(delta)
        nxt = max(0, min(eligible.frame_count - 1, nxt))
        return self.set_frame_index(nxt)

    def current_timestamp_sec(self) -> float | None:
        eligible = self.state.eligible
        if eligible is None:
            return None
        times = np.asarray(eligible.source_timing.prepared_time_sec, dtype=float)
        idx = int(self.state.current_frame_idx)
        if idx < 0 or idx >= times.size:
            return None
        return float(times[idx])

    def load_current_frame(self) -> np.ndarray:
        eligible = self.require_eligible()
        idx = int(self.state.current_frame_idx)
        try:
            frame = self._frame_reader(eligible.prepared_video, idx)
        except (BlobFeaturesError, VideoProbeError) as exc:
            self.state.frame_load_error = str(exc)
            self.state.current_frame_bgr = None
            raise BlobFeaturesUiError(str(exc)) from exc
        self.state.frame_load_error = None
        self.state.current_frame_bgr = frame
        return frame

    def bookmark_list(self) -> tuple[SegmentationBookmark, ...]:
        if self.state.bookmarks is None:
            return ()
        return self.state.bookmarks.bookmarks

    def add_current_bookmark(self, note: str = "") -> SegmentationBookmarkStore:
        eligible = self.require_eligible()
        identity = self.state.source_identity
        store = self.state.bookmarks or load_bookmarks_or_empty(
            session_root=eligible.session_root,
            preprocess_dir=eligible.preprocess_dir,
            prepared_video=eligible.prepared_video,
            prepared_sync=eligible.prepared_sync,
            prepared_video_sha256=(
                identity.prepared_video_sha256 if identity is not None else None
            ),
            prepared_sync_sha256=(
                identity.prepared_sync_sha256 if identity is not None else None
            ),
        )
        updated = add_bookmark(
            store,
            prepared_frame_idx=int(self.state.current_frame_idx),
            note=note,
        )
        save_bookmarks(updated)
        self.state.bookmarks = updated
        return updated

    def remove_bookmark_at(self, prepared_frame_idx: int) -> SegmentationBookmarkStore:
        eligible = self.require_eligible()
        identity = self.state.source_identity
        store = self.state.bookmarks or load_bookmarks_or_empty(
            session_root=eligible.session_root,
            preprocess_dir=eligible.preprocess_dir,
            prepared_video=eligible.prepared_video,
            prepared_sync=eligible.prepared_sync,
            prepared_video_sha256=(
                identity.prepared_video_sha256 if identity is not None else None
            ),
            prepared_sync_sha256=(
                identity.prepared_sync_sha256 if identity is not None else None
            ),
        )
        updated = remove_bookmark(store, prepared_frame_idx)
        save_bookmarks(updated)
        self.state.bookmarks = updated
        return updated

    def go_to_bookmark(self, prepared_frame_idx: int) -> int:
        return self.set_frame_index(int(prepared_frame_idx))

    def begin_preview_request(
        self,
        prepared_frame_indices: Sequence[int] | None = None,
    ) -> PreviewRequestIdentity:
        """Validate settings and capture an identity for an async preview task."""

        eligible = self.require_eligible()
        settings = self.working_settings_validated()
        indices: tuple[int, ...]
        if prepared_frame_indices is None:
            indices = (int(self.state.current_frame_idx),)
        else:
            indices = tuple(int(idx) for idx in prepared_frame_indices)
            if not indices:
                raise BlobFeaturesUiError("Select at least one bookmark to preview.")
            for idx in indices:
                if idx < 0 or idx >= eligible.frame_count:
                    raise BlobFeaturesUiError(
                        f"Bookmark frame out of range: {idx} "
                        f"(valid 0..{eligible.frame_count - 1})."
                    )
        identity = PreviewRequestIdentity(
            request_id=str(uuid4()),
            session_root=eligible.session_root,
            prepared_video=eligible.prepared_video,
            cropped_background=eligible.cropped_background,
            prepared_frame_indices=indices,
            settings=settings,
            settings_fingerprint=settings.segmentation_fingerprint(),
        )
        self.state.active_preview_request_id = identity.request_id
        self.state.preview_error = None
        self.state.task_running = True
        return identity

    def compute_single_frame_preview(
        self,
        identity: PreviewRequestIdentity,
    ) -> SingleFramePreviewResult:
        """Worker-thread entry: segment one prepared frame under the snapshot."""

        settings = identity.settings.validated()
        frame_idx = identity.prepared_frame_indices[0]
        frame = self._frame_reader(identity.prepared_video, frame_idx)
        background = self._background_loader(identity.cropped_background)
        preview = preview_prepared_frame(
            frame=frame,
            background=background,
            settings=settings.segmentation,
            prepared_frame_idx=frame_idx,
        )
        overlay = draw_contour_overlay(frame, preview.contours)
        return SingleFramePreviewResult(
            identity=identity,
            frame_bgr=frame,
            background_bgr=background,
            preview=preview,
            contour_overlay_bgr=overlay,
        )

    def compute_multi_frame_preview(
        self,
        identity: PreviewRequestIdentity,
        progress_callback: ProgressCallback | None = None,
    ) -> MultiFramePreviewResult:
        """Worker-thread entry: preview bookmarked frames under one snapshot.

        Processes every requested frame with the immutable settings snapshot.
        Per-frame failures are recorded without aborting the remainder of the batch.
        """

        settings = identity.settings.validated()
        background = self._background_loader(identity.cropped_background)
        total = len(identity.prepared_frame_indices)
        items: list[MultiFramePreviewItem] = []
        errors: list[tuple[int, str]] = []
        for completed, frame_idx in enumerate(identity.prepared_frame_indices, start=1):
            if progress_callback is not None:
                progress_callback(
                    OperationProgress(
                        phase="preview_bookmarks",
                        message=(
                            f"Previewing bookmarked frame {frame_idx} "
                            f"({completed}/{total})…"
                        ),
                        completed_units=completed - 1,
                        total_units=total,
                        total_is_estimate=False,
                        is_indeterminate=False,
                        processed_video_time_sec=None,
                    )
                )
            try:
                frame = self._frame_reader(identity.prepared_video, frame_idx)
                preview = preview_prepared_frame(
                    frame=frame,
                    background=background,
                    settings=settings.segmentation,
                    prepared_frame_idx=frame_idx,
                )
                items.append(
                    MultiFramePreviewItem(
                        prepared_frame_idx=frame_idx,
                        preview=preview,
                        contour_overlay_bgr=draw_contour_overlay(
                            frame, preview.contours
                        ),
                        frame_bgr=frame,
                    )
                )
            except (BlobFeaturesError, VideoProbeError, OSError, ValueError) as exc:
                errors.append((int(frame_idx), str(exc)))
        if progress_callback is not None:
            progress_callback(
                OperationProgress(
                    phase="preview_bookmarks",
                    message=f"Bookmark preview finished ({len(items)}/{total} ok).",
                    completed_units=total,
                    total_units=total,
                    total_is_estimate=False,
                    is_indeterminate=False,
                    processed_video_time_sec=None,
                )
            )
        return MultiFramePreviewResult(
            identity=identity,
            items=tuple(items),
            errors=tuple(errors),
        )

    def accept_single_preview(
        self,
        result: SingleFramePreviewResult,
        *,
        expected_request_id: str | None = None,
    ) -> bool:
        """Apply a worker result only when it still matches current identity."""

        self.state.task_running = False
        if not self._is_current_preview_result(result.identity, expected_request_id):
            return False
        self.state.last_preview_settings = result.identity.settings
        self.state.preview_error = None
        self.state.active_preview_request_id = None
        self._store_preview_in_cache(
            prepared_frame_idx=result.preview.prepared_frame_idx,
            settings=result.identity.settings,
            frame_bgr=result.frame_bgr,
            background_bgr=result.background_bgr,
            preview=result.preview,
            contour_overlay_bgr=result.contour_overlay_bgr,
        )
        self._sync_current_frame_preview_from_cache()
        return True

    def accept_multi_preview(
        self,
        result: MultiFramePreviewResult,
        *,
        expected_request_id: str | None = None,
    ) -> bool:
        """Accept a batch bookmark preview without changing the selected frame."""

        self.state.task_running = False
        if not self._is_current_preview_result(result.identity, expected_request_id):
            return False
        self.state.last_multi_preview = result
        self.state.last_preview_settings = result.identity.settings
        self.state.preview_error = None
        self.state.active_preview_request_id = None
        background = self.state.background_bgr
        for item in result.items:
            bg = background if background is not None else item.frame_bgr
            self._store_preview_in_cache(
                prepared_frame_idx=item.prepared_frame_idx,
                settings=result.identity.settings,
                frame_bgr=item.frame_bgr,
                background_bgr=bg,
                preview=item.preview,
                contour_overlay_bgr=item.contour_overlay_bgr,
            )
        # Preserve navigation position; restore current-frame result if cached.
        self._sync_current_frame_preview_from_cache()
        return True

    def preview_for_current_frame(self) -> CachedFramePreview | None:
        """Return the cached preview for the current frame when still compatible."""

        identity = self.state.source_identity
        if identity is None or not self._preview_settings_match_working():
            return None
        fingerprint = self.state.working_settings.segmentation_fingerprint()
        return self.state.preview_cache.get(
            source_key=identity.cache_key(),
            prepared_frame_idx=int(self.state.current_frame_idx),
            settings_fingerprint=fingerprint,
        )

    def current_frame_has_valid_preview(self) -> bool:
        return self.preview_for_current_frame() is not None

    def reject_preview(
        self,
        error: BaseException | str,
        *,
        expected_request_id: str | None = None,
    ) -> None:
        self.state.task_running = False
        if (
            expected_request_id is not None
            and self.state.active_preview_request_id is not None
            and expected_request_id != self.state.active_preview_request_id
        ):
            return
        self.state.preview_error = str(error)
        self.state.preview_valid = False
        self.state.active_preview_request_id = None
        if isinstance(error, Exception) and not isinstance(
            error, (BlobFeaturesError, BlobFeaturesUiError, VideoProbeError)
        ):
            self.state.unexpected_error_detail = repr(error)

    def select_s5_run(self, run_id: str | None) -> BlobFeaturesRunSummary | None:
        """Select a discovered S5 run for QC/output inspection."""

        if run_id is None or not str(run_id).strip():
            self.state.selected_s5_run_id = None
            return None
        summary = self.state.summary
        if summary is None:
            raise BlobFeaturesUiError("Refresh discovery before selecting an S5 run.")
        match = next((run for run in summary.runs if run.run_id == run_id), None)
        if match is None:
            raise BlobFeaturesUiError(f"S5 run was not found: {run_id}")
        self.state.selected_s5_run_id = match.run_id
        return match

    def selected_s5_run(self) -> BlobFeaturesRunSummary | None:
        summary = self.state.summary
        selected_id = self.state.selected_s5_run_id
        if summary is None or selected_id is None:
            return None
        return next((run for run in summary.runs if run.run_id == selected_id), None)

    def can_run_extraction(self) -> bool:
        return (
            not self.state.task_running
            and self.state.session_root is not None
            and self.state.eligible is not None
        )

    def cannot_run_reason(self) -> str | None:
        if self.state.task_running:
            return "A Subsystem 5 task is already running."
        if self.state.session_root is None:
            return "Select a project/session first."
        if self.state.eligible is None:
            return (
                self.state.eligibility_error
                or "No eligible S1 result is available for blob feature extraction."
            )
        return None

    def build_request(
        self, *, timestamp: str | None = None
    ) -> tuple[BlobFeaturesRequest, ExtractionRequestIdentity]:
        """Validate S1, snapshot immutable settings, and build a runner request."""

        reason = self.cannot_run_reason()
        if reason is not None:
            raise BlobFeaturesUiError(reason)
        try:
            eligible = validate_s1_for_blob_features(self.require_eligible().session_root)
        except BlobFeaturesError as exc:
            self.state.eligibility_error = str(exc)
            self.state.eligible = None
            raise BlobFeaturesUiError(str(exc)) from exc
        self.state.eligible = eligible
        identity_source = S1SourceIdentity.from_eligible(eligible)
        self.state.source_identity = identity_source
        settings = self.working_settings_validated()
        # Immutable snapshot for this run — later GUI edits never rewrite it.
        config_snapshot = settings.to_config()
        request = BlobFeaturesRequest(
            session_root=eligible.session_root,
            preprocess_dir=eligible.preprocess_dir,
            config=config_snapshot,
            run_purpose="gui",
            timestamp=timestamp,
        )
        extraction_identity = ExtractionRequestIdentity(
            request_id=str(uuid4()),
            session_root=eligible.session_root,
            prepared_sync=eligible.prepared_sync,
            prepared_sync_sha256=identity_source.prepared_sync_sha256,
            settings=settings,
        )
        self.state.active_extraction_identity = extraction_identity
        self.state.task_running = True
        self.state.run_error = None
        self.state.run_stage = "Running blob feature extraction…"
        return request, extraction_identity

    def execute_request(
        self,
        request: BlobFeaturesRequest,
        progress_callback: Callable[[OperationProgress], None] | None = None,
    ) -> BlobFeaturesResult:
        """Run one validated request (worker-thread safe; no Qt updates)."""

        return self._runner(request, progress_callback=progress_callback)

    def record_success(
        self,
        result: BlobFeaturesResult,
        *,
        expected_identity: ExtractionRequestIdentity | None = None,
    ) -> bool:
        """Apply a successful result only when the source identity is still current."""

        self.state.task_running = False
        if not self._is_current_extraction_result(expected_identity):
            return False
        self.state.last_result = result
        self.state.run_stage = result.status
        self.state.run_error = None
        self.state.last_extraction_settings = (
            expected_identity.settings
            if expected_identity is not None
            else self.state.working_settings
        )
        self.state.selected_s5_run_id = result.run_id
        self.state.active_extraction_identity = None
        try:
            self.refresh_discovery()
            self.state.selected_s5_run_id = result.run_id
        except BlobFeaturesUiError:
            pass
        return True

    def record_failure(
        self,
        message: str,
        *,
        expected_identity: ExtractionRequestIdentity | None = None,
    ) -> bool:
        """Record a failed run without marking it completed."""

        self.state.task_running = False
        if (
            expected_identity is not None
            and self.state.active_extraction_identity is not None
            and expected_identity.request_id
            != self.state.active_extraction_identity.request_id
        ):
            return False
        self.state.last_result = None
        self.state.run_stage = "Failed"
        self.state.run_error = message
        self.state.active_extraction_identity = None
        return True

    def record_extraction_settings_snapshot(self) -> BlobFeaturesUiSettings:
        """Remember settings used for a completed extraction (not invalidated by edits)."""

        snapshot = self.working_settings_validated()
        self.state.last_extraction_settings = snapshot
        return snapshot

    def open_selected_s5_folder(self) -> bool:
        """Open the selected S5 run directory in the platform file browser."""

        run = self.selected_s5_run()
        if run is None and self.state.last_result is not None:
            return bool(self._folder_opener(self.state.last_result.run_dir))
        if run is None:
            raise BlobFeaturesUiError("Select a completed S5 run first.")
        return bool(self._folder_opener(run.run_dir))

    def qc_display_rows(self) -> tuple[tuple[str, str], ...]:
        """Compact QC rows for the results panel."""

        result = self.state.last_result
        run = self.selected_s5_run()
        meta_qc: Mapping[str, Any] | None = None
        status = None
        output = None

        use_last_result = result is not None and (
            run is None or run.run_id == result.run_id
        )
        if use_last_result and result is not None:
            status = result.status
            output = (
                str(result.blob_features_path)
                if result.blob_features_path is not None
                else None
            )
            meta_qc = result.qc
        elif run is not None:
            status = run.status
            output = str(run.blob_features_path) if run.blob_features_path else None
            meta_qc = _load_run_qc(run.run_dir)
        else:
            return ()

        rows: list[tuple[str, str]] = [
            ("Status", status or "—"),
            ("Output", output or "—"),
            ("Classification", run.classification if run is not None else "—"),
        ]
        if not isinstance(meta_qc, Mapping):
            return tuple(rows)
        metrics = meta_qc.get("metrics")
        if isinstance(metrics, Mapping):
            for key, label in (
                ("row_count", "Rows"),
                ("expected_row_count", "Expected rows (120 Hz)"),
                ("source_frame_count", "Source frames"),
                ("cc_count_raw_mean", "cc_count_raw mean"),
                ("cc_count_nan_fraction", "cc_count NaN fraction"),
            ):
                value = metrics.get(key)
                if value is not None:
                    if isinstance(value, float):
                        rows.append((label, f"{value:.4g}"))
                    else:
                        rows.append((label, str(value)))
        structural = meta_qc.get("structural_ok")
        if structural is not None:
            rows.append(("Structural QC", "OK" if structural else "FAILED"))
        warnings = meta_qc.get("warnings")
        if isinstance(warnings, list) and warnings:
            rows.append(("Warnings", "; ".join(str(w) for w in warnings)))
        elif isinstance(meta_qc.get("warning_count"), int):
            rows.append(("Warnings", str(meta_qc["warning_count"])))
        failures = meta_qc.get("failures")
        if isinstance(failures, list) and failures:
            rows.append(("Failures", "; ".join(str(f) for f in failures)))
        return tuple(rows)

    def preview_is_stale(self) -> bool:
        if self.state.last_preview_settings is None and len(self.state.preview_cache) == 0:
            return True
        return not self._preview_settings_match_working()

    def diagnostics_summary(
        self, preview: FrameSegmentationPreview | None = None
    ) -> Mapping[str, str]:
        target = preview
        if target is None:
            cached = self.preview_for_current_frame()
            if cached is not None:
                target = cached.preview
            elif self.state.last_single_preview is not None and self.state.preview_valid:
                target = self.state.last_single_preview.preview
        if target is None:
            return {
                "cc_count_raw": "—",
                "cc_count": "—",
                "min_inter_contour_dist": "—",
                "warnings": "No preview yet.",
            }
        cc = target.cc_count
        cc_text = "NaN" if not np.isfinite(cc) else f"{cc:g}"
        dist = target.min_inter_contour_dist
        dist_text = "NaN" if not np.isfinite(dist) else f"{dist:.3f}"
        warnings = "; ".join(target.warnings) if target.warnings else "None"
        return {
            "cc_count_raw": str(target.cc_count_raw),
            "cc_count": cc_text,
            "min_inter_contour_dist": dist_text,
            "warnings": warnings,
        }

    def _is_current_preview_result(
        self,
        identity: PreviewRequestIdentity,
        expected_request_id: str | None,
    ) -> bool:
        if self.state.eligible is None or self.state.session_root is None:
            return False
        if expected_request_id is not None and identity.request_id != expected_request_id:
            return False
        if (
            self.state.active_preview_request_id is not None
            and identity.request_id != self.state.active_preview_request_id
        ):
            return False
        if identity.session_root != self.state.eligible.session_root:
            return False
        if identity.prepared_video != self.state.eligible.prepared_video:
            return False
        return (
            identity.settings_fingerprint
            == self.state.working_settings.segmentation_fingerprint()
        )

    def _is_current_extraction_result(
        self, expected_identity: ExtractionRequestIdentity | None
    ) -> bool:
        if expected_identity is None:
            return self.state.eligible is not None
        active = self.state.active_extraction_identity
        if active is not None and active.request_id != expected_identity.request_id:
            return False
        if self.state.eligible is None or self.state.source_identity is None:
            return False
        if expected_identity.session_root != self.state.eligible.session_root:
            return False
        if expected_identity.prepared_sync != self.state.eligible.prepared_sync:
            return False
        return (
            expected_identity.prepared_sync_sha256.lower()
            == self.state.source_identity.prepared_sync_sha256.lower()
        )

    def _preview_settings_match_working(self) -> bool:
        if self.state.last_preview_settings is None:
            return False
        return (
            self.state.last_preview_settings.segmentation_fingerprint()
            == self.state.working_settings.segmentation_fingerprint()
        )

    def _invalidate_preview(self, *, reason: str) -> None:
        self.state.preview_valid = False
        self.state.last_single_preview = None
        self.state.last_multi_preview = None
        self.state.last_preview_settings = None
        self.state.preview_cache.clear()
        self.state.preview_error = None
        if self.state.active_preview_request_id is not None:
            self.state.active_preview_request_id = f"invalidated:{uuid4()}"
        _ = reason

    def _store_preview_in_cache(
        self,
        *,
        prepared_frame_idx: int,
        settings: BlobFeaturesUiSettings,
        frame_bgr: np.ndarray,
        background_bgr: np.ndarray,
        preview: FrameSegmentationPreview,
        contour_overlay_bgr: np.ndarray,
    ) -> None:
        identity = self.state.source_identity
        if identity is None:
            return
        self.state.preview_cache.put(
            CachedFramePreview(
                source_key=identity.cache_key(),
                prepared_frame_idx=int(prepared_frame_idx),
                settings_fingerprint=settings.segmentation_fingerprint(),
                frame_bgr=frame_bgr,
                background_bgr=background_bgr,
                preview=preview,
                contour_overlay_bgr=contour_overlay_bgr,
            )
        )

    def _sync_current_frame_preview_from_cache(self) -> None:
        cached = self.preview_for_current_frame()
        if cached is None:
            self.state.preview_valid = False
            self.state.last_single_preview = None
            return
        settings = self.state.last_preview_settings or self.state.working_settings
        identity = PreviewRequestIdentity(
            request_id="cache",
            session_root=(
                self.state.eligible.session_root
                if self.state.eligible is not None
                else Path(".")
            ),
            prepared_video=(
                self.state.eligible.prepared_video
                if self.state.eligible is not None
                else Path(".")
            ),
            cropped_background=(
                self.state.eligible.cropped_background
                if self.state.eligible is not None
                else Path(".")
            ),
            prepared_frame_indices=(cached.prepared_frame_idx,),
            settings=settings,
            settings_fingerprint=cached.settings_fingerprint,
        )
        self.state.last_single_preview = SingleFramePreviewResult(
            identity=identity,
            frame_bgr=cached.frame_bgr,
            background_bgr=cached.background_bgr,
            preview=cached.preview,
            contour_overlay_bgr=cached.contour_overlay_bgr,
        )
        self.state.preview_valid = True

    def _persist_working_settings(self) -> None:
        identity = self.state.source_identity
        if identity is None:
            return
        # Persistence failures must not block interactive work.
        with contextlib.suppress(BlobFeaturesError):
            save_working_settings_for_source(identity, self.state.working_settings.to_config())

    def _reconcile_s5_selection(self) -> None:
        summary = self.state.summary
        if summary is None or not summary.runs:
            if self.state.last_result is None:
                self.state.selected_s5_run_id = None
            return
        current = self.state.selected_s5_run_id
        if current is not None and any(run.run_id == current for run in summary.runs):
            return
        # Prefer newest completed run for inspection (not an S6 handoff selection).
        completed = summary.completed_runs
        self.state.selected_s5_run_id = (
            completed[0].run_id if completed else summary.runs[0].run_id
        )


def ensure_bgr(image: np.ndarray) -> np.ndarray:
    """Normalize grayscale / BGRA arrays to BGR for ``VideoFrameView``."""

    array = np.asarray(image)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim == 3 and array.shape[2] == 3:
        return array
    if array.ndim == 3 and array.shape[2] == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2BGR)
    raise BlobFeaturesUiError(f"Unsupported image shape for display: {array.shape}")


def _load_run_qc(run_dir: Path) -> Mapping[str, Any] | None:
    for name in ("qc.json", "run_meta.json"):
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, UnicodeError):
            continue
        if not isinstance(payload, dict):
            continue
        if name == "qc.json":
            return payload
        qc = payload.get("qc")
        if isinstance(qc, Mapping):
            return qc
    return None
