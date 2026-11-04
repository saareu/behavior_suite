"""Export plan contracts for C2-A behavioral and C2-B S4 pose QC clips.

Prepared-frame intervals are inclusive on both ends. Dense bout endpoints remain
scientific authority; prepared seek uses SyncMap / packaged prepared columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from behavioral_review.sync import SyncMap
from overlay.contracts import (
    FrameDomain,
    HudDirectionMode,
    LayerId,
    OverlayContractError,
    OverlayExportScopeKind,
    PlateauPolicy,
    PoseSourceKind,
    default_behavioral_qc_layers,
    default_mixed_pose_behavior_qc_layers,
    default_s2_pose_qc_layers,
    default_s3_pose_qc_layers,
    default_s4_pose_qc_layers,
    default_s5_segmentation_qc_layers,
    validate_layer_selection,
)
from overlay.style import DrawConfig
from results_packaging.media_csv_export.clip_bounds import ClipBounds, compute_clip_bounds


@dataclass(frozen=True, slots=True)
class PreparedFrameInterval:
    """Inclusive prepared-video frame range ``[start, end]``."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if int(self.start) > int(self.end):
            raise OverlayContractError(
                f"Prepared interval start ({self.start}) > end ({self.end})."
            )
        if int(self.start) < 0:
            raise OverlayContractError(
                f"Prepared interval start must be >= 0; got {self.start}."
            )

    @property
    def frame_count(self) -> int:
        return int(self.end) - int(self.start) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "prepared_frame_idx_start_inclusive": int(self.start),
            "prepared_frame_idx_end_inclusive": int(self.end),
            "prepared_frame_count": self.frame_count,
            "boundary_convention": "inclusive_inclusive",
        }


@dataclass(frozen=True, slots=True)
class BehavioralQcExportPlan:
    """Explicit offline prepared-rate behavioral QC export plan."""

    scope_kind: OverlayExportScopeKind
    prepared_interval: PreparedFrameInterval
    hud_mode: HudDirectionMode
    selected_layers: tuple[LayerId, ...]
    plateau_policy: PlateauPolicy = PlateauPolicy.LAST_IN_PLATEAU
    frame_domain: FrameDomain = FrameDomain.PREPARED
    include_contact: bool = True
    include_sbs: bool = True
    style: DrawConfig = field(default_factory=DrawConfig)
    bout_id: str | None = None
    start_sample_idx: int | None = None
    end_sample_idx: int | None = None
    pre_context_sec: float = 0.0
    post_context_sec: float = 0.0
    clip_bounds: ClipBounds | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_kind": self.scope_kind.value,
            "prepared_interval": self.prepared_interval.to_dict(),
            "hud_mode": self.hud_mode.value,
            "selected_layers": [x.value for x in self.selected_layers],
            "plateau_policy": self.plateau_policy.value,
            "frame_domain": self.frame_domain.value,
            "include_contact": self.include_contact,
            "include_sbs": self.include_sbs,
            "bout_id": self.bout_id,
            "start_sample_idx": self.start_sample_idx,
            "end_sample_idx": self.end_sample_idx,
            "pre_context_sec": self.pre_context_sec,
            "post_context_sec": self.post_context_sec,
            "clip_bounds": None if self.clip_bounds is None else self.clip_bounds.to_dict(),
            "style": self.style.to_dict(),
            "notes": list(self.notes),
        }


def plan_prepared_interval(
    *,
    prepared_frame_idx_start: int,
    prepared_frame_idx_end: int,
    source_fps: float,
    source_n_frames: int,
    hud_mode: HudDirectionMode | str = HudDirectionMode.BOTH,
    include_contact: bool = True,
    include_sbs: bool = True,
    include_frame_counter: bool = True,
    selected_layers: list[str | LayerId] | tuple[str | LayerId, ...] | None = None,
    plateau_policy: PlateauPolicy = PlateauPolicy.LAST_IN_PLATEAU,
    pre_context_sec: float = 0.0,
    post_context_sec: float = 0.0,
    style: DrawConfig | None = None,
    notes: tuple[str, ...] = (),
) -> BehavioralQcExportPlan:
    """Build a plan from an explicitly selected inclusive prepared-frame interval."""

    mode = (
        hud_mode
        if isinstance(hud_mode, HudDirectionMode)
        else HudDirectionMode(str(hud_mode))
    )
    interval = PreparedFrameInterval(
        start=int(prepared_frame_idx_start),
        end=int(prepared_frame_idx_end),
    )
    bounds = compute_clip_bounds(
        prepared_frame_idx_start=interval.start,
        prepared_frame_idx_end=interval.end,
        source_fps=float(source_fps),
        source_n_frames=int(source_n_frames),
        pre_context_sec=float(pre_context_sec),
        post_context_sec=float(post_context_sec),
    )
    export_interval = PreparedFrameInterval(
        start=int(bounds.clip_frame_idx_start),
        end=int(bounds.clip_frame_idx_end),
    )
    layers = (
        validate_layer_selection(list(selected_layers))
        if selected_layers is not None
        else default_behavioral_qc_layers(
            hud_mode=mode,
            include_contact=include_contact,
            include_sbs=include_sbs,
            include_frame_counter=include_frame_counter,
        )
    )
    return BehavioralQcExportPlan(
        scope_kind=OverlayExportScopeKind.PREPARED_INTERVAL,
        prepared_interval=export_interval,
        hud_mode=mode,
        selected_layers=layers,
        plateau_policy=plateau_policy,
        include_contact=include_contact,
        include_sbs=include_sbs,
        style=style or DrawConfig(darken_alpha=0.92),
        pre_context_sec=float(pre_context_sec),
        post_context_sec=float(post_context_sec),
        clip_bounds=bounds,
        notes=notes,
    )


def plan_bout_span(
    *,
    sync: SyncMap,
    start_sample_idx: int,
    end_sample_idx: int,
    source_fps: float,
    source_n_frames: int,
    bout_id: str | None = None,
    packaged_prepared_start: int | None = None,
    packaged_prepared_end: int | None = None,
    hud_mode: HudDirectionMode | str = HudDirectionMode.BOTH,
    include_contact: bool = True,
    include_sbs: bool = True,
    include_frame_counter: bool = True,
    selected_layers: list[str | LayerId] | tuple[str | LayerId, ...] | None = None,
    plateau_policy: PlateauPolicy = PlateauPolicy.LAST_IN_PLATEAU,
    pre_context_sec: float = 0.0,
    post_context_sec: float = 0.0,
    style: DrawConfig | None = None,
    notes: tuple[str, ...] = (),
) -> BehavioralQcExportPlan:
    """Build a plan from an inclusive dense bout/span via SyncMap.bout_seek."""

    if int(start_sample_idx) > int(end_sample_idx):
        raise OverlayContractError(
            f"Dense bout start ({start_sample_idx}) > end ({end_sample_idx})."
        )
    seek = sync.bout_seek(
        start_sample_idx=int(start_sample_idx),
        end_sample_idx=int(end_sample_idx),
        bout_id=bout_id,
    )
    prep_start = int(seek.prepared_frame_idx_start)
    prep_end = int(seek.prepared_frame_idx_end)
    if (
        packaged_prepared_start is not None
        and packaged_prepared_end is not None
        and (
            int(packaged_prepared_start) != prep_start
            or int(packaged_prepared_end) != prep_end
        )
    ):
        raise OverlayContractError(
            "Packaged prepared bout endpoints disagree with SyncMap.bout_seek "
            f"(packaged=[{packaged_prepared_start}, {packaged_prepared_end}] "
            f"sync=[{prep_start}, {prep_end}])."
        )
    mode = (
        hud_mode
        if isinstance(hud_mode, HudDirectionMode)
        else HudDirectionMode(str(hud_mode))
    )
    plan = plan_prepared_interval(
        prepared_frame_idx_start=prep_start,
        prepared_frame_idx_end=prep_end,
        source_fps=source_fps,
        source_n_frames=source_n_frames,
        hud_mode=mode,
        include_contact=include_contact,
        include_sbs=include_sbs,
        include_frame_counter=include_frame_counter,
        selected_layers=selected_layers,
        plateau_policy=plateau_policy,
        pre_context_sec=pre_context_sec,
        post_context_sec=post_context_sec,
        style=style,
        notes=notes,
    )
    return BehavioralQcExportPlan(
        scope_kind=OverlayExportScopeKind.BOUT_SPAN,
        prepared_interval=plan.prepared_interval,
        hud_mode=plan.hud_mode,
        selected_layers=plan.selected_layers,
        plateau_policy=plan.plateau_policy,
        frame_domain=plan.frame_domain,
        include_contact=plan.include_contact,
        include_sbs=plan.include_sbs,
        style=plan.style,
        bout_id=bout_id,
        start_sample_idx=int(start_sample_idx),
        end_sample_idx=int(end_sample_idx),
        pre_context_sec=plan.pre_context_sec,
        post_context_sec=plan.post_context_sec,
        clip_bounds=plan.clip_bounds,
        notes=plan.notes,
    )


def dense_sample_window_for_prepared_interval(
    sync: SyncMap,
    interval: PreparedFrameInterval,
) -> tuple[int, int] | None:
    """Return inclusive dense sample bounds covering all plateaus in ``interval``.

    Returns ``None`` when no dense coverage exists for any prepared frame.
    """

    samples: list[int] = []
    for prep in range(int(interval.start), int(interval.end) + 1):
        try:
            plateau = sync.plateau_sample_indices(int(prep))
        except Exception:  # noqa: BLE001 - treat as missing coverage for that frame
            continue
        if plateau.size == 0:
            continue
        samples.extend(int(x) for x in plateau.tolist())
    if not samples:
        return None
    return int(min(samples)), int(max(samples))


@dataclass(frozen=True, slots=True)
class S4PoseQcExportPlan:
    """Explicit offline prepared-rate S4 finalized-pose QC export plan."""

    scope_kind: OverlayExportScopeKind
    prepared_interval: PreparedFrameInterval
    selected_layers: tuple[LayerId, ...]
    plateau_policy: PlateauPolicy = PlateauPolicy.LAST_IN_PLATEAU
    frame_domain: FrameDomain = FrameDomain.PREPARED
    draw_skeleton: bool = True
    include_frame_counter: bool = True
    style: DrawConfig = field(default_factory=DrawConfig)
    pre_context_sec: float = 0.0
    post_context_sec: float = 0.0
    clip_bounds: ClipBounds | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_kind": self.scope_kind.value,
            "prepared_interval": self.prepared_interval.to_dict(),
            "selected_layers": [x.value for x in self.selected_layers],
            "plateau_policy": self.plateau_policy.value,
            "frame_domain": self.frame_domain.value,
            "draw_skeleton": self.draw_skeleton,
            "include_frame_counter": self.include_frame_counter,
            "pre_context_sec": self.pre_context_sec,
            "post_context_sec": self.post_context_sec,
            "clip_bounds": None if self.clip_bounds is None else self.clip_bounds.to_dict(),
            "style": self.style.to_dict(),
            "notes": list(self.notes),
        }


def plan_s4_prepared_interval(
    *,
    prepared_frame_idx_start: int,
    prepared_frame_idx_end: int,
    source_fps: float,
    source_n_frames: int,
    include_frame_counter: bool = True,
    draw_skeleton: bool = True,
    selected_layers: list[str | LayerId] | tuple[str | LayerId, ...] | None = None,
    plateau_policy: PlateauPolicy = PlateauPolicy.LAST_IN_PLATEAU,
    pre_context_sec: float = 0.0,
    post_context_sec: float = 0.0,
    style: DrawConfig | None = None,
    notes: tuple[str, ...] = (),
) -> S4PoseQcExportPlan:
    """Build an S4 pose QC plan from an inclusive prepared-frame interval."""

    interval = PreparedFrameInterval(
        start=int(prepared_frame_idx_start),
        end=int(prepared_frame_idx_end),
    )
    bounds = compute_clip_bounds(
        prepared_frame_idx_start=interval.start,
        prepared_frame_idx_end=interval.end,
        source_fps=float(source_fps),
        source_n_frames=int(source_n_frames),
        pre_context_sec=float(pre_context_sec),
        post_context_sec=float(post_context_sec),
    )
    export_interval = PreparedFrameInterval(
        start=int(bounds.clip_frame_idx_start),
        end=int(bounds.clip_frame_idx_end),
    )
    layers = (
        validate_layer_selection(list(selected_layers))
        if selected_layers is not None
        else default_s4_pose_qc_layers(include_frame_counter=include_frame_counter)
    )
    if LayerId.S4_POSE_FINAL not in layers:
        raise OverlayContractError(
            "S4 pose QC plan requires the s4.pose_final layer."
        )
    draw_style = style or DrawConfig(
        darken_alpha=0.90,
        draw_skeleton=draw_skeleton,
        node_radius=5,
        edge_thickness=2,
        legend_text=True,
    )
    return S4PoseQcExportPlan(
        scope_kind=OverlayExportScopeKind.PREPARED_INTERVAL,
        prepared_interval=export_interval,
        selected_layers=layers,
        plateau_policy=plateau_policy,
        draw_skeleton=draw_skeleton,
        include_frame_counter=include_frame_counter,
        style=draw_style,
        pre_context_sec=float(pre_context_sec),
        post_context_sec=float(post_context_sec),
        clip_bounds=bounds,
        notes=notes,
    )


@dataclass(frozen=True, slots=True)
class S2PoseQcExportPlan:
    """Explicit offline prepared-rate S2 inferred-pose QC export plan."""

    scope_kind: OverlayExportScopeKind
    prepared_interval: PreparedFrameInterval
    selected_layers: tuple[LayerId, ...]
    plateau_policy: PlateauPolicy = PlateauPolicy.LAST_IN_PLATEAU
    frame_domain: FrameDomain = FrameDomain.PREPARED
    draw_skeleton: bool = True
    show_confidence: bool = False
    include_frame_counter: bool = True
    style: DrawConfig = field(default_factory=DrawConfig)
    pre_context_sec: float = 0.0
    post_context_sec: float = 0.0
    clip_bounds: ClipBounds | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_kind": self.scope_kind.value,
            "prepared_interval": self.prepared_interval.to_dict(),
            "selected_layers": [x.value for x in self.selected_layers],
            "plateau_policy": self.plateau_policy.value,
            "frame_domain": self.frame_domain.value,
            "draw_skeleton": self.draw_skeleton,
            "show_confidence": self.show_confidence,
            "include_frame_counter": self.include_frame_counter,
            "pre_context_sec": self.pre_context_sec,
            "post_context_sec": self.post_context_sec,
            "clip_bounds": None if self.clip_bounds is None else self.clip_bounds.to_dict(),
            "style": self.style.to_dict(),
            "notes": list(self.notes),
        }


def plan_s2_prepared_interval(
    *,
    prepared_frame_idx_start: int,
    prepared_frame_idx_end: int,
    source_fps: float,
    source_n_frames: int,
    include_frame_counter: bool = True,
    draw_skeleton: bool = True,
    show_confidence: bool = False,
    selected_layers: list[str | LayerId] | tuple[str | LayerId, ...] | None = None,
    plateau_policy: PlateauPolicy = PlateauPolicy.LAST_IN_PLATEAU,
    pre_context_sec: float = 0.0,
    post_context_sec: float = 0.0,
    style: DrawConfig | None = None,
    notes: tuple[str, ...] = (),
) -> S2PoseQcExportPlan:
    """Build an S2 pose QC plan from an inclusive prepared-frame interval."""

    interval = PreparedFrameInterval(
        start=int(prepared_frame_idx_start),
        end=int(prepared_frame_idx_end),
    )
    bounds = compute_clip_bounds(
        prepared_frame_idx_start=interval.start,
        prepared_frame_idx_end=interval.end,
        source_fps=float(source_fps),
        source_n_frames=int(source_n_frames),
        pre_context_sec=float(pre_context_sec),
        post_context_sec=float(post_context_sec),
    )
    export_interval = PreparedFrameInterval(
        start=int(bounds.clip_frame_idx_start),
        end=int(bounds.clip_frame_idx_end),
    )
    layers = (
        validate_layer_selection(list(selected_layers))
        if selected_layers is not None
        else default_s2_pose_qc_layers(include_frame_counter=include_frame_counter)
    )
    if LayerId.S2_POSE_INFERRED not in layers:
        raise OverlayContractError(
            "S2 pose QC plan requires the s2.pose_inferred layer."
        )
    draw_style = style or DrawConfig(
        darken_alpha=0.90,
        draw_skeleton=draw_skeleton,
        node_radius=5,
        edge_thickness=2,
        legend_text=True,
        extra={"show_confidence": bool(show_confidence)},
    )
    return S2PoseQcExportPlan(
        scope_kind=OverlayExportScopeKind.PREPARED_INTERVAL,
        prepared_interval=export_interval,
        selected_layers=layers,
        plateau_policy=plateau_policy,
        draw_skeleton=draw_skeleton,
        show_confidence=bool(show_confidence),
        include_frame_counter=include_frame_counter,
        style=draw_style,
        pre_context_sec=float(pre_context_sec),
        post_context_sec=float(post_context_sec),
        clip_bounds=bounds,
        notes=notes,
    )


@dataclass(frozen=True, slots=True)
class S3PoseQcExportPlan:
    """Explicit offline prepared-rate S3 accepted tracked-pose QC export plan."""

    scope_kind: OverlayExportScopeKind
    prepared_interval: PreparedFrameInterval
    selected_layers: tuple[LayerId, ...]
    plateau_policy: PlateauPolicy = PlateauPolicy.LAST_IN_PLATEAU
    frame_domain: FrameDomain = FrameDomain.PREPARED
    draw_skeleton: bool = True
    show_machine_corrections: bool = True
    show_manual_corrections: bool = True
    include_frame_counter: bool = True
    style: DrawConfig = field(default_factory=DrawConfig)
    pre_context_sec: float = 0.0
    post_context_sec: float = 0.0
    clip_bounds: ClipBounds | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_kind": self.scope_kind.value,
            "prepared_interval": self.prepared_interval.to_dict(),
            "selected_layers": [x.value for x in self.selected_layers],
            "plateau_policy": self.plateau_policy.value,
            "frame_domain": self.frame_domain.value,
            "draw_skeleton": self.draw_skeleton,
            "show_machine_corrections": self.show_machine_corrections,
            "show_manual_corrections": self.show_manual_corrections,
            "include_frame_counter": self.include_frame_counter,
            "pre_context_sec": self.pre_context_sec,
            "post_context_sec": self.post_context_sec,
            "clip_bounds": None if self.clip_bounds is None else self.clip_bounds.to_dict(),
            "style": self.style.to_dict(),
            "notes": list(self.notes),
        }


def plan_s3_prepared_interval(
    *,
    prepared_frame_idx_start: int,
    prepared_frame_idx_end: int,
    source_fps: float,
    source_n_frames: int,
    include_frame_counter: bool = True,
    draw_skeleton: bool = True,
    show_machine_corrections: bool = True,
    show_manual_corrections: bool = True,
    selected_layers: list[str | LayerId] | tuple[str | LayerId, ...] | None = None,
    plateau_policy: PlateauPolicy = PlateauPolicy.LAST_IN_PLATEAU,
    pre_context_sec: float = 0.0,
    post_context_sec: float = 0.0,
    style: DrawConfig | None = None,
    notes: tuple[str, ...] = (),
) -> S3PoseQcExportPlan:
    """Build an S3 pose QC plan from an inclusive prepared-frame interval."""

    interval = PreparedFrameInterval(
        start=int(prepared_frame_idx_start),
        end=int(prepared_frame_idx_end),
    )
    bounds = compute_clip_bounds(
        prepared_frame_idx_start=interval.start,
        prepared_frame_idx_end=interval.end,
        source_fps=float(source_fps),
        source_n_frames=int(source_n_frames),
        pre_context_sec=float(pre_context_sec),
        post_context_sec=float(post_context_sec),
    )
    export_interval = PreparedFrameInterval(
        start=int(bounds.clip_frame_idx_start),
        end=int(bounds.clip_frame_idx_end),
    )
    layers = (
        validate_layer_selection(list(selected_layers))
        if selected_layers is not None
        else default_s3_pose_qc_layers(include_frame_counter=include_frame_counter)
    )
    if LayerId.S3_POSE_TRACKED not in layers:
        raise OverlayContractError(
            "S3 pose QC plan requires the s3.pose_tracked layer."
        )
    draw_style = style or DrawConfig(
        darken_alpha=0.90,
        draw_skeleton=draw_skeleton,
        node_radius=5,
        edge_thickness=2,
        legend_text=True,
        extra={
            "show_machine_corrections": bool(show_machine_corrections),
            "show_manual_corrections": bool(show_manual_corrections),
        },
    )
    return S3PoseQcExportPlan(
        scope_kind=OverlayExportScopeKind.PREPARED_INTERVAL,
        prepared_interval=export_interval,
        selected_layers=layers,
        plateau_policy=plateau_policy,
        draw_skeleton=draw_skeleton,
        show_machine_corrections=bool(show_machine_corrections),
        show_manual_corrections=bool(show_manual_corrections),
        include_frame_counter=include_frame_counter,
        style=draw_style,
        pre_context_sec=float(pre_context_sec),
        post_context_sec=float(post_context_sec),
        clip_bounds=bounds,
        notes=notes,
    )


@dataclass(frozen=True, slots=True)
class S5SegmentationQcExportPlan:
    """Explicit offline prepared-rate S5 blob-segmentation QC export plan."""

    scope_kind: OverlayExportScopeKind
    prepared_interval: PreparedFrameInterval
    selected_layers: tuple[LayerId, ...]
    plateau_policy: PlateauPolicy = PlateauPolicy.LAST_IN_PLATEAU
    frame_domain: FrameDomain = FrameDomain.PREPARED
    show_contours: bool = True
    show_masks: bool = False
    show_scalar_text: bool = False
    mask_opacity: float = 0.35
    include_frame_counter: bool = True
    style: DrawConfig = field(default_factory=DrawConfig)
    pre_context_sec: float = 0.0
    post_context_sec: float = 0.0
    clip_bounds: ClipBounds | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_kind": self.scope_kind.value,
            "prepared_interval": self.prepared_interval.to_dict(),
            "selected_layers": [x.value for x in self.selected_layers],
            "plateau_policy": self.plateau_policy.value,
            "frame_domain": self.frame_domain.value,
            "show_contours": self.show_contours,
            "show_masks": self.show_masks,
            "show_scalar_text": self.show_scalar_text,
            "mask_opacity": float(self.mask_opacity),
            "include_frame_counter": self.include_frame_counter,
            "pre_context_sec": self.pre_context_sec,
            "post_context_sec": self.post_context_sec,
            "clip_bounds": None if self.clip_bounds is None else self.clip_bounds.to_dict(),
            "style": self.style.to_dict(),
            "notes": list(self.notes),
        }


def plan_s5_prepared_interval(
    *,
    prepared_frame_idx_start: int,
    prepared_frame_idx_end: int,
    source_fps: float,
    source_n_frames: int,
    include_frame_counter: bool = True,
    show_contours: bool = True,
    show_masks: bool = False,
    show_scalar_text: bool = False,
    mask_opacity: float = 0.35,
    selected_layers: list[str | LayerId] | tuple[str | LayerId, ...] | None = None,
    plateau_policy: PlateauPolicy = PlateauPolicy.LAST_IN_PLATEAU,
    pre_context_sec: float = 0.0,
    post_context_sec: float = 0.0,
    style: DrawConfig | None = None,
    notes: tuple[str, ...] = (),
) -> S5SegmentationQcExportPlan:
    """Build an S5 segmentation QC plan from an inclusive prepared-frame interval."""

    interval = PreparedFrameInterval(
        start=int(prepared_frame_idx_start),
        end=int(prepared_frame_idx_end),
    )
    bounds = compute_clip_bounds(
        prepared_frame_idx_start=interval.start,
        prepared_frame_idx_end=interval.end,
        source_fps=float(source_fps),
        source_n_frames=int(source_n_frames),
        pre_context_sec=float(pre_context_sec),
        post_context_sec=float(post_context_sec),
    )
    export_interval = PreparedFrameInterval(
        start=int(bounds.clip_frame_idx_start),
        end=int(bounds.clip_frame_idx_end),
    )
    layers = (
        validate_layer_selection(list(selected_layers))
        if selected_layers is not None
        else default_s5_segmentation_qc_layers(
            include_frame_counter=include_frame_counter
        )
    )
    if LayerId.S5_SEGMENTATION not in layers:
        raise OverlayContractError(
            "S5 segmentation QC plan requires the s5.segmentation layer."
        )
    opacity = float(mask_opacity)
    if opacity < 0.0 or opacity > 1.0:
        raise OverlayContractError(
            f"mask_opacity must be in [0, 1]; got {opacity}."
        )
    draw_style = style or DrawConfig(
        darken_alpha=0.95,
        legend_text=True,
        extra={
            "show_contours": bool(show_contours),
            "show_masks": bool(show_masks),
            "show_scalar_text": bool(show_scalar_text),
            "mask_opacity": opacity,
            "contour_thickness": 2,
        },
    )
    return S5SegmentationQcExportPlan(
        scope_kind=OverlayExportScopeKind.PREPARED_INTERVAL,
        prepared_interval=export_interval,
        selected_layers=layers,
        plateau_policy=plateau_policy,
        show_contours=bool(show_contours),
        show_masks=bool(show_masks),
        show_scalar_text=bool(show_scalar_text),
        mask_opacity=opacity,
        include_frame_counter=include_frame_counter,
        style=draw_style,
        pre_context_sec=float(pre_context_sec),
        post_context_sec=float(post_context_sec),
        clip_bounds=bounds,
        notes=notes,
    )


@dataclass(frozen=True, slots=True)
class MixedPoseBehaviorQcExportPlan:
    """Explicit offline prepared-rate mixed pose + behavioral QC export plan."""

    scope_kind: OverlayExportScopeKind
    prepared_interval: PreparedFrameInterval
    pose_source_kind: PoseSourceKind
    hud_mode: HudDirectionMode
    selected_layers: tuple[LayerId, ...]
    plateau_policy: PlateauPolicy = PlateauPolicy.LAST_IN_PLATEAU
    frame_domain: FrameDomain = FrameDomain.PREPARED
    draw_skeleton: bool = True
    include_contact: bool = True
    include_sbs: bool = True
    include_frame_counter: bool = True
    style: DrawConfig = field(default_factory=DrawConfig)
    pre_context_sec: float = 0.0
    post_context_sec: float = 0.0
    clip_bounds: ClipBounds | None = None
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope_kind": self.scope_kind.value,
            "prepared_interval": self.prepared_interval.to_dict(),
            "pose_source_kind": self.pose_source_kind.value,
            "hud_mode": self.hud_mode.value,
            "selected_layers": [x.value for x in self.selected_layers],
            "plateau_policy": self.plateau_policy.value,
            "frame_domain": self.frame_domain.value,
            "draw_skeleton": self.draw_skeleton,
            "include_contact": self.include_contact,
            "include_sbs": self.include_sbs,
            "include_frame_counter": self.include_frame_counter,
            "pre_context_sec": self.pre_context_sec,
            "post_context_sec": self.post_context_sec,
            "clip_bounds": None if self.clip_bounds is None else self.clip_bounds.to_dict(),
            "style": self.style.to_dict(),
            "notes": list(self.notes),
        }


def plan_mixed_prepared_interval(
    *,
    prepared_frame_idx_start: int,
    prepared_frame_idx_end: int,
    source_fps: float,
    source_n_frames: int,
    pose_source_kind: PoseSourceKind | str,
    hud_mode: HudDirectionMode | str = HudDirectionMode.BOTH,
    include_contact: bool = True,
    include_sbs: bool = True,
    include_frame_counter: bool = True,
    draw_skeleton: bool = True,
    selected_layers: list[str | LayerId] | tuple[str | LayerId, ...] | None = None,
    plateau_policy: PlateauPolicy = PlateauPolicy.LAST_IN_PLATEAU,
    pre_context_sec: float = 0.0,
    post_context_sec: float = 0.0,
    style: DrawConfig | None = None,
    notes: tuple[str, ...] = (),
) -> MixedPoseBehaviorQcExportPlan:
    """Build a mixed pose + behavioral QC plan from an inclusive prepared interval."""

    pose_kind = (
        pose_source_kind
        if isinstance(pose_source_kind, PoseSourceKind)
        else PoseSourceKind(str(pose_source_kind))
    )
    mode = (
        hud_mode
        if isinstance(hud_mode, HudDirectionMode)
        else HudDirectionMode(str(hud_mode))
    )
    interval = PreparedFrameInterval(
        start=int(prepared_frame_idx_start),
        end=int(prepared_frame_idx_end),
    )
    bounds = compute_clip_bounds(
        prepared_frame_idx_start=interval.start,
        prepared_frame_idx_end=interval.end,
        source_fps=float(source_fps),
        source_n_frames=int(source_n_frames),
        pre_context_sec=float(pre_context_sec),
        post_context_sec=float(post_context_sec),
    )
    export_interval = PreparedFrameInterval(
        start=int(bounds.clip_frame_idx_start),
        end=int(bounds.clip_frame_idx_end),
    )
    layers = (
        validate_layer_selection(list(selected_layers))
        if selected_layers is not None
        else default_mixed_pose_behavior_qc_layers(
            pose_source_kind=pose_kind,
            hud_mode=mode,
            include_contact=include_contact,
            include_sbs=include_sbs,
            include_frame_counter=include_frame_counter,
        )
    )
    required_pose = (
        LayerId.S4_POSE_FINAL
        if pose_kind is PoseSourceKind.S4_FINALIZED
        else LayerId.S3_POSE_TRACKED
    )
    if required_pose not in layers:
        raise OverlayContractError(
            f"Mixed QC plan requires the {required_pose.value} pose layer for "
            f"{pose_kind.value}."
        )
    if LayerId.S4_POSE_FINAL in layers and LayerId.S3_POSE_TRACKED in layers:
        raise OverlayContractError(
            "Mixed QC plan must not select both S3 and S4 pose layers."
        )
    if pose_kind is PoseSourceKind.S4_FINALIZED and LayerId.S3_POSE_TRACKED in layers:
        raise OverlayContractError(
            "S4 mixed plan must not include s3.pose_tracked."
        )
    if pose_kind is PoseSourceKind.S3_ACCEPTED and LayerId.S4_POSE_FINAL in layers:
        raise OverlayContractError(
            "S3 mixed plan must not include s4.pose_final "
            "(refusing silent finalization implication)."
        )
    draw_style = style or DrawConfig(
        darken_alpha=0.90,
        draw_skeleton=draw_skeleton,
        node_radius=5,
        edge_thickness=2,
        legend_text=True,
        hud_direction_mode=mode.value,
        contact_led_enabled=include_contact,
        sbs_led_enabled=include_sbs,
    )
    return MixedPoseBehaviorQcExportPlan(
        scope_kind=OverlayExportScopeKind.PREPARED_INTERVAL,
        prepared_interval=export_interval,
        pose_source_kind=pose_kind,
        hud_mode=mode,
        selected_layers=layers,
        plateau_policy=plateau_policy,
        draw_skeleton=draw_skeleton,
        include_contact=include_contact,
        include_sbs=include_sbs,
        include_frame_counter=include_frame_counter,
        style=draw_style,
        pre_context_sec=float(pre_context_sec),
        post_context_sec=float(post_context_sec),
        clip_bounds=bounds,
        notes=notes,
    )
