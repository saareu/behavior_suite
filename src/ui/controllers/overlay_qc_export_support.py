"""QC overlay export GUI settings, request identity, and table helpers (C2-C)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

from overlay.contracts import HudDirectionMode
from overlay.discovery import OverlayQcJobSummary

OverlayQcExportFamily = Literal["behavioral", "s4_pose", "s2_pose", "s3_pose"]
OverlayQcBehaviorSource = Literal["s8_automated", "s10_reviewed"]
OverlayQcScopeKind = Literal["prepared_interval", "bout_span"]


class _S9PackageOption(Protocol):
    """Minimal S9 discovery fields needed for S8 overlay launch planning."""

    @property
    def run_id(self) -> str: ...

    @property
    def run_dir(self) -> Path: ...

    @property
    def source_s8_run_id(self) -> str | None: ...


@dataclass(frozen=True, slots=True)
class OverlayQcUiSettings:
    """GUI working settings snapshotted into each overlay QC export request."""

    export_family: OverlayQcExportFamily = "behavioral"
    source_kind: OverlayQcBehaviorSource = "s8_automated"
    scope_kind: OverlayQcScopeKind = "prepared_interval"
    prepared_frame_idx_start: int = 0
    prepared_frame_idx_end: int = 29
    bout_id: str | None = None
    start_sample_idx: int | None = None
    end_sample_idx: int | None = None
    packaged_prepared_start: int | None = None
    packaged_prepared_end: int | None = None
    hud_mode: str = HudDirectionMode.BOTH.value
    include_contact: bool = True
    include_sbs: bool = True
    include_frame_counter: bool = True
    draw_skeleton: bool = True
    show_confidence: bool = False
    show_machine_corrections: bool = True
    show_manual_corrections: bool = True
    pre_context_sec: float = 0.0
    post_context_sec: float = 0.0

    def validated(self) -> OverlayQcUiSettings:
        family = str(self.export_family).strip().lower()
        if family not in {"behavioral", "s4_pose", "s2_pose", "s3_pose"}:
            raise ValueError(
                "export_family must be behavioral, s4_pose, s2_pose, or s3_pose, "
                f"got {family!r}."
            )
        source = str(self.source_kind).strip().lower()
        if source not in {"s8_automated", "s10_reviewed"}:
            raise ValueError(
                "Behavioral source_kind must be s8_automated or s10_reviewed, "
                f"got {source!r}."
            )
        scope = str(self.scope_kind).strip().lower()
        if scope not in {"prepared_interval", "bout_span"}:
            raise ValueError(f"Unsupported scope_kind {scope!r}.")
        if float(self.pre_context_sec) < 0 or float(self.post_context_sec) < 0:
            raise ValueError("Context padding seconds must be >= 0.")
        hud = str(self.hud_mode).strip().lower()
        try:
            HudDirectionMode(hud)
        except ValueError as exc:
            raise ValueError(
                "hud_mode must be focal_0_only, focal_1_only, or both."
            ) from exc

        start = int(self.prepared_frame_idx_start)
        end = int(self.prepared_frame_idx_end)
        if family in {"s4_pose", "s2_pose", "s3_pose"} or scope == "prepared_interval":
            if start < 0 or end < 0:
                raise ValueError("Prepared frame indices must be >= 0.")
            if start > end:
                raise ValueError(
                    f"Prepared interval start ({start}) > end ({end})."
                )

        bout_id = None if self.bout_id is None else str(self.bout_id).strip() or None
        start_sample = (
            None if self.start_sample_idx is None else int(self.start_sample_idx)
        )
        end_sample = None if self.end_sample_idx is None else int(self.end_sample_idx)
        if family == "behavioral" and scope == "bout_span":
            if start_sample is None or end_sample is None:
                raise ValueError(
                    "Bout-span QC export requires start_sample_idx and end_sample_idx."
                )
            if start_sample > end_sample:
                raise ValueError(
                    f"Dense bout start ({start_sample}) > end ({end_sample})."
                )

        return replace(
            self,
            export_family=family,  # type: ignore[arg-type]
            source_kind=source,  # type: ignore[arg-type]
            scope_kind=scope,  # type: ignore[arg-type]
            prepared_frame_idx_start=start,
            prepared_frame_idx_end=end,
            bout_id=bout_id,
            start_sample_idx=start_sample,
            end_sample_idx=end_sample,
            packaged_prepared_start=(
                None
                if self.packaged_prepared_start is None
                else int(self.packaged_prepared_start)
            ),
            packaged_prepared_end=(
                None
                if self.packaged_prepared_end is None
                else int(self.packaged_prepared_end)
            ),
            hud_mode=hud,
            include_contact=bool(self.include_contact),
            include_sbs=bool(self.include_sbs),
            include_frame_counter=bool(self.include_frame_counter),
            draw_skeleton=bool(self.draw_skeleton),
            show_confidence=bool(self.show_confidence),
            show_machine_corrections=bool(self.show_machine_corrections),
            show_manual_corrections=bool(self.show_manual_corrections),
            pre_context_sec=float(self.pre_context_sec),
            post_context_sec=float(self.post_context_sec),
        )

    def scope_summary(self) -> str:
        settings = self.validated()
        if settings.export_family == "s4_pose":
            return (
                f"S4 pose QC prepared [{settings.prepared_frame_idx_start}, "
                f"{settings.prepared_frame_idx_end}]"
            )
        if settings.export_family == "s2_pose":
            return (
                f"S2 pose QC prepared [{settings.prepared_frame_idx_start}, "
                f"{settings.prepared_frame_idx_end}]"
            )
        if settings.export_family == "s3_pose":
            return (
                f"S3 pose QC prepared [{settings.prepared_frame_idx_start}, "
                f"{settings.prepared_frame_idx_end}]"
            )
        if settings.scope_kind == "bout_span":
            bout = settings.bout_id or "(unnamed)"
            return (
                f"Behavioral bout {bout} dense "
                f"[{settings.start_sample_idx}, {settings.end_sample_idx}]"
            )
        return (
            f"Behavioral prepared [{settings.prepared_frame_idx_start}, "
            f"{settings.prepared_frame_idx_end}]"
        )


@dataclass(frozen=True, slots=True)
class OverlayQcRequestIdentity:
    """Identity used to reject stale asynchronous overlay QC results."""

    request_id: str
    session_root: Path
    export_family: str
    source_kind: str | None
    package_run_id: str | None
    package_dir: Path | None
    review_run_id: str | None
    review_run_dir: Path | None
    s4_run_id: str | None
    s4_run_dir: Path | None
    s2_run_id: str | None
    s2_run_dir: Path | None
    s3_run_id: str | None
    s3_run_dir: Path | None
    settings: OverlayQcUiSettings

    @staticmethod
    def new(
        *,
        session_root: Path,
        export_family: str,
        source_kind: str | None,
        package_run_id: str | None,
        package_dir: Path | None,
        review_run_id: str | None,
        review_run_dir: Path | None,
        s4_run_id: str | None,
        s4_run_dir: Path | None,
        s2_run_id: str | None = None,
        s2_run_dir: Path | None = None,
        s3_run_id: str | None = None,
        s3_run_dir: Path | None = None,
        settings: OverlayQcUiSettings,
    ) -> OverlayQcRequestIdentity:
        return OverlayQcRequestIdentity(
            request_id=str(uuid4()),
            session_root=session_root.resolve(strict=False),
            export_family=str(export_family),
            source_kind=None if source_kind is None else str(source_kind),
            package_run_id=None if package_run_id is None else str(package_run_id),
            package_dir=(
                package_dir.resolve(strict=False) if package_dir is not None else None
            ),
            review_run_id=None if review_run_id is None else str(review_run_id),
            review_run_dir=(
                review_run_dir.resolve(strict=False)
                if review_run_dir is not None
                else None
            ),
            s4_run_id=None if s4_run_id is None else str(s4_run_id),
            s4_run_dir=(
                s4_run_dir.resolve(strict=False) if s4_run_dir is not None else None
            ),
            s2_run_id=None if s2_run_id is None else str(s2_run_id),
            s2_run_dir=(
                s2_run_dir.resolve(strict=False) if s2_run_dir is not None else None
            ),
            s3_run_id=None if s3_run_id is None else str(s3_run_id),
            s3_run_dir=(
                s3_run_dir.resolve(strict=False) if s3_run_dir is not None else None
            ),
            settings=settings.validated(),
        )


@dataclass(frozen=True, slots=True)
class OverlayQcLaunchContext:
    """Prefill context when opening the shared QC Overlay Export dialog."""

    session_root: Path
    export_family: OverlayQcExportFamily
    package_dir: Path | None = None
    package_run_id: str | None = None
    review_run_dir: Path | None = None
    review_run_id: str | None = None
    review_is_draft: bool | None = None
    s4_run_dir: Path | None = None
    s4_run_id: str | None = None
    s2_run_dir: Path | None = None
    s2_run_id: str | None = None
    s3_run_dir: Path | None = None
    s3_run_id: str | None = None
    lock_family: bool = True
    lock_source_kind: bool = False
    locked_source_kind: OverlayQcBehaviorSource | None = None
    preferred_bout_id: str | None = None
    preferred_start_sample_idx: int | None = None
    preferred_end_sample_idx: int | None = None
    preferred_prepared_start: int | None = None
    preferred_prepared_end: int | None = None
    package_options: tuple[tuple[str, Path], ...] = ()
    review_options: tuple[tuple[str, Path, bool, str | None], ...] = ()
    s4_options: tuple[tuple[str, Path], ...] = ()
    s2_options: tuple[tuple[str, Path], ...] = ()
    s3_options: tuple[tuple[str, Path], ...] = ()
    skeleton_nodes_only_disclosure: str | None = None
    unaccepted_s3_message: str | None = None


@dataclass(frozen=True, slots=True)
class S8BehavioralOverlayLaunchPlan:
    """S8 page opener outcome: either a shared-dialog launch or a clear block."""

    launch: OverlayQcLaunchContext | None = None
    block_title: str | None = None
    block_message: str | None = None

    @property
    def blocked(self) -> bool:
        return self.launch is None


_S8_NO_ELIGIBLE_PACKAGE_MESSAGE = (
    "Behavioral QC overlay export resolves automated S8 evidence only through an "
    "eligible S9 results package that pins this S8 run. Package this S8 run in "
    "Results Packaging (S9) first, then reopen Export QC Overlay Video."
)

_S8_NO_SESSION_PACKAGES_MESSAGE = (
    "Behavioral QC overlay export requires an eligible S9 results package. "
    "No currently eligible S9 package was found in this session. Package an S8 "
    "run in Results Packaging (S9) first."
)


def _s8_behavioral_launch_context(
    *,
    session_root: Path,
    package_dir: Path | None,
    package_run_id: str | None,
    package_options: tuple[tuple[str, Path], ...],
) -> OverlayQcLaunchContext:
    """Shared S8 opener context: behavioral family + locked automated source."""

    return OverlayQcLaunchContext(
        session_root=session_root,
        export_family="behavioral",
        package_dir=package_dir,
        package_run_id=package_run_id,
        lock_family=True,
        lock_source_kind=True,
        locked_source_kind="s8_automated",
        package_options=package_options,
        preferred_prepared_start=0,
        preferred_prepared_end=29,
    )


def plan_s8_behavioral_overlay_launch(
    *,
    session_root: Path | str,
    selected_s8_run_id: str | None,
    eligible_s9_runs: Sequence[_S9PackageOption],
) -> S8BehavioralOverlayLaunchPlan:
    """Build S8-page launch context for automated behavioral QC via eligible S9.

    Preserves the exporter contract: never uses a raw S8 directory, never prefills
    an unrelated package, and never substitutes a reviewed S10 source. Locks the
    dialog behavioral source kind to ``s8_automated`` so S8 entry cannot switch to
    published S10. Prefills a package only when exactly one eligible S9 package
    pins the selected S8 run. Otherwise leaves package selection explicit.
    """

    root = Path(session_root).expanduser().resolve(strict=False)
    eligible = tuple(eligible_s9_runs)
    selected = None if selected_s8_run_id is None else str(selected_s8_run_id).strip()
    if selected == "":
        selected = None

    if selected is not None:
        matching = tuple(
            run
            for run in eligible
            if run.source_s8_run_id is not None
            and str(run.source_s8_run_id).strip() == selected
        )
        if not matching:
            return S8BehavioralOverlayLaunchPlan(
                block_title="No eligible S9 package",
                block_message=_S8_NO_ELIGIBLE_PACKAGE_MESSAGE,
            )
        package_options = tuple((run.run_id, Path(run.run_dir)) for run in matching)
        if len(matching) == 1:
            only = matching[0]
            return S8BehavioralOverlayLaunchPlan(
                launch=_s8_behavioral_launch_context(
                    session_root=root,
                    package_dir=Path(only.run_dir),
                    package_run_id=str(only.run_id),
                    package_options=package_options,
                )
            )
        return S8BehavioralOverlayLaunchPlan(
            launch=_s8_behavioral_launch_context(
                session_root=root,
                package_dir=None,
                package_run_id=None,
                package_options=package_options,
            )
        )

    if not eligible:
        return S8BehavioralOverlayLaunchPlan(
            block_title="No eligible S9 package",
            block_message=_S8_NO_SESSION_PACKAGES_MESSAGE,
        )
    return S8BehavioralOverlayLaunchPlan(
        launch=_s8_behavioral_launch_context(
            session_root=root,
            package_dir=None,
            package_run_id=None,
            package_options=tuple(
                (run.run_id, Path(run.run_dir)) for run in eligible
            ),
        )
    )


@dataclass(frozen=True, slots=True)
class OverlayQcBoutOption:
    bout_id: str
    start_sample_idx: int
    end_sample_idx: int
    prepared_frame_idx_start: int | None
    prepared_frame_idx_end: int | None
    channel: str | None
    behavior_name: str | None

    @property
    def label(self) -> str:
        channel = self.channel or "?"
        behavior = self.behavior_name or "?"
        return (
            f"{self.bout_id} · {channel}/{behavior} · dense "
            f"[{self.start_sample_idx},{self.end_sample_idx}]"
        )


def overlay_qc_table_rows(
    jobs: tuple[OverlayQcJobSummary, ...] | list[OverlayQcJobSummary],
) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []
    for job in jobs:
        if job.export_family == "s4_pose":
            source = job.s4_run_id
        elif job.export_family == "s2_pose":
            source = job.s2_run_id
        elif job.export_family == "s3_pose":
            source = job.s3_run_id
        else:
            source = job.review_run_id or job.package_run_id or job.source_kind or "—"
        interval = "—"
        if (
            job.prepared_frame_idx_start is not None
            and job.prepared_frame_idx_end is not None
        ):
            interval = f"[{job.prepared_frame_idx_start}, {job.prepared_frame_idx_end}]"
        layers = ",".join(job.layers) if job.layers else "—"
        rows.append(
            (
                job.job_id,
                job.export_family,
                job.status,
                str(source),
                interval,
                layers,
                str(job.frames_written) if job.frames_written is not None else "—",
                "yes" if job.has_video else "no",
                str(job.job_dir),
            )
        )
    return tuple(rows)


def source_identity_summary(
    *,
    settings: OverlayQcUiSettings,
    package_run_id: str | None,
    review_run_id: str | None,
    s4_run_id: str | None,
    s2_run_id: str | None = None,
    s3_run_id: str | None = None,
) -> str:
    cfg = settings.validated()
    parts = [f"family={cfg.export_family}"]
    if cfg.export_family == "s4_pose":
        parts.append(f"s4={s4_run_id or '—'}")
        parts.append(f"skeleton={'yes' if cfg.draw_skeleton else 'no'}")
    elif cfg.export_family == "s2_pose":
        parts.append(f"s2={s2_run_id or '—'}")
        parts.append(f"skeleton={'yes' if cfg.draw_skeleton else 'no'}")
        parts.append(f"confidence={'yes' if cfg.show_confidence else 'no'}")
    elif cfg.export_family == "s3_pose":
        parts.append(f"s3={s3_run_id or '—'}")
        parts.append(f"skeleton={'yes' if cfg.draw_skeleton else 'no'}")
        parts.append(f"auto={'yes' if cfg.show_machine_corrections else 'no'}")
        parts.append(f"manual={'yes' if cfg.show_manual_corrections else 'no'}")
    else:
        parts.append(f"source={cfg.source_kind}")
        parts.append(f"package={package_run_id or '—'}")
        if cfg.source_kind == "s10_reviewed":
            parts.append(f"review={review_run_id or '—'}")
        parts.append(f"hud={cfg.hud_mode}")
        parts.append(f"contact={'yes' if cfg.include_contact else 'no'}")
        parts.append(f"sbs={'yes' if cfg.include_sbs else 'no'}")
    parts.append(cfg.scope_summary())
    parts.append(f"pad={cfg.pre_context_sec:g}/{cfg.post_context_sec:g}s")
    parts.append(f"counter={'yes' if cfg.include_frame_counter else 'no'}")
    return "; ".join(parts)


def plan_s2_pose_overlay_launch(
    *,
    session_root: Path | str,
    selected_s2_run_id: str | None,
    selected_s2_run_dir: Path | None,
    eligible_s2_runs: Sequence[tuple[str, Path]],
) -> OverlayQcLaunchContext:
    """Build S2-page launch context for inferred-pose QC via eligible S2 runs.

    Prefills only when the selected run is unambiguous/eligible. Never auto-picks
    the newest S2 run.
    """

    root = Path(session_root).expanduser().resolve(strict=False)
    options = tuple((str(run_id), Path(run_dir)) for run_id, run_dir in eligible_s2_runs)
    prefill_id: str | None = None
    prefill_dir: Path | None = None
    if selected_s2_run_id is not None and selected_s2_run_dir is not None:
        selected_id = str(selected_s2_run_id).strip()
        selected_dir = Path(selected_s2_run_dir)
        if any(run_id == selected_id for run_id, _ in options) or (
            options and len(options) == 1 and options[0][0] == selected_id
        ):
            prefill_id = selected_id
            prefill_dir = selected_dir
        else:
            # Selected run may still be valid even if not yet in options list.
            prefill_id = selected_id
            prefill_dir = selected_dir
            if all(run_id != selected_id for run_id, _ in options):
                options = ((selected_id, selected_dir),) + options
    elif len(options) == 1:
        prefill_id, prefill_dir = options[0]

    return OverlayQcLaunchContext(
        session_root=root,
        export_family="s2_pose",
        s2_run_dir=prefill_dir,
        s2_run_id=prefill_id,
        lock_family=True,
        s2_options=options,
        preferred_prepared_start=0,
        preferred_prepared_end=29,
        skeleton_nodes_only_disclosure=(
            "If SLEAP topology cannot be read, export is nodes-only "
            "(no invented edges); disclosed in the QC manifest/log."
        ),
    )


@dataclass(frozen=True, slots=True)
class S3PoseOverlayLaunchPlan:
    """S3 page opener outcome: shared-dialog launch or an actionable block."""

    launch: OverlayQcLaunchContext | None = None
    block_title: str | None = None
    block_message: str | None = None

    @property
    def blocked(self) -> bool:
        return self.launch is None


_S3_UNACCEPTED_MESSAGE = (
    "S3 Tracking Correction QC export requires an accepted S3 run "
    "(acceptance_state == accepted) with authoritative tracked_pose.parquet. "
    "Accept Tracking first, or select a different accepted run. "
    "Working-only / not_accepted / superseded runs are not eligible."
)


def plan_s3_pose_overlay_launch(
    *,
    session_root: Path | str,
    selected_s3_run_id: str | None,
    selected_s3_run_dir: Path | None,
    selected_acceptance_state: str | None,
    accepted_s3_runs: Sequence[tuple[str, Path]],
) -> S3PoseOverlayLaunchPlan:
    """Build S3-page launch context for accepted tracked-pose QC.

    Prefills only an explicitly selected accepted run (or the sole accepted
    option). Never auto-picks newest. Blocks unaccepted selections with an
    actionable explanation.
    """

    root = Path(session_root).expanduser().resolve(strict=False)
    options = tuple((str(run_id), Path(run_dir)) for run_id, run_dir in accepted_s3_runs)
    selected_id = (
        None if selected_s3_run_id is None else str(selected_s3_run_id).strip() or None
    )
    selected_dir = (
        None if selected_s3_run_dir is None else Path(selected_s3_run_dir)
    )
    selected_state = (
        None
        if selected_acceptance_state is None
        else str(selected_acceptance_state).strip() or None
    )

    if selected_id is not None and selected_dir is not None:
        in_accepted = any(run_id == selected_id for run_id, _ in options)
        if not in_accepted:
            # accepted_s3_runs is the dialog authority — never inject a run that
            # discovery did not list, even if page state claims "accepted".
            return S3PoseOverlayLaunchPlan(
                block_title="S3 run not accepted",
                block_message=(
                    f"Selected run {selected_id!r} has acceptance_state="
                    f"{selected_state!r}. {_S3_UNACCEPTED_MESSAGE}"
                ),
            )
        prefill_id = selected_id
        prefill_dir = selected_dir
    elif len(options) == 1:
        prefill_id, prefill_dir = options[0]
    else:
        prefill_id, prefill_dir = None, None

    if not options and prefill_dir is None:
        return S3PoseOverlayLaunchPlan(
            block_title="No accepted S3 run",
            block_message=_S3_UNACCEPTED_MESSAGE,
        )

    return S3PoseOverlayLaunchPlan(
        launch=OverlayQcLaunchContext(
            session_root=root,
            export_family="s3_pose",
            s3_run_dir=prefill_dir,
            s3_run_id=prefill_id,
            lock_family=True,
            s3_options=options,
            preferred_prepared_start=0,
            preferred_prepared_end=29,
            skeleton_nodes_only_disclosure=(
                "If SLEAP topology cannot be read from S2 lineage, export is "
                "nodes-only (no invented edges); disclosed in the QC manifest/log."
            ),
        )
    )


def actionable_overlay_error(exc: BaseException) -> str:
    """Map backend exceptions to short operator-facing messages."""

    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()
    if "draft" in lowered:
        return (
            "Working drafts cannot be used as QC overlay sources. "
            "Publish the review first, or choose an automated S9 package."
        )
    if "digest mismatch" in lowered or "digest changed" in lowered:
        return (
            "Source digest mismatch — the selected science artifacts changed. "
            f"Re-select an eligible source. Details: {message}"
        )
    if "package_dir" in lowered or "ineligible" in lowered or "not eligible" in lowered:
        return f"Ineligible source: {message}"
    if "prepared_video" in lowered or "prepared sync" in lowered or "sync" in lowered:
        return f"Missing or invalid prepared video/sync: {message}"
    if "final_pose" in lowered or ("s4" in lowered and "missing" in lowered):
        return f"Missing S4 pose artifacts: {message}"
    if "pose.parquet" in lowered or ("s2" in lowered and "missing" in lowered):
        return f"Missing or ineligible S2 pose artifacts: {message}"
    if (
        "tracked_pose" in lowered
        or "acceptance" in lowered
        or ("s3" in lowered and ("missing" in lowered or "not eligible" in lowered))
    ):
        return f"Missing or ineligible accepted S3 tracked pose: {message}"
    if "states" in lowered and ("s8" in lowered or "s10" in lowered or "missing" in lowered):
        return f"Missing S8/S10 behavioral states: {message}"
    if "topology" in lowered or "skeleton" in lowered or "sleap" in lowered:
        return (
            f"Skeleton topology unavailable (nodes-only fallback may apply): {message}"
        )
    if "cancel" in lowered:
        return f"QC overlay export canceled: {message}"
    if "ffmpeg" in lowered or "encode" in lowered:
        return f"FFmpeg encode failed: {message}"
    return message


__all__ = [
    "OverlayQcBoutOption",
    "OverlayQcExportFamily",
    "OverlayQcBehaviorSource",
    "OverlayQcLaunchContext",
    "OverlayQcRequestIdentity",
    "OverlayQcScopeKind",
    "OverlayQcUiSettings",
    "S3PoseOverlayLaunchPlan",
    "S8BehavioralOverlayLaunchPlan",
    "actionable_overlay_error",
    "overlay_qc_table_rows",
    "plan_s2_pose_overlay_launch",
    "plan_s3_pose_overlay_launch",
    "plan_s8_behavioral_overlay_launch",
    "source_identity_summary",
]
