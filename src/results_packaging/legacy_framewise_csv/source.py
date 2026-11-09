"""Resolve pinned sources for legacy framewise CSV (S9 automated / S10 reviewed)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_review.contracts import (
    PUBLICATION_STATE_DRAFT,
    REVIEWED_BEHAVIOR_STATES_FILENAME,
)
from behavioral_review.contracts import (
    PublicationError as ReviewPublicationError,
)
from behavioral_review.discovery import reopen_reviewed_run
from behavioral_review.sync import SyncMap
from pose_finalization.accepted_s3 import (
    list_accepted_s3_runs,
    validate_accepted_s3_run,
)
from pose_finalization.contracts import FINAL_POSE_FILENAME, RUN_META_FILENAME
from pose_finalization.dense_pose import load_tracked_pose_table
from pose_finalization.runner import (
    PoseFinalizationError,
    validate_s4_run_for_downstream,
)
from preprocess.sync_writer import load_prepared_sync_npz
from project.file_digest import sha256_file
from results_packaging.legacy_framewise_csv.contracts import (
    BEHAVIOR_MAPPING_PREPARED_NEAREST_DENSE,
    BEHAVIOR_SOURCE_AUTOMATED,
    BEHAVIOR_SOURCE_REVIEWED,
    BEHAVIOR_TIE_BREAK_EARLIER,
    INDEXING_NOTES_V2,
    LEGACY_FRAMEWISE_PROFILE_ID,
    LEGACY_FRAMEWISE_V2_PROFILE_ID,
    LEGACY_FRAMEWISE_V2_SEMANTICS_REVISION,
    LEGACY_POSE_NODE_TRACKS,
    PLATEAU_POLICY_LAST_IN_PLATEAU,
    LegacyFramewiseEligibilityError,
    LegacyFramewiseSourceError,
    is_legacy_framewise_profile,
)
from results_packaging.media_csv_export.source import (
    ExportSourcePackage,
    validate_package_for_export,
)
from results_packaging.sync_index import (
    build_sync_index_from_prepared_sync,
    validate_sync_index,
)

PoseSelectionMode = Literal["lineage_s6", "explicit"]
S3SelectionMode = Literal["lineage_s4_source_s3", "explicit"]


@dataclass(frozen=True, slots=True)
class LegacyFramewiseSource:
    """Pinned inputs for one legacy framewise export."""

    behavior_source_kind: str
    session_root: Path
    package_dir: Path
    package_run_id: str
    review_run_dir: Path | None
    review_run_id: str | None
    sync_index: pd.DataFrame
    sync_index_path: Path
    sync_index_sha256: str
    behavior_states: pd.DataFrame
    behavior_states_path: Path
    behavior_states_sha256: str
    final_pose: pd.DataFrame
    final_pose_path: Path
    final_pose_sha256: str
    pose_run_id: str | None
    pose_run_dir: Path | None
    pose_selection_mode: PoseSelectionMode
    original_frame_mapping_provenance: dict[str, Any]
    package_manifest_sha256: str | None
    warnings: tuple[str, ...]
    digests: dict[str, str | None]
    pose_alignment: dict[str, Any]
    # Prepared-frame / independent timing authorities (required for v2).
    format_profile: str = LEGACY_FRAMEWISE_PROFILE_ID
    prepared_sync_path: Path | None = None
    prepared_sync_sha256: str | None = None
    prepared_frame_idx: np.ndarray | None = None
    raw_decode_frame_idx: np.ndarray | None = None
    prepared_time_sec: np.ndarray | None = None
    n_prepared_frames: int | None = None
    behavior_sync: SyncMap | None = None
    pose_sync: SyncMap | None = None
    pose_sync_build_audit: dict[str, Any] | None = None
    timing_contract: dict[str, Any] | None = None
    # Accepted S3 tracked pose (required for v2; unused for v1).
    tracked_pose: pd.DataFrame | None = None
    tracked_pose_path: Path | None = None
    tracked_pose_sha256: str | None = None
    s3_run_id: str | None = None
    s3_run_dir: Path | None = None
    s3_selection_mode: S3SelectionMode | None = None


def resolve_automated_legacy_framewise_source(
    *,
    session_root: Path | str,
    package_dir: Path | str,
    allow_stale_source: bool = False,
    pose_run_dir: Path | str | None = None,
    format_profile: str = LEGACY_FRAMEWISE_PROFILE_ID,
) -> LegacyFramewiseSource:
    """Resolve S9 package + S8 states + pose authority for the format profile.

    For ``legacy_framewise_v1``, ``pose_run_dir`` is an explicit S4 run.
    For ``legacy_framewise_v2``, ``pose_run_dir`` is an explicit accepted S3 run
    (used only when S6→S4→source_s3 lineage is unavailable or must match).
    """
    root = Path(session_root).expanduser().resolve()
    package = validate_package_for_export(
        package_dir,
        require_video=False,
        allow_stale_source=allow_stale_source,
    )
    return _assemble_from_package(
        root=root,
        package=package,
        behavior_source_kind=BEHAVIOR_SOURCE_AUTOMATED,
        review_run_dir=None,
        review_run_id=None,
        behavior_states_path=_require_s8_states_path(package),
        pose_run_dir=pose_run_dir,
        format_profile=format_profile,
    )


def resolve_reviewed_legacy_framewise_source(
    *,
    session_root: Path | str,
    review_run_dir: Path | str,
    package_dir: Path | str | None = None,
    allow_stale_source: bool = False,
    pose_run_dir: Path | str | None = None,
    format_profile: str = LEGACY_FRAMEWISE_PROFILE_ID,
) -> LegacyFramewiseSource:
    """Resolve published S10 reviewed states + parent S9 package + pose authority.

    Draft / unpublished reviews are refused. Pose selection follows the same
    v1=S4 / v2=accepted-S3 rules as the automated path.
    """

    root = Path(session_root).expanduser().resolve()
    review_dir = Path(review_run_dir).expanduser().resolve()
    try:
        opened = reopen_reviewed_run(review_dir)
    except (ReviewPublicationError, OSError, ValueError) as exc:
        raise LegacyFramewiseEligibilityError(str(exc)) from exc

    meta = opened.get("run_meta") or {}
    publication_state = str(meta.get("publication_state") or "").strip().lower()
    if publication_state == PUBLICATION_STATE_DRAFT:
        raise LegacyFramewiseEligibilityError(
            f"Refusing unpublished/draft S10 review for final CSV export: {review_dir}"
        )

    states_path = review_dir / REVIEWED_BEHAVIOR_STATES_FILENAME
    if not states_path.is_file():
        raise LegacyFramewiseEligibilityError(
            f"Published reviewed_behavior_states.parquet missing: {states_path}"
        )

    validation = opened.get("validation") or {}
    parent_package = Path(
        str(validation.get("parent_package_dir") or meta.get("parent_package_dir"))
    ).expanduser().resolve()
    if package_dir is not None:
        explicit = Path(package_dir).expanduser().resolve()
        if explicit != parent_package:
            raise LegacyFramewiseSourceError(
                "package_dir does not match the published review parent package: "
                f"{explicit} vs {parent_package}"
            )
    package = validate_package_for_export(
        parent_package,
        require_video=False,
        allow_stale_source=allow_stale_source,
    )
    return _assemble_from_package(
        root=root,
        package=package,
        behavior_source_kind=BEHAVIOR_SOURCE_REVIEWED,
        review_run_dir=review_dir,
        review_run_id=str(opened.get("review_run_id") or review_dir.name),
        behavior_states_path=states_path,
        pose_run_dir=pose_run_dir,
        format_profile=format_profile,
    )


def _assemble_from_package(
    *,
    root: Path,
    package: ExportSourcePackage,
    behavior_source_kind: str,
    review_run_dir: Path | None,
    review_run_id: str | None,
    behavior_states_path: Path,
    pose_run_dir: Path | str | None,
    format_profile: str = LEGACY_FRAMEWISE_PROFILE_ID,
) -> LegacyFramewiseSource:
    profile = str(format_profile or LEGACY_FRAMEWISE_PROFILE_ID).strip()
    if not is_legacy_framewise_profile(profile):
        raise LegacyFramewiseEligibilityError(
            f"Unsupported legacy framewise format_profile {profile!r}."
        )

    sync_path = package.package_dir / "sync_index.parquet"
    if not sync_path.is_file():
        raise LegacyFramewiseEligibilityError(f"sync_index.parquet missing: {sync_path}")
    sync = pd.read_parquet(sync_path)
    try:
        validate_sync_index(sync)
    except Exception as exc:  # noqa: BLE001 - surface as eligibility
        raise LegacyFramewiseEligibilityError(
            f"Invalid sync_index.parquet: {exc}"
        ) from exc
    if "raw_decode_frame_idx" not in sync.columns:
        raise LegacyFramewiseEligibilityError(
            "sync_index lacks raw_decode_frame_idx; original-video frame mapping "
            "is unavailable for legacy framewise CSV (refusing dense+trim guess)."
        )

    if not behavior_states_path.is_file():
        raise LegacyFramewiseEligibilityError(
            f"behavior states parquet missing: {behavior_states_path}"
        )
    states = pd.read_parquet(behavior_states_path)

    warnings: list[str] = []
    if package.stale_warning:
        warnings.append(package.stale_warning)

    prepared_sync_path: Path | None = None
    prepared_sync_sha: str | None = None
    prepared_frame_idx: np.ndarray | None = None
    raw_decode_frame_idx: np.ndarray | None = None
    prepared_time_sec: np.ndarray | None = None
    n_prepared_frames: int | None = None
    behavior_sync: SyncMap | None = None
    pose_sync: SyncMap | None = None
    pose_sync_audit: dict[str, Any] | None = None
    timing_contract: dict[str, Any] | None = None
    tracked_pose: pd.DataFrame | None = None
    tracked_pose_path: Path | None = None
    tracked_pose_sha256: str | None = None
    s3_run_id: str | None = None
    s3_run_dir: Path | None = None
    s3_selection_mode: S3SelectionMode | None = None

    if profile == LEGACY_FRAMEWISE_V2_PROFILE_ID:
        s3_resolved = _resolve_accepted_s3_pose(
            root=root,
            package=package,
            s3_run_dir=pose_run_dir,
        )
        warnings.extend(s3_resolved.warnings)
        tracked_pose = s3_resolved.table
        tracked_pose_path = s3_resolved.tracked_pose_path
        tracked_pose_sha256 = s3_resolved.tracked_pose_sha256
        s3_run_id = s3_resolved.s3_run_id
        s3_run_dir = s3_resolved.s3_run_dir
        s3_selection_mode = s3_resolved.selection_mode
        prepared_sync_path = _resolve_prepared_sync_path(
            root=root,
            s3_prepared_sync_path=s3_resolved.prepared_sync_path,
        )
        try:
            prepared = load_prepared_sync_npz(prepared_sync_path)
        except Exception as exc:  # noqa: BLE001
            raise LegacyFramewiseEligibilityError(
                f"Authoritative S1 prepared_sync.npz could not be loaded for "
                f"legacy_framewise_v2: {exc}"
            ) from exc
        prepared_frame_idx = np.asarray(prepared.prepared_frame_idx, dtype=np.int64)
        raw_decode_frame_idx = np.asarray(prepared.raw_decode_frame_idx, dtype=np.int64)
        prepared_time_sec = np.asarray(prepared.prepared_time_sec, dtype=np.float64)
        n_prepared_frames = int(prepared_frame_idx.shape[0])
        prepared_sync_sha = sha256_file(prepared_sync_path)
        if s3_resolved.prepared_sync_path.resolve() != prepared_sync_path.resolve():
            raise LegacyFramewiseSourceError(
                "Accepted S3 prepared_sync path disagrees with the session "
                f"authoritative S1 prepared_sync: {s3_resolved.prepared_sync_path} "
                f"vs {prepared_sync_path}."
            )
        try:
            behavior_sync = SyncMap.from_sync_index(sync)
        except Exception as exc:  # noqa: BLE001
            raise LegacyFramewiseEligibilityError(
                "Behavioral SyncMap could not be built from package sync_index "
                f"for prepared→nearest-dense timing: {exc}"
            ) from exc
        timing_contract = {
            "row_domain": "s1_prepared_video_frames",
            "semantics_revision": LEGACY_FRAMEWISE_V2_SEMANTICS_REVISION,
            "behavior_timing_authority": "results_package_sync_index_time_sec",
            "pose_timing_authority": "accepted_s3_tracked_pose_prepared_frame_idx",
            "behavior_mapping_policy": BEHAVIOR_MAPPING_PREPARED_NEAREST_DENSE,
            "behavior_tie_break": BEHAVIOR_TIE_BREAK_EARLIER,
            "behavior_extrapolation": False,
            "pose_authority": "accepted_s3_tracked_pose",
            "pose_resampling": False,
            "dense_indices_assumed_equal": False,
            "n_prepared_frames": n_prepared_frames,
            "behavior_n_dense": int(behavior_sync.n_dense),
            "prepared_sync_path": str(prepared_sync_path),
            "prepared_sync_sha256": prepared_sync_sha,
            "notes": INDEXING_NOTES_V2,
        }
        # V1-only S4 fields remain populated with placeholders so callers that
        # read pose_run_* for display do not crash; V2 publication uses s3_*.
        pose_resolved = _ResolvedPose(
            table=pd.DataFrame(),
            final_pose_path=tracked_pose_path,
            final_pose_sha256=tracked_pose_sha256,
            pose_run_id=s3_run_id,
            pose_run_dir=s3_run_dir,
            selection_mode="explicit",
            alignment=dict(s3_resolved.alignment),
            warnings=[],
        )
        mapping_prov = {
            "authority": "s1_prepared_sync.raw_decode_frame_idx",
            "upstream": "s1_prepared_sync.npz",
            "indexing": "0_based_raw_decode_order",
            "prepared_indexing": "0_based_prepared_video",
            "dense_indexing": "nullable_behavioral_prepared_nearest_dense",
            "coordinate_domain": "prepared_video_pixels",
            "coordinate_authority": "accepted_s3_tracked_pose",
            "frame_domain": "original_video_raw_decode_idx",
            "row_domain": "s1_prepared_video_frames",
            "semantics_revision": LEGACY_FRAMEWISE_V2_SEMANTICS_REVISION,
            "notes": INDEXING_NOTES_V2,
        }
        digests: dict[str, str | None] = {
            "sync_index_sha256": sha256_file(sync_path),
            "behavior_states_sha256": sha256_file(behavior_states_path),
            "tracked_pose_sha256": tracked_pose_sha256,
            "package_manifest_sha256": package.package_manifest_sha256,
            "prepared_sync_sha256": prepared_sync_sha,
        }
    else:
        pose_resolved = _resolve_final_pose(
            root=root,
            package=package,
            pose_run_dir=pose_run_dir,
            sync_index=sync,
            require_independent_pose_timing=False,
        )
        warnings.extend(pose_resolved.warnings)
        mapping_prov = {
            "authority": "results_package_sync_index.raw_decode_frame_idx",
            "upstream": "s1_prepared_sync.raw_decode_frame_idx[prepared_frame_idx]",
            "indexing": "0_based_raw_decode_order",
            "dense_indexing": "0_based_contiguous",
            "coordinate_domain": "prepared_video_pixels",
            "frame_domain": "original_video_raw_decode_idx",
            "notes": (
                "frame column is original-video decode index from S1 via S9 sync_index; "
                "prepared_frame_idx is not exported as frame. Pose x/y remain in "
                "prepared-video pixel space (different domain from frame)."
            ),
        }
        digests = {
            "sync_index_sha256": sha256_file(sync_path),
            "behavior_states_sha256": sha256_file(behavior_states_path),
            "final_pose_sha256": pose_resolved.final_pose_sha256,
            "package_manifest_sha256": package.package_manifest_sha256,
            "prepared_sync_sha256": None,
        }

    return LegacyFramewiseSource(
        behavior_source_kind=behavior_source_kind,
        session_root=root,
        package_dir=package.package_dir,
        package_run_id=package.package_run_id,
        review_run_dir=review_run_dir,
        review_run_id=review_run_id,
        sync_index=sync,
        sync_index_path=sync_path,
        sync_index_sha256=str(digests["sync_index_sha256"]),
        behavior_states=states,
        behavior_states_path=behavior_states_path,
        behavior_states_sha256=str(digests["behavior_states_sha256"]),
        final_pose=pose_resolved.table,
        final_pose_path=pose_resolved.final_pose_path,
        final_pose_sha256=pose_resolved.final_pose_sha256,
        pose_run_id=pose_resolved.pose_run_id,
        pose_run_dir=pose_resolved.pose_run_dir,
        pose_selection_mode=pose_resolved.selection_mode,
        original_frame_mapping_provenance=mapping_prov,
        package_manifest_sha256=package.package_manifest_sha256,
        warnings=tuple(warnings),
        digests=digests,
        pose_alignment=dict(pose_resolved.alignment),
        format_profile=profile,
        prepared_sync_path=prepared_sync_path,
        prepared_sync_sha256=prepared_sync_sha,
        prepared_frame_idx=prepared_frame_idx,
        raw_decode_frame_idx=raw_decode_frame_idx,
        prepared_time_sec=prepared_time_sec,
        n_prepared_frames=n_prepared_frames,
        behavior_sync=behavior_sync,
        pose_sync=pose_sync,
        pose_sync_build_audit=pose_sync_audit,
        timing_contract=timing_contract,
        tracked_pose=tracked_pose,
        tracked_pose_path=tracked_pose_path,
        tracked_pose_sha256=tracked_pose_sha256,
        s3_run_id=s3_run_id,
        s3_run_dir=s3_run_dir,
        s3_selection_mode=s3_selection_mode,
    )


def _require_s8_states_path(package: ExportSourcePackage) -> Path:
    # Prefer package-validated external reference when present.
    refs = getattr(package, "source_references", None)
    if isinstance(refs, dict):
        s8 = refs.get("source_s8") or {}
        candidate = s8.get("behavior_states")
        if candidate:
            path = Path(str(candidate)).expanduser().resolve()
            if path.is_file():
                return path
    # Fall back to package_manifest / ExportSourcePackage attributes.
    for attr in ("behavior_states_path", "s8_behavior_states_path"):
        value = getattr(package, attr, None)
        if value:
            path = Path(value).expanduser().resolve()
            if path.is_file():
                return path

    manifest_path = package.package_dir / "package_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        source_refs = manifest.get("source_references") or {}
        s8 = source_refs.get("source_s8") or {}
        candidate = s8.get("behavior_states")
        if candidate:
            path = Path(str(candidate)).expanduser().resolve()
            if path.is_file():
                return path
        for item in manifest.get("external_references") or []:
            if item.get("role") == "s8_behavior_states" and item.get("absolute_path"):
                path = Path(str(item["absolute_path"])).expanduser().resolve()
                if path.is_file():
                    return path
    raise LegacyFramewiseEligibilityError(
        "Could not resolve S8 behavior_states.parquet from the S9 package pin."
    )


@dataclass(frozen=True, slots=True)
class _ResolvedPose:
    table: pd.DataFrame
    final_pose_path: Path
    final_pose_sha256: str
    pose_run_id: str
    pose_run_dir: Path
    selection_mode: PoseSelectionMode
    alignment: dict[str, Any]
    warnings: list[str]


def _resolve_final_pose(
    *,
    root: Path,
    package: ExportSourcePackage,
    pose_run_dir: Path | str | None,
    sync_index: pd.DataFrame,
    require_independent_pose_timing: bool = False,
) -> _ResolvedPose:
    """Resolve S4 from exact S6 lineage, else require an explicit S4 run.

    Never silently substitutes the latest eligible S4 run.
    """

    lineage = _lineage_s4_from_package(package)
    explicit_dir = (
        Path(pose_run_dir).expanduser().resolve() if pose_run_dir is not None else None
    )

    if lineage is not None and explicit_dir is not None:
        lineage_dir = lineage["run_dir"]
        if explicit_dir != lineage_dir:
            raise LegacyFramewiseSourceError(
                "Explicit pose_run_dir disagrees with S6 lineage-pinned S4: "
                f"{explicit_dir} vs {lineage_dir}. Refusing ambiguous pose source."
            )

    if lineage is not None:
        return _load_and_validate_pose_run(
            root=root,
            run_dir=lineage["run_dir"],
            expected_path=lineage.get("final_pose"),
            expected_sha256=lineage.get("final_pose_sha256"),
            expected_run_id=lineage.get("run_id"),
            selection_mode="lineage_s6",
            sync_index=sync_index,
            require_independent_pose_timing=require_independent_pose_timing,
        )

    if explicit_dir is None:
        raise LegacyFramewiseEligibilityError(
            "S6 lineage pin for S4 final_pose is unavailable. Provide an explicit "
            "pose_run_dir (completed S4 run) after validating session/timeline/"
            "track-node schema alignment. Automatic latest-S4 fallback is disabled."
        )

    return _load_and_validate_pose_run(
        root=root,
        run_dir=explicit_dir,
        expected_path=None,
        expected_sha256=None,
        expected_run_id=None,
        selection_mode="explicit",
        sync_index=sync_index,
        require_independent_pose_timing=require_independent_pose_timing,
    )


@dataclass(frozen=True, slots=True)
class PoseLineageInfo:
    """Whether S6→S4 lineage is available for a package (never auto-selects S4)."""

    available: bool
    run_id: str | None
    run_dir: Path | None
    final_pose_sha256: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class PoseRunCandidate:
    """One completed S4 run identity offered for explicit GUI selection."""

    run_id: str
    run_dir: Path
    status: str | None
    classification: str
    label: str


def inspect_package_pose_lineage(
    package_dir: Path | str | ExportSourcePackage,
) -> PoseLineageInfo:
    """Return lineage availability for GUI default vs explicit S4 picker."""

    if isinstance(package_dir, ExportSourcePackage):
        package = package_dir
    else:
        pkg_path = Path(package_dir).expanduser().resolve()
        package = ExportSourcePackage(
            package_dir=pkg_path,
            package_run_id=pkg_path.name,
            package_status="unknown",
            session_root=None,
            run_meta=_read_json_object(pkg_path / "run_meta.json"),
            package_manifest=_read_json_object(pkg_path / "package_manifest.json"),
            behavior_bouts=pd.DataFrame(),
            behavior_bouts_sha256="",
            sync_index_sha256="",
            package_manifest_sha256="",
            source_s8_run_id=None,
            source_s8_run_dir=None,
            source_s7_run_id=None,
            source_s6_run_id=None,
            source_video_path=None,
            source_video_kind="",
            source_video_fingerprint=None,
            source_video_sha256=None,
            source_video_verification_strength=None,
            currently_eligible=False,
            stale_warning=None,
            video_probe=None,
        )

    lineage = _lineage_s4_from_package(package)
    if lineage is None:
        return PoseLineageInfo(
            available=False,
            run_id=None,
            run_dir=None,
            final_pose_sha256=None,
            reason=(
                "S6 lineage pin for S4 final_pose is unavailable. Select a "
                "completed S4 run explicitly; automatic latest-S4 fallback is "
                "disabled."
            ),
        )
    return PoseLineageInfo(
        available=True,
        run_id=str(lineage["run_id"]),
        run_dir=Path(lineage["run_dir"]),
        final_pose_sha256=(
            str(lineage["final_pose_sha256"])
            if lineage.get("final_pose_sha256")
            else None
        ),
        reason=f"Using S6 lineage-pinned S4 run {lineage['run_id']}.",
    )


def list_pose_run_candidates(
    session_root: Path | str,
) -> tuple[PoseRunCandidate, ...]:
    """List completed S4 run identities for explicit V1 selection.

    Candidates are ordered newest-first for display only. Callers must never
    auto-select the newest run when lineage is missing.
    """

    from pose_finalization.run_discovery import summarize_pose_finalization_project

    summary = summarize_pose_finalization_project(session_root)
    out: list[PoseRunCandidate] = []
    for run in summary.completed_runs:
        label = (
            f"{run.run_id} · {run.classification}"
            + (f" · status={run.status}" if run.status else "")
        )
        out.append(
            PoseRunCandidate(
                run_id=run.run_id,
                run_dir=Path(run.run_dir).expanduser().resolve(),
                status=run.status,
                classification=run.classification,
                label=label,
            )
        )
    return tuple(out)


@dataclass(frozen=True, slots=True)
class S3LineageInfo:
    """Whether S6→S4→accepted-S3 lineage is available for a package."""

    available: bool
    run_id: str | None
    run_dir: Path | None
    tracked_pose_sha256: str | None
    reason: str


@dataclass(frozen=True, slots=True)
class S3RunCandidate:
    """One accepted S3 run identity offered for explicit GUI selection."""

    run_id: str
    run_dir: Path
    acceptance_state: str
    label: str


def inspect_package_s3_lineage(
    package_dir: Path | str | ExportSourcePackage,
) -> S3LineageInfo:
    """Return accepted-S3 lineage availability for V2 (never auto-selects newest)."""

    if isinstance(package_dir, ExportSourcePackage):
        package = package_dir
    else:
        pkg_path = Path(package_dir).expanduser().resolve()
        package = ExportSourcePackage(
            package_dir=pkg_path,
            package_run_id=pkg_path.name,
            package_status="unknown",
            session_root=None,
            run_meta=_read_json_object(pkg_path / "run_meta.json"),
            package_manifest=_read_json_object(pkg_path / "package_manifest.json"),
            behavior_bouts=pd.DataFrame(),
            behavior_bouts_sha256="",
            sync_index_sha256="",
            package_manifest_sha256="",
            source_s8_run_id=None,
            source_s8_run_dir=None,
            source_s7_run_id=None,
            source_s6_run_id=None,
            source_video_path=None,
            source_video_kind="",
            source_video_fingerprint=None,
            source_video_sha256=None,
            source_video_verification_strength=None,
            currently_eligible=False,
            stale_warning=None,
            video_probe=None,
        )

    lineage = _lineage_s3_from_package(package)
    if lineage is None:
        return S3LineageInfo(
            available=False,
            run_id=None,
            run_dir=None,
            tracked_pose_sha256=None,
            reason=(
                "S6→S4→source_s3 lineage pin for accepted S3 tracked_pose is "
                "unavailable. Select an accepted S3 run explicitly; automatic "
                "latest-S3 fallback is disabled."
            ),
        )
    return S3LineageInfo(
        available=True,
        run_id=str(lineage["run_id"]),
        run_dir=Path(lineage["run_dir"]),
        tracked_pose_sha256=(
            str(lineage["tracked_pose_sha256"])
            if lineage.get("tracked_pose_sha256")
            else None
        ),
        reason=f"Using S6→S4 lineage-pinned accepted S3 run {lineage['run_id']}.",
    )


def list_accepted_s3_run_candidates(
    session_root: Path | str,
) -> tuple[S3RunCandidate, ...]:
    """List validated accepted S3 runs for explicit V2 selection.

    Newest-first for display only. Callers must never auto-select the newest
    run when lineage is missing.
    """

    try:
        accepted = list_accepted_s3_runs(session_root)
    except PoseFinalizationError:
        return ()
    out: list[S3RunCandidate] = []
    for run in accepted:
        out.append(
            S3RunCandidate(
                run_id=run.run_id,
                run_dir=run.run_dir,
                acceptance_state=run.acceptance_state,
                label=f"{run.run_id} · accepted",
            )
        )
    return tuple(out)


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def _lineage_s3_from_package(package: ExportSourcePackage) -> dict[str, Any] | None:
    """Resolve S6→S4→source_s3 pin for accepted tracked pose."""

    s4 = _lineage_s4_from_package(package)
    if s4 is None:
        return None
    s4_dir = Path(s4["run_dir"]).expanduser().resolve()
    meta_path = s4_dir / RUN_META_FILENAME
    if not meta_path.is_file():
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    source_s3 = meta.get("source_s3") or {}
    if not isinstance(source_s3, dict) or not source_s3.get("run_dir"):
        return None
    run_dir = Path(str(source_s3["run_dir"])).expanduser().resolve()
    tracked = source_s3.get("tracked_pose_parquet") or source_s3.get("tracked_pose")
    return {
        "run_dir": run_dir,
        "run_id": str(source_s3.get("run_id") or run_dir.name),
        "tracked_pose": (
            Path(str(tracked)).expanduser().resolve() if tracked else None
        ),
        "tracked_pose_sha256": (
            str(source_s3["tracked_pose_sha256"])
            if source_s3.get("tracked_pose_sha256")
            else None
        ),
        "prepared_sync": (
            Path(str(source_s3["prepared_sync"])).expanduser().resolve()
            if source_s3.get("prepared_sync")
            else None
        ),
        "session_root": (
            Path(str(source_s3["session_root"])).expanduser().resolve()
            if source_s3.get("session_root")
            else None
        ),
        "s4_run_dir": s4_dir,
        "s4_run_id": str(s4.get("run_id") or s4_dir.name),
    }


def _lineage_s4_from_package(package: ExportSourcePackage) -> dict[str, Any] | None:
    """Resolve S6→S4 pin from package manifest / run_meta → S6 run_meta.source_s4."""

    s6_dir_text: str | None = None
    manifest = getattr(package, "package_manifest", None) or {}
    if not isinstance(manifest, dict) or not manifest:
        manifest = _read_json_object(package.package_dir / "package_manifest.json")
    source_refs = manifest.get("source_references") or {}
    s6 = source_refs.get("source_s6") or {}
    if isinstance(s6, dict):
        s6_dir_text = s6.get("run_dir")

    if not s6_dir_text:
        run_meta = getattr(package, "run_meta", None) or {}
        if not isinstance(run_meta, dict) or not run_meta:
            run_meta = _read_json_object(package.package_dir / "run_meta.json")
        s6_meta = run_meta.get("source_s6") or {}
        if isinstance(s6_meta, dict):
            s6_dir_text = s6_meta.get("run_dir")

    if not s6_dir_text:
        return None
    s6_dir = Path(str(s6_dir_text)).expanduser().resolve()
    meta_path = s6_dir / "run_meta.json"
    if not meta_path.is_file():
        return None
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    source_s4 = meta.get("source_s4") or {}
    if not isinstance(source_s4, dict) or not source_s4.get("run_dir"):
        return None
    run_dir = Path(str(source_s4["run_dir"])).expanduser().resolve()
    final_pose = source_s4.get("final_pose")
    return {
        "run_dir": run_dir,
        "run_id": str(source_s4.get("run_id") or run_dir.name),
        "final_pose": (
            Path(str(final_pose)).expanduser().resolve() if final_pose else None
        ),
        "final_pose_sha256": (
            str(source_s4["final_pose_sha256"])
            if source_s4.get("final_pose_sha256")
            else None
        ),
        "session_root": (
            Path(str(source_s4["session_root"])).expanduser().resolve()
            if source_s4.get("session_root")
            else None
        ),
    }


def _load_and_validate_pose_run(
    *,
    root: Path,
    run_dir: Path,
    expected_path: Path | None,
    expected_sha256: str | None,
    expected_run_id: str | None,
    selection_mode: PoseSelectionMode,
    sync_index: pd.DataFrame,
    require_independent_pose_timing: bool = False,
) -> _ResolvedPose:
    warnings: list[str] = []
    try:
        validate_s4_run_for_downstream(run_dir)
    except PoseFinalizationError as exc:
        raise LegacyFramewiseSourceError(
            f"Selected S4 run is not eligible for export: {exc}"
        ) from exc

    meta_path = run_dir / RUN_META_FILENAME
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    pose_path = (run_dir / FINAL_POSE_FILENAME).resolve()
    if expected_path is not None:
        expected_resolved = expected_path.expanduser().resolve()
        if not expected_resolved.is_file():
            raise LegacyFramewiseEligibilityError(
                f"S6-pinned final_pose missing on disk: {expected_resolved}"
            )
        if expected_resolved != pose_path:
            raise LegacyFramewiseSourceError(
                "S6 lineage final_pose path does not match the S4 run artifact: "
                f"{expected_resolved} vs {pose_path}"
            )
    if not pose_path.is_file():
        raise LegacyFramewiseEligibilityError(
            f"final_pose.parquet missing for selected S4 run: {pose_path}"
        )

    digest = sha256_file(pose_path)
    if expected_sha256 is not None and digest.lower() != expected_sha256.lower():
        raise LegacyFramewiseSourceError(
            "Selected final_pose SHA256 does not match the S6 lineage pin: "
            f"{digest} vs {expected_sha256}"
        )

    run_id = str(expected_run_id or meta.get("run_id") or run_dir.name)
    session_from_meta = meta.get("session_root") or (meta.get("source_s3") or {}).get(
        "session_root"
    )
    if session_from_meta:
        meta_root = Path(str(session_from_meta)).expanduser().resolve()
        if meta_root != root:
            raise LegacyFramewiseSourceError(
                "Selected S4 session_root does not match the export session: "
                f"{meta_root} vs {root}"
            )

    table = pd.read_parquet(pose_path)
    if require_independent_pose_timing:
        alignment = _validate_pose_schema_for_independent_timing(table)
        alignment["independent_timing"] = True
        alignment["behavioral_dense_equality_required"] = False
    else:
        alignment = _validate_pose_alignment(table, sync_index=sync_index)
        alignment["independent_timing"] = False
        alignment["behavioral_dense_equality_required"] = True
    return _ResolvedPose(
        table=table,
        final_pose_path=pose_path,
        final_pose_sha256=digest,
        pose_run_id=run_id,
        pose_run_dir=run_dir,
        selection_mode=selection_mode,
        alignment=alignment,
        warnings=warnings,
    )


@dataclass(frozen=True, slots=True)
class _ResolvedS3Pose:
    table: pd.DataFrame
    tracked_pose_path: Path
    tracked_pose_sha256: str
    s3_run_id: str
    s3_run_dir: Path
    prepared_sync_path: Path
    selection_mode: S3SelectionMode
    alignment: dict[str, Any]
    warnings: list[str]


def _resolve_accepted_s3_pose(
    *,
    root: Path,
    package: ExportSourcePackage,
    s3_run_dir: Path | str | None,
) -> _ResolvedS3Pose:
    """Resolve accepted S3 tracked_pose from exact lineage, else explicit pin.

    Never silently substitutes the newest accepted S3 run.
    """

    lineage = _lineage_s3_from_package(package)
    explicit_dir = (
        Path(s3_run_dir).expanduser().resolve() if s3_run_dir is not None else None
    )

    if lineage is not None and explicit_dir is not None:
        lineage_dir = Path(lineage["run_dir"]).expanduser().resolve()
        if explicit_dir != lineage_dir:
            raise LegacyFramewiseSourceError(
                "Explicit s3/pose_run_dir disagrees with S6→S4 lineage-pinned "
                f"accepted S3: {explicit_dir} vs {lineage_dir}. Refusing "
                "ambiguous pose source."
            )

    if lineage is not None:
        return _load_and_validate_accepted_s3(
            root=root,
            run_dir=Path(lineage["run_dir"]),
            expected_path=lineage.get("tracked_pose"),
            expected_sha256=lineage.get("tracked_pose_sha256"),
            expected_run_id=lineage.get("run_id"),
            selection_mode="lineage_s4_source_s3",
        )

    if explicit_dir is None:
        raise LegacyFramewiseEligibilityError(
            "S6→S4→source_s3 lineage pin for accepted S3 tracked_pose is "
            "unavailable. Provide an explicit accepted S3 run directory after "
            "validating session identity, acceptance state, tracked-pose digest, "
            "prepared-video compatibility, and required track/node schema. "
            "Automatic latest-S3 fallback is disabled."
        )

    return _load_and_validate_accepted_s3(
        root=root,
        run_dir=explicit_dir,
        expected_path=None,
        expected_sha256=None,
        expected_run_id=None,
        selection_mode="explicit",
    )


def _load_and_validate_accepted_s3(
    *,
    root: Path,
    run_dir: Path,
    expected_path: Path | None,
    expected_sha256: str | None,
    expected_run_id: str | None,
    selection_mode: S3SelectionMode,
) -> _ResolvedS3Pose:
    warnings: list[str] = []
    try:
        accepted = validate_accepted_s3_run(run_dir)
    except PoseFinalizationError as exc:
        raise LegacyFramewiseSourceError(
            f"Selected S3 run is not eligible accepted tracked pose for export: {exc}"
        ) from exc

    if accepted.session_root != root:
        raise LegacyFramewiseSourceError(
            "Selected accepted S3 session_root does not match the export session: "
            f"{accepted.session_root} vs {root}"
        )

    pose_path = accepted.tracked_pose_path
    if expected_path is not None:
        expected_resolved = Path(expected_path).expanduser().resolve()
        if not expected_resolved.is_file():
            raise LegacyFramewiseEligibilityError(
                f"Lineage-pinned tracked_pose missing on disk: {expected_resolved}"
            )
        if expected_resolved != pose_path:
            raise LegacyFramewiseSourceError(
                "Lineage tracked_pose path does not match the accepted S3 artifact: "
                f"{expected_resolved} vs {pose_path}"
            )

    digest = accepted.tracked_pose_sha256
    if expected_sha256 is not None and digest.lower() != str(expected_sha256).lower():
        raise LegacyFramewiseSourceError(
            "Selected tracked_pose SHA256 does not match the lineage pin: "
            f"{digest} vs {expected_sha256}"
        )

    run_id = str(expected_run_id or accepted.run_id or run_dir.name)
    try:
        table = load_tracked_pose_table(pose_path)
    except PoseFinalizationError as exc:
        raise LegacyFramewiseSourceError(str(exc)) from exc

    alignment = _validate_tracked_pose_schema(table)
    alignment["acceptance_state"] = accepted.acceptance_state
    alignment["selection_mode"] = selection_mode
    alignment["prepared_sync_path"] = str(accepted.prepared_sync_path)
    return _ResolvedS3Pose(
        table=table,
        tracked_pose_path=pose_path,
        tracked_pose_sha256=digest,
        s3_run_id=run_id,
        s3_run_dir=accepted.run_dir,
        prepared_sync_path=accepted.prepared_sync_path,
        selection_mode=selection_mode,
        alignment=alignment,
        warnings=warnings,
    )


def _validate_tracked_pose_schema(tracked_pose: pd.DataFrame) -> dict[str, Any]:
    """Validate accepted S3 schema/tracks/nodes for legacy CSV export."""

    required = {"track", "node", "x", "y"}
    missing = sorted(required - set(tracked_pose.columns))
    if missing:
        raise LegacyFramewiseSourceError(
            f"tracked_pose missing required columns: {missing}"
        )
    frame_col = None
    for name in ("prepared_frame_idx", "frame_idx", "frame"):
        if name in tracked_pose.columns:
            frame_col = name
            break
    if frame_col is None:
        raise LegacyFramewiseSourceError(
            "tracked_pose is missing prepared_frame_idx/frame_idx/frame."
        )

    frames = pd.to_numeric(tracked_pose[frame_col], errors="coerce")
    if frames.isna().any():
        raise LegacyFramewiseSourceError(
            "tracked_pose has non-numeric prepared-frame identities."
        )
    frame_vals = frames.astype(np.int64)
    frame_set = {int(v) for v in frame_vals.unique().tolist()}
    if not frame_set:
        raise LegacyFramewiseSourceError("tracked_pose has no prepared-frame identities.")

    nodes = set(str(v) for v in tracked_pose["node"].astype(str).unique().tolist())
    tracks = set()
    for value in tracked_pose["track"].tolist():
        parsed = _parse_track_label(value)
        if parsed is not None:
            tracks.add(parsed)
    required_nodes = {node for node, _track in LEGACY_POSE_NODE_TRACKS}
    missing_nodes = sorted(required_nodes - nodes)
    if missing_nodes:
        raise LegacyFramewiseSourceError(
            f"tracked_pose lacks required pose nodes for legacy export: {missing_nodes}"
        )
    if not {0, 1}.issubset(tracks):
        raise LegacyFramewiseSourceError(
            f"tracked_pose must include tracks 0 and 1; found {sorted(tracks)}"
        )
    return {
        "n_pose_prepared_frames": len(frame_set),
        "prepared_frame_min": int(min(frame_set)),
        "prepared_frame_max": int(max(frame_set)),
        "frame_column": frame_col,
        "nodes": sorted(nodes),
        "tracks": sorted(tracks),
        "schema_ok": True,
        "timing_mode": "prepared_frame_native_s3",
    }


def _parse_track_label(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not np.isfinite(value) or not float(value).is_integer():
            return None
        return int(value)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return int(text)
    lowered = text.lower()
    if lowered.startswith("track_"):
        suffix = text.split("_", maxsplit=1)[1]
        if suffix.isdigit():
            return int(suffix)
    return None


def _resolve_prepared_sync_path(
    *,
    root: Path,
    s3_prepared_sync_path: Path | None = None,
) -> Path:
    """Resolve authoritative S1 prepared_sync for v2 (fail closed)."""

    candidate = (root / "preprocess" / "prepared_sync.npz").resolve()
    if candidate.is_file():
        return candidate
    if s3_prepared_sync_path is not None:
        path = Path(s3_prepared_sync_path).expanduser().resolve()
        if path.is_file():
            return path
    raise LegacyFramewiseEligibilityError(
        "Authoritative S1 prepared_sync.npz missing; legacy_framewise_v2 cannot "
        f"establish the prepared-video frame universe (looked for {candidate})."
    )


def _build_independent_pose_sync(
    *,
    prepared_sync_path: Path,
    final_pose: pd.DataFrame,
    package_run_id: str | None,
) -> tuple[SyncMap, dict[str, Any]]:
    """Build S4 SyncMap from prepared_sync + S4 dense timeline (overlay authority)."""

    unique_dense = _unique_dense_timeline_from_final_pose(final_pose)
    sync_build = build_sync_index_from_prepared_sync(
        prepared_sync_path,
        s8_unique_dense_times=unique_dense,
        package_run_id=package_run_id,
    )
    sync = SyncMap.from_sync_index(sync_build.sync_index)
    audit = {
        **sync_build.audit_dict(),
        "timing_authority": "s1_prepared_sync_nn_map_validated_against_s4_dense",
        "plateau_policy": PLATEAU_POLICY_LAST_IN_PLATEAU,
    }
    return sync, audit


def _unique_dense_timeline_from_final_pose(final_pose: pd.DataFrame) -> pd.DataFrame:
    if final_pose is None or final_pose.empty:
        raise LegacyFramewiseSourceError("final_pose table is empty.")
    if "sample_idx" not in final_pose.columns or "time_sec" not in final_pose.columns:
        raise LegacyFramewiseSourceError(
            "final_pose must include sample_idx and time_sec to build independent "
            "S4 SyncMap timing."
        )
    grouped = (
        final_pose.groupby("sample_idx", sort=True, as_index=False)["time_sec"]
        .first()
        .sort_values("sample_idx", kind="mergesort")
        .reset_index(drop=True)
    )
    grouped["dense_frame_idx"] = grouped["sample_idx"].astype("int64")
    grouped["sample_idx"] = grouped["sample_idx"].astype("int64")
    return grouped[["dense_frame_idx", "sample_idx", "time_sec"]]


def _validate_pose_schema_for_independent_timing(
    final_pose: pd.DataFrame,
) -> dict[str, Any]:
    """Validate S4 schema/timeline for v2 without requiring behavioral dense equality."""

    required = {"sample_idx", "track", "node", "x", "y", "time_sec"}
    missing = sorted(required - set(final_pose.columns))
    if missing:
        raise LegacyFramewiseSourceError(
            f"final_pose missing required columns for independent timing: {missing}"
        )
    samples = pd.to_numeric(final_pose["sample_idx"], errors="coerce")
    if samples.isna().any():
        raise LegacyFramewiseSourceError("final_pose has NaN sample_idx.")
    sample_vals = samples.astype(np.int64)
    sample_set = set(int(v) for v in sample_vals.unique().tolist())
    if not sample_set:
        raise LegacyFramewiseSourceError("final_pose has no sample_idx values.")
    sample_min = int(min(sample_set))
    sample_max = int(max(sample_set))
    expected = set(range(sample_min, sample_max + 1))
    if sample_min != 0 or sample_set != expected:
        raise LegacyFramewiseSourceError(
            "S4 final_pose sample_idx must be contiguous 0-based for independent "
            f"SyncMap timing (min={sample_min}, max={sample_max}, "
            f"n_unique={len(sample_set)})."
        )

    nodes = set(str(v) for v in final_pose["node"].astype(str).unique().tolist())
    tracks = set(
        int(v)
        for v in pd.to_numeric(final_pose["track"], errors="coerce")
        .dropna()
        .astype(np.int64)
    )
    required_nodes = {node for node, _track in LEGACY_POSE_NODE_TRACKS}
    missing_nodes = sorted(required_nodes - nodes)
    if missing_nodes:
        raise LegacyFramewiseSourceError(
            f"final_pose lacks required pose nodes for legacy export: {missing_nodes}"
        )
    if not {0, 1}.issubset(tracks):
        raise LegacyFramewiseSourceError(
            f"final_pose must include tracks 0 and 1; found {sorted(tracks)}"
        )

    present_pairs = {
        (str(node), int(track))
        for node, track in zip(
            final_pose["node"].astype(str).to_numpy(),
            pd.to_numeric(final_pose["track"], errors="coerce")
            .fillna(-1)
            .astype(np.int64)
            .to_numpy(),
            strict=True,
        )
    }
    missing_pairs = [
        f"{node}_{track}"
        for node, track in LEGACY_POSE_NODE_TRACKS
        if (node, track) not in present_pairs
    ]
    if missing_pairs:
        raise LegacyFramewiseSourceError(
            "final_pose lacks required node/track pairs for legacy export: "
            f"{missing_pairs}"
        )

    return {
        "n_pose_samples": len(sample_set),
        "sample_min": sample_min,
        "sample_max": sample_max,
        "nodes": sorted(nodes),
        "tracks": sorted(tracks),
        "timeline_match": None,
        "schema_ok": True,
        "timing_mode": "independent_s4_sync_map",
    }


def _validate_pose_alignment(
    final_pose: pd.DataFrame,
    *,
    sync_index: pd.DataFrame,
) -> dict[str, Any]:
    """Validate sample timeline + required track/node schema against sync_index."""

    required = {"sample_idx", "track", "node", "x", "y"}
    missing = sorted(required - set(final_pose.columns))
    if missing:
        raise LegacyFramewiseSourceError(
            f"final_pose missing required columns for alignment: {missing}"
        )
    if "dense_frame_idx" not in sync_index.columns:
        raise LegacyFramewiseSourceError(
            "sync_index missing dense_frame_idx for pose alignment."
        )

    dense = pd.to_numeric(sync_index["dense_frame_idx"], errors="coerce")
    if dense.isna().any():
        raise LegacyFramewiseSourceError("sync_index has NaN dense_frame_idx.")
    dense_vals = dense.astype(np.int64).to_numpy()
    if pd.Series(dense_vals).duplicated().any():
        raise LegacyFramewiseSourceError(
            "Authoritative sync_index dense_frame_idx must be unique."
        )
    dense_set = set(int(v) for v in dense_vals)

    samples = pd.to_numeric(final_pose["sample_idx"], errors="coerce")
    if samples.isna().any():
        raise LegacyFramewiseSourceError("final_pose has NaN sample_idx.")
    sample_vals = samples.astype(np.int64)
    sample_set = set(int(v) for v in sample_vals.unique().tolist())

    missing_samples = sorted(dense_set - sample_set)
    extra_samples = sorted(sample_set - dense_set)
    if missing_samples or extra_samples:
        raise LegacyFramewiseSourceError(
            "S4 final_pose sample_idx set does not match authoritative "
            f"sync_index dense_frame_idx "
            f"(missing={len(missing_samples)}, extra={len(extra_samples)}; "
            f"missing_examples={missing_samples[:5]}, "
            f"extra_examples={extra_samples[:5]}). "
            "Refusing silent pose substitution or dense renumbering."
        )

    nodes = set(str(v) for v in final_pose["node"].astype(str).unique().tolist())
    tracks = set(
        int(v)
        for v in pd.to_numeric(final_pose["track"], errors="coerce").dropna().astype(np.int64)
    )
    required_nodes = {node for node, _track in LEGACY_POSE_NODE_TRACKS}
    missing_nodes = sorted(required_nodes - nodes)
    if missing_nodes:
        raise LegacyFramewiseSourceError(
            f"final_pose lacks required pose nodes for legacy export: {missing_nodes}"
        )
    if not {0, 1}.issubset(tracks):
        raise LegacyFramewiseSourceError(
            f"final_pose must include tracks 0 and 1; found {sorted(tracks)}"
        )

    present_pairs = {
        (str(node), int(track))
        for node, track in zip(
            final_pose["node"].astype(str).to_numpy(),
            pd.to_numeric(final_pose["track"], errors="coerce")
            .fillna(-1)
            .astype(np.int64)
            .to_numpy(),
            strict=True,
        )
    }
    missing_pairs = [
        f"{node}_{track}"
        for node, track in LEGACY_POSE_NODE_TRACKS
        if (node, track) not in present_pairs
    ]
    if missing_pairs:
        raise LegacyFramewiseSourceError(
            "final_pose lacks required node/track pairs for legacy export: "
            f"{missing_pairs}"
        )

    return {
        "n_sync_dense": len(dense_set),
        "n_pose_samples": len(sample_set),
        "dense_min": int(min(dense_set)) if dense_set else None,
        "dense_max": int(max(dense_set)) if dense_set else None,
        "sample_min": int(min(sample_set)) if sample_set else None,
        "sample_max": int(max(sample_set)) if sample_set else None,
        "nodes": sorted(nodes),
        "tracks": sorted(tracks),
        "timeline_match": True,
        "schema_ok": True,
    }
