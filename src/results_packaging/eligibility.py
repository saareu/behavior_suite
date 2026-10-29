"""S8 source selection and structural eligibility for Subsystem 09 Sprint 1.

Reuses S8 discovery / ``validate_s8_run_for_downstream`` rather than
re-implementing scientific arbitration checks. Distinguishes historical
completion status from current downstream eligibility. Does not invent
scientific QC thresholds.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_arbitration.contracts import (
    BEHAVIOR_BOUTS_FILENAME,
    BEHAVIOR_STATES_FILENAME,
    BEHAVIORAL_ARBITRATION_DIRNAME,
    COMPLETED_WITH_WARNINGS,
    DIRECTIONAL_IDENTITY_KEYS,
    ELIGIBLE_DOWNSTREAM,
    QC_FILENAME,
    RUN_META_FILENAME,
    SETTINGS_USED_FILENAME,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
)
from behavioral_arbitration.run_discovery import (
    BehavioralArbitrationRunSummary,
    find_latest_eligible_s8_run,
    summarize_behavioral_arbitration_project,
)
from behavioral_arbitration.runner import (
    is_s8_run_eligible_for_downstream,
    validate_s8_run_for_downstream,
)
from behavioral_inference.contracts import ProvenanceError
from project.file_digest import sha256_file
from project.validation_memo import (
    artifact_fingerprint,
    memoized_validation,
    run_dir_fingerprint,
)
from results_packaging.contracts import (
    EXPECTED_S8_BOUTS_SCHEMA_VERSION,
    EXPECTED_S8_STATES_SCHEMA_VERSION,
    FPS_ATOL,
    REQUIRED_BOUT_COLUMNS,
    REQUIRED_STATE_COLUMNS,
    TARGET_FPS,
    TIME_ATOL_SEC,
    IntakeError,
)

_PREPROCESS_DIRNAME = "preprocess"
_PREPARED_SYNC_NAME = "prepared_sync.npz"
_PREPARED_VIDEO_NAME = "prepared_video.mp4"


@dataclass(frozen=True, slots=True)
class ResolvedSyncVideoSources:
    """Resolved prepared_sync / prepared_video identities for packaging."""

    prepared_sync_path: Path
    prepared_sync_sha256: str
    prepared_sync_sha256_recorded: str | None
    prepared_video_path: Path | None
    prepared_video_sha256_recorded: str | None
    source_s4_run_id: str | None
    source_s5_run_id: str | None
    lineage_notes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class S9SourceValidationResult:
    """Validated S8 packaging source plus resolved sync/video lineage."""

    session_root: Path
    s8_run_dir: Path
    s8_run_id: str
    s8_status: str
    s8_classification: str
    s8_eligible: bool
    source_s7_run_id: str
    source_s7_run_dir: Path
    source_s6_run_id: str
    source_s6_run_dir: Path
    states_path: Path
    bouts_path: Path
    states_schema_version: str
    bouts_schema_version: str
    states_sha256: str
    bouts_sha256: str
    n_directional_rows: int
    n_dense_samples: int
    n_bouts: int
    dense_frame_idx_min: int
    dense_frame_idx_max: int
    time_sec_min: float
    time_sec_max: float
    fps: float
    sync_video: ResolvedSyncVideoSources
    unique_dense_times: pd.DataFrame
    warnings: tuple[str, ...] = ()
    downstream_validation: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_root": str(self.session_root),
            "s8_run_dir": str(self.s8_run_dir),
            "s8_run_id": self.s8_run_id,
            "s8_status": self.s8_status,
            "s8_classification": self.s8_classification,
            "s8_eligible": self.s8_eligible,
            "source_s7_run_id": self.source_s7_run_id,
            "source_s7_run_dir": str(self.source_s7_run_dir),
            "source_s6_run_id": self.source_s6_run_id,
            "source_s6_run_dir": str(self.source_s6_run_dir),
            "states_path": str(self.states_path),
            "bouts_path": str(self.bouts_path),
            "states_schema_version": self.states_schema_version,
            "bouts_schema_version": self.bouts_schema_version,
            "states_sha256": self.states_sha256,
            "bouts_sha256": self.bouts_sha256,
            "n_directional_rows": self.n_directional_rows,
            "n_dense_samples": self.n_dense_samples,
            "n_bouts": self.n_bouts,
            "dense_frame_idx_min": self.dense_frame_idx_min,
            "dense_frame_idx_max": self.dense_frame_idx_max,
            "time_sec_min": self.time_sec_min,
            "time_sec_max": self.time_sec_max,
            "fps": self.fps,
            "prepared_sync_path": str(self.sync_video.prepared_sync_path),
            "prepared_sync_sha256": self.sync_video.prepared_sync_sha256,
            "prepared_video_path": (
                str(self.sync_video.prepared_video_path)
                if self.sync_video.prepared_video_path is not None
                else None
            ),
            "warnings": list(self.warnings),
        }


def select_s8_run_for_packaging(
    session_root: Path | str,
    *,
    s8_run_dir: Path | str | None = None,
) -> BehavioralArbitrationRunSummary:
    """Select an explicit S8 run or the latest currently eligible completed run."""

    root = Path(session_root).expanduser().resolve()
    if not root.is_dir():
        raise IntakeError(f"Session root does not exist: {root}")

    if s8_run_dir is not None:
        run_dir = Path(s8_run_dir).expanduser().resolve()
        if not run_dir.is_dir():
            raise IntakeError(f"Explicit S8 run directory does not exist: {run_dir}")
        summary = summarize_behavioral_arbitration_project(root)
        for run in summary.runs:
            if run.run_dir.resolve() == run_dir:
                return run
        # Explicit path outside discovery listing (still usable if valid).
        return BehavioralArbitrationRunSummary(
            run_dir=run_dir,
            run_id=run_dir.name,
            status=_load_status(run_dir),
            classification="explicit",
            source_s7_run_id=None,
            source_s7_run_dir=None,
            source_s6_run_id=None,
            source_s6_run_dir=None,
            behavior_states_path=(
                (run_dir / BEHAVIOR_STATES_FILENAME).resolve()
                if (run_dir / BEHAVIOR_STATES_FILENAME).is_file()
                else None
            ),
            behavior_bouts_path=(
                (run_dir / BEHAVIOR_BOUTS_FILENAME).resolve()
                if (run_dir / BEHAVIOR_BOUTS_FILENAME).is_file()
                else None
            ),
            created_at=None,
            completed_at=None,
            warning_count=0,
            error_message=None,
        )

    latest = find_latest_eligible_s8_run(root)
    if latest is None:
        raise IntakeError(
            "No currently eligible completed S8 run found for packaging. "
            "Select an explicit eligible S8 run or restore upstream digests."
        )
    return latest


def validate_s8_source_for_packaging(
    session_root: Path | str,
    *,
    s8_run_dir: Path | str | None = None,
    require_prepared_video: bool = True,
    states: pd.DataFrame | None = None,
    bouts: pd.DataFrame | None = None,
) -> S9SourceValidationResult:
    """Validate a proposed S9 source run (memoized structural/provenance checks)."""

    root = Path(session_root).expanduser().resolve()
    selected = select_s8_run_for_packaging(root, s8_run_dir=s8_run_dir)
    run_dir = selected.run_dir.resolve()

    fingerprint: tuple[Any, ...] = (
        str(root),
        bool(require_prepared_video),
        run_dir_fingerprint(
            run_dir,
            (BEHAVIOR_STATES_FILENAME, BEHAVIOR_BOUTS_FILENAME, QC_FILENAME),
        ),
    )
    # Invalidate when linked S6 sync / video provenance changes.
    try:
        meta = _load_json_object(run_dir / RUN_META_FILENAME)
        source_s6 = meta.get("source_s6")
        if isinstance(source_s6, Mapping):
            s6_text = _optional_text(source_s6.get("run_dir"))
            if s6_text is not None:
                s6_dir = Path(s6_text)
                fingerprint = (
                    *fingerprint,
                    run_dir_fingerprint(s6_dir, ("run_meta.json",)),
                )
                s6_meta = _load_json_object(s6_dir / RUN_META_FILENAME)
                for key in ("source_s4", "source_s5"):
                    block = s6_meta.get(key)
                    if isinstance(block, Mapping):
                        fingerprint = (
                            *fingerprint,
                            artifact_fingerprint(block.get("prepared_sync")),
                        )
                source_s5 = s6_meta.get("source_s5")
                if isinstance(source_s5, Mapping):
                    source_s1 = source_s5.get("source_s1")
                    if isinstance(source_s1, Mapping):
                        fingerprint = (
                            *fingerprint,
                            artifact_fingerprint(source_s1.get("prepared_video")),
                            artifact_fingerprint(source_s1.get("prepared_sync")),
                        )
    except (IntakeError, OSError, TypeError, ValueError):
        fingerprint = (*fingerprint, "lineage_unreadable")

    # DataFrames are not memo-safe; bypass cache when caller supplies tables.
    if states is not None or bouts is not None:
        return _validate_s8_source_uncached(
            session_root=root,
            selected=selected,
            require_prepared_video=require_prepared_video,
            states=states,
            bouts=bouts,
        )

    return memoized_validation(
        kind="validate_s8_source_for_packaging",
        fingerprint=fingerprint,
        compute=lambda: _validate_s8_source_uncached(
            session_root=root,
            selected=selected,
            require_prepared_video=require_prepared_video,
            states=None,
            bouts=None,
        ),
    )


def _validate_s8_source_uncached(
    *,
    session_root: Path,
    selected: BehavioralArbitrationRunSummary,
    require_prepared_video: bool,
    states: pd.DataFrame | None,
    bouts: pd.DataFrame | None,
) -> S9SourceValidationResult:
    run_dir = selected.run_dir.resolve()
    warnings: list[str] = []

    # 1–2. Explicit/auto selection + eligible completed S8.
    try:
        downstream = validate_s8_run_for_downstream(run_dir)
    except ProvenanceError as exc:
        raise IntakeError(
            f"S8 run is not currently eligible for packaging (stale/provenance): {exc}"
        ) from exc
    except Exception as exc:  # noqa: BLE001 — surface as intake failure
        raise IntakeError(f"S8 downstream validation failed: {exc}") from exc

    if not is_s8_run_eligible_for_downstream(run_dir):
        raise IntakeError(
            f"S8 run is not currently eligible for downstream packaging: {run_dir}"
        )

    meta = _load_json_object(run_dir / RUN_META_FILENAME)
    status = _optional_text(meta.get("status"))
    if status not in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS):
        raise IntakeError(
            f"S8 run is not completed (historical status={status!r}): {run_dir}"
        )
    if status == STATUS_COMPLETED_WITH_WARNINGS:
        warnings.append("Source S8 run historical status is completed_with_warnings.")

    # Required S8 artifact files (states/bouts/qc/meta/settings).
    states_path = run_dir / BEHAVIOR_STATES_FILENAME
    bouts_path = run_dir / BEHAVIOR_BOUTS_FILENAME
    for path, label in (
        (states_path, BEHAVIOR_STATES_FILENAME),
        (bouts_path, BEHAVIOR_BOUTS_FILENAME),
        (run_dir / QC_FILENAME, QC_FILENAME),
        (run_dir / RUN_META_FILENAME, RUN_META_FILENAME),
        (run_dir / SETTINGS_USED_FILENAME, SETTINGS_USED_FILENAME),
    ):
        if not path.is_file():
            raise IntakeError(f"Required S8 artifact missing ({label}): {path}")

    # Exact linked S7/S6 lineage from validated downstream payload + meta.
    source_s7 = meta.get("source_s7")
    source_s6 = meta.get("source_s6")
    if not isinstance(source_s7, Mapping) or not isinstance(source_s6, Mapping):
        raise IntakeError("S8 run_meta.json missing source_s7 / source_s6 blocks.")

    s7_run_id = _require_text(source_s7.get("run_id"), "source_s7.run_id")
    s6_run_id = _require_text(source_s6.get("run_id"), "source_s6.run_id")
    s7_run_dir = Path(
        _require_text(source_s7.get("run_dir"), "source_s7.run_dir")
    ).resolve()
    s6_run_dir = Path(
        _require_text(source_s6.get("run_dir"), "source_s6.run_dir")
    ).resolve()

    # Guard against silent substitution of a different linked run.
    ds_s7 = downstream.get("source_s7_run_id")
    ds_s6 = downstream.get("source_s6_run_id")
    if ds_s7 is not None and str(ds_s7) != s7_run_id:
        raise IntakeError(
            f"Linked S7 identity mismatch: meta={s7_run_id!r} downstream={ds_s7!r}."
        )
    if ds_s6 is not None and str(ds_s6) != s6_run_id:
        raise IntakeError(
            f"Linked S6 identity mismatch: meta={s6_run_id!r} downstream={ds_s6!r}."
        )

    states_df = states if states is not None else pd.read_parquet(states_path)
    bouts_df = bouts if bouts is not None else pd.read_parquet(bouts_path)
    _validate_states_table(states_df)
    _validate_bouts_table(bouts_df, n_dense=_unique_dense_count(states_df))

    states_schema = _unique_schema_version(states_df, EXPECTED_S8_STATES_SCHEMA_VERSION, "states")
    bouts_schema = _unique_schema_version(bouts_df, EXPECTED_S8_BOUTS_SCHEMA_VERSION, "bouts")

    unique_times = _unique_dense_timeline(states_df)
    n_dense = int(len(unique_times))
    fps = float(meta.get("fps", TARGET_FPS))
    if abs(fps - TARGET_FPS) > FPS_ATOL:
        raise IntakeError(f"S8 fps must be {TARGET_FPS}; got {fps!r}.")

    sync_video = resolve_prepared_sync_and_video(
        session_root=session_root,
        s6_run_dir=s6_run_dir,
        require_prepared_video=require_prepared_video,
    )

    classification = (
        COMPLETED_WITH_WARNINGS
        if status == STATUS_COMPLETED_WITH_WARNINGS
        or selected.classification == COMPLETED_WITH_WARNINGS
        else ELIGIBLE_DOWNSTREAM
    )

    return S9SourceValidationResult(
        session_root=session_root,
        s8_run_dir=run_dir,
        s8_run_id=_optional_text(meta.get("run_id")) or run_dir.name,
        s8_status=status,
        s8_classification=classification,
        s8_eligible=True,
        source_s7_run_id=s7_run_id,
        source_s7_run_dir=s7_run_dir,
        source_s6_run_id=s6_run_id,
        source_s6_run_dir=s6_run_dir,
        states_path=states_path.resolve(),
        bouts_path=bouts_path.resolve(),
        states_schema_version=states_schema,
        bouts_schema_version=bouts_schema,
        states_sha256=sha256_file(states_path),
        bouts_sha256=sha256_file(bouts_path),
        n_directional_rows=int(len(states_df)),
        n_dense_samples=n_dense,
        n_bouts=int(len(bouts_df)),
        dense_frame_idx_min=int(unique_times["dense_frame_idx"].iloc[0]),
        dense_frame_idx_max=int(unique_times["dense_frame_idx"].iloc[-1]),
        time_sec_min=float(unique_times["time_sec"].iloc[0]),
        time_sec_max=float(unique_times["time_sec"].iloc[-1]),
        fps=fps,
        sync_video=sync_video,
        unique_dense_times=unique_times,
        warnings=tuple(warnings),
        downstream_validation=dict(downstream),
    )


def resolve_prepared_sync_and_video(
    *,
    session_root: Path,
    s6_run_dir: Path,
    require_prepared_video: bool = True,
) -> ResolvedSyncVideoSources:
    """Resolve prepared_sync / prepared_video from S6→S4/S5/S1 lineage."""

    notes: list[str] = []
    s6_meta = _load_json_object(s6_run_dir / RUN_META_FILENAME)
    source_s4 = s6_meta.get("source_s4")
    source_s5 = s6_meta.get("source_s5")
    if not isinstance(source_s4, Mapping) or not isinstance(source_s5, Mapping):
        raise IntakeError(
            f"S6 run_meta.json missing source_s4 / source_s5: {s6_run_dir}"
        )

    sync_path_text = _optional_text(source_s4.get("prepared_sync")) or _optional_text(
        source_s5.get("prepared_sync")
    )
    if sync_path_text is None:
        raise IntakeError("S6 lineage does not record prepared_sync path.")
    sync_path = Path(sync_path_text).expanduser().resolve()
    if not sync_path.is_file():
        raise IntakeError(f"Prepared sync missing (no silent substitute): {sync_path}")

    recorded_sync_sha = _optional_text(
        source_s4.get("prepared_sync_sha256")
    ) or _optional_text(source_s5.get("prepared_sync_sha256"))
    live_sync_sha = sha256_file(sync_path)
    if recorded_sync_sha is not None and live_sync_sha.lower() != recorded_sync_sha.lower():
        raise IntakeError(
            "Prepared sync SHA256 no longer matches recorded S6 lineage digest "
            f"(recorded={recorded_sync_sha}, live={live_sync_sha})."
        )

    # Cross-check S4 vs S5 recorded paths when both present.
    s4_sync = _optional_text(source_s4.get("prepared_sync"))
    s5_sync = _optional_text(source_s5.get("prepared_sync"))
    if (
        s4_sync
        and s5_sync
        and Path(s4_sync).resolve() != Path(s5_sync).resolve()
    ):
        raise IntakeError(
            "S4 and S5 prepared_sync paths disagree; refusing silent choice."
        )

    source_s1 = source_s5.get("source_s1")
    video_path: Path | None = None
    video_sha_recorded: str | None = None
    if isinstance(source_s1, Mapping):
        video_text = _optional_text(source_s1.get("prepared_video"))
        if video_text is not None:
            video_path = Path(video_text).expanduser().resolve()
        video_sha_recorded = _optional_text(source_s1.get("prepared_video_sha256"))
        s1_sync = _optional_text(source_s1.get("prepared_sync"))
        if s1_sync is not None and Path(s1_sync).resolve() != sync_path:
            raise IntakeError(
                "S1 prepared_sync path disagrees with S4/S5 lineage path."
            )
    else:
        notes.append("S5 source_s1 block absent; falling back to session preprocess paths.")

    if video_path is None:
        fallback = (session_root / _PREPROCESS_DIRNAME / _PREPARED_VIDEO_NAME).resolve()
        if fallback.is_file():
            video_path = fallback
            notes.append(
                f"Prepared video resolved via session preprocess fallback: {fallback}"
            )

    if require_prepared_video:
        if video_path is None or not video_path.is_file():
            raise IntakeError(
                "Prepared video missing (required for S9 packaging; no silent substitute)."
            )
    elif video_path is not None and not video_path.is_file():
        raise IntakeError(
            f"Recorded prepared video path is not available: {video_path}"
        )

    if video_sha_recorded is None and video_path is not None:
        notes.append(
            "Prepared video digest not recorded in lineage; path presence validated only."
        )

    return ResolvedSyncVideoSources(
        prepared_sync_path=sync_path,
        prepared_sync_sha256=live_sync_sha,
        prepared_sync_sha256_recorded=recorded_sync_sha,
        prepared_video_path=video_path,
        prepared_video_sha256_recorded=video_sha_recorded,
        source_s4_run_id=_optional_text(source_s4.get("run_id")),
        source_s5_run_id=_optional_text(source_s5.get("run_id")),
        lineage_notes=tuple(notes),
    )


def _validate_states_table(states: pd.DataFrame) -> None:
    if states is None or states.empty:
        raise IntakeError("S8 behavior_states table is empty.")
    missing = [c for c in REQUIRED_STATE_COLUMNS if c not in states.columns]
    if missing:
        raise IntakeError(f"S8 states missing required columns: {missing}")

    # Directional key uniqueness.
    key_cols = list(DIRECTIONAL_IDENTITY_KEYS)
    if states.duplicated(key_cols).any():
        raise IntakeError(
            f"S8 states have duplicate directional keys {key_cols}."
        )

    sample = pd.to_numeric(states["sample_idx"], errors="coerce")
    dense = pd.to_numeric(states["dense_frame_idx"], errors="coerce")
    if sample.isna().any() or dense.isna().any():
        raise IntakeError("S8 states have non-numeric sample_idx / dense_frame_idx.")
    if not np.array_equal(sample.to_numpy(dtype=np.int64), dense.to_numpy(dtype=np.int64)):
        raise IntakeError("S8 states violate sample_idx == dense_frame_idx.")

    time_sec = pd.to_numeric(states["time_sec"], errors="coerce")
    if time_sec.isna().any():
        raise IntakeError("S8 states have missing time_sec values.")
    if not np.isfinite(time_sec.to_numpy(dtype=np.float64)).all():
        raise IntakeError("S8 states have non-finite time_sec values.")

    # Unique dense timeline must be contiguous from 0.
    unique = (
        states[["dense_frame_idx", "time_sec"]]
        .drop_duplicates(subset=["dense_frame_idx"])
        .sort_values("dense_frame_idx")
        .reset_index(drop=True)
    )
    idxs = unique["dense_frame_idx"].to_numpy(dtype=np.int64)
    if idxs.size == 0:
        raise IntakeError("S8 states have no dense samples.")
    if int(idxs[0]) != 0 or not np.array_equal(idxs, np.arange(idxs.shape[0], dtype=np.int64)):
        raise IntakeError(
            "S8 dense_frame_idx must be contiguous 0 … N-1 unique samples."
        )
    times = unique["time_sec"].to_numpy(dtype=np.float64)
    if times.size > 1 and not np.all(np.diff(times) > 0.0):
        raise IntakeError("S8 unique dense time_sec must be strictly increasing.")

    # Cross-check uniqueness of times per dense index (no conflicting times).
    grouped = states.groupby("dense_frame_idx", sort=False)["time_sec"]
    spreads = grouped.agg(lambda s: float(np.nanmax(s) - np.nanmin(s)))
    if (spreads > TIME_ATOL_SEC).any():
        raise IntakeError(
            "S8 states have conflicting time_sec values for the same dense_frame_idx."
        )


def _validate_bouts_table(bouts: pd.DataFrame, *, n_dense: int) -> None:
    if bouts is None:
        raise IntakeError("S8 behavior_bouts table is missing.")
    if bouts.empty:
        # Empty bout table is structurally allowed but unusual; treat as valid
        # with no further checks.
        return
    missing = [c for c in REQUIRED_BOUT_COLUMNS if c not in bouts.columns]
    if missing:
        raise IntakeError(f"S8 bouts missing required columns: {missing}")

    if bouts["bout_id"].isna().any():
        raise IntakeError("S8 bouts contain null bout_id values.")
    if bouts["bout_id"].duplicated().any():
        raise IntakeError("S8 bouts contain duplicate bout_id values.")

    start = pd.to_numeric(bouts["start_sample_idx"], errors="coerce")
    end = pd.to_numeric(bouts["end_sample_idx"], errors="coerce")
    dur = pd.to_numeric(bouts["bout_duration_frames"], errors="coerce")
    if start.isna().any() or end.isna().any() or dur.isna().any():
        raise IntakeError("S8 bouts have missing start/end/duration index fields.")
    if (start > end).any():
        raise IntakeError("S8 bouts violate start_sample_idx <= end_sample_idx.")
    expected_dur = (end - start + 1).astype(np.int64)
    if not np.array_equal(dur.to_numpy(dtype=np.int64), expected_dur.to_numpy()):
        raise IntakeError(
            "S8 bout_duration_frames must equal inclusive end-start+1 sample count."
        )
    if int(start.min()) < 0 or int(end.max()) >= n_dense:
        raise IntakeError(
            "S8 bout sample indices are outside the dense sample domain "
            f"[0, {n_dense - 1}]."
        )

    # Source identity consistency when present.
    for col in ("source_s7_run_id", "source_s6_run_id", "run_id"):
        if col not in bouts.columns:
            continue
        values = bouts[col].dropna().astype(str).unique()
        if len(values) > 1:
            raise IntakeError(f"S8 bouts have inconsistent {col} values: {list(values)}")


def _unique_dense_timeline(states: pd.DataFrame) -> pd.DataFrame:
    unique = (
        states[["dense_frame_idx", "time_sec"]]
        .drop_duplicates(subset=["dense_frame_idx"])
        .sort_values("dense_frame_idx")
        .reset_index(drop=True)
    )
    unique["dense_frame_idx"] = unique["dense_frame_idx"].astype(np.int64)
    unique["sample_idx"] = unique["dense_frame_idx"]
    unique["time_sec"] = unique["time_sec"].astype(np.float64)
    return unique


def _unique_dense_count(states: pd.DataFrame) -> int:
    return int(states["dense_frame_idx"].nunique())


def _unique_schema_version(
    frame: pd.DataFrame, expected: str, label: str
) -> str:
    if "schema_version" not in frame.columns:
        raise IntakeError(f"S8 {label} missing schema_version column.")
    values = frame["schema_version"].dropna().astype(str).unique()
    if len(values) != 1:
        raise IntakeError(f"S8 {label} schema_version is not unique: {list(values)}")
    version = str(values[0])
    if version != expected:
        raise IntakeError(
            f"S8 {label} schema_version {version!r} != expected {expected!r}."
        )
    return version


def _load_status(run_dir: Path) -> str | None:
    meta_path = run_dir / RUN_META_FILENAME
    if not meta_path.is_file():
        return None
    try:
        meta = _load_json_object(meta_path)
    except IntakeError:
        return None
    return _optional_text(meta.get("status"))


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise IntakeError(f"Required JSON missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntakeError(f"Could not read JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise IntakeError(f"JSON root must be an object: {path}")
    return payload


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_text(value: Any, label: str) -> str:
    text = _optional_text(value)
    if text is None:
        raise IntakeError(f"Missing required field {label}.")
    return text


def s8_package_dirname() -> str:
    """Expose S8 dirname for tests / docs without importing contracts elsewhere."""

    return BEHAVIORAL_ARBITRATION_DIRNAME
