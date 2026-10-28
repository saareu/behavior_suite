"""Joint S7/S6 intake, eligibility, and lineage validation for S8 Sprint 1.

Does **not** substitute the newest S6 run when the S7-linked S6 is unavailable.
Overall S7 eligibility does **not** automatically imply S8 readiness.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from behavioral_arbitration.contracts import (
    FPS_ATOL,
    RECOMMENDED_RESCUE_COLUMNS,
    RECOMMENDED_VETO_COLUMNS,
    REQUIRED_DYADIC_DETECTORS,
    REQUIRED_FOCAL_DETECTORS,
    REQUIRED_HARD_GAP_COLUMNS,
    TARGET_FPS,
    IntakeError,
)
from behavioral_inference.contracts import (
    COMPLETED_WITH_WARNINGS as S7_COMPLETED_WITH_WARNINGS,
)
from behavioral_inference.contracts import (
    DYADIC_WIDE_FILENAME,
    FOCAL_WIDE_FILENAME,
    PREDICTIONS_FILENAME,
    RUN_META_FILENAME,
    ProvenanceError,
    PublicationError,
)
from behavioral_inference.contracts import (
    ELIGIBLE_DOWNSTREAM as S7_ELIGIBLE_DOWNSTREAM,
)
from behavioral_inference.run_discovery import (
    BehavioralInferenceProjectSummary,
    BehavioralInferenceRunSummary,
    find_latest_eligible_s7_run,
)
from behavioral_inference.runner import (
    is_s7_run_eligible_for_downstream,
    validate_s7_run_for_downstream,
)
from combined_features.contracts import (
    DYADIC_FEATURES_FILENAME,
    CombinedFeaturesError,
)
from combined_features.runner import (
    is_s6_run_eligible_for_downstream,
    validate_s6_run_for_downstream,
)
from project.validation_memo import (
    artifact_fingerprint,
    memoized_validation,
    run_dir_fingerprint,
)


@dataclass(frozen=True, slots=True)
class JointIntakeBundle:
    """Validated S7 prediction tables + linked S6 features for arbitration."""

    session_root: Path
    s7_run_dir: Path
    s7_run_id: str
    s6_run_dir: Path
    s6_run_id: str
    exp_name: str
    fps: float
    mouse_ids: tuple[int, ...]
    n_frames: int
    dyadic_wide: pd.DataFrame
    focal_wide: pd.DataFrame
    features: pd.DataFrame
    detectors_dyadic: tuple[str, ...]
    detectors_focal: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    lineage: dict[str, Any] = field(default_factory=dict)


def select_eligible_s7_run(
    session_root: Path | str,
    *,
    s7_run_dir: Path | str | None = None,
    project: BehavioralInferenceProjectSummary | None = None,
) -> BehavioralInferenceRunSummary:
    """Select an eligible S7 run (explicit path or newest eligible)."""

    root = Path(session_root).expanduser().resolve()
    if s7_run_dir is not None:
        run_dir = Path(s7_run_dir).expanduser().resolve()
        if not is_s7_run_eligible_for_downstream(run_dir):
            raise IntakeError(
                f"S7 run is not eligible for downstream use: {run_dir}"
            )
        if project is not None:
            for run in project.runs:
                if run.run_dir.resolve() == run_dir:
                    if run.classification not in (
                        S7_ELIGIBLE_DOWNSTREAM,
                        S7_COMPLETED_WITH_WARNINGS,
                    ):
                        raise IntakeError(
                            f"S7 run classification is {run.classification!r}; "
                            "refusing for S8 intake."
                        )
                    return run
            raise IntakeError(f"S7 run directory not found under session: {run_dir}")
        # Explicit path: summarize only this run (avoid whole-project rescans).
        return _summarize_explicit_eligible_s7_run(run_dir)

    latest = find_latest_eligible_s7_run(root)
    if latest is None:
        raise IntakeError(
            f"No eligible S7 run found under session root: {root}"
        )
    return latest


def _summarize_explicit_eligible_s7_run(run_dir: Path) -> BehavioralInferenceRunSummary:
    """Build a lightweight eligible summary for one already-validated S7 path."""

    meta = _load_json(run_dir / RUN_META_FILENAME)
    status = _optional_text(meta.get("status"))
    source_s6 = (
        meta.get("source_s6") if isinstance(meta.get("source_s6"), Mapping) else {}
    )
    s6_run_id = (
        _optional_text(source_s6.get("run_id")) if isinstance(source_s6, Mapping) else None
    )
    s6_run_dir = (
        _optional_text(source_s6.get("run_dir")) if isinstance(source_s6, Mapping) else None
    )
    detectors = (
        meta.get("detectors_selected")
        if isinstance(meta.get("detectors_selected"), Mapping)
        else {}
    )
    dyadic = tuple(
        str(x)
        for x in (
            detectors.get("dyadic", []) if isinstance(detectors, Mapping) else []
        )
        if str(x).strip()
    )
    focal = tuple(
        str(x)
        for x in (
            detectors.get("focal", []) if isinstance(detectors, Mapping) else []
        )
        if str(x).strip()
    )
    predictions = run_dir / PREDICTIONS_FILENAME
    dyadic_wide = run_dir / DYADIC_WIDE_FILENAME
    focal_wide = run_dir / FOCAL_WIDE_FILENAME
    classification = (
        S7_COMPLETED_WITH_WARNINGS
        if status == "completed_with_warnings"
        else S7_ELIGIBLE_DOWNSTREAM
    )
    return BehavioralInferenceRunSummary(
        run_dir=run_dir.resolve(),
        run_id=_optional_text(meta.get("run_id")) or run_dir.name,
        status=status,
        classification=classification,
        source_s6_run_id=s6_run_id,
        source_s6_run_dir=Path(s6_run_dir).resolve() if s6_run_dir else None,
        predictions_path=predictions.resolve() if predictions.is_file() else None,
        dyadic_wide_path=dyadic_wide.resolve() if dyadic_wide.is_file() else None,
        focal_wide_path=focal_wide.resolve() if focal_wide.is_file() else None,
        detectors_dyadic=dyadic,
        detectors_focal=focal,
        created_at=_optional_text(meta.get("created_at")),
        completed_at=_optional_text(meta.get("completed_at")),
        warning_count=0,
        error_message=_optional_text(meta.get("error")),
        metadata_errors=(),
        staleness_reasons=(),
    )


def resolve_linked_s6_run(s7_run_dir: Path | str) -> dict[str, Any]:
    """Resolve the exact S6 run recorded by S7 — never substitute newest S6."""

    run_dir = Path(s7_run_dir).expanduser().resolve()
    meta_path = run_dir / RUN_META_FILENAME
    if not meta_path.is_file():
        raise IntakeError(f"S7 run_meta.json missing: {meta_path}")
    meta = _load_json(meta_path)
    source_s6 = meta.get("source_s6")
    if not isinstance(source_s6, Mapping):
        raise IntakeError("S7 run_meta.json is missing source_s6 provenance.")

    s6_run_dir_text = _optional_text(source_s6.get("run_dir"))
    s6_run_id = _optional_text(source_s6.get("run_id"))
    if s6_run_dir_text is None:
        raise IntakeError("S7 source_s6.run_dir is missing; refusing newest-S6 substitution.")
    s6_run_dir = Path(s6_run_dir_text).expanduser()
    if not s6_run_dir.is_dir():
        raise IntakeError(
            f"Linked S6 run directory is unavailable: {s6_run_dir}. "
            "S8 will not substitute the newest eligible S6 run."
        )
    if not is_s6_run_eligible_for_downstream(s6_run_dir):
        raise IntakeError(
            f"Linked S6 run is no longer eligible (stale or incomplete): {s6_run_dir}. "
            "S8 will not substitute the newest eligible S6 run."
        )
    return {
        "s6_run_dir": s6_run_dir.resolve(),
        "s6_run_id": s6_run_id or s6_run_dir.name,
        "source_s6": dict(source_s6),
        "s7_meta": meta,
    }


@dataclass(frozen=True, slots=True)
class S7S8ReadinessAssessment:
    """GUI/discovery view of whether an S7 run is S8-ready (vs merely S7-eligible)."""

    run_id: str
    run_dir: Path
    historical_status: str | None
    s7_classification: str
    s8_ready: bool
    linked_s6_run_id: str | None
    linked_s6_run_dir: Path | None
    mouse_ids: tuple[int, ...]
    n_frames: int | None
    detectors_dyadic: tuple[str, ...]
    detectors_focal: tuple[str, ...]
    missing_detectors: tuple[str, ...]
    missing_s6_columns: tuple[str, ...]
    reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    lineage_aligned: bool | None = None


def assess_s7_for_s8(
    session_root: Path | str,
    s7_run_dir: Path | str,
    *,
    require_exactly_two_mice: bool = True,
    project: BehavioralInferenceProjectSummary | None = None,
) -> S7S8ReadinessAssessment:
    """Assess one S7 run for S8 readiness without raising on ordinary failures.

    Uses the authoritative joint validator. Generally eligible S7 runs can still
    fail S8 readiness (missing detectors, linked S6, alignment, K≠2).
    """

    root = Path(session_root).expanduser().resolve()
    run_dir = Path(s7_run_dir).expanduser().resolve()
    linked_s6_fp: tuple[Any, ...] = ("no_s6",)
    try:
        meta = _load_json(run_dir / RUN_META_FILENAME)
        source_s6 = meta.get("source_s6")
        if isinstance(source_s6, Mapping):
            s6_text = _optional_text(source_s6.get("run_dir"))
            if s6_text is not None:
                linked_s6_fp = run_dir_fingerprint(
                    Path(s6_text), (DYADIC_FEATURES_FILENAME,)
                )
    except (IntakeError, OSError):
        linked_s6_fp = ("s6_unreadable",)
    fingerprint = (
        str(root),
        bool(require_exactly_two_mice),
        run_dir_fingerprint(
            run_dir,
            (PREDICTIONS_FILENAME, DYADIC_WIDE_FILENAME, FOCAL_WIDE_FILENAME),
        ),
        artifact_fingerprint(run_dir / RUN_META_FILENAME),
        linked_s6_fp,
    )
    return memoized_validation(
        kind="assess_s7_for_s8",
        fingerprint=fingerprint,
        compute=lambda: _assess_s7_for_s8_uncached(
            root,
            run_dir,
            require_exactly_two_mice=require_exactly_two_mice,
            project=project,
        ),
    )


def _assess_s7_for_s8_uncached(
    root: Path,
    run_dir: Path,
    *,
    require_exactly_two_mice: bool,
    project: BehavioralInferenceProjectSummary | None,
) -> S7S8ReadinessAssessment:
    summary: BehavioralInferenceRunSummary | None = None
    if project is not None:
        summary = next(
            (r for r in project.runs if r.run_dir.resolve() == run_dir), None
        )
    if summary is None:
        try:
            summary = _summarize_explicit_eligible_s7_run(run_dir)
        except (IntakeError, OSError, ValueError):
            summary = None

    historical_status = summary.status if summary is not None else None
    s7_classification = (
        summary.classification if summary is not None else "missing_required_artifacts"
    )
    meta_detectors_dyadic = summary.detectors_dyadic if summary is not None else ()
    meta_detectors_focal = summary.detectors_focal if summary is not None else ()
    linked_s6_id = summary.source_s6_run_id if summary is not None else None
    linked_s6_dir = summary.source_s6_run_dir if summary is not None else None

    try:
        bundle = validate_joint_s7_s6_for_s8(
            root,
            s7_run_dir=run_dir,
            require_exactly_two_mice=require_exactly_two_mice,
        )
    except IntakeError as exc:
        missing_detectors = _parse_missing_detectors(str(exc))
        missing_s6 = _parse_missing_s6_columns(str(exc))
        return S7S8ReadinessAssessment(
            run_id=summary.run_id if summary is not None else run_dir.name,
            run_dir=run_dir,
            historical_status=historical_status,
            s7_classification=s7_classification,
            s8_ready=False,
            linked_s6_run_id=linked_s6_id,
            linked_s6_run_dir=linked_s6_dir,
            mouse_ids=(),
            n_frames=None,
            detectors_dyadic=meta_detectors_dyadic,
            detectors_focal=meta_detectors_focal,
            missing_detectors=missing_detectors,
            missing_s6_columns=missing_s6,
            reasons=(str(exc),),
            warnings=(),
            lineage_aligned=False,
        )

    return S7S8ReadinessAssessment(
        run_id=bundle.s7_run_id,
        run_dir=bundle.s7_run_dir,
        historical_status=historical_status,
        s7_classification=s7_classification,
        s8_ready=True,
        linked_s6_run_id=bundle.s6_run_id,
        linked_s6_run_dir=bundle.s6_run_dir,
        mouse_ids=bundle.mouse_ids,
        n_frames=bundle.n_frames,
        detectors_dyadic=bundle.detectors_dyadic,
        detectors_focal=bundle.detectors_focal,
        missing_detectors=(),
        missing_s6_columns=(),
        reasons=(),
        warnings=bundle.warnings,
        lineage_aligned=True,
    )


def list_s7_options_for_s8(
    session_root: Path | str,
    *,
    require_exactly_two_mice: bool = True,
) -> tuple[S7S8ReadinessAssessment, ...]:
    """Assess every discovered S7 run for S8 readiness (newest first)."""

    root = Path(session_root).expanduser().resolve()
    from behavioral_inference.run_discovery import summarize_behavioral_inference_project

    project = summarize_behavioral_inference_project(root)
    return tuple(
        assess_s7_for_s8(
            root,
            run.run_dir,
            require_exactly_two_mice=require_exactly_two_mice,
            project=project,
        )
        for run in project.runs
    )


def find_latest_s8_ready_s7_run(
    session_root: Path | str,
    *,
    require_exactly_two_mice: bool = True,
) -> S7S8ReadinessAssessment | None:
    """Return the newest S8-ready S7 run (not merely the newest eligible S7)."""

    for assessment in list_s7_options_for_s8(
        session_root, require_exactly_two_mice=require_exactly_two_mice
    ):
        if assessment.s8_ready:
            return assessment
    return None


def validate_joint_s7_s6_for_s8(
    session_root: Path | str,
    *,
    s7_run_dir: Path | str | None = None,
    require_exactly_two_mice: bool = True,
    required_dyadic: Sequence[str] = REQUIRED_DYADIC_DETECTORS,
    required_focal: Sequence[str] = REQUIRED_FOCAL_DETECTORS,
) -> JointIntakeBundle:
    """Full joint intake validation for S8 readiness.

    Checks (in order):
    1. Eligible S7 selection
    2. Exact linked S6 resolution (no newest-S6 fallback)
    3. Revalidate S7 + S6 downstream eligibility
    4. S7 input lineage vs recorded S6 digests
    5. Matching 120 Hz timeline
    6. Identity / sample-key uniqueness
    7. Exactly two mice
    8. Required detector set
    9. Required S6 feature columns
    10. Reject stale / mismatched / missing / structurally invalid inputs
    """

    root = Path(session_root).expanduser().resolve()
    # Explicit paths use a single-run summary (no whole-project rescan).
    s7_summary = select_eligible_s7_run(root, s7_run_dir=s7_run_dir)
    s7_dir = s7_summary.run_dir.resolve()

    try:
        s7_info = validate_s7_run_for_downstream(s7_dir)
    except (PublicationError, ProvenanceError) as exc:
        raise IntakeError(f"S7 downstream revalidation failed: {exc}") from exc

    linked = resolve_linked_s6_run(s7_dir)
    s6_dir: Path = linked["s6_run_dir"]
    # Ensure S7 validate and linked resolve agree on the same S6 path.
    recorded = s7_info.get("source_s6_run_dir")
    if recorded is not None and Path(recorded).resolve() != s6_dir.resolve():
        raise IntakeError(
            "S7 provenance source_s6.run_dir mismatch between validators: "
            f"{recorded} vs {s6_dir}"
        )

    # validate_s7_run_for_downstream already revalidates nested S6; call again so
    # callers still get an explicit S6 info payload (memoized when unchanged).
    try:
        s6_info = validate_s6_run_for_downstream(s6_dir)
    except CombinedFeaturesError as exc:
        raise IntakeError(f"Linked S6 revalidation failed: {exc}") from exc

    warnings: list[str] = []
    dyadic_wide, focal_wide = _load_s7_wide_views(s7_dir, linked["s7_meta"])
    features = _load_s6_features(s6_dir)

    detectors_dyadic, detectors_focal = _resolve_detector_coverage(
        s7_meta=linked["s7_meta"],
        dyadic_wide=dyadic_wide,
        focal_wide=focal_wide,
        required_dyadic=required_dyadic,
        required_focal=required_focal,
    )

    exp_name = _resolve_exp_name(dyadic_wide, focal_wide, features, linked["s7_meta"])
    dyadic_wide = _ensure_exp_name(dyadic_wide, exp_name)
    focal_wide = _ensure_exp_name(focal_wide, exp_name)
    features = _ensure_exp_name(features, exp_name)

    mouse_ids = _require_two_mice(
        dyadic_wide, focal_wide, features, require=require_exactly_two_mice
    )
    n_frames = _validate_timeline_alignment(dyadic_wide, focal_wide, features)
    _validate_key_uniqueness(dyadic_wide, focal_wide, features)
    _validate_s7_s6_lineage_samples(focal_wide, features)
    warnings.extend(_check_recommended_feature_columns(features))

    missing_hard = [c for c in REQUIRED_HARD_GAP_COLUMNS if c not in features.columns]
    if missing_hard:
        raise IntakeError(
            "Linked S6 features missing required hard-gap / optical columns: "
            + ", ".join(missing_hard)
        )

    return JointIntakeBundle(
        session_root=root,
        s7_run_dir=s7_dir,
        s7_run_id=s7_summary.run_id,
        s6_run_dir=s6_dir,
        s6_run_id=str(linked["s6_run_id"]),
        exp_name=exp_name,
        fps=TARGET_FPS,
        mouse_ids=mouse_ids,
        n_frames=n_frames,
        dyadic_wide=dyadic_wide,
        focal_wide=focal_wide,
        features=features,
        detectors_dyadic=tuple(detectors_dyadic),
        detectors_focal=tuple(detectors_focal),
        warnings=tuple(warnings),
        lineage={
            "s7_validate": {
                "status": s7_info.get("status"),
                "source_s6_run_id": s7_info.get("source_s6_run_id"),
                "dyadic_features_sha256": s7_info.get("dyadic_features_sha256"),
            },
            "s6_validate": {
                "run_id": s6_info.get("run_id") if isinstance(s6_info, Mapping) else None,
            },
            "source_s6": linked["source_s6"],
        },
    )


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise IntakeError(f"Could not read JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise IntakeError(f"JSON root must be an object: {path}")
    return payload


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _load_s7_wide_views(
    s7_run_dir: Path,
    s7_meta: Mapping[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dyadic_path = s7_run_dir / DYADIC_WIDE_FILENAME
    focal_path = s7_run_dir / FOCAL_WIDE_FILENAME
    if dyadic_path.is_file() and focal_path.is_file():
        return pd.read_parquet(dyadic_path), pd.read_parquet(focal_path)

    # Dual intake: build from canonical when wide views were not emitted.
    predictions_path = s7_run_dir / PREDICTIONS_FILENAME
    if not predictions_path.is_file():
        raise IntakeError(
            f"S7 run missing both wide views and {PREDICTIONS_FILENAME}."
        )
    raise IntakeError(
        "S8 Sprint 1 requires dyadic_predictions_wide.parquet and "
        "focal_predictions_wide.parquet (emit_wide_views). Canonical-only "
        f"rebuild is deferred (U7). Missing under {s7_run_dir}."
    )


def _load_s6_features(s6_run_dir: Path) -> pd.DataFrame:
    path = s6_run_dir / DYADIC_FEATURES_FILENAME
    if not path.is_file():
        raise IntakeError(f"Linked S6 missing {DYADIC_FEATURES_FILENAME}: {path}")
    return pd.read_parquet(path)


def _resolve_detector_coverage(
    *,
    s7_meta: Mapping[str, Any],
    dyadic_wide: pd.DataFrame,
    focal_wide: pd.DataFrame,
    required_dyadic: Sequence[str],
    required_focal: Sequence[str],
) -> tuple[list[str], list[str]]:
    selected = s7_meta.get("detectors_selected")
    meta_dyadic: list[str] = []
    meta_focal: list[str] = []
    if isinstance(selected, Mapping):
        raw_d = selected.get("dyadic", [])
        raw_f = selected.get("focal", [])
        if isinstance(raw_d, Sequence):
            meta_dyadic = [str(x) for x in raw_d]
        if isinstance(raw_f, Sequence):
            meta_focal = [str(x) for x in raw_f]

    present_dyadic = [
        d for d in required_dyadic if f"prob_{d}" in dyadic_wide.columns
    ]
    present_focal = [
        d for d in required_focal if f"prob_{d}" in focal_wide.columns
    ]

    # S7 eligibility ≠ S8 readiness: require full detector set on artifacts.
    missing_d = [d for d in required_dyadic if d not in present_dyadic]
    missing_f = [d for d in required_focal if d not in present_focal]
    if missing_d or missing_f:
        raise IntakeError(
            "S7 run lacks required detector coverage for S8 full_ethogram profile: "
            f"dyadic_missing={missing_d}, focal_missing={missing_f}. "
            f"meta_selected_dyadic={meta_dyadic}, meta_selected_focal={meta_focal}."
        )

    # Also require validity columns for detectors that use routing.
    for det in required_dyadic:
        if f"input_valid_{det}" not in dyadic_wide.columns:
            raise IntakeError(
                f"Dyadic wide view missing input_valid_{det} "
                "(required to distinguish routing sentinels from valid negatives)."
            )
    for det in required_focal:
        if f"input_valid_{det}" not in focal_wide.columns:
            raise IntakeError(
                f"Focal wide view missing input_valid_{det} "
                "(required to distinguish routing sentinels from valid negatives)."
            )

    return present_dyadic, present_focal


def _resolve_exp_name(
    dyadic: pd.DataFrame,
    focal: pd.DataFrame,
    features: pd.DataFrame,
    s7_meta: Mapping[str, Any],
) -> str:
    for frame in (dyadic, focal, features):
        if "exp_name" in frame.columns and len(frame):
            values = {str(v) for v in frame["exp_name"].dropna().unique().tolist()}
            if len(values) == 1:
                return next(iter(values))
            if len(values) > 1:
                raise IntakeError(f"Multiple exp_name values in intake: {sorted(values)}")
    settings = s7_meta.get("settings")
    if isinstance(settings, Mapping) and settings.get("exp_name"):
        return str(settings["exp_name"])
    return "session"


def _ensure_exp_name(frame: pd.DataFrame, exp_name: str) -> pd.DataFrame:
    out = frame.copy()
    if "exp_name" not in out.columns:
        out["exp_name"] = exp_name
    else:
        out["exp_name"] = out["exp_name"].fillna(exp_name)
    return out


def _require_two_mice(
    dyadic: pd.DataFrame,
    focal: pd.DataFrame,
    features: pd.DataFrame,
    *,
    require: bool,
) -> tuple[int, ...]:
    ids: set[int] = set()
    for frame, cols in (
        (dyadic, ("mouse_a", "mouse_b", "focal_id", "partner_id")),
        (focal, ("focal_id", "partner_id", "mouse_a", "mouse_b")),
        (features, ("focal_id", "partner_id")),
    ):
        for col in cols:
            if col in frame.columns:
                ids.update(int(x) for x in frame[col].dropna().unique().tolist())
    mice = tuple(sorted(ids))
    if require and len(mice) != 2:
        raise IntakeError(
            f"S8 MVP requires exactly two tracked mice; found {len(mice)}: {mice}"
        )
    return mice


def _validate_timeline_alignment(
    dyadic: pd.DataFrame,
    focal: pd.DataFrame,
    features: pd.DataFrame,
) -> int:
    """Confirm matching 120 Hz dense_frame_idx grids across S7/S6."""

    def _frames(frame: pd.DataFrame, label: str) -> set[int]:
        if "dense_frame_idx" not in frame.columns:
            raise IntakeError(f"{label} missing dense_frame_idx.")
        vals = pd.to_numeric(frame["dense_frame_idx"], errors="coerce")
        if vals.isna().any():
            raise IntakeError(f"{label} has NaN dense_frame_idx.")
        return {int(x) for x in vals.unique().tolist()}

    d_frames = _frames(dyadic, "dyadic wide")
    f_frames = _frames(focal, "focal wide")
    s6_frames = _frames(features, "S6 features")

    if d_frames != f_frames:
        raise IntakeError(
            "Dyadic and focal wide views have mismatched dense_frame_idx sets."
        )
    if d_frames != s6_frames:
        raise IntakeError(
            "S7 prediction timeline does not match linked S6 dense_frame_idx set "
            f"(S7 n={len(d_frames)}, S6 n={len(s6_frames)})."
        )

    # sample_idx == dense_frame_idx (S6/S7 contract).
    for frame, label in (
        (dyadic, "dyadic"),
        (focal, "focal"),
        (features, "S6"),
    ):
        if "sample_idx" not in frame.columns:
            continue
        sample = pd.to_numeric(frame["sample_idx"], errors="coerce")
        dense = pd.to_numeric(frame["dense_frame_idx"], errors="coerce")
        if not np.allclose(
            sample.to_numpy(dtype=float),
            dense.to_numpy(dtype=float),
            equal_nan=True,
        ):
            raise IntakeError(f"{label}: sample_idx != dense_frame_idx.")

    # FPS check via time_sec spacing when enough frames exist.
    for frame, label in ((focal, "focal"), (features, "S6")):
        if "time_sec" not in frame.columns or frame["time_sec"].isna().all():
            continue
        # Per unique frame.
        timing = (
            frame.loc[:, ["dense_frame_idx", "time_sec"]]
            .drop_duplicates("dense_frame_idx")
            .sort_values("dense_frame_idx", kind="stable")
        )
        if len(timing) < 3:
            continue
        dt = np.diff(timing["time_sec"].to_numpy(dtype=float))
        dt = dt[np.isfinite(dt) & (dt > 0)]
        if len(dt) == 0:
            continue
        median_dt = float(np.median(dt))
        if median_dt <= 0:
            continue
        fps = 1.0 / median_dt
        if abs(fps - TARGET_FPS) > max(FPS_ATOL, 0.5):  # allow half-Hz lab jitter
            raise IntakeError(
                f"{label} timeline fps≈{fps:.3f} does not match required {TARGET_FPS}."
            )

    return len(d_frames)


def _validate_key_uniqueness(
    dyadic: pd.DataFrame,
    focal: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    dyadic_keys = ["dense_frame_idx", "mouse_a", "mouse_b"]
    if all(c in dyadic.columns for c in dyadic_keys) and bool(
        dyadic.duplicated(subset=dyadic_keys, keep=False).any()
    ):
        raise IntakeError(
            "Dyadic wide view has duplicate undirected keys "
            "(dense_frame_idx, mouse_a, mouse_b)."
        )
    # Also check collapse-survivor IDs.
    alt = ["dense_frame_idx", "focal_id", "partner_id"]
    if all(c in dyadic.columns for c in alt) and bool(
        dyadic.duplicated(subset=alt, keep=False).any()
    ):
        raise IntakeError(
            "Dyadic wide view has duplicate (dense_frame_idx, focal_id, partner_id)."
        )

    focal_keys = ["dense_frame_idx", "focal_id", "partner_id"]
    if bool(focal.duplicated(subset=focal_keys, keep=False).any()):
        raise IntakeError(
            "Focal wide view has duplicate directional keys "
            "(dense_frame_idx, focal_id, partner_id)."
        )
    if bool(features.duplicated(subset=focal_keys, keep=False).any()):
        raise IntakeError(
            "S6 features have duplicate directional keys "
            "(dense_frame_idx, focal_id, partner_id)."
        )


def _validate_s7_s6_lineage_samples(
    focal: pd.DataFrame,
    features: pd.DataFrame,
) -> None:
    """Directional identity keys in focal must be a subset of S6 features."""

    keys = ["dense_frame_idx", "focal_id", "partner_id"]
    focal_keys = focal.loc[:, keys].drop_duplicates()
    feat_keys = features.loc[:, keys].drop_duplicates()
    merged = focal_keys.merge(
        feat_keys.assign(_s6_present=True),
        on=keys,
        how="left",
    )
    missing = merged["_s6_present"].isna().sum()
    if int(missing) > 0:
        raise IntakeError(
            f"{int(missing)} focal directional samples have no matching S6 feature row."
        )


def _check_recommended_feature_columns(features: pd.DataFrame) -> list[str]:
    warnings: list[str] = []
    missing_veto = [c for c in RECOMMENDED_VETO_COLUMNS if c not in features.columns]
    if missing_veto:
        warnings.append(
            "S6 missing recommended postural-veto columns "
            f"(dependent vetoes degrade safely): {missing_veto}"
        )
    missing_rescue = [c for c in RECOMMENDED_RESCUE_COLUMNS if c not in features.columns]
    if missing_rescue:
        warnings.append(
            "S6 missing recommended rescue/stitch columns "
            f"(unused until Sprint 2+): {missing_rescue}"
        )
    return warnings


def _parse_missing_detectors(message: str) -> tuple[str, ...]:
    """Best-effort parse of missing detector names from an IntakeError message."""

    if "detector" not in message.lower():
        return ()
    coverage_hint = (
        "missing" in message.lower() or "lacks required detector" in message.lower()
    )
    if not coverage_hint:
        return ()
    found = [
        name
        for name in (*REQUIRED_DYADIC_DETECTORS, *REQUIRED_FOCAL_DETECTORS)
        if name in message
    ]
    return tuple(dict.fromkeys(found))


def _parse_missing_s6_columns(message: str) -> tuple[str, ...]:
    if "hard-gap" not in message and "optical" not in message.lower():
        return ()
    found = [c for c in REQUIRED_HARD_GAP_COLUMNS if c in message]
    return tuple(dict.fromkeys(found))
