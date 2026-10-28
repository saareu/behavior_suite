"""Headless Sprint 1–3 orchestration + immutable S8 publication.

Sprint 1: intake → expand → validity → merge → hard-distance (pass 1) → vetoes.
Sprint 2: hierarchy → adaptive → contact clean → hard-distance (pass 2).
Sprint 3: temporal stages + finalize → bouts → QC → atomic publish.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from behavioral_arbitration.adaptive import (
    apply_adaptive_approach_thresholding,
    apply_adaptive_retract_thresholding,
)
from behavioral_arbitration.bouts import extract_behavior_bouts
from behavioral_arbitration.config import (
    BehavioralArbitrationConfig,
    load_behavioral_arbitration_config,
)
from behavioral_arbitration.contact_clean import get_cleaned_contact_bouts
from behavioral_arbitration.contracts import (
    BEHAVIOR_BOUTS_FILENAME,
    BEHAVIOR_STATES_FILENAME,
    BEHAVIORAL_ARBITRATION_DIRNAME,
    DIAGNOSTICS_FILENAME,
    PROCESSING_LOG_FILENAME,
    QC_FILENAME,
    RECOMMENDED_RESCUE_COLUMNS,
    RECOMMENDED_VETO_COLUMNS,
    REQUIRED_HARD_GAP_COLUMNS,
    RULE_GAP_FILL,
    RULE_HYSTERESIS_ENGAGED,
    RULE_LABEL_FINALIZE,
    RULE_LABEL_SYNC,
    RULE_LABEL_UNDEFINED,
    RULE_MIN_DURATION,
    RULE_RECIPROCITY_MUTUALLY_DISENGAGED,
    RULE_RECIPROCITY_NOSE_TO_NOSE,
    RUN_META_FILENAME,
    RUN_META_SCHEMA_VERSION,
    SETTINGS_USED_FILENAME,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    STATUS_RUNNING,
    PublicationError,
)
from behavioral_arbitration.expand import expand_dyadic_to_directional
from behavioral_arbitration.export import (
    build_arbitration_diagnostics_table,
    build_behavior_states_table,
    write_behavior_bouts_parquet,
    write_behavior_states_parquet,
    write_optional_diagnostics_parquet,
)
from behavioral_arbitration.gates import (
    apply_hard_distance_probability_gates,
    apply_postural_approach_veto,
    apply_postural_retract_veto,
    assert_parallel_channels_preserved,
    merge_s6_features,
)
from behavioral_arbitration.hierarchy import resolve_hierarchical_conflicts
from behavioral_arbitration.intake import validate_joint_s7_s6_for_s8
from behavioral_arbitration.qc import (
    QcSummary,
    assert_structural_qc,
    evaluate_behavioral_arbitration_qc,
)
from behavioral_arbitration.results import (
    BehavioralArbitrationRunResult,
    Sprint1ArbitrationResult,
    Sprint2ArbitrationResult,
    Sprint3ArbitrationResult,
    StageDiagnostic,
)
from behavioral_arbitration.temporal import apply_sprint3_temporal_stages
from behavioral_arbitration.validity import normalize_arbitration_evidence
from behavioral_inference.contracts import (
    BEHAVIORAL_INFERENCE_DIRNAME,
    DYADIC_WIDE_FILENAME,
    FOCAL_WIDE_FILENAME,
    PREDICTIONS_FILENAME,
    ProvenanceError,
)
from behavioral_inference.runner import (
    is_s7_run_eligible_for_downstream,
    validate_s7_run_for_downstream,
)
from combined_features.contracts import (
    COMBINED_FEATURES_DIRNAME,
    DYADIC_FEATURES_FILENAME,
)
from combined_features.runner import (
    is_s6_run_eligible_for_downstream,
    validate_s6_run_for_downstream,
)
from preprocess.models import OperationProgress
from project.file_digest import sha256_file
from project.validation_memo import (
    artifact_fingerprint,
    memoized_validation,
    run_dir_fingerprint,
)

ProgressCallback = Callable[[OperationProgress], None]

_SPRINT3_EXPECTED_RULE_CODES: tuple[str, ...] = (
    RULE_GAP_FILL,
    RULE_MIN_DURATION,
    RULE_LABEL_SYNC,
    RULE_RECIPROCITY_NOSE_TO_NOSE,
    RULE_RECIPROCITY_MUTUALLY_DISENGAGED,
    RULE_HYSTERESIS_ENGAGED,
)


@dataclass(frozen=True, slots=True)
class BehavioralArbitrationRequest:
    """Immutable request for headless arbitration / publication."""

    session_root: Path
    s7_run_dir: Path | None = None
    output_root: Path | None = None
    config: BehavioralArbitrationConfig | None = None
    config_path: Path | None = None
    run_purpose: str = "development"
    timestamp: str | None = None


@dataclass(frozen=True, slots=True)
class _RunPaths:
    run_dir: Path
    behavior_states: Path
    behavior_bouts: Path
    diagnostics: Path
    run_meta: Path
    settings_used: Path
    processing_log: Path
    qc: Path


def run_behavioral_arbitration_sprint1(
    request: BehavioralArbitrationRequest,
) -> Sprint1ArbitrationResult:
    """Execute Sprint 1 pipeline stages 1–4 (no hierarchy / publish)."""

    session_root = Path(request.session_root).expanduser().resolve()
    config = _resolve_config(request)
    intake = validate_joint_s7_s6_for_s8(
        session_root,
        s7_run_dir=request.s7_run_dir,
        require_exactly_two_mice=config.require_exactly_two_mice,
    )

    diagnostics: list[StageDiagnostic] = []
    warnings = list(intake.warnings)

    frames, diag_expand = expand_dyadic_to_directional(
        intake.dyadic_wide,
        intake.focal_wide,
        required_dyadic=intake.detectors_dyadic,
    )
    diagnostics.append(diag_expand)

    frames, diag_validity = normalize_arbitration_evidence(frames)
    diagnostics.append(diag_validity)

    feature_cols = list(REQUIRED_HARD_GAP_COLUMNS)
    feature_cols.extend(
        c for c in RECOMMENDED_VETO_COLUMNS if c in intake.features.columns
    )
    feature_cols.extend(
        c for c in RECOMMENDED_RESCUE_COLUMNS if c in intake.features.columns
    )
    seen: set[str] = set()
    ordered_cols: list[str] = []
    for col in feature_cols:
        if col not in seen:
            seen.add(col)
            ordered_cols.append(col)

    frames, diag_merge = merge_s6_features(
        frames, intake.features, feature_columns=ordered_cols
    )
    diagnostics.append(diag_merge)

    frames, gate_diags = apply_hard_distance_probability_gates(
        frames, config, pass_id=1
    )
    diagnostics.extend(gate_diags)

    frames, diag_ap = apply_postural_approach_veto(
        frames, config.scientific.postural.approach
    )
    diagnostics.append(diag_ap)
    frames, diag_rt = apply_postural_retract_veto(
        frames, config.scientific.postural.retract
    )
    diagnostics.append(diag_rt)

    assert_parallel_channels_preserved(frames)

    return Sprint1ArbitrationResult(
        frames=frames,
        diagnostics=tuple(diagnostics),
        source_s7_run_id=intake.s7_run_id,
        source_s7_run_dir=intake.s7_run_dir,
        source_s6_run_id=intake.s6_run_id,
        source_s6_run_dir=intake.s6_run_dir,
        session_root=intake.session_root,
        exp_name=intake.exp_name,
        fps=config.fps,
        n_frames=intake.n_frames,
        n_directional_rows=int(len(frames)),
        mouse_ids=intake.mouse_ids,
        detectors_dyadic=intake.detectors_dyadic,
        detectors_focal=intake.detectors_focal,
        warnings=tuple(warnings),
        settings_used=config.to_settings_dict(),
    )


def apply_sprint2_stages(
    frames: pd.DataFrame,
    config: BehavioralArbitrationConfig,
    *,
    prior_diagnostics: tuple[StageDiagnostic, ...] = (),
    prior_warnings: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, list[StageDiagnostic], list[str]]:
    """Apply Stages 5–9 to a Sprint 1 directional frame table."""

    diagnostics: list[StageDiagnostic] = list(prior_diagnostics)
    warnings = list(prior_warnings)

    frames, hier_diags = resolve_hierarchical_conflicts(frames, config)
    diagnostics.extend(hier_diags)

    frames, diag_ap = apply_adaptive_approach_thresholding(
        frames, config.scientific.adaptive
    )
    diagnostics.append(diag_ap)
    frames, diag_rt = apply_adaptive_retract_thresholding(
        frames, config.scientific.adaptive
    )
    diagnostics.append(diag_rt)

    frames, contact_diags = get_cleaned_contact_bouts(
        frames, config.scientific.contact_post_processing
    )
    diagnostics.extend(contact_diags)

    if config.scientific.contact_post_processing.max_gap_to_fill == 0:
        warnings.append(
            "contact morphological closing disabled "
            "(max_gap_to_fill=0; Stage 10 gap fill remains independent)."
        )

    frames, pass2_diags = apply_hard_distance_probability_gates(
        frames, config, pass_id=2
    )
    diagnostics.extend(pass2_diags)

    assert_parallel_channels_preserved(frames)
    return frames, diagnostics, warnings


def run_behavioral_arbitration_sprint2(
    request: BehavioralArbitrationRequest,
    *,
    sprint1_result: Sprint1ArbitrationResult | None = None,
) -> Sprint2ArbitrationResult:
    """Execute Sprint 1 then Stages 5–9 (or continue from an existing Sprint 1)."""

    config = _resolve_config(request)
    if sprint1_result is None:
        sprint1_result = run_behavioral_arbitration_sprint1(request)

    frames, diagnostics, warnings = apply_sprint2_stages(
        sprint1_result.frames,
        config,
        prior_diagnostics=sprint1_result.diagnostics,
        prior_warnings=sprint1_result.warnings,
    )

    return Sprint2ArbitrationResult(
        frames=frames,
        diagnostics=tuple(diagnostics),
        source_s7_run_id=sprint1_result.source_s7_run_id,
        source_s7_run_dir=sprint1_result.source_s7_run_dir,
        source_s6_run_id=sprint1_result.source_s6_run_id,
        source_s6_run_dir=sprint1_result.source_s6_run_dir,
        session_root=sprint1_result.session_root,
        exp_name=sprint1_result.exp_name,
        fps=config.fps,
        n_frames=sprint1_result.n_frames,
        n_directional_rows=int(len(frames)),
        mouse_ids=sprint1_result.mouse_ids,
        detectors_dyadic=sprint1_result.detectors_dyadic,
        detectors_focal=sprint1_result.detectors_focal,
        warnings=tuple(dict.fromkeys(warnings)),
        settings_used=config.to_settings_dict(),
        sprint1_schema_version=sprint1_result.schema_version,
    )


def apply_sprint3_stages(
    frames: pd.DataFrame,
    config: BehavioralArbitrationConfig,
    *,
    prior_diagnostics: tuple[StageDiagnostic, ...] = (),
    prior_warnings: tuple[str, ...] = (),
) -> tuple[pd.DataFrame, list[StageDiagnostic], list[str]]:
    """Apply Stages 10–21 (+ finalize) wrapping ``apply_sprint3_temporal_stages``."""

    frames, temporal_diags, temporal_warnings = apply_sprint3_temporal_stages(
        frames, config
    )
    diagnostics = list(prior_diagnostics) + list(temporal_diags)
    warnings = list(prior_warnings) + list(temporal_warnings)
    assert_parallel_channels_preserved(frames)
    return frames, diagnostics, warnings


def run_behavioral_arbitration_sprint3(
    request: BehavioralArbitrationRequest,
    *,
    sprint2_result: Sprint2ArbitrationResult | None = None,
) -> Sprint3ArbitrationResult:
    """In-memory pipeline through Stage 21 / finalize (no Parquet publish)."""

    config = _resolve_config(request)
    if sprint2_result is None:
        sprint2_result = run_behavioral_arbitration_sprint2(request)

    frames, diagnostics, warnings = apply_sprint3_stages(
        sprint2_result.frames,
        config,
        prior_diagnostics=sprint2_result.diagnostics,
        prior_warnings=sprint2_result.warnings,
    )

    return Sprint3ArbitrationResult(
        frames=frames,
        diagnostics=tuple(diagnostics),
        source_s7_run_id=sprint2_result.source_s7_run_id,
        source_s7_run_dir=sprint2_result.source_s7_run_dir,
        source_s6_run_id=sprint2_result.source_s6_run_id,
        source_s6_run_dir=sprint2_result.source_s6_run_dir,
        session_root=sprint2_result.session_root,
        exp_name=sprint2_result.exp_name,
        fps=config.fps,
        n_frames=sprint2_result.n_frames,
        n_directional_rows=int(len(frames)),
        mouse_ids=sprint2_result.mouse_ids,
        detectors_dyadic=sprint2_result.detectors_dyadic,
        detectors_focal=sprint2_result.detectors_focal,
        warnings=tuple(dict.fromkeys(warnings)),
        settings_used=config.to_settings_dict(),
        sprint1_schema_version=sprint2_result.sprint1_schema_version,
        sprint2_schema_version=sprint2_result.schema_version,
    )


def run_behavioral_arbitration_publication(
    request: BehavioralArbitrationRequest,
    progress_callback: ProgressCallback | None = None,
) -> BehavioralArbitrationRunResult:
    """Full Sprint 1–3 pipeline with immutable S8 publish (never overwrites S7/S6).

    Progress callbacks are coarse phase markers only — no scientific work is
    duplicated for GUI instrumentation.
    """

    purpose = request.run_purpose.strip()
    if not purpose:
        raise PublicationError("run_purpose must not be empty.")

    total_phases = 6
    _report_progress(
        progress_callback,
        phase="intake",
        message="Validating joint S7/S6 readiness…",
        completed_units=0,
        total_units=total_phases,
        is_indeterminate=False,
    )

    config = _resolve_config(request)
    session_root = Path(request.session_root).expanduser().resolve()

    # Resolve sources early for digests + overlap rejection.
    intake = validate_joint_s7_s6_for_s8(
        session_root,
        s7_run_dir=request.s7_run_dir,
        require_exactly_two_mice=config.require_exactly_two_mice,
    )
    s7_run_dir = intake.s7_run_dir.resolve()
    s6_run_dir = intake.s6_run_dir.resolve()

    timestamp = _sanitize_timestamp(request.timestamp or _now_timestamp())
    run_id = f"behavioral_arbitration__{timestamp}"
    paths = _resolve_run_paths(
        session_root=session_root,
        run_id=run_id,
        output_root=request.output_root,
    )
    _reject_output_overlap(
        paths.run_dir,
        protected=(
            session_root / "preprocess",
            session_root / "pose_inference",
            session_root / "tracking_correction",
            session_root / "pose_finalization",
            session_root / "blob_features",
            session_root / COMBINED_FEATURES_DIRNAME,
            session_root / BEHAVIORAL_INFERENCE_DIRNAME,
            s7_run_dir,
            s6_run_dir,
            s6_run_dir / DYADIC_FEATURES_FILENAME,
            s7_run_dir / PREDICTIONS_FILENAME,
            s7_run_dir / DYADIC_WIDE_FILENAME,
            s7_run_dir / FOCAL_WIDE_FILENAME,
        ),
    )
    if paths.run_dir.exists():
        raise PublicationError(f"S8 run directory already exists: {paths.run_dir}")

    created_at = _iso_now()
    try:
        paths.run_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise PublicationError(
            f"S8 run directory cannot be created: {paths.run_dir}: {exc}"
        ) from exc

    predictions_path = s7_run_dir / PREDICTIONS_FILENAME
    dyadic_wide_path = s7_run_dir / DYADIC_WIDE_FILENAME
    focal_wide_path = s7_run_dir / FOCAL_WIDE_FILENAME
    features_path = s6_run_dir / DYADIC_FEATURES_FILENAME
    digests = {
        "predictions_sha256": _sha256(predictions_path) if predictions_path.is_file() else None,
        "dyadic_wide_sha256": _sha256(dyadic_wide_path) if dyadic_wide_path.is_file() else None,
        "focal_wide_sha256": _sha256(focal_wide_path) if focal_wide_path.is_file() else None,
        "dyadic_features_sha256": _sha256(features_path) if features_path.is_file() else None,
    }
    provenance = _build_provenance(
        session_root=session_root,
        s7_run_dir=s7_run_dir,
        s7_run_id=intake.s7_run_id,
        s6_run_dir=s6_run_dir,
        s6_run_id=intake.s6_run_id,
        digests=digests,
        settings=config.to_settings_dict(),
    )
    settings_snapshot = config.to_settings_dict()
    _write_json(
        paths.run_meta,
        _run_meta_payload(
            run_id=run_id,
            run_purpose=purpose,
            created_at=created_at,
            completed_at=None,
            status=STATUS_RUNNING,
            session_root=session_root,
            paths=paths,
            config=config,
            provenance=provenance,
            result=None,
            qc_summary=None,
            error_message=None,
            science_written=False,
            settings_snapshot=settings_snapshot,
            diagnostics_written=False,
        ),
    )
    _write_yaml(paths.settings_used, {**settings_snapshot, "status": STATUS_RUNNING})

    sprint3: Sprint3ArbitrationResult | None = None
    qc_summary: QcSummary | None = None
    error_message: str | None = None
    status = STATUS_FAILED
    success = False
    warnings: tuple[str, ...] = ()
    science_written = False
    diagnostics_written = False
    states_table: pd.DataFrame | None = None
    bouts_table: pd.DataFrame | None = None

    try:
        _report_progress(
            progress_callback,
            phase="arbitration",
            message="Running Pipeline 100 arbitration stages (Sprint 1–3)…",
            completed_units=1,
            total_units=total_phases,
            is_indeterminate=False,
        )
        # Re-enter through request (intake already validated; sprint1 revalidates).
        pub_request = replace(request, s7_run_dir=s7_run_dir)
        sprint3 = run_behavioral_arbitration_sprint3(pub_request)

        _report_progress(
            progress_callback,
            phase="states",
            message="Building framewise behavior states…",
            completed_units=2,
            total_units=total_phases,
            is_indeterminate=False,
        )
        states_table = build_behavior_states_table(
            sprint3.frames,
            session_root=session_root,
            source_s7_run_id=sprint3.source_s7_run_id,
            source_s6_run_id=sprint3.source_s6_run_id,
            run_id=run_id,
        )
        _report_progress(
            progress_callback,
            phase="bouts",
            message="Collapsing exclusive and parallel bouts…",
            completed_units=3,
            total_units=total_phases,
            is_indeterminate=False,
        )
        bouts_table = extract_behavior_bouts(
            states_table,
            fps=sprint3.fps,
            source_s7_run_id=sprint3.source_s7_run_id,
            source_s6_run_id=sprint3.source_s6_run_id,
            run_id=run_id,
        )

        observed = {d.rule_code for d in sprint3.diagnostics}
        expected_rules = list(_SPRINT3_EXPECTED_RULE_CODES)
        if RULE_LABEL_FINALIZE not in observed and RULE_LABEL_UNDEFINED not in observed:
            expected_rules.append(RULE_LABEL_FINALIZE)

        _report_progress(
            progress_callback,
            phase="qc",
            message="Evaluating structural QC and provenance…",
            completed_units=4,
            total_units=total_phases,
            is_indeterminate=False,
        )
        qc_summary = evaluate_behavioral_arbitration_qc(
            states=states_table,
            bouts=bouts_table,
            diagnostics=sprint3.diagnostics,
            n_frames=sprint3.n_frames,
            detectors_dyadic=sprint3.detectors_dyadic,
            detectors_focal=sprint3.detectors_focal,
            source_provenance=provenance,
            linked_s7_eligible=is_s7_run_eligible_for_downstream(s7_run_dir),
            linked_s6_eligible=is_s6_run_eligible_for_downstream(s6_run_dir),
            source_warnings=sprint3.warnings,
            expected_rule_codes=tuple(expected_rules),
        )
        assert_structural_qc(qc_summary)

        _revalidate_sources_before_publish(
            s7_run_dir=s7_run_dir,
            s6_run_dir=s6_run_dir,
            digests=digests,
        )

        _report_progress(
            progress_callback,
            phase="publish",
            message="Publishing immutable S8 artifacts…",
            completed_units=5,
            total_units=total_phases,
            is_indeterminate=False,
        )
        write_behavior_states_parquet(paths.behavior_states, states_table)
        write_behavior_bouts_parquet(paths.behavior_bouts, bouts_table)
        if config.execution.emit_arbitration_diagnostics:
            diag_table = build_arbitration_diagnostics_table(sprint3.diagnostics)
            write_optional_diagnostics_parquet(paths.diagnostics, diag_table)
            diagnostics_written = True
        _write_json(paths.qc, qc_summary.to_metadata())
        science_written = True

        if qc_summary.has_warnings:
            status = STATUS_COMPLETED_WITH_WARNINGS
            warnings = qc_summary.warnings
        else:
            status = STATUS_COMPLETED
            warnings = ()
        success = True
        _report_progress(
            progress_callback,
            phase="complete",
            message=f"Behavioral arbitration {status}.",
            completed_units=total_phases,
            total_units=total_phases,
            is_indeterminate=False,
        )
    except Exception as exc:
        status = STATUS_FAILED
        success = False
        error_message = str(exc)
        _unlink_science_artifacts(paths)
        science_written = False
        diagnostics_written = False
        _report_progress(
            progress_callback,
            phase="failed",
            message=f"Behavioral arbitration failed: {exc}",
            completed_units=None,
            total_units=total_phases,
            is_indeterminate=True,
        )

    completed_at = _iso_now()
    _persist_records(
        paths=paths,
        run_id=run_id,
        run_purpose=purpose,
        created_at=created_at,
        completed_at=completed_at,
        status=status,
        session_root=session_root,
        config=config,
        provenance=provenance,
        result=sprint3,
        qc_summary=qc_summary,
        error_message=error_message,
        science_written=science_written,
        settings_snapshot=settings_snapshot,
        diagnostics_written=diagnostics_written,
    )

    return BehavioralArbitrationRunResult(
        success=success,
        status=status,
        run_id=run_id,
        run_dir=paths.run_dir,
        run_meta_path=paths.run_meta,
        settings_used_path=paths.settings_used,
        processing_log_path=paths.processing_log,
        qc_path=paths.qc,
        behavior_states_path=paths.behavior_states if science_written else None,
        behavior_bouts_path=paths.behavior_bouts if science_written else None,
        diagnostics_path=paths.diagnostics if diagnostics_written else None,
        session_root=session_root,
        source_s7_run_id=intake.s7_run_id,
        source_s6_run_id=intake.s6_run_id,
        warnings=warnings,
        error_message=error_message,
        qc=qc_summary.to_metadata() if qc_summary is not None else None,
    )


def _report_progress(
    progress_callback: ProgressCallback | None,
    *,
    phase: str,
    message: str,
    completed_units: int | None,
    total_units: int | None,
    is_indeterminate: bool,
) -> None:
    if progress_callback is None:
        return
    progress_callback(
        OperationProgress(
            phase=phase,
            message=message,
            completed_units=completed_units,
            total_units=total_units,
            total_is_estimate=False,
            is_indeterminate=is_indeterminate,
            processed_video_time_sec=None,
        )
    )


def is_s8_run_eligible_for_downstream(s8_run_dir: Path | str) -> bool:
    """Return True when a completed S8 run still matches recorded S7/S6 digests."""

    try:
        validate_s8_run_for_downstream(s8_run_dir)
    except (PublicationError, ProvenanceError):
        return False
    return True


def validate_s8_run_for_downstream(s8_run_dir: Path | str) -> dict[str, Any]:
    """Validate a completed S8 run for downstream use (staleness-aware).

    Historical ``status`` on disk is never rewritten. Eligibility becomes
    false when the recorded S7 predictions / S6 features digests no longer
    match, or when the linked S7 run is no longer eligible.
    """

    run_dir = Path(s8_run_dir).expanduser().resolve()
    fingerprint: tuple[Any, ...] = run_dir_fingerprint(
        run_dir,
        (BEHAVIOR_STATES_FILENAME, BEHAVIOR_BOUTS_FILENAME, QC_FILENAME),
    )
    # Invalidate when linked upstream artifacts change even if S8 outputs do not.
    meta_path = run_dir / RUN_META_FILENAME
    if meta_path.is_file():
        try:
            meta = _load_json_object(meta_path)
            source_s7 = meta.get("source_s7")
            source_s6 = meta.get("source_s6")
            upstream: list[Any] = []
            if isinstance(source_s7, Mapping):
                s7_text = _optional_text(source_s7.get("run_dir"))
                if s7_text is not None:
                    upstream.append(
                        run_dir_fingerprint(
                            Path(s7_text),
                            (
                                PREDICTIONS_FILENAME,
                                DYADIC_WIDE_FILENAME,
                                FOCAL_WIDE_FILENAME,
                            ),
                        )
                    )
            if isinstance(source_s6, Mapping):
                s6_text = _optional_text(source_s6.get("run_dir"))
                if s6_text is not None:
                    s6_dir = Path(s6_text)
                    upstream.append(
                        run_dir_fingerprint(s6_dir, (DYADIC_FEATURES_FILENAME,))
                    )
                    # Nested S4/S5 mutations must invalidate even when S6 parquet is unchanged.
                    s6_meta_path = s6_dir / RUN_META_FILENAME
                    if s6_meta_path.is_file():
                        s6_meta = _load_json_object(s6_meta_path)
                        source_s4 = s6_meta.get("source_s4")
                        source_s5 = s6_meta.get("source_s5")
                        if isinstance(source_s4, Mapping):
                            upstream.append(
                                artifact_fingerprint(source_s4.get("final_pose"))
                            )
                            upstream.append(
                                artifact_fingerprint(source_s4.get("prepared_sync"))
                            )
                        if isinstance(source_s5, Mapping):
                            upstream.append(
                                artifact_fingerprint(source_s5.get("blob_features"))
                            )
                            upstream.append(
                                artifact_fingerprint(source_s5.get("prepared_sync"))
                            )
            if upstream:
                fingerprint = (*fingerprint, *upstream)
        except (PublicationError, OSError, TypeError, ValueError):
            fingerprint = (*fingerprint, "upstream_unreadable")
    return memoized_validation(
        kind="validate_s8_run_for_downstream",
        fingerprint=fingerprint,
        compute=lambda: _validate_s8_run_for_downstream_uncached(run_dir),
    )


def _validate_s8_run_for_downstream_uncached(run_dir: Path) -> dict[str, Any]:
    if not run_dir.is_dir():
        raise PublicationError(f"S8 run directory does not exist: {run_dir}")

    meta_path = run_dir / RUN_META_FILENAME
    if not meta_path.is_file():
        raise PublicationError(f"S8 run_meta.json is missing: {meta_path}")
    meta = _load_json_object(meta_path)
    status = _optional_text(meta.get("status"))
    if status not in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS):
        raise PublicationError(
            f"S8 run is not completed (status={status or 'missing'!r})."
        )

    states = run_dir / BEHAVIOR_STATES_FILENAME
    bouts = run_dir / BEHAVIOR_BOUTS_FILENAME
    if not states.is_file():
        raise PublicationError(f"Completed S8 run is missing {BEHAVIOR_STATES_FILENAME}")
    if not bouts.is_file():
        raise PublicationError(f"Completed S8 run is missing {BEHAVIOR_BOUTS_FILENAME}")
    if not (run_dir / QC_FILENAME).is_file():
        raise PublicationError(f"Completed S8 run is missing {QC_FILENAME}")

    source_s7 = meta.get("source_s7")
    source_s6 = meta.get("source_s6")
    if not isinstance(source_s7, Mapping):
        raise PublicationError("S8 run_meta.json is missing source_s7 provenance.")
    if not isinstance(source_s6, Mapping):
        raise PublicationError("S8 run_meta.json is missing source_s6 provenance.")

    s7_run_dir_text = _optional_text(source_s7.get("run_dir"))
    if s7_run_dir_text is None:
        raise PublicationError("S8 run_meta.json source_s7.run_dir is missing.")
    s7_run_dir = Path(s7_run_dir_text)
    if not is_s7_run_eligible_for_downstream(s7_run_dir):
        raise ProvenanceError(
            "S8 source S7 run is no longer eligible for downstream use."
        )
    validate_s7_run_for_downstream(s7_run_dir)

    s6_run_dir_text = _optional_text(source_s6.get("run_dir"))
    if s6_run_dir_text is None:
        raise PublicationError("S8 run_meta.json source_s6.run_dir is missing.")
    s6_run_dir = Path(s6_run_dir_text)
    if not is_s6_run_eligible_for_downstream(s6_run_dir):
        raise ProvenanceError(
            "S8 source S6 run is no longer eligible for downstream use."
        )
    validate_s6_run_for_downstream(s6_run_dir)

    _assert_recorded_digest(
        label="S7 predictions",
        recorded=_optional_text(source_s7.get("predictions_sha256")),
        path=s7_run_dir / PREDICTIONS_FILENAME,
    )
    _assert_recorded_digest(
        label="S7 dyadic_wide",
        recorded=_optional_text(source_s7.get("dyadic_wide_sha256")),
        path=s7_run_dir / DYADIC_WIDE_FILENAME,
        required=False,
    )
    _assert_recorded_digest(
        label="S7 focal_wide",
        recorded=_optional_text(source_s7.get("focal_wide_sha256")),
        path=s7_run_dir / FOCAL_WIDE_FILENAME,
        required=False,
    )
    features_path = Path(
        _optional_text(source_s6.get("dyadic_features"))
        or str(s6_run_dir / DYADIC_FEATURES_FILENAME)
    )
    _assert_recorded_digest(
        label="S6 dyadic_features",
        recorded=_optional_text(source_s6.get("dyadic_features_sha256")),
        path=features_path,
    )

    return {
        "s8_run_dir": run_dir,
        "status": status,
        "session_root": _optional_text(meta.get("session_root")),
        "source_s7_run_dir": s7_run_dir.resolve(),
        "source_s7_run_id": _optional_text(source_s7.get("run_id")),
        "source_s6_run_dir": s6_run_dir.resolve(),
        "source_s6_run_id": _optional_text(source_s6.get("run_id")),
        "predictions_sha256": _optional_text(source_s7.get("predictions_sha256")),
        "dyadic_features_sha256": _optional_text(
            source_s6.get("dyadic_features_sha256")
        ),
    }


def _assert_recorded_digest(
    *,
    label: str,
    recorded: str | None,
    path: Path,
    required: bool = True,
) -> None:
    if recorded is None:
        if required:
            raise ProvenanceError(f"S8 provenance missing recorded digest for {label}.")
        return
    if not path.is_file():
        raise ProvenanceError(f"Recorded {label} artifact missing: {path}")
    current = _sha256(path)
    if current.lower() != recorded.lower():
        raise ProvenanceError(
            f"{label} SHA256 no longer matches the digest recorded by S8."
        )


def _revalidate_sources_before_publish(
    *,
    s7_run_dir: Path,
    s6_run_dir: Path,
    digests: Mapping[str, str | None],
) -> None:
    if not is_s7_run_eligible_for_downstream(s7_run_dir):
        raise ProvenanceError("S7 became ineligible during S8 processing.")
    validate_s7_run_for_downstream(s7_run_dir)
    if not is_s6_run_eligible_for_downstream(s6_run_dir):
        raise ProvenanceError("S6 became ineligible during S8 processing.")
    validate_s6_run_for_downstream(s6_run_dir)

    predictions = s7_run_dir / PREDICTIONS_FILENAME
    if (
        digests.get("predictions_sha256")
        and predictions.is_file()
        and _sha256(predictions).lower() != str(digests["predictions_sha256"]).lower()
    ):
        raise ProvenanceError("S7 predictions SHA256 changed during S8 processing.")
    for name, filename in (
        ("dyadic_wide_sha256", DYADIC_WIDE_FILENAME),
        ("focal_wide_sha256", FOCAL_WIDE_FILENAME),
    ):
        recorded = digests.get(name)
        path = s7_run_dir / filename
        if recorded and path.is_file() and _sha256(path).lower() != str(recorded).lower():
            raise ProvenanceError(
                f"S7 {filename} SHA256 changed during S8 processing."
            )
    features = s6_run_dir / DYADIC_FEATURES_FILENAME
    recorded_feat = digests.get("dyadic_features_sha256")
    if (
        recorded_feat
        and features.is_file()
        and _sha256(features).lower() != str(recorded_feat).lower()
    ):
        raise ProvenanceError(
            "S6 dyadic_features SHA256 changed during S8 processing."
        )


def _build_provenance(
    *,
    session_root: Path,
    s7_run_dir: Path,
    s7_run_id: str,
    s6_run_dir: Path,
    s6_run_id: str,
    digests: Mapping[str, str | None],
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "session_root": _path_text(session_root),
        "settings": dict(settings),
        "source_s7": {
            "run_id": s7_run_id,
            "run_dir": _path_text(s7_run_dir),
            "predictions": _path_text(s7_run_dir / PREDICTIONS_FILENAME),
            "predictions_sha256": digests.get("predictions_sha256"),
            "dyadic_wide": _path_text(s7_run_dir / DYADIC_WIDE_FILENAME),
            "dyadic_wide_sha256": digests.get("dyadic_wide_sha256"),
            "focal_wide": _path_text(s7_run_dir / FOCAL_WIDE_FILENAME),
            "focal_wide_sha256": digests.get("focal_wide_sha256"),
            "eligible_at_publish": is_s7_run_eligible_for_downstream(s7_run_dir),
        },
        "source_s6": {
            "run_id": s6_run_id,
            "run_dir": _path_text(s6_run_dir),
            "dyadic_features": _path_text(s6_run_dir / DYADIC_FEATURES_FILENAME),
            "dyadic_features_sha256": digests.get("dyadic_features_sha256"),
            "eligible_at_publish": is_s6_run_eligible_for_downstream(s6_run_dir),
        },
    }


def _unlink_science_artifacts(paths: _RunPaths) -> None:
    for path in (
        paths.behavior_states,
        paths.behavior_bouts,
        paths.diagnostics,
        paths.qc,
    ):
        if path.exists():
            with suppress(OSError):
                path.unlink()


def _persist_records(
    *,
    paths: _RunPaths,
    run_id: str,
    run_purpose: str,
    created_at: str,
    completed_at: str,
    status: str,
    session_root: Path,
    config: BehavioralArbitrationConfig,
    provenance: Mapping[str, Any],
    result: Sprint3ArbitrationResult | None,
    qc_summary: QcSummary | None,
    error_message: str | None,
    science_written: bool,
    settings_snapshot: Mapping[str, Any],
    diagnostics_written: bool,
) -> None:
    _write_json(
        paths.run_meta,
        _run_meta_payload(
            run_id=run_id,
            run_purpose=run_purpose,
            created_at=created_at,
            completed_at=completed_at,
            status=status,
            session_root=session_root,
            paths=paths,
            config=config,
            provenance=provenance,
            result=result,
            qc_summary=qc_summary,
            error_message=error_message,
            science_written=science_written,
            settings_snapshot=settings_snapshot,
            diagnostics_written=diagnostics_written,
        ),
    )
    snapshot = dict(settings_snapshot)
    snapshot["status"] = status
    _write_yaml(paths.settings_used, snapshot)
    if qc_summary is not None and status in (
        STATUS_COMPLETED,
        STATUS_COMPLETED_WITH_WARNINGS,
    ):
        _write_json(paths.qc, qc_summary.to_metadata())
    _write_processing_log(
        path=paths.processing_log,
        run_id=run_id,
        run_purpose=run_purpose,
        created_at=created_at,
        completed_at=completed_at,
        status=status,
        session_root=session_root,
        paths=paths,
        provenance=provenance,
        result=result,
        qc_summary=qc_summary,
        error_message=error_message,
        science_written=science_written,
    )


def _run_meta_payload(
    *,
    run_id: str,
    run_purpose: str,
    created_at: str,
    completed_at: str | None,
    status: str,
    session_root: Path,
    paths: _RunPaths,
    config: BehavioralArbitrationConfig,
    provenance: Mapping[str, Any],
    result: Sprint3ArbitrationResult | None,
    qc_summary: QcSummary | None,
    error_message: str | None,
    science_written: bool,
    settings_snapshot: Mapping[str, Any],
    diagnostics_written: bool = False,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": RUN_META_SCHEMA_VERSION,
        "run_id": run_id,
        "run_purpose": run_purpose,
        "created_at": created_at,
        "completed_at": completed_at,
        "status": status,
        "session_root": _path_text(session_root),
        "source_s7": dict(provenance.get("source_s7", {})),
        "source_s6": dict(provenance.get("source_s6", {})),
        "fps": float(config.fps),
        "settings": dict(settings_snapshot),
        "detectors_selected": {
            "dyadic": list(result.detectors_dyadic) if result is not None else [],
            "focal": list(result.detectors_focal) if result is not None else [],
        },
        "outputs": {
            "behavior_states_parquet": (
                _path_text(paths.behavior_states) if science_written else None
            ),
            "behavior_bouts_parquet": (
                _path_text(paths.behavior_bouts) if science_written else None
            ),
            "arbitration_diagnostics_parquet": (
                _path_text(paths.diagnostics) if diagnostics_written else None
            ),
            "settings_used": _path_text(paths.settings_used),
            "processing_log": _path_text(paths.processing_log),
            "qc": _path_text(paths.qc),
        },
        "qc": qc_summary.to_metadata() if qc_summary is not None else None,
        "error": error_message,
        "science_written": science_written,
    }
    if result is not None:
        payload["n_frames"] = result.n_frames
        payload["n_directional_rows"] = result.n_directional_rows
        payload["exp_name"] = result.exp_name
        payload["mouse_ids"] = list(result.mouse_ids)
    return payload


def _write_processing_log(
    *,
    path: Path,
    run_id: str,
    run_purpose: str,
    created_at: str,
    completed_at: str,
    status: str,
    session_root: Path,
    paths: _RunPaths,
    provenance: Mapping[str, Any],
    result: Sprint3ArbitrationResult | None,
    qc_summary: QcSummary | None,
    error_message: str | None,
    science_written: bool,
) -> None:
    lines = [
        f"run_id: {run_id}",
        f"run_purpose: {run_purpose}",
        f"created_at: {created_at}",
        f"completed_at: {completed_at}",
        f"status: {status}",
        f"session_root: {_path_text(session_root)}",
        f"run_dir: {_path_text(paths.run_dir)}",
        f"science_written: {science_written}",
        f"source_s7_run_id: {(provenance.get('source_s7') or {}).get('run_id')}",
        f"source_s6_run_id: {(provenance.get('source_s6') or {}).get('run_id')}",
    ]
    if result is not None:
        lines.extend(
            [
                f"exp_name: {result.exp_name}",
                f"n_frames: {result.n_frames}",
                f"n_directional_rows: {result.n_directional_rows}",
                f"n_diagnostics: {len(result.diagnostics)}",
            ]
        )
    if qc_summary is not None:
        lines.extend(
            [
                f"qc_structural_ok: {qc_summary.structural_ok}",
                f"qc_warning_count: {len(qc_summary.warnings)}",
                f"qc_failure_count: {len(qc_summary.failures)}",
            ]
        )
    if error_message:
        lines.append(f"error: {error_message}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _resolve_run_paths(
    *,
    session_root: Path,
    run_id: str,
    output_root: Path | None,
) -> _RunPaths:
    root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else (session_root / BEHAVIORAL_ARBITRATION_DIRNAME)
    )
    run_dir = root / run_id
    return _RunPaths(
        run_dir=run_dir,
        behavior_states=run_dir / BEHAVIOR_STATES_FILENAME,
        behavior_bouts=run_dir / BEHAVIOR_BOUTS_FILENAME,
        diagnostics=run_dir / DIAGNOSTICS_FILENAME,
        run_meta=run_dir / RUN_META_FILENAME,
        settings_used=run_dir / SETTINGS_USED_FILENAME,
        processing_log=run_dir / PROCESSING_LOG_FILENAME,
        qc=run_dir / QC_FILENAME,
    )


def _reject_output_overlap(run_dir: Path, protected: tuple[Path, ...]) -> None:
    resolved_run = run_dir.resolve()
    for protected_path in protected:
        try:
            resolved_protected = Path(protected_path).resolve()
        except OSError:
            continue
        if (
            resolved_run == resolved_protected
            or _is_relative_to(resolved_run, resolved_protected)
            or _is_relative_to(resolved_protected, resolved_run)
        ):
            raise PublicationError(
                "S8 run directory must not overlap S1–S7 artifacts: "
                f"{resolved_run} vs {resolved_protected}"
            )


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return path != parent


def _resolve_config(request: BehavioralArbitrationRequest) -> BehavioralArbitrationConfig:
    if request.config is not None:
        return request.config.validated()
    if request.config_path is not None:
        return load_behavioral_arbitration_config(request.config_path)
    return load_behavioral_arbitration_config(None)


def run_sprint1_from_session(
    session_root: Path | str,
    *,
    s7_run_dir: Path | str | None = None,
    config: BehavioralArbitrationConfig | None = None,
) -> Sprint1ArbitrationResult:
    """Convenience wrapper for headless Sprint 1 execution."""

    return run_behavioral_arbitration_sprint1(
        BehavioralArbitrationRequest(
            session_root=Path(session_root),
            s7_run_dir=Path(s7_run_dir) if s7_run_dir is not None else None,
            config=config,
        )
    )


def run_sprint2_from_session(
    session_root: Path | str,
    *,
    s7_run_dir: Path | str | None = None,
    config: BehavioralArbitrationConfig | None = None,
) -> Sprint2ArbitrationResult:
    """Convenience wrapper for headless Sprint 2 execution."""

    return run_behavioral_arbitration_sprint2(
        BehavioralArbitrationRequest(
            session_root=Path(session_root),
            s7_run_dir=Path(s7_run_dir) if s7_run_dir is not None else None,
            config=config,
        )
    )


def run_sprint3_from_session(
    session_root: Path | str,
    *,
    s7_run_dir: Path | str | None = None,
    config: BehavioralArbitrationConfig | None = None,
) -> Sprint3ArbitrationResult:
    """Convenience wrapper for headless Sprint 3 (in-memory) execution."""

    return run_behavioral_arbitration_sprint3(
        BehavioralArbitrationRequest(
            session_root=Path(session_root),
            s7_run_dir=Path(s7_run_dir) if s7_run_dir is not None else None,
            config=config,
        )
    )


def sprint1_result_summary(result: Sprint1ArbitrationResult) -> dict[str, Any]:
    return result.to_metadata()


def sprint2_result_summary(result: Sprint2ArbitrationResult) -> dict[str, Any]:
    return result.to_metadata()


def sprint3_result_summary(result: Sprint3ArbitrationResult) -> dict[str, Any]:
    return result.to_metadata()


def _sanitize_timestamp(value: str) -> str:
    cleaned = re.sub(r"[^0-9T]", "", value.strip())
    if not cleaned:
        raise PublicationError("timestamp is empty after sanitization.")
    return cleaned


def _now_timestamp() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _iso_now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _path_text(path: Path) -> str:
    return str(path.resolve())


def _sha256(path: Path) -> str:
    return sha256_file(path)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(payload, stream, sort_keys=False)


def _load_json_object(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise PublicationError(f"JSON root must be an object: {path}")
    return payload


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip() if not isinstance(value, str) else value.strip()
    return text or None
