"""Structural and operational QC for Subsystem 08 behavioral arbitration runs.

Structural failures block publication. Ordinary warnings (coverage notes,
nonblocking NaN counts, reciprocity spikes) do not — a completed-with-warnings
run remains eligible for downstream analysis when sources stay eligible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_arbitration.contracts import (
    DIRECTIONAL_IDENTITY_KEYS,
    LABEL_UNDEFINED,
    LABEL_UNRESOLVED,
    PARALLEL_CHANNELS,
    QC_SCHEMA_VERSION,
    REQUIRED_DYADIC_DETECTORS,
    REQUIRED_FOCAL_DETECTORS,
    RULE_RECIPROCITY_MUTUALLY_DISENGAGED,
    RULE_RECIPROCITY_NOSE_TO_NOSE,
    PublicationError,
)
from behavioral_arbitration.results import StageDiagnostic


@dataclass(frozen=True, slots=True)
class QcSummary:
    """Machine-readable QC summary stored in ``qc.json`` / ``run_meta.json``."""

    structural_ok: bool
    warnings: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = QC_SCHEMA_VERSION

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "structural_ok": self.structural_ok,
            "warning_count": len(self.warnings),
            "failure_count": len(self.failures),
            "warnings": list(self.warnings),
            "failures": list(self.failures),
            "metrics": dict(self.metrics),
        }


def evaluate_behavioral_arbitration_qc(
    *,
    states: pd.DataFrame,
    bouts: pd.DataFrame,
    diagnostics: Sequence[StageDiagnostic] = (),
    n_frames: int | None = None,
    detectors_dyadic: Sequence[str] = REQUIRED_DYADIC_DETECTORS,
    detectors_focal: Sequence[str] = REQUIRED_FOCAL_DETECTORS,
    source_provenance: Mapping[str, Any] | None = None,
    linked_s7_eligible: bool | None = None,
    linked_s6_eligible: bool | None = None,
    source_warnings: Sequence[str] = (),
    expected_rule_codes: Sequence[str] | None = None,
) -> QcSummary:
    """Evaluate structural integrity and lightweight operational warnings."""

    failures: list[str] = []
    warnings: list[str] = list(source_warnings)
    metrics: dict[str, Any] = {
        "n_state_rows": int(len(states)),
        "n_bout_rows": int(len(bouts)),
        "n_diagnostic_stages": int(len(diagnostics)),
        "detectors": {
            "dyadic": list(detectors_dyadic),
            "focal": list(detectors_focal),
        },
    }
    if source_provenance is not None:
        metrics["source_provenance"] = dict(source_provenance)

    if states.empty:
        failures.append("behavior_states table is empty")

    _eligibility_flags(
        metrics,
        failures,
        warnings,
        linked_s7_eligible=linked_s7_eligible,
        linked_s6_eligible=linked_s6_eligible,
        source_provenance=source_provenance,
    )
    _detector_coverage(states, detectors_dyadic, detectors_focal, metrics, failures)
    _directional_uniqueness(states, n_frames=n_frames, metrics=metrics, failures=failures)
    _stage_completion(diagnostics, expected_rule_codes, metrics, failures, warnings)
    _undefined_unobserved_counts(states, metrics, warnings)
    _rule_change_counts(diagnostics, metrics)
    _parallel_coverage(states, metrics, warnings)
    _reciprocity_counts(diagnostics, metrics, warnings)
    _bout_duration_summary(bouts, metrics, warnings)
    _nonfinite_checks(states, bouts, metrics, warnings, failures)
    _provenance_present(source_provenance, metrics, failures)

    warnings = list(dict.fromkeys(str(w) for w in warnings if str(w).strip()))
    if failures:
        return QcSummary(
            structural_ok=False,
            warnings=tuple(warnings),
            failures=tuple(failures),
            metrics=metrics,
        )
    return QcSummary(
        structural_ok=True,
        warnings=tuple(warnings),
        failures=(),
        metrics=metrics,
    )


def assert_structural_qc(summary: QcSummary) -> None:
    if not summary.structural_ok:
        raise PublicationError(
            "S8 structural QC failed: " + "; ".join(summary.failures)
        )


def _eligibility_flags(
    metrics: dict[str, Any],
    failures: list[str],
    warnings: list[str],
    *,
    linked_s7_eligible: bool | None,
    linked_s6_eligible: bool | None,
    source_provenance: Mapping[str, Any] | None,
) -> None:
    flags = {
        "linked_s7_eligible": linked_s7_eligible,
        "linked_s6_eligible": linked_s6_eligible,
    }
    if source_provenance is not None:
        src7 = source_provenance.get("source_s7")
        src6 = source_provenance.get("source_s6")
        if isinstance(src7, Mapping) and "eligible_at_publish" in src7:
            flags["linked_s7_eligible"] = bool(src7.get("eligible_at_publish"))
        if isinstance(src6, Mapping) and "eligible_at_publish" in src6:
            flags["linked_s6_eligible"] = bool(src6.get("eligible_at_publish"))
    metrics["eligibility"] = flags
    if flags.get("linked_s7_eligible") is False:
        failures.append("linked S7 run is not eligible for downstream use")
    if flags.get("linked_s6_eligible") is False:
        failures.append("linked S6 run is not eligible for downstream use")
    if flags.get("linked_s7_eligible") is None:
        warnings.append("linked S7 eligibility flag not provided to QC")
    if flags.get("linked_s6_eligible") is None:
        warnings.append("linked S6 eligibility flag not provided to QC")


def _detector_coverage(
    states: pd.DataFrame,
    detectors_dyadic: Sequence[str],
    detectors_focal: Sequence[str],
    metrics: dict[str, Any],
    failures: list[str],
) -> None:
    present_dyadic = [d for d in detectors_dyadic if f"pred_{d}" in states.columns]
    present_focal = [d for d in detectors_focal if f"pred_{d}" in states.columns]
    metrics["detector_coverage"] = {
        "dyadic_present": present_dyadic,
        "focal_present": present_focal,
        "dyadic_missing": [d for d in detectors_dyadic if d not in present_dyadic],
        "focal_missing": [d for d in detectors_focal if d not in present_focal],
    }
    missing = metrics["detector_coverage"]["dyadic_missing"] + metrics[
        "detector_coverage"
    ]["focal_missing"]
    if missing:
        failures.append(
            "states table missing required detector pred columns: "
            + ", ".join(str(x) for x in missing)
        )


def _directional_uniqueness(
    states: pd.DataFrame,
    *,
    n_frames: int | None,
    metrics: dict[str, Any],
    failures: list[str],
) -> None:
    if states.empty:
        return
    keys = [c for c in DIRECTIONAL_IDENTITY_KEYS if c in states.columns]
    if len(keys) != len(DIRECTIONAL_IDENTITY_KEYS):
        failures.append(
            "states missing directional identity keys: "
            + ", ".join(DIRECTIONAL_IDENTITY_KEYS)
        )
        return
    n_dup = int(states.duplicated(subset=list(keys), keep=False).sum())
    n_unique_frames = int(
        pd.to_numeric(states["dense_frame_idx"], errors="coerce").nunique(dropna=True)
    )
    n_rows = int(len(states))
    metrics["directional"] = {
        "n_duplicate_key_rows": n_dup,
        "n_unique_frames": n_unique_frames,
        "n_directional_rows": n_rows,
        "expected_rows_k2": (2 * int(n_frames)) if n_frames is not None else None,
    }
    if n_dup:
        failures.append(
            f"directional uniqueness violated: {n_dup} rows share "
            "(dense_frame_idx, focal_id, partner_id)"
        )
    expected = metrics["directional"]["expected_rows_k2"]
    if expected is not None and n_rows != expected:
        failures.append(
            f"K=2 cardinality mismatch: got {n_rows} directional rows, "
            f"expected ~2*n_frames={expected}"
        )


def _stage_completion(
    diagnostics: Sequence[StageDiagnostic],
    expected_rule_codes: Sequence[str] | None,
    metrics: dict[str, Any],
    failures: list[str],
    warnings: list[str],
) -> None:
    observed = [d.rule_code for d in diagnostics]
    metrics["stage_rule_codes_observed"] = list(dict.fromkeys(observed))
    if not diagnostics:
        warnings.append("no stage diagnostics recorded")
        return
    if expected_rule_codes:
        missing = [c for c in expected_rule_codes if c not in set(observed)]
        metrics["stage_rule_codes_missing"] = missing
        if missing:
            failures.append(
                "expected Sprint-3 rule codes missing from diagnostics: "
                + ", ".join(missing)
            )


def _undefined_unobserved_counts(
    states: pd.DataFrame,
    metrics: dict[str, Any],
    warnings: list[str],
) -> None:
    if states.empty or "behavior_label" not in states.columns:
        return
    labels = states["behavior_label"].astype(str)
    n_undefined = int(labels.eq(LABEL_UNDEFINED).sum())
    n_unresolved = int(labels.eq(LABEL_UNRESOLVED).sum())
    label_counts = {str(k): int(v) for k, v in labels.value_counts().items()}
    metrics["label_counts"] = label_counts
    metrics["n_undefined"] = n_undefined
    metrics["n_unresolved"] = n_unresolved

    n_unobserved = 0
    for col in states.columns:
        if col.startswith("score_origin_"):
            n_unobserved += int(
                states[col].astype(str).eq("routing_sentinel").sum()
            )
    metrics["n_routing_sentinel_cells"] = n_unobserved
    if n_unresolved:
        warnings.append(
            f"{n_unresolved} rows still have provisional unresolved labels at QC"
        )


def _rule_change_counts(
    diagnostics: Sequence[StageDiagnostic],
    metrics: dict[str, Any],
) -> None:
    counts: dict[str, int] = {}
    for diag in diagnostics:
        counts[diag.rule_code] = counts.get(diag.rule_code, 0) + int(
            diag.n_rows_changed
        )
    metrics["rule_change_counts"] = dict(sorted(counts.items()))


def _parallel_coverage(
    states: pd.DataFrame,
    metrics: dict[str, Any],
    warnings: list[str],
) -> None:
    coverage: dict[str, Any] = {}
    for channel in PARALLEL_CHANNELS:
        pred_col = f"pred_{channel}"
        if pred_col not in states.columns:
            coverage[channel] = {"present": False}
            if channel == "contact":
                warnings.append("pred_contact missing from states (parallel channel)")
            continue
        preds = pd.to_numeric(states[pred_col], errors="coerce").fillna(0).astype(int)
        coverage[channel] = {
            "present": True,
            "n_positive": int(preds.eq(1).sum()),
            "positive_fraction": float(preds.eq(1).mean()) if len(preds) else 0.0,
        }
    metrics["parallel_coverage"] = coverage


def _reciprocity_counts(
    diagnostics: Sequence[StageDiagnostic],
    metrics: dict[str, Any],
    _warnings: list[str],
) -> None:
    # Counts only — no arbitrary biological spike threshold for warnings.
    nn = sum(
        int(d.n_rows_changed)
        for d in diagnostics
        if d.rule_code == RULE_RECIPROCITY_NOSE_TO_NOSE
    )
    md = sum(
        int(d.n_rows_changed)
        for d in diagnostics
        if d.rule_code == RULE_RECIPROCITY_MUTUALLY_DISENGAGED
    )
    metrics["reciprocity_repair_counts"] = {
        "nose_to_nose": nn,
        "mutually_disengaged": md,
    }


def _bout_duration_summary(
    bouts: pd.DataFrame,
    metrics: dict[str, Any],
    warnings: list[str],
) -> None:
    if bouts.empty or "bout_duration_frames" not in bouts.columns:
        metrics["bout_summary"] = {
            "n_bouts": int(len(bouts)),
            "by_channel": {},
            "duration_frames": None,
        }
        if bouts.empty:
            warnings.append("bout table is empty")
        return
    durations = pd.to_numeric(bouts["bout_duration_frames"], errors="coerce").dropna()
    by_channel: dict[str, int] = {}
    if "channel" in bouts.columns:
        by_channel = {
            str(k): int(v) for k, v in bouts["channel"].value_counts().items()
        }
    summary = {
        "n_bouts": int(len(bouts)),
        "by_channel": by_channel,
        "duration_frames": {
            "min": float(durations.min()) if len(durations) else None,
            "median": float(durations.median()) if len(durations) else None,
            "max": float(durations.max()) if len(durations) else None,
        },
    }
    metrics["bout_summary"] = summary


def _nonfinite_checks(
    states: pd.DataFrame,
    bouts: pd.DataFrame,
    metrics: dict[str, Any],
    warnings: list[str],
    failures: list[str],
) -> None:
    n_nonfinite_probs = 0
    for col in states.columns:
        if not col.startswith("prob_"):
            continue
        vals = pd.to_numeric(states[col], errors="coerce")
        # NaN is allowed for unobserved; Inf is structural.
        arr = vals.to_numpy(dtype=float)
        n_inf = int(np.isinf(arr).sum()) if len(arr) else 0
        n_nonfinite_probs += n_inf
    metrics["n_infinite_prob_values"] = n_nonfinite_probs
    if n_nonfinite_probs:
        failures.append(f"{n_nonfinite_probs} infinite probability values in states")

    if not bouts.empty and "bout_duration_frames" in bouts.columns:
        dur = pd.to_numeric(bouts["bout_duration_frames"], errors="coerce")
        if (dur <= 0).any():
            failures.append("bout_duration_frames contains non-positive values")
        if (~np.isfinite(dur.to_numpy(dtype=float))).any():
            warnings.append("bout_duration_frames contains non-finite values")


def _provenance_present(
    source_provenance: Mapping[str, Any] | None,
    metrics: dict[str, Any],
    failures: list[str],
) -> None:
    if source_provenance is None:
        failures.append("source/config provenance missing from QC evaluation")
        return
    src7 = source_provenance.get("source_s7")
    src6 = source_provenance.get("source_s6")
    settings = source_provenance.get("settings")
    present = {
        "source_s7": isinstance(src7, Mapping) and bool(src7),
        "source_s6": isinstance(src6, Mapping) and bool(src6),
        "settings": isinstance(settings, Mapping) and bool(settings),
    }
    metrics["provenance_present"] = present
    if not present["source_s7"]:
        failures.append("source_s7 provenance block missing")
    if not present["source_s6"]:
        failures.append("source_s6 provenance block missing")
    if not present["settings"]:
        failures.append("settings provenance block missing")
