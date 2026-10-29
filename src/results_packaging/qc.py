"""Structural and scientific QC for Subsystem 09 package publication.

Structural failures block publication. Scientific reconstruction statistics and
packaging notes become nonblocking warnings (U-R8). No new biological quality
thresholds are invented.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from results_packaging.bouts import assert_lossless_bout_copy, authoritative_bout_content_key
from results_packaging.contracts import (
    AUTHORITATIVE_BOUT_KEY_COLUMNS,
    BOUTS_SCHEMA_VERSION,
    ETHOGRAM_SUMMARY_SCHEMA_VERSION,
    PACKAGE_BOUT_STAMP_COLUMNS,
    QC_SCHEMA_VERSION,
    SYNC_INDEX_REQUIRED_COLUMNS,
    SYNC_SCHEMA_VERSION,
    TARGET_FPS,
    BoutPackagingError,
    PublicationError,
)
from results_packaging.export_profiles import ExportProfileResult
from results_packaging.sync_index import SyncIndexBuildResult


@dataclass(frozen=True, slots=True)
class QcSummary:
    """Machine-readable QC summary stored in ``qc.json`` / ``run_meta.json``."""

    structural_ok: bool
    warnings: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    informational: tuple[str, ...] = ()
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
            "informational_count": len(self.informational),
            "warnings": list(self.warnings),
            "failures": list(self.failures),
            "informational": list(self.informational),
            "metrics": dict(self.metrics),
        }


def evaluate_results_package_qc(
    *,
    sync: SyncIndexBuildResult,
    packaged_bouts: pd.DataFrame,
    ethogram_summary: pd.DataFrame,
    source_s8_bouts: pd.DataFrame,
    states: pd.DataFrame | None,
    export_profiles: Mapping[str, ExportProfileResult] | None = None,
    source_warnings: Sequence[str] = (),
    dependency_notes: Sequence[str] = (),
    source_s8_eligible: bool = True,
    source_s8_status: str | None = None,
    prepared_video_present: bool = True,
    require_prepared_video: bool = True,
    package_mode: str = "reference_lightweight",
    full_bout_preservation: bool = True,
) -> QcSummary:
    """Evaluate structural blockers, scientific warnings, and informational notes."""

    failures: list[str] = []
    warnings: list[str] = []
    informational: list[str] = []
    metrics: dict[str, Any] = {
        "package_mode": package_mode,
        "full_bout_preservation": bool(full_bout_preservation),
        "fps": TARGET_FPS,
    }

    if not source_s8_eligible:
        failures.append("Source S8 run is not currently eligible for packaging.")
    if source_s8_status == "completed_with_warnings":
        warnings.append("Source S8 run completed_with_warnings (inherited).")
        informational.append("Inherited S8 completed_with_warnings status.")

    for note in source_warnings:
        text = str(note).strip()
        if text:
            warnings.append(f"Inherited S8/source warning: {text}")
    for note in dependency_notes:
        text = str(note).strip()
        if text and "digest not hashed" in text.lower():
            informational.append(text)
        elif text:
            warnings.append(text)

    if require_prepared_video and not prepared_video_present:
        failures.append("Required prepared video is missing.")

    _sync_structural(sync, metrics, failures, warnings, informational)
    _bout_structural(
        packaged_bouts,
        source_s8_bouts,
        metrics,
        failures,
    )
    _ethogram_structural(ethogram_summary, metrics, failures)
    _scientific_channel_metrics(packaged_bouts, states, metrics, informational)
    _reconstruction_qc(packaged_bouts, states, metrics, warnings, failures)
    _profile_qc(export_profiles or {}, metrics, warnings, informational)

    warnings = list(dict.fromkeys(str(w) for w in warnings if str(w).strip()))
    informational = list(
        dict.fromkeys(str(i) for i in informational if str(i).strip())
    )
    failures = list(dict.fromkeys(str(f) for f in failures if str(f).strip()))

    if failures:
        return QcSummary(
            structural_ok=False,
            warnings=tuple(warnings),
            failures=tuple(failures),
            informational=tuple(informational),
            metrics=metrics,
        )
    return QcSummary(
        structural_ok=True,
        warnings=tuple(warnings),
        failures=(),
        informational=tuple(informational),
        metrics=metrics,
    )


def assert_structural_qc(summary: QcSummary) -> None:
    if not summary.structural_ok:
        raise PublicationError(
            "S9 structural QC failed: " + "; ".join(summary.failures)
        )


def _sync_structural(
    sync: SyncIndexBuildResult,
    metrics: dict[str, Any],
    failures: list[str],
    warnings: list[str],
    informational: list[str],
) -> None:
    table = sync.sync_index
    metrics["sync_audit"] = sync.audit_dict()
    if table is None or table.empty:
        failures.append("sync_index is empty")
        return
    missing = [c for c in SYNC_INDEX_REQUIRED_COLUMNS if c not in table.columns]
    if missing:
        failures.append("sync_index missing columns: " + ", ".join(missing))
    if "schema_version" in table.columns:
        bad = table["schema_version"].astype(str) != SYNC_SCHEMA_VERSION
        if bool(bad.any()):
            failures.append("sync_index schema_version mismatch")
    if "sample_idx" in table.columns and "dense_frame_idx" in table.columns:
        if not np.array_equal(
            table["sample_idx"].to_numpy(dtype=np.int64),
            table["dense_frame_idx"].to_numpy(dtype=np.int64),
        ):
            failures.append("sample_idx must equal dense_frame_idx")
        sample = table["sample_idx"].to_numpy(dtype=np.int64)
        if len(sample) != int(sample[-1] - sample[0] + 1) if len(sample) else True:
            failures.append("sync_index sample_idx is not contiguous")
        if len(sample) and (int(sample[0]) != 0 or not np.all(np.diff(sample) == 1)):
            failures.append("sync_index sample_idx must be contiguous from 0")
        if len(sample) != len(np.unique(sample)):
            failures.append("duplicate sync_index sample_idx values")
    if "prepared_frame_idx" in table.columns:
        prep = table["prepared_frame_idx"].to_numpy(dtype=np.int64)
        if np.any(prep < 0):
            failures.append("prepared_frame_idx contains negative values")
        if len(prep) > 1 and np.any(np.diff(prep) < 0):
            failures.append("prepared_frame_idx is not monotonically non-decreasing")
        metrics["n_prepared_unique"] = int(len(np.unique(prep)))
        metrics["max_prepared_frame_idx"] = int(prep.max()) if len(prep) else None
    if "timing_residual_sec" in table.columns:
        resid = table["timing_residual_sec"].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(resid)):
            failures.append("non-finite timing_residual_sec values")
        else:
            metrics["timing_residual"] = {
                "min": float(np.min(resid)),
                "max": float(np.max(resid)),
                "mean": float(np.mean(resid)),
                "abs_p95": float(np.percentile(np.abs(resid), 95)),
            }
    if "is_exact_source_sample" in table.columns:
        exact = table["is_exact_source_sample"].to_numpy(dtype=bool)
        metrics["n_exact_source_samples"] = int(exact.sum())
        metrics["frac_non_exact"] = float(1.0 - (exact.mean() if len(exact) else 1.0))
    audit = sync.audit_dict()
    max_many = audit.get("max_many_to_one")
    if isinstance(max_many, int) and max_many > 1:
        informational.append(
            f"Many-to-one video-frame mapping present (max multiplicity={max_many})."
        )
    if sync.warnings:
        warnings.extend(str(w) for w in sync.warnings)


def _bout_structural(
    packaged: pd.DataFrame,
    source: pd.DataFrame,
    metrics: dict[str, Any],
    failures: list[str],
) -> None:
    metrics["n_bouts_source"] = int(len(source))
    metrics["n_bouts_packaged"] = int(len(packaged))
    if packaged is None:
        failures.append("packaged behavior_bouts missing")
        return
    if int(len(packaged)) != int(len(source)):
        failures.append(
            f"Bout count mismatch vs S8 (packaged={len(packaged)}, source={len(source)})."
        )
    try:
        assert_lossless_bout_copy(source, packaged)
        metrics["lossless_bout_copy"] = True
    except BoutPackagingError as exc:
        failures.append(str(exc))
        metrics["lossless_bout_copy"] = False

    for col in PACKAGE_BOUT_STAMP_COLUMNS:
        if col not in packaged.columns:
            failures.append(f"packaged bouts missing stamp column {col}")
    if "schema_version" in packaged.columns and not (
        packaged["schema_version"].astype(str) == BOUTS_SCHEMA_VERSION
    ).all():
        failures.append("behavior_bouts schema_version mismatch")
    if "bout_id" in packaged.columns:
        ids = packaged["bout_id"].astype(str)
        if ids.duplicated().any():
            failures.append("duplicate package bout_id values")
    if "source_s8_bout_id" in packaged.columns:
        src_ids = packaged["source_s8_bout_id"].astype(str)
        if src_ids.duplicated().any():
            failures.append("duplicate source_s8_bout_id values in package")
        source_ids = set(source["bout_id"].astype(str).tolist())
        packaged_ids = set(src_ids.tolist())
        if source_ids != packaged_ids:
            failures.append("package source_s8_bout_id set differs from S8 bout_id set")

    # Boundary integrity vs dense inclusive duration.
    if {
        "start_sample_idx",
        "end_sample_idx",
        "bout_duration_frames",
    }.issubset(packaged.columns):
        start = packaged["start_sample_idx"].to_numpy(dtype=np.int64)
        end = packaged["end_sample_idx"].to_numpy(dtype=np.int64)
        dur = packaged["bout_duration_frames"].to_numpy(dtype=np.int64)
        if np.any(end < start):
            failures.append("bout end_sample_idx < start_sample_idx")
        if np.any(dur != (end - start + 1)):
            failures.append("bout_duration_frames inconsistent with inclusive endpoints")

    channel_counts = {
        str(k): int(v)
        for k, v in packaged["channel"].astype(str).value_counts().sort_index().items()
    }
    metrics["channel_counts"] = channel_counts
    short_ex = int(
        (
            (packaged["channel"].astype(str) == "exclusive")
            & (packaged["bout_duration_frames"] < 10)
        ).sum()
    )
    metrics["n_short_exclusive_lt_10"] = short_ex
    metrics["authoritative_key_columns"] = list(AUTHORITATIVE_BOUT_KEY_COLUMNS)
    # short_ex retained in metrics only — not a quality gate (no emit floors).


def _ethogram_structural(
    summary: pd.DataFrame,
    metrics: dict[str, Any],
    failures: list[str],
) -> None:
    if summary is None:
        failures.append("ethogram_summary missing")
        return
    metrics["n_ethogram_summary_rows"] = int(len(summary))
    if summary.empty:
        failures.append("ethogram_summary is empty")
        return
    if "schema_version" in summary.columns and not (
        summary["schema_version"].astype(str) == ETHOGRAM_SUMMARY_SCHEMA_VERSION
    ).all():
        failures.append("ethogram_summary schema_version mismatch")
    if "summary_kind" in summary.columns:
        metrics["summary_kinds"] = sorted(
            summary["summary_kind"].astype(str).unique().tolist()
        )


def _scientific_channel_metrics(
    packaged: pd.DataFrame,
    states: pd.DataFrame | None,
    metrics: dict[str, Any],
    informational: list[str],
) -> None:
    grain_counts = {
        str(k): int(v)
        for k, v in packaged["identity_grain"].astype(str).value_counts().sort_index().items()
    }
    metrics["identity_grain_counts"] = grain_counts
    metrics["n_directional_bouts"] = int(grain_counts.get("directional", 0))
    metrics["n_symmetric_bouts"] = int(grain_counts.get("symmetric_pair", 0))
    informational.append(
        "Contact/SBS parallel channels are not mutually exclusive with exclusive labels."
    )

    if states is None or states.empty or "behavior_label" not in states.columns:
        metrics["states_available"] = False
        return
    metrics["states_available"] = True
    labels = states["behavior_label"].astype(str)
    metrics["n_state_rows"] = int(len(states))
    metrics["engaged_coverage"] = {
        "n_rows": int((labels == "engaged").sum()),
        "fraction": float((labels == "engaged").mean()),
    }
    metrics["undefined_coverage"] = {
        "n_rows": int((labels == "undefined").sum()),
        "fraction": float((labels == "undefined").mean()),
    }
    if "label_reconstruction_mode" in packaged.columns:
        recon = packaged["label_reconstruction_mode"].astype(str)
        metrics["bout_reconstruction_modes"] = {
            str(k): int(v) for k, v in recon.value_counts().sort_index().items()
        }


def _reconstruction_qc(
    packaged: pd.DataFrame,
    states: pd.DataFrame | None,
    metrics: dict[str, Any],
    warnings: list[str],
    failures: list[str],
) -> None:
    """Report bout↔state reconstruction statistics without biological thresholds.

    Limited mismatches are warnings with counts (U-R8). A total exclusive miss
    against available states is treated as a structural wrong-source signal.
    """

    if states is None or states.empty:
        metrics["reconstruction"] = {"checked": False, "reason": "states_unavailable"}
        return
    required = {"dense_frame_idx", "focal_id", "partner_id", "behavior_label"}
    if not required.issubset(states.columns):
        metrics["reconstruction"] = {"checked": False, "reason": "states_missing_columns"}
        return

    exclusive = packaged[packaged["channel"].astype(str) == "exclusive"].copy()
    n_checked = 0
    n_mismatch = 0
    mismatch_examples: list[dict[str, Any]] = []
    # Index states for directional lookups.
    state_keys = states.assign(
        dense_frame_idx=states["dense_frame_idx"].astype(np.int64),
        focal_id=states["focal_id"].astype(np.int64),
        partner_id=states["partner_id"].astype(np.int64),
    )
    grouped = {
        (int(f), int(p)): g.sort_values("dense_frame_idx")
        for (f, p), g in state_keys.groupby(["focal_id", "partner_id"], sort=False)
    }

    for _, row in exclusive.iterrows():
        if pd.isna(row.get("focal_id")) or pd.isna(row.get("partner_id")):
            continue
        key = (int(row["focal_id"]), int(row["partner_id"]))
        block = grouped.get(key)
        if block is None:
            n_mismatch += 1
            continue
        start = int(row["start_sample_idx"])
        end = int(row["end_sample_idx"])
        window = block[
            (block["dense_frame_idx"] >= start) & (block["dense_frame_idx"] <= end)
        ]
        n_checked += 1
        expected = str(row["behavior_name"])
        labels = window["behavior_label"].astype(str).tolist()
        if len(labels) != (end - start + 1) or any(lab != expected for lab in labels):
            n_mismatch += 1
            if len(mismatch_examples) < 5:
                mismatch_examples.append(
                    {
                        "source_s8_bout_id": str(row.get("source_s8_bout_id")),
                        "behavior_name": expected,
                        "start_sample_idx": start,
                        "end_sample_idx": end,
                        "observed_labels": sorted(set(labels)),
                    }
                )

    # Parallel channels: contiguous pred_* == 1 where available.
    parallel_checked = 0
    parallel_mismatch = 0
    for channel, col in (("contact", "pred_contact"), ("side_by_side", "pred_side_by_side")):
        if col not in states.columns:
            continue
        subset = packaged[packaged["channel"].astype(str) == channel]
        undirected = states.copy()
        undirected["mouse_a_id"] = np.minimum(
            undirected["focal_id"].astype(np.int64),
            undirected["partner_id"].astype(np.int64),
        )
        undirected["mouse_b_id"] = np.maximum(
            undirected["focal_id"].astype(np.int64),
            undirected["partner_id"].astype(np.int64),
        )
        # Use one direction only to avoid double-counting pair samples.
        undirected = undirected[
            undirected["focal_id"].astype(np.int64)
            < undirected["partner_id"].astype(np.int64)
        ]
        for _, row in subset.iterrows():
            start = int(row["start_sample_idx"])
            end = int(row["end_sample_idx"])
            a = int(row["mouse_a_id"])
            b = int(row["mouse_b_id"])
            window = undirected[
                (undirected["mouse_a_id"] == a)
                & (undirected["mouse_b_id"] == b)
                & (undirected["dense_frame_idx"] >= start)
                & (undirected["dense_frame_idx"] <= end)
            ]
            parallel_checked += 1
            vals = window[col].to_numpy()
            if len(vals) != (end - start + 1) or not np.all(vals.astype(np.int64) == 1):
                parallel_mismatch += 1

    metrics["reconstruction"] = {
        "checked": True,
        "exclusive_checked": n_checked,
        "exclusive_mismatch": n_mismatch,
        "parallel_checked": parallel_checked,
        "parallel_mismatch": parallel_mismatch,
        "mismatch_examples": mismatch_examples,
        "note": (
            "Reconstruction mismatches are reported as counts. Legitimate S8 "
            "reconstruction provenance is not classified as corruption (U-R8)."
        ),
    }
    if n_checked > 0 and n_mismatch == n_checked:
        failures.append(
            "All checked exclusive bouts disagree with referenced S8 states "
            "(likely wrong source selection)."
        )
    elif n_mismatch > 0:
        warnings.append(
            f"Exclusive bout-state reconstruction mismatches: {n_mismatch}/{n_checked}."
        )
    if parallel_mismatch > 0:
        warnings.append(
            "Parallel bout-pred reconstruction mismatches: "
            f"{parallel_mismatch}/{parallel_checked}."
        )


def _profile_qc(
    profiles: Mapping[str, ExportProfileResult],
    metrics: dict[str, Any],
    warnings: list[str],
    informational: list[str],
) -> None:
    profile_metrics: dict[str, Any] = {}
    for profile_id, result in profiles.items():
        profile_metrics[profile_id] = {
            "n_included": int(result.n_included),
            "n_excluded": int(result.n_excluded),
            "exclusion_reason_counts": dict(result.exclusion_reason_counts),
        }
        informational.append(
            f"Optional profile {profile_id}: included={result.n_included}, "
            f"excluded={result.n_excluded}."
        )
        warnings.extend(str(w) for w in result.warnings)
    metrics["export_profiles"] = profile_metrics
    if not profiles:
        informational.append(
            "No optional export profiles selected; analysis_full only."
        )


def packaged_bout_content_checksum(packaged: pd.DataFrame) -> str:
    """Deterministic checksum over authoritative bout content keys (order-independent)."""

    keys = sorted(authoritative_bout_content_key(row) for _, row in packaged.iterrows())
    payload = "\n".join(keys).encode("utf-8")
    import hashlib

    return hashlib.sha256(payload).hexdigest()
