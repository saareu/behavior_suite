"""Lightweight automated QC for Subsystem 04 final pose outputs.

Distinguishes structural failures (block completion) from quality warnings
(recorded without blocking). Thresholds are practical defaults, not
scientifically validated pass/fail criteria.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from pose_finalization.contracts import (
    FINAL_POSE_COLUMNS,
    FINAL_POSE_SCHEMA_VERSION,
    PROVENANCE_KALMAN_GAP_FILL,
    PROVENANCE_LINEAR_GAP_FILL,
    PROVENANCE_MISSING,
    PROVENANCE_PROTECTED_MANUAL_BLANK,
    PROVENANCE_TEMPORAL_RESAMPLE,
    QC_IMPUTED_FRACTION_WARN_DEFAULT,
    QC_JUMP_PX_WARN_DEFAULT,
    QC_MISSING_FRACTION_WARN_DEFAULT,
    QC_SCHEMA_VERSION,
    QC_SMOOTH_DISPLACEMENT_PX_WARN_DEFAULT,
    PoseFinalizationError,
)
from pose_finalization.dense_pose import DensePoseResult
from pose_finalization.smoothing import SmoothResult


@dataclass(frozen=True, slots=True)
class QcConfig:
    """Configurable practical QC warning thresholds."""

    missing_fraction_warn: float = QC_MISSING_FRACTION_WARN_DEFAULT
    imputed_fraction_warn: float = QC_IMPUTED_FRACTION_WARN_DEFAULT
    jump_px_warn: float = QC_JUMP_PX_WARN_DEFAULT
    smooth_displacement_px_warn: float = QC_SMOOTH_DISPLACEMENT_PX_WARN_DEFAULT

    def __post_init__(self) -> None:
        for name, value in (
            ("missing_fraction_warn", self.missing_fraction_warn),
            ("imputed_fraction_warn", self.imputed_fraction_warn),
        ):
            if not np.isfinite(value) or value < 0.0 or value > 1.0:
                raise PoseFinalizationError(f"{name} must be in [0, 1].")
        for name, value in (
            ("jump_px_warn", self.jump_px_warn),
            ("smooth_displacement_px_warn", self.smooth_displacement_px_warn),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise PoseFinalizationError(f"{name} must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class QcSummary:
    """Machine-readable QC summary stored inside ``run_meta.json``."""

    structural_ok: bool
    warnings: tuple[str, ...] = ()
    failures: tuple[str, ...] = ()
    metrics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_warnings(self) -> bool:
        return bool(self.warnings)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": QC_SCHEMA_VERSION,
            "structural_ok": self.structural_ok,
            "warning_count": len(self.warnings),
            "failure_count": len(self.failures),
            "warnings": list(self.warnings),
            "failures": list(self.failures),
            "metrics": dict(self.metrics),
        }


def evaluate_finalization_qc(
    *,
    dense: DensePoseResult,
    smooth: SmoothResult,
    final_table: pd.DataFrame,
    config: QcConfig | None = None,
) -> QcSummary:
    """Evaluate structural integrity and lightweight quality warnings."""

    cfg = config if config is not None else QcConfig()
    failures: list[str] = []
    warnings: list[str] = []
    metrics: dict[str, Any] = {
        "thresholds": {
            "missing_fraction_warn": cfg.missing_fraction_warn,
            "imputed_fraction_warn": cfg.imputed_fraction_warn,
            "jump_px_warn": cfg.jump_px_warn,
            "smooth_displacement_px_warn": cfg.smooth_displacement_px_warn,
        }
    }

    _check_schema_and_row_count(dense, final_table, failures, metrics)
    _check_timing_integrity(dense, failures, metrics)
    _check_protected_blank_preservation(dense, smooth, failures, metrics)
    _check_coordinate_validity(dense, smooth, failures, metrics)

    if failures:
        return QcSummary(
            structural_ok=False,
            warnings=tuple(warnings),
            failures=tuple(failures),
            metrics=metrics,
        )

    _check_missing_percentages(dense, smooth, cfg, warnings, metrics)
    _check_provenance_percentages(dense, cfg, warnings, metrics)
    _check_implausible_jumps(smooth, dense.protected, cfg, warnings, metrics)
    _check_smoothing_displacement(smooth, cfg, warnings, metrics)

    return QcSummary(
        structural_ok=True,
        warnings=tuple(warnings),
        failures=(),
        metrics=metrics,
    )


def _check_schema_and_row_count(
    dense: DensePoseResult,
    final_table: pd.DataFrame,
    failures: list[str],
    metrics: dict[str, Any],
) -> None:
    expected_rows = dense.n_samples * dense.n_tracks * dense.n_nodes
    metrics["row_count"] = int(len(final_table))
    metrics["expected_row_count"] = int(expected_rows)
    if list(final_table.columns) != list(FINAL_POSE_COLUMNS):
        failures.append("final_pose schema columns do not match the authoritative contract")
    if len(final_table) != expected_rows:
        failures.append(
            f"final_pose row count {len(final_table)} != expected {expected_rows}"
        )
    if "schema_version" in final_table.columns and len(final_table) > 0:
        versions = {str(v) for v in final_table["schema_version"].unique().tolist()}
        metrics["schema_versions"] = sorted(versions)
        if versions != {FINAL_POSE_SCHEMA_VERSION}:
            failures.append(
                "final_pose schema_version mismatch: " + ", ".join(sorted(versions))
            )


def _check_timing_integrity(
    dense: DensePoseResult,
    failures: list[str],
    metrics: dict[str, Any],
) -> None:
    timing = dense.timing_map
    n = dense.n_samples
    metrics["target_sample_count"] = int(n)
    metrics["source_frame_count"] = int(timing.source_frame_count)

    if timing.target_sample_count != n:
        failures.append("TimingMap target sample count does not match dense pose")
        return

    left = np.asarray(dense.left_source_frame_idx, dtype=np.int64)
    right = np.asarray(dense.right_source_frame_idx, dtype=np.int64)
    weight = np.asarray(dense.interpolation_weight, dtype=np.float64)
    exact = np.asarray(dense.is_exact_source_sample, dtype=bool)
    times = np.asarray(timing.target_time_sec, dtype=np.float64)

    if left.shape != (n,) or right.shape != (n,) or weight.shape != (n,) or exact.shape != (n,):
        failures.append("source-mapping arrays have inconsistent lengths")
        return
    if times.shape != (n,):
        failures.append("target_time_sec length does not match sample count")
        return
    if not np.all(np.isfinite(times)):
        failures.append("target_time_sec contains non-finite values")
    if n >= 2 and not np.all(np.diff(times) > 0):
        failures.append("target_time_sec is not strictly increasing")
    if np.any(left < 0) or np.any(right < 0):
        failures.append("source-frame mapping contains negative indices")
    if np.any(left >= timing.source_frame_count) or np.any(right >= timing.source_frame_count):
        failures.append("source-frame mapping indices exceed source frame count")
    if np.any(left > right):
        failures.append("left_source_frame_idx exceeds right_source_frame_idx")
    if not np.all((weight >= 0.0) & (weight <= 1.0) & np.isfinite(weight)):
        failures.append("interpolation_weight is outside [0, 1] or non-finite")
    if np.any(exact & ((left != right) | (weight != 0.0))):
        failures.append("exact source samples have inconsistent left/right/weight mapping")


def _check_protected_blank_preservation(
    dense: DensePoseResult,
    smooth: SmoothResult,
    failures: list[str],
    metrics: dict[str, Any],
) -> None:
    protected = dense.protected
    n_protected = int(np.count_nonzero(protected))
    metrics["protected_sample_count"] = n_protected
    if n_protected == 0:
        metrics["protected_blank_violations"] = 0
        return

    finite_xy = np.isfinite(smooth.x) & np.isfinite(smooth.y)
    violations = int(np.count_nonzero(protected & finite_xy))
    smoothed_protected = int(np.count_nonzero(protected & smooth.was_smoothed))
    bad_prov = int(
        np.count_nonzero(
            protected & (dense.provenance != PROVENANCE_PROTECTED_MANUAL_BLANK)
        )
    )
    metrics["protected_blank_violations"] = violations
    metrics["protected_smoothed_count"] = smoothed_protected
    metrics["protected_provenance_mismatches"] = bad_prov
    if violations:
        failures.append(
            f"protected manual blanks have finite coordinates ({violations} samples)"
        )
    if smoothed_protected:
        failures.append(
            f"protected manual blanks were marked smoothed ({smoothed_protected} samples)"
        )
    if bad_prov:
        failures.append(
            "protected samples lack protected_manual_blank provenance "
            f"({bad_prov} samples)"
        )


def _check_coordinate_validity(
    dense: DensePoseResult,
    smooth: SmoothResult,
    failures: list[str],
    metrics: dict[str, Any],
) -> None:
    finite = np.isfinite(smooth.x) & np.isfinite(smooth.y)
    one_finite = np.isfinite(smooth.x) ^ np.isfinite(smooth.y)
    metrics["finite_coordinate_count"] = int(np.count_nonzero(finite))
    metrics["partial_coordinate_count"] = int(np.count_nonzero(one_finite))
    if np.any(one_finite):
        failures.append("some samples have only one of x/y finite")

    # Imputed/resampled provenance must carry finite coordinates.
    imputed = np.isin(
        dense.provenance,
        (
            PROVENANCE_LINEAR_GAP_FILL,
            PROVENANCE_KALMAN_GAP_FILL,
            PROVENANCE_TEMPORAL_RESAMPLE,
        ),
    )
    imputed_nan = int(np.count_nonzero(imputed & ~finite))
    metrics["imputed_with_nan_coordinates"] = imputed_nan
    if imputed_nan:
        failures.append(
            f"imputed/resampled provenance samples lack finite coordinates ({imputed_nan})"
        )


def _check_missing_percentages(
    dense: DensePoseResult,
    smooth: SmoothResult,
    config: QcConfig,
    warnings: list[str],
    metrics: dict[str, Any],
) -> None:
    finite = np.isfinite(smooth.x) & np.isfinite(smooth.y)
    total = int(finite.size)
    missing = int(total - np.count_nonzero(finite))
    overall = float(missing / total) if total else 0.0
    metrics["missing_fraction_overall"] = overall
    per_series: list[dict[str, Any]] = []
    for track_i, track_id in enumerate(dense.track_ids.tolist()):
        for node_i, node_name in enumerate(dense.node_names.tolist()):
            series = finite[:, track_i, node_i]
            frac = float(1.0 - (np.count_nonzero(series) / series.size))
            entry = {
                "track": int(track_id),
                "node": str(node_name),
                "missing_fraction": frac,
            }
            per_series.append(entry)
            if frac >= config.missing_fraction_warn:
                warnings.append(
                    f"high missing fraction for track={track_id} node={node_name}: "
                    f"{frac:.3f} >= {config.missing_fraction_warn:.3f}"
                )
    metrics["missing_fraction_by_track_node"] = per_series
    if overall >= config.missing_fraction_warn:
        warnings.append(
            f"overall missing coordinate fraction {overall:.3f} >= "
            f"{config.missing_fraction_warn:.3f}"
        )


def _check_provenance_percentages(
    dense: DensePoseResult,
    config: QcConfig,
    warnings: list[str],
    metrics: dict[str, Any],
) -> None:
    total = int(dense.provenance.size)
    counts: dict[str, int] = {}
    for code, name in (
        (PROVENANCE_MISSING, "missing"),
        (PROVENANCE_LINEAR_GAP_FILL, "linear_gap_fill"),
        (PROVENANCE_KALMAN_GAP_FILL, "kalman_gap_fill"),
        (PROVENANCE_TEMPORAL_RESAMPLE, "temporal_resample"),
        (PROVENANCE_PROTECTED_MANUAL_BLANK, "protected_manual_blank"),
    ):
        counts[name] = int(np.count_nonzero(dense.provenance == code))
    metrics["provenance_counts"] = counts
    imputed = counts["linear_gap_fill"] + counts["kalman_gap_fill"]
    imputed_frac = float(imputed / total) if total else 0.0
    metrics["imputed_fraction"] = imputed_frac
    if imputed_frac >= config.imputed_fraction_warn:
        warnings.append(
            f"imputed coordinate fraction {imputed_frac:.3f} >= "
            f"{config.imputed_fraction_warn:.3f}"
        )


def _check_implausible_jumps(
    smooth: SmoothResult,
    protected: np.ndarray,
    config: QcConfig,
    warnings: list[str],
    metrics: dict[str, Any],
) -> None:
    jump_count = 0
    max_jump = 0.0
    for track_i in range(smooth.n_tracks):
        for node_i in range(smooth.n_nodes):
            x = smooth.x[:, track_i, node_i]
            y = smooth.y[:, track_i, node_i]
            prot = protected[:, track_i, node_i]
            for i in range(1, smooth.n_samples):
                if prot[i] or prot[i - 1]:
                    continue
                if not (
                    np.isfinite(x[i])
                    and np.isfinite(y[i])
                    and np.isfinite(x[i - 1])
                    and np.isfinite(y[i - 1])
                ):
                    continue
                dist = float(np.hypot(x[i] - x[i - 1], y[i] - y[i - 1]))
                if dist > max_jump:
                    max_jump = dist
                if dist >= config.jump_px_warn:
                    jump_count += 1
    metrics["implausible_jump_count"] = jump_count
    metrics["max_consecutive_displacement_px"] = max_jump
    if jump_count:
        warnings.append(
            f"implausible consecutive displacements detected ({jump_count} steps "
            f">= {config.jump_px_warn:.1f} px; max={max_jump:.1f} px)"
        )


def _check_smoothing_displacement(
    smooth: SmoothResult,
    config: QcConfig,
    warnings: list[str],
    metrics: dict[str, Any],
) -> None:
    mask = smooth.was_smoothed
    if not np.any(mask):
        metrics["smoothed_sample_count"] = 0
        metrics["mean_smooth_displacement_px"] = 0.0
        metrics["max_smooth_displacement_px"] = 0.0
        return
    dx = smooth.x[mask] - smooth.x_pre_smooth[mask]
    dy = smooth.y[mask] - smooth.y_pre_smooth[mask]
    disp = np.hypot(dx, dy)
    mean_disp = float(np.mean(disp))
    max_disp = float(np.max(disp))
    metrics["smoothed_sample_count"] = int(np.count_nonzero(mask))
    metrics["mean_smooth_displacement_px"] = mean_disp
    metrics["max_smooth_displacement_px"] = max_disp
    if max_disp >= config.smooth_displacement_px_warn:
        warnings.append(
            f"smoothing displacement max {max_disp:.3f} px >= "
            f"{config.smooth_displacement_px_warn:.3f} px "
            f"(mean {mean_disp:.3f} px over {int(np.count_nonzero(mask))} samples)"
        )

