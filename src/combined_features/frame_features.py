"""Frame-level dyadic pose features (vendored from legacy ``frame_features``).

Scientific formulas are preserved from the lab Pipeline 50 producer. Input is an
``AdaptedPose`` from Subsystem 06 Sprint 1 (not Pipeline 30 NPZ files).

Features are computed at the dyad-frame level and include:
- Kinematics (velocities, accelerations)
- Orientation (heading alignment, relative angles)
- Contact geometry (pairwise keypoint distances)
- Motion projections (radial velocities)
- Quality (SLEAP confidence, imputation flags)

All features are body-length normalized for size invariance.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from combined_features.contracts import (
    IMPLANTATION_RESOLVED,
    IMPUTE_STAGE_KALMAN,
    IMPUTE_STAGE_PCHIP,
)
from combined_features.pose_adapter import AdaptedPose, ImplantationResolution

# ============================================================================
# Constants
# ============================================================================

EPS = 1e-6
PEARSON_CORR_ABS_LIMIT = 1.0 + 1e-6

# Impute-stage codes are imported from combined_features.contracts
# (legacy Pipeline 30 encoding: 0=none, 1=linear, 2=PCHIP, 3=Kalman).


def sanitize_pearson_correlation(
    values: pd.Series,
    *,
    abs_limit: float = PEARSON_CORR_ABS_LIMIT,
) -> pd.Series:
    """Keep only mathematically valid Pearson values in ``[-1, 1]``.

    Undefined / non-finite / wildly out-of-range rolling correlations become
    NaN. Tiny float overshoot within ``abs_limit`` is clipped into ``[-1, 1]``.
    Neighbor fill is intentionally not applied: that would invent synchrony.
    """

    arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64, copy=True)
    valid = np.isfinite(arr) & (np.abs(arr) <= float(abs_limit))
    cleaned = np.where(valid, np.clip(arr, -1.0, 1.0), np.nan)
    return pd.Series(cleaned, index=values.index, dtype=np.float64)

# ============================================================================
# Dataclasses
# ============================================================================


@dataclass
class FrameFeatureSummary:
    """Summary statistics for frame features."""

    n_rows: int
    n_dyads: int
    n_dense_frames_unique: int
    missingness: dict[str, float]
    feature_quantiles: dict[str, dict[str, float]]


@dataclass
class FrameFeatureResult:
    """Result container for frame features."""

    features_df: pd.DataFrame
    summary: FrameFeatureSummary


# ============================================================================
# Helper Functions
# ============================================================================


def _normalize_track_id(val: Any) -> Any:
    """
    Normalize track ID value.

    Converts bytes to string, then to int if possible.

    Args:
        val: Track ID value.

    Returns:
        Normalized track ID.
    """
    if isinstance(val, bytes):
        val = val.decode("utf-8")

    # Try to coerce to int
    try:
        return int(val)
    except (ValueError, TypeError):
        return val


def _rolling_median_np(arr: np.ndarray, win: int) -> np.ndarray:
    """
    Apply rolling median with centered window.

    Args:
        arr: 1D array.
        win: Window size (use odd number for symmetry).

    Returns:
        Smoothed array with same length.
    """
    if win <= 1:
        return arr.copy()

    result = np.full_like(arr, np.nan, dtype=float)
    half = win // 2

    for i in range(len(arr)):
        start = max(0, i - half)
        end = min(len(arr), i + half + 1)
        window = arr[start:end]
        valid = window[np.isfinite(window)]
        if len(valid) > 0:
            result[i] = np.median(valid)

    return result


def _finite_diff(arr: np.ndarray, t: np.ndarray, step: int) -> np.ndarray:
    """
    Compute finite difference derivative.

    Args:
        arr: Array to differentiate.
        t: Time array.
        step: Step size for finite difference.

    Returns:
        Derivative array with NaN where not computable.
    """
    if step < 1:
        step = 1

    deriv = np.full_like(arr, np.nan, dtype=float)

    for i in range(step, len(arr)):
        dt = t[i] - t[i - step]
        if dt > 0 and np.isfinite(arr[i]) and np.isfinite(arr[i - step]):
            deriv[i] = (arr[i] - arr[i - step]) / dt

    return deriv


def _unit_vec(dx: np.ndarray, dy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute unit vector from components.

    Args:
        dx: X component.
        dy: Y component.

    Returns:
        Tuple of (ux, uy) unit vectors, with NaN where norm is zero.
    """
    norm = np.sqrt(dx**2 + dy**2)
    valid = (norm > EPS) & np.isfinite(norm)

    # Use masked divide to avoid eager evaluation warnings from np.where(dx / norm, ...).
    ux = np.full_like(norm, np.nan, dtype=float)
    uy = np.full_like(norm, np.nan, dtype=float)
    np.divide(dx, norm, out=ux, where=valid)
    np.divide(dy, norm, out=uy, where=valid)
    return ux, uy


def _safe_dot(ax, ay, bx, by) -> np.ndarray:
    """
    Compute safe dot product with NaN propagation.

    Args:
        ax, ay: First vector components.
        bx, by: Second vector components.

    Returns:
        Dot product array.
    """
    return ax * bx + ay * by


def _pairwise_distance(
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    scale: np.ndarray,
) -> np.ndarray:
    """
    Compute normalized Euclidean distance.

    Args:
        ax, ay: First point coordinates.
        bx, by: Second point coordinates.
        scale: Normalization scale (body length).

    Returns:
        Normalized distance array.
    """
    dx = ax - bx
    dy = ay - by
    dist = np.hypot(dx, dy)
    scale_safe = np.maximum(scale, EPS)
    return dist / scale_safe


def dist_point_to_segment_vectorized(px, py, ax, ay, bx, by):
    """Calculates min distance from point P to segment AB (vectorized)."""
    dx, dy = bx - ax, by - ay
    l2 = dx**2 + dy**2
    l2 = np.where(l2 == 0, 1e-9, l2)

    t = ((px - ax) * dx + (py - ay) * dy) / l2
    t = np.clip(t, 0, 1)

    closest_x = ax + t * dx
    closest_y = ay + t * dy
    return np.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)


def _resolve_side_coord_cols(df: pd.DataFrame, prefix: str, node: str) -> tuple[str, str]:
    """Resolve x/y coordinate column names for both prefix and suffix naming styles."""
    x_candidates = [f"{prefix}_{node}_x", f"{node}_x_{prefix}"]
    y_candidates = [f"{prefix}_{node}_y", f"{node}_y_{prefix}"]

    x_col = next((c for c in x_candidates if c in df.columns), None)
    y_col = next((c for c in y_candidates if c in df.columns), None)

    if x_col is None or y_col is None:
        raise ValueError(
            "Missing columns for self-healing side-distance feature with "
            f"prefix='{prefix}', node='{node}'. Tried: x={x_candidates}, y={y_candidates}"
        )

    return x_col, y_col


def calculate_self_healing_side_distance(
    df,
    prefix_i,
    prefix_j,
    origin_node="nose",
    rear_exclusion_ratio=0.33,
    anterior_ratio=0.9,
):
    """
    Calculates distance to a self-healing side polyline.
    - Interpolates Virtual Neck if neck is missing.
    - Extrapolates Virtual Neck forward if entire anterior half is missing.
    - Truncates Virtual Hip to exclude posterior 1/3.
    """
    origin_i_x_col, origin_i_y_col = _resolve_side_coord_cols(df, prefix_i, origin_node)
    nose_j_x_col, nose_j_y_col = _resolve_side_coord_cols(df, prefix_j, "nose")
    neck_j_x_col, neck_j_y_col = _resolve_side_coord_cols(df, prefix_j, "neck")
    spine_j_x_col, spine_j_y_col = _resolve_side_coord_cols(df, prefix_j, "spine")
    tail_j_x_col, tail_j_y_col = _resolve_side_coord_cols(df, prefix_j, "tail_base")

    px = pd.to_numeric(df[origin_i_x_col], errors="coerce").to_numpy(dtype=float, copy=False)
    py = pd.to_numeric(df[origin_i_y_col], errors="coerce").to_numpy(dtype=float, copy=False)

    nj_x = pd.to_numeric(df[nose_j_x_col], errors="coerce").to_numpy(dtype=float, copy=False)
    nj_y = pd.to_numeric(df[nose_j_y_col], errors="coerce").to_numpy(dtype=float, copy=False)
    nx = pd.to_numeric(df[neck_j_x_col], errors="coerce").to_numpy(dtype=float, copy=False)
    ny = pd.to_numeric(df[neck_j_y_col], errors="coerce").to_numpy(dtype=float, copy=False)
    sx = pd.to_numeric(df[spine_j_x_col], errors="coerce").to_numpy(dtype=float, copy=False)
    sy = pd.to_numeric(df[spine_j_y_col], errors="coerce").to_numpy(dtype=float, copy=False)
    tx = pd.to_numeric(df[tail_j_x_col], errors="coerce").to_numpy(dtype=float, copy=False)
    ty = pd.to_numeric(df[tail_j_y_col], errors="coerce").to_numpy(dtype=float, copy=False)

    vec_st_x = sx - tx
    vec_st_y = sy - ty

    # Anterior anchor (virtual neck).
    vn_x = np.where(
        ~np.isnan(nx),
        nx,
        np.where(
            ~np.isnan(nj_x) & ~np.isnan(sx),
            (nj_x + sx) / 2.0,
            np.where(~np.isnan(sx) & ~np.isnan(tx), sx + vec_st_x * anterior_ratio, sx),
        ),
    )

    vn_y = np.where(
        ~np.isnan(ny),
        ny,
        np.where(
            ~np.isnan(nj_y) & ~np.isnan(sy),
            (nj_y + sy) / 2.0,
            np.where(~np.isnan(sy) & ~np.isnan(ty), sy + vec_st_y * anterior_ratio, sy),
        ),
    )

    # Posterior anchor (virtual hip).
    vh_target_x = np.where(~np.isnan(sx), sx, vn_x)
    vh_target_y = np.where(~np.isnan(sy), sy, vn_y)

    vh_x = np.where(
        ~np.isnan(tx),
        tx + rear_exclusion_ratio * (vh_target_x - tx),
        vh_target_x,
    )
    vh_y = np.where(
        ~np.isnan(ty),
        ty + rear_exclusion_ratio * (vh_target_y - ty),
        vh_target_y,
    )

    # Core node (mid-spine).
    core_x = np.where(~np.isnan(sx), sx, (vn_x + vh_x) / 2.0)
    core_y = np.where(~np.isnan(sy), sy, (vn_y + vh_y) / 2.0)

    d1 = dist_point_to_segment_vectorized(px, py, vn_x, vn_y, core_x, core_y)
    d2 = dist_point_to_segment_vectorized(px, py, core_x, core_y, vh_x, vh_y)

    return np.minimum(d1, d2)


def _scalar_distance(ax: float, ay: float, bx: float, by: float) -> float:
    """Scalar Euclidean distance with NaN propagation."""
    vals = np.array([ax, ay, bx, by], dtype=float)
    if not np.isfinite(vals).all():
        return float(np.nan)
    return float(np.sqrt((ax - bx) ** 2 + (ay - by) ** 2))


def _safe_normalize_scalar(value: float, denom: float) -> float:
    """Normalize scalar by denominator, returning NaN for invalid denominator/value."""
    if not np.isfinite(value) or not np.isfinite(denom) or denom <= 0:
        return float(np.nan)
    return float(value / denom)


def _resolve_body_ref(track: dict[str, Any]) -> tuple[float, str]:
    """
    Resolve a stable per-track body reference for cross-mouse normalization.

    Fallback order:
      1) nose_tail_median
      2) chain_length_median
      3) bl_median (configured body-length proxy median)
    """
    candidates = [
        ("nose_tail_median", track.get("nose_tail_median", np.nan)),
        ("chain_length_median", track.get("chain_length_median", np.nan)),
        ("bl_median", track.get("bl_median", np.nan)),
    ]
    for source, value in candidates:
        value_f = float(value)
        if np.isfinite(value_f) and value_f > 0:
            return value_f, source
    return float(np.nan), "none"


def _symmetric_dyad_body_scale(
    focal: dict[str, Any],
    partner: dict[str, Any],
) -> tuple[float, float, float, str, str]:
    """
    Compute symmetric dyad body scale for cross-mouse distance normalization.

    Preferred default: 0.5 * (body_ref_i + body_ref_j)
    Fallback:
      - use the valid side when only one body_ref is valid
      - return NaN when neither side is valid
    """
    body_ref_i, src_i = _resolve_body_ref(focal)
    body_ref_j, src_j = _resolve_body_ref(partner)

    i_ok = np.isfinite(body_ref_i) and body_ref_i > 0
    j_ok = np.isfinite(body_ref_j) and body_ref_j > 0

    if i_ok and j_ok:
        dyad_scale = 0.5 * (body_ref_i + body_ref_j)
    elif i_ok:
        dyad_scale = body_ref_i
    elif j_ok:
        dyad_scale = body_ref_j
    else:
        dyad_scale = float(np.nan)

    if not np.isfinite(dyad_scale) or dyad_scale <= 0:
        dyad_scale = float(np.nan)

    return float(dyad_scale), float(body_ref_i), float(body_ref_j), src_i, src_j


def _scalar_cosine(ax: float, ay: float, bx: float, by: float) -> float:
    """Cosine similarity between two scalar 2D vectors with NaN safety."""
    vals = np.array([ax, ay, bx, by], dtype=float)
    if not np.isfinite(vals).all():
        return float(np.nan)

    norm_a = float(np.sqrt(ax**2 + ay**2))
    norm_b = float(np.sqrt(bx**2 + by**2))
    if norm_a <= EPS or norm_b <= EPS:
        return float(np.nan)

    return float((ax * bx + ay * by) / (norm_a * norm_b))


def _nanmax2(a: float, b: float) -> float:
    """NaN-safe max for two scalars without RuntimeWarning on all-NaN input."""
    a_finite = np.isfinite(a)
    b_finite = np.isfinite(b)
    if a_finite and b_finite:
        return float(max(a, b))
    if a_finite:
        return float(a)
    if b_finite:
        return float(b)
    return float(np.nan)


def _nanmin2(a: float, b: float) -> float:
    """NaN-safe min for two scalars without RuntimeWarning on all-NaN input."""
    a_finite = np.isfinite(a)
    b_finite = np.isfinite(b)
    if a_finite and b_finite:
        return float(min(a, b))
    if a_finite:
        return float(a)
    if b_finite:
        return float(b)
    return float(np.nan)


def _rowwise_nanmin(df: pd.DataFrame, cols: Sequence[str]) -> pd.Series:
    """Row-wise NaN-safe minimum across columns, preserving float dtype."""
    values = df.loc[:, list(cols)].to_numpy(dtype=float, copy=False)
    any_finite = np.isfinite(values).any(axis=1)
    out = np.full(values.shape[0], np.nan, dtype=float)
    if len(out) == 0:
        return pd.Series(out, index=df.index, dtype=float)
    out[any_finite] = np.nanmin(values[any_finite], axis=1)
    return pd.Series(out, index=df.index, dtype=float)


def _rowwise_mean_of_k_smallest(
    df: pd.DataFrame, cols: Sequence[str], k: int
) -> pd.Series:
    """Mean of the k smallest finite row-wise values, or available finite values."""
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    values = df.loc[:, list(cols)].to_numpy(dtype=float, copy=False)
    out = np.full(values.shape[0], np.nan, dtype=float)

    if len(out) == 0:
        return pd.Series(out, index=df.index, dtype=float)

    finite_mask = np.isfinite(values)
    finite_counts = finite_mask.sum(axis=1)
    has_any = finite_counts > 0
    if not np.any(has_any):
        return pd.Series(out, index=df.index, dtype=float)

    safe_values = np.where(finite_mask, values, np.inf)
    partition_k = min(max(k - 1, 0), safe_values.shape[1] - 1)
    smallest = np.partition(safe_values, partition_k, axis=1)[:, : min(k, safe_values.shape[1])]
    smallest = np.where(np.isfinite(smallest), smallest, np.nan)

    with np.errstate(invalid="ignore"):
        out[has_any] = np.nanmean(smallest[has_any], axis=1)

    return pd.Series(out, index=df.index, dtype=float)


def _primitive_distance_feature_cols() -> list[str]:
    """Primitive cross-mouse CONTACT distance columns used for robustness stats."""
    return [
        "distance_nose_i_nose_j_norm",
        "distance_nose_i_neck_j_norm",
        "distance_nose_i_spine_j_norm",
        "distance_nose_i_tail_j_norm",
        "distance_neck_i_nose_j_norm",
        "distance_neck_i_neck_j_norm",
        "distance_neck_i_spine_j_norm",
        "distance_neck_i_tail_j_norm",
        "distance_spine_i_nose_j_norm",
        "distance_spine_i_neck_j_norm",
        "distance_spine_i_spine_j_norm",
        "distance_spine_i_tail_j_norm",
        "distance_tail_i_nose_j_norm",
        "distance_tail_i_neck_j_norm",
        "distance_tail_i_spine_j_norm",
        "distance_tail_i_tail_j_norm",
    ]


def _add_primitive_distance_missingness_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Add primitive-distance missingness summary features."""
    primitive_cols = _primitive_distance_feature_cols()
    missing_cols = [col for col in primitive_cols if col not in features_df.columns]
    if missing_cols:
        raise ValueError(
            "Missing primitive cross-mouse distance columns required for CONTACT robustness features: "
            f"{missing_cols}"
        )

    values = features_df.loc[:, primitive_cols].to_numpy(dtype=float, copy=False)
    finite_mask = np.isfinite(values)
    n_finite = finite_mask.sum(axis=1).astype(float)

    out = features_df.copy()
    out["n_primitive_distance_finite"] = n_finite
    out["frac_primitive_distance_missing"] = 1.0 - (n_finite / float(len(primitive_cols)))
    out["all_primitive_distances_missing"] = n_finite == 0
    return out


def _append_contact_robustness_features(
    rec: dict[str, Any],
    *,
    focal: dict[str, Any],
    partner: dict[str, Any],
    dense_idx: int,
) -> None:
    """Append morphology and directionality CONTACT robustness features for one row."""
    nose_i_x = focal["nose_x"][dense_idx]
    nose_i_y = focal["nose_y"][dense_idx]
    neck_i_x = focal["neck_x"][dense_idx]
    neck_i_y = focal["neck_y"][dense_idx]
    spine_i_x = focal["spine_x"][dense_idx]
    spine_i_y = focal["spine_y"][dense_idx]
    tail_i_x = focal["tail_x"][dense_idx]
    tail_i_y = focal["tail_y"][dense_idx]

    nose_j_x = partner["nose_x"][dense_idx]
    nose_j_y = partner["nose_y"][dense_idx]
    neck_j_x = partner["neck_x"][dense_idx]
    neck_j_y = partner["neck_y"][dense_idx]
    spine_j_x = partner["spine_x"][dense_idx]
    spine_j_y = partner["spine_y"][dense_idx]
    tail_j_x = partner["tail_x"][dense_idx]
    tail_j_y = partner["tail_y"][dense_idx]

    body_length_i = _scalar_distance(nose_i_x, nose_i_y, tail_i_x, tail_i_y)
    body_length_j = _scalar_distance(nose_j_x, nose_j_y, tail_j_x, tail_j_y)

    nose_neck_i = _scalar_distance(nose_i_x, nose_i_y, neck_i_x, neck_i_y)
    neck_spine_i = _scalar_distance(neck_i_x, neck_i_y, spine_i_x, spine_i_y)
    spine_tail_i = _scalar_distance(spine_i_x, spine_i_y, tail_i_x, tail_i_y)
    chain_length_i = float(np.nansum(np.array([nose_neck_i, neck_spine_i, spine_tail_i], dtype=float)))
    if not np.isfinite(np.array([nose_neck_i, neck_spine_i, spine_tail_i], dtype=float)).all():
        chain_length_i = float(np.nan)

    nose_neck_j = _scalar_distance(nose_j_x, nose_j_y, neck_j_x, neck_j_y)
    neck_spine_j = _scalar_distance(neck_j_x, neck_j_y, spine_j_x, spine_j_y)
    spine_tail_j = _scalar_distance(spine_j_x, spine_j_y, tail_j_x, tail_j_y)
    chain_length_j = float(np.nansum(np.array([nose_neck_j, neck_spine_j, spine_tail_j], dtype=float)))
    if not np.isfinite(np.array([nose_neck_j, neck_spine_j, spine_tail_j], dtype=float)).all():
        chain_length_j = float(np.nan)

    rec["body_length_i_norm"] = _safe_normalize_scalar(body_length_i, focal["nose_tail_median"])
    rec["body_length_j_norm"] = _safe_normalize_scalar(body_length_j, partner["nose_tail_median"])
    rec["chain_length_i_norm"] = _safe_normalize_scalar(chain_length_i, focal["chain_length_median"])
    rec["chain_length_j_norm"] = _safe_normalize_scalar(chain_length_j, partner["chain_length_median"])

    rec["compactness_i"] = (
        float(chain_length_i / (body_length_i + EPS))
        if np.isfinite(chain_length_i) and np.isfinite(body_length_i)
        else float(np.nan)
    )
    rec["compactness_j"] = (
        float(chain_length_j / (body_length_j + EPS))
        if np.isfinite(chain_length_j) and np.isfinite(body_length_j)
        else float(np.nan)
    )

    rec["neck_spine_norm_i"] = _safe_normalize_scalar(neck_spine_i, focal["neck_spine_median"])
    rec["neck_spine_norm_j"] = _safe_normalize_scalar(neck_spine_j, partner["neck_spine_median"])
    rec["spine_tail_norm_i"] = _safe_normalize_scalar(spine_tail_i, focal["spine_tail_median"])
    rec["spine_tail_norm_j"] = _safe_normalize_scalar(spine_tail_j, partner["spine_tail_median"])

    rec["curvature_i"] = _scalar_cosine(
        spine_i_x - neck_i_x,
        spine_i_y - neck_i_y,
        tail_i_x - spine_i_x,
        tail_i_y - spine_i_y,
    )
    rec["curvature_j"] = _scalar_cosine(
        spine_j_x - neck_j_x,
        spine_j_y - neck_j_y,
        tail_j_x - spine_j_x,
        tail_j_y - spine_j_y,
    )

    rec["heading_to_partner_i"] = _scalar_cosine(
        neck_i_x - spine_i_x,
        neck_i_y - spine_i_y,
        spine_j_x - spine_i_x,
        spine_j_y - spine_i_y,
    )
    rec["heading_to_partner_j"] = _scalar_cosine(
        neck_j_x - spine_j_x,
        neck_j_y - spine_j_y,
        spine_i_x - spine_j_x,
        spine_i_y - spine_j_y,
    )

    mutual_vals = np.array(
        [rec["heading_to_partner_i"], rec["heading_to_partner_j"]],
        dtype=float,
    )
    rec["mutual_facing"] = (
        float(np.nanmean(mutual_vals)) if np.isfinite(mutual_vals).any() else float(np.nan)
    )


def _print_contact_robustness_feature_summary(features_df: pd.DataFrame) -> None:
    """Print concise summary for new CONTACT robustness features."""
    print("Added CONTACT robustness features: missingness=3, morphology=12, directionality=3")
    summary_cols = [
        "n_primitive_distance_finite",
        "frac_primitive_distance_missing",
        "all_primitive_distances_missing",
        "body_length_i_norm",
        "compactness_i",
        "mutual_facing",
    ]
    for col in summary_cols:
        if col in features_df.columns:
            miss = float(features_df[col].isna().mean()) if len(features_df) > 0 else float("nan")
            print(f"  {col} missingness: {miss:.4f}")


def _headstage_feature_cols() -> list[str]:
    """Headstage distance feature columns for CONTACT detector."""
    return [
        "distance_headstage_i_nose_j_norm",
        "distance_headstage_i_neck_j_norm",
        "distance_headstage_i_spine_j_norm",
        "distance_headstage_i_tail_j_norm",
        "distance_headstage_i_nose_i_norm",
        "distance_headstage_i_neck_i_norm",
        "distance_headstage_i_spine_i_norm",
        "distance_headstage_i_tail_i_norm",
    ]


def _headstage_direction_feature_cols() -> list[str]:
    """Headstage direction/alignment feature columns for CONTACT detector."""
    return [
        "headstage_alignment_i",
        "headstage_neck_alignment_i",
        "headstage_offset_from_body_axis_i_norm",
        "headstage_forward_projection_i_norm",
    ]


def _all_headstage_feature_cols() -> list[str]:
    """All headstage-derived feature columns for CONTACT detector."""
    return _headstage_feature_cols() + _headstage_direction_feature_cols()


def _append_headstage_direction_features(
    rec: dict[str, Any],
    *,
    implanted: dict[str, Any],
    dense_idx: int,
) -> None:
    """Append implanted-headstage direction/alignment features for one row."""
    spine_i_x = implanted["spine_x"][dense_idx]
    spine_i_y = implanted["spine_y"][dense_idx]
    neck_i_x = implanted["neck_x"][dense_idx]
    neck_i_y = implanted["neck_y"][dense_idx]
    headstage_i_x = implanted["headstage_x"][dense_idx]
    headstage_i_y = implanted["headstage_y"][dense_idx]

    axis_dx = neck_i_x - spine_i_x
    axis_dy = neck_i_y - spine_i_y

    rec["headstage_alignment_i"] = _scalar_cosine(
        axis_dx,
        axis_dy,
        headstage_i_x - spine_i_x,
        headstage_i_y - spine_i_y,
    )
    rec["headstage_neck_alignment_i"] = _scalar_cosine(
        axis_dx,
        axis_dy,
        headstage_i_x - neck_i_x,
        headstage_i_y - neck_i_y,
    )

    vals = np.array(
        [spine_i_x, spine_i_y, neck_i_x, neck_i_y, headstage_i_x, headstage_i_y],
        dtype=float,
    )
    if not np.isfinite(vals).all():
        for col in [
            "headstage_offset_from_body_axis_i_norm",
            "headstage_forward_projection_i_norm",
        ]:
            rec[col] = float(np.nan)
        return

    axis_norm = float(np.sqrt(axis_dx**2 + axis_dy**2))
    axis_scale = float(implanted.get("neck_spine_median", np.nan))
    if axis_norm <= EPS or not np.isfinite(axis_scale) or axis_scale <= 0:
        for col in [
            "headstage_offset_from_body_axis_i_norm",
            "headstage_forward_projection_i_norm",
        ]:
            rec[col] = float(np.nan)
        return

    hs_from_spine_dx = headstage_i_x - spine_i_x
    hs_from_spine_dy = headstage_i_y - spine_i_y
    cross_mag = abs(axis_dx * hs_from_spine_dy - axis_dy * hs_from_spine_dx)
    perp_dist = cross_mag / axis_norm
    rec["headstage_offset_from_body_axis_i_norm"] = _safe_normalize_scalar(
        float(perp_dist),
        axis_scale,
    )

    unit_axis_x = axis_dx / axis_norm
    unit_axis_y = axis_dy / axis_norm
    hs_from_neck_dx = headstage_i_x - neck_i_x
    hs_from_neck_dy = headstage_i_y - neck_i_y
    forward_projection = hs_from_neck_dx * unit_axis_x + hs_from_neck_dy * unit_axis_y
    rec["headstage_forward_projection_i_norm"] = _safe_normalize_scalar(
        float(forward_projection),
        axis_scale,
    )


def _print_headstage_feature_summary(features_df: pd.DataFrame) -> None:
    """Print concise summary for newly added headstage features."""
    dist_cols = _headstage_feature_cols()
    direction_cols = _headstage_direction_feature_cols()
    print("Added CONTACT headstage-direction features: 4")
    print("Added CONTACT headstage distance features: 8")
    for col in direction_cols:
        if col in features_df.columns:
            miss = float(features_df[col].isna().mean()) if len(features_df) > 0 else float("nan")
            print(f"  {col} missingness: {miss:.4f}")
    for col in dist_cols[:2]:
        if col in features_df.columns:
            miss = float(features_df[col].isna().mean()) if len(features_df) > 0 else float("nan")
            print(f"  {col} missingness: {miss:.4f}")


def _add_body_only_fallback_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """Add compact body-only fallback CONTACT geometry features."""
    body_only_cols = [
        "distance_neck_i_neck_j_norm",
        "distance_neck_i_spine_j_norm",
        "distance_neck_i_tail_j_norm",
        "distance_spine_i_neck_j_norm",
        "distance_spine_i_spine_j_norm",
        "distance_spine_i_tail_j_norm",
        "distance_tail_i_neck_j_norm",
        "distance_tail_i_spine_j_norm",
        "distance_tail_i_tail_j_norm",
    ]
    body_axis_cols = [
        "distance_neck_i_neck_j_norm",
        "distance_neck_i_spine_j_norm",
        "distance_spine_i_neck_j_norm",
        "distance_spine_i_spine_j_norm",
    ]
    tail_rear_cols = [
        "distance_tail_i_neck_j_norm",
        "distance_tail_i_spine_j_norm",
        "distance_tail_i_tail_j_norm",
        "distance_neck_i_tail_j_norm",
        "distance_spine_i_tail_j_norm",
    ]

    required_cols = sorted(set(body_only_cols + body_axis_cols + tail_rear_cols))
    missing_cols = [col for col in required_cols if col not in features_df.columns]
    if missing_cols:
        raise ValueError(
            "Missing required primitive body-distance columns for CONTACT fallback features: "
            f"{missing_cols}"
        )

    features_df = features_df.copy()
    features_df["body_only_contact_best_norm"] = _rowwise_nanmin(features_df, body_only_cols)
    features_df["body_only_contact_mean_best3_norm"] = _rowwise_mean_of_k_smallest(
        features_df, body_only_cols, k=3
    )
    features_df["body_axis_contact_best_norm"] = _rowwise_nanmin(features_df, body_axis_cols)
    features_df["tail_rear_contact_best_norm"] = _rowwise_nanmin(features_df, tail_rear_cols)

    print("Added fallback body-only features: 4")
    for col in [
        "body_only_contact_best_norm",
        "body_only_contact_mean_best3_norm",
        "body_axis_contact_best_norm",
        "tail_rear_contact_best_norm",
    ]:
        missingness = float(features_df[col].isna().mean()) if len(features_df) > 0 else float("nan")
        print(f"  {col} missingness: {missingness:.4f}")

    return features_df


def _add_side_by_side_geometry_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add side-by-side geometry features.

    These features describe whether the partner is positioned laterally beside the
    focal mouse rather than in front/behind, and whether corresponding body-axis
    landmarks are closer than crossed body-axis landmarks.

    Feature groups:
      - forward_offset_*: signed normalized projection of partner position onto body axis
      - lateral_offset_*: signed normalized projection of partner position onto lateral axis
      - lateral_dominance_*: fraction of partner position explained by lateral rather than forward offset
      - body-axis distance symmetry: matched body-part distances vs crossed body-part distances
    """
    required_cols = [
        "neck_x_i", "neck_y_i", "spine_x_i", "spine_y_i",
        "neck_x_j", "neck_y_j", "spine_x_j", "spine_y_j",
        "dyad_body_scale",
        "distance_neck_i_neck_j_norm",
        "distance_spine_i_spine_j_norm",
        "distance_tail_i_tail_j_norm",
        "distance_neck_i_tail_j_norm",
        "distance_tail_i_neck_j_norm",
    ]

    missing_cols = [col for col in required_cols if col not in features_df.columns]
    if missing_cols:
        raise ValueError(
            "Missing required columns for side-by-side geometry features: "
            f"{missing_cols}"
        )

    out = features_df.copy()

    scale = pd.to_numeric(out["dyad_body_scale"], errors="coerce").to_numpy(dtype=float, copy=False)
    valid_scale = np.isfinite(scale) & (scale > EPS)
    scale_safe = np.where(valid_scale, scale, np.nan)

    # Body-axis unit vectors.
    axis_i_x = out["neck_x_i"].to_numpy(dtype=float, copy=False) - out["spine_x_i"].to_numpy(dtype=float, copy=False)
    axis_i_y = out["neck_y_i"].to_numpy(dtype=float, copy=False) - out["spine_y_i"].to_numpy(dtype=float, copy=False)
    axis_j_x = out["neck_x_j"].to_numpy(dtype=float, copy=False) - out["spine_x_j"].to_numpy(dtype=float, copy=False)
    axis_j_y = out["neck_y_j"].to_numpy(dtype=float, copy=False) - out["spine_y_j"].to_numpy(dtype=float, copy=False)

    u_i_x, u_i_y = _unit_vec(axis_i_x, axis_i_y)
    u_j_x, u_j_y = _unit_vec(axis_j_x, axis_j_y)

    # Lateral axes are 90-degree rotations of the heading axes.
    p_i_x = -u_i_y
    p_i_y = u_i_x
    p_j_x = -u_j_y
    p_j_y = u_j_x

    # Partner position vectors, normalized by dyad body scale.
    r_ij_x = out["spine_x_j"].to_numpy(dtype=float, copy=False) - out["spine_x_i"].to_numpy(dtype=float, copy=False)
    r_ij_y = out["spine_y_j"].to_numpy(dtype=float, copy=False) - out["spine_y_i"].to_numpy(dtype=float, copy=False)
    r_ji_x = -r_ij_x
    r_ji_y = -r_ij_y

    r_ij_x_norm = r_ij_x / scale_safe
    r_ij_y_norm = r_ij_y / scale_safe
    r_ji_x_norm = r_ji_x / scale_safe
    r_ji_y_norm = r_ji_y / scale_safe

    # Forward offset: signed distance of partner along focal body axis.
    out["forward_offset_i_to_j"] = r_ij_x_norm * u_i_x + r_ij_y_norm * u_i_y
    out["forward_offset_j_to_i"] = r_ji_x_norm * u_j_x + r_ji_y_norm * u_j_y

    out["abs_forward_offset_i_to_j"] = np.abs(out["forward_offset_i_to_j"])
    out["abs_forward_offset_j_to_i"] = np.abs(out["forward_offset_j_to_i"])
    out["mean_abs_forward_offset"] = out[
        ["abs_forward_offset_i_to_j", "abs_forward_offset_j_to_i"]
    ].mean(axis=1, skipna=True)

    # Lateral offset: signed distance of partner along focal lateral axis.
    out["lateral_offset_i_to_j"] = r_ij_x_norm * p_i_x + r_ij_y_norm * p_i_y
    out["lateral_offset_j_to_i"] = r_ji_x_norm * p_j_x + r_ji_y_norm * p_j_y

    out["abs_lateral_offset_i_to_j"] = np.abs(out["lateral_offset_i_to_j"])
    out["abs_lateral_offset_j_to_i"] = np.abs(out["lateral_offset_j_to_i"])
    out["mean_abs_lateral_offset"] = out[
        ["abs_lateral_offset_i_to_j", "abs_lateral_offset_j_to_i"]
    ].mean(axis=1, skipna=True)

    # Lateral dominance: higher values mean the partner position is explained more by lateral offset.
    out["lateral_dominance_i_to_j"] = (
        out["abs_lateral_offset_i_to_j"]
        / (out["abs_lateral_offset_i_to_j"] + out["abs_forward_offset_i_to_j"] + EPS)
    )
    out["lateral_dominance_j_to_i"] = (
        out["abs_lateral_offset_j_to_i"]
        / (out["abs_lateral_offset_j_to_i"] + out["abs_forward_offset_j_to_i"] + EPS)
    )
    out["mean_lateral_dominance"] = out[
        ["lateral_dominance_i_to_j", "lateral_dominance_j_to_i"]
    ].mean(axis=1, skipna=True)

    # Body-axis distance symmetry: matched landmarks should be closer than crossed ones.
    matched_cols = [
        "distance_neck_i_neck_j_norm",
        "distance_spine_i_spine_j_norm",
        "distance_tail_i_tail_j_norm",
    ]
    crossed_cols = [
        "distance_neck_i_tail_j_norm",
        "distance_tail_i_neck_j_norm",
    ]

    out["matched_body_axis_distance_norm"] = out[matched_cols].mean(axis=1, skipna=True)
    out["crossed_body_axis_distance_norm"] = out[crossed_cols].mean(axis=1, skipna=True)
    out["body_axis_symmetry"] = (
        out["crossed_body_axis_distance_norm"] - out["matched_body_axis_distance_norm"]
    )

    crossed_safe = pd.to_numeric(out["crossed_body_axis_distance_norm"], errors="coerce").to_numpy(dtype=float, copy=False)
    matched_safe = pd.to_numeric(out["matched_body_axis_distance_norm"], errors="coerce").to_numpy(dtype=float, copy=False)
    ratio = np.full(len(out), np.nan, dtype=float)
    valid_ratio = np.isfinite(matched_safe) & np.isfinite(crossed_safe) & (crossed_safe > EPS)
    ratio[valid_ratio] = matched_safe[valid_ratio] / crossed_safe[valid_ratio]
    out["body_axis_symmetry_ratio"] = ratio

    print("Added side-by-side geometry features: 17")
    for col in [
        "forward_offset_i_to_j",
        "lateral_offset_i_to_j",
        "mean_lateral_dominance",
        "body_axis_symmetry",
        "body_axis_symmetry_ratio",
    ]:
        missingness = float(out[col].isna().mean()) if len(out) > 0 else float("nan")
        print(f"  {col} missingness: {missingness:.4f}")

    return out


def _summarize_features(
    df: pd.DataFrame, feature_cols: Sequence[str]
) -> FrameFeatureSummary:
    """
    Compute summary statistics for frame features.

    Args:
        df: Features dataframe.
        feature_cols: List of feature column names to summarize.

    Returns:
        FrameFeatureSummary with statistics.
    """
    n_rows = len(df)
    n_dyads = df.groupby(["focal_id", "partner_id"]).ngroups
    n_dense_frames_unique = df["dense_frame_idx"].nunique()

    # Missingness
    missingness = {}
    for col in feature_cols:
        if col in df.columns:
            missingness[col] = float(df[col].isnull().sum() / n_rows)

    # Quantiles
    quantiles_to_compute = [0.01, 0.05, 0.5, 0.95, 0.99]
    feature_quantiles = {}

    for col in feature_cols:
        if (
            col in df.columns
            and pd.api.types.is_numeric_dtype(df[col])
            and not pd.api.types.is_bool_dtype(df[col])
        ):
            q_values = df[col].quantile(quantiles_to_compute)
            feature_quantiles[col] = {
                f"q{int(q*100)}": float(v) for q, v in zip(quantiles_to_compute, q_values)
            }

    return FrameFeatureSummary(
        n_rows=n_rows,
        n_dyads=n_dyads,
        n_dense_frames_unique=n_dense_frames_unique,
        missingness=missingness,
        feature_quantiles=feature_quantiles,
    )


# ============================================================================
# Main Public Function
# ============================================================================


def build_frame_features(
    adapted: AdaptedPose,
    *,
    exp_name: str = "session",
    iteration_name: str = "suite",
    smooth_win: int = 9,
    diff_dt: int = 1,
    bl_proxy_nodes: tuple[str, str] = ("neck", "tail_base"),
    heading_from: tuple[str, str] = ("spine_base", "neck"),
) -> FrameFeatureResult:
    """Build frame-level dyad features from an ``AdaptedPose`` tensor bundle.

    Scientific formulas match legacy ``frame_features.build_frame_features``.
    Input is the Sprint 1 S4 adapter (not Pipeline 30 NPZ files).

    Args:
        adapted: Pose tensors / quality arrays from ``adapt_final_pose_to_frame_feature_input``.
        exp_name: Session / experiment label written on directional rows.
        iteration_name: Compatibility iteration label (default ``suite``).
        smooth_win: Window size for trajectory smoothing (odd recommended).
        diff_dt: Step size for finite difference derivatives.
        bl_proxy_nodes: Nodes to use for body length proxy.
        heading_from: Nodes to use for heading vector (tail to head).

    Returns:
        FrameFeatureResult containing features dataframe and summary.
    """
    # ========================================================================
    # 1. Consume AdaptedPose (replaces dense_pose_120_imputed.npz + impute_meta)
    # ========================================================================
    arrays = adapted.to_frame_feature_arrays()
    x = np.asarray(arrays["x"], dtype=np.float64)
    y = np.asarray(arrays["y"], dtype=np.float64)
    t_120 = np.asarray(arrays["t_120"], dtype=np.float64)
    orig_frame_idx = np.asarray(arrays["orig_frame_idx"], dtype=np.int64)
    impute_stage = np.asarray(arrays["impute_stage"])
    is_imputed = np.asarray(arrays["is_imputed"], dtype=bool)
    was_refined = np.asarray(arrays["was_refined"], dtype=bool)
    point_score = np.asarray(arrays["point_score"], dtype=np.float64)

    tracks = [_normalize_track_id(t) for t in adapted.tracks]
    nodes = list(adapted.nodes)
    exp = exp_name

    if not tracks or not nodes:
        raise ValueError(f"Invalid AdaptedPose metadata: tracks={tracks}, nodes={nodes}")

    n_tracks, n_nodes_arr, T = x.shape
    if n_tracks != len(tracks) or n_nodes_arr != len(nodes) or int(t_120.shape[0]) != T:
        raise ValueError(
            "AdaptedPose array shapes do not match tracks/nodes/t_120: "
            f"x={x.shape}, tracks={len(tracks)}, nodes={len(nodes)}, T={t_120.shape[0]}"
        )
    # ========================================================================
    # 2. Validate and resolve node names
    # ========================================================================
    needed_nodes = set([
        "nose",
        "neck",
        "tail_base",
    ] + list(bl_proxy_nodes) + list(heading_from))

    # Support spine_base vs spine fallback
    node_map = {}
    for node in needed_nodes:
        if node in nodes:
            node_map[node] = node
        elif node == "spine_base" and "spine" in nodes:
            node_map[node] = "spine"
        elif node == "spine" and "spine_base" in nodes:
            node_map[node] = "spine_base"
        else:
            # Check if we can skip optional nodes
            if node in ["spine_base", "spine"]:
                # Try the other variant
                if "spine" in nodes:
                    node_map["spine_base"] = "spine"
                    node_map["spine"] = "spine"
                elif "spine_base" in nodes:
                    node_map["spine_base"] = "spine_base"
                    node_map["spine"] = "spine_base"
                else:
                    raise ValueError(f"Missing required node: {node} (or spine variant)")

    # Ensure all heading_from and bl_proxy_nodes are mapped
    for node in list(heading_from) + list(bl_proxy_nodes):
        if node not in node_map:
            if node == "spine_base" and "spine" in nodes:
                node_map[node] = "spine"
            elif node == "spine" and "spine_base" in nodes:
                node_map[node] = "spine_base"
            else:
                raise ValueError(f"Cannot resolve node '{node}'")

    # Get node indices
    def get_node_idx(name: str) -> int:
        resolved = node_map.get(name, name)
        if resolved not in nodes:
            raise ValueError(f"Node {name} (resolved to {resolved}) not in nodes list")
        return nodes.index(resolved)

    nose_idx = get_node_idx("nose")
    neck_idx = get_node_idx("neck")
    tail_base_idx = get_node_idx("tail_base")

    # For spine, prefer spine_base if available
    if "spine_base" in nodes:
        spine_idx = nodes.index("spine_base")
    elif "spine" in nodes:
        spine_idx = nodes.index("spine")
    else:
        raise ValueError("No spine or spine_base node found")

    # Heading nodes
    heading_tail_idx = get_node_idx(heading_from[0])
    heading_head_idx = get_node_idx(heading_from[1])

    # BL proxy nodes
    bl_node1_idx = get_node_idx(bl_proxy_nodes[0])
    bl_node2_idx = get_node_idx(bl_proxy_nodes[1])

    # Optional headstage node
    headstage_idx: int | None = None
    if "headstage" in nodes:
        headstage_idx = nodes.index("headstage")

    # ========================================================================
    # 3. Compute per-track kinematics
    # ========================================================================
    track_data = {}

    for track_idx, track_id in enumerate(tracks):
        # Base point for velocity (use neck)
        base_x = x[track_idx, neck_idx, :]
        base_y = y[track_idx, neck_idx, :]

        # Smooth
        base_x_smooth = _rolling_median_np(base_x, smooth_win)
        base_y_smooth = _rolling_median_np(base_y, smooth_win)

        # Velocity
        vx = _finite_diff(base_x_smooth, t_120, diff_dt)
        vy = _finite_diff(base_y_smooth, t_120, diff_dt)
        speed = np.sqrt(vx**2 + vy**2)

        # Heading vector (from tail to head)
        head_x = x[track_idx, heading_head_idx, :]
        head_y = y[track_idx, heading_head_idx, :]
        tail_x = x[track_idx, heading_tail_idx, :]
        tail_y = y[track_idx, heading_tail_idx, :]

        hx = head_x - tail_x
        hy = head_y - tail_y
        heading_x, heading_y = _unit_vec(hx, hy)

        # Body length proxy
        bl_x1 = x[track_idx, bl_node1_idx, :]
        bl_y1 = y[track_idx, bl_node1_idx, :]
        bl_x2 = x[track_idx, bl_node2_idx, :]
        bl_y2 = y[track_idx, bl_node2_idx, :]

        bl_raw = np.sqrt((bl_x1 - bl_x2)**2 + (bl_y1 - bl_y2)**2)
        bl_median = np.nanmedian(bl_raw[np.isfinite(bl_raw)])
        if not np.isfinite(bl_median) or bl_median <= 0:
            bl_median = 1.0  # Fallback

        nose_x = x[track_idx, nose_idx, :]
        nose_y = y[track_idx, nose_idx, :]
        neck_x = x[track_idx, neck_idx, :]
        neck_y = y[track_idx, neck_idx, :]
        spine_x = x[track_idx, spine_idx, :]
        spine_y = y[track_idx, spine_idx, :]
        tail_x_track = x[track_idx, tail_base_idx, :]
        tail_y_track = y[track_idx, tail_base_idx, :]

        nose_tail_raw = np.sqrt((nose_x - tail_x_track) ** 2 + (nose_y - tail_y_track) ** 2)
        neck_spine_raw = np.sqrt((neck_x - spine_x) ** 2 + (neck_y - spine_y) ** 2)
        spine_tail_raw = np.sqrt((spine_x - tail_x_track) ** 2 + (spine_y - tail_y_track) ** 2)
        nose_neck_raw = np.sqrt((nose_x - neck_x) ** 2 + (nose_y - neck_y) ** 2)
        chain_length_raw = nose_neck_raw + neck_spine_raw + spine_tail_raw

        nose_tail_median = np.nanmedian(nose_tail_raw[np.isfinite(nose_tail_raw)])
        chain_length_median = np.nanmedian(chain_length_raw[np.isfinite(chain_length_raw)])
        neck_spine_median = np.nanmedian(neck_spine_raw[np.isfinite(neck_spine_raw)])
        spine_tail_median = np.nanmedian(spine_tail_raw[np.isfinite(spine_tail_raw)])

        track_data[track_id] = {
            "vx": vx,
            "vy": vy,
            "speed": speed,
            "heading_x": heading_x,
            "heading_y": heading_y,
            "bl_median": bl_median,
            "nose_x": nose_x,
            "nose_y": nose_y,
            "neck_x": neck_x,
            "neck_y": neck_y,
            "spine_x": spine_x,
            "spine_y": spine_y,
            "tail_x": tail_x_track,
            "tail_y": tail_y_track,
            "nose_tail_median": nose_tail_median,
            "chain_length_median": chain_length_median,
            "neck_spine_median": neck_spine_median,
            "spine_tail_median": spine_tail_median,
            "headstage_x": (
                x[track_idx, headstage_idx, :]
                if headstage_idx is not None
                else np.full(T, np.nan, dtype=float)
            ),
            "headstage_y": (
                y[track_idx, headstage_idx, :]
                if headstage_idx is not None
                else np.full(T, np.nan, dtype=float)
            ),
            # Quality arrays for nose
            "nose_impute_stage": impute_stage[track_idx, nose_idx, :],
            "nose_is_imputed": is_imputed[track_idx, nose_idx, :],
            "nose_was_refined": was_refined[track_idx, nose_idx, :],
            "nose_point_score": point_score[track_idx, nose_idx, :],
            # Quality arrays for tail
            "tail_impute_stage": impute_stage[track_idx, tail_base_idx, :],
            "tail_is_imputed": is_imputed[track_idx, tail_base_idx, :],
            "tail_was_refined": was_refined[track_idx, tail_base_idx, :],
            "tail_point_score": point_score[track_idx, tail_base_idx, :],
            # Quality arrays for neck and spine (for robustness features)
            "neck_impute_stage": impute_stage[track_idx, neck_idx, :],
            "neck_is_imputed": is_imputed[track_idx, neck_idx, :],
            "neck_was_refined": was_refined[track_idx, neck_idx, :],
            "neck_point_score": point_score[track_idx, neck_idx, :],
            "spine_impute_stage": impute_stage[track_idx, spine_idx, :],
            "spine_is_imputed": is_imputed[track_idx, spine_idx, :],
            "spine_was_refined": was_refined[track_idx, spine_idx, :],
            "spine_point_score": point_score[track_idx, spine_idx, :],
        }

    # Sprint 1 implantation resolution only (never invent track 0 / argmax).
    implanted_track_id: Any | None = None
    implantation: ImplantationResolution = adapted.implantation
    if (
        implantation.status == IMPLANTATION_RESOLVED
        and implantation.implanted_track_id is not None
    ):
        candidate = _normalize_track_id(implantation.implanted_track_id)
        if candidate in tracks:
            implanted_track_id = candidate

    # ========================================================================
    # 4. Build dyad frame table for all dyad-frame combinations
    # ========================================================================
    records = []
    print("  -> Using symmetric dyad normalization for cross-mouse distances")
    dyad_scales: list[float] = []
    logged_dyads = 0

    # Generate all dyad pairs and frame indices
    for i, focal_id in enumerate(tracks):
        for j, partner_id in enumerate(tracks):
            if focal_id == partner_id:
                continue  # Skip self-pairs

            focal = track_data[focal_id]
            partner = track_data[partner_id]
            dyad_scale, body_ref_i, body_ref_j, body_ref_src_i, body_ref_src_j = _symmetric_dyad_body_scale(
                focal,
                partner,
            )
            if np.isfinite(dyad_scale):
                dyad_scales.append(float(dyad_scale))
            if logged_dyads < 4:
                print(
                    "    "
                    f"dyad {focal_id}->{partner_id}: "
                    f"body_ref_i({body_ref_src_i})={body_ref_i:.3f}, "
                    f"body_ref_j({body_ref_src_j})={body_ref_j:.3f}, "
                    f"dyad_body_scale={dyad_scale:.3f}"
                )
                logged_dyads += 1

            for dense_idx in range(T):
                # Metadata
                rec = {
                    "exp_name": exp,
                    "iteration_name": iteration_name,
                    "focal_id": focal_id,
                    "partner_id": partner_id,
                    "dense_frame_idx": dense_idx,
                    "orig_frame_idx": int(orig_frame_idx[dense_idx]),
                    "t_120": float(t_120[dense_idx]),
                    "dyad_body_scale": float(dyad_scale),
                }

                _append_contact_robustness_features(
                    rec,
                    focal=focal,
                    partner=partner,
                    dense_idx=dense_idx,
                )

                bl_i = focal["bl_median"]
                bl_j = partner["bl_median"]
                bl_mean = (bl_i + bl_j) / 2
                scale_cross = np.array([dyad_scale], dtype=float)
                scale_focal = np.array([bl_i], dtype=float)
                scale_bl_mean = np.array([bl_mean], dtype=float)

                for key in _all_headstage_feature_cols():
                    rec[key] = np.nan

                if implanted_track_id is not None and len(tracks) >= 2:
                    implanted = track_data[implanted_track_id]
                    other_track_id = (
                        partner_id if focal_id == implanted_track_id else focal_id
                    )
                    if other_track_id in track_data and other_track_id != implanted_track_id:
                        other = track_data[other_track_id]

                        hs_i_x = implanted["headstage_x"][dense_idx]
                        hs_i_y = implanted["headstage_y"][dense_idx]

                        rec["distance_headstage_i_nose_j_norm"] = _pairwise_distance(
                            np.array([hs_i_x]), np.array([hs_i_y]),
                            np.array([other["nose_x"][dense_idx]]), np.array([other["nose_y"][dense_idx]]),
                            scale_cross,
                        )[0]
                        rec["distance_headstage_i_neck_j_norm"] = _pairwise_distance(
                            np.array([hs_i_x]), np.array([hs_i_y]),
                            np.array([other["neck_x"][dense_idx]]), np.array([other["neck_y"][dense_idx]]),
                            scale_cross,
                        )[0]
                        rec["distance_headstage_i_spine_j_norm"] = _pairwise_distance(
                            np.array([hs_i_x]), np.array([hs_i_y]),
                            np.array([other["spine_x"][dense_idx]]), np.array([other["spine_y"][dense_idx]]),
                            scale_cross,
                        )[0]
                        rec["distance_headstage_i_tail_j_norm"] = _pairwise_distance(
                            np.array([hs_i_x]), np.array([hs_i_y]),
                            np.array([other["tail_x"][dense_idx]]), np.array([other["tail_y"][dense_idx]]),
                            scale_cross,
                        )[0]

                        rec["distance_headstage_i_nose_i_norm"] = _pairwise_distance(
                            np.array([hs_i_x]), np.array([hs_i_y]),
                            np.array([implanted["nose_x"][dense_idx]]), np.array([implanted["nose_y"][dense_idx]]),
                            scale_bl_mean,
                        )[0]
                        rec["distance_headstage_i_neck_i_norm"] = _pairwise_distance(
                            np.array([hs_i_x]), np.array([hs_i_y]),
                            np.array([implanted["neck_x"][dense_idx]]), np.array([implanted["neck_y"][dense_idx]]),
                            scale_bl_mean,
                        )[0]
                        rec["distance_headstage_i_spine_i_norm"] = _pairwise_distance(
                            np.array([hs_i_x]), np.array([hs_i_y]),
                            np.array([implanted["spine_x"][dense_idx]]), np.array([implanted["spine_y"][dense_idx]]),
                            scale_bl_mean,
                        )[0]
                        rec["distance_headstage_i_tail_i_norm"] = _pairwise_distance(
                            np.array([hs_i_x]), np.array([hs_i_y]),
                            np.array([implanted["tail_x"][dense_idx]]), np.array([implanted["tail_y"][dense_idx]]),
                            scale_bl_mean,
                        )[0]

                        _append_headstage_direction_features(
                            rec,
                            implanted=implanted,
                            dense_idx=dense_idx,
                        )

                # A. Compute distance
                nose_i_x = focal["nose_x"][dense_idx]
                nose_i_y = focal["nose_y"][dense_idx]
                nose_j_x = partner["nose_x"][dense_idx]
                nose_j_y = partner["nose_y"][dense_idx]
                nose_i_valid = bool(np.isfinite(nose_i_x) and np.isfinite(nose_i_y))
                nose_j_valid = bool(np.isfinite(nose_j_x) and np.isfinite(nose_j_y))
                both_noses_valid = nose_i_valid and nose_j_valid

                # Extract spine and neck coordinates early (needed for projected velocities)
                neck_i_x = focal["neck_x"][dense_idx]
                neck_i_y = focal["neck_y"][dense_idx]
                spine_i_x = focal["spine_x"][dense_idx]
                spine_i_y = focal["spine_y"][dense_idx]
                
                neck_j_x = partner["neck_x"][dense_idx]
                neck_j_y = partner["neck_y"][dense_idx]
                spine_j_x = partner["spine_x"][dense_idx]
                spine_j_y = partner["spine_y"][dense_idx]
                tail_i_x = focal["tail_x"][dense_idx]
                tail_i_y = focal["tail_y"][dense_idx]
                tail_j_x = partner["tail_x"][dense_idx]
                tail_j_y = partner["tail_y"][dense_idx]
                headstage_i_x = focal["headstage_x"][dense_idx]
                headstage_i_y = focal["headstage_y"][dense_idx]
                headstage_j_x = partner["headstage_x"][dense_idx]
                headstage_j_y = partner["headstage_y"][dense_idx]

                # Expose all raw node coordinates for debugging and feature replay.
                rec.update(
                    {
                        "nose_x_i": nose_i_x,
                        "nose_y_i": nose_i_y,
                        "neck_x_i": neck_i_x,
                        "neck_y_i": neck_i_y,
                        "spine_x_i": spine_i_x,
                        "spine_y_i": spine_i_y,
                        "tail_x_i": tail_i_x,
                        "tail_y_i": tail_i_y,
                        "headstage_x_i": headstage_i_x,
                        "headstage_y_i": headstage_i_y,
                        "nose_x_j": nose_j_x,
                        "nose_y_j": nose_j_y,
                        "neck_x_j": neck_j_x,
                        "neck_y_j": neck_j_y,
                        "spine_x_j": spine_j_x,
                        "spine_y_j": spine_j_y,
                        "tail_x_j": tail_j_x,
                        "tail_y_j": tail_j_y,
                        "headstage_x_j": headstage_j_x,
                        "headstage_y_j": headstage_j_y,
                        # True-name aliases to match pose/impute metadata.
                        "spine_base_x_i": spine_i_x,
                        "spine_base_y_i": spine_i_y,
                        "spine_base_x_j": spine_j_x,
                        "spine_base_y_j": spine_j_y,
                        "tail_base_x_i": tail_i_x,
                        "tail_base_y_i": tail_i_y,
                        "tail_base_x_j": tail_j_x,
                        "tail_base_y_j": tail_j_y,
                    }
                )

                if both_noses_valid:
                    dx = nose_i_x - nose_j_x
                    dy = nose_i_y - nose_j_y
                    dist_raw = np.sqrt(dx**2 + dy**2)
                    rec["d_nose_nose_norm"] = _safe_normalize_scalar(dist_raw, dyad_scale)
                else:
                    dx = np.nan
                    dy = np.nan
                    rec["d_nose_nose_norm"] = np.nan

                # Distance derivative
                if both_noses_valid and dense_idx >= diff_dt:
                    prev_idx = dense_idx - diff_dt
                    nose_i_prev_x = focal["nose_x"][prev_idx]
                    nose_i_prev_y = focal["nose_y"][prev_idx]
                    nose_j_prev_x = partner["nose_x"][prev_idx]
                    nose_j_prev_y = partner["nose_y"][prev_idx]
                    if (
                        np.isfinite(nose_i_prev_x)
                        and np.isfinite(nose_i_prev_y)
                        and np.isfinite(nose_j_prev_x)
                        and np.isfinite(nose_j_prev_y)
                    ):
                        dx_prev = nose_i_prev_x - nose_j_prev_x
                        dy_prev = nose_i_prev_y - nose_j_prev_y
                        dist_prev = _safe_normalize_scalar(np.sqrt(dx_prev**2 + dy_prev**2), dyad_scale)
                        dt = t_120[dense_idx] - t_120[prev_idx]
                        if dt > 0 and np.isfinite(dist_prev):
                            rec["ddist_dt"] = (rec["d_nose_nose_norm"] - dist_prev) / dt
                        else:
                            rec["ddist_dt"] = np.nan
                    else:
                        rec["ddist_dt"] = np.nan
                else:
                    rec["ddist_dt"] = np.nan

                rec["abs_ddist_dt"] = abs(rec["ddist_dt"]) if np.isfinite(rec["ddist_dt"]) else np.nan

                # B. BL-normalized kinematics
                v_i = focal["speed"][dense_idx]
                v_j = partner["speed"][dense_idx]

                rec["v_i_bl"] = v_i / bl_i
                rec["v_j_bl"] = v_j / bl_j
                rec["speed_i"] = v_i
                rec["speed_j"] = v_j

                # Relative velocity
                vx_i = focal["vx"][dense_idx]
                vy_i = focal["vy"][dense_idx]
                vx_j = partner["vx"][dense_idx]
                vy_j = partner["vy"][dense_idx]

                dvx = vx_i - vx_j
                dvy = vy_i - vy_j
                v_rel = np.sqrt(dvx**2 + dvy**2)
                rec["v_rel_bl"] = v_rel / bl_mean

                # Neck displacement over 15-frame window
                if dense_idx >= 15:
                    neck_i_x_prev = focal["neck_x"][dense_idx - 15]
                    neck_i_y_prev = focal["neck_y"][dense_idx - 15]
                    neck_j_x_prev = partner["neck_x"][dense_idx - 15]
                    neck_j_y_prev = partner["neck_y"][dense_idx - 15]
                    
                    if np.isfinite(neck_i_x) and np.isfinite(neck_i_y) and np.isfinite(neck_i_x_prev) and np.isfinite(neck_i_y_prev):
                        neck_displacement_15_i = np.sqrt(
                            (neck_i_x - neck_i_x_prev)**2 + (neck_i_y - neck_i_y_prev)**2
                        )
                        rec["neck_displacement_15_i"] = neck_displacement_15_i
                        rec["neck_directed_vel_15_i"] = neck_displacement_15_i / 15.0
                    else:
                        rec["neck_displacement_15_i"] = np.nan
                        rec["neck_directed_vel_15_i"] = np.nan
                    
                    if np.isfinite(neck_j_x) and np.isfinite(neck_j_y) and np.isfinite(neck_j_x_prev) and np.isfinite(neck_j_y_prev):
                        neck_displacement_15_j = np.sqrt(
                            (neck_j_x - neck_j_x_prev)**2 + (neck_j_y - neck_j_y_prev)**2
                        )
                        rec["neck_displacement_15_j"] = neck_displacement_15_j
                        rec["neck_directed_vel_15_j"] = neck_displacement_15_j / 15.0
                    else:
                        rec["neck_displacement_15_j"] = np.nan
                        rec["neck_directed_vel_15_j"] = np.nan
                else:
                    rec["neck_displacement_15_i"] = np.nan
                    rec["neck_directed_vel_15_i"] = np.nan
                    rec["neck_displacement_15_j"] = np.nan
                    rec["neck_directed_vel_15_j"] = np.nan

                # ===== Relational Physics: Projected Velocities (Intent Signals) =====
                # Computing unit vector from spine_i to spine_j for projection
                spine_dx = spine_j_x - spine_i_x
                spine_dy = spine_j_y - spine_i_y
                
                if np.isfinite(spine_i_x) and np.isfinite(spine_i_y) and np.isfinite(spine_j_x) and np.isfinite(spine_j_y):
                    ux_spine_ij, uy_spine_ij = _unit_vec(np.array([spine_dx]), np.array([spine_dy]))

                    # Velocity intercept: cosine between focal velocity and i->j spine vector.
                    rec["cos_velocity_i_to_j"] = _scalar_cosine(
                        vx_i,
                        vy_i,
                        spine_dx,
                        spine_dy,
                    )
                    
                    if np.isfinite(ux_spine_ij[0]) and np.isfinite(uy_spine_ij[0]):
                        # Velocity of i projected onto direction to j
                        proj_vel_i_to_j = _safe_dot(np.array([vx_i]), np.array([vy_i]), ux_spine_ij, uy_spine_ij)[0]
                        rec["projected_vel_i_to_j"] = proj_vel_i_to_j / bl_i
                        
                        # Velocity of j projected onto direction back to i (unit vector reversed)
                        ux_spine_ji, uy_spine_ji = -ux_spine_ij, -uy_spine_ij
                        proj_vel_j_to_i = _safe_dot(np.array([vx_j]), np.array([vy_j]), ux_spine_ji, uy_spine_ji)[0]
                        rec["projected_vel_j_to_i"] = proj_vel_j_to_i / bl_j
                    else:
                        rec["projected_vel_i_to_j"] = np.nan
                        rec["projected_vel_j_to_i"] = np.nan
                else:
                    rec["cos_velocity_i_to_j"] = np.nan
                    rec["projected_vel_i_to_j"] = np.nan
                    rec["projected_vel_j_to_i"] = np.nan

                rec["max_projected_vel"] = _nanmax2(
                    rec["projected_vel_i_to_j"],
                    rec["projected_vel_j_to_i"],
                )

                # Geometric intent: bearing cosine from mouse i body axis to partner direction.
                body_axis_i_dx = neck_i_x - spine_i_x
                body_axis_i_dy = neck_i_y - spine_i_y
                rel_partner_i_dx = spine_j_x - spine_i_x
                rel_partner_i_dy = spine_j_y - spine_i_y
                rec["cos_bearing_i_to_j"] = _scalar_cosine(
                    body_axis_i_dx,
                    body_axis_i_dy,
                    rel_partner_i_dx,
                    rel_partner_i_dy,
                )

                # Bearing asymmetry: partner mouse body-axis bearing back to focal.
                body_axis_j_dx = neck_j_x - spine_j_x
                body_axis_j_dy = neck_j_y - spine_j_y
                rel_partner_j_dx = spine_i_x - spine_j_x
                rel_partner_j_dy = spine_i_y - spine_j_y
                rec["cos_bearing_j_to_i"] = _scalar_cosine(
                    body_axis_j_dx,
                    body_axis_j_dy,
                    rel_partner_j_dx,
                    rel_partner_j_dy,
                )

                # Distance-gated bearing intent: strong only when dyad is physically close.
                spine_dist_norm_local = _safe_normalize_scalar(
                    float(np.hypot(spine_dx, spine_dy)),
                    dyad_scale,
                )
                if np.isfinite(rec["cos_bearing_i_to_j"]) and np.isfinite(spine_dist_norm_local):
                    rec["bearing_while_close"] = float(
                        rec["cos_bearing_i_to_j"] * np.exp(-spine_dist_norm_local / 0.5)
                    )
                else:
                    rec["bearing_while_close"] = np.nan

                if np.isfinite(rec["cos_bearing_j_to_i"]) and np.isfinite(spine_dist_norm_local):
                    rec["bearing_while_close_j_to_i"] = float(
                        rec["cos_bearing_j_to_i"] * np.exp(-spine_dist_norm_local / 0.5)
                    )
                else:
                    rec["bearing_while_close_j_to_i"] = np.nan

                # Tangential dynamics around spine-to-spine axis.
                if dense_idx >= diff_dt and np.isfinite(spine_dx) and np.isfinite(spine_dy):
                    prev_idx = dense_idx - diff_dt
                    spine_i_prev_x = focal["spine_x"][prev_idx]
                    spine_i_prev_y = focal["spine_y"][prev_idx]
                    spine_j_prev_x = partner["spine_x"][prev_idx]
                    spine_j_prev_y = partner["spine_y"][prev_idx]

                    if (
                        np.isfinite(spine_i_prev_x)
                        and np.isfinite(spine_i_prev_y)
                        and np.isfinite(spine_j_prev_x)
                        and np.isfinite(spine_j_prev_y)
                    ):
                        dt_spine = t_120[dense_idx] - t_120[prev_idx]
                        if dt_spine > 0:
                            vx_spine_i = (spine_i_x - spine_i_prev_x) / dt_spine
                            vy_spine_i = (spine_i_y - spine_i_prev_y) / dt_spine
                            vx_spine_j = (spine_j_x - spine_j_prev_x) / dt_spine
                            vy_spine_j = (spine_j_y - spine_j_prev_y) / dt_spine

                            v_rel_spine_x = vx_spine_i - vx_spine_j
                            v_rel_spine_y = vy_spine_i - vy_spine_j

                            if np.isfinite(ux_spine_ij[0]) and np.isfinite(uy_spine_ij[0]):
                                v_radial_scalar = (
                                    v_rel_spine_x * ux_spine_ij[0]
                                    + v_rel_spine_y * uy_spine_ij[0]
                                )
                                v_radial_x = v_radial_scalar * ux_spine_ij[0]
                                v_radial_y = v_radial_scalar * uy_spine_ij[0]
                                v_tangential = np.hypot(
                                    v_rel_spine_x - v_radial_x,
                                    v_rel_spine_y - v_radial_y,
                                )
                                rec["vel_tangential_spine"] = (
                                    v_tangential / bl_mean if bl_mean > EPS else np.nan
                                )
                            else:
                                rec["vel_tangential_spine"] = np.nan

                            prev_spine_dx = spine_j_prev_x - spine_i_prev_x
                            prev_spine_dy = spine_j_prev_y - spine_i_prev_y
                            if np.isfinite(prev_spine_dx) and np.isfinite(prev_spine_dy):
                                angle_curr = np.arctan2(spine_dy, spine_dx)
                                angle_prev = np.arctan2(prev_spine_dy, prev_spine_dx)
                                dtheta = np.arctan2(
                                    np.sin(angle_curr - angle_prev),
                                    np.cos(angle_curr - angle_prev),
                                )
                                rec["angular_velocity_rel"] = float(np.degrees(dtheta) / dt_spine)
                            else:
                                rec["angular_velocity_rel"] = np.nan
                        else:
                            rec["vel_tangential_spine"] = np.nan
                            rec["angular_velocity_rel"] = np.nan
                    else:
                        rec["vel_tangential_spine"] = np.nan
                        rec["angular_velocity_rel"] = np.nan
                else:
                    rec["vel_tangential_spine"] = np.nan
                    rec["angular_velocity_rel"] = np.nan
                
                # Magnitude of relative velocity
                rec["relative_velocity_mag"] = v_rel / bl_mean

                # Relative speed difference
                max_speed_bl = max(rec["v_i_bl"], rec["v_j_bl"])
                if max_speed_bl > EPS:
                    rec["rel_speed_diff"] = abs(rec["v_i_bl"] - rec["v_j_bl"]) / max_speed_bl
                else:
                    rec["rel_speed_diff"] = 0.0

                # C. Motion projection (radial velocity)
                # Unit vector from focal to partner
                if both_noses_valid:
                    ux_ij, uy_ij = _unit_vec(np.array([dx]), np.array([dy]))
                else:
                    ux_ij = np.array([np.nan], dtype=float)
                    uy_ij = np.array([np.nan], dtype=float)

                if np.isfinite(ux_ij[0]) and np.isfinite(uy_ij[0]):
                    v_rad_i = _safe_dot(np.array([vx_i]), np.array([vy_i]), ux_ij, uy_ij)[0]
                    rec["v_rad_i_bl"] = v_rad_i / bl_i
                else:
                    rec["v_rad_i_bl"] = np.nan

                # D. Orientation
                hx_i = focal["heading_x"][dense_idx]
                hy_i = focal["heading_y"][dense_idx]
                hx_j = partner["heading_x"][dense_idx]
                hy_j = partner["heading_y"][dense_idx]

                # Internal Dynamics: per-mouse angular velocity (deg/s).
                if dense_idx >= diff_dt and np.isfinite(hx_i) and np.isfinite(hy_i):
                    prev_idx = dense_idx - diff_dt
                    hx_i_prev = focal["heading_x"][prev_idx]
                    hy_i_prev = focal["heading_y"][prev_idx]
                    if np.isfinite(hx_i_prev) and np.isfinite(hy_i_prev):
                        dt_heading_i = t_120[dense_idx] - t_120[prev_idx]
                        if dt_heading_i > 0:
                            angle_i_curr = np.arctan2(hy_i, hx_i)
                            angle_i_prev = np.arctan2(hy_i_prev, hx_i_prev)
                            dtheta_i = np.arctan2(
                                np.sin(angle_i_curr - angle_i_prev),
                                np.cos(angle_i_curr - angle_i_prev),
                            )
                            rec["angular_velocity_i"] = float(np.degrees(dtheta_i) / dt_heading_i)
                        else:
                            rec["angular_velocity_i"] = np.nan
                    else:
                        rec["angular_velocity_i"] = np.nan
                else:
                    rec["angular_velocity_i"] = np.nan

                if dense_idx >= diff_dt and np.isfinite(hx_j) and np.isfinite(hy_j):
                    prev_idx = dense_idx - diff_dt
                    hx_j_prev = partner["heading_x"][prev_idx]
                    hy_j_prev = partner["heading_y"][prev_idx]
                    if np.isfinite(hx_j_prev) and np.isfinite(hy_j_prev):
                        dt_heading_j = t_120[dense_idx] - t_120[prev_idx]
                        if dt_heading_j > 0:
                            angle_j_curr = np.arctan2(hy_j, hx_j)
                            angle_j_prev = np.arctan2(hy_j_prev, hx_j_prev)
                            dtheta_j = np.arctan2(
                                np.sin(angle_j_curr - angle_j_prev),
                                np.cos(angle_j_curr - angle_j_prev),
                            )
                            rec["angular_velocity_j"] = float(np.degrees(dtheta_j) / dt_heading_j)
                        else:
                            rec["angular_velocity_j"] = np.nan
                    else:
                        rec["angular_velocity_j"] = np.nan
                else:
                    rec["angular_velocity_j"] = np.nan

                if np.isfinite(ux_ij[0]) and np.isfinite(uy_ij[0]):
                    rec["cos_heading_i_to_j"] = _safe_dot(
                        np.array([hx_i]), np.array([hy_i]), ux_ij, uy_ij
                    )[0]

                    # Unit vector from partner to focal
                    ux_ji, uy_ji = _unit_vec(-np.array([dx]), -np.array([dy]))
                    rec["cos_heading_j_to_i"] = _safe_dot(
                        np.array([hx_j]), np.array([hy_j]), ux_ji, uy_ji
                    )[0]
                else:
                    rec["cos_heading_i_to_j"] = np.nan
                    rec["cos_heading_j_to_i"] = np.nan

                # Asymmetric crosshair angles from focal heading to partner anchors.
                tail_j_x_anchor = partner["tail_x"][dense_idx]
                tail_j_y_anchor = partner["tail_y"][dense_idx]
                spine_j_x_anchor = partner["spine_x"][dense_idx]
                spine_j_y_anchor = partner["spine_y"][dense_idx]

                if np.isfinite(hx_i) and np.isfinite(hy_i) and np.isfinite(nose_i_x) and np.isfinite(nose_i_y):
                    tail_vec_dx = tail_j_x_anchor - nose_i_x
                    tail_vec_dy = tail_j_y_anchor - nose_i_y
                    ux_i_tail_j, uy_i_tail_j = _unit_vec(
                        np.array([tail_vec_dx]),
                        np.array([tail_vec_dy]),
                    )
                    if np.isfinite(ux_i_tail_j[0]) and np.isfinite(uy_i_tail_j[0]):
                        rec["cos_heading_i_to_tail_j"] = _safe_dot(
                            np.array([hx_i]),
                            np.array([hy_i]),
                            ux_i_tail_j,
                            uy_i_tail_j,
                        )[0]
                    else:
                        rec["cos_heading_i_to_tail_j"] = np.nan

                    spine_vec_dx = spine_j_x_anchor - nose_i_x
                    spine_vec_dy = spine_j_y_anchor - nose_i_y
                    ux_i_spine_j, uy_i_spine_j = _unit_vec(
                        np.array([spine_vec_dx]),
                        np.array([spine_vec_dy]),
                    )
                    if np.isfinite(ux_i_spine_j[0]) and np.isfinite(uy_i_spine_j[0]):
                        rec["cos_heading_i_to_spine_j"] = _safe_dot(
                            np.array([hx_i]),
                            np.array([hy_i]),
                            ux_i_spine_j,
                            uy_i_spine_j,
                        )[0]
                    else:
                        rec["cos_heading_i_to_spine_j"] = np.nan
                else:
                    rec["cos_heading_i_to_tail_j"] = np.nan
                    rec["cos_heading_i_to_spine_j"] = np.nan

                if np.isfinite(hx_j) and np.isfinite(hy_j) and np.isfinite(nose_j_x) and np.isfinite(nose_j_y):
                    tail_vec_dx = tail_i_x - nose_j_x
                    tail_vec_dy = tail_i_y - nose_j_y
                    ux_j_tail_i, uy_j_tail_i = _unit_vec(
                        np.array([tail_vec_dx]),
                        np.array([tail_vec_dy]),
                    )
                    if np.isfinite(ux_j_tail_i[0]) and np.isfinite(uy_j_tail_i[0]):
                        rec["cos_heading_j_to_tail_i"] = _safe_dot(
                            np.array([hx_j]),
                            np.array([hy_j]),
                            ux_j_tail_i,
                            uy_j_tail_i,
                        )[0]
                    else:
                        rec["cos_heading_j_to_tail_i"] = np.nan

                    spine_vec_dx = spine_i_x - nose_j_x
                    spine_vec_dy = spine_i_y - nose_j_y
                    ux_j_spine_i, uy_j_spine_i = _unit_vec(
                        np.array([spine_vec_dx]),
                        np.array([spine_vec_dy]),
                    )
                    if np.isfinite(ux_j_spine_i[0]) and np.isfinite(uy_j_spine_i[0]):
                        rec["cos_heading_j_to_spine_i"] = _safe_dot(
                            np.array([hx_j]),
                            np.array([hy_j]),
                            ux_j_spine_i,
                            uy_j_spine_i,
                        )[0]
                    else:
                        rec["cos_heading_j_to_spine_i"] = np.nan
                else:
                    rec["cos_heading_j_to_tail_i"] = np.nan
                    rec["cos_heading_j_to_spine_i"] = np.nan

                # Heading alignment
                rec["cos_heading_alignment"] = _safe_dot(
                    np.array([hx_i]), np.array([hy_i]),
                    np.array([hx_j]), np.array([hy_j])
                )[0]

                # ===== Relational Physics: Angular Dynamics =====
                # cos_heading_alignment_delta: First derivative of heading alignment  
                if dense_idx >= diff_dt and np.isfinite(rec["cos_heading_alignment"]):
                    prev_idx = dense_idx - diff_dt
                    hx_i_prev = focal["heading_x"][prev_idx]
                    hy_i_prev = focal["heading_y"][prev_idx]
                    hx_j_prev = partner["heading_x"][prev_idx]
                    hy_j_prev = partner["heading_y"][prev_idx]
                    if (np.isfinite(hx_i_prev) and np.isfinite(hy_i_prev) and 
                        np.isfinite(hx_j_prev) and np.isfinite(hy_j_prev)):
                        cos_align_prev = _safe_dot(
                            np.array([hx_i_prev]), np.array([hy_i_prev]),
                            np.array([hx_j_prev]), np.array([hy_j_prev])
                        )[0]
                        if np.isfinite(cos_align_prev):
                            dt = t_120[dense_idx] - t_120[prev_idx]
                            if dt > 0:
                                rec["cos_heading_alignment_delta"] = (rec["cos_heading_alignment"] - cos_align_prev) / dt
                            else:
                                rec["cos_heading_alignment_delta"] = np.nan
                        else:
                            rec["cos_heading_alignment_delta"] = np.nan
                    else:
                        rec["cos_heading_alignment_delta"] = np.nan
                else:
                    rec["cos_heading_alignment_delta"] = np.nan

                # ===== Angular Intent: Heading Angle to Partner =====
                # Compute angle from focal nose to partner center (spine), in radians
                # This captures the instantaneous direction focal is orienting toward partner.
                if np.isfinite(nose_i_x) and np.isfinite(nose_i_y) and np.isfinite(spine_j_x) and np.isfinite(spine_j_y):
                    angle_to_partner_dx = spine_j_x - nose_i_x
                    angle_to_partner_dy = spine_j_y - nose_i_y
                    rec["heading_angle_to_j"] = float(np.arctan2(angle_to_partner_dy, angle_to_partner_dx))
                else:
                    rec["heading_angle_to_j"] = np.nan

                if np.isfinite(nose_j_x) and np.isfinite(nose_j_y) and np.isfinite(spine_i_x) and np.isfinite(spine_i_y):
                    angle_to_partner_dx = spine_i_x - nose_j_x
                    angle_to_partner_dy = spine_i_y - nose_j_y
                    rec["heading_angle_to_i"] = float(np.arctan2(angle_to_partner_dy, angle_to_partner_dx))
                else:
                    rec["heading_angle_to_i"] = np.nan

                # Snout alignment (neck -> nose), captures face orientation agreement.
                snout_dx_i = nose_i_x - focal["neck_x"][dense_idx]
                snout_dy_i = nose_i_y - focal["neck_y"][dense_idx]
                snout_dx_j = nose_j_x - partner["neck_x"][dense_idx]
                snout_dy_j = nose_j_y - partner["neck_y"][dense_idx]

                mag_i = np.hypot(snout_dx_i, snout_dy_i)
                mag_j = np.hypot(snout_dx_j, snout_dy_j)
                dot_product = (snout_dx_i * snout_dx_j) + (snout_dy_i * snout_dy_j)

                with np.errstate(divide="ignore", invalid="ignore"):
                    rec["cos_snout_alignment"] = dot_product / (mag_i * mag_j)
                rec["cos_snout_alignment"] = float(np.nan_to_num(rec["cos_snout_alignment"], nan=0.0))

                # Targeting: is focal mouse's headstage pointed at partner's nose?
                hs_x = implanted["headstage_x"][dense_idx] if implanted_track_id is not None and focal_id == implanted_track_id else focal["headstage_x"][dense_idx]
                hs_y = implanted["headstage_y"][dense_idx] if implanted_track_id is not None and focal_id == implanted_track_id else focal["headstage_y"][dense_idx]
                neck_i_x_val = focal["neck_x"][dense_idx]
                neck_i_y_val = focal["neck_y"][dense_idx]

                target_dx = nose_j_x - neck_i_x_val
                target_dy = nose_j_y - neck_i_y_val
                hs_dx = hs_x - neck_i_x_val
                hs_dy = hs_y - neck_i_y_val

                mag_target = np.hypot(target_dx, target_dy)
                mag_hs = np.hypot(hs_dx, hs_dy)
                dot_hs_target = (hs_dx * target_dx) + (hs_dy * target_dy)

                with np.errstate(divide="ignore", invalid="ignore"):
                    rec["cos_hs_i_to_nose_j"] = dot_hs_target / (mag_hs * mag_target)
                rec["cos_hs_i_to_nose_j"] = float(np.nan_to_num(rec["cos_hs_i_to_nose_j"], nan=0.0))

                # E. Contact geometry distances
                # (neck_i_x, neck_i_y, spine_i_x, spine_i_y, neck_j_x, neck_j_y, spine_j_x, spine_j_y already extracted above)
                # Use mean BL for normalization

                rec["distance_nose_i_nose_j_norm"] = _pairwise_distance(
                    np.array([nose_i_x]), np.array([nose_i_y]),
                    np.array([nose_j_x]), np.array([nose_j_y]), scale_cross
                )[0]

                rec["distance_nose_i_neck_j_norm"] = _pairwise_distance(
                    np.array([nose_i_x]), np.array([nose_i_y]),
                    np.array([neck_j_x]), np.array([neck_j_y]), scale_focal
                )[0]

                rec["distance_nose_i_spine_j_norm"] = _pairwise_distance(
                    np.array([nose_i_x]), np.array([nose_i_y]),
                    np.array([spine_j_x]), np.array([spine_j_y]), scale_focal
                )[0]

                rec["distance_nose_i_tail_j_norm"] = _pairwise_distance(
                    np.array([nose_i_x]), np.array([nose_i_y]),
                    np.array([tail_j_x]), np.array([tail_j_y]), scale_focal
                )[0]

                rec["distance_neck_i_nose_j_norm"] = _pairwise_distance(
                    np.array([neck_i_x]), np.array([neck_i_y]),
                    np.array([nose_j_x]), np.array([nose_j_y]), scale_focal
                )[0]

                rec["distance_spine_i_nose_j_norm"] = _pairwise_distance(
                    np.array([spine_i_x]), np.array([spine_i_y]),
                    np.array([nose_j_x]), np.array([nose_j_y]), scale_cross
                )[0]

                rec["distance_tail_i_nose_j_norm"] = _pairwise_distance(
                    np.array([tail_i_x]), np.array([tail_i_y]),
                    np.array([nose_j_x]), np.array([nose_j_y]), scale_cross
                )[0]

                rec["distance_spine_i_spine_j_norm"] = _pairwise_distance(
                    np.array([spine_i_x]), np.array([spine_i_y]),
                    np.array([spine_j_x]), np.array([spine_j_y]), scale_cross
                )[0]

                # ===== Relational Physics: Radial Velocities (Distance Derivatives) =====
                # Compute first derivatives of spine, neck, nose distances
                # (rate of approach/separation)
                def _compute_distance_derivative(curr_dist_norm, prev_idx_val):
                    """Helper to compute distance derivative at 120 FPS sampling."""
                    if dense_idx >= diff_dt and np.isfinite(curr_dist_norm):
                        return curr_dist_norm, prev_idx_val
                    return None, None
                
                # vel_radial_spine: d(distance_spine_i_spine_j_norm) / dt
                if (dense_idx >= diff_dt and np.isfinite(rec["distance_spine_i_spine_j_norm"]) and 
                    np.isfinite(spine_i_x) and np.isfinite(spine_i_y) and 
                    np.isfinite(spine_j_x) and np.isfinite(spine_j_y)):
                    prev_idx = dense_idx - diff_dt
                    spine_i_prev_x = focal["spine_x"][prev_idx]
                    spine_i_prev_y = focal["spine_y"][prev_idx]
                    spine_j_prev_x = partner["spine_x"][prev_idx]
                    spine_j_prev_y = partner["spine_y"][prev_idx]
                    if (np.isfinite(spine_i_prev_x) and np.isfinite(spine_i_prev_y) and 
                        np.isfinite(spine_j_prev_x) and np.isfinite(spine_j_prev_y)):
                        dist_prev = _pairwise_distance(
                            np.array([spine_i_prev_x]), np.array([spine_i_prev_y]),
                            np.array([spine_j_prev_x]), np.array([spine_j_prev_y]), scale_cross
                        )[0]
                        dt = t_120[dense_idx] - t_120[prev_idx]
                        if dt > 0 and np.isfinite(dist_prev):
                            rec["vel_radial_spine"] = (rec["distance_spine_i_spine_j_norm"] - dist_prev) / dt
                        else:
                            rec["vel_radial_spine"] = np.nan
                    else:
                        rec["vel_radial_spine"] = np.nan
                else:
                    rec["vel_radial_spine"] = np.nan

                # vel_radial_neck: d(distance_neck_i_neck_j_norm) / dt
                # Compute distance_neck_i_neck_j_norm first, then derivative
                dist_neck_i_neck_j_norm = _pairwise_distance(
                    np.array([neck_i_x]), np.array([neck_i_y]),
                    np.array([neck_j_x]), np.array([neck_j_y]), scale_focal
                )[0]
                if (dense_idx >= diff_dt and np.isfinite(dist_neck_i_neck_j_norm) and 
                    np.isfinite(neck_i_x) and np.isfinite(neck_i_y) and 
                    np.isfinite(neck_j_x) and np.isfinite(neck_j_y)):
                    prev_idx = dense_idx - diff_dt
                    neck_i_prev_x = focal["neck_x"][prev_idx]
                    neck_i_prev_y = focal["neck_y"][prev_idx]
                    neck_j_prev_x = partner["neck_x"][prev_idx]
                    neck_j_prev_y = partner["neck_y"][prev_idx]
                    if (np.isfinite(neck_i_prev_x) and np.isfinite(neck_i_prev_y) and 
                        np.isfinite(neck_j_prev_x) and np.isfinite(neck_j_prev_y)):
                        dist_prev = _pairwise_distance(
                            np.array([neck_i_prev_x]), np.array([neck_i_prev_y]),
                            np.array([neck_j_prev_x]), np.array([neck_j_prev_y]), scale_focal
                        )[0]
                        dt = t_120[dense_idx] - t_120[prev_idx]
                        if dt > 0 and np.isfinite(dist_prev):
                            rec["vel_radial_neck"] = (dist_neck_i_neck_j_norm - dist_prev) / dt
                        else:
                            rec["vel_radial_neck"] = np.nan
                    else:
                        rec["vel_radial_neck"] = np.nan
                else:
                    rec["vel_radial_neck"] = np.nan

                # vel_radial_nose: d(distance_nose_i_nose_j_norm) / dt
                if (dense_idx >= diff_dt and np.isfinite(rec["distance_nose_i_nose_j_norm"]) and 
                    both_noses_valid):
                    prev_idx = dense_idx - diff_dt
                    nose_i_prev_x = focal["nose_x"][prev_idx]
                    nose_i_prev_y = focal["nose_y"][prev_idx]
                    nose_j_prev_x = partner["nose_x"][prev_idx]
                    nose_j_prev_y = partner["nose_y"][prev_idx]
                    if (np.isfinite(nose_i_prev_x) and np.isfinite(nose_i_prev_y) and 
                        np.isfinite(nose_j_prev_x) and np.isfinite(nose_j_prev_y)):
                        dist_prev = _pairwise_distance(
                            np.array([nose_i_prev_x]), np.array([nose_i_prev_y]),
                            np.array([nose_j_prev_x]), np.array([nose_j_prev_y]), scale_cross
                        )[0]
                        dt = t_120[dense_idx] - t_120[prev_idx]
                        if dt > 0 and np.isfinite(dist_prev):
                            rec["vel_radial_nose"] = (rec["distance_nose_i_nose_j_norm"] - dist_prev) / dt
                        else:
                            rec["vel_radial_nose"] = np.nan
                    else:
                        rec["vel_radial_nose"] = np.nan
                else:
                    rec["vel_radial_nose"] = np.nan

                # vel_radial_nose_i_tail_j: d(distance_nose_i_tail_j_norm) / dt
                if (
                    dense_idx >= diff_dt
                    and np.isfinite(rec["distance_nose_i_tail_j_norm"])
                    and np.isfinite(nose_i_x)
                    and np.isfinite(nose_i_y)
                    and np.isfinite(tail_j_x)
                    and np.isfinite(tail_j_y)
                ):
                    prev_idx = dense_idx - diff_dt
                    nose_i_prev_x = focal["nose_x"][prev_idx]
                    nose_i_prev_y = focal["nose_y"][prev_idx]
                    tail_j_prev_x = partner["tail_x"][prev_idx]
                    tail_j_prev_y = partner["tail_y"][prev_idx]
                    if (
                        np.isfinite(nose_i_prev_x)
                        and np.isfinite(nose_i_prev_y)
                        and np.isfinite(tail_j_prev_x)
                        and np.isfinite(tail_j_prev_y)
                    ):
                        dist_prev = _pairwise_distance(
                            np.array([nose_i_prev_x]), np.array([nose_i_prev_y]),
                            np.array([tail_j_prev_x]), np.array([tail_j_prev_y]), scale_focal
                        )[0]
                        dt = t_120[dense_idx] - t_120[prev_idx]
                        if dt > 0 and np.isfinite(dist_prev):
                            rec["vel_radial_nose_i_tail_j"] = (
                                rec["distance_nose_i_tail_j_norm"] - dist_prev
                            ) / dt
                        else:
                            rec["vel_radial_nose_i_tail_j"] = np.nan
                    else:
                        rec["vel_radial_nose_i_tail_j"] = np.nan
                else:
                    rec["vel_radial_nose_i_tail_j"] = np.nan

                # vel_radial_tail_i_nose_j: d(distance_tail_i_nose_j_norm) / dt
                if (
                    dense_idx >= diff_dt
                    and np.isfinite(tail_i_x)
                    and np.isfinite(tail_i_y)
                    and np.isfinite(nose_j_x)
                    and np.isfinite(nose_j_y)
                ):
                    curr_dist_tail_i_nose_j = _pairwise_distance(
                        np.array([tail_i_x]), np.array([tail_i_y]),
                        np.array([nose_j_x]), np.array([nose_j_y]), scale_cross
                    )[0]
                    prev_idx = dense_idx - diff_dt
                    tail_i_prev_x = focal["tail_x"][prev_idx]
                    tail_i_prev_y = focal["tail_y"][prev_idx]
                    nose_j_prev_x = partner["nose_x"][prev_idx]
                    nose_j_prev_y = partner["nose_y"][prev_idx]
                    if (
                        np.isfinite(curr_dist_tail_i_nose_j)
                        and np.isfinite(tail_i_prev_x)
                        and np.isfinite(tail_i_prev_y)
                        and np.isfinite(nose_j_prev_x)
                        and np.isfinite(nose_j_prev_y)
                    ):
                        prev_dist_tail_i_nose_j = _pairwise_distance(
                            np.array([tail_i_prev_x]), np.array([tail_i_prev_y]),
                            np.array([nose_j_prev_x]), np.array([nose_j_prev_y]), scale_cross
                        )[0]
                        dt = t_120[dense_idx] - t_120[prev_idx]
                        if dt > 0 and np.isfinite(prev_dist_tail_i_nose_j):
                            rec["vel_radial_tail_i_nose_j"] = (
                                curr_dist_tail_i_nose_j - prev_dist_tail_i_nose_j
                            ) / dt
                        else:
                            rec["vel_radial_tail_i_nose_j"] = np.nan
                    else:
                        rec["vel_radial_tail_i_nose_j"] = np.nan
                else:
                    rec["vel_radial_tail_i_nose_j"] = np.nan

                # Conservative CONTACT-v2 body-distance block
                rec["distance_neck_i_neck_j_norm"] = dist_neck_i_neck_j_norm

                rec["distance_neck_i_spine_j_norm"] = _pairwise_distance(
                    np.array([neck_i_x]), np.array([neck_i_y]),
                    np.array([spine_j_x]), np.array([spine_j_y]), scale_focal
                )[0]

                rec["distance_neck_i_tail_j_norm"] = _pairwise_distance(
                    np.array([neck_i_x]), np.array([neck_i_y]),
                    np.array([tail_j_x]), np.array([tail_j_y]), scale_focal
                )[0]

                rec["distance_spine_i_neck_j_norm"] = _pairwise_distance(
                    np.array([spine_i_x]), np.array([spine_i_y]),
                    np.array([neck_j_x]), np.array([neck_j_y]), scale_cross
                )[0]

                rec["distance_spine_i_tail_j_norm"] = _pairwise_distance(
                    np.array([spine_i_x]), np.array([spine_i_y]),
                    np.array([tail_j_x]), np.array([tail_j_y]), scale_cross
                )[0]

                rec["distance_tail_i_neck_j_norm"] = _pairwise_distance(
                    np.array([tail_i_x]), np.array([tail_i_y]),
                    np.array([neck_j_x]), np.array([neck_j_y]), scale_cross
                )[0]

                rec["distance_tail_i_spine_j_norm"] = _pairwise_distance(
                    np.array([tail_i_x]), np.array([tail_i_y]),
                    np.array([spine_j_x]), np.array([spine_j_y]), scale_cross
                )[0]

                rec["distance_tail_i_tail_j_norm"] = _pairwise_distance(
                    np.array([tail_i_x]), np.array([tail_i_y]),
                    np.array([tail_j_x]), np.array([tail_j_y]), scale_cross
                )[0]

                # F. Reduced contact geometry summaries
                rec["distance_nose_i_partner_body_min_norm"] = min(
                    rec["distance_nose_i_neck_j_norm"],
                    rec["distance_nose_i_spine_j_norm"],
                    rec["distance_nose_i_tail_j_norm"],
                )

                rec["distance_nose_j_focal_body_min_norm"] = min(
                    rec["distance_neck_i_nose_j_norm"],
                    rec["distance_spine_i_nose_j_norm"],
                    rec["distance_tail_i_nose_j_norm"],
                )

                # Body-body min
                body_body_dists = []
                for i_node_x, i_node_y in [(neck_i_x, neck_i_y), (spine_i_x, spine_i_y), (tail_i_x, tail_i_y)]:
                    for j_node_x, j_node_y in [(neck_j_x, neck_j_y), (spine_j_x, spine_j_y), (tail_j_x, tail_j_y)]:
                        d = _pairwise_distance(
                            np.array([i_node_x]), np.array([i_node_y]),
                            np.array([j_node_x]), np.array([j_node_y]), scale_cross
                        )[0]
                        body_body_dists.append(d)

                body_body_arr = np.array(body_body_dists, dtype=float)
                finite_body_body = np.isfinite(body_body_arr)
                if np.any(finite_body_body):
                    rec["distance_body_body_min_norm"] = float(np.nanmin(body_body_arr))
                    rec["distance_body_body_mean_norm"] = float(np.nanmean(body_body_arr))
                else:
                    rec["distance_body_body_min_norm"] = np.nan
                    rec["distance_body_body_mean_norm"] = np.nan

                # H. Behind score
                # Vector from partner to focal
                dx_ji = nose_i_x - nose_j_x
                dy_ji = nose_i_y - nose_j_y

                # Project onto partner heading
                behind_raw = _safe_dot(
                    np.array([dx_ji]), np.array([dy_ji]),
                    np.array([hx_j]), np.array([hy_j])
                )[0]
                rec["behind_score_bl"] = behind_raw / bl_j
                # Partner-normalized behind score (negative of focal score)
                rec["behind_score_bl_j"] = -rec["behind_score_bl"]

                # I. Quality block (matching pipeline 60 features, but at frame level)
                # Focal nose quality
                impute_stage_i = focal["nose_impute_stage"][dense_idx]
                rec["nose_i_is_pchip"] = 1 if impute_stage_i == IMPUTE_STAGE_PCHIP else 0
                rec["nose_i_is_kf"] = 1 if impute_stage_i == IMPUTE_STAGE_KALMAN else 0
                
                # Sigma from point_score (score = 1/(1+sigma) -> sigma = 1/score - 1)
                score_i = focal["nose_point_score"][dense_idx]
                if score_i > 0 and np.isfinite(score_i):
                    rec["nose_i_sigma"] = max(0.0, 1.0 / score_i - 1.0)
                else:
                    rec["nose_i_sigma"] = np.nan
                
                rec["nose_i_was_refined"] = int(focal["nose_was_refined"][dense_idx])
                
                # Partner nose quality
                impute_stage_j = partner["nose_impute_stage"][dense_idx]
                rec["nose_j_is_pchip"] = 1 if impute_stage_j == IMPUTE_STAGE_PCHIP else 0
                rec["nose_j_is_kf"] = 1 if impute_stage_j == IMPUTE_STAGE_KALMAN else 0
                
                # Sigma from point_score
                score_j = partner["nose_point_score"][dense_idx]
                if score_j > 0 and np.isfinite(score_j):
                    rec["nose_j_sigma"] = max(0.0, 1.0 / score_j - 1.0)
                else:
                    rec["nose_j_sigma"] = np.nan
                
                rec["nose_j_was_refined"] = int(partner["nose_was_refined"][dense_idx])

                # Tail quality
                tail_impute_stage_i = focal["tail_impute_stage"][dense_idx]
                rec["tail_i_is_pchip"] = 1 if tail_impute_stage_i == IMPUTE_STAGE_PCHIP else 0
                rec["tail_i_is_kf"] = 1 if tail_impute_stage_i == IMPUTE_STAGE_KALMAN else 0

                tail_impute_stage_j = partner["tail_impute_stage"][dense_idx]
                rec["tail_j_is_pchip"] = 1 if tail_impute_stage_j == IMPUTE_STAGE_PCHIP else 0
                rec["tail_j_is_kf"] = 1 if tail_impute_stage_j == IMPUTE_STAGE_KALMAN else 0

                # Neck quality
                neck_impute_stage_i = focal["neck_impute_stage"][dense_idx]
                rec["neck_i_is_pchip"] = 1 if neck_impute_stage_i == IMPUTE_STAGE_PCHIP else 0
                rec["neck_i_is_kf"] = 1 if neck_impute_stage_i == IMPUTE_STAGE_KALMAN else 0

                neck_impute_stage_j = partner["neck_impute_stage"][dense_idx]
                rec["neck_j_is_pchip"] = 1 if neck_impute_stage_j == IMPUTE_STAGE_PCHIP else 0
                rec["neck_j_is_kf"] = 1 if neck_impute_stage_j == IMPUTE_STAGE_KALMAN else 0

                # Spine quality
                spine_impute_stage_i = focal["spine_impute_stage"][dense_idx]
                rec["spine_i_is_pchip"] = 1 if spine_impute_stage_i == IMPUTE_STAGE_PCHIP else 0
                rec["spine_i_is_kf"] = 1 if spine_impute_stage_i == IMPUTE_STAGE_KALMAN else 0

                spine_impute_stage_j = partner["spine_impute_stage"][dense_idx]
                rec["spine_j_is_pchip"] = 1 if spine_impute_stage_j == IMPUTE_STAGE_PCHIP else 0
                rec["spine_j_is_kf"] = 1 if spine_impute_stage_j == IMPUTE_STAGE_KALMAN else 0

                # J. CONTACT-v3 derived features
                required_v3_base_cols = [
                    "distance_nose_i_partner_body_min_norm",
                    "distance_nose_j_focal_body_min_norm",
                    "distance_nose_i_nose_j_norm",
                    "distance_body_body_mean_norm",
                    "distance_body_body_min_norm",
                    "nose_i_sigma",
                    "nose_j_sigma",
                    "nose_i_was_refined",
                    "nose_j_was_refined",
                    "nose_i_is_pchip",
                    "nose_i_is_kf",
                    "nose_j_is_pchip",
                    "nose_j_is_kf",
                ]
                missing_v3_cols = [c for c in required_v3_base_cols if c not in rec]
                if missing_v3_cols:
                    raise ValueError(
                        "Missing required component columns for CONTACT-v3 derived features: "
                        f"{missing_v3_cols}"
                    )

                # Asymmetric-contact summary block
                rec["contact_asymmetry_min_norm"] = min(
                    rec["distance_nose_i_partner_body_min_norm"],
                    rec["distance_nose_j_focal_body_min_norm"],
                )
                rec["contact_asymmetry_diff_norm"] = abs(
                    rec["distance_nose_i_partner_body_min_norm"]
                    - rec["distance_nose_j_focal_body_min_norm"]
                )
                rec["nose_contact_best_norm"] = min(
                    rec["distance_nose_i_nose_j_norm"],
                    rec["distance_nose_i_partner_body_min_norm"],
                    rec["distance_nose_j_focal_body_min_norm"],
                )

                # Body-vs-nose separation block
                rec["body_contact_gap_norm"] = (
                    rec["distance_body_body_mean_norm"] - rec["distance_body_body_min_norm"]
                )
                rec["nose_vs_body_gap_i_norm"] = (
                    rec["distance_nose_i_partner_body_min_norm"] - rec["distance_body_body_min_norm"]
                )
                rec["nose_vs_body_gap_j_norm"] = (
                    rec["distance_nose_j_focal_body_min_norm"] - rec["distance_body_body_min_norm"]
                )

                # Nose-quality summary block
                rec["nose_sigma_max"] = _nanmax2(
                    rec["nose_i_sigma"], rec["nose_j_sigma"]
                )
                rec["nose_sigma_min"] = _nanmin2(
                    rec["nose_i_sigma"], rec["nose_j_sigma"]
                )
                rec["nose_any_refined"] = bool(
                    rec["nose_i_was_refined"] or rec["nose_j_was_refined"]
                )
                rec["nose_any_imputed_i"] = int(bool(
                    rec["nose_i_is_pchip"]
                    or rec["nose_i_is_kf"]
                ))
                rec["nose_any_imputed_j"] = int(bool(
                    rec["nose_j_is_pchip"]
                    or rec["nose_j_is_kf"]
                ))
                rec["tail_any_imputed_i"] = int(bool(
                    rec["tail_i_is_pchip"]
                    or rec["tail_i_is_kf"]
                ))
                rec["tail_any_imputed_j"] = int(bool(
                    rec["tail_j_is_pchip"]
                    or rec["tail_j_is_kf"]
                ))
                rec["neck_any_imputed_i"] = int(bool(
                    rec["neck_i_is_pchip"]
                    or rec["neck_i_is_kf"]
                ))
                rec["neck_any_imputed_j"] = int(bool(
                    rec["neck_j_is_pchip"]
                    or rec["neck_j_is_kf"]
                ))
                rec["spine_any_imputed_i"] = int(bool(
                    rec["spine_i_is_pchip"]
                    or rec["spine_i_is_kf"]
                ))
                rec["spine_any_imputed_j"] = int(bool(
                    rec["spine_j_is_pchip"]
                    or rec["spine_j_is_kf"]
                ))

                records.append(rec)

    if dyad_scales:
        dyad_scales_arr = np.array(dyad_scales, dtype=float)
        print(
            "  -> dyad_body_scale summary: "
            f"min={float(np.nanmin(dyad_scales_arr)):.3f}, "
            f"median={float(np.nanmedian(dyad_scales_arr)):.3f}, "
            f"max={float(np.nanmax(dyad_scales_arr)):.3f}"
        )
    else:
        print(
            "  -> dyad_body_scale summary: no finite dyad scales "
            "(all cross-mouse distances will be NaN)"
        )

    # ========================================================================
    # 5. Build DataFrame
    # ========================================================================
    features_df = pd.DataFrame(records)

    if features_df.empty:
        print(f"Warning: No dyad-frame combinations generated for {exp}/{iteration_name}")
        return FrameFeatureResult(
            features_df=features_df,
            summary=FrameFeatureSummary(
                n_rows=0,
                n_dyads=0,
                n_dense_frames_unique=0,
                missingness={},
                feature_quantiles={},
            ),
        )

    # Validate unique keys
    key_cols = ["exp_name", "iteration_name", "focal_id", "partner_id", "dense_frame_idx"]
    duplicates = features_df.duplicated(subset=key_cols, keep=False)

    if duplicates.sum() > 0:
        dup_preview = features_df[duplicates].head(10)
        raise ValueError(
            f"Found {duplicates.sum()} duplicate keys in frame features.\n"
            f"Preview:\n{dup_preview}"
        )

    # Sort
    features_df = features_df.sort_values(
        ["focal_id", "partner_id", "dense_frame_idx"]
    ).reset_index(drop=True)

    # Self-healing dynamic side polyline distance (normalized by dyad body scale).
    dyad_scale_arr = pd.to_numeric(features_df["dyad_body_scale"], errors="coerce").to_numpy(
        dtype=float, copy=False
    )
    dyad_scale_safe = np.maximum(dyad_scale_arr, EPS)
    invalid_scale = ~np.isfinite(dyad_scale_arr) | (dyad_scale_arr <= 0)

    dist_nose_i_side_j = calculate_self_healing_side_distance(
        features_df,
        prefix_i="i",
        prefix_j="j",
    )
    dist_side_i_nose_j = calculate_self_healing_side_distance(
        features_df,
        prefix_i="j",
        prefix_j="i",
    )
    dist_headstage_i_side_j = calculate_self_healing_side_distance(
        features_df,
        prefix_i="i",
        prefix_j="j",
        origin_node="headstage",
    )

    features_df["distance_nose_i_side_j_norm"] = dist_nose_i_side_j / dyad_scale_safe
    features_df["distance_side_i_nose_j_norm"] = dist_side_i_nose_j / dyad_scale_safe
    features_df["distance_headstage_i_side_j_norm"] = dist_headstage_i_side_j / dyad_scale_safe

    if np.any(invalid_scale):
        features_df.loc[invalid_scale, "distance_nose_i_side_j_norm"] = np.nan
        features_df.loc[invalid_scale, "distance_side_i_nose_j_norm"] = np.nan
        features_df.loc[invalid_scale, "distance_headstage_i_side_j_norm"] = np.nan

    # Backward-compatible alias for existing downstream consumers.
    features_df["distance_nose_j_side_i_norm"] = features_df["distance_side_i_nose_j_norm"]

    features_df = _add_body_only_fallback_features(features_df)
    features_df = _add_side_by_side_geometry_features(features_df)
    features_df = _add_primitive_distance_missingness_features(features_df)
    _print_contact_robustness_feature_summary(features_df)
    _print_headstage_feature_summary(features_df)

    # Robust nose distances for implanted occlusion fallback.
    # Legacy hardcoded focal/partner ids 0/1; Sprint 2 uses AdaptedPose implantation.
    # When implantation is unresolved, do not invent a fallback — leave NaNs.
    if implanted_track_id is not None:
        implanted_str = str(implanted_track_id)
        involves_implanted = (
            features_df["focal_id"].astype(str).eq(implanted_str)
            | features_df["partner_id"].astype(str).eq(implanted_str)
        )
    else:
        involves_implanted = pd.Series(False, index=features_df.index)

    robust_distance_map = [
        (
            "distance_nose_i_tail_j_norm",
            "distance_headstage_i_tail_j_norm",
            "distance_nose_robust_tail_j_norm",
        ),
        (
            "distance_nose_i_nose_j_norm",
            "distance_headstage_i_nose_j_norm",
            "distance_nose_robust_nose_j_norm",
        ),
        (
            "distance_nose_i_side_j_norm",
            "distance_headstage_i_side_j_norm",
            "distance_nose_robust_side_j_norm",
        ),
    ]

    for raw_col, headstage_col, robust_col in robust_distance_map:
        if raw_col not in features_df.columns:
            raise ValueError(f"Missing raw nose-distance column required for robustness: {raw_col}")
        if headstage_col not in features_df.columns:
            raise ValueError(
                f"Missing headstage distance column required for robustness: {headstage_col}"
            )

        robust_values = pd.to_numeric(features_df[raw_col], errors="coerce").copy()
        headstage_values = pd.to_numeric(features_df[headstage_col], errors="coerce")

        missing_mask = robust_values.isna() & involves_implanted
        robust_values.loc[missing_mask] = (headstage_values.loc[missing_mask] - 0.05).clip(lower=0.0)
        features_df[robust_col] = robust_values

    robust_grouped = features_df.groupby(["exp_name", "focal_id", "partner_id"])
    for _, _, robust_col in robust_distance_map:
        features_df[robust_col] = robust_grouped[robust_col].transform(
            lambda x: x.rolling(window=7, center=True, min_periods=1).mean()
        )

    # ------------------------------------------------------------------
    # Robust-distance radial velocities
    # Compute first derivatives of the robust nose-distance measures
    # (d(distance_nose_robust_*_norm) / dt) grouped by dyad.
    # ------------------------------------------------------------------
    robust_vel_map = [
        ("distance_nose_robust_tail_j_norm", "vel_radial_nose_robust_tail_j"),
        ("distance_nose_robust_nose_j_norm", "vel_radial_nose_robust_nose_j"),
        ("distance_nose_robust_side_j_norm", "vel_radial_nose_robust_side_j"),
    ]

    dyad_group = ["exp_name", "focal_id", "partner_id"]
    for dist_col, vel_col in robust_vel_map:
        if dist_col not in features_df.columns:
            features_df[vel_col] = np.nan
            continue

        dist_vals = pd.to_numeric(features_df[dist_col], errors="coerce")
        prev_dist = features_df.groupby(dyad_group)[dist_col].shift(diff_dt)
        dt_series = features_df["t_120"] - features_df.groupby(dyad_group)["t_120"].shift(diff_dt)

        vel = (dist_vals - prev_dist) / dt_series

        # Invalidate where inputs are not finite or dt <= 0
        invalid_mask = ~(
            np.isfinite(dist_vals) & np.isfinite(prev_dist) & np.isfinite(dt_series) & (dt_series > 0)
        )
        vel[invalid_mask] = np.nan

        features_df[vel_col] = vel

    # ========================================================================
    # 5.5 Add Rolling Window Features
    # ========================================================================
    window_size = 15
    grouped = features_df.groupby(["exp_name", "focal_id", "partner_id"])

    # Rolling mean features
    features_df["distance_nose_i_nose_j_norm_rolling_mean"] = grouped[
        "distance_nose_i_nose_j_norm"
    ].transform(lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean())

    features_df["d_nose_nose_norm_rolling_mean"] = grouped[
        "d_nose_nose_norm"
    ].transform(lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean())

    features_df["cos_heading_alignment_rolling_mean"] = grouped[
        "cos_heading_alignment"
    ].transform(lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean())

    features_df["distance_neck_i_neck_j_norm_rolling_mean"] = grouped[
        "distance_neck_i_neck_j_norm"
    ].transform(lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean())

    features_df["distance_nose_i_neck_j_norm_rolling_mean"] = grouped[
        "distance_nose_i_neck_j_norm"
    ].transform(lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean())

    features_df["distance_nose_i_spine_j_norm_rolling_mean"] = grouped[
        "distance_nose_i_spine_j_norm"
    ].transform(lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean())

    features_df["distance_nose_i_tail_j_norm_rolling_mean"] = grouped[
        "distance_nose_i_tail_j_norm"
    ].transform(lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean())
    features_df["distance_spine_i_nose_j_norm_rolling_mean"] = grouped[
        "distance_spine_i_nose_j_norm"
    ].transform(lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean())
    features_df["distance_tail_i_nose_j_norm_rolling_mean"] = grouped[
        "distance_tail_i_nose_j_norm"
    ].transform(lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean())

    features_df["distance_neck_i_nose_j_norm_rolling_mean"] = grouped[
        "distance_neck_i_nose_j_norm"
    ].transform(lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean())

    features_df["distance_neck_i_spine_j_norm_rolling_mean"] = grouped[
        "distance_neck_i_spine_j_norm"
    ].transform(lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean())

    features_df["distance_neck_i_tail_j_norm_rolling_mean"] = grouped[
        "distance_neck_i_tail_j_norm"
    ].transform(lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean())
    features_df["distance_spine_i_neck_j_norm_rolling_mean"] = grouped[
        "distance_spine_i_neck_j_norm"
    ].transform(lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean())
    features_df["distance_tail_i_neck_j_norm_rolling_mean"] = grouped[
        "distance_tail_i_neck_j_norm"
    ].transform(lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean())

    features_df["cos_snout_alignment_rolling_mean"] = grouped[
        "cos_snout_alignment"
    ].transform(lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean())

    features_df["cos_hs_i_to_nose_j_rolling_mean"] = grouped[
        "cos_hs_i_to_nose_j"
    ].transform(lambda x: x.rolling(window=window_size, center=True, min_periods=1).mean())

    # Rolling std features
    features_df["distance_nose_i_nose_j_norm_rolling_std"] = grouped[
        "distance_nose_i_nose_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size, center=True, min_periods=1).std().fillna(0)
    )

    # ===== Relational Physics: 60-frame Rolling Entropy Signals =====
    # Rolling std for radial velocity (rate of change variability)
    window_size_60 = 60
    features_df["rolling_std_vel_radial_spine_60"] = grouped[
        "vel_radial_spine"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).std().fillna(0)
    )
    
    # Rolling std for projected velocity (intent/motivation variability)
    features_df["rolling_std_projected_vel_i_to_j_60"] = grouped[
        "projected_vel_i_to_j"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).std().fillna(0)
    )
    features_df["rolling_std_projected_vel_j_to_i_60"] = grouped[
        "projected_vel_j_to_i"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).std().fillna(0)
    )
    features_df["rolling_mean_projected_vel_i_to_j_60"] = grouped[
        "projected_vel_i_to_j"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).mean().bfill().ffill()
    )
    features_df["rolling_mean_projected_vel_j_to_i_60"] = grouped[
        "projected_vel_j_to_i"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).mean().bfill().ffill()
    )
    features_df["rolling_mean_vel_radial_spine_60"] = grouped[
        "vel_radial_spine"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).mean().bfill().ffill()
    )
    features_df["rolling_speed_i_60"] = grouped[
        "speed_i"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).mean().bfill().ffill()
    )
    features_df["rolling_speed_j_60"] = grouped[
        "speed_j"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).mean().bfill().ffill()
    )
    features_df["rolling_speed_i_15"] = grouped[
        "speed_i"
    ].transform(
        lambda x: x.rolling(window=15, center=True, min_periods=1).mean().bfill().ffill()
    )
    features_df["rolling_speed_j_15"] = grouped[
        "speed_j"
    ].transform(
        lambda x: x.rolling(window=15, center=True, min_periods=1).mean().bfill().ffill()
    )
    
    features_df["rolling_mean_cos_heading_i_to_spine_j_60"] = grouped[
        "cos_heading_i_to_spine_j"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).mean().bfill().ffill()
    )
    features_df["rolling_mean_cos_heading_i_to_tail_j_60"] = grouped[
        "cos_heading_i_to_tail_j"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).mean().bfill().ffill()
    )

    # ===== Physics Delta: Acceleration & Synchrony =====
    # Radial acceleration: frame-to-frame difference of vel_radial_spine.
    features_df["accel_radial_spine"] = grouped["vel_radial_spine"].diff().bfill()

    # Angular closing rate: frame-to-frame derivative of heading angle to partner
    # Negative values indicate focal mouse is orienting toward partner
    features_df["angular_closing_rate"] = grouped["heading_angle_to_j"].transform(
        lambda x: x.diff().fillna(0.0)
    )

    # Weighted rotation: angular closing rate scaled by partner speed
    # Captures intent to follow a moving target: is focal reorienting while partner moves?
    features_df["weighted_rotation"] = (
        features_df["angular_closing_rate"] * features_df["rolling_speed_j_60"]
    ).fillna(0.0)

    # Movement asymmetry signals.
    features_df["velocity_mag_diff"] = (features_df["v_i_bl"] - features_df["v_j_bl"]).abs()
    features_df["compactness_diff"] = (
        features_df["compactness_i"] - features_df["compactness_j"]
    ).abs()
    v_max = np.maximum(features_df["v_i_bl"].to_numpy(), features_df["v_j_bl"].to_numpy())
    v_min = np.minimum(features_df["v_i_bl"].to_numpy(), features_df["v_j_bl"].to_numpy())
    features_df["relative_speed_ratio"] = v_min / (v_max + EPS)

    # Speed correlation: centered 60-frame rolling Pearson correlation of speed_i and speed_j.
    # Undefined windows (insufficient pairs / zero variance) and Inf/out-of-range
    # numeric artifacts become NaN — never Inf, and never neighbor-filled.
    features_df["speed_correlation_60"] = sanitize_pearson_correlation(
        grouped["speed_i"].transform(
            lambda x: x.rolling(window=window_size_60, center=True, min_periods=2).corr(
                features_df.loc[x.index, "speed_j"]
            )
        )
    )

    # Synchrony entropy: rolling angular-velocity coupling across the dyad.
    features_df["cross_velocity_correlation"] = grouped["angular_velocity_i"].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=2).corr(
            features_df.loc[x.index, "angular_velocity_j"]
        )
    ).fillna(0.0)

    # Angular stability: rolling std of heading alignment (chaotic movement = high std)
    features_df["rolling_std_cos_heading_60"] = grouped[
        "cos_heading_alignment"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).std().fillna(0)
    )

    # High-resolution entropy of per-mouse rotational dynamics.
    features_df["angular_velocity_mean"] = (
        features_df["angular_velocity_i"].abs() + features_df["angular_velocity_j"].abs()
    ) / 2.0
    features_df["rolling_std_angular_velocity_60"] = grouped[
        "angular_velocity_mean"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).std().fillna(0)
    )

    features_df["rolling_std_cos_bearing_60"] = grouped[
        "cos_bearing_i_to_j"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).std().fillna(0)
    )
    features_df["rolling_std_cos_bearing_j_to_i_60"] = grouped[
        "cos_bearing_j_to_i"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).std().fillna(0)
    )

    features_df["rolling_mean_mutual_facing_60"] = grouped[
        "mutual_facing"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).mean().bfill().ffill()
    )

    features_df["rolling_std_mutual_facing_60"] = grouped[
        "mutual_facing"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).std().bfill().ffill()
    )

    features_df["rolling_std_rel_speed_ratio_60"] = grouped[
        "relative_speed_ratio"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).std().fillna(0)
    )

    # Temporal stasis: rolling count of low-velocity frames (< 0.5 units/s).
    features_df["velocity_low_duration_i"] = grouped["speed_i"].transform(
        lambda x: pd.Series(np.where(np.isfinite(x) & (x < 0.5), 1.0, 0.0), index=x.index)
        .rolling(window=window_size_60, center=True, min_periods=1)
        .sum()
    )
    features_df["velocity_low_duration_j"] = grouped["speed_j"].transform(
        lambda x: pd.Series(np.where(np.isfinite(x) & (x < 0.5), 1.0, 0.0), index=x.index)
        .rolling(window=window_size_60, center=True, min_periods=1)
        .sum()
    )

    # Geometric ratio: nose-to-spine distance ratio (huddle vs greeting)
    # Add small epsilon to avoid division by zero
    features_df["nose_to_spine_ratio"] = features_df["distance_nose_i_nose_j_norm"] / (
        features_df["distance_spine_i_spine_j_norm"] + EPS
    )
    features_df["log_dist_ratio_nose_spine"] = np.log10(
        features_df["distance_nose_i_nose_j_norm"]
        / (features_df["distance_spine_i_spine_j_norm"] + EPS)
    )

    # ===== 3-Second (360-Frame) Memory Features =====
    # These features use centered windows to capture local context around each frame.
    window_size_360 = 360

    # Historical Minimums: Did they touch recently?
    features_df["rolling_min_dist_spine_360"] = grouped[
        "distance_spine_i_spine_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).min().bfill().ffill()
    )

    features_df["rolling_min_dist_nose_360"] = grouped[
        "distance_nose_i_nose_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_neck_360"] = grouped[
        "distance_neck_i_neck_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_tail_360"] = grouped[
        "distance_tail_i_tail_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_nose_i_neck_j_360"] = grouped[
        "distance_nose_i_neck_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_nose_i_spine_j_360"] = grouped[
        "distance_nose_i_spine_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_nose_i_tail_j_360"] = grouped[
        "distance_nose_i_tail_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_neck_i_nose_j_360"] = grouped[
        "distance_neck_i_nose_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_neck_i_spine_j_360"] = grouped[
        "distance_neck_i_spine_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_neck_i_tail_j_360"] = grouped[
        "distance_neck_i_tail_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_spine_i_nose_j_360"] = grouped[
        "distance_spine_i_nose_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_spine_i_neck_j_360"] = grouped[
        "distance_spine_i_neck_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_spine_i_tail_j_360"] = grouped[
        "distance_spine_i_tail_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_tail_i_nose_j_360"] = grouped[
        "distance_tail_i_neck_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_tail_i_neck_j_360"] = grouped[
        "distance_tail_i_neck_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_tail_i_spine_j_360"] = grouped[
        "distance_tail_i_spine_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_headstage_i_nose_j_360"] = grouped[
        "distance_headstage_i_nose_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_headstage_i_neck_j_360"] = grouped[
        "distance_headstage_i_neck_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_headstage_i_spine_j_360"] = grouped[
        "distance_headstage_i_spine_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_headstage_i_tail_j_360"] = grouped[
        "distance_headstage_i_tail_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["_nose_i_body_dist_j"] = features_df[
        ["distance_nose_i_neck_j_norm", "distance_nose_i_spine_j_norm"]
    ].min(axis=1, skipna=True)
    features_df["rolling_min_dist_nose_i_body_j_360"] = grouped[
        "_nose_i_body_dist_j"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["_body_i_nose_j_dist"] = features_df[
        ["distance_neck_i_nose_j_norm", "distance_spine_i_nose_j_norm"]
    ].min(axis=1, skipna=True)
    features_df["rolling_min_dist_body_i_nose_j_360"] = grouped[
        "_body_i_nose_j_dist"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df = features_df.drop(columns=["_nose_i_body_dist_j", "_body_i_nose_j_dist"])





    # =====  (15-Frame) Memory Features =====
    window_size_15 = 15
    features_df["rolling_min_dist_nose_i_nose_j_15"] = grouped[
        "distance_nose_i_nose_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_15, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_mean_dist_nose_i_nose_j_15"] = grouped[
        "distance_nose_i_nose_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_15, center=True, min_periods=1).mean().bfill().ffill()
    )
    features_df["rolling_min_dist_nose_i_side_j_15"] = grouped[
        "distance_nose_i_side_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_15, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_mean_dist_nose_i_side_j_15"] = grouped[
        "distance_nose_i_side_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_15, center=True, min_periods=1).mean().bfill().ffill()
    )
    features_df["rolling_min_dist_side_i_nose_j_15"] = grouped[
        "distance_side_i_nose_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_15, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_nose_i_tail_j_15"] = grouped[
        "distance_nose_i_tail_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_15, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_mean_dist_nose_i_tail_j_15"] = grouped[
        "distance_nose_i_tail_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_15, center=True, min_periods=1).mean().bfill().ffill()
    )
    features_df["rolling_min_dist_tail_i_nose_j_15"] = grouped[
        "distance_tail_i_nose_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_15, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_nose_i_neck_j_15"] = grouped[
        "distance_nose_i_neck_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_15, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_neck_i_nose_j_15"] = grouped[
        "distance_neck_i_nose_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_15, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_spine_i_spine_j_15"] = grouped[
        "distance_spine_i_spine_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_15, center=True, min_periods=1).min().bfill().ffill()
    )
    # add rolling min for nose-spine
    features_df["rolling_min_dist_nose_i_spine_j_15"] = grouped[
        "distance_nose_i_spine_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_15, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_mean_dist_nose_i_spine_j_15"] = grouped[
        "distance_nose_i_spine_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_15, center=True, min_periods=1).mean().bfill().ffill()
    )
    features_df["rolling_min_dist_spine_i_nose_j_15"] = grouped[
        "distance_spine_i_nose_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_15, center=True, min_periods=1).min().bfill().ffill()
    )
    # ===== (30-Frame) Memory Features =====
    window_size_30 = 30
    features_df["rolling_min_dist_nose_i_side_j_30"] = grouped[
        "distance_nose_i_side_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_30, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_mean_dist_nose_i_side_j_30"] = grouped[
        "distance_nose_i_side_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_30, center=True, min_periods=1).mean().bfill().ffill()
    )
    features_df["rolling_min_dist_nose_i_nose_j_30"] = grouped[
        "distance_nose_i_nose_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_30, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_mean_dist_nose_i_nose_j_30"] = grouped[
        "distance_nose_i_nose_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_30, center=True, min_periods=1).mean().bfill().ffill()
    )
    features_df["rolling_min_dist_nose_i_tail_j_30"] = grouped[
        "distance_nose_i_tail_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_30, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_mean_dist_nose_i_tail_j_30"] = grouped[
        "distance_nose_i_tail_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_30, center=True, min_periods=1).mean().bfill().ffill()
    )
    features_df["rolling_min_dist_nose_i_spine_j_30"] = grouped[
        "distance_nose_i_spine_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_30, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_mean_dist_nose_i_spine_j_30"] = grouped[
        "distance_nose_i_spine_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_30, center=True, min_periods=1).mean().bfill().ffill()
    )
    # =====  (60-Frame) Memory Features =====
    window_size_60 = 60
    features_df["rolling_min_dist_nose_i_nose_j_60"] = grouped[
        "distance_nose_i_nose_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_mean_dist_nose_i_nose_j_60"] = grouped[
        "distance_nose_i_nose_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).mean().bfill().ffill()
    )
    features_df["rolling_min_dist_nose_i_side_j_60"] = grouped[
        "distance_nose_i_side_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_mean_dist_nose_i_side_j_60"] = grouped[
        "distance_nose_i_side_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).mean().bfill().ffill()
    )
    features_df["rolling_min_dist_side_i_nose_j_60"] = grouped[
        "distance_side_i_nose_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_nose_i_tail_j_60"] = grouped[
        "distance_nose_i_tail_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_mean_dist_nose_i_tail_j_60"] = grouped[
        "distance_nose_i_tail_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).mean().bfill().ffill()
    )
    features_df["rolling_min_dist_tail_i_nose_j_60"] = grouped[
        "distance_tail_i_nose_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_spine_i_spine_j_60"] = grouped[
        "distance_spine_i_spine_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).min().bfill().ffill()
    )
    # add rolling min for nose-spine
    features_df["rolling_min_dist_nose_i_spine_j_60"] = grouped[
        "distance_nose_i_spine_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_mean_dist_nose_i_spine_j_60"] = grouped[
        "distance_nose_i_spine_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).mean().bfill().ffill()
    )
    features_df["rolling_min_dist_spine_i_nose_j_60"] = grouped[
        "distance_spine_i_nose_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_nose_i_neck_j_60"] = grouped[
        "distance_nose_i_neck_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_dist_neck_i_nose_j_60"] = grouped[
        "distance_neck_i_nose_j_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_min_nose_contact_dist_60"] = grouped[
        "nose_contact_best_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).min().bfill().ffill()
    )
    features_df["rolling_mean_nose_contact_dist_60"] = grouped[
        "nose_contact_best_norm"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).mean().bfill().ffill()
    )
    features_df["rolling_mean_heading_to_partner_i_60"] = grouped[
        "heading_to_partner_i"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).mean().bfill().ffill()
    )
    features_df["rolling_mean_heading_to_partner_j_60"] = grouped[
        "heading_to_partner_j"
    ].transform(
        lambda x: x.rolling(window=window_size_60, center=True, min_periods=1).mean().bfill().ffill()
    )

    # Historical Averages: General State over the 3-second window
    features_df["rolling_mean_heading_to_partner_i_360"] = grouped[
        "heading_to_partner_i"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).mean().bfill().ffill()
    )
    features_df["rolling_mean_heading_to_partner_j_360"] = grouped[
        "heading_to_partner_j"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).mean().bfill().ffill()
    )
    features_df["rolling_mean_cos_heading_360"] = grouped[
        "cos_heading_alignment"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).mean().bfill().ffill()
    )

    features_df["rolling_mean_speed_diff_360"] = grouped[
        "velocity_mag_diff"
    ].transform(
        lambda x: x.rolling(window=window_size_360, center=True, min_periods=1).mean().bfill().ffill()
    )

    # A bool indicator for whether nose is closer to tail than spine. True also when nose_tail is below a threshold (0.7) gets 1 if (rolling_min_dist_nose_i_tail_j_60 < rolling_min_dist_nose_i_side_j_60) or (rolling_min_dist_nose_i_tail_j_15 < rolling_min_dist_nose_i_side_j_15) or (rolling_min_dist_nose_i_tail_j_60 < rolling_min_dist_nose_i_spine_j_60) or (rolling_min_dist_nose_i_tail_j_15 < rolling_min_dist_nose_i_spine_j_15), else 0
    features_df["nose_closer_to_tail"] = ((
        (features_df["rolling_min_dist_nose_i_tail_j_60"] < features_df["rolling_min_dist_nose_i_side_j_60"])
        | (features_df["rolling_min_dist_nose_i_tail_j_15"] < features_df["rolling_min_dist_nose_i_side_j_15"])
        | (features_df["rolling_min_dist_nose_i_tail_j_60"] < features_df["rolling_min_dist_nose_i_spine_j_60"])
        | (features_df["rolling_min_dist_nose_i_tail_j_15"] < features_df["rolling_min_dist_nose_i_spine_j_15"])
    ) & (features_df["distance_nose_robust_tail_j_norm"] < 0.7) & (features_df["behind_score_bl"] < -1.5) & (features_df["rolling_min_dist_nose_i_tail_j_60"] < 0.3)).astype(float)
    # set to Nan if conditions are NaN
    features_df.loc[features_df["nose_closer_to_tail"].isnull(), "nose_closer_to_tail"] = np.nan
    # A bool indicator for whether nose is closer to nose than neck.
    features_df["nose_closer_to_nose"] = ((
        (features_df["rolling_min_dist_nose_i_nose_j_60"] < features_df["rolling_min_dist_nose_i_side_j_60"])
        | (features_df["rolling_min_dist_nose_i_nose_j_15"] < features_df["rolling_min_dist_nose_i_side_j_15"])
        | (features_df["rolling_min_dist_nose_i_nose_j_60"] < features_df["rolling_min_dist_nose_i_neck_j_60"])
        | (features_df["rolling_min_dist_nose_i_nose_j_15"] < features_df["rolling_min_dist_nose_i_neck_j_15"])
    ) & (features_df["distance_nose_robust_nose_j_norm"] < 0.7)  & (features_df["behind_score_bl"] > - 0.5) & (features_df["rolling_min_dist_nose_i_nose_j_60"] < 0.4)).astype(float)
    # set to Nan if conditions are NaN
    features_df.loc[features_df["nose_closer_to_nose"].isnull(), "nose_closer_to_nose"] = np.nan
    # A bool indicator for whether nose is closer to side than both nose and tail. So it is true when both flags are false and distance_nose_robust_side_j_norm is less than 0.7 and rolling_min_dist_nose_i_side_j_60 is less than 0.5, else false. Also set to NaN if conditions are NaN
    features_df["nose_closer_to_side"] = ((features_df["nose_closer_to_tail"] == 0) & (features_df["nose_closer_to_nose"] == 0) & (features_df["distance_nose_robust_side_j_norm"] < 0.7) & (features_df["rolling_min_dist_nose_i_side_j_60"] < 0.5)).astype(float)
    features_df.loc[features_df["nose_closer_to_side"].isnull(), "nose_closer_to_side"] = np.nan
    # ========================================================================
    # SOTA Kinematic Proofs (Sweep, Contraction, Dominance)
    # ========================================================================
    
    # 1. The Sweep (Bearing Correction Rate)
    # How fast is the core body sweeping to face the partner?
    features_df["bearing_correction_rate_15"] = (
        features_df["cos_bearing_i_to_j"] - 
        grouped["cos_bearing_i_to_j"].shift(window_size_15)
    ).fillna(0.0)

    # 2. The Flinch Veto (Robust Body Contraction Rate)
    # Uses Nose for maximum flinch sensitivity, seamlessly falling back to Neck if occluded
    front_x_i = np.where(
        np.isfinite(features_df["nose_x_i"]), 
        features_df["nose_x_i"], 
        features_df["neck_x_i"]
    )
    front_y_i = np.where(
        np.isfinite(features_df["nose_y_i"]), 
        features_df["nose_y_i"], 
        features_df["neck_y_i"]
    )

    features_df["robust_front_to_spine_i"] = np.sqrt(
        (front_x_i - features_df["spine_x_i"])**2 + 
        (front_y_i - features_df["spine_y_i"])**2
    )

    features_df["body_contraction_rate_7"] = (
        features_df.groupby(["exp_name", "focal_id", "partner_id"])["robust_front_to_spine_i"].shift(7)
        - features_df["robust_front_to_spine_i"]
    ).fillna(0.0)
    features_df["body_extension_rate_7"] = -features_df.get("body_contraction_rate_7", pd.Series([0.0]*len(features_df)))
    # 3. The Passive Receiver Veto (Effort Ratio)
    # What percentage of the dyad's total kinetic energy belongs to the focal mouse?
    features_df["focal_effort_ratio_15"] = features_df["rolling_speed_i_15"] / (
        features_df["rolling_speed_i_15"] + features_df["rolling_speed_j_15"] + EPS
    )
    features_df["spine_displacement_15_i"] = np.sqrt(
        (features_df["spine_x_i"] - grouped["spine_x_i"].shift(window_size_15))**2 +
        (features_df["spine_y_i"] - grouped["spine_y_i"].shift(window_size_15))**2
    ).fillna(0.0)
    
    # Convert 15-frame displacement to pixels/sec (assuming 120fps)
    features_df["spine_directed_vel_15_i"] = features_df["spine_displacement_15_i"] * (120.0 / 15.0)
    

    # ========================================================================
    # 6. Summarize
    # ========================================================================
    feature_cols_to_summarize = [
        "body_extension_rate_7",
        "spine_directed_vel_15_i",
        "bearing_correction_rate_15",
        "body_contraction_rate_7",
        "focal_effort_ratio_15",
        "rolling_mean_intent_15",
        "d_nose_nose_norm",
        "ddist_dt",
        "abs_ddist_dt",
        "v_i_bl",
        "v_j_bl",
        "speed_i",
        "speed_j",
        "v_rel_bl",
        "rel_speed_diff",
        "angular_velocity_i",
        "angular_velocity_j",
        "cos_bearing_i_to_j",
        "cos_bearing_j_to_i",
        "bearing_while_close",
        "bearing_while_close_j_to_i",
        "v_rad_i_bl",
        "cos_heading_i_to_j",
        "cos_heading_j_to_i",
        "cos_heading_i_to_spine_j",
        "cos_heading_j_to_spine_i",
        "cos_heading_i_to_tail_j",
        "cos_heading_j_to_tail_i",
        "heading_angle_to_j",
        "heading_angle_to_i",
        "cos_heading_alignment",
        "cos_snout_alignment",
        "cos_hs_i_to_nose_j",
        "distance_nose_i_nose_j_norm",
        "distance_nose_i_partner_body_min_norm",
        "distance_nose_j_focal_body_min_norm",
        "distance_body_body_min_norm",
        "distance_neck_i_neck_j_norm",
        "distance_neck_i_spine_j_norm",
        "distance_neck_i_tail_j_norm",
        "distance_spine_i_neck_j_norm",
        "distance_spine_i_tail_j_norm",
        "distance_tail_i_neck_j_norm",
        "distance_tail_i_spine_j_norm",
        "distance_tail_i_tail_j_norm",
        "distance_body_body_mean_norm",
        "distance_spine_i_spine_j_norm",
        "distance_nose_i_side_j_norm",
        "distance_side_i_nose_j_norm",
        "distance_nose_j_side_i_norm",
        "distance_headstage_i_side_j_norm",
        "distance_nose_robust_tail_j_norm",
        "distance_nose_robust_nose_j_norm",
        "distance_nose_robust_side_j_norm",
        "distance_headstage_i_nose_j_norm",
        "distance_headstage_i_neck_j_norm",
        "distance_headstage_i_spine_j_norm",
        "distance_headstage_i_tail_j_norm",
        "distance_headstage_i_nose_i_norm",
        "distance_headstage_i_neck_i_norm",
        "distance_headstage_i_spine_i_norm",
        "distance_headstage_i_tail_i_norm",
        "headstage_alignment_i",
        "headstage_neck_alignment_i",
        "headstage_offset_from_body_axis_i_norm",
        "headstage_forward_projection_i_norm",
        "behind_score_bl",
        "nose_i_sigma",
        "nose_j_sigma",
        "nose_i_is_pchip",
        "nose_i_is_kf",
        "nose_j_is_pchip",
        "nose_j_is_kf",
        "tail_i_is_pchip",
        "tail_i_is_kf",
        "tail_j_is_pchip",
        "tail_j_is_kf",
        "contact_asymmetry_min_norm",
        "contact_asymmetry_diff_norm",
        "nose_contact_best_norm",
        "body_contact_gap_norm",
        "nose_vs_body_gap_i_norm",
        "nose_vs_body_gap_j_norm",
        "body_only_contact_best_norm",
        "body_only_contact_mean_best3_norm",
        "body_axis_contact_best_norm",
        "tail_rear_contact_best_norm",
        "n_primitive_distance_finite",
        "frac_primitive_distance_missing",
        "all_primitive_distances_missing",
        "body_length_i_norm",
        "body_length_j_norm",
        "chain_length_i_norm",
        "chain_length_j_norm",
        "compactness_i",
        "compactness_j",
        "neck_spine_norm_i",
        "neck_spine_norm_j",
        "spine_tail_norm_i",
        "spine_tail_norm_j",
        "curvature_i",
        "curvature_j",
        "heading_to_partner_i",
        "heading_to_partner_j",
        "mutual_facing",
        # Side-by-side geometry features
        "forward_offset_i_to_j",
        "forward_offset_j_to_i",
        "abs_forward_offset_i_to_j",
        "abs_forward_offset_j_to_i",
        "mean_abs_forward_offset",
        "lateral_offset_i_to_j",
        "lateral_offset_j_to_i",
        "abs_lateral_offset_i_to_j",
        "abs_lateral_offset_j_to_i",
        "mean_abs_lateral_offset",
        "lateral_dominance_i_to_j",
        "lateral_dominance_j_to_i",
        "mean_lateral_dominance",
        "matched_body_axis_distance_norm",
        "crossed_body_axis_distance_norm",
        "body_axis_symmetry",
        "body_axis_symmetry_ratio",
        # side-side end
        "nose_sigma_max",
        "nose_sigma_min",
        "nose_any_refined",
        "nose_any_imputed_i",
        "nose_any_imputed_j",
        "tail_any_imputed_i",
        "tail_any_imputed_j",
        # Relational Physics Features (decoupling signals)
        "vel_radial_spine",
        "vel_radial_neck",
        "vel_radial_nose",
        "vel_radial_nose_i_tail_j",
        "vel_radial_tail_i_nose_j",
        "projected_vel_i_to_j",
        "projected_vel_j_to_i",
        "max_projected_vel",
        "cos_velocity_i_to_j",
        "relative_velocity_mag",
        "cos_heading_alignment_delta",
        # Physics Delta Features (synchrony & acceleration)
        "accel_radial_spine",
        "speed_correlation_60",
        "rolling_std_cos_heading_60",
        "nose_to_spine_ratio",
        "rolling_std_angular_velocity_60",
        "rolling_std_cos_bearing_60",
        "cross_velocity_correlation",
        "log_dist_ratio_nose_spine",
        "velocity_mag_diff",
        "compactness_diff",
        "relative_speed_ratio",
        "vel_tangential_spine",
        "angular_velocity_rel",
        "rolling_mean_mutual_facing_60",
        "rolling_std_mutual_facing_60",
        "rolling_std_rel_speed_ratio_60",
        "velocity_low_duration_i",
        "velocity_low_duration_j",
        # Rolling window features
        "distance_nose_i_nose_j_norm_rolling_mean",
        "distance_nose_i_neck_j_norm_rolling_mean",
        "distance_nose_i_spine_j_norm_rolling_mean",
        "distance_nose_i_tail_j_norm_rolling_mean",
        "distance_spine_i_nose_j_norm_rolling_mean",
        "distance_spine_i_neck_j_norm_rolling_mean",
        "distance_tail_i_nose_j_norm_rolling_mean",
        "distance_tail_i_neck_j_norm_rolling_mean",
        "distance_neck_i_nose_j_norm_rolling_mean",
        "d_nose_nose_norm_rolling_mean",
        "cos_heading_alignment_rolling_mean",
        "distance_nose_i_nose_j_norm_rolling_std",
        "distance_neck_i_neck_j_norm_rolling_mean",
        "distance_neck_i_spine_j_norm_rolling_mean",
        "distance_neck_i_tail_j_norm_rolling_mean",
        "cos_snout_alignment_rolling_mean",
        "cos_hs_i_to_nose_j_rolling_mean",
        "rolling_std_vel_radial_spine_60",
        "rolling_std_projected_vel_i_to_j_60",
        "rolling_std_projected_vel_j_to_i_60",
        "rolling_mean_projected_vel_i_to_j_60",
        "rolling_mean_projected_vel_j_to_i_60",
        "rolling_std_cos_bearing_60",
        "rolling_std_cos_bearing_j_to_i_60",
        # 3-Second Memory Features (360-frame backward-looking context)
        "rolling_min_dist_spine_360",
        "rolling_min_dist_nose_360",
        "rolling_min_dist_nose_i_body_j_360",
        "rolling_min_dist_nose_i_tail_j_360",
        "rolling_min_dist_body_i_nose_j_360",
        "rolling_min_dist_tail_i_nose_j_360",
        "rolling_mean_cos_heading_360",
        "rolling_mean_speed_diff_360",
        "rolling_min_dist_neck_360",
        "rolling_min_dist_tail_360",
        "rolling_min_dist_nose_i_neck_j_360",
        "rolling_min_dist_nose_i_spine_j_360",
        "rolling_min_dist_nose_i_tail_j_360",
        "rolling_min_dist_neck_i_nose_j_360",
        "rolling_min_dist_neck_i_spine_j_360",
        "rolling_min_dist_neck_i_tail_j_360",
        "rolling_min_dist_spine_i_nose_j_360",
        "rolling_min_dist_spine_i_neck_j_360",
        "rolling_min_dist_spine_i_tail_j_360",
        "rolling_min_dist_tail_i_neck_j_360",
        "rolling_min_dist_tail_i_spine_j_360",
        "rolling_min_dist_headstage_i_nose_j_360",
        "rolling_min_dist_headstage_i_neck_j_360",
        "rolling_min_dist_headstage_i_spine_j_360",
        "rolling_min_dist_headstage_i_tail_j_360",
        "rolling_min_dist_nose_i_nose_j_15",
        "rolling_min_dist_nose_i_side_j_15",
        "rolling_min_dist_side_i_nose_j_15",
        "rolling_min_dist_nose_i_tail_j_15",
        "rolling_min_dist_tail_i_nose_j_15",
        "rolling_min_dist_spine_i_spine_j_15",
        "rolling_min_dist_nose_i_spine_j_15",
        "rolling_min_dist_spine_i_nose_j_15",
        "rolling_speed_i_15",   
        "rolling_speed_j_15",
        "rolling_min_dist_nose_i_nose_j_60",
        "rolling_min_dist_nose_i_side_j_60",
        "rolling_min_dist_side_i_nose_j_60",
        "rolling_min_dist_nose_i_tail_j_60",
        "rolling_min_dist_tail_i_nose_j_60",
        "rolling_min_dist_spine_i_spine_j_60",
        "rolling_min_dist_nose_i_spine_j_60",
        "rolling_min_dist_spine_i_nose_j_60",
        "nose_closer_to_tail",
        "nose_closer_to_nose",
        "nose_closer_to_side",
        "rolling_min_nose_contact_dist_60",
        "rolling_mean_nose_contact_dist_60",
        "rolling_mean_heading_to_partner_i_60",
        "rolling_mean_heading_to_partner_j_60",
        "rolling_mean_heading_to_partner_i_360",
        "rolling_mean_heading_to_partner_j_360",
        "rolling_mean_dist_nose_i_side_j_15",
        "rolling_mean_dist_nose_i_side_j_60",
        "rolling_min_dist_nose_i_side_j_30",
        "rolling_mean_dist_nose_i_side_j_30",
        "rolling_mean_dist_nose_i_nose_j_15",
        "rolling_mean_dist_nose_i_nose_j_15",
        "rolling_mean_dist_nose_i_tail_j_15",
        "rolling_min_dist_nose_i_nose_j_30",
        "rolling_mean_dist_nose_i_nose_j_30",
        "rolling_min_dist_nose_i_tail_j_30",
        "rolling_mean_dist_nose_i_tail_j_30",
        "rolling_mean_dist_nose_i_tail_j_60",
        "rolling_mean_dist_nose_i_nose_j_60",
        "rolling_mean_dist_nose_i_spine_j_15",
        "rolling_mean_dist_nose_i_spine_j_30",
        "rolling_mean_dist_nose_i_spine_j_60",
        "rolling_min_dist_nose_i_spine_j_30",



    ]

    summary = _summarize_features(features_df, feature_cols_to_summarize)

    return FrameFeatureResult(features_df=features_df, summary=summary)


# ============================================================================
# Utilities
# ============================================================================


def print_frame_feature_summary(result: FrameFeatureResult) -> None:
    """
    Print a concise readable summary of frame features.

    Args:
        result: FrameFeatureResult to summarize.
    """
    s = result.summary

    print("\n" + "=" * 70)
    print("FRAME FEATURES SUMMARY")
    print("=" * 70)

    print("\nData Shape:")
    print(f"  Total rows: {s.n_rows}")
    print(f"  Unique dyads: {s.n_dyads}")
    print(f"  Unique frames: {s.n_dense_frames_unique}")

    print("\nFeature Missingness (top 10):")
    sorted_miss = sorted(s.missingness.items(), key=lambda x: x[1], reverse=True)[:10]
    for feat, miss_frac in sorted_miss:
        print(f"  {feat}: {miss_frac:.2%}")

    print("\nKey Feature Quantiles:")
    key_features = ["d_nose_nose_norm", "v_i_bl", "v_rel_bl", "distance_nose_i_nose_j_norm"]
    for feat in key_features:
        if feat in s.feature_quantiles:
            q = s.feature_quantiles[feat]
            print(f"  {feat}:")
            print(f"    q1={q.get('q1', np.nan):.3f}, "
                  f"q50={q.get('q50', np.nan):.3f}, "
                  f"q99={q.get('q99', np.nan):.3f}")

    print("=" * 70 + "\n")


def collapse_directional_contact_features_to_dyad(features_df: pd.DataFrame) -> pd.DataFrame:
    """Collapse directional CONTACT frame features to one dyad row per frame.

    Deterministic convention:
    - Keep canonical direction where str(focal_id) <= str(partner_id)
    - Validate uniqueness on (exp_name, iteration_name, dense_frame_idx)

    This preserves the existing directional-style feature names (i/j) while
    removing directional duplication for symmetric CONTACT training.
    """
    required_cols = [
        "exp_name",
        "iteration_name",
        "dense_frame_idx",
        "focal_id",
        "partner_id",
    ]
    missing = [c for c in required_cols if c not in features_df.columns]
    if missing:
        raise ValueError(
            "Cannot collapse directional CONTACT features; missing required columns: "
            f"{missing}"
        )

    if features_df.empty:
        return features_df.copy()

    df = features_df.copy()
    focal_key = df["focal_id"].astype(str)
    partner_key = df["partner_id"].astype(str)
    canonical_mask = focal_key <= partner_key
    dyad_df = df.loc[canonical_mask].copy()

    dyad_key_cols = ["exp_name", "iteration_name", "dense_frame_idx"]
    dups = dyad_df.duplicated(subset=dyad_key_cols, keep=False)
    if int(dups.sum()) > 0:
        preview = dyad_df.loc[dups, dyad_key_cols + ["focal_id", "partner_id"]].head(10)
        raise ValueError(
            "Dyad collapse produced duplicate frame keys on "
            f"{dyad_key_cols}. Preview:\n{preview.to_string(index=False)}"
        )

    return dyad_df.reset_index(drop=True)
