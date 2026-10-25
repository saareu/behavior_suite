"""Pipeline 50 post-processing steps (vendored; formulas unchanged).

Order (binding):
1. attach blob features
2. geometric echo / shifts
3. head-sweep / intent
4. total occlusion flags
5. last-known-good echoes
6. is_implanted
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from combined_features.contracts import (
    BLOB_FEATURE_COLS,
    IMPLANTATION_RESOLVED,
    MANDATORY_BLOB_FEATURE_COLS,
    OPTIONAL_BLOB_FEATURE_COLS,
    CombinedFeaturesError,
)
from combined_features.pose_adapter import ImplantationResolution




def attach_blob_features(
    directional_df: pd.DataFrame,
    blob_features: pd.DataFrame,
    *,
    require_complete_keys: bool = True,
) -> pd.DataFrame:
    """Left-join S5 blob columns onto directional rows on ``dense_frame_idx``.

    Preserves directional row count (many-to-one). Rejects missing/duplicate
    blob join keys when ``require_complete_keys`` is True.

    Mandatory scientific columns must be present on the S5 table; missing
    mandatory columns raise. Optional columns (none today) may be NaN-filled.
    Legitimate measurement NaNs inside present columns are preserved.
    """
    if "dense_frame_idx" not in directional_df.columns:
        raise CombinedFeaturesError(
            "Directional features are missing dense_frame_idx required for blob join."
        )

    missing_in_directional = [c for c in BLOB_FEATURE_COLS if c not in directional_df.columns]
    if not missing_in_directional:
        return directional_df

    if blob_features.empty:
        raise CombinedFeaturesError("S5 blob_features table is empty; cannot join.")

    if "dense_frame_idx" in blob_features.columns:
        blob_key = "dense_frame_idx"
    elif "frame_idx" in blob_features.columns:
        blob_key = "frame_idx"
    else:
        raise CombinedFeaturesError(
            "S5 blob_features is missing dense_frame_idx/frame_idx join key."
        )

    blob_keys = blob_features[blob_key]
    if blob_keys.duplicated().any():
        raise CombinedFeaturesError(
            "S5 blob_features has duplicate 120 Hz join keys; refusing merge."
        )
    if require_complete_keys:
        expected = set(np.asarray(directional_df["dense_frame_idx"], dtype=np.int64).tolist())
        present = set(np.asarray(blob_keys, dtype=np.int64).tolist())
        missing_keys = sorted(expected - present)
        if missing_keys:
            raise CombinedFeaturesError(
                "S5 blob_features is missing dense_frame_idx values required by "
                f"directional rows (e.g. {missing_keys[:5]})."
            )

    missing_mandatory = [
        c for c in MANDATORY_BLOB_FEATURE_COLS if c not in blob_features.columns
    ]
    if missing_mandatory:
        raise CombinedFeaturesError(
            "S5 blob_features is missing mandatory scientific columns: "
            + ", ".join(missing_mandatory)
        )

    join_cols = [
        c
        for c in missing_in_directional
        if c in blob_features.columns
        and (c in MANDATORY_BLOB_FEATURE_COLS or c in OPTIONAL_BLOB_FEATURE_COLS)
    ]
    blob_slim = blob_features[[blob_key, *join_cols]].drop_duplicates(subset=[blob_key])
    n_before = len(directional_df)
    out = directional_df.merge(
        blob_slim,
        how="left",
        left_on="dense_frame_idx",
        right_on=blob_key,
        validate="many_to_one",
    )
    if blob_key in out.columns and blob_key != "dense_frame_idx":
        out = out.drop(columns=[blob_key])

    if len(out) != n_before:
        raise CombinedFeaturesError(
            "Blob feature merge changed directional row count "
            f"({n_before} -> {len(out)}). Check dense_frame_idx uniqueness."
        )

    for col in OPTIONAL_BLOB_FEATURE_COLS:
        if col not in out.columns:
            out[col] = np.nan
    return out


# Backward-compatible private alias used by apply_pipeline50_post.
_attach_blob_features = attach_blob_features


def _add_geometric_echo_features(directional_df: pd.DataFrame) -> pd.DataFrame:
    """Add shifted social anchors and cc_count persistence features."""
    out = directional_df.sort_values(
        ["exp_name", "focal_id", "partner_id", "dense_frame_idx"]
    ).reset_index(drop=True)

    grouped = out.groupby(["exp_name", "focal_id", "partner_id"], sort=False)

    shift_specs = [
        ("distance_nose_i_spine_j_norm", "distance_nose_i_spine_j_norm_shift_10", 10),
        ("distance_nose_i_spine_j_norm", "distance_nose_i_spine_j_norm_shift_30", 30),
        ("distance_spine_i_nose_j_norm", "distance_spine_i_nose_j_norm_shift_10", 10),
        ("distance_spine_i_nose_j_norm", "distance_spine_i_nose_j_norm_shift_30", 30),
        ("distance_nose_i_tail_j_norm", "distance_nose_i_tail_j_norm_shift_10", 10),
        ("distance_tail_i_nose_j_norm", "distance_tail_i_nose_j_norm_shift_10", 10),
    ]
    for src_col, out_col, lag in shift_specs:
        if src_col in out.columns:
            out[out_col] = grouped[src_col].shift(lag)
        else:
            out[out_col] = np.nan
            print(
                "  WARNING: geometric echo source column missing; "
                f"initialized '{out_col}' to NaN (missing '{src_col}')"
            )

    # Persistence over the last 0.5 seconds (60 frames @ 120 fps).
    out["rolling_sum_cc_count_60"] = grouped["cc_count"].transform(
        lambda x: x.clip(lower=0).fillna(0).rolling(window=60, min_periods=1).sum()
    )
    return out


def _signed_kinematic_intent_series(
    projected_vel: pd.Series,
    nose_kinetic_energy: pd.Series,
    *,
    enter_thresh: float = 0.15,
    exit_thresh: float = 0.40,
    deadzone: float = 0.02,
    midrange_window: int = 25,
    state_smooth_window: int = 31,
) -> pd.Series:
    """Compute signed kinematic intent from projected velocity and kinetic energy."""
    v = pd.to_numeric(projected_vel, errors="coerce")
    energy = pd.to_numeric(nose_kinetic_energy, errors="coerce")
    valid_mask = v.notna() & energy.notna()

    v_upper = v.rolling(window=midrange_window, center=True, min_periods=1).max()
    v_lower = v.rolling(window=midrange_window, center=True, min_periods=1).min()
    v_midrange = (v_upper + v_lower) / 2.0

    state = 0
    states = np.zeros(len(v_midrange), dtype=float)

    for idx, val in enumerate(v_midrange.to_numpy(dtype=float, copy=False)):
        if not np.isfinite(val):
            states[idx] = np.nan
            continue

        if abs(val) <= deadzone:
            state = 0
        elif state == 1:
            if val < -exit_thresh:
                state = -1
        elif state == -1:
            if val > exit_thresh:
                state = 1
        else:
            if val >= enter_thresh:
                state = 1
            elif val <= -enter_thresh:
                state = -1
            else:
                state = 0

        states[idx] = state

    directional_state = pd.Series(states, index=projected_vel.index).rolling(
        window=state_smooth_window,
        center=True,
        min_periods=1,
    ).mean()
    directional_state = directional_state.where(valid_mask)
    return directional_state * energy





def _interpolate_short_gaps(series: pd.Series, max_gap: int = 30, grouped: Any = None) -> pd.Series:
    """Smart gap-fill via linear interpolation for short NaN gaps within groups."""
    if grouped is None:
        # Simple ungrouped case
        return series.interpolate(method='linear', limit=max_gap)
    
    # Grouped case: interpolate within each group up to max_gap frames
    result = series.copy()
    for _, group_idx in grouped.indices.items():
        idx = pd.Index(group_idx)
        group_series = series.loc[idx]
        interpolated = group_series.interpolate(method='linear', limit=max_gap)
        result.loc[idx] = interpolated
    
    return result


def add_head_sweep_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add vectorized nose/neck momentum features for rotational approach cases."""
    out = df.copy()

    sort_cols = [
        col
        for col in ["exp_name", "focal_id", "partner_id", "dense_frame_idx", "frame_idx"]
        if col in out.columns
    ]
    if sort_cols:
        out = out.sort_values(sort_cols).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)

    group_cols = [c for c in ["exp_name", "focal_id", "partner_id"] if c in out.columns]
    grouped = out.groupby(group_cols, sort=False) if group_cols else None

    def _safe_unit_components(dx: pd.Series, dy: pd.Series) -> tuple[pd.Series, pd.Series]:
        norm = np.sqrt(dx**2 + dy**2)
        valid = np.isfinite(norm) & (norm > 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            unit_x = np.where(valid, dx / norm, np.nan)
            unit_y = np.where(valid, dy / norm, np.nan)
        return pd.Series(unit_x, index=dx.index), pd.Series(unit_y, index=dy.index)

    def _diff_by_group(column: str) -> pd.Series:
        if column not in out.columns:
            return pd.Series(np.nan, index=out.index, dtype=float)
        if grouped is None:
            return out[column].diff()
        return grouped[column].diff()

    def _weighted_component_mean(specs: list[tuple[str, str]]) -> pd.Series:
        weighted_sum = pd.Series(0.0, index=out.index, dtype=float)
        weight_sum = pd.Series(0.0, index=out.index, dtype=float)

        for vel_col, impute_flag_col in specs:
            if vel_col not in out.columns:
                continue

            component = pd.to_numeric(out[vel_col], errors="coerce")
            if impute_flag_col in out.columns:
                imputed = pd.to_numeric(out[impute_flag_col], errors="coerce")
                weight = 1.0 - (imputed * 0.5)
            else:
                weight = pd.Series(1.0, index=out.index, dtype=float)

            weight = pd.to_numeric(weight, errors="coerce").clip(lower=0.0)
            valid = component.notna() & weight.notna()
            weighted_sum = weighted_sum.add(component.where(valid, 0.0) * weight.where(valid, 0.0), fill_value=0.0)
            weight_sum = weight_sum.add(weight.where(valid, 0.0), fill_value=0.0)

        return weighted_sum / weight_sum.replace(0.0, np.nan)

    for column in [
        "nose_vx_i",
        "nose_vy_i",
        "nose_vx_j",
        "nose_vy_j",
        "neck_vx_i",
        "neck_vy_i",
        "neck_vx_j",
        "neck_vy_j",
        "nose_dir_to_spine_j_x",
        "nose_dir_to_spine_j_y",
        "nose_dir_to_spine_i_x",
        "nose_dir_to_spine_i_y",
        "neck_dir_to_spine_j_x",
        "neck_dir_to_spine_j_y",
        "neck_dir_to_spine_i_x",
        "neck_dir_to_spine_i_y",
        "projected_vel_nose_i_to_spine_j",
        "projected_vel_neck_i_to_spine_j",
        "projected_vel_nose_j_to_spine_i",
        "projected_vel_neck_j_to_spine_i",
        "nose_reach_delta",
        "projected_vel_nose_max_15",
        "projected_vel_neck_max_15",
        "projected_vel_nose_max_15_j_to_i",
        "projected_vel_neck_max_15_j_to_i",
        "nose_kinetic_energy_15",
        "neck_kinetic_energy_15",
        "nose_kinetic_energy_15_j",
        "neck_kinetic_energy_15_j",
        "vel_radial_nose_i_spine_j",
        "vel_radial_neck_i_spine_j",
        "vel_radial_neck_i_tail_j",
        "projected_vel_nose_mean_5",
        "rolling_projected_vel_nose_15",
        "rolling_projected_vel_neck_15",
        "signed_kinematic_intent",
        "signed_kinematic_intent_neck",
        "signed_kinematic_intent_nose_or_neck",
        "signed_kinematic_intent_used_neck_fallback",
        "signed_kinematic_intent_j_to_i",
        "signed_kinematic_intent_neck_j_to_i",
        "signed_kinematic_intent_nose_or_neck_j_to_i",
        "signed_kinematic_intent_used_neck_fallback_j_to_i",
        "rolling_mean_cos_heading_j_to_tail_i_60",
        "vel_radial_spine_i_neck_j",
        "vel_radial_tail_i_neck_j",
        "vel_radial_spine_i_nose_j",
        "projected_vel_neck_i_to_nose_j",
        "projected_vel_nose_i_to_nose_j",
        "projected_vel_nose_i_to_neck_j",
        "projected_vel_nose_i_to_tail_j",
        "projected_vel_nose_i_to_j",
        "projected_vel_neck_i_to_neck_j",
        "projected_vel_neck_i_to_tail_j",
        "projected_vel_neck_i_to_j",
        "projected_vel_combined_i_to_j",
        "projected_vel_nose_j_to_neck_i",
        "projected_vel_nose_j_to_nose_i",
        "projected_vel_nose_j_to_tail_i",
        "projected_vel_nose_j_to_i",
        "projected_vel_neck_j_to_neck_i",
        "projected_vel_neck_j_to_nose_i",
        "projected_vel_neck_j_to_tail_i",
        "projected_vel_neck_j_to_i",
        "projected_vel_combined_j_to_i",
        "rolling_mean_projected_vel_combined_i_to_j_7",
        "rolling_mean_projected_vel_combined_j_to_i_7",
        "rolling_mean_projected_vel_neck_i_to_nose_j_60",
        "rolling_mean_projected_vel_nose_i_to_neck_j_60",
        "rolling_mean_projected_vel_neck_i_to_nose_j_15",
        "rolling_mean_projected_vel_nose_i_to_neck_j_15",
        "rolling_mean_projected_vel_neck_i_to_nose_j_8",
        "rolling_mean_projected_vel_nose_i_to_neck_j_8",
    ]:
        if column not in out.columns:
            out[column] = np.nan

    required_cols = [
        "nose_x_i",
        "nose_y_i",
        "neck_x_i",
        "neck_y_i",
        "spine_x_j",
        "spine_y_j",
    ]
    if not all(col in out.columns for col in required_cols):
        return out

    out["nose_vx_i"] = _diff_by_group("nose_x_i")
    out["nose_vy_i"] = _diff_by_group("nose_y_i")
    out["neck_vx_i"] = _diff_by_group("neck_x_i")
    out["neck_vy_i"] = _diff_by_group("neck_y_i")
    out["nose_vx_j"] = _diff_by_group("nose_x_j")
    out["nose_vy_j"] = _diff_by_group("nose_y_j")
    out["neck_vx_j"] = _diff_by_group("neck_x_j")
    out["neck_vy_j"] = _diff_by_group("neck_y_j")

    nose_dx = out["spine_x_j"] - out["nose_x_i"]
    nose_dy = out["spine_y_j"] - out["nose_y_i"]
    neck_dx = out["spine_x_j"] - out["neck_x_i"]
    neck_dy = out["spine_y_j"] - out["neck_y_i"]

    out["nose_dir_to_spine_j_x"], out["nose_dir_to_spine_j_y"] = _safe_unit_components(nose_dx, nose_dy)
    out["neck_dir_to_spine_j_x"], out["neck_dir_to_spine_j_y"] = _safe_unit_components(neck_dx, neck_dy)

    out["projected_vel_nose_i_to_spine_j"] = (
        out["nose_vx_i"] * out["nose_dir_to_spine_j_x"]
        + out["nose_vy_i"] * out["nose_dir_to_spine_j_y"]
    )
    out["projected_vel_neck_i_to_spine_j"] = (
        out["neck_vx_i"] * out["neck_dir_to_spine_j_x"]
        + out["neck_vy_i"] * out["neck_dir_to_spine_j_y"]
    )

    if "projected_vel_i_to_j" in out.columns:
        out["nose_reach_delta"] = (
            out["projected_vel_nose_i_to_spine_j"] - out["projected_vel_i_to_j"]
        )

    # Compute direction vectors and projected velocities for neck-to-nose and nose-to-neck.
    if all(col in out.columns for col in ["nose_x_j", "nose_y_j", "neck_x_i", "neck_y_i"]):
        neck_to_nose_dx = out["nose_x_j"] - out["neck_x_i"]
        neck_to_nose_dy = out["nose_y_j"] - out["neck_y_i"]
        out["neck_dir_to_nose_j_x"], out["neck_dir_to_nose_j_y"] = _safe_unit_components(neck_to_nose_dx, neck_to_nose_dy)

        out["projected_vel_neck_i_to_nose_j"] = (
            out["neck_vx_i"] * out["neck_dir_to_nose_j_x"]
            + out["neck_vy_i"] * out["neck_dir_to_nose_j_y"]
        )

    if all(col in out.columns for col in ["neck_x_j", "neck_y_j", "nose_x_i", "nose_y_i"]):
        nose_to_neck_dx = out["neck_x_j"] - out["nose_x_i"]
        nose_to_neck_dy = out["neck_y_j"] - out["nose_y_i"]
        out["nose_dir_to_neck_j_x"], out["nose_dir_to_neck_j_y"] = _safe_unit_components(nose_to_neck_dx, nose_to_neck_dy)

        out["projected_vel_nose_i_to_neck_j"] = (
            out["nose_vx_i"] * out["nose_dir_to_neck_j_x"]
            + out["nose_vy_i"] * out["nose_dir_to_neck_j_y"]
        )

    if all(col in out.columns for col in ["tail_x_j", "tail_y_j", "nose_x_i", "nose_y_i"]):
        nose_to_tail_dx = out["tail_x_j"] - out["nose_x_i"]
        nose_to_tail_dy = out["tail_y_j"] - out["nose_y_i"]
        out["nose_dir_to_tail_j_x"], out["nose_dir_to_tail_j_y"] = _safe_unit_components(nose_to_tail_dx, nose_to_tail_dy)

        out["projected_vel_nose_i_to_tail_j"] = (
            out["nose_vx_i"] * out["nose_dir_to_tail_j_x"]
            + out["nose_vy_i"] * out["nose_dir_to_tail_j_y"]
        )

    if all(col in out.columns for col in ["neck_x_j", "neck_y_j", "neck_x_i", "neck_y_i"]):
        neck_to_neck_dx = out["neck_x_j"] - out["neck_x_i"]
        neck_to_neck_dy = out["neck_y_j"] - out["neck_y_i"]
        out["neck_dir_to_neck_j_x"], out["neck_dir_to_neck_j_y"] = _safe_unit_components(neck_to_neck_dx, neck_to_neck_dy)

        out["projected_vel_neck_i_to_neck_j"] = (
            out["neck_vx_i"] * out["neck_dir_to_neck_j_x"]
            + out["neck_vy_i"] * out["neck_dir_to_neck_j_y"]
        )

    # Component: nose_i -> nose_j (explicit nose-to-nose projected velocity)
    if all(col in out.columns for col in ["nose_x_i", "nose_y_i", "nose_x_j", "nose_y_j"]):
        nose_to_nose_dx = out["nose_x_j"] - out["nose_x_i"]
        nose_to_nose_dy = out["nose_y_j"] - out["nose_y_i"]
        out["nose_dir_to_nose_j_x"], out["nose_dir_to_nose_j_y"] = _safe_unit_components(nose_to_nose_dx, nose_to_nose_dy)
        out["projected_vel_nose_i_to_nose_j"] = (
            out["nose_vx_i"] * out["nose_dir_to_nose_j_x"]
            + out["nose_vy_i"] * out["nose_dir_to_nose_j_y"]
        )

    if all(col in out.columns for col in ["tail_x_j", "tail_y_j", "neck_x_i", "neck_y_i"]):
        neck_to_tail_dx = out["tail_x_j"] - out["neck_x_i"]
        neck_to_tail_dy = out["tail_y_j"] - out["neck_y_i"]
        out["neck_dir_to_tail_j_x"], out["neck_dir_to_tail_j_y"] = _safe_unit_components(neck_to_tail_dx, neck_to_tail_dy)

        out["projected_vel_neck_i_to_tail_j"] = (
            out["neck_vx_i"] * out["neck_dir_to_tail_j_x"]
            + out["neck_vy_i"] * out["neck_dir_to_tail_j_y"]
        )

    # Compute combined projected velocity as weighted mean across all partner body parts
    # Downweight components where the target body part was imputed
    # All imputation flags are already created in frame_features.py
    nose_to_j_specs = [
        ("projected_vel_nose_i_to_nose_j", "nose_any_imputed_j"),
        ("projected_vel_nose_i_to_neck_j", "neck_any_imputed_j"),
        ("projected_vel_nose_i_to_spine_j", "spine_any_imputed_j"),
        ("projected_vel_nose_i_to_tail_j", "tail_any_imputed_j"),
    ]
    
    out["projected_vel_nose_i_to_j"] = _weighted_component_mean(nose_to_j_specs)
    
    # Compute combined neck projected velocity as weighted mean across all partner body parts
    neck_to_j_specs = [
        ("projected_vel_neck_i_to_nose_j", "nose_any_imputed_j"),
        ("projected_vel_neck_i_to_neck_j", "neck_any_imputed_j"),
        ("projected_vel_neck_i_to_spine_j", "spine_any_imputed_j"),
        ("projected_vel_neck_i_to_tail_j", "tail_any_imputed_j"),
    ]
    
    out["projected_vel_neck_i_to_j"] = _weighted_component_mean(neck_to_j_specs)

    # Compute combined nose+neck feature with imputation awareness and gap-fill post-processing
    nose_combined = out["projected_vel_nose_i_to_j"].copy() if "projected_vel_nose_i_to_j" in out.columns else pd.Series(np.nan, index=out.index)
    neck_combined = out["projected_vel_neck_i_to_j"].copy() if "projected_vel_neck_i_to_j" in out.columns else pd.Series(np.nan, index=out.index)
    
    # Build weights based on focal imputation status (nose_i and neck_i)
    nose_weight = 1.0 - (out["nose_any_imputed_i"] * 0.5) if "nose_any_imputed_i" in out.columns else pd.Series(1.0, index=out.index)
    neck_weight = 1.0 - (out["neck_any_imputed_i"] * 0.5) if "neck_any_imputed_i" in out.columns else pd.Series(1.0, index=out.index)
    
    # Combine with intelligent fallback: if one is missing, use the other
    combined = pd.Series(np.nan, index=out.index)
    both_valid = nose_combined.notna() & neck_combined.notna()
    only_nose = nose_combined.notna() & neck_combined.isna()
    only_neck = nose_combined.isna() & neck_combined.notna()
    
    combined[both_valid] = (
        (nose_combined[both_valid] * nose_weight[both_valid] + 
         neck_combined[both_valid] * neck_weight[both_valid]) / 
        (nose_weight[both_valid] + neck_weight[both_valid])
    )
    combined[only_nose] = nose_combined[only_nose]
    combined[only_neck] = neck_combined[only_neck]
    
    # Post-process with smart gap-fill interpolation within groups
    combined = _interpolate_short_gaps(combined, max_gap=30, grouped=grouped)
    out["projected_vel_combined_i_to_j"] = combined

    # Mirror features for j→i direction (partner approaching focal)
    if all(col in out.columns for col in ["neck_x_i", "neck_y_i", "nose_x_j", "nose_y_j"]):
        nose_j_to_neck_i_dx = out["neck_x_i"] - out["nose_x_j"]
        nose_j_to_neck_i_dy = out["neck_y_i"] - out["nose_y_j"]
        out["nose_j_dir_to_neck_i_x"], out["nose_j_dir_to_neck_i_y"] = _safe_unit_components(nose_j_to_neck_i_dx, nose_j_to_neck_i_dy)

        out["projected_vel_nose_j_to_neck_i"] = (
            out["nose_vx_j"] * out["nose_j_dir_to_neck_i_x"]
            + out["nose_vy_j"] * out["nose_j_dir_to_neck_i_y"]
        )

    if all(col in out.columns for col in ["tail_x_i", "tail_y_i", "nose_x_j", "nose_y_j"]):
        nose_j_to_tail_i_dx = out["tail_x_i"] - out["nose_x_j"]
        nose_j_to_tail_i_dy = out["tail_y_i"] - out["nose_y_j"]
        out["nose_j_dir_to_tail_i_x"], out["nose_j_dir_to_tail_i_y"] = _safe_unit_components(nose_j_to_tail_i_dx, nose_j_to_tail_i_dy)

        out["projected_vel_nose_j_to_tail_i"] = (
            out["nose_vx_j"] * out["nose_j_dir_to_tail_i_x"]
            + out["nose_vy_j"] * out["nose_j_dir_to_tail_i_y"]
        )

    if all(col in out.columns for col in ["neck_x_i", "neck_y_i", "neck_x_j", "neck_y_j"]):
        neck_j_to_neck_i_dx = out["neck_x_i"] - out["neck_x_j"]
        neck_j_to_neck_i_dy = out["neck_y_i"] - out["neck_y_j"]
        out["neck_j_dir_to_neck_i_x"], out["neck_j_dir_to_neck_i_y"] = _safe_unit_components(neck_j_to_neck_i_dx, neck_j_to_neck_i_dy)

        out["projected_vel_neck_j_to_neck_i"] = (
            out["neck_vx_j"] * out["neck_j_dir_to_neck_i_x"]
            + out["neck_vy_j"] * out["neck_j_dir_to_neck_i_y"]
        )

    if all(col in out.columns for col in ["tail_x_i", "tail_y_i", "neck_x_j", "neck_y_j"]):
        neck_j_to_tail_i_dx = out["tail_x_i"] - out["neck_x_j"]
        neck_j_to_tail_i_dy = out["tail_y_i"] - out["neck_y_j"]
        out["neck_j_dir_to_tail_i_x"], out["neck_j_dir_to_tail_i_y"] = _safe_unit_components(neck_j_to_tail_i_dx, neck_j_to_tail_i_dy)

        out["projected_vel_neck_j_to_tail_i"] = (
            out["neck_vx_j"] * out["neck_j_dir_to_tail_i_x"]
            + out["neck_vy_j"] * out["neck_j_dir_to_tail_i_y"]
        )

    # Compute weighted means for j→i direction (downweight where partner j is imputed on target body parts)
    nose_j_to_i_specs = [
        ("projected_vel_nose_j_to_nose_i", "nose_any_imputed_i"),
        ("projected_vel_nose_j_to_neck_i", "neck_any_imputed_i"),
        ("projected_vel_nose_j_to_spine_i", "spine_any_imputed_i"),
        ("projected_vel_nose_j_to_tail_i", "tail_any_imputed_i"),
    ]

    neck_j_to_i_specs = [
        ("projected_vel_neck_j_to_nose_i", "nose_any_imputed_i"),
        ("projected_vel_neck_j_to_neck_i", "neck_any_imputed_i"),
        ("projected_vel_neck_j_to_spine_i", "spine_any_imputed_i"),
        ("projected_vel_neck_j_to_tail_i", "tail_any_imputed_i"),
    ]

    if grouped is None:
        out["projected_vel_nose_max_15"] = out["projected_vel_nose_i_to_spine_j"].rolling(
            window=15, center=True, min_periods=1
        ).max()
        out["projected_vel_neck_max_15"] = out["projected_vel_neck_i_to_spine_j"].rolling(
            window=15, center=True, min_periods=1
        ).max()
        out["nose_kinetic_energy_15"] = out["projected_vel_nose_i_to_spine_j"].abs().rolling(
            window=15, center=True, min_periods=1
        ).mean()
        out["neck_kinetic_energy_15"] = out["projected_vel_neck_i_to_spine_j"].abs().rolling(
            window=15, center=True, min_periods=1
        ).mean()
    else:
        out["projected_vel_nose_max_15"] = grouped["projected_vel_nose_i_to_spine_j"].transform(
            lambda s: s.rolling(window=15, center=True, min_periods=1).max()
        )
        out["projected_vel_neck_max_15"] = grouped["projected_vel_neck_i_to_spine_j"].transform(
            lambda s: s.rolling(window=15, center=True, min_periods=1).max()
        )
        out["nose_kinetic_energy_15"] = grouped["projected_vel_nose_i_to_spine_j"].transform(
            lambda s: s.abs().rolling(window=15, center=True, min_periods=1).mean()
        )
        out["neck_kinetic_energy_15"] = grouped["projected_vel_neck_i_to_spine_j"].transform(
            lambda s: s.abs().rolling(window=15, center=True, min_periods=1).mean()
        )

    reverse_required_cols = ["nose_x_j", "nose_y_j", "neck_x_j", "neck_y_j", "spine_x_i", "spine_y_i"]
    if all(col in out.columns for col in reverse_required_cols):
        nose_dx_j = out["spine_x_i"] - out["nose_x_j"]
        nose_dy_j = out["spine_y_i"] - out["nose_y_j"]
        neck_dx_j = out["spine_x_i"] - out["neck_x_j"]
        neck_dy_j = out["spine_y_i"] - out["neck_y_j"]

        out["nose_dir_to_spine_i_x"], out["nose_dir_to_spine_i_y"] = _safe_unit_components(nose_dx_j, nose_dy_j)
        out["neck_dir_to_spine_i_x"], out["neck_dir_to_spine_i_y"] = _safe_unit_components(neck_dx_j, neck_dy_j)

        out["projected_vel_nose_j_to_spine_i"] = (
            out["nose_vx_j"] * out["nose_dir_to_spine_i_x"]
            + out["nose_vy_j"] * out["nose_dir_to_spine_i_y"]
        )
        out["projected_vel_neck_j_to_spine_i"] = (
            out["neck_vx_j"] * out["neck_dir_to_spine_i_x"]
            + out["neck_vy_j"] * out["neck_dir_to_spine_i_y"]
        )

        # Component: nose_j -> nose_i (explicit nose-to-nose projected velocity)
        if all(col in out.columns for col in ["nose_x_i", "nose_y_i", "nose_x_j", "nose_y_j"]):
            nose_j_to_nose_i_dx = out["nose_x_i"] - out["nose_x_j"]
            nose_j_to_nose_i_dy = out["nose_y_i"] - out["nose_y_j"]
            out["nose_j_dir_to_nose_i_x"], out["nose_j_dir_to_nose_i_y"] = _safe_unit_components(nose_j_to_nose_i_dx, nose_j_to_nose_i_dy)
            out["projected_vel_nose_j_to_nose_i"] = (
                out["nose_vx_j"] * out["nose_j_dir_to_nose_i_x"]
                + out["nose_vy_j"] * out["nose_j_dir_to_nose_i_y"]
            )

        # Component: neck_j -> nose_i (partner neck projected toward focal nose)
        if all(col in out.columns for col in ["neck_x_j", "neck_y_j", "nose_x_i", "nose_y_i"]):
            neck_j_to_nose_i_dx = out["nose_x_i"] - out["neck_x_j"]
            neck_j_to_nose_i_dy = out["nose_y_i"] - out["neck_y_j"]
            out["neck_j_dir_to_nose_i_x"], out["neck_j_dir_to_nose_i_y"] = _safe_unit_components(neck_j_to_nose_i_dx, neck_j_to_nose_i_dy)
            out["projected_vel_neck_j_to_nose_i"] = (
                out["neck_vx_j"] * out["neck_j_dir_to_nose_i_x"]
                + out["neck_vy_j"] * out["neck_j_dir_to_nose_i_y"]
            )

        if grouped is None:
            out["projected_vel_nose_max_15_j_to_i"] = out["projected_vel_nose_j_to_spine_i"].rolling(
                window=15, center=True, min_periods=1
            ).max()
            out["projected_vel_neck_max_15_j_to_i"] = out["projected_vel_neck_j_to_spine_i"].rolling(
                window=15, center=True, min_periods=1
            ).max()
            out["nose_kinetic_energy_15_j"] = out["projected_vel_nose_j_to_spine_i"].abs().rolling(
                window=15, center=True, min_periods=1
            ).mean()
            out["neck_kinetic_energy_15_j"] = out["projected_vel_neck_j_to_spine_i"].abs().rolling(
                window=15, center=True, min_periods=1
            ).mean()
        else:
            out["projected_vel_nose_max_15_j_to_i"] = grouped["projected_vel_nose_j_to_spine_i"].transform(
                lambda s: s.rolling(window=15, center=True, min_periods=1).max()
            )
            out["projected_vel_neck_max_15_j_to_i"] = grouped["projected_vel_neck_j_to_spine_i"].transform(
                lambda s: s.rolling(window=15, center=True, min_periods=1).max()
            )
            out["nose_kinetic_energy_15_j"] = grouped["projected_vel_nose_j_to_spine_i"].transform(
                lambda s: s.abs().rolling(window=15, center=True, min_periods=1).mean()
            )
            out["neck_kinetic_energy_15_j"] = grouped["projected_vel_neck_j_to_spine_i"].transform(
                lambda s: s.abs().rolling(window=15, center=True, min_periods=1).mean()
            )

        out["signed_kinematic_intent_j_to_i"] = _signed_kinematic_intent_series(
            out["projected_vel_nose_j_to_spine_i"],
            out["nose_kinetic_energy_15_j"],
        )
        out["signed_kinematic_intent_neck_j_to_i"] = _signed_kinematic_intent_series(
            out["projected_vel_neck_j_to_spine_i"],
            out["neck_kinetic_energy_15_j"],
        )

        nose_absent_mask_j = out["nose_x_j"].isna() | out["nose_y_j"].isna()
        out["signed_kinematic_intent_used_neck_fallback_j_to_i"] = nose_absent_mask_j.astype(np.uint8)
        out["signed_kinematic_intent_nose_or_neck_j_to_i"] = out["signed_kinematic_intent_j_to_i"]
        out.loc[nose_absent_mask_j, "signed_kinematic_intent_nose_or_neck_j_to_i"] = out.loc[
            nose_absent_mask_j, "signed_kinematic_intent_neck_j_to_i"
        ]

    out["projected_vel_nose_j_to_i"] = _weighted_component_mean(nose_j_to_i_specs)
    out["projected_vel_neck_j_to_i"] = _weighted_component_mean(neck_j_to_i_specs)

    nose_j_combined = out["projected_vel_nose_j_to_i"].copy()
    neck_j_combined = out["projected_vel_neck_j_to_i"].copy()
    nose_j_weight = 1.0 - (out["nose_any_imputed_j"] * 0.5) if "nose_any_imputed_j" in out.columns else pd.Series(1.0, index=out.index)
    neck_j_weight = 1.0 - (out["neck_any_imputed_j"] * 0.5) if "neck_any_imputed_j" in out.columns else pd.Series(1.0, index=out.index)

    combined_j_to_i = pd.Series(np.nan, index=out.index)
    both_valid_j = nose_j_combined.notna() & neck_j_combined.notna()
    only_nose_j = nose_j_combined.notna() & neck_j_combined.isna()
    only_neck_j = nose_j_combined.isna() & neck_j_combined.notna()

    combined_j_to_i[both_valid_j] = (
        (nose_j_combined[both_valid_j] * nose_j_weight[both_valid_j] +
         neck_j_combined[both_valid_j] * neck_j_weight[both_valid_j]) /
        (nose_j_weight[both_valid_j] + neck_j_weight[both_valid_j])
    )
    combined_j_to_i[only_nose_j] = nose_j_combined[only_nose_j]
    combined_j_to_i[only_neck_j] = neck_j_combined[only_neck_j]

    combined_j_to_i = _interpolate_short_gaps(combined_j_to_i, max_gap=30, grouped=grouped)
    out["projected_vel_combined_j_to_i"] = combined_j_to_i

    if grouped is None:
        out["rolling_mean_projected_vel_combined_i_to_j_7"] = out[
            "projected_vel_combined_i_to_j"
        ].rolling(window=7, min_periods=1).mean()
        out["rolling_mean_projected_vel_combined_j_to_i_7"] = out[
            "projected_vel_combined_j_to_i"
        ].rolling(window=7, min_periods=1).mean()
    else:
        out["rolling_mean_projected_vel_combined_i_to_j_7"] = grouped[
            "projected_vel_combined_i_to_j"
        ].transform(lambda s: s.rolling(window=7, min_periods=1).mean())
        out["rolling_mean_projected_vel_combined_j_to_i_7"] = grouped[
            "projected_vel_combined_j_to_i"
        ].transform(lambda s: s.rolling(window=7, min_periods=1).mean())

    # Radial velocity features are defined on normalized distances.
    radial_source_cols = {
        "vel_radial_nose_i_spine_j": "distance_nose_i_spine_j_norm",
        "vel_radial_neck_i_spine_j": "distance_neck_i_spine_j_norm",
        "vel_radial_neck_i_tail_j": "distance_neck_i_tail_j_norm",
        "vel_radial_spine_i_neck_j": "distance_spine_i_neck_j_norm",
        "vel_radial_tail_i_neck_j": "distance_tail_i_neck_j_norm",
        "vel_radial_spine_i_nose_j": "distance_spine_i_nose_j_norm",
    }
    for radial_col, source_col in radial_source_cols.items():
        if source_col not in out.columns:
            continue
        out[radial_col] = (
            out[source_col].diff() if grouped is None else grouped[source_col].diff()
        )

    if "cos_heading_j_to_tail_i" in out.columns:
        if grouped is None:
            out["rolling_mean_cos_heading_j_to_tail_i_60"] = out[
                "cos_heading_j_to_tail_i"
            ].rolling(window=60, min_periods=1).mean()
        else:
            out["rolling_mean_cos_heading_j_to_tail_i_60"] = grouped[
                "cos_heading_j_to_tail_i"
            ].transform(lambda s: s.rolling(window=60, min_periods=1).mean())

    if grouped is None:
        out["projected_vel_nose_mean_5"] = out["projected_vel_nose_i_to_spine_j"].rolling(
            window=5, min_periods=1
        ).mean()
    else:
        out["projected_vel_nose_mean_5"] = grouped["projected_vel_nose_i_to_spine_j"].transform(
            lambda s: s.rolling(window=5, min_periods=1).mean()
        )

    if grouped is None:
        out["rolling_projected_vel_nose_15"] = out["projected_vel_nose_i_to_spine_j"].rolling(
            window=15, center=True, min_periods=1
        ).mean()
        out["rolling_projected_vel_neck_15"] = out["projected_vel_neck_i_to_spine_j"].rolling(
            window=15, center=True, min_periods=1
        ).mean()
    else:
        out["rolling_projected_vel_nose_15"] = grouped["projected_vel_nose_i_to_spine_j"].transform(
            lambda s: s.rolling(window=15, center=True, min_periods=1).mean()
        )
        out["rolling_projected_vel_neck_15"] = grouped["projected_vel_neck_i_to_spine_j"].transform(
            lambda s: s.rolling(window=15, center=True, min_periods=1).mean()
        )

    # Rolling mean (60 frames) for nose-to-neck/neck-to-nose projected velocities.
    if grouped is None:
        if "projected_vel_neck_i_to_nose_j" in out.columns:
            out["rolling_mean_projected_vel_neck_i_to_nose_j_60"] = out["projected_vel_neck_i_to_nose_j"].rolling(
                window=60, min_periods=1
            ).mean()
        if "projected_vel_nose_i_to_neck_j" in out.columns:
            out["rolling_mean_projected_vel_nose_i_to_neck_j_60"] = out["projected_vel_nose_i_to_neck_j"].rolling(
                window=60, min_periods=1
            ).mean()
        if "projected_vel_neck_i_to_nose_j" in out.columns:
            out["rolling_mean_projected_vel_neck_i_to_nose_j_15"] = out["projected_vel_neck_i_to_nose_j"].rolling(
                window=15, min_periods=1
            ).mean()
        if "projected_vel_nose_i_to_neck_j" in out.columns:
            out["rolling_mean_projected_vel_nose_i_to_neck_j_15"] = out["projected_vel_nose_i_to_neck_j"].rolling(
                window=15, min_periods=1
            ).mean()
        if "projected_vel_neck_i_to_nose_j" in out.columns:
            out["rolling_mean_projected_vel_neck_i_to_nose_j_8"] = out["projected_vel_neck_i_to_nose_j"].rolling(
                window=8, min_periods=1
            ).mean()
        if "projected_vel_nose_i_to_neck_j" in out.columns:
            out["rolling_mean_projected_vel_nose_i_to_neck_j_8"] = out["projected_vel_nose_i_to_neck_j"].rolling(
                window=8, min_periods=1
            ).mean() 
    else:
        if "projected_vel_neck_i_to_nose_j" in out.columns:
            out["rolling_mean_projected_vel_neck_i_to_nose_j_60"] = grouped["projected_vel_neck_i_to_nose_j"].transform(
                lambda s: s.rolling(window=60, min_periods=1).mean()
            )
        if "projected_vel_nose_i_to_neck_j" in out.columns:
            out["rolling_mean_projected_vel_nose_i_to_neck_j_60"] = grouped["projected_vel_nose_i_to_neck_j"].transform(
                lambda s: s.rolling(window=60, min_periods=1).mean()
            )
        if "projected_vel_neck_i_to_nose_j" in out.columns:
            out["rolling_mean_projected_vel_neck_i_to_nose_j_15"] = grouped["projected_vel_neck_i_to_nose_j"].transform(
                lambda s: s.rolling(window=15, min_periods=1).mean()
            )
        if "projected_vel_nose_i_to_neck_j" in out.columns:
            out["rolling_mean_projected_vel_nose_i_to_neck_j_15"] = grouped["projected_vel_nose_i_to_neck_j"].transform(
                lambda s: s.rolling(window=15, min_periods=1).mean()
            )
        if "projected_vel_neck_i_to_nose_j" in out.columns:
            out["rolling_mean_projected_vel_neck_i_to_nose_j_8"] = grouped["projected_vel_neck_i_to_nose_j"].transform(
                lambda s: s.rolling(window=8, min_periods=1).mean()
            )
        if "projected_vel_nose_i_to_neck_j" in out.columns:
            out["rolling_mean_projected_vel_nose_i_to_neck_j_8"] = grouped["projected_vel_nose_i_to_neck_j"].transform(
                lambda s: s.rolling(window=8, min_periods=1).mean()
            )

    if grouped is None:
        out["signed_kinematic_intent"] = _signed_kinematic_intent_series(
            out["projected_vel_nose_i_to_spine_j"],
            out["nose_kinetic_energy_15"],
        )
        out["signed_kinematic_intent_neck"] = _signed_kinematic_intent_series(
            out["projected_vel_neck_i_to_spine_j"],
            out["neck_kinetic_energy_15"],
        )
    else:
        out["signed_kinematic_intent"] = 0.0
        out["signed_kinematic_intent_neck"] = 0.0
        for _, group_index in grouped.indices.items():
            idx = pd.Index(group_index)
            out.loc[idx, "signed_kinematic_intent"] = _signed_kinematic_intent_series(
                out.loc[idx, "projected_vel_nose_i_to_spine_j"],
                out.loc[idx, "nose_kinetic_energy_15"],
            ).to_numpy()
            out.loc[idx, "signed_kinematic_intent_neck"] = _signed_kinematic_intent_series(
                out.loc[idx, "projected_vel_neck_i_to_spine_j"],
                out.loc[idx, "neck_kinetic_energy_15"],
            ).to_numpy()

    nose_absent_mask = out["nose_x_i"].isna() | out["nose_y_i"].isna()
    out["signed_kinematic_intent_used_neck_fallback"] = nose_absent_mask.astype(np.uint8)
    out["signed_kinematic_intent_nose_or_neck"] = out["signed_kinematic_intent"]
    out.loc[nose_absent_mask, "signed_kinematic_intent_nose_or_neck"] = out.loc[
        nose_absent_mask, "signed_kinematic_intent_neck"
    ]

    if "y_approach" in out.columns and "projected_vel_nose_max_15" in out.columns:
        approach_mask = out["y_approach"] == 1
        if approach_mask.any():
            mean_lift = out.loc[approach_mask, "projected_vel_nose_max_15"].mean()
            print(
                "    projected_vel_nose_max_15 mean where y_approach == 1: "
                f"{mean_lift:.6f}"
            )
    # add rolling_mean_intent_15, rolling mean of signed_kinematic_intent_nose_or_neck over 15 frames, like the rolling features above, but with a 15-frame window
    if grouped is None:
        out["rolling_mean_intent_15"] = out["signed_kinematic_intent_nose_or_neck"].rolling(
            window=15, min_periods=1
        ).mean()
    else:
        out["rolling_mean_intent_15"] = grouped["signed_kinematic_intent_nose_or_neck"].transform(
            lambda s: s.rolling(window=15, min_periods=1).mean()
        )



    return out


def add_total_occlusion_flag(df: pd.DataFrame) -> pd.DataFrame:
    """Add focal/partner lost flags and total occlusion flag using coordinate columns."""
    out = df.copy()

    def _is_track_coord(col: str, track_token: str) -> bool:
        tokens = [t for t in col.lower().split("_") if t]
        return track_token in tokens and any(tok in {"x", "y"} for tok in tokens)

    focal_coord_cols = [c for c in out.columns if _is_track_coord(c, "i")]
    partner_coord_cols = [c for c in out.columns if _is_track_coord(c, "j")]

    if focal_coord_cols and partner_coord_cols:
        focal_lost = out[focal_coord_cols].isna().all(axis=1)
        partner_lost = out[partner_coord_cols].isna().all(axis=1)
    else:
        # Fallback for frames where explicit x/y coordinates are not persisted.
        paired_i_cols: list[str] = []
        paired_j_cols: list[str] = []
        for col in out.columns:
            if not col.endswith("_i"):
                continue
            col_j = f"{col[:-2]}_j"
            if col_j not in out.columns:
                continue
            if not (
                pd.api.types.is_numeric_dtype(out[col])
                and pd.api.types.is_numeric_dtype(out[col_j])
            ):
                continue
            if out[col].isna().any() or out[col_j].isna().any():
                paired_i_cols.append(col)
                paired_j_cols.append(col_j)

        if paired_i_cols and paired_j_cols:
            focal_lost = out[paired_i_cols].isna().all(axis=1)
            partner_lost = out[paired_j_cols].isna().all(axis=1)
        else:
            focal_lost = pd.Series(False, index=out.index)
            partner_lost = pd.Series(False, index=out.index)

    out["is_focal_lost"] = focal_lost.astype(np.uint8)
    out["is_partner_lost"] = partner_lost.astype(np.uint8)
    out["is_total_occlusion"] = (focal_lost | partner_lost).astype(np.uint8)
    return out


def add_last_known_good_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add echo features by carrying forward recent valid values up to 60 frames."""
    out = df.copy()

    feature_specs: dict[str, dict[str, Any]] = {}

    def _register_echo(src_col: str, echo_col: str, trigger_col: str) -> None:
        spec = feature_specs.setdefault(echo_col, {"src_col": src_col, "trigger_cols": []})
        if spec["src_col"] != src_col:
            raise ValueError(
                f"Conflicting source columns for '{echo_col}': '{spec['src_col']}' vs '{src_col}'"
            )
        if trigger_col not in spec["trigger_cols"]:
            spec["trigger_cols"].append(trigger_col)

    _register_echo("cos_heading_alignment", "cos_heading_alignment_echo", "is_total_occlusion")
    _register_echo("projected_vel_i_to_j", "projected_vel_i_to_j_echo", "is_total_occlusion")
    _register_echo("projected_vel_j_to_i", "projected_vel_j_to_i_echo", "is_total_occlusion")
    _register_echo("speed_i", "speed_i_echo", "is_total_occlusion")
    _register_echo("speed_j", "speed_j_echo", "is_total_occlusion")
    _register_echo("distance_nose_i_spine_j_norm", "distance_nose_i_spine_j_norm_echo", "is_total_occlusion")
    _register_echo("distance_spine_i_nose_j_norm", "distance_spine_i_nose_j_norm_echo", "is_total_occlusion")
    _register_echo("distance_nose_i_nose_j_norm", "distance_nose_i_nose_j_norm_echo_imputed", "nose_any_imputed_i")
    _register_echo("distance_nose_i_neck_j_norm", "distance_nose_i_neck_j_norm_echo_imputed", "nose_any_imputed_i")
    _register_echo("distance_nose_i_spine_j_norm", "distance_nose_i_spine_j_norm_echo_imputed", "nose_any_imputed_i")
    _register_echo("distance_nose_i_tail_j_norm", "distance_nose_i_tail_j_norm_echo_imputed", "nose_any_imputed_i")
    _register_echo("distance_headstage_i_nose_i_norm", "distance_headstage_i_nose_i_norm_echo_imputed", "nose_any_imputed_i")
    _register_echo("vel_radial_nose", "vel_radial_nose_echo_imputed", "nose_any_imputed_i")
    _register_echo("vel_radial_nose_i_tail_j", "vel_radial_nose_i_tail_j_echo_imputed", "nose_any_imputed_i")
    _register_echo("distance_nose_i_side_j_norm", "distance_nose_i_side_j_norm_echo_imputed", "nose_any_imputed_i")
    _register_echo("distance_neck_i_tail_j_norm", "distance_neck_i_tail_j_norm_echo_imputed", "tail_any_imputed_j")
    _register_echo("distance_spine_i_tail_j_norm", "distance_spine_i_tail_j_norm_echo_imputed", "tail_any_imputed_j")
    _register_echo("distance_tail_i_tail_j_norm", "distance_tail_i_tail_j_norm_echo_imputed", "tail_any_imputed_j")
    _register_echo("distance_headstage_i_tail_j_norm", "distance_headstage_i_tail_j_norm_echo_imputed", "tail_any_imputed_j")
    _register_echo("cos_heading_i_to_tail_j", "cos_heading_i_to_tail_j_echo_imputed", "tail_any_imputed_j")

    group_cols = [c for c in ["exp_name", "focal_id", "partner_id"] if c in out.columns]

    for echo_col, spec in feature_specs.items():
        src_col = spec["src_col"]
        if src_col not in out.columns:
            continue

        trigger_cols = [col for col in spec["trigger_cols"] if col in out.columns]
        if not trigger_cols:
            continue

        valid_mask = pd.Series(True, index=out.index)
        for trigger_col in trigger_cols:
            valid_mask &= out[trigger_col].fillna(0).eq(0)

        masked = out[src_col].where(valid_mask)
        if group_cols:
            out[echo_col] = masked.groupby(
                [out[c] for c in group_cols],
                sort=False,
            ).transform(lambda s: s.ffill(limit=60))
        else:
            out[echo_col] = masked.ffill(limit=60)

    return out


def add_is_implanted_feature(
    df: pd.DataFrame,
    implantation: ImplantationResolution,
) -> pd.DataFrame:
    """Add focal ``is_implanted`` from Sprint 1 headstage resolution.

    When implantation is resolved: ``1.0`` for the implanted focal, ``0.0`` for
    confirmed non-implanted focals. When unresolved: ``NaN`` for all rows so
    uncertainty is never encoded as ``0``. Reason is retained on frame attrs
    for QC.
    """
    out = df.copy()
    id_col = "track_id" if "track_id" in out.columns else "focal_id"
    if id_col not in out.columns:
        out["is_implanted"] = np.full(len(out), np.nan, dtype=np.float64)
        out.attrs["implantation_status"] = "missing_focal_id"
        out.attrs["implantation_reason"] = "no track_id/focal_id column"
        return out

    if (
        implantation.status == IMPLANTATION_RESOLVED
        and implantation.implanted_track_id is not None
    ):
        implanted_id = int(implantation.implanted_track_id)
        id_as_numeric = pd.to_numeric(out[id_col], errors="coerce")
        out["is_implanted"] = np.where(
            id_as_numeric.isna(),
            np.nan,
            id_as_numeric.eq(implanted_id).astype(np.float64),
        )
        out.attrs["implantation_status"] = IMPLANTATION_RESOLVED
        out.attrs["implanted_track_id"] = implanted_id
        out.attrs["implantation_reason"] = None
        return out

    out["is_implanted"] = np.full(len(out), np.nan, dtype=np.float64)
    out.attrs["implantation_status"] = implantation.status
    out.attrs["implanted_track_id"] = None
    out.attrs["implantation_reason"] = (
        f"implantation unresolved ({implantation.status}); "
        "is_implanted set to NaN (not 0) to preserve uncertainty"
    )
    return out


def apply_pipeline50_post(
    directional_df: pd.DataFrame,
    blob_features: pd.DataFrame,
    implantation: ImplantationResolution,
) -> pd.DataFrame:
    """Run Pipeline 50 post steps in the legacy order after base frame features."""

    out = attach_blob_features(directional_df, blob_features)
    out = _add_geometric_echo_features(out)
    out = add_head_sweep_features(out)
    out = add_total_occlusion_flag(out)
    out = add_last_known_good_features(out)
    out = add_is_implanted_feature(out, implantation)
    return out


