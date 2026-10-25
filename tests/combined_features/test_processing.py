"""Sprint 2: frame-feature migration and Pipeline 50 postprocessing tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from combined_features import (
    IMPLANTATION_ABSENT,
    IMPLANTATION_RESOLVED,
    AdaptedPose,
    CombinedFeaturesError,
    adapt_final_pose_to_frame_feature_input,
)
from combined_features.frame_features import build_frame_features
from combined_features.pipeline50_post import (
    BLOB_FEATURE_COLS,
    add_is_implanted_feature,
    attach_blob_features,
)
from combined_features.pose_adapter import ImplantationResolution
from combined_features.processing import compute_combined_features_from_adapted
from tests.combined_features.helpers import (
    PROVENANCE,
    build_blob_features_table,
    build_final_pose_table,
)


def _adapted(
    *,
    n_samples: int = 8,
    tracks: tuple[int, ...] = (0, 1),
    headstage_owner: int | None = 0,
    provenance_by_key: dict | None = None,
    blank_keys: set | None = None,
) -> tuple[AdaptedPose, pd.DataFrame]:
    pose = build_final_pose_table(
        n_samples=n_samples,
        tracks=tracks,
        headstage_owner=headstage_owner,
        provenance_by_key=provenance_by_key,
        blank_keys=blank_keys,
    )
    blob = build_blob_features_table(n_samples=n_samples)
    adapted = adapt_final_pose_to_frame_feature_input(
        pose,
        orig_frame_idx=blob["orig_frame_idx"].to_numpy(dtype=np.int64),
    )
    return adapted, blob


def test_node_aliases_share_one_anatomical_landmark() -> None:
    adapted, _ = _adapted(n_samples=4)
    spine = adapted.nodes.index("spine")
    spine_base = adapted.nodes.index("spine_base")
    assert spine != spine_base
    np.testing.assert_allclose(adapted.x[:, spine, :], adapted.x[:, spine_base, :])
    # FF named lookups still resolve without inventing a second body length.
    ff = build_frame_features(adapted, exp_name="alias")
    assert "distance_nose_i_spine_j_norm" in ff.features_df.columns
    assert ff.features_df["dyad_body_scale"].notna().any()


def test_two_mouse_directional_rows_and_swapped_track_order() -> None:
    adapted_a, blob = _adapted(tracks=(0, 1), n_samples=5)
    adapted_b, _ = _adapted(tracks=(5, 2), n_samples=5, headstage_owner=5)
    result_a = compute_combined_features_from_adapted(adapted_a, blob, exp_name="a")
    result_b = compute_combined_features_from_adapted(
        adapted_b, build_blob_features_table(n_samples=5), exp_name="b"
    )

    assert len(result_a.features) == 5 * 2
    assert set(
        zip(
            result_a.features["focal_id"].tolist(),
            result_a.features["partner_id"].tolist(),
            strict=True,
        )
    ) == {(0, 1), (1, 0)}

    # Track tensor order is sorted, but identity columns keep real track ids.
    assert set(result_b.adapted_pose.tracks) == {2, 5}
    pairs = set(
        zip(
            result_b.features["focal_id"].tolist(),
            result_b.features["partner_id"].tolist(),
            strict=True,
        )
    )
    assert pairs == {(2, 5), (5, 2)}

    # i/j semantics: nose_i follows focal_id, not tensor index order.
    row = result_b.features[
        (result_b.features.focal_id == 5) & (result_b.features.dense_frame_idx == 0)
    ].iloc[0]
    track_idx = result_b.adapted_pose.tracks.index(5)
    nose_i = result_b.adapted_pose.nodes.index("nose")
    assert row["nose_x_i"] == pytest.approx(result_b.adapted_pose.x[track_idx, nose_i, 0])


def test_kalman_only_nose_any_imputed_mapping() -> None:
    provenance_by_key = {
        (0, 0, "nose"): PROVENANCE["kalman_gap_fill"],
        (1, 0, "nose"): PROVENANCE["linear_gap_fill"],
        (2, 0, "nose"): PROVENANCE["temporal_resample"],
    }
    adapted, blob = _adapted(n_samples=3, provenance_by_key=provenance_by_key)
    result = compute_combined_features_from_adapted(adapted, blob)
    focal0 = result.features[
        (result.features.focal_id == 0) & (result.features.dense_frame_idx == 0)
    ].iloc[0]
    focal0_linear = result.features[
        (result.features.focal_id == 0) & (result.features.dense_frame_idx == 1)
    ].iloc[0]
    focal0_resample = result.features[
        (result.features.focal_id == 0) & (result.features.dense_frame_idx == 2)
    ].iloc[0]

    assert int(focal0["nose_i_is_kf"]) == 1
    assert int(focal0["nose_i_is_pchip"]) == 0
    assert int(focal0["nose_any_imputed_i"]) == 1

    assert int(focal0_linear["nose_i_is_kf"]) == 0
    assert int(focal0_linear["nose_any_imputed_i"]) == 0

    assert int(focal0_resample["nose_i_is_kf"]) == 0
    assert int(focal0_resample["nose_any_imputed_i"]) == 0
    assert int(focal0["nose_any_refined"]) == 0


def test_implantation_resolved_and_unresolved() -> None:
    adapted_res, blob = _adapted(n_samples=4, headstage_owner=1, tracks=(0, 1))
    result = compute_combined_features_from_adapted(adapted_res, blob)
    assert adapted_res.implantation.status == IMPLANTATION_RESOLVED
    assert adapted_res.implantation.implanted_track_id == 1
    assert set(
        result.features.loc[result.features.focal_id == 1, "is_implanted"].unique().tolist()
    ) == {1}
    assert set(
        result.features.loc[result.features.focal_id == 0, "is_implanted"].unique().tolist()
    ) == {0}

    pose = build_final_pose_table(
        n_samples=3,
        nodes=("nose", "neck", "spine_base", "tail_base"),
        headstage_owner=None,
    )
    blob_abs = build_blob_features_table(n_samples=3)
    adapted_abs = adapt_final_pose_to_frame_feature_input(
        pose, orig_frame_idx=blob_abs["orig_frame_idx"].to_numpy(dtype=np.int64)
    )
    assert adapted_abs.implantation.status == IMPLANTATION_ABSENT
    unresolved = compute_combined_features_from_adapted(adapted_abs, blob_abs)
    assert unresolved.features["is_implanted"].isna().all()
    assert any("unresolved" in w or "NaN" in w for w in unresolved.warnings)
    # Must not invent track-0 implantation merely because id==0 exists.
    assert unresolved.features.attrs["implanted_track_id"] is None


def test_geometry_motion_formula_smoke() -> None:
    adapted, blob = _adapted(n_samples=10)
    # Known neck displacement for track 0: x increases by 1 each sample at 120 Hz.
    result = compute_combined_features_from_adapted(adapted, blob)
    row0 = result.features[
        (result.features.focal_id == 0) & (result.features.dense_frame_idx == 1)
    ].iloc[0]
    # Finite-difference over dt=1/120 with rolling-median (win=9) on nearly linear series
    # should remain near 120 px/s for neck-based speed proxy used in kinematics.
    assert np.isfinite(row0["speed_i"])
    assert row0["speed_i"] > 0
    # Pairwise nose distance normalized by dyad body scale is finite and positive.
    assert row0["distance_nose_i_nose_j_norm"] > 0
    assert "rolling_speed_i_15" in result.features.columns


def test_blob_merge_cardinality_and_alignment() -> None:
    adapted, blob = _adapted(n_samples=6)
    ff = build_frame_features(adapted)
    out = attach_blob_features(ff.features_df, blob)
    assert len(out) == len(ff.features_df)
    assert out["cc_count"].notna().all()
    for col in ("cc_count", "min_inter_contour_dist", "solidity"):
        assert col in out.columns

    dup = pd.concat([blob, blob.iloc[[0]]], ignore_index=True)
    with pytest.raises(CombinedFeaturesError, match="duplicate"):
        attach_blob_features(ff.features_df, dup)

    truncated = blob.iloc[:-1].copy()
    with pytest.raises(CombinedFeaturesError, match="missing dense_frame_idx"):
        attach_blob_features(ff.features_df, truncated)


def test_occlusion_and_echo_features_present() -> None:
    adapted, blob = _adapted(n_samples=12)
    result = compute_combined_features_from_adapted(adapted, blob)
    for col in (
        "is_focal_lost",
        "is_partner_lost",
        "is_total_occlusion",
        "distance_nose_i_spine_j_norm_shift_10",
        "rolling_sum_cc_count_60",
    ):
        assert col in result.features.columns


def test_protected_blank_missingness() -> None:
    blank = {(2, 0, "nose")}
    adapted, blob = _adapted(n_samples=4, blank_keys=blank)
    result = compute_combined_features_from_adapted(adapted, blob)
    row = result.features[
        (result.features.focal_id == 0) & (result.features.dense_frame_idx == 2)
    ].iloc[0]
    assert np.isnan(row["nose_x_i"])
    assert int(row["nose_i_is_kf"]) == 0
    assert int(row["nose_any_imputed_i"]) == 0


def test_full_in_memory_pipeline_and_active_feature_names() -> None:
    adapted, blob = _adapted(n_samples=16, headstage_owner=0)
    result = compute_combined_features_from_adapted(adapted, blob, exp_name="synth")
    assert result.n_rows == 16 * 2
    assert result.features.duplicated(
        subset=["focal_id", "partner_id", "dense_frame_idx"]
    ).sum() == 0
    assert "sample_idx" in result.features.columns
    assert "time_sec" in result.features.columns
    np.testing.assert_array_equal(
        result.features["sample_idx"].to_numpy(),
        result.features["dense_frame_idx"].to_numpy(),
    )

    # Representative active-model columns generated by this sprint's stages.
    required = [
        "distance_nose_i_nose_j_norm",
        "nose_any_imputed_i",
        "nose_any_imputed_j",
        "nose_i_is_kf",
        "nose_j_is_kf",
        "nose_i_sigma",
        "cc_count",
        "min_inter_contour_dist",
        "is_total_occlusion",
        "is_implanted",
        "signed_kinematic_intent",
        "projected_vel_i_to_j",
    ]
    missing = [c for c in required if c not in result.features.columns]
    assert missing == []
    # Blob feature contract columns should all be present after join.
    missing_blob = [c for c in BLOB_FEATURE_COLS if c not in result.features.columns]
    assert missing_blob == []


def test_is_implanted_never_defaults_to_zero_without_resolution() -> None:
    df = pd.DataFrame(
        {
            "focal_id": [0, 1, 0],
            "partner_id": [1, 0, 1],
            "dense_frame_idx": [0, 0, 1],
        }
    )
    implantation = ImplantationResolution(
        status=IMPLANTATION_ABSENT,
        implanted_track_id=None,
        evidence={},
    )
    out = add_is_implanted_feature(df, implantation)
    assert out["is_implanted"].isna().all()
    assert out.attrs["implanted_track_id"] is None
    assert "NaN" in str(out.attrs["implantation_reason"])


def test_mandatory_blob_columns_rejected_when_missing() -> None:
    adapted, blob = _adapted(n_samples=4)
    ff = build_frame_features(adapted)
    truncated = blob.drop(columns=["cc_count", "solidity"])
    with pytest.raises(CombinedFeaturesError, match="mandatory scientific columns"):
        attach_blob_features(ff.features_df, truncated)


def test_legitimate_blob_measurement_nans_preserved() -> None:
    adapted, blob = _adapted(n_samples=4)
    blob = blob.copy()
    blob.loc[1, "min_inter_contour_dist"] = np.nan
    blob.loc[2, "ellipse_eccentricity"] = np.nan
    ff = build_frame_features(adapted)
    out = attach_blob_features(ff.features_df, blob)
    sample1 = out[out["dense_frame_idx"] == 1]
    sample2 = out[out["dense_frame_idx"] == 2]
    assert sample1["min_inter_contour_dist"].isna().all()
    assert sample2["ellipse_eccentricity"].isna().all()
    assert sample1["cc_count"].notna().all()
