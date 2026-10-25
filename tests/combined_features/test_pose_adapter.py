"""S4 pose adapter tests: pivot, provenance, aliases, implantation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from combined_features import (
    ADAPTER_POLICY_VERSION,
    IMPLANTATION_ABSENT,
    IMPLANTATION_AMBIGUOUS,
    IMPLANTATION_RESOLVED,
    IMPUTE_STAGE_KALMAN,
    IMPUTE_STAGE_LINEAR,
    IMPUTE_STAGE_NONE,
    AdaptedPose,
    CombinedFeaturesError,
    adapt_final_pose_to_frame_feature_input,
    map_provenance_to_quality,
    resolve_implanted_track,
)
from pose_finalization.contracts import (
    PROVENANCE_IDENTITY_CORRECTED,
    PROVENANCE_KALMAN_GAP_FILL,
    PROVENANCE_LINEAR_GAP_FILL,
    PROVENANCE_MISSING,
    PROVENANCE_ORIGINAL,
    PROVENANCE_PROTECTED_MANUAL_BLANK,
    PROVENANCE_TEMPORAL_RESAMPLE,
)
from tests.combined_features.helpers import PROVENANCE, build_final_pose_table


def _adapt(pose: pd.DataFrame, *, n_samples: int | None = None) -> AdaptedPose:
    n = n_samples if n_samples is not None else int(pose["sample_idx"].nunique())
    orig = np.arange(n, dtype=np.int64)
    return adapt_final_pose_to_frame_feature_input(pose, orig_frame_idx=orig)


def test_track_and_node_normalization_aliases() -> None:
    pose = build_final_pose_table(
        n_samples=3,
        nodes=("nose", "neck", "spine_base", "tail_base", "headstage"),
    )
    adapted = _adapt(pose)
    assert adapted.tracks == (0, 1)
    assert "spine" in adapted.nodes and "spine_base" in adapted.nodes
    assert "tail" in adapted.nodes and "tail_base" in adapted.nodes
    assert "headstage" in adapted.nodes
    # Alias slots receive identical coordinates.
    spine_i = adapted.nodes.index("spine")
    spine_base_i = adapted.nodes.index("spine_base")
    np.testing.assert_allclose(adapted.x[:, spine_i, :], adapted.x[:, spine_base_i, :])


def test_directional_identity_pairs_preserved() -> None:
    pose = build_final_pose_table(n_samples=2, tracks=(2, 5))
    adapted = _adapt(pose)
    assert adapted.tracks == (2, 5)
    assert adapted.directional_track_pairs() == ((2, 5), (5, 2))


def test_post_smooth_coords_and_t120_and_orig_mapping() -> None:
    pose = build_final_pose_table(n_samples=4)
    # Distinct pre-smooth vs post-smooth to prove post-smooth selection.
    pose.loc[:, "x_pre_smooth"] = pose["x"] + 100.0
    pose.loc[:, "y_pre_smooth"] = pose["y"] + 100.0
    orig = np.asarray([0, 0, 1, 2], dtype=np.int64)
    adapted = adapt_final_pose_to_frame_feature_input(pose, orig_frame_idx=orig)
    nose_i = adapted.nodes.index("nose")
    # Post-smooth values match S4 x/y, not pre-smooth.
    row = pose[(pose.sample_idx == 1) & (pose.track == 0) & (pose.node == "nose")].iloc[0]
    assert adapted.x[0, nose_i, 1] == pytest.approx(float(row["x"]))
    assert adapted.x_pre_smooth[0, nose_i, 1] == pytest.approx(float(row["x"]) + 100.0)
    np.testing.assert_allclose(adapted.t_120, np.arange(4) / 120.0, atol=1e-12)
    np.testing.assert_array_equal(adapted.orig_frame_idx, orig)
    assert adapted.adapter_policy_version == ADAPTER_POLICY_VERSION


def test_protected_blanks_and_nan_propagation() -> None:
    blank = {(1, 0, "nose")}
    pose = build_final_pose_table(n_samples=3, blank_keys=blank)
    adapted = _adapt(pose)
    nose_i = adapted.nodes.index("nose")
    assert np.isnan(adapted.x[0, nose_i, 1])
    assert np.isnan(adapted.y[0, nose_i, 1])
    assert bool(adapted.protected_manual_blank[0, nose_i, 1]) is True
    assert not bool(adapted.is_kf[0, nose_i, 1])
    assert not bool(adapted.is_imputed[0, nose_i, 1])
    assert adapted.impute_stage[0, nose_i, 1] == IMPUTE_STAGE_NONE


def test_kalman_versus_linear_and_resample_provenance() -> None:
    provenance_by_key = {
        (0, 0, "nose"): PROVENANCE["kalman_gap_fill"],
        (1, 0, "nose"): PROVENANCE["linear_gap_fill"],
        (2, 0, "nose"): PROVENANCE["temporal_resample"],
        (0, 0, "neck"): PROVENANCE["identity_corrected"],
    }
    pose = build_final_pose_table(n_samples=3, provenance_by_key=provenance_by_key)
    adapted = _adapt(pose)
    nose_i = adapted.nodes.index("nose")
    neck_i = adapted.nodes.index("neck")

    assert adapted.is_kf[0, nose_i, 0]
    assert adapted.impute_stage[0, nose_i, 0] == IMPUTE_STAGE_KALMAN
    assert adapted.is_imputed[0, nose_i, 0]
    assert not adapted.is_pchip[0, nose_i, 0]

    assert not adapted.is_kf[0, nose_i, 1]
    assert adapted.impute_stage[0, nose_i, 1] == IMPUTE_STAGE_LINEAR
    assert adapted.is_imputed[0, nose_i, 1]  # linear counts as imputed raw
    # Legacy *_any_imputed is pchip∨kf; linear excluded — covered by is_kf/is_pchip.

    assert not adapted.is_kf[0, nose_i, 2]
    assert not adapted.is_imputed[0, nose_i, 2]
    assert adapted.impute_stage[0, nose_i, 2] == IMPUTE_STAGE_NONE

    assert not adapted.is_kf[0, neck_i, 0]
    assert not adapted.is_imputed[0, neck_i, 0]
    assert not adapted.was_refined.any()


def test_map_provenance_quality_table() -> None:
    prov = np.asarray(
        [
            PROVENANCE_MISSING,
            PROVENANCE_ORIGINAL,
            PROVENANCE_IDENTITY_CORRECTED,
            PROVENANCE_TEMPORAL_RESAMPLE,
            PROVENANCE_LINEAR_GAP_FILL,
            PROVENANCE_KALMAN_GAP_FILL,
            PROVENANCE_PROTECTED_MANUAL_BLANK,
        ],
        dtype=np.uint8,
    )
    stage, imputed, kf, pchip, refined = map_provenance_to_quality(prov)
    assert list(stage) == [
        IMPUTE_STAGE_NONE,
        IMPUTE_STAGE_NONE,
        IMPUTE_STAGE_NONE,
        IMPUTE_STAGE_NONE,
        IMPUTE_STAGE_LINEAR,
        IMPUTE_STAGE_KALMAN,
        IMPUTE_STAGE_NONE,
    ]
    assert list(imputed.astype(int)) == [0, 0, 0, 0, 1, 1, 0]
    assert list(kf.astype(int)) == [0, 0, 0, 0, 0, 1, 0]
    assert not pchip.any()
    assert not refined.any()


def test_point_score_pass_through_no_fabrication() -> None:
    scores = {(0, 0, "nose"): 0.77}
    provenance_by_key = {(1, 0, "nose"): PROVENANCE["kalman_gap_fill"]}
    pose = build_final_pose_table(
        n_samples=2,
        point_scores=scores,
        provenance_by_key=provenance_by_key,
    )
    adapted = _adapt(pose)
    nose_i = adapted.nodes.index("nose")
    assert adapted.point_score[0, nose_i, 0] == pytest.approx(0.77)
    assert np.isnan(adapted.point_score[0, nose_i, 1])


def test_headstage_implantation_resolved_unique_owner() -> None:
    pose = build_final_pose_table(n_samples=4, headstage_owner=1, tracks=(0, 1))
    adapted = _adapt(pose)
    assert adapted.implantation.status == IMPLANTATION_RESOLVED
    assert adapted.implantation.implanted_track_id == 1


def test_headstage_implantation_absent() -> None:
    pose = build_final_pose_table(
        n_samples=2,
        nodes=("nose", "neck", "spine_base", "tail_base"),
        headstage_owner=None,
    )
    adapted = _adapt(pose)
    assert adapted.implantation.status == IMPLANTATION_ABSENT
    assert adapted.implantation.implanted_track_id is None


def test_headstage_implantation_ambiguous() -> None:
    pose = build_final_pose_table(n_samples=2, headstage_owner=None)
    # Force finite headstage on both tracks.
    mask = pose["node"] == "headstage"
    pose.loc[mask, "x"] = 1.0
    pose.loc[mask, "y"] = 2.0
    pose.loc[mask, "provenance"] = PROVENANCE_ORIGINAL
    adapted = _adapt(pose)
    assert adapted.implantation.status == IMPLANTATION_AMBIGUOUS
    assert adapted.implantation.implanted_track_id is None


def test_resolve_implanted_track_does_not_default_to_zero() -> None:
    tracks = (0, 1)
    nodes = ("nose", "headstage")
    x = np.full((2, 2, 3), np.nan)
    y = np.full((2, 2, 3), np.nan)
    result = resolve_implanted_track(tracks=tracks, nodes=nodes, x=x, y=y)
    assert result.implanted_track_id is None
    assert result.status == IMPLANTATION_ABSENT


def test_missing_required_nodes_hard_error() -> None:
    pose = build_final_pose_table(n_samples=1, nodes=("nose", "neck", "headstage"))
    with pytest.raises(CombinedFeaturesError, match="required scientific nodes"):
        _adapt(pose)


def test_duplicate_keys_hard_error() -> None:
    pose = build_final_pose_table(n_samples=1)
    pose = pd.concat([pose, pose.iloc[[0]]], ignore_index=True)
    with pytest.raises(CombinedFeaturesError, match="duplicate"):
        _adapt(pose)


def test_to_frame_feature_arrays_shapes() -> None:
    adapted = _adapt(build_final_pose_table(n_samples=3))
    arrays = adapted.to_frame_feature_arrays()
    meta = adapted.to_impute_meta()
    assert arrays["x"].shape == (2, adapted.n_nodes, 3)
    assert meta["tracks"] == [0, 1]
    assert "implantation" in meta
