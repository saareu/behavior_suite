"""Source-rate gap fill, bout temporal features, and 120 Hz mapping."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from blob_features import (
    OCCLUSION_STATIC_FEATURE_COLS,
    add_temporal_occlusion_features,
    apply_source_temporal_pipeline,
    build_nearest_neighbor_timing_map,
    impute_cc_count_and_min_inter_contour_dist,
    map_source_features_to_target,
)


def test_gap_fill_same_flanking_values() -> None:
    df = pd.DataFrame(
        {
            "frame_idx": [0, 1, 2, 3, 4],
            "cc_count": [2.0, np.nan, np.nan, 2.0, 1.0],
            "cc_count_raw": [2, 3, 3, 2, 1],
            "min_inter_contour_dist": [10.0, np.nan, np.nan, 20.0, 0.0],
        }
    )
    filled = impute_cc_count_and_min_inter_contour_dist(df, max_gap_frames=5)
    assert filled["cc_count"].tolist()[:4] == pytest.approx([2.0, 2.0, 2.0, 2.0])
    assert filled["min_inter_contour_dist"].iloc[1:3].tolist() == pytest.approx(
        [10.0 + 10.0 / 3.0, 10.0 + 20.0 / 3.0]
    )
    # Raw component counts must remain untouched by effective-count gap fill.
    assert filled["cc_count_raw"].tolist() == [2, 3, 3, 2, 1]


def test_gap_fill_preserves_boundary_nans_and_max_gap() -> None:
    df = pd.DataFrame(
        {
            "frame_idx": [0, 1, 2, 3, 4],
            "cc_count": [np.nan, 2.0, np.nan, 2.0, np.nan],
            "cc_count_raw": [0, 2, 3, 2, 0],
            "min_inter_contour_dist": [np.nan, 5.0, np.nan, 7.0, np.nan],
        }
    )
    filled = impute_cc_count_and_min_inter_contour_dist(df, max_gap_frames=5)
    # Leading / trailing gaps are never filled (legacy boundary behavior).
    assert not np.isfinite(filled["cc_count"].iloc[0])
    assert not np.isfinite(filled["cc_count"].iloc[4])
    assert filled["cc_count"].iloc[2] == pytest.approx(2.0)
    assert filled["cc_count_raw"].tolist() == [0, 2, 3, 2, 0]


def test_gap_fill_skips_mismatched_flanks_and_overlong_gaps() -> None:
    df = pd.DataFrame(
        {
            "frame_idx": list(range(6)),
            "cc_count": [1.0, np.nan, 2.0, 2.0, np.nan, np.nan],
            "min_inter_contour_dist": [0.0, np.nan, 5.0, 5.0, np.nan, np.nan],
        }
    )
    # Transition 1→2 must remain NaN.
    filled = impute_cc_count_and_min_inter_contour_dist(df, max_gap_frames=5)
    assert not np.isfinite(filled["cc_count"].iloc[1])

    # Over-long gap (2 frames with max_gap=1) must remain NaN.
    long_gap = pd.DataFrame(
        {
            "frame_idx": [0, 1, 2, 3],
            "cc_count": [2.0, np.nan, np.nan, 2.0],
            "min_inter_contour_dist": [1.0, np.nan, np.nan, 3.0],
        }
    )
    filled_long = impute_cc_count_and_min_inter_contour_dist(long_gap, max_gap_frames=1)
    assert not np.isfinite(filled_long["cc_count"].iloc[1])
    assert not np.isfinite(filled_long["cc_count"].iloc[2])


def test_gap_fill_disabled_when_max_gap_zero() -> None:
    df = pd.DataFrame(
        {
            "frame_idx": [0, 1, 2],
            "cc_count": [2.0, np.nan, 2.0],
            "min_inter_contour_dist": [1.0, np.nan, 3.0],
        }
    )
    filled = impute_cc_count_and_min_inter_contour_dist(df, max_gap_frames=0)
    assert not np.isfinite(filled["cc_count"].iloc[1])


def test_temporal_features_bout_scoped_and_nan_outside_occlusion() -> None:
    n = 20
    cc = np.full(n, 2.0)
    cc[5:12] = 1.0  # bout A
    cc[14:18] = 1.0  # bout B
    static = np.arange(n, dtype=float)
    df = pd.DataFrame(
        {
            "frame_idx": np.arange(n),
            "cc_count": cc,
            "ellipse_eccentricity": static,
            "circularity": static,
            "solidity": static,
            "num_dt_peaks": static,
            "furthest_endpoint_distance_px": static,
        }
    )
    out = add_temporal_occlusion_features(df)

    # Non-occlusion rows remain NaN for temporal columns.
    assert not np.isfinite(out.loc[0, "ellipse_eccentricity_mean_5"])
    assert not np.isfinite(out.loc[13, "circularity_delta_5"])

    # Bout A mean_5 uses source-frame windows within the bout only.
    bout_a = out.loc[5:11, "ellipse_eccentricity_mean_5"].to_numpy()
    assert np.isfinite(bout_a).all()
    assert bout_a[0] == pytest.approx(5.0)
    assert bout_a[4] == pytest.approx(np.mean([5, 6, 7, 8, 9]))

    # Delta at bout-local lag 5: first five of bout are NaN, then value-5.
    delta = out.loc[5:11, "ellipse_eccentricity_delta_5"].to_numpy()
    assert not np.isfinite(delta[:5]).any()
    assert delta[5] == pytest.approx(10.0 - 5.0)

    # Second bout does not leak history from first bout.
    assert out.loc[14, "ellipse_eccentricity_mean_5"] == pytest.approx(14.0)


def test_source_to_target_mapping_uses_prepared_frame_space() -> None:
    source_n = 5
    fps = 20.0
    source_times = np.arange(source_n, dtype=float) / fps
    timing = build_nearest_neighbor_timing_map(
        source_frame_idx=np.arange(source_n, dtype=np.int64),
        source_time_sec=source_times,
        target_fps=120.0,
    )
    df_source = pd.DataFrame(
        {
            "frame_idx": np.arange(source_n),
            "cc_count": [2.0, 2.0, 1.0, 1.0, 2.0],
            "cc_count_raw": [2, 2, 1, 1, 2],
            "min_inter_contour_dist": [4.0, 5.0, 0.0, 0.0, 6.0],
            **{name: np.arange(source_n, dtype=float) for name in OCCLUSION_STATIC_FEATURE_COLS},
        }
    )
    df_source = apply_source_temporal_pipeline(df_source, max_gap_frames=0)
    mapped = map_source_features_to_target(df_source, timing)

    assert len(mapped) == timing.target_sample_count
    assert mapped["dense_frame_idx"].tolist() == list(range(len(mapped)))
    assert mapped["frame_idx"].tolist() == mapped["dense_frame_idx"].tolist()
    assert mapped["sample_idx"].tolist() == mapped["dense_frame_idx"].tolist()
    np.testing.assert_array_equal(mapped["orig_frame_idx"], timing.orig_frame_idx)
    # Feature values come from the mapped source frame, not interpolated.
    for row_i, src_i in enumerate(timing.orig_frame_idx.tolist()):
        assert mapped.loc[row_i, "cc_count"] == pytest.approx(
            float(df_source.loc[src_i, "cc_count"])
        )
