"""Tests for lossless S9 bout packaging and stable identities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from results_packaging.bouts import (
    assert_lossless_bout_copy,
    authoritative_bout_content_key,
    build_package_bout_id,
    package_behavior_bouts,
)
from results_packaging.contracts import BOUTS_SCHEMA_VERSION, BoutPackagingError
from results_packaging.sync_index import build_sync_index_from_prepared_sync
from tests.results_packaging._fixtures import (
    MOUSE_A,
    MOUSE_B,
    build_packaging_session,
    make_multi_channel_bouts,
)


def _sync_for_session(fixture: dict) -> pd.DataFrame:
    unique = pd.DataFrame(
        {
            "dense_frame_idx": np.arange(fixture["n_dense"], dtype=np.int64),
            "time_sec": np.asarray(fixture["time_sec"], dtype=np.float64),
        }
    )
    result = build_sync_index_from_prepared_sync(
        fixture["prepared_sync"],
        s8_unique_dense_times=unique,
        target_fps=120.0,
    )
    return result.sync_index


def test_lossless_preservation_and_cardinality(tmp_path) -> None:
    fixture = build_packaging_session(tmp_path, n_prepared=20, prepared_fps=30.0)
    s8 = make_multi_channel_bouts()
    # Ensure intervals fit dense domain.
    assert int(s8["end_sample_idx"].max()) < int(fixture["n_dense"])
    sync = _sync_for_session(fixture)

    packaged = package_behavior_bouts(
        s8,
        sync,
        source_s8_run_id="behavioral_arbitration__synth",
        package_run_id="results_package__test",
    )
    assert len(packaged) == len(s8) == 9
    assert_lossless_bout_copy(s8, packaged)
    assert (packaged["schema_version"] == BOUTS_SCHEMA_VERSION).all()
    assert (packaged["s8_bouts_schema_version"] == s8["schema_version"]).all()
    assert set(packaged["source_s8_bout_id"]) == set(s8["bout_id"].astype(str))


def test_stable_deterministic_package_ids_independent_of_row_order(tmp_path) -> None:
    fixture = build_packaging_session(tmp_path, n_prepared=20, prepared_fps=30.0)
    s8 = make_multi_channel_bouts()
    sync = _sync_for_session(fixture)

    pkg_a = package_behavior_bouts(
        s8, sync, source_s8_run_id="behavioral_arbitration__synth"
    )
    pkg_b = package_behavior_bouts(
        s8.sample(frac=1.0, random_state=0).reset_index(drop=True),
        sync,
        source_s8_run_id="behavioral_arbitration__synth",
    )
    # Same multiset of package ids regardless of input row order.
    assert sorted(pkg_a["bout_id"]) == sorted(pkg_b["bout_id"])
    assert sorted(pkg_a["source_s8_bout_id"]) == sorted(pkg_b["source_s8_bout_id"])
    # Deterministic sort order of the packaged table.
    assert pkg_a["bout_id"].tolist() == pkg_b["bout_id"].tolist()

    for _, row in pkg_a.iterrows():
        expected = build_package_bout_id(
            source_s8_run_id="behavioral_arbitration__synth",
            source_s8_bout_id=str(row["source_s8_bout_id"]),
        )
        assert row["bout_id"] == expected


def test_directional_vs_symmetric_identity(tmp_path) -> None:
    fixture = build_packaging_session(tmp_path, n_prepared=20, prepared_fps=30.0)
    s8 = make_multi_channel_bouts()
    sync = _sync_for_session(fixture)
    packaged = package_behavior_bouts(
        s8, sync, source_s8_run_id="behavioral_arbitration__synth"
    )

    directional = packaged[packaged["identity_grain"] == "directional"]
    symmetric = packaged[packaged["identity_grain"] == "symmetric_pair"]
    assert not directional.empty and not symmetric.empty
    assert directional["focal_id"].notna().all()
    assert directional["partner_id"].notna().all()
    assert symmetric["focal_id"].isna().all()
    assert symmetric["partner_id"].isna().all()

    keys = [authoritative_bout_content_key(r) for _, r in packaged.iterrows()]
    assert len(keys) == len(set(keys))
    # Opposite directions produce distinct content keys.
    ab = packaged[
        (packaged["channel"] == "exclusive")
        & (packaged["focal_id"] == MOUSE_A)
        & (packaged["partner_id"] == MOUSE_B)
    ].iloc[0]
    ba = packaged[
        (packaged["channel"] == "exclusive")
        & (packaged["focal_id"] == MOUSE_B)
        & (packaged["partner_id"] == MOUSE_A)
    ].iloc[0]
    assert authoritative_bout_content_key(ab) != authoritative_bout_content_key(ba)


def test_inclusive_endpoints_and_prepared_navigation(tmp_path) -> None:
    fixture = build_packaging_session(tmp_path, n_prepared=5, prepared_fps=30.0)
    sync = _sync_for_session(fixture)
    # Many dense samples share prepared frames on 30→120 upsample.
    s8 = make_multi_channel_bouts().iloc[0:1].copy()
    s8.loc[:, "start_sample_idx"] = 2
    s8.loc[:, "end_sample_idx"] = 4
    s8.loc[:, "bout_duration_frames"] = 3
    s8.loc[:, "bout_duration_sec"] = 3.0 / 120.0
    s8.loc[:, "bout_id"] = "bout_00000000"

    packaged = package_behavior_bouts(
        s8, sync, source_s8_run_id="behavioral_arbitration__synth"
    )
    row = packaged.iloc[0]
    assert int(row["start_sample_idx"]) == 2
    assert int(row["end_sample_idx"]) == 4
    assert int(row["bout_duration_frames"]) == 3
    # Navigation annotations present; duration not recomputed from prepared span.
    assert pd.notna(row["prepared_frame_idx_start"])
    assert pd.notna(row["prepared_frame_idx_end"])
    prepared_span = int(row["prepared_frame_idx_end"]) - int(
        row["prepared_frame_idx_start"]
    )
    assert prepared_span <= 2  # may be 0 when many-to-one
    assert float(row["bout_duration_sec"]) == pytest.approx(3.0 / 120.0)


def test_repeated_prepared_frame_mappings_allowed(tmp_path) -> None:
    fixture = build_packaging_session(tmp_path, n_prepared=5, prepared_fps=30.0)
    sync = _sync_for_session(fixture)
    prepared = sync["prepared_frame_idx"].to_numpy(dtype=np.int64)
    # Confirm many-to-one exists in this fixture.
    assert len(prepared) > len(np.unique(prepared))

    s8 = make_multi_channel_bouts()
    s8 = s8[s8["end_sample_idx"] < fixture["n_dense"]].reset_index(drop=True)
    packaged = package_behavior_bouts(
        s8, sync, source_s8_run_id="behavioral_arbitration__synth"
    )
    # Shared prepared start/end with dense duration > 1 is valid.
    multi = packaged[packaged["bout_duration_frames"] > 1]
    assert (
        multi["prepared_frame_idx_start"] <= multi["prepared_frame_idx_end"]
    ).all()


def test_rejects_duplicate_source_bout_ids(tmp_path) -> None:
    fixture = build_packaging_session(tmp_path, n_prepared=5, prepared_fps=30.0)
    sync = _sync_for_session(fixture)
    s8 = make_multi_channel_bouts().iloc[0:2].copy()
    s8["bout_id"] = "dup"
    with pytest.raises(BoutPackagingError, match="not unique"):
        package_behavior_bouts(
            s8, sync, source_s8_run_id="behavioral_arbitration__synth"
        )
