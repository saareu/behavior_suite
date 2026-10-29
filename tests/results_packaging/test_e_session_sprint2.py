"""Read-only E-session Sprint 2 packaging validation.

Does not modify published E-session artifacts. Skips when the lab session or
requested S8 run is absent. Filtered export counts are measured, not invented.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from project.file_digest import sha256_file
from results_packaging.bouts import assert_lossless_bout_copy
from results_packaging.config import ExecutionSettings, ResultsPackagingConfig
from results_packaging.contracts import REVIEW_EMIT_PROFILE_ID
from results_packaging.runner import (
    ResultsPackagingRequest,
    run_results_packaging_sprint2,
)

SESSION = Path(r"C:\behavior_suite\test_cases\E_full_session_integration")
S8_RUN = SESSION / "behavioral_arbitration" / "behavioral_arbitration__20261028T133953"


def _e_session_s8_currently_eligible() -> bool:
    if not S8_RUN.is_dir():
        return False
    try:
        from behavioral_arbitration.runner import is_s8_run_eligible_for_downstream

        return bool(is_s8_run_eligible_for_downstream(S8_RUN))
    except Exception:  # noqa: BLE001
        return False


pytestmark = pytest.mark.skipif(
    not _e_session_s8_currently_eligible(),
    reason=(
        "E-session S8 not currently eligible (upstream S7 model artifacts "
        "missing/stale); packaging smoke deferred — not an S9 export regression"
    ),
)


@pytest.fixture(scope="module")
def sprint2_result():
    cfg = ResultsPackagingConfig(
        execution=ExecutionSettings(
            selected_export_profiles=(REVIEW_EMIT_PROFILE_ID,),
        )
    )
    return run_results_packaging_sprint2(
        ResultsPackagingRequest(
            session_root=SESSION,
            s8_run_dir=S8_RUN,
            config=cfg,
            package_run_id="results_package__e_session_sprint2_probe",
        )
    )


def test_e_session_sync_and_bout_cardinality(sprint2_result) -> None:
    s8_bouts = pd.read_parquet(S8_RUN / "behavior_bouts.parquet")
    assert len(sprint2_result.sync.sync_index) == 36528
    assert sprint2_result.source.n_dense_samples == 36528
    assert len(s8_bouts) == 609
    assert len(sprint2_result.behavior_bouts) == 609

    by_channel = (
        sprint2_result.behavior_bouts.groupby("channel").size().to_dict()
    )
    assert by_channel.get("exclusive") == 538
    assert by_channel.get("contact") == 60
    assert by_channel.get("side_by_side") == 11

    short = sprint2_result.behavior_bouts[
        (sprint2_result.behavior_bouts["channel"] == "exclusive")
        & (sprint2_result.behavior_bouts["bout_duration_frames"] < 10)
    ]
    assert len(short) == 31


def test_e_session_full_table_authoritative_comparison(sprint2_result) -> None:
    s8 = pd.read_parquet(S8_RUN / "behavior_bouts.parquet")
    packaged = sprint2_result.behavior_bouts
    assert_lossless_bout_copy(s8, packaged)

    # Independent column-wise comparison of authoritative fields.
    left = s8.assign(source_s8_bout_id=s8["bout_id"].astype(str)).set_index(
        "source_s8_bout_id"
    )
    right = packaged.set_index("source_s8_bout_id")
    cols = [
        "channel",
        "behavior_name",
        "identity_grain",
        "start_sample_idx",
        "end_sample_idx",
        "bout_duration_frames",
        "bout_duration_sec",
        "mouse_a_id",
        "mouse_b_id",
        "pair_id",
    ]
    for col in cols:
        if col in {"start_sample_idx", "end_sample_idx", "bout_duration_frames", "mouse_a_id", "mouse_b_id"}:
            assert np.array_equal(
                left[col].to_numpy(dtype=np.int64),
                right.loc[left.index, col].to_numpy(dtype=np.int64),
            )
        elif col == "bout_duration_sec":
            assert np.allclose(
                left[col].to_numpy(dtype=np.float64),
                right.loc[left.index, col].to_numpy(dtype=np.float64),
                rtol=0.0,
                atol=0.0,
            )
        else:
            assert (
                left[col].astype(str).to_numpy()
                == right.loc[left.index, col].astype(str).to_numpy()
            ).all()

    # All source bout identities preserved.
    assert set(packaged["source_s8_bout_id"]) == set(s8["bout_id"].astype(str))


def test_e_session_navigation_annotations_valid(sprint2_result) -> None:
    packaged = sprint2_result.behavior_bouts
    sync = sprint2_result.sync.sync_index
    assert packaged["prepared_frame_idx_start"].notna().all()
    assert packaged["prepared_frame_idx_end"].notna().all()
    assert (
        packaged["prepared_frame_idx_start"] <= packaged["prepared_frame_idx_end"]
    ).all()
    max_prep = int(sync["prepared_frame_idx"].max())
    assert int(packaged["prepared_frame_idx_end"].max()) <= max_prep
    assert int(packaged["prepared_frame_idx_start"].min()) >= 0


def test_e_session_summary_consistency(sprint2_result) -> None:
    summary = sprint2_result.ethogram_summary
    assert not summary.empty
    bout_total = summary[
        (summary["summary_kind"] == "bout_counts")
        & (summary["metric_name"] == "n_bouts")
        & (summary["channel"] == "*")
    ]
    assert int(bout_total["metric_value"].iloc[0]) == 609
    session_n = summary[
        (summary["summary_kind"] == "session_timing")
        & (summary["metric_name"] == "n_samples")
    ]
    assert int(session_n["metric_value"].iloc[0]) == 36528


def test_e_session_optional_profile_separate_and_measured(sprint2_result) -> None:
    assert len(sprint2_result.behavior_bouts) == 609
    profile = sprint2_result.export_profiles[REVIEW_EMIT_PROFILE_ID]
    assert profile.n_included + profile.n_excluded == 609
    # Measured results — do not hard-code filtered expectations beyond invariants.
    assert profile.parameters["include_contact"] is False
    assert profile.parameters["include_engaged"] is True
    assert profile.parameters["min_required_frames"] == 12
    assert profile.n_included >= 0
    assert profile.n_excluded >= 0
    # Contact must be absent from the review index under default include_contact=false.
    if not profile.bout_index.empty:
        assert (profile.bout_index["channel"] != "contact").all()
    # Short exclusive (<10) remain in the complete package.
    short_pkg = sprint2_result.behavior_bouts[
        (sprint2_result.behavior_bouts["channel"] == "exclusive")
        & (sprint2_result.behavior_bouts["bout_duration_frames"] < 10)
    ]
    assert len(short_pkg) == 31
    if not profile.bout_index.empty:
        short_ids = set(short_pkg["source_s8_bout_id"].astype(str))
        assert short_ids.isdisjoint(
            set(profile.bout_index["source_s8_bout_id"].astype(str))
        )


def test_e_session_no_upstream_mutation(sprint2_result) -> None:
    states_path = S8_RUN / "behavior_states.parquet"
    bouts_path = S8_RUN / "behavior_bouts.parquet"
    sync_path = sprint2_result.source.sync_video.prepared_sync_path
    assert sha256_file(states_path) == sprint2_result.source.states_sha256
    assert sha256_file(bouts_path) == sprint2_result.source.bouts_sha256
    assert (
        sha256_file(sync_path)
        == sprint2_result.source.sync_video.prepared_sync_sha256
    )
    assert sprint2_result.to_metadata()["published"] is False
    # Sprint 2 itself does not publish; Sprint 3 may create results_package/.
