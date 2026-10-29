"""Read-only E-session synchronization smoke test for S9 Sprint 1.

Does not modify published E-session artifacts. Skips when the lab session or
requested S8 run is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from results_packaging.bout_mapping import map_bout_boundaries_to_prepared_frames
from results_packaging.runner import (
    ResultsPackagingRequest,
    run_results_packaging_sprint1,
)
from results_packaging.sync_index import validate_sync_index

SESSION = Path(r"C:\behavior_suite\test_cases\E_full_session_integration")
S8_RUN = SESSION / "behavioral_arbitration" / "behavioral_arbitration__20261028T133953"


def _e_session_s8_currently_eligible() -> bool:
    """Packaging smoke requires live S8→S7 eligibility (not only directory presence)."""

    if not S8_RUN.is_dir():
        return False
    try:
        from behavioral_arbitration.runner import is_s8_run_eligible_for_downstream

        return bool(is_s8_run_eligible_for_downstream(S8_RUN))
    except Exception:  # noqa: BLE001 - treat any eligibility failure as skip
        return False


pytestmark = pytest.mark.skipif(
    not _e_session_s8_currently_eligible(),
    reason=(
        "E-session S8 behavioral_arbitration__20261028T133953 is not currently "
        "eligible (upstream S7 model artifacts missing/stale under legacy/models); "
        "packaging smoke deferred — not an S9 export regression"
    ),
)


@pytest.fixture(scope="module")
def sprint1_result():
    return run_results_packaging_sprint1(
        ResultsPackagingRequest(
            session_root=SESSION,
            s8_run_dir=S8_RUN,
        )
    )


def test_e_session_dense_count_and_sync_rows(sprint1_result) -> None:
    # Authoritative dense sample count from published S8 states.
    states = pd.read_parquet(S8_RUN / "behavior_states.parquet", columns=["dense_frame_idx"])
    n_dense = int(states["dense_frame_idx"].nunique())
    assert n_dense == 36528
    assert sprint1_result.source.n_dense_samples == n_dense
    assert sprint1_result.sync.n_dense == n_dense
    assert len(sprint1_result.sync.sync_index) == n_dense
    validate_sync_index(sprint1_result.sync.sync_index)


def test_e_session_prepared_mapping_audit(sprint1_result) -> None:
    sync = sprint1_result.sync.sync_index
    prepared = sync["prepared_frame_idx"].to_numpy(dtype=np.int64)
    unique = np.unique(prepared)
    audit = sprint1_result.sync.audit_dict()

    assert audit["n_prepared_unique"] == int(unique.shape[0])
    assert audit["prepared_frame_idx_min"] == int(prepared.min())
    assert audit["prepared_frame_idx_max"] == int(prepared.max())
    # Dense timeline is not interchangeable with prepared indices on E-session.
    assert audit["n_dense"] != audit["n_prepared_unique"]
    assert audit["monotonic_non_decreasing"] is True
    assert np.all(np.diff(prepared) >= 0)

    residuals = sync["timing_residual_sec"].to_numpy(dtype=np.float64)
    assert np.all(np.isfinite(residuals))
    assert audit["timing_residual_abs_max"] == pytest.approx(float(np.max(np.abs(residuals))))
    # Exact-match flag consistency already enforced by validate_sync_index.
    assert audit["n_exact_matches"] == int(sync["is_exact_source_sample"].sum())

    # Out-of-domain samples must be zero for a successful build.
    assert int(prepared.min()) >= 0
    assert int(prepared.max()) < audit["n_prepared_source"]


def test_e_session_full_bout_membership_and_boundary_map(sprint1_result) -> None:
    bouts = pd.read_parquet(S8_RUN / "behavior_bouts.parquet")
    assert len(bouts) == 609
    assert sprint1_result.source.n_bouts == 609
    assert len(sprint1_result.bout_boundary_map) == 609

    mapped = map_bout_boundaries_to_prepared_frames(
        bouts, sprint1_result.sync.sync_index, fps=120.0
    )
    assert np.array_equal(
        mapped["bout_duration_frames"].to_numpy(dtype=np.int64),
        bouts["bout_duration_frames"].to_numpy(dtype=np.int64),
    )
    assert np.array_equal(
        mapped["start_sample_idx"].to_numpy(dtype=np.int64),
        bouts["start_sample_idx"].to_numpy(dtype=np.int64),
    )
    assert np.array_equal(
        mapped["end_sample_idx"].to_numpy(dtype=np.int64),
        bouts["end_sample_idx"].to_numpy(dtype=np.int64),
    )
    # Prepared endpoints may be equal even when dense duration > 1 (many-to-one).
    assert (mapped["prepared_frame_idx_start"] <= mapped["prepared_frame_idx_end"]).all()


def test_e_session_no_artifact_mutation(sprint1_result) -> None:
    from project.file_digest import sha256_file

    # Re-hash after sprint1; values must match pre-recorded digests in meta where present.
    states = S8_RUN / "behavior_states.parquet"
    bouts = S8_RUN / "behavior_bouts.parquet"
    sync = sprint1_result.source.sync_video.prepared_sync_path
    before_states = sprint1_result.source.states_sha256
    before_bouts = sprint1_result.source.bouts_sha256
    before_sync = sprint1_result.source.sync_video.prepared_sync_sha256
    assert sha256_file(states) == before_states
    assert sha256_file(bouts) == before_bouts
    assert sha256_file(sync) == before_sync
    # Sprint 1 itself does not publish; Sprint 3 may create results_package/.
    meta = sprint1_result.to_metadata()
    # Capture audit numbers for documentation (assert shape only).
    assert meta["sync_audit"]["n_dense"] == 36528
    assert meta["n_bouts"] == 609
