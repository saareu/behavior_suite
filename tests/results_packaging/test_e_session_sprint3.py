"""E-session Sprint 3 immutable publication validation.

Publishes one real S9 run against the validated S8 source and verifies
cardinality, eligibility, and no upstream mutation. Skips when absent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from project.file_digest import sha256_file
from results_packaging.config import ExecutionSettings, ResultsPackagingConfig
from results_packaging.contracts import (
    BEHAVIOR_BOUTS_FILENAME,
    ETHOGRAM_SUMMARY_FILENAME,
    PACKAGE_MANIFEST_FILENAME,
    QC_FILENAME,
    REVIEW_EMIT_PROFILE_ID,
    RUN_META_FILENAME,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    SYNC_INDEX_FILENAME,
)
from results_packaging.publication import (
    is_s9_run_eligible_for_downstream,
    reopen_results_package,
    run_results_packaging_publication,
)
from results_packaging.run_discovery import find_latest_eligible_s9_run
from results_packaging.runner import ResultsPackagingRequest

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
def published_run():
    states_sha = sha256_file(S8_RUN / "behavior_states.parquet")
    bouts_sha = sha256_file(S8_RUN / "behavior_bouts.parquet")
    cfg = ResultsPackagingConfig(
        execution=ExecutionSettings(
            selected_export_profiles=(REVIEW_EMIT_PROFILE_ID,),
            publish_run=True,
        )
    )
    result = run_results_packaging_publication(
        ResultsPackagingRequest(
            session_root=SESSION,
            s8_run_dir=S8_RUN,
            config=cfg,
            run_purpose="e_session_sprint3_validation",
        )
    )
    assert result.success, result.error_message
    assert result.status in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS)
    # Upstream S8 unchanged.
    assert sha256_file(S8_RUN / "behavior_states.parquet") == states_sha
    assert sha256_file(S8_RUN / "behavior_bouts.parquet") == bouts_sha
    return result


def test_e_session_publication_layout_and_cardinality(published_run) -> None:
    run_dir = published_run.run_dir
    for name in (
        SYNC_INDEX_FILENAME,
        BEHAVIOR_BOUTS_FILENAME,
        ETHOGRAM_SUMMARY_FILENAME,
        PACKAGE_MANIFEST_FILENAME,
        QC_FILENAME,
        RUN_META_FILENAME,
        "settings_used.yaml",
        "processing_log.txt",
    ):
        assert (run_dir / name).is_file()

    sync = pd.read_parquet(run_dir / SYNC_INDEX_FILENAME)
    bouts = pd.read_parquet(run_dir / BEHAVIOR_BOUTS_FILENAME)
    assert len(sync) == 36528
    assert len(bouts) == 609
    by_channel = bouts.groupby("channel").size().to_dict()
    assert by_channel.get("exclusive") == 538
    assert by_channel.get("contact") == 60
    assert by_channel.get("side_by_side") == 11

    # Authoritative dense boundaries / durations preserved vs S8.
    # Package table is stably re-sorted; compare by source id, not row order.
    s8 = pd.read_parquet(S8_RUN / "behavior_bouts.parquet")
    left = (
        s8.assign(source_s8_bout_id=s8["bout_id"].astype(str))
        .set_index("source_s8_bout_id")
        .sort_index()
    )
    right = bouts.set_index("source_s8_bout_id").sort_index()
    for col in (
        "start_sample_idx",
        "end_sample_idx",
        "bout_duration_frames",
        "bout_duration_sec",
        "channel",
        "behavior_name",
    ):
        assert left[col].equals(right[col])

    # Prepared navigation aids present.
    for col in (
        "prepared_frame_idx_start",
        "prepared_frame_idx_end",
        "prepared_time_sec_start",
        "prepared_time_sec_end",
    ):
        assert col in bouts.columns


def test_e_session_manifest_qc_eligibility(published_run) -> None:
    run_dir = published_run.run_dir
    manifest = json.loads((run_dir / PACKAGE_MANIFEST_FILENAME).read_text(encoding="utf-8"))
    qc = json.loads((run_dir / QC_FILENAME).read_text(encoding="utf-8"))
    meta = json.loads((run_dir / RUN_META_FILENAME).read_text(encoding="utf-8"))

    assert manifest["full_bout_preservation"] is True
    assert manifest["portability"]["self_contained"] is False
    assert manifest["row_counts"]["sync_index"] == 36528
    assert manifest["row_counts"]["behavior_bouts"] == 609
    assert manifest["source_references"]["source_s8"]["run_id"] == (
        "behavioral_arbitration__20261028T133953"
    )
    assert qc["structural_ok"] is True
    assert meta["status"] == published_run.status
    assert meta["science_written"] is True
    assert is_s9_run_eligible_for_downstream(run_dir)

    latest = find_latest_eligible_s9_run(SESSION)
    assert latest is not None
    assert latest.run_id == published_run.run_id

    opened = reopen_results_package(run_dir)
    assert opened["currently_eligible"] is True
    assert len(opened["behavior_bouts"]) == 609


def test_e_session_optional_review_emit_profile(published_run) -> None:
    profile_dir = (
        published_run.run_dir / "export_profiles" / REVIEW_EMIT_PROFILE_ID
    )
    assert profile_dir.is_dir()
    meta = json.loads((profile_dir / "profile_manifest.json").read_text(encoding="utf-8"))
    # Measured Sprint 2 result on this S8 source/settings: 514 included / 95 excluded.
    assert meta["n_included"] == 514
    assert meta["n_excluded"] == 95
    assert meta["n_included"] + meta["n_excluded"] == 609
