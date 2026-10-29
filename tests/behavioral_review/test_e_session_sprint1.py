"""E-session smoke for S10 Sprint 1 (read-only; no upstream mutation).

Uses eligible package ``results_package__20261028T181714`` when present and
currently eligible. Reports missing dependencies via skip rather than weakening
eligibility checks.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from behavioral_review.session import open_inspection_session
from behavioral_review.source import (
    assert_no_upstream_writes,
    open_review_source,
    snapshot_upstream_fingerprints,
)
from behavioral_review.sync import SyncMap
from results_packaging.publication import (
    ProvenanceError,
    PublicationError,
    is_s9_run_eligible_for_downstream,
)

SESSION = Path(r"C:\behavior_suite\test_cases\E_full_session_integration")
PACKAGE = (
    SESSION
    / "results_package"
    / "results_package__20261028T181714"
)


def _e_session_package_currently_eligible() -> bool:
    if not PACKAGE.is_dir():
        return False
    try:
        return bool(is_s9_run_eligible_for_downstream(PACKAGE))
    except Exception:  # noqa: BLE001 - treat any eligibility failure as skip
        return False


def _skip_reason() -> str:
    if not PACKAGE.is_dir():
        return (
            "E-session package results_package__20261028T181714 is missing; "
            "S10 Sprint 1 smoke deferred"
        )
    try:
        from results_packaging.publication import validate_s9_run_for_downstream

        validate_s9_run_for_downstream(PACKAGE)
        return "unexpected: package became eligible during skip-reason build"
    except (PublicationError, ProvenanceError) as exc:
        return (
            "E-session package results_package__20261028T181714 is not currently "
            f"eligible for S10 open ({exc}); report missing dependencies rather "
            "than weakening eligibility"
        )
    except Exception as exc:  # noqa: BLE001
        return f"E-session package eligibility check failed: {exc}"


pytestmark = pytest.mark.skipif(
    not _e_session_package_currently_eligible(),
    reason=_skip_reason(),
)


@pytest.fixture(scope="module")
def e_source():
    return open_review_source(PACKAGE, require_video=True, require_s8_states=False)


@pytest.fixture(scope="module")
def e_session(e_source):
    return open_inspection_session(
        PACKAGE,
        require_video=True,
        require_s8_states=False,
        decode_initial_frame=False,
        video_cache_size=4,
    )


def test_e_session_exact_source_identity(e_source) -> None:
    assert e_source.package_run_id == "results_package__20261028T181714"
    assert e_source.currently_eligible is True
    assert e_source.source_s8.run_id == "behavioral_arbitration__20261028T133953"
    assert e_source.n_dense_samples == 36528
    assert e_source.n_bouts == 609
    assert e_source.source_video.available is True
    assert e_source.source_video.path is not None
    assert e_source.source_video.path.name == "prepared_video.mp4"


def test_e_session_sync_mapping_counts(e_source) -> None:
    sync = SyncMap.from_sync_index(e_source.sync_index)
    audit = sync.audit_dict()
    assert sync.n_dense == 36528
    assert audit["n_prepared_unique"] == 36238
    assert audit["n_dense"] != audit["n_prepared_unique"]
    assert audit["prepared_frame_idx_min"] == 0
    assert audit["prepared_frame_idx_max"] == 36237
    # First / middle / last sample mappings are deterministic.
    first = sync.mapping_at_sample(0)
    mid = sync.mapping_at_sample(18264)
    last = sync.mapping_at_sample(36527)
    assert first.prepared_frame_idx == int(e_source.sync_index.loc[0, "prepared_frame_idx"])
    assert mid.prepared_frame_idx == int(
        e_source.sync_index.loc[18264, "prepared_frame_idx"]
    )
    assert last.prepared_frame_idx == int(
        e_source.sync_index.loc[36527, "prepared_frame_idx"]
    )


def test_e_session_first_middle_last_video_seek(e_session) -> None:
    for sample in (0, 18264, 36527):
        mapping = e_session.seek_sample(sample, decode=True)
        frame = e_session.decode_current_frame()
        assert mapping.sample_idx == sample
        assert frame.prepared_frame_idx == mapping.prepared_frame_idx
        assert isinstance(frame.image_bgr, np.ndarray)
        assert frame.image_bgr.ndim == 3


def test_e_session_representative_bout_seeks(e_session) -> None:
    bouts = e_session.source.behavior_bouts
    # Directional exclusive, contact, side_by_side representatives.
    exclusive = bouts.loc[bouts["channel"] == "exclusive"].iloc[0]
    contact = bouts.loc[bouts["channel"] == "contact"].iloc[0]
    sbs = bouts.loc[bouts["channel"] == "side_by_side"].iloc[0]

    m_ex = e_session.seek_bout_start(str(exclusive["bout_id"]), decode=True)
    assert e_session.selected_channel == "exclusive"
    assert e_session.selected_direction.focal_id == int(exclusive["focal_id"])
    assert m_ex.prepared_frame_idx_start == int(exclusive["prepared_frame_idx_start"])

    m_ct = e_session.seek_bout_end(str(contact["bout_id"]), decode=True)
    assert e_session.selected_channel == "contact"
    assert e_session.selected_direction.focal_id is None
    assert m_ct.prepared_frame_idx_end == int(contact["prepared_frame_idx_end"])

    m_sbs = e_session.seek_bout_start(str(sbs["bout_id"]), decode=True)
    assert e_session.selected_channel == "side_by_side"
    assert m_sbs.start.sample_idx == int(sbs["start_sample_idx"])


def test_e_session_no_upstream_writes(e_source, e_session) -> None:
    snaps = snapshot_upstream_fingerprints(e_source)
    e_session.seek_sample(100, decode=True)
    e_session.seek_sample(101, decode=True)  # possible plateau keep-frame
    assert_no_upstream_writes(
        package_dir=e_source.package_dir,
        s8_run_dir=e_source.source_s8.run_dir,
        video_path=e_source.source_video.path,
        snapshots=snaps,
    )
    # Opening must not create a review draft.
    assert not (SESSION / "behavioral_review").exists() or not any(
        (SESSION / "behavioral_review").glob("behavioral_review__*")
    )


def test_e_session_packaged_bouts_match_row_count(e_source) -> None:
    assert len(e_source.behavior_bouts) == 609
    grain = e_source.behavior_bouts["identity_grain"].value_counts().to_dict()
    assert grain.get("directional", 0) + grain.get("symmetric_pair", 0) == 609
