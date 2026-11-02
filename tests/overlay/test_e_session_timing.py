"""E-session timing evidence for overlay C1 (read-only; skip if ineligible)."""

from __future__ import annotations

from pathlib import Path

import pytest

from behavioral_review.source import (
    assert_no_upstream_writes,
    open_review_source,
    snapshot_upstream_fingerprints,
)
from behavioral_review.sync import SyncMap
from overlay.contracts import PlateauPolicy
from overlay.timing import resolve_prepared_frame_timing
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
    except Exception:  # noqa: BLE001
        return False


def _skip_reason() -> str:
    if not PACKAGE.is_dir():
        return (
            "E-session package results_package__20261028T181714 is missing; "
            "overlay C1 timing smoke deferred"
        )
    try:
        from results_packaging.publication import validate_s9_run_for_downstream

        validate_s9_run_for_downstream(PACKAGE)
        return "unexpected: package became eligible during skip-reason build"
    except (PublicationError, ProvenanceError) as exc:
        return (
            "E-session package results_package__20261028T181714 is not currently "
            f"eligible for overlay timing smoke ({exc}); report missing "
            "dependencies rather than weakening eligibility"
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


def test_e_session_last_in_plateau_on_many_to_one(e_source) -> None:
    sync = SyncMap.from_sync_index(e_source.sync_index)
    # Find a prepared frame with plateau size > 1.
    prepared = sync.prepared_frame_idx
    values, counts = __import__("numpy").unique(prepared, return_counts=True)
    multi = values[counts > 1]
    assert multi.size > 0
    prep = int(multi[0])
    plateau = sync.plateau_sample_indices(prep)
    timing = resolve_prepared_frame_timing(
        sync, prep, plateau_policy=PlateauPolicy.LAST_IN_PLATEAU
    )
    assert timing.available
    assert timing.dense_sample_idx == int(plateau[-1])
    assert timing.dense_sample_idx == int(max(plateau))

    nearest = resolve_prepared_frame_timing(
        sync, prep, plateau_policy=PlateauPolicy.NEAREST_TIME
    )
    assert nearest.available
    assert nearest.dense_sample_idx == sync.mapping_from_prepared_frame(prep).sample_idx


def test_e_session_first_last_prepared_frames(e_source) -> None:
    sync = SyncMap.from_sync_index(e_source.sync_index)
    first_prep = int(sync.prepared_frame_idx.min())
    last_prep = int(sync.prepared_frame_idx.max())
    first = resolve_prepared_frame_timing(sync, first_prep)
    last = resolve_prepared_frame_timing(sync, last_prep)
    assert first.available and last.available
    assert first.dense_sample_idx == int(max(sync.plateau_sample_indices(first_prep)))
    assert last.dense_sample_idx == int(max(sync.plateau_sample_indices(last_prep)))


def test_e_session_timing_read_only(e_source) -> None:
    snaps = snapshot_upstream_fingerprints(e_source)
    sync = SyncMap.from_sync_index(e_source.sync_index)
    for prep in (0, 1000, int(sync.prepared_frame_idx.max())):
        resolve_prepared_frame_timing(sync, prep)
    assert_no_upstream_writes(
        package_dir=e_source.package_dir,
        s8_run_dir=e_source.source_s8.run_dir,
        video_path=e_source.source_video.path,
        snapshots=snaps,
    )
