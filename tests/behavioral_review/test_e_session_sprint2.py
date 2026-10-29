"""E-session smoke for S10 Sprint 2 inspection (read-only).

Uses eligible package ``results_package__20261028T181714`` when present and
currently eligible.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from behavioral_review.bouts import BoutFilter
from behavioral_review.contracts import EVIDENCE_UNAVAILABLE
from behavioral_review.session import open_inspection_session
from behavioral_review.source import (
    assert_no_upstream_writes,
    open_review_source,
    snapshot_upstream_fingerprints,
)
from results_packaging.publication import (
    ProvenanceError,
    PublicationError,
    is_s9_run_eligible_for_downstream,
)

SESSION = Path(r"C:\behavior_suite\test_cases\E_full_session_integration")
PACKAGE = SESSION / "results_package" / "results_package__20261028T181714"


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
            "S10 Sprint 2 smoke deferred"
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
def e_session():
    return open_inspection_session(
        PACKAGE,
        require_video=True,
        require_s8_states=True,
        decode_initial_frame=False,
        video_cache_size=4,
    )


def test_e_session_sprint2_scale_and_lanes(e_session) -> None:
    assert e_session.n_dense_samples == 36528
    assert e_session.n_bouts == 609
    assert e_session.source.s8_states_available is True
    directions = e_session.ethogram.discover_directions(sample_idx=1000, half_window=30)
    assert len(directions) == 2
    e_session.select_direction(focal_id=directions[0][0], partner_id=directions[0][1])
    e_session.seek_sample(1000)
    window = e_session.ethogram_window(half_window=60)
    assert len(window.directions) == 2
    assert window.contact is not None
    assert window.side_by_side is not None
    assert window.contact.lane.channel == "contact"
    assert window.side_by_side.lane.channel == "side_by_side"


def test_e_session_bout_navigation(e_session) -> None:
    exclusive = e_session.bouts.list_bouts(
        BoutFilter(channels=frozenset({"exclusive"}))
    )
    contact = e_session.bouts.list_bouts(BoutFilter(channels=frozenset({"contact"})))
    sbs = e_session.bouts.list_bouts(
        BoutFilter(channels=frozenset({"side_by_side"}))
    )
    assert len(exclusive) + len(contact) + len(sbs) == 609
    bout_id = str(exclusive.iloc[0]["bout_id"])
    e_session.set_bout_filter(BoutFilter(channels=frozenset({"exclusive"})))
    mapped = e_session.seek_bout_start(bout_id, decode=False)
    assert mapped.start.sample_idx == int(exclusive.iloc[0]["start_sample_idx"])
    nxt = e_session.bouts.next_bout()
    assert nxt is not None
    e_session.seek_bout_end(nxt, decode=False)
    assert e_session.selected_bout_id == nxt


def test_e_session_lazy_s7_and_s8_provenance(e_session) -> None:
    directions = e_session.ethogram.discover_directions(sample_idx=5000, half_window=10)
    e_session.select_direction(focal_id=directions[0][0], partner_id=directions[0][1])
    e_session.seek_sample(5000)
    echo = e_session.s8_echoed_evidence(half_window=30)
    assert echo.availability == "available"
    assert len(echo.points) > 0
    s7 = e_session.s7_evidence(half_window=15, detectors=["approach", "contact"])
    assert e_session.evidence.s7_available is True
    assert s7.availability in {"available", "missing"}
    prov = e_session.provenance_at_playhead()
    assert prov is not None
    assert prov.behavior_label is not None
    assert prov.label_reconstruction is not None
    # E-session S8 run has no arbitration_diagnostics.parquet.
    assert prov.stage_diagnostics_availability == EVIDENCE_UNAVAILABLE


def test_e_session_windowed_performance(e_session) -> None:
    e_session.seek_sample(18000)
    directions = e_session.ethogram.discover_directions(sample_idx=18000, half_window=5)
    e_session.select_direction(focal_id=directions[0][0], partner_id=directions[0][1])
    t0 = time.perf_counter()
    window = e_session.ethogram_window(half_window=240)
    ethogram_s = time.perf_counter() - t0
    t1 = time.perf_counter()
    _ = e_session.s8_echoed_evidence(half_window=120)
    echo_s = time.perf_counter() - t1
    t2 = time.perf_counter()
    _ = e_session.s7_evidence(half_window=60, detectors=["approach"])
    s7_s = time.perf_counter() - t2
    t3 = time.perf_counter()
    view = e_session.timeline_view(half_window=240, max_segments=2000)
    timeline_s = time.perf_counter() - t3
    assert window.inclusive_span == 481  # ±240 inclusive
    assert view.scientific_values_unchanged is True
    # Soft budgets for workstation smoke (not CI hard gates).
    assert ethogram_s < 5.0
    assert echo_s < 5.0
    assert s7_s < 10.0
    assert timeline_s < 5.0
    # Stash timings on the session warnings for the sprint report / script.
    e_session.warnings.append(
        "perf_sprint2:"
        f"ethogram={ethogram_s:.3f}s,"
        f"echo={echo_s:.3f}s,"
        f"s7={s7_s:.3f}s,"
        f"timeline={timeline_s:.3f}s"
    )


def test_e_session_sprint2_no_writes(e_session) -> None:
    source = open_review_source(PACKAGE, require_video=True, require_s8_states=True)
    snaps = snapshot_upstream_fingerprints(source)
    e_session.seek_sample(100)
    e_session.ethogram_window(half_window=30)
    e_session.s7_evidence(half_window=10, detectors=["contact"])
    e_session.timeline_view(half_window=30)
    assert_no_upstream_writes(
        package_dir=source.package_dir,
        s8_run_dir=source.source_s8.run_dir,
        video_path=source.source_video.path,
        snapshots=snaps,
    )
    assert not (SESSION / "behavioral_review").exists() or not any(
        (SESSION / "behavioral_review").glob("behavioral_review__*")
    )
