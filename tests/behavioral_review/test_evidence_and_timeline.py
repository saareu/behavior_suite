"""Sprint 2 lazy evidence and timeline LOD tests."""

from __future__ import annotations

from pathlib import Path

from behavioral_review.bouts import BoutFilter
from behavioral_review.contracts import (
    EVIDENCE_INVALID,
    EVIDENCE_UNAVAILABLE,
    EVIDENCE_UNOBSERVED,
    EVIDENCE_VALID_NEGATIVE,
    EVIDENCE_VALID_POSITIVE,
    LABEL_ENGAGED_VALUE,
)
from behavioral_review.evidence import classify_prediction_display
from behavioral_review.session import open_inspection_session
from behavioral_review.source import (
    assert_no_upstream_writes,
    snapshot_upstream_fingerprints,
)
from behavioral_review.timeline import timeline_view_to_dict
from tests.behavioral_review._fixtures import (
    build_mini_review_package,
    patch_s9_currently_eligible,
)


def _open_session(tmp_path: Path):
    built = build_mini_review_package(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        session = open_inspection_session(
            built["package_dir"],
            require_video=True,
            require_s8_states=True,
        )
    return built, session


def test_s8_echo_and_lazy_s7_evidence(tmp_path: Path) -> None:
    _, session = _open_session(tmp_path)
    session.select_direction(focal_id=0, partner_id=1)
    echo = session.s8_echoed_evidence(start_sample_idx=0, end_sample_idx=3)
    assert echo.availability == "available"
    assert len(echo.points) > 0
    assert all(p.source == "s8_echo" for p in echo.points)

    s7 = session.s7_evidence(
        start_sample_idx=0,
        end_sample_idx=3,
        detectors=["approach", "contact"],
    )
    assert session.evidence.s7_available is True
    assert s7.availability == "available"
    assert not s7.frame.empty
    assert set(s7.frame["detector_name"].astype(str)) <= {"approach", "contact"}
    row = s7.frame.iloc[0]
    assert int(row["prediction"]) in (0, 1)


def test_missing_vs_valid_negative_and_invalid(tmp_path: Path) -> None:
    assert (
        classify_prediction_display(
            input_valid=True, score_origin="model_observed", prediction=0, available=True
        )
        == EVIDENCE_VALID_NEGATIVE
    )
    assert (
        classify_prediction_display(
            input_valid=True, score_origin="model_observed", prediction=1, available=True
        )
        == EVIDENCE_VALID_POSITIVE
    )
    assert (
        classify_prediction_display(
            input_valid=False,
            score_origin="routing_sentinel",
            prediction=0,
            available=True,
        )
        == EVIDENCE_INVALID
    )
    assert (
        classify_prediction_display(
            input_valid=True,
            score_origin="routing_sentinel",
            prediction=0,
            available=True,
        )
        == EVIDENCE_UNOBSERVED
    )
    assert (
        classify_prediction_display(
            input_valid=None, score_origin=None, prediction=None, available=False
        )
        == "missing"
    )

    _, session = _open_session(tmp_path)
    s7 = session.s7_evidence(
        start_sample_idx=8, end_sample_idx=8, detectors=["approach"]
    )
    invalid_points = [p for p in s7.points if p.display_state == EVIDENCE_INVALID]
    assert invalid_points, "expected invalid/routing approach evidence at sample 8"


def test_s8_reconstruction_provenance_and_missing_diagnostics(tmp_path: Path) -> None:
    _, session = _open_session(tmp_path)
    session.select_direction(focal_id=0, partner_id=1)
    session.seek_sample(5)
    prov = session.provenance_at_playhead()
    assert prov is not None
    assert prov.behavior_label == LABEL_ENGAGED_VALUE
    assert prov.label_reconstruction is not None
    assert prov.stage_diagnostics_availability == EVIDENCE_UNAVAILABLE
    assert prov.stage_diagnostics == ()
    assert any("diagnostics" in w.lower() for w in prov.warnings)

    bout = session.source.behavior_bouts.iloc[0]
    session.select_bout(str(bout["bout_id"]))
    bout_prov = session.provenance_for_selected_bout()
    assert bout_prov is not None
    assert bout_prov.scope == "bout"
    assert bout_prov.bout_id == str(bout["bout_id"])


def test_timeline_lod_does_not_change_science(tmp_path: Path) -> None:
    _, session = _open_session(tmp_path)
    session.select_direction(focal_id=0, partner_id=1)
    full = session.timeline_view(
        start_sample_idx=0, end_sample_idx=9, max_segments=10_000
    )
    assert full.scientific_values_unchanged is True
    assert full.lod_level == "full"
    assert len(full.lanes) >= 3
    overview = session.timeline_view(
        start_sample_idx=0, end_sample_idx=9, max_segments=2
    )
    assert overview.scientific_values_unchanged is True
    assert overview.lod_level in {"full", "overview"}
    window = session.ethogram_window(start_sample_idx=0, end_sample_idx=9)
    track = next(t for t in window.directions if t.lane.focal_id == 0)
    assert LABEL_ENGAGED_VALUE in track.behavior_label or "approach" in track.behavior_label
    summary = timeline_view_to_dict(overview)
    assert summary["scientific_values_unchanged"] is True


def test_playhead_sync_plateau_flag(tmp_path: Path) -> None:
    _, session = _open_session(tmp_path)
    session.seek_sample(0)
    view0 = session.timeline_view(start_sample_idx=0, end_sample_idx=3)
    session.seek_sample(1)
    view1 = session.timeline_view(start_sample_idx=0, end_sample_idx=3)
    assert view1.playhead.prepared_frame_idx == view0.playhead.prepared_frame_idx
    assert view1.playhead.prepared_frame_unchanged is True
    assert len(view1.playhead.plateau_sample_indices) >= 2


def test_read_only_no_upstream_writes(tmp_path: Path) -> None:
    _, session = _open_session(tmp_path)
    snaps = snapshot_upstream_fingerprints(session.source)
    session.seek_sample(3)
    session.ethogram_window(half_window=2)
    session.s8_echoed_evidence(half_window=2)
    session.s7_evidence(half_window=2, detectors=["approach"])
    session.set_bout_filter(BoutFilter(channels=frozenset({"exclusive"})))
    session.next_bout()
    session.timeline_view(half_window=2)
    assert_no_upstream_writes(
        package_dir=session.source.package_dir,
        s8_run_dir=session.source.source_s8.run_dir,
        video_path=session.source.source_video.path,
        snapshots=snaps,
    )
    session_root = session.source.session_root
    assert session_root is not None
    review_dir = session_root / "behavioral_review"
    assert not review_dir.exists() or not any(review_dir.glob("behavioral_review__*"))


def test_missing_s7_reports_unavailable(tmp_path: Path) -> None:
    built = build_mini_review_package(tmp_path)
    preds = (
        built["session_root"]
        / "behavioral_inference"
        / "behavioral_inference__synth"
        / "predictions.parquet"
    )
    preds.unlink()
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        session = open_inspection_session(
            built["package_dir"],
            require_video=True,
            require_s8_states=True,
        )
    s7 = session.s7_evidence(start_sample_idx=0, end_sample_idx=2)
    assert s7.availability == EVIDENCE_UNAVAILABLE
    assert s7.frame.empty
