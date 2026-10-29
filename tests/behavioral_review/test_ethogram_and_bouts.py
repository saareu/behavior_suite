"""Sprint 2 ethogram windowing and bout navigation tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from behavioral_review.bouts import BoutBrowser, BoutFilter
from behavioral_review.contracts import (
    EVIDENCE_UNOBSERVED,
    LABEL_ENGAGED_VALUE,
    LABEL_UNDEFINED_VALUE,
    InspectionError,
)
from behavioral_review.ethogram import run_length_segments
from behavioral_review.session import open_inspection_session
from behavioral_review.source import (
    assert_no_upstream_writes,
    snapshot_upstream_fingerprints,
)
from behavioral_review.sync import SyncMap
from tests.behavioral_review._fixtures import (
    build_mini_review_package,
    patch_s9_currently_eligible,
)


def _open_session(tmp_path: Path, **kwargs):
    built = build_mini_review_package(tmp_path, **kwargs)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        session = open_inspection_session(
            built["package_dir"],
            require_video=True,
            require_s8_states=True,
            decode_initial_frame=False,
        )
    return built, session


def test_window_bounds_and_time_alignment(tmp_path: Path) -> None:
    built, session = _open_session(tmp_path)
    window = session.ethogram_window(start_sample_idx=2, end_sample_idx=5)
    assert window.start_sample_idx == 2
    assert window.end_sample_idx == 5
    assert window.states_available is True
    assert len(window.directions) == 2
    for track in window.directions:
        assert track.sample_idx.min() >= 2
        assert track.sample_idx.max() <= 5
        # time_sec matches sync authority
        for s, t in zip(track.sample_idx, track.time_sec, strict=True):
            assert t == pytest.approx(float(built["sync"].loc[int(s), "time_sec"]))


def test_both_directional_ethogram_tracks(tmp_path: Path) -> None:
    _, session = _open_session(tmp_path)
    window = session.ethogram_window(start_sample_idx=0, end_sample_idx=3)
    pairs = {(t.lane.focal_id, t.lane.partner_id) for t in window.directions}
    assert pairs == {(0, 1), (1, 0)}
    labels_a = [
        t.behavior_label for t in window.directions if t.lane.focal_id == 0
    ][0]
    labels_b = [
        t.behavior_label for t in window.directions if t.lane.focal_id == 1
    ][0]
    assert set(labels_a.tolist()) == {"approach"}
    assert set(labels_b.tolist()) == {"retract"}


def test_independent_contact_and_sbs_tracks(tmp_path: Path) -> None:
    _, session = _open_session(tmp_path)
    window = session.ethogram_window(start_sample_idx=0, end_sample_idx=9)
    assert window.contact is not None
    assert window.side_by_side is not None
    assert window.contact.lane.channel == "contact"
    assert window.side_by_side.lane.channel == "side_by_side"
    assert window.contact.lane.identity_grain == "symmetric_pair"
    # Contact positives exist while exclusive labels vary.
    assert int(window.contact.pred.sum()) > 0
    assert int(window.side_by_side.pred.sum()) > 0
    # Parallel tracks are not folded into exclusive lanes.
    exclusive_channels = {t.lane.channel for t in window.directions}
    assert exclusive_channels == {"exclusive"}


def test_engaged_versus_undefined_distinct(tmp_path: Path) -> None:
    _, session = _open_session(tmp_path)
    window = session.ethogram_window(start_sample_idx=4, end_sample_idx=9)
    track_a = next(t for t in window.directions if t.lane.focal_id == 0)
    assert LABEL_ENGAGED_VALUE in track_a.behavior_label
    assert LABEL_UNDEFINED_VALUE in track_a.behavior_label
    engaged_idx = np.flatnonzero(track_a.behavior_label == LABEL_ENGAGED_VALUE)[0]
    undef_idx = np.flatnonzero(track_a.behavior_label == LABEL_UNDEFINED_VALUE)[0]
    assert track_a.behavior_label[engaged_idx] != track_a.behavior_label[undef_idx]
    assert track_a.undefined_reason[undef_idx] == "unresolved_invalid_evidence"
    assert track_a.evidence_state[undef_idx] in {EVIDENCE_UNOBSERVED, "missing"}


def test_reconstruction_indicators(tmp_path: Path) -> None:
    _, session = _open_session(tmp_path)
    window = session.ethogram_window(start_sample_idx=4, end_sample_idx=6)
    track_b = next(t for t in window.directions if t.lane.focal_id == 1)
    assert "stitched" in track_b.label_reconstruction
    track_a = next(t for t in window.directions if t.lane.focal_id == 0)
    assert "engaged_valid" in track_a.hierarchy_residual_kind


def test_column_projection_avoids_full_load(tmp_path: Path) -> None:
    built, session = _open_session(tmp_path)
    window = session.ethogram.read_window(
        0, 2, include_echo_probs=False, include_validity=False, keep_raw=True
    )
    assert window.raw_frame is not None
    assert "behavior_label" in window.columns_loaded
    # Heavy optional columns not requested should be absent when not needed.
    assert "prob_approach" not in window.columns_loaded
    snaps = snapshot_upstream_fingerprints(session.source)
    assert_no_upstream_writes(
        package_dir=session.source.package_dir,
        s8_run_dir=session.source.source_s8.run_dir,
        video_path=session.source.source_video.path,
        snapshots=snaps,
    )
    assert built["n_dense"] == session.n_dense_samples


def test_bout_filtering_and_stable_selection(tmp_path: Path) -> None:
    _, session = _open_session(tmp_path)
    browser = session.bouts
    exclusive = browser.list_bouts(BoutFilter(channels=frozenset({"exclusive"})))
    contact = browser.list_bouts(BoutFilter(channels=frozenset({"contact"})))
    assert len(exclusive) == 2
    assert len(contact) == 1
    assert set(contact["identity_grain"]) == {"symmetric_pair"}
    bout_id = str(exclusive.iloc[0]["bout_id"])
    selected = browser.select(bout_id)
    assert selected is not None
    assert browser.selected_bout_id == bout_id
    overlay = browser.overlay()
    assert overlay.selected is True
    assert overlay.start_sample_idx <= overlay.end_sample_idx
    assert overlay.bout_duration_frames == (
        overlay.end_sample_idx - overlay.start_sample_idx + 1
    )


def test_previous_next_bout_navigation(tmp_path: Path) -> None:
    _, session = _open_session(tmp_path)
    filt = BoutFilter(channels=frozenset({"exclusive"}))
    session.set_bout_filter(filt)
    ids = session.bouts.list_bouts(filt)["bout_id"].astype(str).tolist()
    session.select_bout(ids[0])
    assert session.bouts.previous_bout() is None
    nxt = session.bouts.next_bout()
    assert nxt == ids[1]
    session.select_bout(ids[1])
    assert session.bouts.previous_bout() == ids[0]
    assert session.bouts.next_bout() is None


def test_inclusive_endpoint_overlays_and_overlap(tmp_path: Path) -> None:
    _, session = _open_session(tmp_path)
    contact = session.bouts.list_bouts(BoutFilter(channels=frozenset({"contact"}))).iloc[0]
    overlay = session.bouts.overlay(str(contact["bout_id"]))
    assert overlay.inclusive_endpoints == (
        int(contact["start_sample_idx"]),
        int(contact["end_sample_idx"]),
    )
    mapped = session.bouts.navigation_annotations(str(contact["bout_id"]))
    assert mapped.start.sample_idx == overlay.start_sample_idx
    assert mapped.end.sample_idx == overlay.end_sample_idx
    hits = session.bouts.overlapping(3, 3)
    assert str(contact["bout_id"]) in set(hits["bout_id"].astype(str))


def test_dense_prepared_plateau_with_bout_seek(tmp_path: Path) -> None:
    _, session = _open_session(tmp_path, plateau_repeats=2)
    bout = session.source.behavior_bouts.iloc[0]
    mapped = session.seek_bout_start(str(bout["bout_id"]), decode=True)
    assert mapped.start.sample_idx == int(bout["start_sample_idx"])
    # Step within plateau: ethogram advances, prepared may stay.
    prepared = session.playhead.prepared_frame_idx
    session.step(1, decode=True)
    assert session.playhead.sample_idx == mapped.start.sample_idx + 1
    assert session.playhead.prepared_frame_idx == prepared


def test_invalid_window_raises(tmp_path: Path) -> None:
    _, session = _open_session(tmp_path)
    with pytest.raises(InspectionError, match="outside dense range"):
        session.ethogram.read_window(-1, 2)
    with pytest.raises(InspectionError, match="start .* > end"):
        session.ethogram.read_window(5, 2)


def test_run_length_splits_on_dense_gap() -> None:
    sample_idx = np.asarray([0, 1, 2, 5, 6], dtype=np.int64)
    values = np.asarray(["a", "a", "a", "a", "a"], dtype=object)
    segs = run_length_segments(sample_idx, values)
    assert segs == [(0, 2, "a"), (5, 6, "a")]


def test_bout_browser_rejects_empty() -> None:
    import pandas as pd

    from tests.behavioral_review._fixtures import build_hand_computed_sync

    sync = SyncMap.from_sync_index(
        build_hand_computed_sync(n_prepared=2, plateau_repeats=1)
    )
    with pytest.raises(InspectionError, match="empty"):
        BoutBrowser(pd.DataFrame(), sync)
