"""C3-2 S3 correction journal parsing and attribution rules."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from overlay.corrections_s3 import (
    load_machine_correction_index,
    load_manual_correction_index,
)
from overlay.layers.pose import (
    _correction_anchor,
    _draw_s3_correction_events,
    _events_for_prepared_frame,
)


def test_machine_vs_manual_parsing_and_attribution(tmp_path: Path) -> None:
    machine = tmp_path / "machine_corrections.json"
    machine.write_text(
        json.dumps(
            [
                {"frame": 2, "type": "node_overlap", "track": 0},
                {"frame": 3, "type": "track_reassignment"},
                {"frame": 4, "type": "orientation", "track": 1, "node": "nose"},
                {"frame": 5, "type": "bad"},  # still usable: frame+type
                {"not_a_frame": 9, "type": "skip"},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manual = tmp_path / "manual_corrections.json"
    manual.write_text(
        json.dumps(
            [
                {
                    "action": "swap_identities",
                    "start_frame": 10,
                    "end_frame": 12,
                    "tracks": [0, 1],
                    "source": "user",
                },
                {
                    "action": "blank_node",
                    "start_frame": 11,
                    "end_frame": 11,
                    "track": 0,
                    "node": "neck",
                    "tracks": [0],
                    "source": "user",
                },
                {
                    "action": "swap_node",
                    "start_frame": 13,
                    "end_frame": 14,
                    "node": "nose",
                    "tracks": [0, 1],
                    "source": "user",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    machine_idx = load_machine_correction_index(machine)
    manual_idx = load_manual_correction_index(manual)
    assert machine_idx.status == "available"
    assert manual_idx.status == "available"

    by_track = machine_idx.events_for_frame(2)
    assert len(by_track) == 1
    assert by_track[0].attribution == "track"
    assert by_track[0].action_type == "node_overlap"

    unlocalized = machine_idx.events_for_frame(3)
    assert len(unlocalized) == 1
    assert unlocalized[0].attribution == "unlocalized"

    track_node = machine_idx.events_for_frame(4)
    assert track_node[0].attribution == "track_node"
    assert track_node[0].node == "nose"

    # Free-text action must not invent a structured node.
    assert all(
        event.node is None or event.attribution == "track_node"
        for event in machine_idx.events
    )

    # Inclusive interval coverage.
    assert len(manual_idx.events_for_frame(10)) == 1
    assert len(manual_idx.events_for_frame(11)) == 2
    assert len(manual_idx.events_for_frame(12)) == 1
    assert len(manual_idx.events_for_frame(13)) == 1
    assert len(manual_idx.events_for_frame(9)) == 0

    swap_ids = manual_idx.events_for_frame(10)[0]
    assert swap_ids.attribution == "unlocalized"
    blank = [e for e in manual_idx.events_for_frame(11) if e.action_type == "blank_node"][0]
    assert blank.attribution == "track_node"
    swap_node = manual_idx.events_for_frame(13)[0]
    assert swap_node.attribution == "unlocalized"  # multi-track list, no dedicated track


def test_missing_and_malformed_journals(tmp_path: Path) -> None:
    missing = load_machine_correction_index(tmp_path / "nope.json")
    assert missing.status == "absent"
    assert missing.events == ()

    empty = tmp_path / "empty.json"
    empty.write_text("[]\n", encoding="utf-8")
    empty_idx = load_manual_correction_index(empty)
    assert empty_idx.status == "empty"

    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "a list"}\n', encoding="utf-8")
    malformed = load_machine_correction_index(bad)
    assert malformed.status == "malformed"
    assert "unavailable" in malformed.disclosure.lower()


def test_no_was_track_swapped_fabrication_and_anchor_rules() -> None:
    # Journal loader never reads parquet flags; anchors require accepted pose.
    by_track = {
        0: {
            "nose": {"x": 10.0, "y": 20.0},
            "neck": {"x": 12.0, "y": 22.0},
        }
    }
    assert (
        _correction_anchor(
            by_track=by_track,
            attribution="unlocalized",
            track=0,
            node="nose",
        )
        is None
    )
    assert _correction_anchor(
        by_track=by_track,
        attribution="track",
        track=0,
        node=None,
    ) == (10, 2)
    assert _correction_anchor(
        by_track=by_track,
        attribution="track_node",
        track=0,
        node="neck",
    ) == (12, 22)
    # Missing track → unlocalized draw path.
    assert (
        _correction_anchor(
            by_track=by_track,
            attribution="track",
            track=1,
            node=None,
        )
        is None
    )
    # Missing attributed node (e.g. blank_node NaNs) must NOT fall back to nose.
    assert (
        _correction_anchor(
            by_track=by_track,
            attribution="track_node",
            track=0,
            node="spine_base",
        )
        is None
    )


def test_event_window_dict_and_draw_smoke() -> None:
    from overlay.corrections_s3 import CorrectionEvent, CorrectionJournalIndex

    index = CorrectionJournalIndex(
        origin="machine",
        status="available",
        path=None,
        events=(
            CorrectionEvent(
                origin="machine",
                action_type="node_overlap",
                start_frame=5,
                end_frame=5,
                track=0,
                node=None,
                attribution="track",
            ),
            CorrectionEvent(
                origin="machine",
                action_type="identity_swap",
                start_frame=7,
                end_frame=7,
                track=None,
                node=None,
                attribution="unlocalized",
            ),
        ),
        disclosure="fixture",
    )
    windowed = index.by_frame_dict((5, 6))
    assert 5 in windowed
    assert 7 not in windowed
    frame = np.zeros((64, 96, 3), dtype=np.uint8)
    events = _events_for_prepared_frame(windowed, 5)
    _draw_s3_correction_events(
        frame,
        machine_events=events,
        manual_events=[],
        by_track={0: {"nose": {"x": 30.0, "y": 20.0}}},
    )
    assert frame.sum() > 0


def _count_near_bgr(
    canvas: np.ndarray, target: tuple[int, int, int], *, tol: int = 12
) -> int:
    diff = np.abs(canvas.astype(np.int16) - np.array(target, dtype=np.int16))
    return int(np.sum(np.all(diff <= tol, axis=2)))


def test_manual_inclusive_interval_pixel_boundaries_and_styling(tmp_path: Path) -> None:
    """Controlled fixture: inclusive multi-frame manual interval + overlap + styling.

    Interval under test: blank_node track_node on [10, 14] inclusive.
    Overlapping second manual: swap_identities unlocalized on [12, 13].
    Third manual: unlocalized review_note across [10, 14].
    Machine single-frame event at frame 12 verifies distinct AUTO vs MAN styling and
    no neighbor bleed for machine single-frame events.
    """

    from overlay.layers.pose import _MACHINE_EVENT_BGR, _MANUAL_EVENT_BGR

    machine = tmp_path / "machine_corrections.json"
    machine.write_text(
        json.dumps(
            [
                # Exact single-frame machine event — must not extend to 11 or 13.
                {"frame": 12, "type": "node_overlap", "track": 0},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    manual = tmp_path / "manual_corrections.json"
    manual.write_text(
        json.dumps(
            [
                {
                    "action": "blank_node",
                    "start_frame": 10,
                    "end_frame": 14,
                    "track": 1,
                    "node": "neck",
                    "tracks": [1],
                    "source": "user",
                },
                {
                    "action": "swap_identities",
                    "start_frame": 12,
                    "end_frame": 13,
                    "tracks": [0, 1],
                    "source": "user",
                },
                {
                    # Unlocalized across the whole primary interval (no track/node).
                    "action": "review_note",
                    "start_frame": 10,
                    "end_frame": 14,
                    "source": "user",
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    machine_idx = load_machine_correction_index(machine)
    manual_idx = load_manual_correction_index(manual)
    assert machine_idx.status == "available"
    assert manual_idx.status == "available"

    # Inclusive coverage + no off-by-one.
    assert [e.action_type for e in manual_idx.events_for_frame(9)] == []
    assert {e.action_type for e in manual_idx.events_for_frame(10)} == {
        "blank_node",
        "review_note",
    }
    assert {e.action_type for e in manual_idx.events_for_frame(12)} == {
        "blank_node",
        "swap_identities",
        "review_note",
    }
    assert {e.action_type for e in manual_idx.events_for_frame(14)} == {
        "blank_node",
        "review_note",
    }
    assert [e.action_type for e in manual_idx.events_for_frame(15)] == []
    assert [e.action_type for e in machine_idx.events_for_frame(11)] == []
    assert [e.action_type for e in machine_idx.events_for_frame(12)] == ["node_overlap"]
    assert [e.action_type for e in machine_idx.events_for_frame(13)] == []

    blank = next(e for e in manual_idx.events if e.action_type == "blank_node")
    assert blank.attribution == "track_node"
    assert blank.node == "neck"
    swap = next(e for e in manual_idx.events if e.action_type == "swap_identities")
    assert swap.attribution == "unlocalized"
    note = next(e for e in manual_idx.events if e.action_type == "review_note")
    assert note.attribution == "unlocalized"

    # Keep localized anchors below the unlocalized pill band (y≈36) so machine and
    # manual indicators remain spatially distinguishable on the combined canvas.
    by_track = {
        0: {"nose": {"x": 30.0, "y": 95.0}, "neck": {"x": 32.0, "y": 100.0}},
        1: {"nose": {"x": 120.0, "y": 95.0}, "neck": {"x": 122.0, "y": 100.0}},
    }
    neck_xy = (122, 100)
    machine_track_anchor = _correction_anchor(
        by_track=by_track, attribution="track", track=0, node=None
    )
    assert machine_track_anchor == (30, 77)

    machine_by_frame = machine_idx.by_frame_dict((9, 15))
    manual_by_frame = manual_idx.by_frame_dict((9, 15))

    probes = {
        "before": 9,
        "first": 10,
        "intermediate": 12,
        "last": 14,
        "after": 15,
    }
    rendered: dict[str, np.ndarray] = {}
    for name, frame_idx in probes.items():
        canvas = np.zeros((140, 180, 3), dtype=np.uint8)
        _draw_s3_correction_events(
            canvas,
            machine_events=_events_for_prepared_frame(machine_by_frame, frame_idx),
            manual_events=_events_for_prepared_frame(manual_by_frame, frame_idx),
            by_track=by_track,
        )
        rendered[name] = canvas

    # Boundary pixels: no indicators before/after the inclusive interval.
    assert rendered["before"].sum() == 0
    assert rendered["after"].sum() == 0

    # First / last include localized blank_node diamond at neck + unlocalized pill.
    for name in ("first", "last"):
        canvas = rendered[name]
        assert _count_near_bgr(canvas, _MANUAL_EVENT_BGR) > 0
        assert _count_near_bgr(canvas, _MACHINE_EVENT_BGR) == 0
        # Localized diamond sits on structured neck geometry (accepted pose only).
        assert tuple(int(v) for v in canvas[neck_xy[1], neck_xy[0]]) != (0, 0, 0)

    # Intermediate overlap: both manual actions + machine single-frame present.
    mid = rendered["intermediate"]
    mid_events = _events_for_prepared_frame(manual_by_frame, 12)
    assert {e["action_type"] for e in mid_events} == {
        "blank_node",
        "swap_identities",
        "review_note",
    }
    manual_px = _count_near_bgr(mid, _MANUAL_EVENT_BGR)
    machine_px = _count_near_bgr(mid, _MACHINE_EVENT_BGR)
    assert manual_px > 0
    assert machine_px > 0
    # Distinct styling: machine orange vs manual teal both present and separable.
    assert manual_px != machine_px
    # Localized blank_node still at neck; machine track diamond elsewhere.
    assert tuple(int(v) for v in mid[neck_xy[1], neck_xy[0]]) != (0, 0, 0)
    mx, my = machine_track_anchor
    assert tuple(int(v) for v in mid[my, mx]) == _MACHINE_EVENT_BGR
    assert abs(mx - neck_xy[0]) > 20 or abs(my - neck_xy[1]) > 20
    # Unlocalized manual pills occupy the upper HUD band; localized diamond stays low.
    top_band = mid[20:55, :, :]
    assert _count_near_bgr(top_band, _MANUAL_EVENT_BGR) > 0
    assert _count_near_bgr(top_band, _MACHINE_EVENT_BGR) == 0
    neck_roi = mid[neck_xy[1] - 8 : neck_xy[1] + 9, neck_xy[0] - 8 : neck_xy[0] + 9]
    assert _count_near_bgr(neck_roi, _MANUAL_EVENT_BGR) > 0
    assert _count_near_bgr(neck_roi, _MACHINE_EVENT_BGR) == 0

    # Machine single-frame must not bleed onto first/last of the manual interval.
    assert _count_near_bgr(rendered["first"], _MACHINE_EVENT_BGR) == 0
    assert _count_near_bgr(rendered["last"], _MACHINE_EVENT_BGR) == 0
