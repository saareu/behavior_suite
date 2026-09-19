"""S3 review skeleton topology resolution and overlay edge sourcing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tests.tracking_correction.test_review import _completed_s2, _pose_rows, _write_pose

from tracking_correction.contracts import TrackingCorrectionRequest
from tracking_correction.overlay_draw import draw_pose_overlay
from tracking_correction.review import PoseNodePoint, load_review_session
from tracking_correction.runner import run_tracking_correction
from tracking_correction.skeleton import (
    edges_from_skeleton,
    resolve_review_skeleton_edges,
    skeleton_edges_from_labels,
)

# Hardcoded incorrect / lab-specific edges that must NOT be assumed by review.
_FORBIDDEN_DEFAULT_EDGE = ("neck", "tail_base")
_S2_TOPOLOGY = (
    ("headstage", "nose"),
    ("nose", "neck"),
    ("neck", "spine_base"),
    ("spine_base", "tail_base"),
    ("headstage", "neck"),
)


@dataclass
class _FakeNode:
    name: str


@dataclass
class _FakeEdge:
    source: _FakeNode
    destination: _FakeNode


@dataclass
class _FakeSkeleton:
    edges: list[_FakeEdge]
    edge_names: list[tuple[str, str]] | None = None


@dataclass
class _FakeLabels:
    skeletons: list[_FakeSkeleton]


def test_overlay_edges_come_from_resolved_s2_skeleton_not_hardcoded_list(
    tmp_path: Path, monkeypatch
) -> None:
    s2_run, _s1 = _completed_s2(tmp_path)
    pose_slp = s2_run / "pose.slp"
    pose_slp.write_bytes(b"not-a-real-slp")

    fake_labels = _FakeLabels(
        skeletons=[
            _FakeSkeleton(
                edges=[
                    _FakeEdge(_FakeNode(a), _FakeNode(b)) for a, b in _S2_TOPOLOGY
                ],
                edge_names=list(_S2_TOPOLOGY),
            )
        ]
    )
    monkeypatch.setattr(
        "tracking_correction.skeleton._load_sleap_labels",
        lambda _path: fake_labels,
    )

    result = run_tracking_correction(
        TrackingCorrectionRequest(
            s2_run_dir=s2_run,
            dry_run=False,
            timestamp="20260301T130000",
        )
    )
    assert result.success is True
    session = load_review_session(result.run_dir)

    assert session.skeleton_edges == _S2_TOPOLOGY
    assert _FORBIDDEN_DEFAULT_EDGE not in session.skeleton_edges
    assert "s2_pose_slp:" in session.skeleton_source

    frame = np.zeros((120, 120, 3), dtype=np.uint8)
    points = (
        PoseNodePoint(track=0, node="nose", x=10, y=10),
        PoseNodePoint(track=0, node="neck", x=20, y=20),
        PoseNodePoint(track=0, node="spine_base", x=30, y=30),
        PoseNodePoint(track=0, node="tail_base", x=40, y=40),
        PoseNodePoint(track=0, node="headstage", x=15, y=5),
    )
    drawn = draw_pose_overlay(frame, points, skeleton_edges=session.skeleton_edges)
    assert drawn.shape == frame.shape
    # Nodes-only fallback uses empty edges and must not invent forbidden topology.
    empty = resolve_review_skeleton_edges(None)
    assert empty == ()
    assert _FORBIDDEN_DEFAULT_EDGE not in empty


def test_skeleton_edges_from_labels_and_fallback_nodes_only() -> None:
    labels = _FakeLabels(
        skeletons=[
            _FakeSkeleton(
                edges=[
                    _FakeEdge(_FakeNode("nose"), _FakeNode("neck")),
                    _FakeEdge(_FakeNode("neck"), _FakeNode("spine_base")),
                ]
            )
        ]
    )
    assert skeleton_edges_from_labels(labels) == (
        ("nose", "neck"),
        ("neck", "spine_base"),
    )
    assert edges_from_skeleton(labels.skeletons[0]) == (
        ("nose", "neck"),
        ("neck", "spine_base"),
    )
    assert resolve_review_skeleton_edges(Path("missing.slp")) == ()


def test_session_without_pose_slp_uses_nodes_only_overlay(tmp_path: Path) -> None:
    s2_run, _s1 = _completed_s2(tmp_path)
    _write_pose(s2_run / "pose.parquet", _pose_rows())
    result = run_tracking_correction(
        TrackingCorrectionRequest(
            s2_run_dir=s2_run,
            dry_run=False,
            timestamp="20260301T140000",
        )
    )
    session = load_review_session(result.run_dir)
    assert session.skeleton_edges == ()
    assert session.skeleton_source == "unavailable"
