"""C2-B skeleton topology: real edges when available; never invent nodes-only fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.behavioral_review._fixtures import build_hand_computed_sync
from tests.overlay._fixtures import solid_frame

from behavioral_review.sync import SyncMap
from overlay import (
    DrawConfig,
    LayerId,
    OverlayCompositor,
    collect_evidence,
    make_s4_final_pose_adapter,
)
from overlay.contracts import (
    S4_PROVENANCE_ORIGINAL,
    S4_PROVENANCE_TEMPORAL_RESAMPLE,
)
from overlay.layers.pose import PoseLayer
from overlay.source_s4 import OverlayS4QcSource, _resolve_skeleton_edges
from tracking_correction.skeleton import resolve_review_skeleton_edges

# Lab topology that must come from resolved labels — never a hardcoded default.
_REAL_TOPOLOGY = (
    ("headstage", "nose"),
    ("nose", "neck"),
    ("neck", "spine_base"),
    ("spine_base", "tail_base"),
    ("headstage", "neck"),
)
_FORBIDDEN_GUESSED_EDGE = ("neck", "tail_base")


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


def _fake_topology_labels() -> _FakeLabels:
    return _FakeLabels(
        skeletons=[
            _FakeSkeleton(
                edges=[
                    _FakeEdge(_FakeNode(a), _FakeNode(b)) for a, b in _REAL_TOPOLOGY
                ],
                edge_names=list(_REAL_TOPOLOGY),
            )
        ]
    )


def test_resolve_skeleton_edges_never_guesses_when_unreadable(tmp_path: Path) -> None:
    s3 = tmp_path / "tracking_correction" / "tracking_correction__x"
    s3.mkdir(parents=True)
    s2 = tmp_path / "pose_inference" / "bottomup__x"
    s2.mkdir(parents=True)
    (s2 / "pose.slp").write_bytes(b"not-a-real-slp")
    (s3 / "run_meta.json").write_text(
        json.dumps({"input": {"s2_run_dir": str(s2)}}),
        encoding="utf-8",
    )
    edges, source = _resolve_skeleton_edges(s3_run_dir=s3)
    assert edges == ()
    assert source is not None
    assert source.startswith("unresolved:")
    assert _FORBIDDEN_GUESSED_EDGE not in edges


def test_resolve_skeleton_edges_uses_injected_sleap_topology(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    s3 = tmp_path / "tracking_correction" / "tracking_correction__x"
    s3.mkdir(parents=True)
    s2 = tmp_path / "pose_inference" / "bottomup__x"
    s2.mkdir(parents=True)
    pose_slp = s2 / "pose.slp"
    pose_slp.write_bytes(b"placeholder")
    (s3 / "run_meta.json").write_text(
        json.dumps({"input": {"s2_run_dir": str(s2)}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "tracking_correction.skeleton._load_sleap_labels",
        lambda _path: _fake_topology_labels(),
    )
    edges, source = _resolve_skeleton_edges(s3_run_dir=s3)
    assert edges == _REAL_TOPOLOGY
    assert _FORBIDDEN_GUESSED_EDGE not in edges
    assert source == f"s2_pose_slp:{pose_slp.resolve()}"


def test_s4_pose_layer_draws_real_topology_edges(tmp_path: Path) -> None:
    """Focused fixture: known topology lights the edge midpoint; nodes-only does not."""

    rows = [
        {
            "sample_idx": 0,
            "track": 0,
            "node": "nose",
            "x": 10.0,
            "y": 20.0,
            "provenance": S4_PROVENANCE_ORIGINAL,
            "provenance_name": "original",
        },
        {
            "sample_idx": 0,
            "track": 0,
            "node": "neck",
            "x": 50.0,
            "y": 20.0,
            "provenance": S4_PROVENANCE_TEMPORAL_RESAMPLE,
            "provenance_name": "temporal_resample",
        },
        {
            "sample_idx": 0,
            "track": 0,
            "node": "spine_base",
            "x": 90.0,
            "y": 20.0,
            "provenance": S4_PROVENANCE_ORIGINAL,
            "provenance_name": "original",
        },
    ]
    path = tmp_path / "final_pose.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    adapter = make_s4_final_pose_adapter(path)
    sync = SyncMap.from_sync_index(
        build_hand_computed_sync(n_prepared=1, plateau_repeats=1)
    )
    evidence = collect_evidence(
        sync, 0, {LayerId.S4_POSE_FINAL: adapter}, selected_layers=[LayerId.S4_POSE_FINAL]
    )
    frame = solid_frame((0, 0, 0), h=48, w=120)

    with_edges = OverlayCompositor().render_frame(
        frame,
        prepared_frame_idx=0,
        evidence=evidence,
        selected_layers=[LayerId.S4_POSE_FINAL],
        style=DrawConfig(
            draw_skeleton=True,
            legend_text=False,
            extra={"skeleton_edges": [("nose", "neck"), ("neck", "spine_base")]},
        ),
    )
    nodes_only = OverlayCompositor().render_frame(
        frame,
        prepared_frame_idx=0,
        evidence=evidence,
        selected_layers=[LayerId.S4_POSE_FINAL],
        style=DrawConfig(
            draw_skeleton=True,
            legend_text=False,
            extra={"skeleton_edges": []},
        ),
    )
    # Midpoint of nose→neck at (30, 20) should be lit only when topology edges exist.
    mid = with_edges.frame_bgr[20, 30]
    mid_nodes = nodes_only.frame_bgr[20, 30]
    assert int(mid.sum()) > 0
    assert int(mid_nodes.sum()) == 0
    # Identity mapping: track 0 color remains on the observed nose center.
    assert not np.array_equal(
        with_edges.frame_bgr[20, 10], np.array([0, 0, 0], dtype=np.uint8)
    )


def test_overlay_s4_source_discloses_nodes_only(tmp_path: Path) -> None:
    sync = SyncMap.from_sync_index(
        build_hand_computed_sync(n_prepared=2, plateau_repeats=1)
    )
    pose = tmp_path / "final_pose.parquet"
    pd.DataFrame(
        [
            {
                "sample_idx": 0,
                "track": 0,
                "node": "nose",
                "x": 1.0,
                "y": 2.0,
                "provenance": S4_PROVENANCE_ORIGINAL,
                "provenance_name": "original",
            }
        ]
    ).to_parquet(pose, index=False)
    from results_packaging.media_csv_export.video import VideoProbe

    source = OverlayS4QcSource(
        s4_run_id="synth",
        s4_run_dir=tmp_path,
        s4_status="completed",
        final_pose_path=pose,
        final_pose_sha256="a" * 64,
        session_root=tmp_path,
        prepared_sync_path=tmp_path / "prepared_sync.npz",
        prepared_sync_sha256="b" * 64,
        prepared_video_path=tmp_path / "prepared_video.mp4",
        prepared_video_sha256="c" * 64,
        video_probe=VideoProbe(
            path=tmp_path / "prepared_video.mp4",
            fps=10.0,
            n_frames=2,
            width=64,
            height=48,
        ),
        sync=sync,
        sync_build_audit={},
        source_s3_run_id="s3",
        source_s3_run_dir=tmp_path / "s3",
        source_s3_tracked_pose_sha256="d" * 64,
        skeleton_edges=(),
        skeleton_source="unresolved:/fake/pose.slp",
        settings_used_path=None,
        run_meta={},
        source_digests={},
    )
    disclosure = source.skeleton_disclosure()
    assert source.skeleton_mode == "nodes_only"
    assert disclosure["skeleton_nodes_only"] is True
    assert "Nodes-only" in disclosure["skeleton_disclosure"]
    assert disclosure["skeleton_edge_count"] == 0


def test_resolve_review_skeleton_edges_empty_without_labels() -> None:
    assert resolve_review_skeleton_edges(None) == ()
    assert _FORBIDDEN_GUESSED_EDGE not in resolve_review_skeleton_edges(None)


def test_pose_layer_skips_edge_when_endpoint_undrawable() -> None:
    from overlay.context import OverlayContext, OverlayObservation
    from overlay.contracts import FrameDomain, PlateauPolicy
    from overlay.timing import resolve_prepared_frame_timing

    sync = SyncMap.from_sync_index(
        build_hand_computed_sync(n_prepared=1, plateau_repeats=1)
    )
    timing = resolve_prepared_frame_timing(sync, 0)
    obs = OverlayObservation(
        layer_id=LayerId.S4_POSE_FINAL,
        available=True,
        payload={
            "legend": "final pose",
            "points": [
                {
                    "track": 0,
                    "node": "nose",
                    "x": 10.0,
                    "y": 10.0,
                    "drawable": True,
                    "provenance": S4_PROVENANCE_ORIGINAL,
                    "visual_class": "observed",
                },
                {
                    "track": 0,
                    "node": "neck",
                    "x": None,
                    "y": None,
                    "drawable": False,
                    "provenance": 0,
                    "visual_class": "unavailable",
                },
                {
                    "track": 0,
                    "node": "spine_base",
                    "x": 40.0,
                    "y": 10.0,
                    "drawable": True,
                    "provenance": S4_PROVENANCE_ORIGINAL,
                    "visual_class": "observed",
                },
            ],
        },
    )
    ctx = OverlayContext(
        prepared_frame_idx=0,
        timing=timing,
        frame_domain=FrameDomain.PREPARED,
        plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
        style=DrawConfig(
            draw_skeleton=True,
            legend_text=False,
            extra={
                "skeleton_edges": [
                    ("nose", "neck"),
                    ("neck", "spine_base"),
                    ("nose", "spine_base"),
                ]
            },
        ),
        selected_layers=(LayerId.S4_POSE_FINAL,),
    )
    canvas = PoseLayer(LayerId.S4_POSE_FINAL).render(
        solid_frame((0, 0, 0), h=32, w=64), ctx, obs
    )
    # Valid nose→spine midpoint lit; nose→neck midpoint not forced by missing neck.
    assert int(canvas[10, 25].sum()) > int(canvas[10, 15].sum())
