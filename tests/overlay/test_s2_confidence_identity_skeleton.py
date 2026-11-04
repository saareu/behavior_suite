"""C3-1 S2 confidence styling, provisional identity, skeleton nodes-only fallback."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from tests.overlay._fixtures import solid_frame

from behavioral_review.sync import SyncMap
from overlay import (
    DrawConfig,
    LayerId,
    OverlayCompositor,
    collect_evidence,
    make_s2_inferred_pose_adapter,
    s2_score_visual_intensity,
)
from overlay.contracts import (
    S2_SCORE_VISUAL_RANGE_MAX,
    S2_SCORE_VISUAL_RANGE_MIN,
)
from overlay.layers.pose import PoseLayer, _s2_drawable_points
from overlay.source_s2 import _resolve_skeleton_edges
from overlay.timing import PreparedFrameTiming
from results_packaging.contracts import SYNC_SCHEMA_VERSION
from tracking_correction.skeleton import resolve_review_skeleton_edges


def _tiny_sync(n: int = 3, fps: float = 10.0) -> SyncMap:
    idx = np.arange(n, dtype=np.int64)
    times = idx.astype(np.float64) / fps
    return SyncMap.from_sync_index(
        pd.DataFrame(
            {
                "schema_version": SYNC_SCHEMA_VERSION,
                "sample_idx": idx,
                "dense_frame_idx": idx,
                "time_sec": times,
                "prepared_frame_idx": idx,
                "prepared_time_sec": times,
                "is_exact_source_sample": np.ones(n, dtype=bool),
                "timing_residual_sec": np.zeros(n, dtype=np.float64),
            }
        ),
        fps=fps,
    )


def test_s2_score_visual_intensity_distinguishes_missing_and_zero() -> None:
    assert s2_score_visual_intensity(None) is None
    assert s2_score_visual_intensity(float("nan")) is None
    assert s2_score_visual_intensity(0.0) == 0.0
    assert s2_score_visual_intensity(1.0) == 1.0
    assert s2_score_visual_intensity(0.5) == pytest.approx(0.5)
    # Outside assumed range: clamp for drawing only.
    assert s2_score_visual_intensity(-0.2) == 0.0
    assert s2_score_visual_intensity(1.27) == 1.0
    assert S2_SCORE_VISUAL_RANGE_MIN == 0.0
    assert S2_SCORE_VISUAL_RANGE_MAX == 1.0


def test_s2_adapter_preserves_scores_and_provisional_track_labels(
    tmp_path: Path,
) -> None:
    path = tmp_path / "pose.parquet"
    pd.DataFrame(
        [
            {
                "frame_idx": 0,
                "prepared_frame_idx": 0,
                "instance_index": 0,
                "track": "track_0",
                "node": "nose",
                "x": 10.0,
                "y": 11.0,
                "score": 0.0,
            },
            {
                "frame_idx": 0,
                "prepared_frame_idx": 0,
                "instance_index": 0,
                "track": "track_0",
                "node": "neck",
                "x": 12.0,
                "y": 13.0,
                "score": None,
            },
            {
                "frame_idx": 0,
                "prepared_frame_idx": 0,
                "instance_index": 1,
                "track": "track_1",
                "node": "nose",
                "x": 30.0,
                "y": 31.0,
                "score": 1.15,
            },
        ]
    ).to_parquet(path, index=False)

    adapter = make_s2_inferred_pose_adapter(path, run_id="s2x", run_dir=tmp_path)
    from overlay.contracts import PlateauPolicy

    timing = PreparedFrameTiming(
        prepared_frame_idx=0,
        plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
        available=True,
        dense_sample_idx=0,
        plateau_sample_indices=(0,),
        mapping=None,
        reason=None,
    )
    obs = adapter.observe(timing, prepared_frame_idx=0)
    assert obs.available
    assert obs.payload["identity_kind"] == "provisional"
    by_key = {(int(p["track"]), str(p["node"])): p for p in obs.payload["points"]}
    assert by_key[(0, "nose")]["score"] == 0.0
    assert by_key[(0, "nose")]["score_missing"] is False
    assert by_key[(0, "neck")]["score"] is None
    assert by_key[(0, "neck")]["score_missing"] is True
    # Original >1 preserved (not clamped in payload).
    assert by_key[(1, "nose")]["score"] == pytest.approx(1.15)
    tracks = {int(p["track"]) for p in obs.payload["points"]}
    assert tracks == {0, 1}


def test_s2_no_confidence_based_filtering_in_drawable() -> None:
    points = _s2_drawable_points(
        [
            {"track": 0, "node": "nose", "x": 1.0, "y": 2.0, "score": 0.0},
            {"track": 0, "node": "neck", "x": 3.0, "y": 4.0, "score": None},
            {"track": 1, "node": "nose", "x": 5.0, "y": 6.0, "score": 0.01},
        ]
    )
    assert len(points) == 3
    assert {p["node"] for p in points} == {"nose", "neck"}


def test_s2_render_provisional_labels_and_confidence(tmp_path: Path) -> None:
    path = tmp_path / "pose.parquet"
    pd.DataFrame(
        [
            {
                "frame_idx": 0,
                "prepared_frame_idx": 0,
                "track": 0,
                "node": "nose",
                "x": 20.0,
                "y": 20.0,
                "score": 0.9,
            },
            {
                "frame_idx": 0,
                "prepared_frame_idx": 0,
                "track": 1,
                "node": "nose",
                "x": 50.0,
                "y": 22.0,
                "score": None,
            },
        ]
    ).to_parquet(path, index=False)
    sync = _tiny_sync()
    adapter = make_s2_inferred_pose_adapter(path)
    evidence = collect_evidence(
        sync,
        0,
        {LayerId.S2_POSE_INFERRED: adapter},
        selected_layers=[LayerId.S2_POSE_INFERRED],
    )
    frame = solid_frame(h=64, w=80)
    style = DrawConfig(
        draw_skeleton=True,
        legend_text=True,
        extra={
            "skeleton_edges": [["nose", "neck"]],
            "show_confidence": True,
        },
    )
    composed = OverlayCompositor().render_frame(
        frame,
        prepared_frame_idx=0,
        evidence=evidence,
        selected_layers=[LayerId.S2_POSE_INFERRED],
        style=style,
    )
    assert composed.frame_bgr.shape == frame.shape
    # Source frame immutability.
    assert np.array_equal(frame, solid_frame(h=64, w=80))


def test_s2_confidence_fixture_pixels_and_legend(tmp_path: Path) -> None:
    """Pixel-level acceptance for s2_score_fill_intensity_v1.

    Fixture scores: 0, low+, mid, 1, missing, out-of-range (>1).
    """

    from overlay.contracts import (
        S2_SCORE_VISUAL_MAPPING_ID,
        S2_SCORE_VISUAL_MAPPING_SUMMARY,
    )
    from tracking_correction.review import TRACK_COLORS_BGR

    # Spaced nodes on one provisional track so identity color is shared.
    fixture = (
        ("score_0", 40.0, 0.0),
        ("score_low", 100.0, 0.05),
        ("score_mid", 160.0, 0.5),
        ("score_1", 220.0, 1.0),
        ("score_missing", 280.0, None),
        ("score_oor", 340.0, 1.5),
    )
    rows = [
        {
            "frame_idx": 0,
            "prepared_frame_idx": 0,
            "track": 0,
            "node": name,
            "x": x,
            "y": 60.0,
            "score": score,
        }
        for name, x, score in fixture
    ]
    path = tmp_path / "pose.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)

    adapter = make_s2_inferred_pose_adapter(path)
    sync = _tiny_sync()
    evidence = collect_evidence(
        sync,
        0,
        {LayerId.S2_POSE_INFERRED: adapter},
        selected_layers=[LayerId.S2_POSE_INFERRED],
    )
    obs = evidence.get(LayerId.S2_POSE_INFERRED)
    assert obs is not None and obs.available
    by_node = {str(p["node"]): p for p in obs.payload["points"]}
    # Originals preserved (display clamp must not mutate payload scores).
    assert by_node["score_0"]["score"] == 0.0
    assert by_node["score_low"]["score"] == pytest.approx(0.05)
    assert by_node["score_mid"]["score"] == pytest.approx(0.5)
    assert by_node["score_1"]["score"] == 1.0
    assert by_node["score_missing"]["score"] is None
    assert by_node["score_oor"]["score"] == pytest.approx(1.5)

    bg = (30, 40, 50)
    frame = solid_frame(bg, h=120, w=400)
    # Pixel samples without legend so legend glyphs cannot overlap marker centers.
    style_pixels = DrawConfig(
        draw_skeleton=False,
        legend_text=False,
        node_radius=5,
        extra={"show_confidence": True},
    )
    compositor = OverlayCompositor()
    composed_a = compositor.render_frame(
        frame,
        prepared_frame_idx=0,
        evidence=evidence,
        selected_layers=[LayerId.S2_POSE_INFERRED],
        style=style_pixels,
    )
    composed_b = compositor.render_frame(
        frame.copy(),
        prepared_frame_idx=0,
        evidence=evidence,
        selected_layers=[LayerId.S2_POSE_INFERRED],
        style=style_pixels,
    )
    canvas = composed_a.frame_bgr
    assert np.array_equal(canvas, composed_b.frame_bgr)

    def _center(x: float) -> np.ndarray:
        return canvas[60, int(x)].astype(np.int16)

    def _patch_delta(x: float) -> int:
        x_i = int(x)
        patch = canvas[55:66, x_i - 5 : x_i + 6].astype(np.int16)
        return int(np.abs(patch - np.array(bg, dtype=np.int16)).sum())

    centers = {name: _center(x) for name, x, _ in fixture}
    deltas = {name: _patch_delta(x) for name, x, _ in fixture}
    fill_sum = {name: int(centers[name].sum()) for name in centers}

    # Every marker remains drawable (outline/halo/fill vs background).
    for name, delta in deltas.items():
        assert delta > 5_000, f"{name} marker disappeared (delta={delta})"

    # Numerical zero: filled minimum-intensity disc (not background).
    assert not np.array_equal(centers["score_0"], np.array(bg, dtype=np.int16))
    # Missing: hollow ring — center stays background; patch still differs via ring.
    assert np.array_equal(centers["score_missing"], np.array(bg, dtype=np.int16))
    assert not np.array_equal(
        canvas[55:66, 35:46], canvas[55:66, 275:286]
    ), "score 0 and missing must be visually distinct"

    # Low confidence still filled (distinct from missing hollow center).
    assert not np.array_equal(centers["score_low"], np.array(bg, dtype=np.int16))
    # Fill intensity ordering for continuous scores (missing excluded).
    assert fill_sum["score_0"] < fill_sum["score_low"] < fill_sum["score_mid"] < fill_sum["score_1"]
    color = np.array(TRACK_COLORS_BGR[0], dtype=np.int16)
    assert np.array_equal(centers["score_1"], color)
    # Out-of-range (>1) is display-clamped to full intensity.
    assert np.array_equal(centers["score_oor"], centers["score_1"])
    assert s2_score_visual_intensity(1.5) == 1.0
    assert by_node["score_oor"]["score"] == pytest.approx(1.5)

    # Legend / mapping disclosure on a separate legend-enabled render.
    with_legend = compositor.render_frame(
        frame.copy(),
        prepared_frame_idx=0,
        evidence=evidence,
        selected_layers=[LayerId.S2_POSE_INFERRED],
        style=DrawConfig(
            draw_skeleton=False,
            legend_text=True,
            node_radius=5,
            extra={"show_confidence": True},
        ),
    ).frame_bgr
    legend_strip = with_legend[16:110, 0:260]
    assert int(np.abs(legend_strip.astype(np.int16) - np.array(bg, dtype=np.int16)).sum()) > 0
    assert S2_SCORE_VISUAL_MAPPING_ID == "s2_score_fill_intensity_v1"
    assert "hollow ring" in S2_SCORE_VISUAL_MAPPING_SUMMARY.lower()
    assert "clamped for drawing only" in S2_SCORE_VISUAL_MAPPING_SUMMARY.lower()
    # Source frame immutability.
    assert np.array_equal(frame, solid_frame(bg, h=120, w=400))


def test_s2_skeleton_nodes_only_when_unreadable(tmp_path: Path) -> None:
    slp = tmp_path / "pose.slp"
    slp.write_bytes(b"not-a-real-slp")
    edges, source = _resolve_skeleton_edges(slp)
    assert edges == ()
    assert source is not None
    assert source.startswith("unresolved:")
    assert resolve_review_skeleton_edges(slp) == ()


def test_s2_skeleton_injected_topology_edges(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    topology = (("nose", "neck"), ("neck", "spine_base"))

    class _Labels:
        pass

    def _fake_resolve(path, *, labels=None):  # type: ignore[no-untyped-def]
        return topology

    monkeypatch.setattr(
        "overlay.source_s2.resolve_review_skeleton_edges",
        _fake_resolve,
    )
    edges, source = _resolve_skeleton_edges(tmp_path / "pose.slp")
    assert edges == topology
    assert source is not None
    assert "s2_pose_slp" in source


def test_pose_layer_s2_uses_p_prefix_not_m() -> None:
    layer = PoseLayer(LayerId.S2_POSE_INFERRED)
    frame = solid_frame(h=48, w=64)
    from overlay.context import OverlayContext, OverlayObservation
    from overlay.contracts import FrameDomain, PlateauPolicy

    obs = OverlayObservation(
        layer_id=LayerId.S2_POSE_INFERRED,
        available=True,
        reason=None,
        payload={
            "legend": "inferred pose (S2 predictions; provisional IDs)",
            "points": [
                {"track": 0, "node": "nose", "x": 10.0, "y": 10.0, "score": 0.8},
            ],
        },
    )
    timing = PreparedFrameTiming(
        prepared_frame_idx=0,
        plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
        available=True,
        dense_sample_idx=0,
        plateau_sample_indices=(0,),
        mapping=None,
        reason=None,
    )
    ctx = OverlayContext(
        prepared_frame_idx=0,
        timing=timing,
        frame_domain=FrameDomain.PREPARED,
        plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
        style=DrawConfig(legend_text=True, extra={"show_confidence": False}),
        selected_layers=(LayerId.S2_POSE_INFERRED,),
    )
    out = layer.render(frame, ctx, obs)
    assert out is not None
