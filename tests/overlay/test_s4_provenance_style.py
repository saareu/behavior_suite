"""C2-B S4 provenance mapping, drawing honesty, and mouse-identity tests."""

from __future__ import annotations

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
    OverlayAdapterError,
    OverlayCompositor,
    OverlayContractError,
    S4ProvenanceVisualClass,
    collect_evidence,
    make_s4_final_pose_adapter,
    s4_provenance_visual_class,
)
from overlay.contracts import (
    S4_PROVENANCE_IDENTITY_CORRECTED,
    S4_PROVENANCE_KALMAN_GAP_FILL,
    S4_PROVENANCE_LINEAR_GAP_FILL,
    S4_PROVENANCE_MISSING,
    S4_PROVENANCE_NAMES,
    S4_PROVENANCE_ORIGINAL,
    S4_PROVENANCE_PROTECTED_MANUAL_BLANK,
    S4_PROVENANCE_TEMPORAL_RESAMPLE,
    S4_PROVENANCE_VISUAL_CLASS_BY_CODE,
)
from overlay.layers.pose import PoseLayer
from pose_finalization.contracts import PROVENANCE_NAMES


def test_exact_s4_provenance_code_mapping() -> None:
    expected = {
        0: S4ProvenanceVisualClass.UNAVAILABLE,
        1: S4ProvenanceVisualClass.OBSERVED,
        2: S4ProvenanceVisualClass.OBSERVED,
        3: S4ProvenanceVisualClass.TEMPORAL_RESAMPLE,
        4: S4ProvenanceVisualClass.GAP_FILL,
        5: S4ProvenanceVisualClass.GAP_FILL,
        6: S4ProvenanceVisualClass.UNAVAILABLE,
    }
    assert dict(S4_PROVENANCE_VISUAL_CLASS_BY_CODE) == expected
    for code, visual in expected.items():
        assert s4_provenance_visual_class(code) is visual
        assert S4_PROVENANCE_NAMES[code] == PROVENANCE_NAMES[code]
    with pytest.raises(OverlayContractError, match="Unknown S4 provenance"):
        s4_provenance_visual_class(99)


def _write_mixed_s4(path: Path) -> Path:
    rows = [
        # Track 0 mixed provenance skeleton: nose observed, neck resample, spine gap-fill.
        {
            "sample_idx": 1,
            "track": 0,
            "node": "nose",
            "x": 10.0,
            "y": 10.0,
            "provenance": S4_PROVENANCE_ORIGINAL,
            "provenance_name": "original",
        },
        {
            "sample_idx": 1,
            "track": 0,
            "node": "neck",
            "x": 20.0,
            "y": 12.0,
            "provenance": S4_PROVENANCE_TEMPORAL_RESAMPLE,
            "provenance_name": "temporal_resample",
        },
        {
            "sample_idx": 1,
            "track": 0,
            "node": "spine_base",
            "x": 30.0,
            "y": 14.0,
            "provenance": S4_PROVENANCE_LINEAR_GAP_FILL,
            "provenance_name": "linear_gap_fill",
        },
        # Protected blank + missing must not become drawable.
        {
            "sample_idx": 1,
            "track": 0,
            "node": "tail_base",
            "x": 40.0,
            "y": 16.0,
            "provenance": S4_PROVENANCE_PROTECTED_MANUAL_BLANK,
            "provenance_name": "protected_manual_blank",
        },
        {
            "sample_idx": 1,
            "track": 0,
            "node": "headstage",
            "x": np.nan,
            "y": np.nan,
            "provenance": S4_PROVENANCE_MISSING,
            "provenance_name": "missing",
        },
        # Track 1 identity-corrected + kalman (second mouse).
        {
            "sample_idx": 1,
            "track": 1,
            "node": "nose",
            "x": 50.0,
            "y": 40.0,
            "provenance": S4_PROVENANCE_IDENTITY_CORRECTED,
            "provenance_name": "identity_corrected",
        },
        {
            "sample_idx": 1,
            "track": 1,
            "node": "neck",
            "x": 58.0,
            "y": 42.0,
            "provenance": S4_PROVENANCE_KALMAN_GAP_FILL,
            "provenance_name": "kalman_gap_fill",
        },
        # Missing coords with observed provenance → not drawable.
        {
            "sample_idx": 1,
            "track": 1,
            "node": "spine_base",
            "x": None,
            "y": None,
            "provenance": S4_PROVENANCE_ORIGINAL,
            "provenance_name": "original",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_adapter_preserves_codes_and_visual_classes(tmp_path: Path) -> None:
    path = _write_mixed_s4(tmp_path / "final_pose.parquet")
    adapter = make_s4_final_pose_adapter(path, run_id="pose_finalization__t")
    sync = SyncMap.from_sync_index(
        build_hand_computed_sync(n_prepared=2, plateau_repeats=2)
    )
    # prepared 0 → dense plateau {0,1}; LAST_IN_PLATEAU → 1
    from overlay.timing import resolve_prepared_frame_timing

    timing = resolve_prepared_frame_timing(sync, 0)
    assert timing.dense_sample_idx == 1
    obs = adapter.observe(timing, prepared_frame_idx=0)
    assert obs.available
    by_key = {(p["track"], p["node"]): p for p in obs.payload["points"]}

    assert by_key[(0, "nose")]["provenance"] == S4_PROVENANCE_ORIGINAL
    assert by_key[(0, "nose")]["visual_class"] == "observed"
    assert by_key[(0, "nose")]["drawable"] is True

    assert by_key[(0, "neck")]["visual_class"] == "temporal_resample"
    assert by_key[(0, "spine_base")]["visual_class"] == "gap_fill"

    assert by_key[(0, "tail_base")]["drawable"] is False
    assert by_key[(0, "tail_base")]["visual_class"] == "unavailable"
    assert by_key[(0, "headstage")]["drawable"] is False
    assert by_key[(0, "headstage")]["x"] is None

    assert by_key[(1, "nose")]["visual_class"] == "observed"
    assert by_key[(1, "neck")]["visual_class"] == "gap_fill"
    assert by_key[(1, "spine_base")]["drawable"] is False


def test_observed_resample_gapfill_visually_distinct(tmp_path: Path) -> None:
    path = _write_mixed_s4(tmp_path / "final_pose.parquet")
    adapter = make_s4_final_pose_adapter(path)
    sync = SyncMap.from_sync_index(
        build_hand_computed_sync(n_prepared=2, plateau_repeats=2)
    )
    evidence = collect_evidence(
        sync,
        0,
        {LayerId.S4_POSE_FINAL: adapter},
        selected_layers=[LayerId.S4_POSE_FINAL],
    )
    style = DrawConfig(
        draw_skeleton=True,
        legend_text=True,
        extra={"skeleton_edges": [("nose", "neck"), ("neck", "spine_base")]},
    )
    bright = solid_frame((230, 230, 230), h=80, w=96)
    dark = solid_frame((20, 20, 20), h=80, w=96)
    compositor = OverlayCompositor()
    composed_bright = compositor.render_frame(
        bright,
        prepared_frame_idx=0,
        evidence=evidence,
        selected_layers=[LayerId.S4_POSE_FINAL],
        style=style,
    )
    composed_dark = compositor.render_frame(
        dark,
        prepared_frame_idx=0,
        evidence=evidence,
        selected_layers=[LayerId.S4_POSE_FINAL],
        style=style,
    )
    assert not np.array_equal(composed_bright.frame_bgr, bright)
    assert not np.array_equal(composed_dark.frame_bgr, dark)

    # Direct layer render for pixel-level class distinction.
    from overlay.context import OverlayContext
    from overlay.contracts import FrameDomain, PlateauPolicy

    obs = evidence.get(LayerId.S4_POSE_FINAL)
    assert obs is not None and obs.available
    ctx = OverlayContext(
        prepared_frame_idx=0,
        timing=evidence.timing,
        frame_domain=FrameDomain.PREPARED,
        plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
        style=style,
        selected_layers=(LayerId.S4_POSE_FINAL,),
    )
    canvas = PoseLayer(LayerId.S4_POSE_FINAL).render(
        solid_frame((40, 40, 40), h=80, w=96), ctx, obs
    )

    # Nose (observed, filled) should have identity color at center.
    assert not np.array_equal(canvas[10, 10], np.array([40, 40, 40], dtype=np.uint8))
    # Protected blank / missing coordinates must leave those pixels unchanged.
    blank_ref = solid_frame((40, 40, 40), h=80, w=96)
    assert np.array_equal(canvas[16, 40], blank_ref[16, 40])  # protected blank loc
    # Missing headstage had NaN coords — nothing drawn at a sentinel far pixel.
    assert np.array_equal(canvas[70, 90], blank_ref[70, 90])

    # Marker shapes differ: sample patches around nose / neck / spine.
    nose_patch = canvas[8:13, 8:13].copy()
    neck_patch = canvas[10:15, 18:23].copy()
    spine_patch = canvas[12:17, 28:33].copy()
    assert not np.array_equal(nose_patch, neck_patch)
    assert not np.array_equal(neck_patch, spine_patch)
    assert not np.array_equal(nose_patch, spine_patch)


def test_no_edges_across_missing_nodes(tmp_path: Path) -> None:
    rows = [
        {
            "sample_idx": 0,
            "track": 0,
            "node": "nose",
            "x": 10.0,
            "y": 10.0,
            "provenance": S4_PROVENANCE_ORIGINAL,
            "provenance_name": "original",
        },
        {
            "sample_idx": 0,
            "track": 0,
            "node": "neck",
            "x": np.nan,
            "y": np.nan,
            "provenance": S4_PROVENANCE_MISSING,
            "provenance_name": "missing",
        },
        {
            "sample_idx": 0,
            "track": 0,
            "node": "spine_base",
            "x": 30.0,
            "y": 10.0,
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
    frame = solid_frame((0, 0, 0), h=40, w=64)
    composed = OverlayCompositor().render_frame(
        frame,
        prepared_frame_idx=0,
        evidence=evidence,
        selected_layers=[LayerId.S4_POSE_FINAL],
        style=DrawConfig(
            draw_skeleton=True,
            legend_text=False,
            extra={"skeleton_edges": [("nose", "neck"), ("neck", "spine_base"), ("nose", "spine_base")]},
        ),
    )
    # Midpoint of nose→spine edge (20,10) may be drawn (both drawable).
    # Midpoint of nose→neck would be near (10,10)-(nan) — not drawn via missing neck.
    # Ensure protected path: pixels around missing neck location stay mostly black
    # except possible nearby node halos; the far midpoint of a missing edge stays dark.
    mid_missing = composed.frame_bgr[10, 15]
    # nose-spine midpoint at x=20 should be lit by the valid edge.
    mid_valid = composed.frame_bgr[10, 20]
    assert int(mid_valid.sum()) > int(mid_missing.sum())


def test_mouse_identity_consistency(tmp_path: Path) -> None:
    path = _write_mixed_s4(tmp_path / "final_pose.parquet")
    adapter = make_s4_final_pose_adapter(path)
    sync = SyncMap.from_sync_index(
        build_hand_computed_sync(n_prepared=2, plateau_repeats=2)
    )
    evidence = collect_evidence(
        sync, 0, {LayerId.S4_POSE_FINAL: adapter}, selected_layers=[LayerId.S4_POSE_FINAL]
    )
    obs = evidence.get(LayerId.S4_POSE_FINAL)
    assert obs is not None
    tracks = {int(p["track"]) for p in obs.payload["points"] if p.get("drawable")}
    assert tracks == {0, 1}
    # Coordinates must remain unchanged from artifact.
    art = pd.read_parquet(path)
    for p in obs.payload["points"]:
        if not p.get("drawable"):
            continue
        row = art[
            (art.sample_idx == 1)
            & (art.track == p["track"])
            & (art.node == p["node"])
        ].iloc[0]
        assert float(p["x"]) == pytest.approx(float(row.x))
        assert float(p["y"]) == pytest.approx(float(row.y))


def test_unknown_provenance_still_rejected(tmp_path: Path) -> None:
    path = tmp_path / "final_pose.parquet"
    pd.DataFrame(
        [
            {
                "sample_idx": 0,
                "track": 0,
                "node": "nose",
                "x": 1.0,
                "y": 2.0,
                "provenance": 42,
                "provenance_name": "invented",
            }
        ]
    ).to_parquet(path, index=False)
    adapter = make_s4_final_pose_adapter(path)
    sync = SyncMap.from_sync_index(
        build_hand_computed_sync(n_prepared=1, plateau_repeats=1)
    )
    from overlay.timing import resolve_prepared_frame_timing

    timing = resolve_prepared_frame_timing(sync, 0)
    with pytest.raises(OverlayAdapterError, match="Unknown S4 provenance"):
        adapter.observe(timing, prepared_frame_idx=0)
