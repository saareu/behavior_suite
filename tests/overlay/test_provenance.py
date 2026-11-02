"""Manifest serialization and no-scientific-write tests."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from tests.behavioral_review._fixtures import build_hand_computed_sync
from tests.overlay._fixtures import (
    solid_frame,
    write_behavior_states_parquet,
    write_s2_pose_parquet,
)

from behavioral_review.sync import SyncMap
from overlay import (
    OVERLAY_ARTIFACT_MANIFEST_SCHEMA,
    OVERLAY_RENDERER_VERSION,
    DrawConfig,
    FrameDomain,
    LayerId,
    OverlayCompositor,
    PlateauPolicy,
    PreparedVideoIdentity,
    build_overlay_artifact_manifest,
    collect_evidence,
    make_s2_inferred_pose_adapter,
    make_s8_exclusive_a_to_b,
    parse_overlay_artifact_manifest,
)
from project.validation_memo import artifact_fingerprint


def test_manifest_serialization_roundtrip(tmp_path: Path) -> None:
    sync_df = build_hand_computed_sync(n_prepared=3, plateau_repeats=2)
    sync = SyncMap.from_sync_index(sync_df)
    s2_path = write_s2_pose_parquet(tmp_path / "pose.parquet")
    states_path = write_behavior_states_parquet(tmp_path / "states.parquet", sync_df)
    s2 = make_s2_inferred_pose_adapter(s2_path, run_id="pose_inference__t")
    s8 = make_s8_exclusive_a_to_b(states_path, run_id="behavioral_arbitration__t")
    layers = [LayerId.S2_POSE_INFERRED, LayerId.S8_EXCLUSIVE_A_TO_B]
    evidence = collect_evidence(
        sync, 1, {LayerId.S2_POSE_INFERRED: s2, LayerId.S8_EXCLUSIVE_A_TO_B: s8},
        selected_layers=layers,
        plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
    )
    style = DrawConfig()
    composed = OverlayCompositor(plateau_policy=PlateauPolicy.LAST_IN_PLATEAU).render_frame(
        solid_frame(),
        prepared_frame_idx=1,
        evidence=evidence,
        selected_layers=layers,
        style=style,
    )
    video = PreparedVideoIdentity(
        path=tmp_path / "prepared_video.mp4",
        digest_sha256="abc123",
        fps=10.0,
        frame_count=3,
    )
    manifest = build_overlay_artifact_manifest(
        sync=sync,
        prepared_video=video,
        source_identities=(s2.source_identity, s8.source_identity),
        selected_layers=layers,
        plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
        style=style,
        composed=composed,
        timing=evidence.timing,
        output_frame_domain=FrameDomain.PREPARED,
        output_fps=10.0,
    )
    assert manifest.schema_version == OVERLAY_ARTIFACT_MANIFEST_SCHEMA
    assert manifest.renderer_version == OVERLAY_RENDERER_VERSION
    assert manifest.plateau_policy == PlateauPolicy.LAST_IN_PLATEAU.value
    assert manifest.output_frame_domain == FrameDomain.PREPARED.value
    assert "s2.pose_inferred" in manifest.layer_selections

    out = manifest.write_json(tmp_path / "overlay_artifact_manifest.json")
    loaded = parse_overlay_artifact_manifest(json.loads(out.read_text(encoding="utf-8")))
    assert loaded.to_dict() == manifest.to_dict()


def test_no_scientific_writes(tmp_path: Path) -> None:
    """Compositor + adapters must not mutate science parquet contents."""

    sync_df = build_hand_computed_sync(n_prepared=3, plateau_repeats=2)
    sync = SyncMap.from_sync_index(sync_df)
    s2_path = write_s2_pose_parquet(tmp_path / "science" / "pose.parquet")
    states_path = write_behavior_states_parquet(
        tmp_path / "science" / "behavior_states.parquet", sync_df
    )
    before = {
        "pose": artifact_fingerprint(s2_path),
        "states": artifact_fingerprint(states_path),
        "pose_df": pd.read_parquet(s2_path).copy(),
        "states_df": pd.read_parquet(states_path).copy(),
    }
    s2 = make_s2_inferred_pose_adapter(s2_path)
    s8 = make_s8_exclusive_a_to_b(states_path)
    layers = [LayerId.S2_POSE_INFERRED, LayerId.S8_EXCLUSIVE_A_TO_B]
    evidence = collect_evidence(
        sync,
        1,
        {LayerId.S2_POSE_INFERRED: s2, LayerId.S8_EXCLUSIVE_A_TO_B: s8},
        selected_layers=layers,
    )
    OverlayCompositor().render_frame(
        solid_frame(),
        prepared_frame_idx=1,
        evidence=evidence,
        selected_layers=layers,
    )
    build_overlay_artifact_manifest(
        sync=sync,
        prepared_video=PreparedVideoIdentity(path=None, fps=10.0),
        source_identities=(s2.source_identity, s8.source_identity),
        selected_layers=layers,
        plateau_policy=PlateauPolicy.LAST_IN_PLATEAU,
        style=DrawConfig(),
        timing=evidence.timing,
    ).write_json(tmp_path / "overlay_only" / "manifest.json")

    assert artifact_fingerprint(s2_path) == before["pose"]
    assert artifact_fingerprint(states_path) == before["states"]
    pd.testing.assert_frame_equal(pd.read_parquet(s2_path), before["pose_df"])
    pd.testing.assert_frame_equal(pd.read_parquet(states_path), before["states_df"])
    # Overlay output is separate from science tree.
    assert (tmp_path / "overlay_only" / "manifest.json").is_file()
    assert not (tmp_path / "science" / "overlay_artifact_manifest.json").exists()
