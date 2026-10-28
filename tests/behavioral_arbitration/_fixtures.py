"""Synthetic S6/S7 fixtures based on real wide/feature schemas."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from behavioral_arbitration.contracts import (
    REQUIRED_DYADIC_DETECTORS,
    REQUIRED_FOCAL_DETECTORS,
)
from behavioral_inference.contracts import (
    BEHAVIORAL_INFERENCE_DIRNAME,
    DYADIC_WIDE_FILENAME,
    FOCAL_WIDE_FILENAME,
    MODEL_MANIFEST_FILENAME,
    PREDICTIONS_FILENAME,
    QC_FILENAME,
    RUN_META_FILENAME,
    SCORE_ORIGIN_MODEL_OBSERVED,
    SCORE_ORIGIN_ROUTING_SENTINEL,
    SETTINGS_USED_FILENAME,
    STATUS_COMPLETED,
)
from combined_features.contracts import (
    COMBINED_FEATURES_DIRNAME,
    DYADIC_FEATURES_FILENAME,
)
from combined_features.contracts import (
    RUN_META_FILENAME as S6_RUN_META_FILENAME,
)
from combined_features.contracts import (
    STATUS_COMPLETED as S6_STATUS_COMPLETED,
)

MOUSE_A = 1
MOUSE_B = 2
N_FRAMES = 4
N_FRAMES_TEMPORAL = 48
FPS = 120.0


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def make_frame_indices(n: int = N_FRAMES) -> np.ndarray:
    return np.arange(n, dtype=np.int64)


def make_dyadic_wide(
    *,
    n_frames: int = N_FRAMES,
    detectors: tuple[str, ...] = REQUIRED_DYADIC_DETECTORS,
    invalid_contact_frames: set[int] | None = None,
    exp_name: str = "synth",
) -> pd.DataFrame:
    """Undirected pair rows matching S7 dyadic wide schema."""

    frames = make_frame_indices(n_frames)
    invalid_contact_frames = invalid_contact_frames or set()
    rows: list[dict[str, Any]] = []
    for fi in frames:
        row: dict[str, Any] = {
            "schema_version": "dyadic_predictions_wide_v1",
            "exp_name": exp_name,
            "sample_idx": int(fi),
            "dense_frame_idx": int(fi),
            "time_sec": float(fi) / FPS,
            "pair_id": f"{MOUSE_A}_{MOUSE_B}",
            "mouse_a": MOUSE_A,
            "mouse_b": MOUSE_B,
            "focal_id": MOUSE_A,
            "partner_id": MOUSE_B,
        }
        for det in detectors:
            if det == "contact" and int(fi) in invalid_contact_frames:
                row[f"prob_{det}"] = 0.0
                row[f"pred_{det}"] = 0
                row[f"input_valid_{det}"] = False
                row[f"score_origin_{det}"] = SCORE_ORIGIN_ROUTING_SENTINEL
                row[f"invalid_reason_{det}"] = "occlusion_mask_excluded"
                row[f"threshold_{det}"] = 0.5
            else:
                # Mix of valid positives and valid negatives.
                positive = int(fi) % 2 == 0 and det in ("nose_to_nose", "contact")
                row[f"prob_{det}"] = 0.9 if positive else 0.01
                row[f"pred_{det}"] = 1 if positive else 0
                row[f"input_valid_{det}"] = True
                row[f"score_origin_{det}"] = SCORE_ORIGIN_MODEL_OBSERVED
                row[f"invalid_reason_{det}"] = None
                row[f"threshold_{det}"] = 0.1
        rows.append(row)
    return pd.DataFrame(rows)


def make_focal_wide(
    *,
    n_frames: int = N_FRAMES,
    detectors: tuple[str, ...] = REQUIRED_FOCAL_DETECTORS,
    exp_name: str = "synth",
) -> pd.DataFrame:
    """Directional rows matching S7 focal wide schema (both directions)."""

    frames = make_frame_indices(n_frames)
    rows: list[dict[str, Any]] = []
    for fi in frames:
        for focal, partner in ((MOUSE_A, MOUSE_B), (MOUSE_B, MOUSE_A)):
            row: dict[str, Any] = {
                "schema_version": "focal_predictions_wide_v1",
                "exp_name": exp_name,
                "sample_idx": int(fi),
                "dense_frame_idx": int(fi),
                "time_sec": float(fi) / FPS,
                "focal_id": focal,
                "partner_id": partner,
                "is_total_occlusion": 0,
            }
            for det in detectors:
                # Directional signal differs by direction for approach.
                positive = det == "approach" and focal == MOUSE_A and int(fi) == 1
                row[f"prob_{det}"] = 0.8 if positive else 0.05
                row[f"pred_{det}"] = 1 if positive else 0
                row[f"input_valid_{det}"] = True
                row[f"score_origin_{det}"] = SCORE_ORIGIN_MODEL_OBSERVED
                row[f"invalid_reason_{det}"] = None
                row[f"threshold_{det}"] = 0.4
                if det == "nose_to_body":
                    row["prob_nose_to_body_raw"] = row[f"prob_{det}"]
            rows.append(row)
    return pd.DataFrame(rows)


def make_s6_features(
    *,
    n_frames: int = N_FRAMES,
    exp_name: str = "synth",
    include_veto: bool = True,
    drop_hard_gap: tuple[str, ...] = (),
    distance_overrides: dict[tuple[int, int, int], dict[str, float]] | None = None,
) -> pd.DataFrame:
    """Directional S6-like feature table."""

    frames = make_frame_indices(n_frames)
    rows: list[dict[str, Any]] = []
    overrides = distance_overrides or {}
    for fi in frames:
        for focal, partner in ((MOUSE_A, MOUSE_B), (MOUSE_B, MOUSE_A)):
            row: dict[str, Any] = {
                "schema_version": "dyadic_features_full_v1",
                "session_root": "synth",
                "exp_name": exp_name,
                "focal_id": focal,
                "partner_id": partner,
                "sample_idx": int(fi),
                "dense_frame_idx": int(fi),
                "time_sec": float(fi) / FPS,
                "orig_frame_idx": int(fi),
                "source_s4_run_id": "s4_synth",
                "source_s5_run_id": "s5_synth",
                "distance_nose_i_nose_j_norm": 0.1,
                "distance_nose_i_side_j_norm": 0.2,
                "distance_nose_i_spine_j_norm": 0.3,
                "distance_nose_i_tail_j_norm": 0.2,
                "distance_spine_i_spine_j_norm": 0.4,
                "cc_count": 2,
                "is_total_occlusion": 0,
            }
            key = (int(fi), focal, partner)
            if key in overrides:
                row.update(overrides[key])
            if include_veto:
                row.update(
                    {
                        "body_contraction_rate_7": 0.0,
                        "behind_score_bl_j": 0.0,
                        "cos_bearing_i_to_j": 1.0,
                        "rolling_mean_intent_15": 0.0,
                        "rolling_mean_projected_vel_i_to_j_60": 0.0,
                        "rolling_speed_i_15": 50.0,
                        "rolling_speed_j_15": 50.0,
                        "focal_effort_ratio_15": 1.0,
                        "spine_directed_vel_15_i": -10.0,
                        "rolling_mean_projected_vel_combined_i_to_j_7": 0.0,
                        "signed_kinematic_intent": 0.0,
                        "signed_kinematic_intent_nose_or_neck": 0.0,
                        "signed_kinematic_intent_neck": 0.0,
                        "rolling_speed_i_60": 50.0,
                    }
                )
            for col in drop_hard_gap:
                row.pop(col, None)
            rows.append(row)
    return pd.DataFrame(rows)


def _write_json(path: Path, payload: MappingLike) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


MappingLike = dict[str, Any]


def build_synthetic_session(
    root: Path,
    *,
    n_frames: int = N_FRAMES,
    stale_s6: bool = False,
    missing_detectors: tuple[str, ...] = (),
    drop_hard_gap: tuple[str, ...] = (),
    three_mice: bool = False,
    invalid_contact_frames: set[int] | None = None,
    distance_overrides: dict[tuple[int, int, int], dict[str, float]] | None = None,
    include_veto: bool = True,
) -> dict[str, Any]:
    """Create a mini session with linked S6 + eligible-looking S7 artifacts.

    Staleness / eligibility: when ``stale_s6=True``, the S7 meta points at a
    missing/non-eligible S6 directory while a *different* newer S6 exists.
    """

    session = root / "session"
    session.mkdir(parents=True, exist_ok=True)

    s6_run_id = "combined_features__20260101T000000"
    s6_dir = session / COMBINED_FEATURES_DIRNAME / s6_run_id
    s6_dir.mkdir(parents=True, exist_ok=True)

    features = make_s6_features(
        n_frames=n_frames,
        drop_hard_gap=drop_hard_gap,
        distance_overrides=distance_overrides,
        include_veto=include_veto,
    )
    if three_mice:
        # Add a third mouse direction to break K=2.
        extra = features.iloc[:1].copy()
        extra["focal_id"] = 3
        extra["partner_id"] = MOUSE_A
        features = pd.concat([features, extra], ignore_index=True)

    feat_path = s6_dir / DYADIC_FEATURES_FILENAME
    features.to_parquet(feat_path, index=False)
    feat_sha = _sha256_file(feat_path)

    # Minimal S6 meta — enough for path existence; full S6 validate needs S4/S5.
    # Tests that need full validate_joint will monkeypatch eligibility helpers.
    _write_json(
        s6_dir / S6_RUN_META_FILENAME,
        {
            "schema_version": "combined_features_run_meta_v1",
            "run_id": s6_run_id,
            "status": S6_STATUS_COMPLETED,
            "session_root": str(session),
            "source_s4": {
                "run_id": "s4_synth",
                "run_dir": str(session / "pose_finalization" / "s4_synth"),
                "final_pose_sha256": "a" * 64,
                "prepared_sync_sha256": "b" * 64,
            },
            "source_s5": {
                "run_id": "s5_synth",
                "run_dir": str(session / "blob_features" / "s5_synth"),
                "blob_features_sha256": "c" * 64,
                "prepared_sync_sha256": "b" * 64,
            },
        },
    )

    dyadic_dets = tuple(d for d in REQUIRED_DYADIC_DETECTORS if d not in missing_detectors)
    focal_dets = tuple(d for d in REQUIRED_FOCAL_DETECTORS if d not in missing_detectors)
    dyadic = make_dyadic_wide(
        n_frames=n_frames,
        detectors=dyadic_dets,
        invalid_contact_frames=invalid_contact_frames,
    )
    focal = make_focal_wide(n_frames=n_frames, detectors=focal_dets)
    if three_mice:
        extra_f = focal.iloc[:1].copy()
        extra_f["focal_id"] = 3
        extra_f["partner_id"] = MOUSE_A
        focal = pd.concat([focal, extra_f], ignore_index=True)

    s7_run_id = "behavioral_inference__20260101T010000"
    s7_dir = session / BEHAVIORAL_INFERENCE_DIRNAME / s7_run_id
    s7_dir.mkdir(parents=True, exist_ok=True)
    dyadic.to_parquet(s7_dir / DYADIC_WIDE_FILENAME, index=False)
    focal.to_parquet(s7_dir / FOCAL_WIDE_FILENAME, index=False)
    # Minimal canonical stub so discovery sees predictions artifact.
    (s7_dir / PREDICTIONS_FILENAME).write_bytes(b"parquet-stub")
    (s7_dir / QC_FILENAME).write_text("{}\n", encoding="utf-8")
    (s7_dir / SETTINGS_USED_FILENAME).write_text("schema_version: x\n", encoding="utf-8")

    linked_s6_dir = s6_dir
    if stale_s6:
        # Point S7 at a missing linked S6; create a newer eligible-looking S6.
        linked_s6_dir = session / COMBINED_FEATURES_DIRNAME / "combined_features__MISSING"
        newer = session / COMBINED_FEATURES_DIRNAME / "combined_features__20260102T000000"
        newer.mkdir(parents=True, exist_ok=True)
        features.to_parquet(newer / DYADIC_FEATURES_FILENAME, index=False)
        _write_json(
            newer / S6_RUN_META_FILENAME,
            {
                "schema_version": "combined_features_run_meta_v1",
                "run_id": newer.name,
                "status": S6_STATUS_COMPLETED,
                "session_root": str(session),
                "source_s4": {"run_id": "s4", "run_dir": str(session / "s4")},
                "source_s5": {"run_id": "s5", "run_dir": str(session / "s5")},
            },
        )

    _write_json(
        s7_dir / RUN_META_FILENAME,
        {
            "schema_version": "behavioral_inference_run_meta_v1",
            "run_id": s7_run_id,
            "status": STATUS_COMPLETED,
            "session_root": str(session),
            "completed_at": "2026-01-01T01:00:00+00:00",
            "created_at": "2026-01-01T01:00:00+00:00",
            "detectors_selected": {
                "dyadic": list(dyadic_dets),
                "focal": list(focal_dets),
            },
            "source_s6": {
                "run_id": s6_run_id if not stale_s6 else "combined_features__MISSING",
                "run_dir": str(linked_s6_dir),
                "dyadic_features": str(
                    (linked_s6_dir / DYADIC_FEATURES_FILENAME)
                    if not stale_s6
                    else linked_s6_dir / DYADIC_FEATURES_FILENAME
                ),
                "dyadic_features_sha256": feat_sha,
            },
        },
    )
    _write_json(
        s7_dir / MODEL_MANIFEST_FILENAME,
        {"schema_version": "behavioral_inference_model_manifest_v1", "bundles": []},
    )

    return {
        "session_root": session,
        "s6_run_dir": s6_dir,
        "s7_run_dir": s7_dir,
        "s6_run_id": s6_run_id,
        "s7_run_id": s7_run_id,
        "dyadic": dyadic,
        "focal": focal,
        "features": features,
        "feat_sha": feat_sha,
    }


def make_directional_temporal_frame(
    *,
    frame: int,
    focal: int = MOUSE_A,
    partner: int = MOUSE_B,
    exp_name: str = "synth",
    behavior_label: str = "engaged",
) -> dict[str, Any]:
    """Minimal directional row for Sprint-3 temporal / bout unit tests."""

    from behavioral_arbitration.contracts import (
        REQUIRED_DYADIC_DETECTORS,
        REQUIRED_FOCAL_DETECTORS,
    )
    from behavioral_inference.contracts import SCORE_ORIGIN_MODEL_OBSERVED

    detectors = tuple(
        dict.fromkeys((*REQUIRED_DYADIC_DETECTORS, *REQUIRED_FOCAL_DETECTORS, "engaged"))
    )
    row: dict[str, Any] = {
        "exp_name": exp_name,
        "sample_idx": int(frame),
        "dense_frame_idx": int(frame),
        "time_sec": float(frame) / FPS,
        "focal_id": int(focal),
        "partner_id": int(partner),
        "mouse_a": min(int(focal), int(partner)),
        "mouse_b": max(int(focal), int(partner)),
        "pair_id": f"{min(int(focal), int(partner))}|{max(int(focal), int(partner))}",
        "cc_count": 2,
        "is_total_occlusion": 0,
        "behavior_label": behavior_label,
        "hierarchy_residual_kind": "",
        "undefined_reason": "",
        "label_reconstruction": "",
        "distance_nose_i_nose_j_norm": 1.0,
        "distance_nose_i_side_j_norm": 1.0,
        "distance_nose_i_spine_j_norm": 1.0,
        "distance_nose_i_tail_j_norm": 1.0,
        "distance_spine_i_spine_j_norm": 1.0,
        "signed_kinematic_intent": 0.0,
        "signed_kinematic_intent_nose_or_neck": 0.0,
        "signed_kinematic_intent_neck": 0.0,
        "rolling_speed_i_60": 50.0,
    }
    for det in detectors:
        row[f"prob_{det}"] = 0.01
        row[f"pred_{det}"] = 1 if det == behavior_label else 0
        row[f"arbitration_input_valid_{det}"] = True
        row[f"input_valid_{det}"] = True
        row[f"score_origin_{det}"] = SCORE_ORIGIN_MODEL_OBSERVED
        row[f"s7_prob_{det}"] = row[f"prob_{det}"]
        row[f"s7_pred_{det}"] = row[f"pred_{det}"]
    return row


def make_directional_timeline(
    labels: list[str],
    *,
    focal: int = MOUSE_A,
    partner: int = MOUSE_B,
    start_frame: int = 0,
) -> pd.DataFrame:
    """Build a contiguous directional timeline from exclusive labels."""

    rows = [
        make_directional_temporal_frame(
            frame=start_frame + i,
            focal=focal,
            partner=partner,
            behavior_label=label,
        )
        for i, label in enumerate(labels)
    ]
    return pd.DataFrame(rows)


@pytest.fixture
def synth_tables() -> dict[str, pd.DataFrame]:
    return {
        "dyadic": make_dyadic_wide(),
        "focal": make_focal_wide(),
        "features": make_s6_features(),
    }


@pytest.fixture
def default_config():
    from behavioral_arbitration.config import load_behavioral_arbitration_config

    return load_behavioral_arbitration_config(None)
