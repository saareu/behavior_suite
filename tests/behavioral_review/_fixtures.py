"""Synthetic S9 packages and videos for S10 Sprint 1–2 tests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

import cv2  # type: ignore[import-untyped]
import numpy as np
import pandas as pd
import yaml

from project.file_digest import sha256_file
from project.validation_memo import artifact_fingerprint
from results_packaging.contracts import (
    BEHAVIOR_BOUTS_FILENAME,
    ETHOGRAM_SUMMARY_FILENAME,
    PACKAGE_MANIFEST_FILENAME,
    QC_FILENAME,
    RESULTS_PACKAGE_DIRNAME,
    RUN_META_FILENAME,
    SETTINGS_USED_FILENAME,
    STATUS_COMPLETED,
    SYNC_INDEX_FILENAME,
    TIME_ATOL_SEC,
    VERIFICATION_STRENGTH_CONTENT_HASH,
)


def build_mini_behavior_states(sync: pd.DataFrame) -> pd.DataFrame:
    """Dense dual-direction states with contact/SBS, engaged, and undefined."""

    rows: list[dict[str, Any]] = []
    n = int(len(sync))
    for i in range(n):
        t = float(sync.loc[i, "time_sec"])
        # Direction 0→1
        if i < 4:
            lab_a, recon_a, undef_a = "approach", "selected", ""
        elif i < 7:
            lab_a, recon_a, undef_a = "engaged", "selected", ""
        else:
            lab_a, recon_a, undef_a = "undefined", "selected", "unresolved_invalid_evidence"
        # Direction 1→0
        if i < 4:
            lab_b, recon_b, undef_b = "retract", "selected", ""
        elif i < 7:
            lab_b, recon_b, undef_b = "mutually_disengaged", "stitched", ""
        else:
            lab_b, recon_b, undef_b = "nose_to_body", "selected", ""
        pred_contact = 1 if 2 <= i <= 7 else 0
        pred_sbs = 1 if i >= n - 2 else 0
        for focal, partner, lab, recon, undef in (
            (0, 1, lab_a, recon_a, undef_a),
            (1, 0, lab_b, recon_b, undef_b),
        ):
            rows.append(
                {
                    "schema_version": "behavioral_arbitration_states_v1",
                    "dense_frame_idx": i,
                    "sample_idx": i,
                    "time_sec": t,
                    "focal_id": focal,
                    "partner_id": partner,
                    "mouse_a": 0,
                    "mouse_b": 1,
                    "pair_id": "0|1",
                    "behavior_label": lab,
                    "label_reconstruction": recon,
                    "undefined_reason": undef,
                    "hierarchy_residual_kind": (
                        "engaged_valid" if lab == "engaged" else ""
                    ),
                    "pred_contact": pred_contact,
                    "prob_contact": 0.9 if pred_contact else 0.1,
                    "pred_side_by_side": pred_sbs,
                    "prob_side_by_side": 0.8 if pred_sbs else 0.05,
                    "pred_approach": 1 if lab == "approach" else 0,
                    "prob_approach": 0.7 if lab == "approach" else 0.1,
                    "pred_retract": 1 if lab == "retract" else 0,
                    "prob_retract": 0.6 if lab == "retract" else 0.1,
                    "pred_engaged": 1 if lab == "engaged" else 0,
                    "prob_engaged": 0.5 if lab == "engaged" else 0.0,
                    "pred_nose_to_body": 1 if lab == "nose_to_body" else 0,
                    "prob_nose_to_body": 0.55 if lab == "nose_to_body" else 0.1,
                    "pred_mutually_disengaged": 1 if lab == "mutually_disengaged" else 0,
                    "prob_mutually_disengaged": (
                        0.65 if lab == "mutually_disengaged" else 0.1
                    ),
                    "input_valid_contact": True,
                    "score_origin_contact": "model_observed",
                    "arbitration_input_valid_contact": True,
                    "input_valid_side_by_side": True,
                    "score_origin_side_by_side": "model_observed",
                    "arbitration_input_valid_side_by_side": True,
                    "input_valid_approach": i != 8,  # one invalid sample
                    "score_origin_approach": (
                        "routing_sentinel" if i == 8 else "model_observed"
                    ),
                    "arbitration_input_valid_approach": i != 8,
                    "input_valid_retract": True,
                    "score_origin_retract": "model_observed",
                    "arbitration_input_valid_retract": True,
                }
            )
    return pd.DataFrame(rows)


def build_mini_s7_predictions(sync: pd.DataFrame) -> pd.DataFrame:
    """Long-form S7 predictions covering a few detectors and validity states."""

    rows: list[dict[str, Any]] = []
    for i in range(int(len(sync))):
        t = float(sync.loc[i, "time_sec"])
        for focal, partner, det, family in (
            (0, 1, "approach", "focal"),
            (1, 0, "retract", "focal"),
            (0, 1, "contact", "dyadic"),
        ):
            valid = not (det == "approach" and i == 8)
            origin = "routing_sentinel" if not valid else "model_observed"
            prob = 0.0 if not valid else (0.8 if i < 4 else 0.2)
            pred = 0 if not valid else int(prob >= 0.5)
            rows.append(
                {
                    "schema_version": "behavioral_inference_predictions_v1",
                    "sample_idx": i,
                    "dense_frame_idx": i,
                    "time_sec": t,
                    "prediction_grain": "directional",
                    "pair_id": "0|1",
                    "mouse_a": 0,
                    "mouse_b": 1,
                    "focal_id": focal,
                    "partner_id": partner,
                    "family": family,
                    "detector_name": det,
                    "specialist_name": None,
                    "output_role": "primary",
                    "model_version": "test",
                    "model_bundle_dir": None,
                    "model_sha256": None,
                    "prob_raw": prob,
                    "prob_effective": prob,
                    "threshold": 0.5,
                    "threshold_source": "test",
                    "prediction": pred,
                    "input_valid": valid,
                    "invalid_reason": "" if valid else "routing",
                    "score_origin": origin,
                    "source_s6_run_id": "combined_features__synth",
                    "session_root": "",
                }
            )
    return pd.DataFrame(rows)


def write_tiny_avi(path: Path, *, frame_count: int = 8, fps: float = 10.0) -> Path:
    """Write a tiny MJPG AVI with identifiable per-frame intensity."""

    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        float(fps),
        (32, 24),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not create test video at {path}")
    try:
        for i in range(int(frame_count)):
            frame = np.full((24, 32, 3), int(i * 20) % 256, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path


@contextmanager
def patch_s9_currently_eligible(
    *,
    source_s8_run_id: str = "behavioral_arbitration__synth",
) -> Iterator[None]:
    """Treat mini packages as currently eligible (no full S8 chain required)."""

    def _ok(run_dir: Path | str) -> dict[str, Any]:
        path = Path(run_dir).resolve()
        return {
            "s9_run_dir": path,
            "status": STATUS_COMPLETED,
            "session_root": str(path.parent.parent),
            "source_s8_run_dir": path.parent.parent
            / "behavioral_arbitration"
            / source_s8_run_id,
            "source_s8_run_id": source_s8_run_id,
            "behavior_states_sha256": None,
            "behavior_bouts_sha256": None,
        }

    def _eligible(_run_dir: Path | str) -> bool:
        return True

    with (
        patch(
            "behavioral_review.source.validate_s9_run_for_downstream",
            side_effect=_ok,
        ),
        patch(
            "behavioral_review.eligibility.validate_s9_run_for_downstream",
            side_effect=_ok,
        ),
        patch(
            "behavioral_review.eligibility.is_s9_run_eligible_for_downstream",
            side_effect=_eligible,
        ),
        patch(
            "behavioral_review.discovery.validate_s9_run_for_downstream",
            side_effect=_ok,
        ),
        patch(
            "behavioral_review.discovery.is_s9_run_eligible_for_downstream",
            side_effect=_eligible,
        ),
    ):
        yield


def build_hand_computed_sync(
    *,
    n_prepared: int = 5,
    plateau_repeats: int = 2,
    prepared_fps: float = 10.0,
    dense_fps: float = 120.0,
) -> pd.DataFrame:
    """Build a contiguous sync_index with intentional many-to-one plateaus.

    Dense samples map as ``prepared_frame_idx = sample_idx // plateau_repeats``
    (clipped). Residuals and exact-match flags are hand-consistent.
    """

    n_dense = int(n_prepared) * int(plateau_repeats)
    sample_idx = np.arange(n_dense, dtype=np.int64)
    prepared = (sample_idx // int(plateau_repeats)).astype(np.int64)
    time_sec = sample_idx.astype(np.float64) / float(dense_fps)
    prepared_time = prepared.astype(np.float64) / float(prepared_fps)
    residual = time_sec - prepared_time
    exact = np.abs(residual) <= TIME_ATOL_SEC
    return pd.DataFrame(
        {
            "schema_version": "results_package_sync_v1",
            "sample_idx": sample_idx,
            "dense_frame_idx": sample_idx.copy(),
            "time_sec": time_sec,
            "prepared_frame_idx": prepared,
            "prepared_time_sec": prepared_time,
            "is_exact_source_sample": exact,
            "timing_residual_sec": residual,
        }
    )


def build_mini_review_package(
    root: Path,
    *,
    n_prepared: int = 5,
    plateau_repeats: int = 2,
    package_run_id: str = "results_package__review_synth",
    s8_run_id: str = "behavioral_arbitration__synth",
    include_video: bool = True,
    include_s8_states: bool = True,
    video_sha_mismatch: bool = False,
    corrupt_sync: bool = False,
) -> dict[str, Any]:
    """Create a mini session + completed S9 package suitable for S10 open tests."""

    session = root / "session"
    preprocess = session / "preprocess"
    preprocess.mkdir(parents=True)
    video_path = preprocess / "prepared_video.avi"
    if include_video:
        write_tiny_avi(video_path, frame_count=n_prepared, fps=10.0)

    sync = build_hand_computed_sync(
        n_prepared=n_prepared,
        plateau_repeats=plateau_repeats,
        prepared_fps=10.0,
        dense_fps=120.0,
    )
    n_dense = int(len(sync))
    if corrupt_sync:
        sync.loc[0, "sample_idx"] = -1

    # Inclusive bouts: directional exclusive + contact + side_by_side.
    bouts = pd.DataFrame(
        [
            {
                "bout_id": f"{package_run_id}__bout_dir_a",
                "schema_version": "results_package_bouts_v1",
                "behavior_name": "approach",
                "channel": "exclusive",
                "identity_grain": "directional",
                "focal_id": 0,
                "partner_id": 1,
                "mouse_a_id": 0,
                "mouse_b_id": 1,
                "pair_id": "0|1",
                "start_sample_idx": 0,
                "end_sample_idx": 3,
                "start_time_sec": 0.0,
                "end_time_sec": 3 / 120.0,
                "bout_duration_frames": 4,
                "bout_duration_sec": 4 / 120.0,
                "mean_prob": 0.5,
                "max_prob": 0.5,
                "median_prob": 0.5,
                "frac_input_valid": 1.0,
                "label_reconstruction_mode": "test",
                "undefined_reason": "",
                "source_s7_run_id": "s7",
                "source_s6_run_id": "s6",
                "run_id": s8_run_id,
                "exp_name": "synth",
                "source_s8_bout_id": "bout_dir_a",
                "source_s8_run_id": s8_run_id,
                "package_run_id": package_run_id,
                "s8_bouts_schema_version": "behavioral_arbitration_bouts_v1",
                "prepared_frame_idx_start": int(sync.loc[0, "prepared_frame_idx"]),
                "prepared_frame_idx_end": int(sync.loc[3, "prepared_frame_idx"]),
                "prepared_time_sec_start": float(sync.loc[0, "prepared_time_sec"]),
                "prepared_time_sec_end": float(sync.loc[3, "prepared_time_sec"]),
            },
            {
                "bout_id": f"{package_run_id}__bout_dir_b",
                "schema_version": "results_package_bouts_v1",
                "behavior_name": "retract",
                "channel": "exclusive",
                "identity_grain": "directional",
                "focal_id": 1,
                "partner_id": 0,
                "mouse_a_id": 0,
                "mouse_b_id": 1,
                "pair_id": "0|1",
                "start_sample_idx": 4,
                "end_sample_idx": 5,
                "start_time_sec": 4 / 120.0,
                "end_time_sec": 5 / 120.0,
                "bout_duration_frames": 2,
                "bout_duration_sec": 2 / 120.0,
                "mean_prob": 0.4,
                "max_prob": 0.4,
                "median_prob": 0.4,
                "frac_input_valid": 1.0,
                "label_reconstruction_mode": "test",
                "undefined_reason": "",
                "source_s7_run_id": "s7",
                "source_s6_run_id": "s6",
                "run_id": s8_run_id,
                "exp_name": "synth",
                "source_s8_bout_id": "bout_dir_b",
                "source_s8_run_id": s8_run_id,
                "package_run_id": package_run_id,
                "s8_bouts_schema_version": "behavioral_arbitration_bouts_v1",
                "prepared_frame_idx_start": int(sync.loc[4, "prepared_frame_idx"]),
                "prepared_frame_idx_end": int(sync.loc[5, "prepared_frame_idx"]),
                "prepared_time_sec_start": float(sync.loc[4, "prepared_time_sec"]),
                "prepared_time_sec_end": float(sync.loc[5, "prepared_time_sec"]),
            },
            {
                "bout_id": f"{package_run_id}__bout_contact",
                "schema_version": "results_package_bouts_v1",
                "behavior_name": "contact",
                "channel": "contact",
                "identity_grain": "symmetric_pair",
                "focal_id": None,
                "partner_id": None,
                "mouse_a_id": 0,
                "mouse_b_id": 1,
                "pair_id": "0|1",
                "start_sample_idx": 2,
                "end_sample_idx": 7,
                "start_time_sec": 2 / 120.0,
                "end_time_sec": 7 / 120.0,
                "bout_duration_frames": 6,
                "bout_duration_sec": 6 / 120.0,
                "mean_prob": 0.9,
                "max_prob": 0.9,
                "median_prob": 0.9,
                "frac_input_valid": 1.0,
                "label_reconstruction_mode": "test",
                "undefined_reason": "",
                "source_s7_run_id": "s7",
                "source_s6_run_id": "s6",
                "run_id": s8_run_id,
                "exp_name": "synth",
                "source_s8_bout_id": "bout_contact",
                "source_s8_run_id": s8_run_id,
                "package_run_id": package_run_id,
                "s8_bouts_schema_version": "behavioral_arbitration_bouts_v1",
                "prepared_frame_idx_start": int(sync.loc[2, "prepared_frame_idx"]),
                "prepared_frame_idx_end": int(sync.loc[7, "prepared_frame_idx"]),
                "prepared_time_sec_start": float(sync.loc[2, "prepared_time_sec"]),
                "prepared_time_sec_end": float(sync.loc[7, "prepared_time_sec"]),
            },
            {
                "bout_id": f"{package_run_id}__bout_sbs",
                "schema_version": "results_package_bouts_v1",
                "behavior_name": "side_by_side",
                "channel": "side_by_side",
                "identity_grain": "symmetric_pair",
                "focal_id": None,
                "partner_id": None,
                "mouse_a_id": 0,
                "mouse_b_id": 1,
                "pair_id": "0|1",
                "start_sample_idx": n_dense - 2,
                "end_sample_idx": n_dense - 1,
                "start_time_sec": (n_dense - 2) / 120.0,
                "end_time_sec": (n_dense - 1) / 120.0,
                "bout_duration_frames": 2,
                "bout_duration_sec": 2 / 120.0,
                "mean_prob": 0.8,
                "max_prob": 0.8,
                "median_prob": 0.8,
                "frac_input_valid": 1.0,
                "label_reconstruction_mode": "test",
                "undefined_reason": "",
                "source_s7_run_id": "s7",
                "source_s6_run_id": "s6",
                "run_id": s8_run_id,
                "exp_name": "synth",
                "source_s8_bout_id": "bout_sbs",
                "source_s8_run_id": s8_run_id,
                "package_run_id": package_run_id,
                "s8_bouts_schema_version": "behavioral_arbitration_bouts_v1",
                "prepared_frame_idx_start": int(sync.loc[n_dense - 2, "prepared_frame_idx"]),
                "prepared_frame_idx_end": int(sync.loc[n_dense - 1, "prepared_frame_idx"]),
                "prepared_time_sec_start": float(
                    sync.loc[n_dense - 2, "prepared_time_sec"]
                ),
                "prepared_time_sec_end": float(
                    sync.loc[n_dense - 1, "prepared_time_sec"]
                ),
            },
        ]
    )

    s8_dir = session / "behavioral_arbitration" / s8_run_id
    s8_dir.mkdir(parents=True)
    s8_bouts_path = s8_dir / "behavior_bouts.parquet"
    # Minimal S8 bout table (schema stamp only for digest/schema checks).
    s8_bouts = bouts[
        [
            c
            for c in bouts.columns
            if c
            not in {
                "bout_id",
                "source_s8_bout_id",
                "source_s8_run_id",
                "package_run_id",
                "s8_bouts_schema_version",
                "prepared_frame_idx_start",
                "prepared_frame_idx_end",
                "prepared_time_sec_start",
                "prepared_time_sec_end",
            }
        ]
    ].copy()
    s8_bouts.insert(0, "bout_id", [f"bout_{i:08d}" for i in range(len(s8_bouts))])
    s8_bouts["schema_version"] = "behavioral_arbitration_bouts_v1"
    s8_bouts.to_parquet(s8_bouts_path, index=False)
    s8_bouts_sha = sha256_file(s8_bouts_path)

    states_path: Path | None = None
    states_sha: str | None = None
    if include_s8_states:
        states_path = s8_dir / "behavior_states.parquet"
        states = build_mini_behavior_states(sync)
        states.to_parquet(states_path, index=False)
        states_sha = sha256_file(states_path)

    # Optional S7 predictions for lazy evidence tests.
    s7_run_id = "behavioral_inference__synth"
    s7_dir = session / "behavioral_inference" / s7_run_id
    s7_dir.mkdir(parents=True, exist_ok=True)
    s7_preds = build_mini_s7_predictions(sync)
    s7_preds.to_parquet(s7_dir / "predictions.parquet", index=False)

    package_dir = session / RESULTS_PACKAGE_DIRNAME / package_run_id
    package_dir.mkdir(parents=True)
    bouts.to_parquet(package_dir / BEHAVIOR_BOUTS_FILENAME, index=False)
    sync.to_parquet(package_dir / SYNC_INDEX_FILENAME, index=False)
    pd.DataFrame(
        [
            {
                "schema_version": "results_package_ethogram_summary_v1",
                "summary_kind": "bout_counts",
                "channel": "exclusive",
                "behavior_name": "approach",
                "identity_key": "all",
                "identity_grain": "session",
                "metric_name": "n_bouts",
                "metric_value": float(len(bouts)),
                "denominator_name": None,
                "denominator_value": None,
                "unit": "count",
                "notes": None,
            }
        ]
    ).to_parquet(package_dir / ETHOGRAM_SUMMARY_FILENAME, index=False)

    bouts_sha = sha256_file(package_dir / BEHAVIOR_BOUTS_FILENAME)
    sync_sha = sha256_file(package_dir / SYNC_INDEX_FILENAME)
    video_sha = sha256_file(video_path) if include_video and video_path.is_file() else None
    recorded_video_sha = (
        "0" * 64 if video_sha_mismatch and video_sha is not None else video_sha
    )

    manifest = {
        "schema_version": "results_package_manifest_v1",
        "package_run_id": package_run_id,
        "package_artifacts": [
            {
                "role": "behavior_bouts",
                "sha256": bouts_sha,
                "verification_strength": "content_hash",
            },
            {
                "role": "sync_index",
                "sha256": sync_sha,
                "verification_strength": "content_hash",
            },
        ],
        "source_references": {
            "source_s8": {
                "run_id": s8_run_id,
                "run_dir": str(s8_dir),
            }
        },
        "row_counts": {"sync_index": n_dense, "behavior_bouts": int(len(bouts))},
    }
    (package_dir / PACKAGE_MANIFEST_FILENAME).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (package_dir / QC_FILENAME).write_text(
        json.dumps({"schema_version": "results_package_qc_v1", "blocking": []}, indent=2),
        encoding="utf-8",
    )
    (package_dir / SETTINGS_USED_FILENAME).write_text(
        yaml.safe_dump({"schema_version": "results_package_settings_v1"}),
        encoding="utf-8",
    )

    source_video: dict[str, Any]
    if include_video:
        source_video = {
            "path": str(video_path),
            "sha256": recorded_video_sha,
            "fingerprint": list(artifact_fingerprint(video_path)),
            "fingerprint_type": "sha256",
            "verification_strength": VERIFICATION_STRENGTH_CONTENT_HASH,
            "required": True,
        }
    else:
        source_video = {
            "path": str(video_path),
            "sha256": None,
            "fingerprint": None,
            "fingerprint_type": "sha256",
            "verification_strength": VERIFICATION_STRENGTH_CONTENT_HASH,
            "required": True,
        }

    run_meta = {
        "schema_version": "results_package_run_meta_v1",
        "run_id": package_run_id,
        "status": STATUS_COMPLETED,
        "session_root": str(session),
        "source_s8": {
            "run_id": s8_run_id,
            "run_dir": str(s8_dir),
            "behavior_states": str(states_path) if states_path else None,
            "behavior_bouts": str(s8_bouts_path),
            "behavior_states_sha256": states_sha,
            "behavior_bouts_sha256": s8_bouts_sha,
        },
        "source_s7": {
            "run_id": s7_run_id,
            "run_dir": str(s7_dir),
        },
        "source_s6": {"run_id": "combined_features__synth"},
        "source_video": source_video,
        "source_sync": {
            "path": str(session / "preprocess" / "prepared_sync.npz"),
            "sha256": "a" * 64,
        },
    }
    (package_dir / RUN_META_FILENAME).write_text(
        json.dumps(run_meta, indent=2), encoding="utf-8"
    )

    return {
        "session_root": session,
        "package_dir": package_dir,
        "package_run_id": package_run_id,
        "video_path": video_path if include_video else None,
        "s8_run_dir": s8_dir,
        "s8_run_id": s8_run_id,
        "sync": sync,
        "bouts": bouts,
        "n_dense": n_dense,
        "n_prepared": n_prepared,
    }
