"""Headless in-memory S6 feature calculation (Sprint 2).

Combines Sprint 1 joint eligibility + pose adapter with vendored frame-feature
and Pipeline 50 post-processing. Does not write Parquet run artifacts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from combined_features.contracts import CombinedFeaturesError
from combined_features.frame_features import FrameFeatureResult, build_frame_features
from combined_features.pipeline50_post import apply_pipeline50_post
from combined_features.pose_adapter import AdaptedPose, adapt_final_pose_to_frame_feature_input
from combined_features.upstream import JointEligiblePair, validate_joint_s4_s5_pair


@dataclass(frozen=True, slots=True)
class CombinedFeaturesComputation:
    """In-memory result of the Sprint 2 calculation chain."""

    features: pd.DataFrame
    adapted_pose: AdaptedPose
    frame_feature_result: FrameFeatureResult
    warnings: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_rows(self) -> int:
        return int(len(self.features))

    @property
    def n_samples(self) -> int:
        if "dense_frame_idx" not in self.features.columns or self.features.empty:
            return 0
        return int(self.features["dense_frame_idx"].nunique())


def compute_combined_features_from_adapted(
    adapted: AdaptedPose,
    blob_features: pd.DataFrame,
    *,
    exp_name: str = "session",
    iteration_name: str = "suite",
    smooth_win: int = 9,
    diff_dt: int = 1,
) -> CombinedFeaturesComputation:
    """Run FF + Pipeline 50 post on an already-adapted pose + S5 blob table."""

    ff = build_frame_features(
        adapted,
        exp_name=exp_name,
        iteration_name=iteration_name,
        smooth_win=smooth_win,
        diff_dt=diff_dt,
    )
    directional = ff.features_df.copy()
    if directional.empty:
        raise CombinedFeaturesError("Frame-feature calculation produced an empty table.")

    # Normalize timing aliases for Suite consumers while keeping legacy t_120.
    if "dense_frame_idx" in directional.columns:
        directional["sample_idx"] = directional["dense_frame_idx"].astype(np.int64)
    if "t_120" in directional.columns and "time_sec" not in directional.columns:
        directional["time_sec"] = directional["t_120"].astype(np.float64)

    features = apply_pipeline50_post(
        directional,
        blob_features,
        adapted.implantation,
    )
    _validate_directional_keys(features)

    warnings: list[str] = []
    reason = features.attrs.get("implantation_reason")
    status = features.attrs.get("implantation_status")
    if reason:
        warnings.append(str(reason))
    if status and status != "resolved":
        warnings.append(
            f"implantation_status={status}; headstage-dependent robust fallbacks "
            "were not applied via invented track 0."
        )

    metadata = {
        "exp_name": exp_name,
        "iteration_name": iteration_name,
        "n_tracks": adapted.n_tracks,
        "n_nodes": adapted.n_nodes,
        "n_samples": adapted.n_samples,
        "tracks": list(adapted.tracks),
        "nodes": list(adapted.nodes),
        "implantation": adapted.implantation.to_metadata(),
        "adapter_policy_version": adapted.adapter_policy_version,
        "n_directional_rows": int(len(features)),
        "expected_directional_rows": int(
            adapted.n_samples * adapted.n_tracks * max(adapted.n_tracks - 1, 0)
        ),
    }
    return CombinedFeaturesComputation(
        features=features,
        adapted_pose=adapted,
        frame_feature_result=ff,
        warnings=tuple(warnings),
        metadata=metadata,
    )


def compute_combined_features(
    *,
    session_root: Path | str,
    s4_run_dir: Path | str,
    s5_run_dir: Path | str,
    exp_name: str | None = None,
    iteration_name: str = "suite",
    smooth_win: int = 9,
    diff_dt: int = 1,
) -> CombinedFeaturesComputation:
    """Validate a joint S4/S5 pair and compute the complete directional feature table."""

    pair = validate_joint_s4_s5_pair(
        session_root=session_root,
        s4_run_dir=s4_run_dir,
        s5_run_dir=s5_run_dir,
    )
    return compute_combined_features_from_pair(
        pair,
        exp_name=exp_name,
        iteration_name=iteration_name,
        smooth_win=smooth_win,
        diff_dt=diff_dt,
    )


def compute_combined_features_from_pair(
    pair: JointEligiblePair,
    *,
    exp_name: str | None = None,
    iteration_name: str = "suite",
    smooth_win: int = 9,
    diff_dt: int = 1,
) -> CombinedFeaturesComputation:
    """Compute features from an already-validated ``JointEligiblePair``."""

    adapted = adapt_final_pose_to_frame_feature_input(
        pair.final_pose,
        orig_frame_idx=pair.blob_features["orig_frame_idx"].to_numpy(dtype=np.int64),
        run_meta=pair.s4_meta,
    )
    label = exp_name if exp_name is not None else Path(pair.session_root).name
    result = compute_combined_features_from_adapted(
        adapted,
        pair.blob_features,
        exp_name=label,
        iteration_name=iteration_name,
        smooth_win=smooth_win,
        diff_dt=diff_dt,
    )
    metadata = dict(result.metadata)
    metadata.update(
        {
            "session_root": str(pair.session_root),
            "s4_run_id": pair.s4_run_id,
            "s5_run_id": pair.s5_run_id,
            "prepared_sync": str(pair.prepared_sync),
            "prepared_sync_sha256": pair.prepared_sync_sha256,
            "alignment": pair.alignment.to_metadata(),
        }
    )
    return CombinedFeaturesComputation(
        features=result.features,
        adapted_pose=result.adapted_pose,
        frame_feature_result=result.frame_feature_result,
        warnings=result.warnings,
        metadata=metadata,
    )


def _validate_directional_keys(features: pd.DataFrame) -> None:
    required = ("focal_id", "partner_id", "dense_frame_idx")
    missing = [name for name in required if name not in features.columns]
    if missing:
        raise CombinedFeaturesError(
            "Directional feature table missing required identity columns: "
            + ", ".join(missing)
        )
    if features.empty:
        raise CombinedFeaturesError("Directional feature table is empty.")
    key = features.duplicated(subset=list(required), keep=False)
    if bool(key.any()):
        raise CombinedFeaturesError(
            "Directional feature table has duplicate (focal_id, partner_id, dense_frame_idx) keys."
        )
    if (features["focal_id"] == features["partner_id"]).any():
        raise CombinedFeaturesError("Directional feature table contains focal_id == partner_id rows.")
