"""Adapter for the current-lab minimal-intervention corrector.

Prepares S2 pose.parquet for the validated Pipeline 20 corrector, invokes it
without redesigning correction rules, and writes S3-owned working outputs.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import yaml

from tracking_correction.backends.current_lab.finalize_tracks import finalize_tracks_df_for_save
from tracking_correction.backends.current_lab.id_corrector_minimal import process_video_minimal
from tracking_correction.contracts import (
    CURRENT_BACKEND_ID,
    CURRENT_PROFILE_ID,
    MACHINE_CORRECTIONS_FILENAME,
    WORKING_TRACKED_POSE_FILENAME,
    BackendResult,
    TrackingCorrectionError,
)

_PROFILE_CONFIG_PATH = Path(__file__).resolve().parent / "backends" / "current_lab" / "profile.yaml"

# Timing / identity fields from S2 that should survive into the working pose when
# compatible with the historical finalizer schema (frame-level merge).
_S2_TIMING_COLUMNS = (
    "prepared_frame_idx",
    "raw_decode_frame_idx",
    "original_frame_idx",
    "source_video_frame_idx",
    "time_sec",
    "prepared_time_sec",
    "raw_pts_time_sec",
    "external_time_sec",
    "timestamp_source",
    "video_index",
    "fps_header",
    "video_fps",
    "raw_fps_effective",
)


def load_current_profile_config(path: Path | None = None) -> dict[str, Any]:
    """Load validated current-lab profile thresholds from profile.yaml."""

    config_path = Path(path) if path is not None else _PROFILE_CONFIG_PATH
    if not config_path.is_file():
        raise TrackingCorrectionError(f"Profile config does not exist: {config_path}")
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TrackingCorrectionError(f"Profile config is unreadable: {exc}") from exc
    if not isinstance(payload, dict):
        raise TrackingCorrectionError("Profile config must contain a mapping.")
    return dict(payload)


def adapt_s2_pose_to_legacy_input(pose: Any) -> Any:
    """Adapt S2 pose.parquet columns into the legacy corrector's expected shape.

    Explicit mappings:
    - ``frame_idx`` → ``frame`` (``frame_idx`` retained as an extra column)
    - ``score`` → ``point_score`` when point_score is absent
    - track labels such as ``track_0`` / ``0`` → integer track indices
    """

    if not isinstance(pose, pd.DataFrame):
        raise TrackingCorrectionError("S2 pose input must be a DataFrame.")
    frame = pose.copy()
    if "frame_idx" not in frame.columns:
        raise TrackingCorrectionError(
            "S2 pose.parquet is missing frame_idx required for legacy adaptation."
        )
    if "frame" in frame.columns:
        mismatched = frame["frame"].to_numpy() != frame["frame_idx"].to_numpy()
        if bool(np.any(mismatched)):
            raise TrackingCorrectionError(
                "S2 pose.parquet has conflicting frame and frame_idx values."
            )
    else:
        frame["frame"] = frame["frame_idx"]

    frame["frame"] = pd.to_numeric(frame["frame"], errors="coerce")
    if frame["frame"].isna().any():
        raise TrackingCorrectionError("S2 pose frame_idx/frame has non-numeric values.")
    frame["frame"] = frame["frame"].astype(np.int64)

    frame["track"] = [_normalize_track_label(value) for value in frame["track"].tolist()]
    if "score" in frame.columns and "point_score" not in frame.columns:
        frame["point_score"] = frame["score"]
    elif "point_score" in frame.columns and "score" not in frame.columns:
        frame["score"] = frame["point_score"]
    elif "score" not in frame.columns and "point_score" not in frame.columns:
        raise TrackingCorrectionError(
            "S2 pose.parquet is missing both score and point_score."
        )
    if "instance_score" not in frame.columns:
        frame["instance_score"] = np.nan
    return frame


def _normalize_track_label(value: object) -> int:
    if isinstance(value, bool) or value is None:
        raise TrackingCorrectionError(f"Unsupported track label: {value!r}")
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not np.isfinite(value) or not float(value).is_integer():
            raise TrackingCorrectionError(f"Unsupported track label: {value!r}")
        return int(value)
    text = str(value).strip()
    if text.lower().startswith("track_"):
        text = text.split("_", 1)[1]
    try:
        return int(text)
    except ValueError as exc:
        raise TrackingCorrectionError(f"Unsupported track label: {value!r}") from exc


def _preserve_raw_columns(frame: Any) -> Any:
    out = frame.copy()
    out["x_raw"] = out["x"].copy()
    out["y_raw"] = out["y"].copy()
    out["point_score_raw"] = out["point_score"].copy()
    out["instance_score_raw"] = out["instance_score"].copy()
    return out


def _maybe_transpose_tracks_for_headstage(
    frame: Any,
    *,
    headstage_node: str = "headstage",
    first_n_frames: int = 480,
) -> tuple[Any, set[int]]:
    """Preserve Pipeline 20 headstage-ownership transposition behavior."""

    swapped_frames: set[int] = set()
    work = frame
    first_frames = work[work["frame"] < first_n_frames]
    if first_frames.empty:
        return work, swapped_frames

    first_frames = first_frames.copy()
    first_frames["_track_int"] = pd.to_numeric(first_frames["track"], errors="coerce")
    hs_data = first_frames[first_frames["node"] == headstage_node]
    hs_valid = hs_data[["x", "y"]].notna().all(axis=1).sum()
    if hs_data.empty or hs_valid == 0 or not hs_data["_track_int"].notna().any():
        return work, swapped_frames

    valid_mask = hs_data[["x", "y"]].notna().all(axis=1) & hs_data["_track_int"].notna()
    hs_counts = hs_data.loc[valid_mask, "_track_int"].value_counts()
    totals = (
        first_frames[first_frames["_track_int"].notna()]
        .groupby("_track_int")["frame"]
        .nunique()
    )
    hs_ratio = {
        int(key): float(value)
        for key, value in (hs_counts / totals).fillna(0.0).items()
    }
    track_0_has_hs = hs_ratio.get(0, 0.0) > hs_ratio.get(1, 0.0)
    if track_0_has_hs:
        return work, swapped_frames

    work = work.copy()
    work["track"] = pd.to_numeric(work["track"], errors="coerce")
    if work["track"].isna().any():
        raise TrackingCorrectionError(
            "Track transposition failed: non-numeric track values."
        )
    work["track"] = work["track"].astype(int)
    swapped_frames.update(int(value) for value in work["frame"].unique().tolist())
    work.loc[work["track"] == 0, "track"] = -999
    work.loc[work["track"] == 1, "track"] = 0
    work.loc[work["track"] == -999, "track"] = 1
    return work, swapped_frames


def _timing_lookup(frame: Any) -> Any:
    present = [column for column in _S2_TIMING_COLUMNS if column in frame.columns]
    if not present:
        return pd.DataFrame(columns=["frame"])
    lookup = (
        frame[["frame", *present]]
        .drop_duplicates(subset=["frame"], keep="first")
        .sort_values("frame")
        .reset_index(drop=True)
    )
    return lookup


def _attach_s3_identity_and_timing(finalized: Any, timing_by_frame: Any) -> Any:
    out = finalized.copy()
    out["frame_idx"] = out["frame"].astype(np.int64)
    if timing_by_frame is not None and len(timing_by_frame.columns) > 1:
        merge_cols = [column for column in timing_by_frame.columns if column != "frame"]
        if merge_cols:
            out = out.merge(timing_by_frame, on="frame", how="left", validate="many_to_one")
    # Stable column order: historical finalizer columns, then frame_idx, then timing.
    historical = [
        "frame",
        "track",
        "node",
        "x",
        "y",
        "point_score",
        "instance_score",
        "x_raw",
        "y_raw",
        "point_score_raw",
        "instance_score_raw",
        "was_track_swapped",
        "was_node_blanked",
    ]
    extras = [column for column in out.columns if column not in historical and column != "frame_idx"]
    ordered = [*historical, "frame_idx", *extras]
    return out[[column for column in ordered if column in out.columns]]


def _correction_summary(corrections: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    type_counts = Counter(str(item.get("type", "unknown")) for item in corrections)
    return {
        "total_corrections": int(len(corrections)),
        "by_type": dict(sorted(type_counts.items())),
    }


def run_current_lab_backend(
    *,
    pose_parquet: Path,
    output_dir: Path,
    fps: float | None = None,
    profile_config: Mapping[str, Any] | None = None,
    show_progress: bool = False,
) -> BackendResult:
    """Run the validated current-lab corrector and write working S3 artifacts."""

    pose_path = Path(pose_parquet)
    out_dir = Path(output_dir)
    if not pose_path.is_file():
        raise TrackingCorrectionError(f"pose.parquet does not exist: {pose_path}")
    try:
        raw_pose = pd.read_parquet(pose_path)
    except Exception as exc:
        raise TrackingCorrectionError(f"Could not read pose.parquet: {exc}") from exc

    config = dict(profile_config) if profile_config is not None else load_current_profile_config()
    # Validated Pipeline 20 reads config['fps'] (profile baseline), not detected video FPS.
    if "fps" not in config or config["fps"] is None:
        if fps is None:
            raise TrackingCorrectionError("Profile config is missing fps.")
        config["fps"] = float(fps)
    config["show_progress"] = bool(show_progress)

    adapted = adapt_s2_pose_to_legacy_input(raw_pose)
    timing_by_frame = _timing_lookup(adapted)
    adapted = _preserve_raw_columns(adapted)
    adapted, transposed_frames = _maybe_transpose_tracks_for_headstage(
        adapted,
        headstage_node=str(config.get("headstage_node", "headstage")),
    )
    input_rows = int(len(adapted))

    try:
        corrected, corrections, blanked_nodes, corrector_swapped = process_video_minimal(
            adapted,
            config,
            show_progress=bool(config.get("show_progress", False)),
        )
    except Exception as exc:
        raise TrackingCorrectionError(f"Current-lab corrector failed: {exc}") from exc

    swapped_frames = set(transposed_frames)
    swapped_frames.update(corrector_swapped)

    for required in ("point_score", "instance_score", "x_raw", "y_raw"):
        if required not in corrected.columns:
            raise TrackingCorrectionError(
                f"Corrector dropped required provenance column: {required}"
            )

    try:
        finalized = finalize_tracks_df_for_save(
            corrected,
            swapped_frames=swapped_frames,
            blanked_nodes=blanked_nodes,
        )
    except Exception as exc:
        raise TrackingCorrectionError(f"Track finalization failed: {exc}") from exc

    working = _attach_s3_identity_and_timing(finalized, timing_by_frame)
    summary = _correction_summary(corrections)

    working_path = out_dir / WORKING_TRACKED_POSE_FILENAME
    corrections_path = out_dir / MACHINE_CORRECTIONS_FILENAME
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        working.to_parquet(working_path, engine="pyarrow", index=False)
        corrections_path.write_text(
            json.dumps(list(corrections), ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
            newline="\n",
        )
    except OSError as exc:
        raise TrackingCorrectionError(f"Could not write backend outputs: {exc}") from exc

    return BackendResult(
        backend_id=CURRENT_BACKEND_ID,
        profile_id=CURRENT_PROFILE_ID,
        working_tracked_pose_path=working_path,
        machine_corrections_path=corrections_path,
        correction_summary=summary,
        blanked_node_count=int(len(blanked_nodes)),
        swapped_frame_count=int(len(swapped_frames)),
        input_row_count=input_rows,
        output_row_count=int(len(working)),
        warnings=(),
    )
