"""Pose-quality QC summary for Subsystem 02 pose.parquet outputs."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

POSE_QC_SCHEMA_VERSION = "pose_qc_v1"
DEFAULT_EXPECTED_ANIMALS = 2
DEFAULT_SCORE_THRESHOLD = 0.2
DEFAULT_DUPLICATE_DISTANCE_PX = 15.0
DEFAULT_IMPLAUSIBLE_MAD_THRESHOLD = 8.0
DEFAULT_REVIEW_ONE_ANIMAL_FRACTION_THRESHOLD = 0.90
DEFAULT_REVIEW_MISSING_KEYPOINT_FRACTION_THRESHOLD = 0.90
DEFAULT_REVIEW_FRAME_MISSING_KEYPOINT_FRACTION_THRESHOLD = 0.90
MAX_REVIEW_INTERVALS_PER_WARNING = 10
GEOMETRY_NODE_PAIRS = (("nose", "tail_base"), ("neck", "tail_base"))
EXAMPLE_LIMIT = 25


class PoseQcError(RuntimeError):
    """Expected failure while computing pose-quality QC."""


@dataclass(frozen=True)
class PoseQcSummary:
    """Machine-readable summary of a pose-quality QC attempt."""

    status: str
    payload: Mapping[str, Any]
    error: str | None = None

    def to_metadata(self) -> dict[str, Any]:
        metadata = dict(self.payload)
        metadata.setdefault("status", self.status)
        metadata.setdefault("schema_version", POSE_QC_SCHEMA_VERSION)
        if self.error:
            metadata["error"] = self.error
        return metadata


def _fraction(count: int, total: int) -> float:
    return float(count / total) if total else 0.0


def _finite_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return numeric if math.isfinite(numeric) else None


def _example_frame_indices(values: object) -> list[int]:
    examples: list[int] = []
    for value in list(values)[:EXAMPLE_LIMIT]:
        numeric = _finite_float(value)
        if numeric is not None:
            examples.append(int(numeric))
    return examples


def _count_summary(count: int, total: int, frame_indices: object | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "count": int(count),
        "fraction": _fraction(int(count), int(total)),
    }
    if frame_indices is not None:
        summary["example_frame_indices"] = _example_frame_indices(frame_indices)
    return summary


def _prepared_frame_indices(frame_keys: object) -> list[int]:
    return [int(key[1]) for key in frame_keys]


def _require_columns(columns: object, required: set[str]) -> None:
    present = {str(column) for column in columns}
    missing = sorted(required - present)
    if missing:
        raise PoseQcError("pose.parquet is missing required QC columns: " + ", ".join(missing))


def _prepare_dataframe(pose_parquet_path: Path) -> Any:
    try:
        import pandas as pd
    except ImportError as exc:
        raise PoseQcError("pandas is required to compute pose-quality QC from pose.parquet.") from exc

    path = Path(pose_parquet_path)
    if not path.is_file():
        raise PoseQcError(f"pose.parquet does not exist: {path}")
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:
        raise PoseQcError(f"Could not read pose.parquet for pose-quality QC: {exc}") from exc

    _require_columns(
        frame.columns,
        {"frame_idx", "instance_index", "node", "x", "y"},
    )
    work = frame.copy()
    for column in ("frame_idx", "instance_index"):
        numeric = pd.to_numeric(work[column], errors="coerce")
        if numeric.isna().any():
            raise PoseQcError(f"pose.parquet column {column} has missing or non-numeric values.")
        work[column] = numeric.astype("int64")

    if "video_index" in work.columns:
        video_indices = pd.to_numeric(work["video_index"], errors="coerce")
        if video_indices.isna().any():
            raise PoseQcError(
                "pose.parquet video_index has missing or non-numeric values."
            )
        work["video_index"] = video_indices.astype("int64")
    else:
        work["video_index"] = 0

    if "prepared_frame_idx" in work.columns:
        prepared_indices = pd.to_numeric(work["prepared_frame_idx"], errors="coerce")
        if prepared_indices.isna().any():
            raise PoseQcError(
                "pose.parquet prepared_frame_idx has missing or non-numeric values."
            )
        work["prepared_frame_idx"] = prepared_indices.astype("int64")
        if not np.array_equal(
            work["prepared_frame_idx"].to_numpy(), work["frame_idx"].to_numpy()
        ):
            raise PoseQcError(
                "pose.parquet prepared_frame_idx does not match prediction frame_idx."
            )
    else:
        work["prepared_frame_idx"] = work["frame_idx"]

    if "sync_frame_count_used_for_sleap" in work.columns and not work.empty:
        sync_counts = pd.to_numeric(
            work["sync_frame_count_used_for_sleap"], errors="coerce"
        )
        if sync_counts.isna().any() or sync_counts.nunique() != 1:
            raise PoseQcError(
                "pose.parquet has an invalid or inconsistent S1 sync frame count."
            )
        sync_count = int(sync_counts.iloc[0])
        if sync_count < 1 or (work["frame_idx"] < 0).any() or (
            work["frame_idx"] >= sync_count
        ).any():
            raise PoseQcError(
                "pose.parquet frame indices fall outside the authoritative S1 frame range."
            )

    if {"prepared_frame_idx", "prepared_time_sec", "fps_header"}.issubset(
        work.columns
    ):
        prepared_times = pd.to_numeric(work["prepared_time_sec"], errors="coerce")
        fps_values = pd.to_numeric(work["fps_header"], errors="coerce")
        available = prepared_times.notna() & fps_values.notna()
        if available.any():
            expected_times = work.loc[available, "prepared_frame_idx"] / fps_values[available]
            if not np.allclose(
                prepared_times[available].to_numpy(),
                expected_times.to_numpy(),
                rtol=0,
                atol=1e-12,
            ):
                raise PoseQcError(
                    "pose.parquet prepared-frame timing does not match the S1 mapping."
                )

    work["node"] = work["node"].astype("string").fillna("unknown")
    work["x"] = pd.to_numeric(work["x"], errors="coerce")
    work["y"] = pd.to_numeric(work["y"], errors="coerce")
    work["_valid_xy"] = np.isfinite(work["x"].to_numpy()) & np.isfinite(work["y"].to_numpy())

    if "score" in work.columns:
        work["score"] = pd.to_numeric(work["score"], errors="coerce")
    else:
        work["score"] = pd.NA
    work["_low_score"] = work["score"].notna() & (work["score"] < 0)
    return work


def _frame_and_instance_summary(
    work: Any,
    *,
    expected_animals: int,
    score_threshold: float,
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    frame_cols = ["video_index", "prepared_frame_idx"]
    group_cols = [*frame_cols, "instance_index"]

    valid_instances = work.groupby(group_cols, dropna=False)["_valid_xy"].any()
    valid_instances_per_frame = valid_instances.groupby(level=[0, 1]).sum().astype(int)
    frame_total = int(len(valid_instances_per_frame))
    zero_frames = valid_instances_per_frame[valid_instances_per_frame == 0].index
    one_animal_frames = valid_instances_per_frame[valid_instances_per_frame == 1].index
    fewer_frames = valid_instances_per_frame[valid_instances_per_frame < expected_animals].index
    expected_frames = valid_instances_per_frame[valid_instances_per_frame == expected_animals].index
    extra_frames = valid_instances_per_frame[valid_instances_per_frame > expected_animals].index

    frames = {
        "total": frame_total,
        "expected_animals": int(expected_animals),
        "zero_animals": _count_summary(
            len(zero_frames), frame_total, _prepared_frame_indices(zero_frames)
        ),
        "exactly_one_animal": _count_summary(
            len(one_animal_frames),
            frame_total,
            _prepared_frame_indices(one_animal_frames),
        ),
        "fewer_than_expected": _count_summary(
            len(fewer_frames),
            frame_total,
            _prepared_frame_indices(fewer_frames),
        ),
        "expected_count": _count_summary(len(expected_frames), frame_total),
        "more_than_expected": _count_summary(
            len(extra_frames),
            frame_total,
            _prepared_frame_indices(extra_frames),
        ),
    }

    instance_score_summary: dict[str, Any] = {"status": "unavailable", "reason": "missing column"}
    if "instance_score" in work.columns:
        scores = work.copy()
        scores["instance_score"] = work["instance_score"].map(_finite_float)
        instance_scores = scores.groupby(group_cols, dropna=False)["instance_score"].first()
        scored = instance_scores.dropna()
        low = scored[scored < score_threshold]
        instance_score_summary = {
            "status": "computed",
            "scored_instances": int(len(scored)),
            "low_confidence_instances": int(len(low)),
            "low_confidence_fraction": _fraction(int(len(low)), int(len(scored))),
        }

    instances = {
        "total": int(len(valid_instances)),
        "valid": int(valid_instances.sum()),
        "invalid": int((~valid_instances).sum()),
        "instance_score": instance_score_summary,
    }
    return frames, instances, valid_instances_per_frame


def _keypoint_summary(work: Any, score_threshold: float) -> dict[str, Any]:
    work = work.copy()
    work["_low_score"] = work["score"].notna() & (work["score"] < score_threshold)
    total_rows = int(len(work))
    missing = int((~work["_valid_xy"]).sum())
    low_score = int(work["_low_score"].sum())

    by_node = []
    for node, group in work.groupby("node", dropna=False, sort=True):
        node_total = int(len(group))
        node_missing = int((~group["_valid_xy"]).sum())
        node_low = int(group["_low_score"].sum())
        by_node.append(
            {
                "node": str(node),
                "rows": node_total,
                "missing_coordinates": node_missing,
                "missing_fraction": _fraction(node_missing, node_total),
                "low_confidence": node_low,
                "low_confidence_fraction": _fraction(node_low, node_total),
            }
        )

    return {
        "rows": total_rows,
        "nodes": int(work["node"].nunique()),
        "missing_coordinates": _count_summary(missing, total_rows),
        "low_confidence": _count_summary(low_score, total_rows),
        "by_node": by_node,
    }


def _skeleton_summary(work: Any) -> dict[str, Any]:
    group_cols = ["frame_idx", "instance_index"]
    expected_node_count = int(work["node"].nunique())
    valid_keypoints = work.groupby(group_cols, dropna=False)["_valid_xy"].sum().astype(int)
    partial = valid_keypoints[valid_keypoints < expected_node_count]
    return {
        "expected_keypoints_per_instance": expected_node_count,
        "total_instances": int(len(valid_keypoints)),
        "partial_skeleton_instances": int(len(partial)),
        "partial_skeleton_fraction": _fraction(int(len(partial)), int(len(valid_keypoints))),
    }


def _duplicate_candidate_risk(
    work: Any,
    *,
    expected_animals: int,
    duplicate_distance_px: float,
) -> dict[str, Any]:
    group_cols = ["frame_idx", "instance_index"]
    valid_instances = work.groupby(group_cols, dropna=False)["_valid_xy"].any()
    frame_indices = sorted(int(value) for value in work["frame_idx"].dropna().unique())
    valid_instances_per_frame = (
        valid_instances.groupby(level=0).sum().reindex(frame_indices, fill_value=0).astype(int)
    )
    extra_frames = set(
        int(index) for index in valid_instances_per_frame[valid_instances_per_frame > expected_animals].index
    )

    examples = []
    flagged_frames: set[int] = set()
    valid_rows = work[work["_valid_xy"]]
    for frame_idx in sorted(extra_frames):
        frame_rows = valid_rows[valid_rows["frame_idx"] == frame_idx]
        for node, node_rows in frame_rows.groupby("node", dropna=False, sort=True):
            records = list(
                node_rows[["instance_index", "x", "y"]].itertuples(index=False, name=None)
            )
            for left_index, left in enumerate(records):
                for right in records[left_index + 1 :]:
                    if int(left[0]) == int(right[0]):
                        continue
                    distance = math.hypot(float(left[1]) - float(right[1]), float(left[2]) - float(right[2]))
                    if distance < duplicate_distance_px:
                        flagged_frames.add(frame_idx)
                        if len(examples) < EXAMPLE_LIMIT:
                            examples.append(
                                {
                                    "frame_idx": frame_idx,
                                    "node": str(node),
                                    "instance_indices": [int(left[0]), int(right[0])],
                                    "distance_px": distance,
                                }
                            )

    return {
        "status": "computed",
        "threshold_px": float(duplicate_distance_px),
        "frames_with_extra_animals": int(len(extra_frames)),
        "frames_flagged": int(len(flagged_frames)),
        "fraction": _fraction(int(len(flagged_frames)), len(frame_indices)),
        "example_frame_indices": sorted(flagged_frames)[:EXAMPLE_LIMIT],
        "examples": examples,
    }


def _geometry_distances_for_pair(work: Any, node_a: str, node_b: str) -> list[dict[str, Any]]:
    pair_rows = work[work["node"].isin([node_a, node_b]) & work["_valid_xy"]]
    if pair_rows.empty:
        return []

    distances = []
    for (frame_idx, instance_index), group in pair_rows.groupby(
        ["frame_idx", "instance_index"], dropna=False
    ):
        points = {
            str(row.node): (float(row.x), float(row.y))
            for row in group[["node", "x", "y"]].itertuples(index=False)
        }
        if node_a not in points or node_b not in points:
            continue
        ax, ay = points[node_a]
        bx, by = points[node_b]
        distances.append(
            {
                "frame_idx": int(frame_idx),
                "instance_index": int(instance_index),
                "distance_px": math.hypot(ax - bx, ay - by),
            }
        )
    return distances


def _implausible_geometry(work: Any, threshold: float) -> dict[str, Any]:
    selected_pair = None
    distances: list[dict[str, Any]] = []
    for pair in GEOMETRY_NODE_PAIRS:
        distances = _geometry_distances_for_pair(work, pair[0], pair[1])
        if len(distances) >= 3:
            selected_pair = pair
            break

    if selected_pair is None:
        return {
            "status": "unavailable",
            "reason": "common node pair absent or too few distances",
            "method": "conservative_body_length_outlier",
            "frames_flagged": 0,
            "instances_flagged": 0,
            "example_frame_indices": [],
        }

    values = np.asarray([item["distance_px"] for item in distances], dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    if mad > 0:
        robust_z = np.abs(values - median) / (1.4826 * mad)
        flagged_mask = robust_z > threshold
        upper_threshold = None
    elif median > 0:
        upper_threshold = median * 5.0
        flagged_mask = values > upper_threshold
    else:
        upper_threshold = None
        flagged_mask = np.zeros(values.shape, dtype=bool)

    flagged = [item for item, is_flagged in zip(distances, flagged_mask, strict=True) if is_flagged]
    flagged_frames = sorted({int(item["frame_idx"]) for item in flagged})
    return {
        "status": "computed",
        "method": "conservative_body_length_outlier",
        "node_pair": list(selected_pair),
        "median_distance_px": median,
        "mad_distance_px": mad,
        "mad_threshold": float(threshold),
        "fallback_upper_threshold_px": upper_threshold,
        "frames_flagged": int(len(flagged_frames)),
        "instances_flagged": int(len(flagged)),
        "example_frame_indices": flagged_frames[:EXAMPLE_LIMIT],
        "examples": flagged[:EXAMPLE_LIMIT],
    }


def _timestamp_sources(work: Any) -> list[str]:
    if "timestamp_source" not in work.columns:
        return []
    values = [str(value) for value in work["timestamp_source"].dropna().unique()]
    return sorted(values)


def _validate_fraction_threshold(name: str, value: float) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise PoseQcError(f"{name} must be between 0 and 1 inclusive.")


def _frame_times(work: Any) -> dict[tuple[int, int], float]:
    for column in ("time_sec", "prepared_time_sec"):
        if column not in work.columns:
            continue
        times: dict[tuple[int, int], float] = {}
        for frame_key, values in work.groupby(
            ["video_index", "prepared_frame_idx"], sort=True
        )[column]:
            for value in values:
                numeric = _finite_float(value)
                if numeric is not None:
                    times[(int(frame_key[0]), int(frame_key[1]))] = numeric
                    break
        if times:
            return times
    return {}


def _contiguous_intervals(
    frame_keys: object,
    *,
    warning_type: str,
    frame_times: Mapping[tuple[int, int], float],
    frame_values: Mapping[tuple[int, int], float] | None = None,
) -> list[dict[str, Any]]:
    keys = sorted({(int(key[0]), int(key[1])) for key in frame_keys})
    groups: list[list[tuple[int, int]]] = []
    for frame_key in keys:
        if (
            not groups
            or frame_key[0] != groups[-1][-1][0]
            or frame_key[1] != groups[-1][-1][1] + 1
        ):
            groups.append([frame_key])
        else:
            groups[-1].append(frame_key)

    intervals: list[dict[str, Any]] = []
    for group in groups:
        video_index = group[0][0]
        start_frame = group[0][1]
        end_frame = group[-1][1]
        start_time = frame_times.get(group[0])
        end_time = frame_times.get(group[-1])
        interval: dict[str, Any] = {
            "warning_type": warning_type,
            "video_index": video_index,
            "start_frame": start_frame,
            "end_frame": end_frame,
            "frame_count": len(group),
            "start_time_sec": start_time,
            "end_time_sec": end_time,
            "time_span_sec": (
                max(0.0, end_time - start_time)
                if start_time is not None and end_time is not None
                else None
            ),
        }
        if frame_values is not None:
            values = [frame_values[frame_key] for frame_key in group]
            interval["mean_missing_keypoint_fraction"] = float(np.mean(values))
            interval["max_missing_keypoint_fraction"] = float(max(values))
        intervals.append(interval)
    return sorted(
        intervals,
        key=lambda interval: (
            -int(interval["frame_count"]),
            int(interval["video_index"]),
            int(interval["start_frame"]),
        ),
    )[:MAX_REVIEW_INTERVALS_PER_WARNING]


def _technical_outcome(
    work: Any,
    *,
    frames: Mapping[str, Any],
    keypoints: Mapping[str, Any],
    valid_instances_per_frame: Any,
    expected_animals: int,
    review_one_animal_fraction_threshold: float,
    review_missing_keypoint_fraction_threshold: float,
    review_frame_missing_keypoint_fraction_threshold: float,
) -> dict[str, Any]:
    represented_finite_frames = int((valid_instances_per_frame > 0).sum())
    failure_reasons = (
        ["zero_represented_frames_with_finite_pose"]
        if represented_finite_frames == 0
        else []
    )
    reasons: list[str] = []
    warning_summaries: list[dict[str, Any]] = []
    flagged_intervals: dict[str, list[dict[str, Any]]] = {}
    frame_times = _frame_times(work)

    one_animal_fraction = float(frames["exactly_one_animal"]["fraction"])
    if (
        expected_animals == 2
        and one_animal_fraction >= review_one_animal_fraction_threshold
    ):
        warning_type = "one_detected_animal"
        reasons.append(warning_type)
        one_animal_frames = valid_instances_per_frame[
            valid_instances_per_frame == 1
        ].index
        intervals = _contiguous_intervals(
            one_animal_frames,
            warning_type=warning_type,
            frame_times=frame_times,
        )
        flagged_intervals[warning_type] = intervals
        warning_summaries.append(
            {
                "warning_type": warning_type,
                "fraction": one_animal_fraction,
                "threshold": float(review_one_animal_fraction_threshold),
                "interval_count_retained": len(intervals),
            }
        )

    missing_fraction = float(keypoints["missing_coordinates"]["fraction"])
    if missing_fraction >= review_missing_keypoint_fraction_threshold:
        warning_type = "high_missing_keypoint_fraction"
        reasons.append(warning_type)
        frame_missing = work.groupby(
            ["video_index", "prepared_frame_idx"], sort=True
        )["_valid_xy"].apply(lambda values: float((~values).mean()))
        flagged = frame_missing[
            frame_missing >= review_frame_missing_keypoint_fraction_threshold
        ]
        frame_values = {
            (int(index[0]), int(index[1])): float(value)
            for index, value in flagged.items()
        }
        intervals = _contiguous_intervals(
            flagged.index,
            warning_type=warning_type,
            frame_times=frame_times,
            frame_values=frame_values,
        )
        flagged_intervals[warning_type] = intervals
        warning_summaries.append(
            {
                "warning_type": warning_type,
                "fraction": missing_fraction,
                "threshold": float(review_missing_keypoint_fraction_threshold),
                "per_frame_threshold": float(
                    review_frame_missing_keypoint_fraction_threshold
                ),
                "interval_count_retained": len(intervals),
            }
        )

    if failure_reasons:
        outcome = "failed"
    elif reasons:
        outcome = "review_recommended"
    else:
        outcome = "pass"
    return {
        "outcome": outcome,
        "represented_frames_with_finite_pose": represented_finite_frames,
        "failure_reasons": failure_reasons,
        "review_recommendation_reasons": reasons,
        "review_warning_count": len(reasons),
        "review_warnings": warning_summaries,
        "flagged_intervals": flagged_intervals,
        "review_thresholds": {
            "one_animal_fraction": float(review_one_animal_fraction_threshold),
            "missing_keypoint_fraction": float(
                review_missing_keypoint_fraction_threshold
            ),
            "frame_missing_keypoint_fraction": float(
                review_frame_missing_keypoint_fraction_threshold
            ),
            "max_intervals_per_warning_type": MAX_REVIEW_INTERVALS_PER_WARNING,
        },
    }


def _warnings(
    *,
    frames: Mapping[str, Any],
    keypoints: Mapping[str, Any],
    skeletons: Mapping[str, Any],
    duplicate_risk: Mapping[str, Any],
    geometry: Mapping[str, Any],
) -> list[str]:
    warnings = []
    if frames["total"] == 0:
        warnings.append("pose.parquet contains no represented frames")
    if frames["zero_animals"]["count"]:
        warnings.append("some represented frames contain zero valid animals")
    if frames["fewer_than_expected"]["count"]:
        warnings.append("some represented frames contain fewer animals than expected")
    if frames["more_than_expected"]["count"]:
        warnings.append("some represented frames contain more animals than expected")
    if keypoints["missing_coordinates"]["count"]:
        warnings.append("missing keypoint coordinates are present")
    if keypoints["low_confidence"]["count"]:
        warnings.append("low-confidence keypoints are present")
    if skeletons["partial_skeleton_instances"]:
        warnings.append("partial skeletons are present")
    if duplicate_risk["frames_flagged"]:
        warnings.append("duplicate candidate risk was flagged")
    if geometry.get("frames_flagged"):
        warnings.append("implausible geometry was flagged")
    return warnings


def compute_pose_qc_from_parquet(
    *,
    pose_parquet_path: Path,
    expected_animals: int = DEFAULT_EXPECTED_ANIMALS,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    duplicate_distance_px: float = DEFAULT_DUPLICATE_DISTANCE_PX,
    implausible_mad_threshold: float = DEFAULT_IMPLAUSIBLE_MAD_THRESHOLD,
    review_one_animal_fraction_threshold: float = (
        DEFAULT_REVIEW_ONE_ANIMAL_FRACTION_THRESHOLD
    ),
    review_missing_keypoint_fraction_threshold: float = (
        DEFAULT_REVIEW_MISSING_KEYPOINT_FRACTION_THRESHOLD
    ),
    review_frame_missing_keypoint_fraction_threshold: float = (
        DEFAULT_REVIEW_FRAME_MISSING_KEYPOINT_FRACTION_THRESHOLD
    ),
) -> PoseQcSummary:
    """Compute ``pose_qc_v1`` from a long/tidy pose.parquet file."""

    if expected_animals < 1:
        raise PoseQcError("expected_animals must be at least 1.")
    if score_threshold < 0:
        raise PoseQcError("score_threshold must be non-negative.")
    if duplicate_distance_px <= 0:
        raise PoseQcError("duplicate_distance_px must be positive.")
    _validate_fraction_threshold(
        "review_one_animal_fraction_threshold",
        review_one_animal_fraction_threshold,
    )
    _validate_fraction_threshold(
        "review_missing_keypoint_fraction_threshold",
        review_missing_keypoint_fraction_threshold,
    )
    _validate_fraction_threshold(
        "review_frame_missing_keypoint_fraction_threshold",
        review_frame_missing_keypoint_fraction_threshold,
    )

    work = _prepare_dataframe(Path(pose_parquet_path))
    frames, instances, valid_instances_per_frame = _frame_and_instance_summary(
        work,
        expected_animals=expected_animals,
        score_threshold=score_threshold,
    )
    keypoints = _keypoint_summary(work, score_threshold)
    skeletons = _skeleton_summary(work)
    duplicate_risk = _duplicate_candidate_risk(
        work,
        expected_animals=expected_animals,
        duplicate_distance_px=duplicate_distance_px,
    )
    geometry = _implausible_geometry(work, implausible_mad_threshold)
    warnings = _warnings(
        frames=frames,
        keypoints=keypoints,
        skeletons=skeletons,
        duplicate_risk=duplicate_risk,
        geometry=geometry,
    )
    outcome = _technical_outcome(
        work,
        frames=frames,
        keypoints=keypoints,
        valid_instances_per_frame=valid_instances_per_frame,
        expected_animals=expected_animals,
        review_one_animal_fraction_threshold=(
            review_one_animal_fraction_threshold
        ),
        review_missing_keypoint_fraction_threshold=(
            review_missing_keypoint_fraction_threshold
        ),
        review_frame_missing_keypoint_fraction_threshold=(
            review_frame_missing_keypoint_fraction_threshold
        ),
    )

    payload = {
        "status": "computed",
        "schema_version": POSE_QC_SCHEMA_VERSION,
        **outcome,
        "expected_animals": int(expected_animals),
        "score_threshold": float(score_threshold),
        "frames": frames,
        "instances": instances,
        "keypoints": keypoints,
        "skeletons": skeletons,
        "duplicate_candidate_risk": duplicate_risk,
        "implausible_geometry": geometry,
        "timestamp_sources": _timestamp_sources(work),
        "warnings": warnings,
    }
    return PoseQcSummary(status="computed", payload=payload)
