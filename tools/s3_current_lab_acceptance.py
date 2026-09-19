"""Optional local acceptance helper for Gate 3 golden baseline comparison.

This script is intentionally not committed with large artifacts. Point it at a
local pose.parquet / accepted tracks.parquet pair when available:

    python tools/s3_current_lab_acceptance.py \\
        --pose path/to/pose.parquet \\
        --accepted path/to/tracks.parquet \\
        --output-dir path/to/out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tracking_correction.legacy_backend import run_current_lab_backend


def _compare(working: pd.DataFrame, accepted: pd.DataFrame) -> dict[str, object]:
    left = working.copy()
    right = accepted.copy()
    if "frame_idx" in left.columns and "frame" not in right.columns and "frame_idx" in right.columns:
        right = right.rename(columns={"frame_idx": "frame"})
    keys = ["frame", "track", "node"]
    for frame in (left, right):
        for key in keys:
            if key not in frame.columns:
                raise SystemExit(f"Missing comparison key {key}")
    merged = left.merge(right, on=keys, how="outer", suffixes=("_s3", "_accepted"), indicator=True)
    both = merged[merged["_merge"] == "both"]
    coord_ok = bool(
        np.allclose(
            both["x_s3"].to_numpy(dtype=float),
            both["x_accepted"].to_numpy(dtype=float),
            equal_nan=True,
        )
        and np.allclose(
            both["y_s3"].to_numpy(dtype=float),
            both["y_accepted"].to_numpy(dtype=float),
            equal_nan=True,
        )
    )
    return {
        "rows_working": int(len(left)),
        "rows_accepted": int(len(right)),
        "rows_matched": int(len(both)),
        "rows_only_working": int((merged["_merge"] == "left_only").sum()),
        "rows_only_accepted": int((merged["_merge"] == "right_only").sum()),
        "coordinates_match": coord_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pose", type=Path, required=True)
    parser.add_argument("--accepted", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    # Gate 3 legacy pose uses `frame`; S3 adapter expects `frame_idx`.
    pose = pd.read_parquet(args.pose)
    if "frame_idx" not in pose.columns and "frame" in pose.columns:
        pose = pose.copy()
        pose["frame_idx"] = pose["frame"]
        adapted_pose = args.output_dir / "adapted_s2_style_pose.parquet"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        pose.to_parquet(adapted_pose, index=False)
        pose_path = adapted_pose
    else:
        pose_path = args.pose

    result = run_current_lab_backend(pose_parquet=pose_path, output_dir=args.output_dir)
    working = pd.read_parquet(result.working_tracked_pose_path)
    accepted = pd.read_parquet(args.accepted)
    summary = {
        "backend": result.backend_id,
        "profile": result.profile_id,
        "correction_summary": result.correction_summary,
        "comparison": _compare(working, accepted),
    }
    summary_path = args.output_dir / "acceptance_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
