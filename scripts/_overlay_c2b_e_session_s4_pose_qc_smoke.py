"""E-session C2-B S4 finalized-pose QC clip smoke (read-only science; no commit)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from overlay import (
    export_s4_pose_qc_clip,
    plan_s4_prepared_interval,
    resolve_overlay_s4_qc_source,
)
from project.file_digest import sha256_file

SESSION = Path(r"C:\behavior_suite\test_cases\E_full_session_integration")
S4_RUN = (
    SESSION
    / "pose_finalization"
    / "pose_finalization__20261025T173206"
)
OUT_ROOT = SESSION / "results_exports"
EVIDENCE_DIR = Path(r"C:\behavior_suite\tmp_overlay_c2b_e_session_s4_pose_qc")

# Prepared-frame window previously verified to contain observed (identity_corrected),
# temporal_resample, and kalman_gap_fill (plus missing). No original / linear /
# protected_manual_blank in this short window.
PREP_START = 971
PREP_END = 1000


def main() -> int:
    if not S4_RUN.is_dir():
        print(f"SKIP: S4 run missing: {S4_RUN}")
        return 2

    t_resolve = time.perf_counter()
    source = resolve_overlay_s4_qc_source(s4_run_dir=S4_RUN)
    resolve_open_sec = time.perf_counter() - t_resolve
    before = {
        "final_pose": sha256_file(source.final_pose_path),
        "prepared_video": sha256_file(source.prepared_video_path),
        "prepared_sync": sha256_file(source.prepared_sync_path),
    }

    plan = plan_s4_prepared_interval(
        prepared_frame_idx_start=PREP_START,
        prepared_frame_idx_end=PREP_END,
        source_fps=source.video_probe.fps,
        source_n_frames=source.video_probe.n_frames,
        draw_skeleton=True,
        notes=(
            "E-session C2-B lab clip: prep [971,1000] mixed provenance window",
        ),
    )
    print(
        "exporting S4 pose QC prepared inclusive "
        f"[{plan.prepared_interval.start}, {plan.prepared_interval.end}] "
        f"({plan.prepared_interval.frame_count} frames) @ {source.video_probe.fps} fps"
    )
    print(f"s4_run_id={source.s4_run_id}")
    print(f"final_pose_sha256={source.final_pose_sha256}")
    disclosure = source.skeleton_disclosure()
    print(
        f"skeleton_mode={disclosure['skeleton_mode']} "
        f"edges={disclosure['skeleton_edge_count']} source={source.skeleton_source}"
    )
    print(f"skeleton_disclosure={disclosure['skeleton_disclosure']}")

    perf_from_export: dict[str, float] = {}

    def _progress(event: dict[object, object]) -> None:
        stage = event.get("stage")
        print(f"progress: {stage} frames={event.get('frames_written')}")
        if stage == "published" and isinstance(event.get("performance_breakdown_sec"), dict):
            perf_from_export.update(
                {str(k): float(v) for k, v in event["performance_breakdown_sec"].items()}
            )

    result = export_s4_pose_qc_clip(
        source=source,
        plan=plan,
        output_root=OUT_ROOT,
        progress_callback=_progress,
    )

    after = {
        "final_pose": sha256_file(source.final_pose_path),
        "prepared_video": sha256_file(source.prepared_video_path),
        "prepared_sync": sha256_file(source.prepared_sync_path),
    }
    assert before == after, "Science artifacts mutated during S4 pose QC export"

    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(result.video_path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    assert len(frames) == result.frames_written

    # Align frame samples with last-in-plateau dense samples and record provenance.
    pose = pd.read_parquet(
        source.final_pose_path,
        columns=["sample_idx", "track", "node", "provenance", "provenance_name", "x", "y"],
    )
    sample_idxs = sorted(
        {
            0,
            len(frames) // 2,
            len(frames) - 1,
        }
    )
    evidence_rows: list[dict[object, object]] = []
    for i in sample_idxs:
        prep = plan.prepared_interval.start + i
        plateau = source.sync.plateau_sample_indices(prep)
        dense = int(plateau.max()) if plateau.size else None
        out_png = EVIDENCE_DIR / f"frame_{i:04d}_prep{prep}_dense{dense}.png"
        cv2.imwrite(str(out_png), frames[i])
        codes: list[int] = []
        names: list[str] = []
        if dense is not None:
            g = pose[pose.sample_idx == dense]
            codes = sorted({int(x) for x in g["provenance"].tolist()})
            names = sorted({str(x) for x in g["provenance_name"].tolist()})
        evidence_rows.append(
            {
                "frame_index_in_clip": i,
                "prepared_frame_idx": prep,
                "dense_sample_idx": dense,
                "png": str(out_png),
                "provenance_codes": codes,
                "provenance_names": names,
            }
        )
        print(
            f"frame clip={i} prep={prep} dense={dense} "
            f"codes={codes} names={names} -> {out_png.name}"
        )

    performance = {
        "source_resolve_open_sec": round(resolve_open_sec, 6),
        **(perf_from_export or dict(result.manifest.extra.get("performance_breakdown_sec") or {})),
    }
    print("performance_breakdown_sec:")
    for key in sorted(performance):
        print(f"  {key}: {performance[key]}")

    report = {
        "s4_run_id": source.s4_run_id,
        "s4_run_dir": str(source.s4_run_dir),
        "final_pose_sha256": source.final_pose_sha256,
        "prepared_video_sha256": source.prepared_video_sha256,
        "prepared_sync_sha256": source.prepared_sync_sha256,
        "job_dir": str(result.job_dir),
        "video_path": str(result.video_path),
        "manifest_path": str(result.manifest_path),
        "frames_written": result.frames_written,
        "output_fps": result.output_fps,
        "qc_status": result.qc_status,
        "prepared_interval": plan.prepared_interval.to_dict(),
        "skeleton": disclosure,
        "skeleton_source": source.skeleton_source,
        "skeleton_edge_count": len(source.skeleton_edges),
        "performance_breakdown_sec": performance,
        "manifest_observed_provenance_codes": result.manifest.extra.get(
            "observed_provenance_codes_in_clip"
        ),
        "manifest_observed_visual_classes": result.manifest.extra.get(
            "observed_visual_classes_in_clip"
        ),
        "sampled_frames": evidence_rows,
        "source_digests_unchanged": before == after,
        "categories_absent_from_clip_note": (
            "This short clip contains identity_corrected (observed), "
            "temporal_resample, kalman_gap_fill, and missing. It does not "
            "contain original, linear_gap_fill, or protected_manual_blank; "
            "those codes were not visually validated in this clip."
        ),
    }
    report_path = EVIDENCE_DIR / "E_SESSION_S4_POSE_QC_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"published: {result.job_dir}")
    print(f"evidence: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
