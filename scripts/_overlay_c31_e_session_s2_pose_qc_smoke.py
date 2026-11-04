"""C3-1 lab smoke: short S2 pose QC clip from the E-session eligible S2 run."""

from __future__ import annotations

import json
from pathlib import Path

from overlay import (
    export_s2_pose_qc_clip,
    plan_s2_prepared_interval,
    resolve_overlay_s2_qc_source,
)
from project.file_digest import sha256_file

SESSION = Path("test_cases/E_full_session_integration")
S2_RUN = SESSION / "pose_inference" / "bottomup__20260709T195039"
OUT_ROOT = SESSION / "results_exports"
EVIDENCE = Path("tmp_overlay_c31_e_session_s2_pose_qc")


def main() -> None:
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    before_pose = sha256_file(S2_RUN / "pose.parquet")
    source = resolve_overlay_s2_qc_source(s2_run_dir=S2_RUN)
    plan = plan_s2_prepared_interval(
        prepared_frame_idx_start=971,
        prepared_frame_idx_end=1000,
        source_fps=float(source.video_probe.fps),
        source_n_frames=int(source.video_probe.n_frames),
        draw_skeleton=True,
        show_confidence=True,
        include_frame_counter=True,
    )
    result = export_s2_pose_qc_clip(
        source=source,
        plan=plan,
        output_root=OUT_ROOT,
    )
    after_pose = sha256_file(S2_RUN / "pose.parquet")
    report = {
        "s2_run_id": source.s2_run_id,
        "job_dir": str(result.job_dir),
        "video_path": str(result.video_path),
        "manifest_path": str(result.manifest_path),
        "frames_written": result.frames_written,
        "output_fps": result.output_fps,
        "qc_status": result.qc_status,
        "skeleton": source.skeleton_disclosure(),
        "pose_digest_before": before_pose,
        "pose_digest_after": after_pose,
        "pose_digest_unchanged": before_pose == after_pose,
        "existing_overlay_mp4_present": (S2_RUN / "overlay.mp4").is_file(),
        "notes": list(result.notes),
        "manifest_extra_keys": sorted(result.manifest.extra.keys()),
        "confidence": result.manifest.extra.get("confidence"),
        "identity_semantics": result.manifest.extra.get("identity_semantics"),
        "output": result.manifest.extra.get("output"),
    }
    report_path = EVIDENCE / "C31_S2_POSE_QC_SMOKE_REPORT.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
