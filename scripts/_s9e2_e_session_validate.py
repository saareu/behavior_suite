"""One-shot E-session validation for S9-E2 media/CSV export (not a unit test)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from results_packaging.media_csv_export.config import SelectionSpec
from results_packaging.media_csv_export.contracts import SELECTION_BOUT_ID_LIST
from results_packaging.media_csv_export.discovery import (
    reopen_export_job,
    summarize_export_project,
)
from results_packaging.media_csv_export.runner import (
    CsvExportRequest,
    MediaExportRequest,
    run_csv_export,
    run_media_export,
)
from results_packaging.media_csv_export.video import probe_video


def main() -> None:
    session = Path("test_cases/E_full_session_integration").resolve()
    package = session / "results_package" / "results_package__20261028T181714"
    assert package.is_dir(), package

    pkg_files = {p: p.read_bytes() for p in package.rglob("*") if p.is_file()}
    bouts = pd.read_parquet(package / "behavior_bouts.parquet")
    assert len(bouts) == 609

    csv_result = run_csv_export(
        CsvExportRequest(
            session_root=session,
            package_dir=package,
            timestamp="20261029T153000",
            allow_stale_source=True,
        )
    )
    print(
        "CSV status",
        csv_result.status,
        "rows",
        csv_result.n_rows,
        "warnings",
        len(csv_result.warnings),
    )
    assert csv_result.success and csv_result.n_rows == 609
    assert csv_result.csv_path is not None
    csv = pd.read_csv(csv_result.csv_path)
    assert len(csv) == 609
    assert set(csv["export_provenance"]) == {"automated"}
    assert set(csv["source_package_run_id"]) == {"results_package__20261028T181714"}
    assert csv["bout_id"].nunique() == 609
    assert csv["source_s8_bout_id"].nunique() == 609

    edge_start = bouts.nsmallest(1, "prepared_frame_idx_start").iloc[0]["bout_id"]
    edge_end = bouts.nlargest(1, "prepared_frame_idx_end").iloc[0]["bout_id"]
    short_dir = (
        bouts[(bouts.channel == "exclusive") & (bouts.bout_duration_frames <= 5)]
        .iloc[0]["bout_id"]
    )
    contact = (
        bouts[bouts.channel == "contact"]
        .nsmallest(1, "bout_duration_frames")
        .iloc[0]["bout_id"]
    )
    sbs = bouts[bouts.channel == "side_by_side"].iloc[0]["bout_id"]
    ids = list(
        dict.fromkeys(
            [
                str(edge_start),
                str(edge_end),
                str(short_dir),
                str(contact),
                str(sbs),
            ]
        )
    )
    print("Media bout ids:", ids)

    media = run_media_export(
        MediaExportRequest(
            session_root=session,
            package_dir=package,
            timestamp="20261029T153001",
            selection=SelectionSpec(mode=SELECTION_BOUT_ID_LIST, bout_ids=tuple(ids)),
            pre_context_sec=0.1,
            post_context_sec=0.1,
            allow_stale_source=True,
        )
    )
    print(
        "MEDIA status",
        media.status,
        "selected",
        media.n_selected,
        "ok",
        media.n_succeeded,
        "fail",
        media.n_failed,
    )
    assert media.status == "completed"
    assert media.n_succeeded == len(ids)
    assert media.clip_index_path is not None
    ci = pd.read_parquet(media.clip_index_path)
    report = []
    for _, row in ci.iterrows():
        clip = media.job_dir / "clips" / Path(str(row["clip_relpath"]))
        assert clip.is_file()
        probed = probe_video(clip)
        report.append(
            {
                "bout_id": row["bout_id"],
                "channel": row["channel"],
                "behavior_name": row["behavior_name"],
                "identity_grain": row["identity_grain"],
                "source_s8_bout_id": row["source_s8_bout_id"],
                "prep": [
                    int(row["prepared_frame_idx_start"]),
                    int(row["prepared_frame_idx_end"]),
                ],
                "clip": [
                    int(row["clip_frame_idx_start"]),
                    int(row["clip_frame_idx_end"]),
                ],
                "planned_frames": int(row["clip_frame_count"]),
                "encoded_frames": probed.n_frames,
                "source_fps": float(row["source_fps"]),
                "encoded_fps": probed.fps,
                "planned_dur": float(row["clip_duration_sec"]),
                "clip_exists": bool(row["clip_exists"]),
            }
        )
    print(json.dumps(report, indent=2))

    for path, payload in pkg_files.items():
        assert path.read_bytes() == payload, path

    summary = summarize_export_project(session)
    print(
        "export jobs",
        [(j.export_job_id, j.job_kind, j.status) for j in summary.jobs[:5]],
    )
    re_csv = reopen_export_job(csv_result.job_dir)
    re_media = reopen_export_job(media.job_dir)
    print(
        "reopen csv",
        re_csv["status"],
        "media",
        re_media["status"],
        "stale",
        list(re_media["stale_reasons"])[:1],
    )

    out = session / "results_exports" / "_s9e2_validation_report.json"
    out.write_text(
        json.dumps(
            {
                "package": str(package),
                "csv": {
                    "job_id": csv_result.export_job_id,
                    "n_rows": csv_result.n_rows,
                    "status": csv_result.status,
                    "warnings": list(csv_result.warnings),
                },
                "media": {
                    "job_id": media.export_job_id,
                    "n_selected": media.n_selected,
                    "n_succeeded": media.n_succeeded,
                    "status": media.status,
                    "clips": report,
                    "warnings": list(media.warnings),
                },
                "package_immutable": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print("Wrote", out)


if __name__ == "__main__":
    main()
