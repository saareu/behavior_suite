"""Thin S4 discovery helper used by S6 pair selection."""

from __future__ import annotations

import json
from pathlib import Path

from pose_finalization.contracts import (
    FINAL_POSE_FILENAME,
    POSE_FINALIZATION_DIRNAME,
    RUN_META_FILENAME,
)
from pose_finalization.run_discovery import find_latest_eligible_s4_run


def test_find_latest_eligible_s4_run_returns_newest(monkeypatch, tmp_path: Path) -> None:
    session = tmp_path / "session"
    older = session / POSE_FINALIZATION_DIRNAME / "pose_finalization__20260101T000000"
    newer = session / POSE_FINALIZATION_DIRNAME / "pose_finalization__20260102T000000"
    for run_dir, stamp in (
        (older, "2026-01-01T00:00:00"),
        (newer, "2026-01-02T00:00:00"),
    ):
        run_dir.mkdir(parents=True)
        (run_dir / FINAL_POSE_FILENAME).write_bytes(b"parquet-placeholder")
        (run_dir / RUN_META_FILENAME).write_text(
            json.dumps(
                {
                    "run_id": run_dir.name,
                    "status": "completed",
                    "created_at": stamp,
                    "completed_at": stamp,
                    "source_s3": {"run_dir": str(session / "s3"), "run_id": "s3"},
                }
            )
            + "\n",
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "pose_finalization.run_discovery.is_s4_run_eligible_for_downstream",
        lambda _path: True,
    )
    monkeypatch.setattr(
        "pose_finalization.run_discovery.list_accepted_s3_runs",
        lambda _root: [],
    )
    monkeypatch.setattr(
        "pose_finalization.run_discovery.explain_missing_accepted_s3",
        lambda _root: "none",
    )

    latest = find_latest_eligible_s4_run(session)
    assert latest is not None
    assert latest.run_dir == newer.resolve()
