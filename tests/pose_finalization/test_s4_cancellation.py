"""S4 cooperative cancellation, publication gate, and eligibility tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from pose_finalization import (
    FAILED_OR_INCOMPLETE,
    FINAL_POSE_FILENAME,
    STATUS_CANCELED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    PoseFinalizationCancelledError,
    PoseFinalizationError,
    PoseFinalizationRequest,
    is_s4_run_eligible_for_downstream,
    load_final_pose_table,
    run_pose_finalization,
    summarize_pose_finalization_project,
    validate_s4_run_for_downstream,
)
from pose_finalization.contracts import RUN_META_FILENAME
from preprocess.sync_writer import build_prepared_sync, write_prepared_sync_npz
from tracking_correction.contracts import (
    ACCEPTANCE_ACCEPTED,
    FINAL_TRACKED_POSE_FILENAME,
)
from tracking_correction.contracts import (
    RUN_META_FILENAME as S3_RUN_META_FILENAME,
)


class _CancelOnNthCheck:
    def __init__(self, n: int) -> None:
        self.n = n
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls >= self.n


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_sync(path: Path, *, frame_count: int, fps: float = 120.0) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    sync = build_prepared_sync(
        prepared_frame_count=frame_count,
        start_frame=0,
        end_frame_exclusive=frame_count,
        fps_header=fps,
        raw_fps_effective=fps,
        raw_pts_time_sec=np.arange(frame_count, dtype=float) / fps,
        raw_pts_status="valid",
        external_ttl_vector=None,
        external_time_status="not_provided",
        external_time_source=None,
        external_time_variable_name=None,
        external_time_units="unknown",
        raw_frame_count_opencv_readable=frame_count,
        prepared_frame_count_opencv_reported=frame_count,
        prepared_frame_count_opencv_readable=frame_count,
    )
    write_prepared_sync_npz(path, sync)
    return path


def _write_accepted_session(
    session_root: Path,
    *,
    n_frames: int = 24,
    run_id: str = "current_lab_two_mouse_headstage_v1__20260101T000000",
) -> Path:
    session_root.mkdir(parents=True, exist_ok=True)
    sync = _write_sync(session_root / "preprocess" / "prepared_sync.npz", frame_count=n_frames)
    run_dir = session_root / "tracking_correction" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for frame_idx in range(n_frames):
        for track in (0, 1):
            for node in ("nose", "neck"):
                missing = 8 <= frame_idx <= 12 and node == "nose" and track == 0
                rows.append(
                    {
                        "frame": frame_idx,
                        "frame_idx": frame_idx,
                        "prepared_frame_idx": frame_idx,
                        "track": track,
                        "node": node,
                        "x": np.nan if missing else float(track * 10 + frame_idx),
                        "y": np.nan if missing else float(track * 20 + 2 * frame_idx),
                        "point_score": 0.9,
                        "instance_score": 1.5,
                        "was_track_swapped": np.uint8(0),
                        "was_node_blanked": np.uint8(0),
                        "time_sec": frame_idx / 120.0,
                        "timestamp_source": "prepared_sync:prepared_time_sec",
                        "video_index": 0,
                        "video_fps": 120.0,
                    }
                )
    pose_path = run_dir / FINAL_TRACKED_POSE_FILENAME
    pd.DataFrame(rows).to_parquet(pose_path, index=False)
    digest = _sha256(pose_path)
    meta = {
        "run_id": run_id,
        "status": "correction_complete",
        "backend_status": "completed",
        "acceptance_state": ACCEPTANCE_ACCEPTED,
        "completed_at": "2026-01-01T00:00:00",
        "profile_id": "current_lab_two_mouse_headstage_v1",
        "acceptance": {
            "accepted_at": "2026-01-01T00:05:00",
            "tracked_pose_sha256": digest,
        },
        "input": {
            "session_root": str(session_root),
            "prepared_sync": str(sync),
        },
    }
    (run_dir / S3_RUN_META_FILENAME).write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return run_dir


def _run_meta(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / RUN_META_FILENAME).read_text(encoding="utf-8"))


def _artifact_fingerprint(result_path: Path) -> dict[str, Any]:
    table = load_final_pose_table(result_path)
    return {
        "columns": list(table.columns),
        "n_rows": len(table),
        "x": table["x"].to_numpy(dtype=float).copy(),
        "y": table["y"].to_numpy(dtype=float).copy(),
        "provenance": table["provenance"].to_numpy(dtype=np.uint8).copy(),
        "was_smoothed": table["was_smoothed"].to_numpy(dtype=bool).copy(),
        "point_score": table["point_score"].to_numpy(dtype=float).copy(),
        "sample_idx": table["sample_idx"].to_numpy(dtype=np.int64).copy(),
        "sha256": _sha256(result_path),
    }


def test_no_callback_preserves_successful_output(tmp_path: Path) -> None:
    session = tmp_path / "session"
    s3 = _write_accepted_session(session, n_frames=20)
    result = run_pose_finalization(
        PoseFinalizationRequest(
            s3_run_dir=s3,
            run_purpose="test",
            timestamp="20260101T100000",
        )
    )
    assert result.success is True
    assert result.status == STATUS_COMPLETED
    assert result.final_pose_path is not None
    assert result.final_pose_path.is_file()


def test_false_callback_preserves_successful_output(tmp_path: Path) -> None:
    session = tmp_path / "session"
    s3 = _write_accepted_session(session, n_frames=20)
    result = run_pose_finalization(
        PoseFinalizationRequest(
            s3_run_dir=s3,
            run_purpose="test",
            timestamp="20260101T100001",
        ),
        cancellation_requested=lambda: False,
    )
    assert result.success is True
    assert result.final_pose_path is not None


def test_scientific_invariance_no_callback_vs_false_callback(tmp_path: Path) -> None:
    session_a = tmp_path / "a"
    session_b = tmp_path / "b"
    s3_a = _write_accepted_session(session_a, n_frames=24)
    s3_b = _write_accepted_session(session_b, n_frames=24)
    baseline = run_pose_finalization(
        PoseFinalizationRequest(
            s3_run_dir=s3_a,
            run_purpose="test",
            timestamp="20260101T100010",
        )
    )
    false_cb = run_pose_finalization(
        PoseFinalizationRequest(
            s3_run_dir=s3_b,
            run_purpose="test",
            timestamp="20260101T100011",
        ),
        cancellation_requested=lambda: False,
    )
    assert baseline.final_pose_path is not None
    assert false_cb.final_pose_path is not None
    left = _artifact_fingerprint(baseline.final_pose_path)
    right = _artifact_fingerprint(false_cb.final_pose_path)
    assert left["columns"] == right["columns"]
    assert left["n_rows"] == right["n_rows"]
    assert left["sha256"] == right["sha256"]
    np.testing.assert_array_equal(left["sample_idx"], right["sample_idx"])
    np.testing.assert_allclose(left["x"], right["x"], equal_nan=True)
    np.testing.assert_allclose(left["y"], right["y"], equal_nan=True)
    np.testing.assert_array_equal(left["provenance"], right["provenance"])
    np.testing.assert_array_equal(left["was_smoothed"], right["was_smoothed"])
    np.testing.assert_allclose(left["point_score"], right["point_score"], equal_nan=True)
    assert baseline.qc == false_cb.qc
    assert baseline.source_s3_run_id == false_cb.source_s3_run_id


def test_cancel_before_run_creation(tmp_path: Path) -> None:
    session = tmp_path / "session"
    s3 = _write_accepted_session(session, n_frames=16)
    with pytest.raises(PoseFinalizationCancelledError):
        run_pose_finalization(
            PoseFinalizationRequest(
                s3_run_dir=s3,
                run_purpose="test",
                timestamp="20260101T100100",
            ),
            cancellation_requested=lambda: True,
        )
    root = session / "pose_finalization"
    assert not root.exists() or list(root.iterdir()) == []


@pytest.mark.parametrize(
    ("nth", "label"),
    [
        (2, "before_timing"),
        (3, "after_timing"),
        (4, "before_dense"),
        (5, "after_dense_before_gap_fill"),
        (6, "after_gap_fill"),
        (7, "after_smoothing"),
        (8, "after_qc"),
        (9, "final_publication_gate"),
    ],
)
def test_cancel_at_safe_checkpoints(tmp_path: Path, nth: int, label: str) -> None:
    del label
    session = tmp_path / "session"
    s3 = _write_accepted_session(session, n_frames=20)
    cancel = _CancelOnNthCheck(nth)
    with pytest.raises(PoseFinalizationCancelledError):
        run_pose_finalization(
            PoseFinalizationRequest(
                s3_run_dir=s3,
                run_purpose="test",
                timestamp=f"20260101T11{nth:04d}",
            ),
            cancellation_requested=cancel,
        )
    runs = list((session / "pose_finalization").iterdir())
    assert len(runs) == 1
    meta = _run_meta(runs[0])
    assert meta["status"] == STATUS_CANCELED
    assert not (runs[0] / FINAL_POSE_FILENAME).exists()
    assert (runs[0] / RUN_META_FILENAME).is_file()
    assert (runs[0] / "settings_used.yaml").is_file()
    assert (runs[0] / "processing_log.txt").is_file()
    err = (meta.get("error") or "").lower()
    assert "cancel" in err


def test_cancel_after_publication_gate_completes(tmp_path: Path) -> None:
    session = tmp_path / "session"
    s3 = _write_accepted_session(session, n_frames=16)
    state = {"in_publication": False}

    def cancel() -> bool:
        return state["in_publication"]

    def publication_started() -> None:
        state["in_publication"] = True

    result = run_pose_finalization(
        PoseFinalizationRequest(
            s3_run_dir=s3,
            run_purpose="test",
            timestamp="20260101T120200",
        ),
        cancellation_requested=cancel,
        publication_started=publication_started,
    )
    assert state["in_publication"] is True
    assert result.success is True
    assert result.status == STATUS_COMPLETED
    assert result.final_pose_path is not None
    assert result.final_pose_path.is_file()


def test_canceled_run_rejected_downstream(tmp_path: Path) -> None:
    session = tmp_path / "session"
    s3 = _write_accepted_session(session, n_frames=16)
    with pytest.raises(PoseFinalizationCancelledError):
        run_pose_finalization(
            PoseFinalizationRequest(
                s3_run_dir=s3,
                run_purpose="test",
                timestamp="20260101T130000",
            ),
            cancellation_requested=_CancelOnNthCheck(2),
        )
    run_dir = next((session / "pose_finalization").iterdir())
    assert is_s4_run_eligible_for_downstream(run_dir) is False
    with pytest.raises(PoseFinalizationError, match="not completed"):
        validate_s4_run_for_downstream(run_dir)
    summary = summarize_pose_finalization_project(session)
    assert summary.runs
    assert summary.runs[0].classification == FAILED_OR_INCOMPLETE
    assert summary.eligible_runs == ()


def test_canceled_run_does_not_affect_prior_completed(tmp_path: Path) -> None:
    session = tmp_path / "session"
    s3 = _write_accepted_session(session, n_frames=16)
    completed = run_pose_finalization(
        PoseFinalizationRequest(
            s3_run_dir=s3,
            run_purpose="test",
            timestamp="20260101T140000",
        )
    )
    with pytest.raises(PoseFinalizationCancelledError):
        run_pose_finalization(
            PoseFinalizationRequest(
                s3_run_dir=s3,
                run_purpose="test",
                timestamp="20260101T140100",
            ),
            cancellation_requested=_CancelOnNthCheck(2),
        )
    assert completed.final_pose_path is not None
    assert completed.final_pose_path.is_file()
    assert is_s4_run_eligible_for_downstream(completed.run_dir) is True
    summary = summarize_pose_finalization_project(session)
    assert len(summary.completed_runs) == 1
    assert summary.eligible_runs
    assert summary.eligible_runs[0].run_id == completed.run_id


def test_ordinary_failure_remains_failed_even_if_cancel_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = tmp_path / "session"
    s3 = _write_accepted_session(session, n_frames=16)
    state = {"cancel": False}

    def boom(*_args: object, **_kwargs: object) -> None:
        state["cancel"] = True
        raise PoseFinalizationError("injected structural failure")

    monkeypatch.setattr(
        "pose_finalization.runner.evaluate_finalization_qc",
        boom,
    )
    with pytest.raises(PoseFinalizationError, match="injected structural failure"):
        run_pose_finalization(
            PoseFinalizationRequest(
                s3_run_dir=s3,
                run_purpose="test",
                timestamp="20260101T150000",
            ),
            cancellation_requested=lambda: state["cancel"],
        )
    assert state["cancel"] is True
    run_dir = next((session / "pose_finalization").iterdir())
    assert _run_meta(run_dir)["status"] == STATUS_FAILED


def test_rerun_after_cancellation(tmp_path: Path) -> None:
    session = tmp_path / "session"
    s3 = _write_accepted_session(session, n_frames=16)
    with pytest.raises(PoseFinalizationCancelledError):
        run_pose_finalization(
            PoseFinalizationRequest(
                s3_run_dir=s3,
                run_purpose="test",
                timestamp="20260101T160000",
            ),
            cancellation_requested=_CancelOnNthCheck(3),
        )
    result = run_pose_finalization(
        PoseFinalizationRequest(
            s3_run_dir=s3,
            run_purpose="test",
            timestamp="20260101T160100",
        )
    )
    assert result.success is True
    assert result.status == STATUS_COMPLETED
    assert is_s4_run_eligible_for_downstream(result.run_dir) is True


def test_progress_stages_reported(tmp_path: Path) -> None:
    session = tmp_path / "session"
    s3 = _write_accepted_session(session, n_frames=12)
    phases: list[str] = []

    def progress(event: object) -> None:
        phases.append(getattr(event, "phase", ""))

    result = run_pose_finalization(
        PoseFinalizationRequest(
            s3_run_dir=s3,
            run_purpose="test",
            timestamp="20260101T170000",
        ),
        progress_callback=progress,
        cancellation_requested=lambda: False,
    )
    assert result.success is True
    assert "validating_inputs" in phases
    assert "preparing_timing" in phases
    assert "building_dense_pose" in phases
    assert "reconstructing_gaps" in phases
    assert "smoothing" in phases
    assert "computing_qc" in phases
    assert "revalidating_sources" in phases
    assert "writing_artifacts" in phases
    assert "completed" in phases
