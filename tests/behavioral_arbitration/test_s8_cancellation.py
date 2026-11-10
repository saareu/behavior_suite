"""S8 cooperative cancellation, publication gate, and eligibility tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from behavioral_arbitration import (
    STATUS_CANCELED,
    BehavioralArbitrationCancelledError,
    BehavioralArbitrationError,
    BehavioralArbitrationRequest,
    is_s8_run_eligible_for_downstream,
    load_behavioral_arbitration_config,
    run_behavioral_arbitration_publication,
    summarize_behavioral_arbitration_project,
    validate_s8_run_for_downstream,
)
from behavioral_arbitration.contracts import (
    BEHAVIOR_BOUTS_FILENAME,
    BEHAVIOR_STATES_FILENAME,
    FAILED_OR_INCOMPLETE,
    QC_FILENAME,
    RUN_META_FILENAME,
    STATUS_COMPLETED,
    STATUS_FAILED,
)
from tests.behavioral_arbitration._fixtures import build_synthetic_session
from tests.behavioral_arbitration.test_export_qc_discovery import (
    _enter_all,
    _patch_publication,
)


class _CancelOnNthCheck:
    def __init__(self, n: int) -> None:
        self.n = n
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls >= self.n


def _run_meta(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / RUN_META_FILENAME).read_text(encoding="utf-8"))


def _request(
    info: dict[str, Any],
    *,
    timestamp: str,
) -> BehavioralArbitrationRequest:
    return BehavioralArbitrationRequest(
        session_root=info["session_root"],
        s7_run_dir=info["s7_run_dir"],
        config=load_behavioral_arbitration_config(None),
        run_purpose="test",
        timestamp=timestamp,
    )


def test_no_callback_preserves_successful_output(tmp_path: Path) -> None:
    info = build_synthetic_session(tmp_path, n_frames=6)
    with _enter_all(_patch_publication(info)):
        result = run_behavioral_arbitration_publication(
            _request(info, timestamp="20260101T100000")
        )
    assert result.success is True
    assert result.status in {STATUS_COMPLETED, "completed_with_warnings"}
    assert result.behavior_states_path is not None
    assert result.behavior_states_path.is_file()
    assert result.behavior_bouts_path is not None
    assert result.behavior_bouts_path.is_file()


def test_false_callback_preserves_successful_output(tmp_path: Path) -> None:
    info = build_synthetic_session(tmp_path, n_frames=6)
    with _enter_all(_patch_publication(info)):
        result = run_behavioral_arbitration_publication(
            _request(info, timestamp="20260101T100001"),
            cancellation_requested=lambda: False,
        )
    assert result.success is True
    assert result.behavior_states_path is not None


def test_cancel_before_intake_raises_without_run_dir(tmp_path: Path) -> None:
    info = build_synthetic_session(tmp_path, n_frames=6)
    with (
        _enter_all(_patch_publication(info)),
        pytest.raises(BehavioralArbitrationCancelledError),
    ):
        run_behavioral_arbitration_publication(
            _request(info, timestamp="20260101T100100"),
            cancellation_requested=lambda: True,
        )
    arb_root = info["session_root"] / "behavioral_arbitration"
    assert not arb_root.exists() or list(arb_root.iterdir()) == []


@pytest.mark.parametrize(
    ("nth", "label"),
    [
        (2, "before_arbitration"),
        (3, "between_sprint1_sprint2"),
        (4, "between_sprint2_sprint3"),
        (5, "after_arbitration_before_states"),
        (6, "after_states_before_bouts"),
        (7, "after_bouts_before_qc"),
        (8, "final_publication_gate"),
    ],
)
def test_cancel_at_safe_checkpoints(tmp_path: Path, nth: int, label: str) -> None:
    del label
    info = build_synthetic_session(tmp_path, n_frames=6)
    cancel = _CancelOnNthCheck(nth)
    with (
        _enter_all(_patch_publication(info)),
        pytest.raises(BehavioralArbitrationCancelledError),
    ):
        run_behavioral_arbitration_publication(
            _request(info, timestamp=f"20260101T11{nth:04d}"),
            cancellation_requested=cancel,
        )
    runs = list((info["session_root"] / "behavioral_arbitration").iterdir())
    assert len(runs) == 1
    meta = _run_meta(runs[0])
    assert meta["status"] == STATUS_CANCELED
    assert not (runs[0] / BEHAVIOR_STATES_FILENAME).exists()
    assert not (runs[0] / BEHAVIOR_BOUTS_FILENAME).exists()
    assert not (runs[0] / QC_FILENAME).exists()
    assert (runs[0] / RUN_META_FILENAME).is_file()
    assert (runs[0] / "settings_used.yaml").is_file()
    assert (runs[0] / "processing_log.txt").is_file()
    err = (meta.get("error") or "").lower()
    assert "cancel" in err


def test_cancel_after_publication_gate_completes(tmp_path: Path) -> None:
    info = build_synthetic_session(tmp_path, n_frames=6)
    state = {"in_publication": False}

    def cancel() -> bool:
        return state["in_publication"]

    def publication_started() -> None:
        state["in_publication"] = True

    with _enter_all(_patch_publication(info)):
        result = run_behavioral_arbitration_publication(
            _request(info, timestamp="20260101T120200"),
            cancellation_requested=cancel,
            publication_started=publication_started,
        )
    assert state["in_publication"] is True
    assert result.success is True
    assert result.status in {STATUS_COMPLETED, "completed_with_warnings"}
    assert result.behavior_states_path is not None
    assert result.behavior_states_path.is_file()


def test_canceled_run_rejected_downstream(tmp_path: Path) -> None:
    info = build_synthetic_session(tmp_path, n_frames=6)
    with (
        _enter_all(_patch_publication(info)),
        pytest.raises(BehavioralArbitrationCancelledError),
    ):
        run_behavioral_arbitration_publication(
            _request(info, timestamp="20260101T130000"),
            cancellation_requested=_CancelOnNthCheck(2),
        )
    run_dir = next((info["session_root"] / "behavioral_arbitration").iterdir())
    assert is_s8_run_eligible_for_downstream(run_dir) is False
    with pytest.raises(BehavioralArbitrationError, match="not completed"):
        validate_s8_run_for_downstream(run_dir)
    summary = summarize_behavioral_arbitration_project(info["session_root"])
    assert summary.runs
    assert summary.runs[0].classification == FAILED_OR_INCOMPLETE
    assert summary.eligible_runs == ()


def test_canceled_run_does_not_affect_prior_completed(tmp_path: Path) -> None:
    info = build_synthetic_session(tmp_path, n_frames=6)
    with _enter_all(_patch_publication(info)):
        completed = run_behavioral_arbitration_publication(
            _request(info, timestamp="20260101T140000")
        )
        with pytest.raises(BehavioralArbitrationCancelledError):
            run_behavioral_arbitration_publication(
                _request(info, timestamp="20260101T140100"),
                cancellation_requested=_CancelOnNthCheck(2),
            )
        assert completed.behavior_states_path is not None
        assert completed.behavior_states_path.is_file()
        # Eligibility uses live S7/S6 validators; keep patches active for the check.
        assert is_s8_run_eligible_for_downstream(completed.run_dir) is True
        summary = summarize_behavioral_arbitration_project(info["session_root"])
        assert len(summary.completed_runs) == 1
        assert summary.eligible_runs
        assert summary.eligible_runs[0].run_id == completed.run_id


def test_ordinary_failure_remains_failed_even_if_cancel_requested(
    tmp_path: Path,
) -> None:
    info = build_synthetic_session(tmp_path, n_frames=6)
    state = {"cancel": False}

    def boom(*_args: object, **_kwargs: object) -> None:
        state["cancel"] = True
        raise BehavioralArbitrationError("injected structural failure")

    patches = _patch_publication(info) + (
        patch("behavioral_arbitration.runner.assert_structural_qc", boom),
    )
    with _enter_all(patches):
        result = run_behavioral_arbitration_publication(
            _request(info, timestamp="20260101T150000"),
            cancellation_requested=lambda: state["cancel"],
        )
    assert state["cancel"] is True
    assert result.success is False
    assert result.status == STATUS_FAILED
    run_dir = next((info["session_root"] / "behavioral_arbitration").iterdir())
    assert _run_meta(run_dir)["status"] == STATUS_FAILED


def test_successful_outputs_unchanged_with_false_cancel_callback(
    tmp_path: Path,
) -> None:
    info = build_synthetic_session(tmp_path, n_frames=8)
    with _enter_all(_patch_publication(info)):
        baseline = run_behavioral_arbitration_publication(
            _request(info, timestamp="20260101T160000")
        )
        with_cancel = run_behavioral_arbitration_publication(
            _request(info, timestamp="20260101T160001"),
            cancellation_requested=lambda: False,
        )
    assert baseline.behavior_states_path is not None
    assert with_cancel.behavior_states_path is not None
    assert baseline.behavior_bouts_path is not None
    assert with_cancel.behavior_bouts_path is not None
    states_a = pd.read_parquet(baseline.behavior_states_path)
    states_b = pd.read_parquet(with_cancel.behavior_states_path)
    bouts_a = pd.read_parquet(baseline.behavior_bouts_path)
    bouts_b = pd.read_parquet(with_cancel.behavior_bouts_path)
    drop_volatile = {
        "run_id",
        "source_s7_run_id",
        "source_s6_run_id",
        "schema_version",
    }
    for frame_a, frame_b in ((states_a, states_b), (bouts_a, bouts_b)):
        cols = [c for c in frame_a.columns if c not in drop_volatile]
        pd.testing.assert_frame_equal(
            frame_a[cols].reset_index(drop=True),
            frame_b[cols].reset_index(drop=True),
            check_dtype=False,
        )
    assert baseline.qc is not None and with_cancel.qc is not None
    assert baseline.qc.get("warning_count") == with_cancel.qc.get("warning_count")
    assert baseline.qc.get("structural_ok") == with_cancel.qc.get("structural_ok")
    assert baseline.qc.get("metrics") == with_cancel.qc.get("metrics")


def test_progress_cancel_polling_is_independent(tmp_path: Path) -> None:
    info = build_synthetic_session(tmp_path, n_frames=6)
    cancel_calls = {"n": 0}
    progress_events: list[str] = []

    def cancel() -> bool:
        cancel_calls["n"] += 1
        return False

    def progress(event: object) -> None:
        progress_events.append(getattr(event, "phase", ""))

    with _enter_all(_patch_publication(info)):
        result = run_behavioral_arbitration_publication(
            _request(info, timestamp="20260101T170000"),
            progress_callback=progress,
            cancellation_requested=cancel,
        )
    assert result.success is True
    # Intake + pre-arbitration + sprint boundaries + states/bouts/qc + publish gate.
    assert cancel_calls["n"] >= 8
    assert "arbitration" in progress_events
    assert "publish" in progress_events
