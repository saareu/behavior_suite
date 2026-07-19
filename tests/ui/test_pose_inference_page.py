from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import replace
from pathlib import Path

from PySide6.QtWidgets import QApplication

from pose_inference.run_discovery import (
    COMPLETE_REVIEWABLE,
    PoseInferenceProjectSummary,
    PoseInferenceRunSummary,
    S1HandoffStatus,
)
from pose_inference.runner import (
    PoseInferenceError,
    PoseInferencePreflightError,
    PoseInferencePreflightFailure,
    PoseInferencePreflightResult,
    PoseInferencePreflightStage,
    PoseInferenceResult,
    SleapRuntime,
    validate_s1_handoff,
)
from ui.controllers.pose_inference_controller import (
    InferenceMode,
    PoseInferenceController,
)
from ui.controllers.preprocess_setup_controller import PreprocessSetupController
from ui.main_window import MainWindow
from ui.pages.pose_inference_page import PoseInferencePage

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_APPLICATION = QApplication.instance() or QApplication([])


def _write_s1(root: Path) -> None:
    preprocess = root / "preprocess"
    preprocess.mkdir(parents=True, exist_ok=True)
    (preprocess / "prepared_video.mp4").write_bytes(b"video")
    (preprocess / "prepared_sync.npz").write_bytes(b"sync")
    (preprocess / "prepare_meta.json").write_text(
        json.dumps(
            {
                "prepared_video": {"frame_count_used_for_sleap": 12},
                "sync": {"frame_count_used_for_sleap": 12},
                "external_time": {"provided": False},
            }
        ),
        encoding="utf-8",
    )


def _handoff(root: Path) -> S1HandoffStatus:
    preprocess = root / "preprocess"
    return S1HandoffStatus(
        session_root=root,
        preprocess_dir=preprocess,
        prepared_video=preprocess / "prepared_video.mp4",
        prepare_meta=preprocess / "prepare_meta.json",
        prepared_sync=preprocess / "prepared_sync.npz",
        artifact_presence={
            "prepared_video": True,
            "prepare_meta": True,
            "prepared_sync": True,
        },
        missing_files=(),
        is_complete=True,
    )


def _summary(
    root: Path, runs: tuple[PoseInferenceRunSummary, ...] = ()
) -> PoseInferenceProjectSummary:
    return PoseInferenceProjectSummary(
        session_root=root,
        s1_handoff=_handoff(root),
        runs=runs,
    )


def _run(
    root: Path,
    run_id: str,
    *,
    qc_outcome: str = "pass",
    overlay: bool = True,
    classification: str = COMPLETE_REVIEWABLE,
) -> PoseInferenceRunSummary:
    run_dir = root / "pose_inference" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model = root / "models" / "model"
    model.mkdir(parents=True, exist_ok=True)
    profile = root / "bottomup-profile.yaml"
    profile.write_text(
        "profile_id: ui-test\ninference_mode: bottomup\ntracking:\n  enabled: false\n",
        encoding="utf-8",
    )
    for name in ("pose.slp", "pose.parquet", "pose_meta.json"):
        (run_dir / name).write_bytes(b"x")
    if overlay:
        (run_dir / "overlay.mp4").write_bytes(b"x")
    return PoseInferenceRunSummary(
        run_dir=run_dir,
        run_id=run_id,
        status="success",
        classification=classification,
        artifact_presence={
            "pose_slp": True,
            "pose_parquet": True,
            "overlay_mp4": overlay,
            "pose_meta": True,
        },
        missing_required_artifacts=() if overlay else ("overlay.mp4",),
        model_id="model",
        model_path=str(model),
        inference_mode="bottomup",
        bottomup_model_path=str(model),
        profile_path=str(profile),
        pose_qc_outcome=qc_outcome,
        review_recommendation_reasons=("missing_keypoints",)
        if qc_outcome == "review_recommended"
        else (),
        flagged_intervals={
            "missing_keypoints": [
                {
                    "start_frame": 4,
                    "end_frame": 8,
                    "frame_count": 5,
                    "time_span_sec": 0.2,
                }
            ]
        }
        if qc_outcome == "review_recommended"
        else {},
    )


def _wait_until(predicate, timeout_sec: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while not predicate():
        _APPLICATION.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for Qt state change")
        time.sleep(0.005)
    _APPLICATION.processEvents()


def _result(run_dir: Path, *, success: bool = True) -> PoseInferenceResult:
    log = run_dir / "processing_log.txt"
    log.write_text(
        f"status: {'success' if success else 'inference_failed'}\n", encoding="utf-8"
    )
    return PoseInferenceResult(
        success=success,
        status="success" if success else "inference_failed",
        run_id=run_dir.name,
        run_dir=run_dir,
        pose_slp_path=run_dir / "pose.slp",
        pose_meta_path=run_dir / "pose_meta.json",
        settings_used_path=run_dir / "settings_used.yaml",
        job_manifest_path=run_dir / "job_manifest.yaml",
        processing_log_path=log,
        command=("C:/env/sleap-nn.exe", "predict"),
        returncode=0 if success else 1,
    )


def test_mode_fields_switch_between_bottomup_and_topdown(tmp_path: Path) -> None:
    _write_s1(tmp_path)
    controller = PoseInferenceController(discovery=lambda _root: _summary(tmp_path))
    controller.set_session(tmp_path)
    page = PoseInferencePage(controller)
    page.show()
    _APPLICATION.processEvents()

    assert page.bottomup_row.isHidden() is False
    assert page.centroid_row.isHidden() is True
    assert page.centered_row.isHidden() is True

    page.mode_combo.setCurrentIndex(
        page.mode_combo.findData(InferenceMode.TOPDOWN.value)
    )
    _APPLICATION.processEvents()

    assert page.bottomup_row.isHidden() is True
    assert page.centroid_row.isHidden() is False
    assert page.centered_row.isHidden() is False
    page.close()


def test_invalid_handoff_disables_run_and_malformed_run_does_not_crash(
    tmp_path: Path,
) -> None:
    tmp_path.mkdir(exist_ok=True)
    summary = PoseInferenceProjectSummary(
        session_root=tmp_path,
        s1_handoff=_summary(tmp_path).s1_handoff,
        runs=(),
    )
    controller = PoseInferenceController(discovery=lambda _root: summary)
    page = PoseInferencePage(controller)

    page.set_session(tmp_path)

    assert page.run_button.isEnabled() is False
    assert "metadata" in page.s1_status_label.text().lower()
    page.close()


def test_review_intervals_are_displayed_and_missing_overlay_disables_action(
    tmp_path: Path,
) -> None:
    _write_s1(tmp_path)
    review = _run(
        tmp_path,
        "review-run",
        qc_outcome="review_recommended",
        overlay=False,
        classification="missing_required_artifacts",
    )
    controller = PoseInferenceController(
        discovery=lambda _root: _summary(tmp_path, (review,))
    )
    controller.set_session(tmp_path)
    controller.select_run(review.run_id)
    page = PoseInferencePage(controller)
    page.refresh_from_state()

    assert "Flagged intervals: 1" in page.selected_summary.text()
    assert '"start_frame": 4' in page.technical_details.toPlainText()
    assert page.open_overlay_button.isEnabled() is False
    assert page.continue_s3_button.isEnabled() is False
    page.close()


def test_backend_preflight_failure_is_shown_from_background_worker(
    tmp_path: Path,
) -> None:
    _write_s1(tmp_path)

    def reject(_request):
        raise PoseInferenceError("Model role is incompatible with top-down centroid")

    controller = PoseInferenceController(
        discovery=lambda _root: _summary(tmp_path), runner=reject
    )
    controller.set_session(tmp_path)
    page = PoseInferencePage(controller)
    page.bottomup_model.setText(str(tmp_path / "model"))
    page.show()

    page.run_button.click()
    _wait_until(lambda: not page.task_runner.is_running)

    assert "Model role is incompatible" in page.error_label.text()
    assert controller.state.run_stage == "Failed"
    page.close()


def test_mocked_background_execution_stays_responsive_and_selects_new_run(
    tmp_path: Path,
) -> None:
    _write_s1(tmp_path)
    completed = False
    release = threading.Event()
    run = _run(tmp_path, "new-run")

    def discover(_root):
        return _summary(tmp_path, (run,) if completed else ())

    def execute(_request):
        nonlocal completed
        release.wait(timeout=2)
        completed = True
        return _result(run.run_dir)

    controller = PoseInferenceController(discovery=discover, runner=execute)
    controller.set_session(tmp_path)
    page = PoseInferencePage(controller)
    page.bottomup_model.setText(str(tmp_path / "model"))
    page.show()

    page.run_button.click()
    _wait_until(lambda: page.task_runner.is_running)
    assert page.progress.isHidden() is False
    assert page.session_path.text() == str(tmp_path)
    release.set()
    _wait_until(lambda: not page.task_runner.is_running)

    assert controller.state.selected_run_id == "new-run"
    assert "status: success" in page.log_excerpt.toPlainText()
    assert page.runs_table.rowCount() == 1
    page.close()


def test_s1_completion_and_main_navigation_preserve_same_session(
    tmp_path: Path,
) -> None:
    _write_s1(tmp_path)
    pose_controller = PoseInferenceController(
        discovery=lambda _root: _summary(tmp_path)
    )
    window = MainWindow(
        PreprocessSetupController(), pose_controller=pose_controller
    )

    window.wizard.preprocessing_completed.emit(tmp_path)
    _APPLICATION.processEvents()

    assert window.pages.currentWidget() is window.pose_page
    assert pose_controller.state.session_root == tmp_path.resolve()

    window.show_subsystem_1()
    window.wizard.session_changed.emit(tmp_path)
    window.s2_button.click()
    _APPLICATION.processEvents()

    assert window.pages.currentWidget() is window.pose_page
    assert pose_controller.state.session_root == tmp_path.resolve()

    window.pose_page.back_to_s1_requested.emit()
    _APPLICATION.processEvents()
    assert window.pages.currentWidget() is window.wizard
    assert window._session_root == tmp_path.resolve()
    window.close()


def test_empty_run_list_shows_explicit_message(tmp_path: Path) -> None:
    _write_s1(tmp_path)
    controller = PoseInferenceController(discovery=lambda _root: _summary(tmp_path))
    controller.set_session(tmp_path)

    page = PoseInferencePage(controller)

    assert page.empty_runs_label.isHidden() is False
    assert (
        "No Subsystem 2 runs were found for this session."
        in page.empty_runs_label.text()
    )
    assert "configure a new inference run" in page.empty_runs_label.text().lower()
    page.close()


def test_row_selection_immediately_populates_technical_details(tmp_path: Path) -> None:
    _write_s1(tmp_path)
    newest = _run(tmp_path, "newest")
    older = _run(tmp_path, "older")
    controller = PoseInferenceController(
        discovery=lambda _root: _summary(tmp_path, (newest, older))
    )
    controller.set_session(tmp_path)
    page = PoseInferencePage(controller)
    page.show()
    _APPLICATION.processEvents()

    assert controller.state.selected_run_id == "newest"
    assert page.scroll_area.verticalScrollBar().value() == 0
    page.runs_table.selectRow(1)
    _APPLICATION.processEvents()

    assert controller.state.selected_run_id == "older"
    assert '"run_id": "older"' in page.technical_details.toPlainText()
    assert page.runs_table.item(1, 0).isSelected()
    page.runs_table.itemActivated.emit(page.runs_table.item(1, 0))
    _APPLICATION.processEvents()
    assert page.technical_details.hasFocus()
    assert page.runs_table.item(1, 0).isSelected()
    page.close()


def test_controls_disable_immediately_and_double_run_cannot_clear_active_state(
    tmp_path: Path,
) -> None:
    _write_s1(tmp_path)
    release = threading.Event()
    run = _run(tmp_path, "eventual-result")

    def execute(_request):
        release.wait(timeout=2)
        return _result(run.run_dir)

    controller = PoseInferenceController(
        discovery=lambda _root: _summary(tmp_path, (run,)),
        runner=execute,
    )
    controller.set_session(tmp_path)
    page = PoseInferencePage(controller)
    page.bottomup_model.setText(str(tmp_path / "models" / "model"))
    page.show()

    page.run_button.click()
    _wait_until(lambda: page.task_runner.is_running)

    assert controller.state.task_running is True
    assert page.run_button.isEnabled() is False
    assert page.validate_button.isEnabled() is False
    assert page.session_path.isEnabled() is False
    assert page.runs_table.isEnabled() is False
    assert page.bottomup_model.isEnabled() is False
    page._run_inference()
    assert controller.state.task_running is True
    assert "task is active" in page.error_label.text()
    page._continue_to_s3()
    assert "task is active" in page.error_label.text()

    release.set()
    _wait_until(lambda: not page.task_runner.is_running)
    assert controller.state.task_running is False
    page.close()


def test_malformed_flagged_intervals_do_not_crash_rendering(tmp_path: Path) -> None:
    _write_s1(tmp_path)
    run = replace(
        _run(tmp_path, "malformed"),
        flagged_intervals={
            "wrong_container": "bad",
            "mixed": [7, {"video_index": 3, "start_frame": 2, "end_frame": 2}],
        },
    )
    controller = PoseInferenceController(
        discovery=lambda _root: _summary(tmp_path, (run,))
    )
    controller.set_session(tmp_path)

    page = PoseInferencePage(controller)

    assert "Metadata diagnostic" in page.selected_summary.text()
    assert "not a list/tuple" in page.technical_details.toPlainText()
    assert "video_index=3" in page.technical_details.toPlainText()
    page.close()


def test_validate_configuration_uses_preflight_without_inference_or_run_dir(
    tmp_path: Path,
) -> None:
    _write_s1(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    executable = tmp_path / "sleap-nn.exe"
    executable.write_bytes(b"exe")
    preflight_calls = []

    def reject_inference(_request):
        raise AssertionError("validation action must not invoke inference")

    def validate(request):
        preflight_calls.append(request)
        handoff = validate_s1_handoff(request.session_root)
        prospective = tmp_path / "pose_inference" / "prospective"
        return PoseInferencePreflightResult(
            s1_handoff=handoff,
            profile_path=request.profile_path.resolve(),
            profile_id="ui-default",
            inference_mode="bottomup",
            model_id="model",
            sleap_runtime=SleapRuntime(executable.resolve(), "0.3.0"),
            prospective_run_id="prospective",
            prospective_run_dir=prospective,
            command=(
                str(executable.resolve()),
                "predict",
                "--output_path",
                str(prospective / "pose.slp"),
            ),
        )

    controller = PoseInferenceController(
        discovery=lambda _root: _summary(tmp_path),
        runner=reject_inference,
        preflight=validate,
    )
    controller.set_session(tmp_path)
    page = PoseInferencePage(controller)
    page.bottomup_model.setText(str(model))
    page.show()

    page.validate_button.click()
    _wait_until(lambda: not page.task_runner.is_running)

    assert len(preflight_calls) == 1
    assert controller.state.last_result is None
    assert controller.state.run_stage == "Configuration valid"
    assert "Resolved command (not executed)" in page.log_excerpt.toPlainText()
    assert not (tmp_path / "pose_inference").exists()
    page.close()


def test_validate_configuration_displays_structured_runtime_failure(
    tmp_path: Path,
) -> None:
    _write_s1(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    executable = (tmp_path / "sleap-nn.exe").resolve()
    executable.write_bytes(b"exe")
    stage_statuses = (
        "passed",
        "passed",
        "passed",
        "passed",
        "passed",
        "failed",
        "not reached",
    )
    stages = tuple(
        PoseInferencePreflightStage(name=name, status=status)
        for name, status in zip(
            (
                "S1 handoff",
                "prepared-video/frame/timing contract",
                "inference mode and model bundle",
                "profile compatibility",
                "SLEAP executable resolution",
                "SLEAP version compatibility",
                "command construction",
            ),
            stage_statuses,
            strict=True,
        )
    )

    def fail_runtime(_request):
        raise PoseInferencePreflightError(
            PoseInferencePreflightFailure(
                stages=stages,
                error=(
                    f"SLEAP-NN 0.2.0 at {executable} is incompatible; "
                    "Behavior Suite currently requires SLEAP-NN 0.3.x."
                ),
                resolved_executable_path=executable,
                found_sleap_version="0.2.0",
            )
        )

    controller = PoseInferenceController(
        discovery=lambda _root: _summary(tmp_path),
        runner=lambda _request: (_ for _ in ()).throw(
            AssertionError("inference must not run")
        ),
        preflight=fail_runtime,
    )
    controller.set_session(tmp_path)
    page = PoseInferencePage(controller)
    page.bottomup_model.setText(str(model))
    page.show()

    page.validate_button.click()
    _wait_until(lambda: not page.task_runner.is_running)

    report = page.log_excerpt.toPlainText()
    assert "Configuration inputs validated up to runtime check." in report
    assert "Local SLEAP-NN runtime is incompatible:" in report
    assert "found 0.2.0; Behavior Suite currently requires 0.3.x." in report
    assert f"Resolved SLEAP executable: {executable}" in report
    assert "profile compatibility: passed" in report
    assert "SLEAP version compatibility: failed" in report
    assert "command construction: not reached" in report
    assert "Full actionable error:" in report
    assert controller.state.run_stage == "Failed"
    assert not (tmp_path / "pose_inference").exists()
    page.close()
