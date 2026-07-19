from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from pose_inference.run_discovery import (
    COMPLETE_REVIEWABLE,
    FAILED_OR_INCOMPLETE,
    PoseInferenceProjectSummary,
    PoseInferenceRunSummary,
    S1HandoffStatus,
)
from pose_inference.runner import (
    PoseInferenceError,
    PoseInferenceResult,
)
from ui.controllers.pose_inference_controller import (
    InferenceMode,
    PoseInferenceController,
    PoseInferenceUiError,
    S1UiStatus,
)


class MemorySettings:
    def __init__(self, values: dict[str, object] | None = None) -> None:
        self.values = values or {}

    def value(self, key: str, default_value: object = None) -> object:
        return self.values.get(key, default_value)

    def setValue(self, key: str, value: object) -> None:
        self.values[key] = value


def _write_s1(root: Path, *, metadata: str | None = None) -> None:
    preprocess = root / "preprocess"
    preprocess.mkdir(parents=True)
    (preprocess / "prepared_video.mp4").write_bytes(b"video")
    (preprocess / "prepared_sync.npz").write_bytes(b"sync")
    payload = {
        "prepared_video": {
            "path": str(preprocess / "prepared_video.mp4"),
            "frame_count_used_for_sleap": 42,
        },
        "sync": {"frame_count_used_for_sleap": 42},
        "external_time": {
            "provided": True,
            "validation_status": "valid",
            "source_path": "timing.mat",
        },
    }
    (preprocess / "prepare_meta.json").write_text(
        metadata if metadata is not None else json.dumps(payload), encoding="utf-8"
    )


def _handoff(root: Path, *, ready: bool = True) -> S1HandoffStatus:
    preprocess = root / "preprocess"
    paths = {
        "prepared_video": preprocess / "prepared_video.mp4",
        "prepare_meta": preprocess / "prepare_meta.json",
        "prepared_sync": preprocess / "prepared_sync.npz",
    }
    missing = () if ready else ("prepared_sync.npz",)
    return S1HandoffStatus(
        session_root=root,
        preprocess_dir=preprocess,
        prepared_video=paths["prepared_video"],
        prepare_meta=paths["prepare_meta"],
        prepared_sync=paths["prepared_sync"],
        artifact_presence={key: ready for key in paths},
        missing_files=missing,
        is_complete=ready,
    )


def _run(
    root: Path,
    run_id: str,
    *,
    classification: str = COMPLETE_REVIEWABLE,
    qc_outcome: str = "pass",
    mode: str = "bottomup",
    overlay: bool = True,
) -> PoseInferenceRunSummary:
    run_dir = root / "pose_inference" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    model_root = root / "models"
    bottomup_model = model_root / "bottomup"
    centroid_model = model_root / "centroid"
    centered_model = model_root / "centered"
    for path in (bottomup_model, centroid_model, centered_model):
        path.mkdir(parents=True, exist_ok=True)
    profile = root / f"{mode}-profile.yaml"
    profile.write_text(
        f"profile_id: test-{mode}\ninference_mode: {mode}\ntracking:\n  enabled: false\n",
        encoding="utf-8",
    )
    for filename in ("pose.slp", "pose.parquet", "pose_meta.json"):
        (run_dir / filename).write_bytes(b"x")
    if overlay:
        (run_dir / "overlay.mp4").write_bytes(b"x")
    presence = {
        "pose_slp": True,
        "pose_parquet": True,
        "overlay_mp4": overlay,
        "pose_meta": True,
        "settings_used": True,
        "job_manifest": True,
        "processing_log": True,
    }
    return PoseInferenceRunSummary(
        run_dir=run_dir,
        run_id=run_id,
        status="success" if classification == COMPLETE_REVIEWABLE else "failed",
        classification=classification,
        artifact_presence=presence,
        missing_required_artifacts=() if overlay else ("overlay.mp4",),
        model_id="legacy-model" if mode == "bottomup" else "topdown-bundle",
        model_path=str(bottomup_model) if mode == "bottomup" else None,
        inference_mode=mode,
        bottomup_model_path=str(bottomup_model) if mode == "bottomup" else None,
        centroid_model_path=str(centroid_model) if mode == "topdown" else None,
        centered_instance_model_path=(
            str(centered_model) if mode == "topdown" else None
        ),
        profile_path=str(profile),
        created_at="2026-07-19T10:00:00+03:00",
        pose_qc_outcome=qc_outcome,
        review_recommendation_reasons=(
            ("high_missing_keypoint_fraction",)
            if qc_outcome == "review_recommended"
            else ()
        ),
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


def _summary(
    root: Path, runs: tuple[PoseInferenceRunSummary, ...] = ()
) -> PoseInferenceProjectSummary:
    return PoseInferenceProjectSummary(
        session_root=root,
        s1_handoff=_handoff(root),
        runs=runs,
    )


def test_valid_and_invalid_s1_handoff_control_submission(tmp_path: Path) -> None:
    valid = tmp_path / "valid"
    valid.mkdir()
    _write_s1(valid)
    controller = PoseInferenceController(discovery=lambda _root: _summary(valid))
    controller.set_session(valid)

    assert controller.state.s1_status is S1UiStatus.READY
    assert controller.state.prepared_frame_count == 42
    assert controller.state.timing_status == "external timing: valid (timing.mat)"
    assert controller.can_submit is False

    invalid = tmp_path / "invalid"
    invalid.mkdir()
    controller = PoseInferenceController(
        discovery=lambda _root: PoseInferenceProjectSummary(
            session_root=invalid,
            s1_handoff=_handoff(invalid, ready=False),
            runs=(),
        )
    )
    controller.set_session(invalid)

    assert controller.state.s1_status is S1UiStatus.INCOMPLETE
    assert controller.can_submit is False


def test_malformed_prepare_metadata_is_nonfatal_and_blocks_run(tmp_path: Path) -> None:
    _write_s1(tmp_path, metadata="{not json")
    controller = PoseInferenceController(discovery=lambda _root: _summary(tmp_path))

    controller.set_session(tmp_path)

    assert controller.state.s1_status is S1UiStatus.METADATA_UNREADABLE
    assert "unreadable or invalid" in controller.state.s1_message


def test_bottomup_and_topdown_requests_use_explicit_model_specs(tmp_path: Path) -> None:
    _write_s1(tmp_path)
    controller = PoseInferenceController(discovery=lambda _root: _summary(tmp_path))
    controller.set_session(tmp_path)
    controller.state.bottomup_model_path = "models/bottomup"

    bottomup = controller.build_request()

    assert bottomup.model_path is None
    assert bottomup.model_spec is not None
    assert bottomup.model_spec.inference_mode == "bottomup"
    assert bottomup.model_spec.bottomup_model_path == Path("models/bottomup")
    assert bottomup.model_spec.centroid_model_path is None

    controller.set_mode(InferenceMode.TOPDOWN)
    controller.state.centroid_model_path = "models/centroid"
    controller.state.centered_instance_model_path = "models/centered"
    topdown = controller.build_request()

    assert topdown.model_spec is not None
    assert topdown.model_spec.inference_mode == "topdown"
    assert topdown.model_spec.bottomup_model_path is None
    assert topdown.model_spec.centroid_model_path == Path("models/centroid")
    assert topdown.model_spec.centered_instance_model_path == Path("models/centered")


@pytest.mark.parametrize(
    ("mode", "field", "message"),
    [
        (InferenceMode.BOTTOMUP, "bottomup", "bottom-up model"),
        (InferenceMode.TOPDOWN, "centroid", "centroid model"),
        (InferenceMode.TOPDOWN, "centered", "centered-instance model"),
    ],
)
def test_incomplete_mode_model_combinations_are_rejected(
    tmp_path: Path, mode: InferenceMode, field: str, message: str
) -> None:
    _write_s1(tmp_path)
    controller = PoseInferenceController(discovery=lambda _root: _summary(tmp_path))
    controller.set_session(tmp_path)
    controller.set_mode(mode)
    if field != "bottomup":
        controller.state.bottomup_model_path = "unused"
    if field != "centroid":
        controller.state.centroid_model_path = "centroid"
    if field != "centered":
        controller.state.centered_instance_model_path = "centered"

    with pytest.raises(PoseInferenceUiError, match=message):
        controller.build_request()


def test_saved_paths_preserve_unavailable_local_text_for_user_correction(
    tmp_path: Path,
) -> None:
    bottomup = tmp_path / "bottomup"
    bottomup.mkdir()
    executable = tmp_path / "sleap-nn.exe"
    executable.write_bytes(b"exe")
    settings = MemorySettings(
        {
            "pose_inference/last_bottomup_model": str(bottomup),
            "pose_inference/last_centroid_model": str(tmp_path / "missing"),
            "pose_inference/last_sleap_executable": str(executable),
        }
    )

    controller = PoseInferenceController(settings=settings)

    assert controller.state.bottomup_model_path == str(bottomup.resolve())
    assert controller.state.centroid_model_path == str((tmp_path / "missing").resolve())
    assert any("currently unavailable" in notice for notice in controller.state.notices)
    assert controller.state.sleap_executable_path == str(executable.resolve())
    assert controller.state.bottomup_profile_path.endswith(
        "sleapnn_default_profile.yaml"
    )


def test_backend_preflight_error_is_preserved_as_actionable_failure(
    tmp_path: Path,
) -> None:
    _write_s1(tmp_path)

    def reject(_request):
        raise PoseInferenceError("SLEAP-NN 0.2.x is unsupported; install 0.3.x")

    controller = PoseInferenceController(
        discovery=lambda _root: _summary(tmp_path), runner=reject
    )
    controller.set_session(tmp_path)
    controller.state.bottomup_model_path = "model"
    request = controller.begin_run()

    with pytest.raises(PoseInferenceError):
        controller.execute_request(request)
    try:
        controller.execute_request(request)
    except PoseInferenceError as exc:
        controller.apply_error(exc)

    assert controller.state.run_stage == "Failed"
    assert "install 0.3.x" in (controller.state.run_error or "")


@pytest.mark.parametrize("qc_outcome", ["pass", "review_recommended"])
def test_complete_pass_and_review_runs_can_feed_s3(
    tmp_path: Path, qc_outcome: str
) -> None:
    _write_s1(tmp_path)
    run = _run(tmp_path, "run-1", qc_outcome=qc_outcome)
    controller = PoseInferenceController(
        discovery=lambda _root: _summary(tmp_path, (run,))
    )
    controller.set_session(tmp_path)
    controller.select_run(run.run_id)

    handoff = controller.build_s3_handoff()

    assert handoff.pose_slp_path == run.run_dir / "pose.slp"
    assert handoff.pose_parquet_path == run.run_dir / "pose.parquet"
    assert handoff.qc_outcome == qc_outcome


@pytest.mark.parametrize(
    "run",
    [
        pytest.param("failed", id="failed"),
        pytest.param("missing_overlay", id="missing-overlay"),
    ],
)
def test_incomplete_runs_cannot_feed_s3(tmp_path: Path, run: str) -> None:
    _write_s1(tmp_path)
    selected = _run(
        tmp_path,
        "run-1",
        classification=(
            FAILED_OR_INCOMPLETE if run == "failed" else "missing_required_artifacts"
        ),
        overlay=run != "missing_overlay",
    )
    controller = PoseInferenceController(
        discovery=lambda _root: _summary(tmp_path, (selected,))
    )
    controller.set_session(tmp_path)
    controller.select_run(selected.run_id)

    assert controller.selected_run_can_feed_s3() is False
    with pytest.raises(PoseInferenceUiError, match="technically complete"):
        controller.build_s3_handoff()


def test_completion_refreshes_discovery_and_selects_new_run(tmp_path: Path) -> None:
    _write_s1(tmp_path)
    run = _run(tmp_path, "new-run")
    calls = 0

    def discover(_root: Path | str) -> PoseInferenceProjectSummary:
        nonlocal calls
        calls += 1
        return _summary(tmp_path, (run,) if calls > 1 else ())

    result = PoseInferenceResult(
        success=True,
        status="success",
        run_id=run.run_id,
        run_dir=run.run_dir,
        pose_slp_path=run.run_dir / "pose.slp",
        pose_meta_path=run.run_dir / "pose_meta.json",
        settings_used_path=run.run_dir / "settings_used.yaml",
        job_manifest_path=run.run_dir / "job_manifest.yaml",
        processing_log_path=run.run_dir / "processing_log.txt",
        command=("sleap-nn", "predict"),
    )
    controller = PoseInferenceController(discovery=discover)
    controller.set_session(tmp_path)

    controller.apply_result(result)

    assert controller.state.selected_run_id == "new-run"
    assert controller.selected_run is run


def test_legacy_bottomup_run_remains_visible_and_copyable(tmp_path: Path) -> None:
    _write_s1(tmp_path)
    legacy = _run(tmp_path, "legacy", mode="bottomup")
    controller = PoseInferenceController(
        discovery=lambda _root: _summary(tmp_path, (legacy,))
    )
    controller.set_session(tmp_path)
    controller.select_run("legacy")

    controller.copy_selected_run_settings()

    assert controller.runs == (legacy,)
    assert controller.state.inference_mode is InferenceMode.BOTTOMUP
    assert controller.state.bottomup_model_path == str(tmp_path / "models" / "bottomup")


def test_technical_details_include_flagged_intervals_and_runtime(tmp_path: Path) -> None:
    _write_s1(tmp_path)
    run = _run(tmp_path, "review", qc_outcome="review_recommended")
    run = replace(
        run,
        sleap_executable_path="C:/env/sleap-nn.exe",
        sleap_executable_version="0.3.0",
        command=("C:/env/sleap-nn.exe", "predict"),
        diagnostic_findings=("partial skeletons are present",),
    )
    controller = PoseInferenceController(
        discovery=lambda _root: _summary(tmp_path, (run,))
    )
    controller.set_session(tmp_path)
    controller.select_run("review")

    details = controller.technical_details()

    assert '"start_frame": 4' in details
    assert "partial skeletons are present" in details
    assert "C:/env/sleap-nn.exe" in details


def test_s3_revalidates_deleted_artifacts_at_action_time(tmp_path: Path) -> None:
    _write_s1(tmp_path)
    run = _run(tmp_path, "deleted-before-s3")
    controller = PoseInferenceController(
        discovery=lambda _root: _summary(tmp_path, (run,))
    )
    controller.set_session(tmp_path)
    controller.select_run(run.run_id)
    (run.run_dir / "pose.parquet").unlink()

    with pytest.raises(PoseInferenceUiError, match="pose.parquet is missing"):
        controller.build_s3_handoff()


@pytest.mark.parametrize("qc_outcome", ["failed", None])
def test_s3_rejects_contradictory_or_failed_qc(
    tmp_path: Path,
    qc_outcome: str | None,
) -> None:
    _write_s1(tmp_path)
    run = replace(_run(tmp_path, "bad-qc"), pose_qc_outcome=qc_outcome)
    controller = PoseInferenceController(
        discovery=lambda _root: _summary(tmp_path, (run,))
    )
    controller.set_session(tmp_path)
    controller.select_run(run.run_id)

    with pytest.raises(PoseInferenceUiError, match="pose QC metadata"):
        controller.build_s3_handoff()


def test_active_task_rejects_s3_and_run_selection_actions(tmp_path: Path) -> None:
    _write_s1(tmp_path)
    run = _run(tmp_path, "selected")
    controller = PoseInferenceController(
        discovery=lambda _root: _summary(tmp_path, (run,))
    )
    controller.set_session(tmp_path)
    controller.select_run(run.run_id)
    controller.state.bottomup_model_path = str(tmp_path / "models" / "bottomup")
    controller.begin_run()
    token = controller.active_task_token

    with pytest.raises(PoseInferenceUiError, match="task is active"):
        controller.build_s3_handoff()
    with pytest.raises(PoseInferenceUiError, match="task is active"):
        controller.select_run(run.run_id)

    controller.finish_run(token=token)


def test_malformed_and_incomplete_rerun_metadata_are_actionable(
    tmp_path: Path,
) -> None:
    _write_s1(tmp_path)
    malformed = replace(_run(tmp_path, "malformed"), inference_mode="sideways")
    incomplete = replace(
        _run(tmp_path, "incomplete"),
        bottomup_model_path=None,
        model_path=None,
    )
    selected = malformed

    def discover(_root):
        return _summary(tmp_path, (selected,))

    controller = PoseInferenceController(discovery=discover)
    controller.set_session(tmp_path)
    controller.select_run(malformed.run_id)

    with pytest.raises(PoseInferenceUiError, match="unsupported or missing"):
        controller.copy_selected_run_settings()

    selected = incomplete
    controller.refresh_discovery()
    controller.select_run(incomplete.run_id)
    assert "authoritative bottom-up model path" in (
        controller.rerun_unavailable_reason() or ""
    )


def test_task_tokens_reject_stale_progress_and_terminal_updates(
    tmp_path: Path,
) -> None:
    _write_s1(tmp_path)
    controller = PoseInferenceController(discovery=lambda _root: _summary(tmp_path))
    controller.set_session(tmp_path)
    model = tmp_path / "model"
    model.mkdir()
    controller.state.bottomup_model_path = str(model)

    controller.begin_run()
    first_token = controller.active_task_token
    assert controller.apply_progress("running inference", None, token=first_token)
    controller.apply_error(PoseInferenceError("first failed"), token=first_token)
    controller.finish_run(token=first_token)

    controller.begin_run()
    second_token = controller.active_task_token
    assert second_token > first_token
    assert not controller.apply_progress(
        "stale queued progress", None, token=first_token
    )
    assert not controller.apply_error(
        PoseInferenceError("stale terminal"), token=first_token
    )
    assert controller.state.run_stage == "Validating backend preflight"
    assert controller.state.task_running is True
    assert controller.apply_progress("building command", None, token=second_token)
    assert controller.state.run_stage == "building command"
    controller.finish_run(token=second_token)


def test_malformed_flagged_intervals_are_capped_and_diagnosed(
    tmp_path: Path,
) -> None:
    _write_s1(tmp_path)
    entries: list[object] = ["bad"] + [
        {
            "video_index": 2,
            "start_frame": index,
            "end_frame": index,
            "frame_count": 1,
            "time_span_sec": 0.0,
        }
        for index in range(12)
    ]
    run = replace(
        _run(tmp_path, "malformed-intervals"),
        flagged_intervals={"bad_warning": "not-a-list", "valid_warning": entries},
    )
    controller = PoseInferenceController(
        discovery=lambda _root: _summary(tmp_path, (run,))
    )
    controller.set_session(tmp_path)
    controller.select_run(run.run_id)

    intervals, diagnostics = controller.normalized_flagged_intervals()

    assert len(intervals["valid_warning"]) == 10
    assert intervals["valid_warning"][0]["video_index"] == 2
    assert any("not a list/tuple" in item for item in diagnostics)
    assert any("not a mapping" in item for item in diagnostics)
    assert any("capped at 10" in item for item in diagnostics)
    assert "inclusive frames" in controller.technical_details()
    assert "timestamp span (time_span_sec)" in controller.technical_details()


def test_persisted_unc_path_is_loaded_without_startup_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unc = r"\\offline-server\models\centroid"
    settings = MemorySettings({"pose_inference/last_centroid_model": unc})
    original_exists = Path.exists

    def guarded_exists(path: Path) -> bool:
        if str(path).startswith(r"\\offline-server"):
            raise AssertionError("startup must not probe persisted UNC paths")
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", guarded_exists)

    controller = PoseInferenceController(settings=settings)

    assert controller.state.centroid_model_path == unc
    assert unc in controller.state.unverified_network_paths
