"""C2-C GUI: QC overlay export controller gates, worker safety, discovery."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PySide6.QtWidgets import QApplication

from overlay.contracts import (
    OVERLAY_ARTIFACT_MANIFEST_SCHEMA,
    PIXEL_FORMAT_BGR,
    OverlayExportCancelled,
)
from overlay.discovery import STATUS_CANCELED
from overlay.export import OverlayQcExportResult, OverlayS4PoseQcExportResult
from overlay.provenance import OverlayArtifactManifest
from ui.controllers.overlay_qc_export_controller import (
    OverlayQcExportController,
    OverlayQcExportUiError,
)
from ui.controllers.overlay_qc_export_support import (
    OverlayQcLaunchContext,
    OverlayQcUiSettings,
    actionable_overlay_error,
)
from ui.dialogs.overlay_qc_export_dialog import OverlayQcExportDialog
from ui.tasks import GuiTaskKind

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_APPLICATION = QApplication.instance() or QApplication([])


def _manifest() -> OverlayArtifactManifest:
    return OverlayArtifactManifest(
        schema_version=OVERLAY_ARTIFACT_MANIFEST_SCHEMA,
        renderer_version="overlay_renderer_c2a_v1",
        source_runs=(),
        source_digests={},
        prepared_video={"path": "prepared_video.mp4", "fps": 30.0},
        timing_mapping={"kind": "test"},
        plateau_policy="last_in_plateau",
        layer_selections=("debug.frame_counter",),
        unavailable_layers=(),
        output_frame_domain="prepared",
        output_fps=30.0,
        render_settings={},
        pixel_format=PIXEL_FORMAT_BGR,
    )


def _behavioral_result(session: Path) -> OverlayQcExportResult:
    job = session / "results_exports" / "overlay_qc_behavioral__test"
    job.mkdir(parents=True, exist_ok=True)
    video = job / "behavioral_qc.mp4"
    video.write_bytes(b"x")
    manifest_path = job / "overlay_artifact_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    return OverlayQcExportResult(
        job_dir=job,
        video_path=video,
        manifest_path=manifest_path,
        manifest=_manifest(),
        frames_written=30,
        output_fps=30.0,
        unavailable_layers=(),
    )


def _s4_result(session: Path) -> OverlayS4PoseQcExportResult:
    job = session / "results_exports" / "overlay_qc_s4_pose__test"
    job.mkdir(parents=True, exist_ok=True)
    video = job / "s4_pose_qc.mp4"
    video.write_bytes(b"x")
    manifest_path = job / "overlay_artifact_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    return OverlayS4PoseQcExportResult(
        job_dir=job,
        video_path=video,
        manifest_path=manifest_path,
        manifest=_manifest(),
        frames_written=30,
        output_fps=30.0,
        unavailable_layers=(),
        qc_status="completed",
    )


def test_s4_versus_behavioral_settings_and_selection(tmp_path: Path) -> None:
    behavioral = OverlayQcUiSettings(
        export_family="behavioral",
        source_kind="s8_automated",
        scope_kind="prepared_interval",
        prepared_frame_idx_start=10,
        prepared_frame_idx_end=20,
        hud_mode="both",
        include_contact=True,
        include_sbs=False,
    ).validated()
    assert behavioral.export_family == "behavioral"
    assert behavioral.include_sbs is False

    s4 = OverlayQcUiSettings(
        export_family="s4_pose",
        prepared_frame_idx_start=971,
        prepared_frame_idx_end=1000,
        draw_skeleton=True,
        include_frame_counter=True,
    ).validated()
    assert s4.export_family == "s4_pose"
    assert s4.draw_skeleton is True

    with pytest.raises(ValueError, match="start"):
        OverlayQcUiSettings(
            export_family="behavioral",
            prepared_frame_idx_start=20,
            prepared_frame_idx_end=10,
        ).validated()


def test_automated_versus_published_and_draft_rejection(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    package = session / "pkg"
    package.mkdir()
    draft = session / "draft"
    draft.mkdir()
    ctrl = OverlayQcExportController(
        behavioral_exporter=MagicMock(return_value=_behavioral_result(session)),
        s4_exporter=MagicMock(return_value=_s4_result(session)),
    )
    ctrl.set_session(session)

    reason = ctrl.cannot_export_reason(
        session_root=session,
        settings=OverlayQcUiSettings(
            export_family="behavioral", source_kind="s10_reviewed"
        ),
        package_dir=package,
        package_run_id="pkg",
        review_run_dir=draft,
        review_run_id="draft",
        review_is_draft=True,
    )
    assert reason is not None
    assert "draft" in reason.lower()

    # Automated must refuse silent review merge.
    reason = ctrl.cannot_export_reason(
        session_root=session,
        settings=OverlayQcUiSettings(
            export_family="behavioral", source_kind="s8_automated"
        ),
        package_dir=package,
        package_run_id="pkg",
        review_run_dir=draft,
        review_run_id="draft",
        review_is_draft=False,
    )
    assert reason is not None
    assert "silent merge" in reason.lower()

    # Explicit S4 requires run selection.
    reason = ctrl.cannot_export_reason(
        session_root=session,
        settings=OverlayQcUiSettings(export_family="s4_pose"),
        s4_run_dir=None,
        s4_run_id=None,
    )
    assert reason is not None
    assert "s4" in reason.lower()


def test_settings_propagation_into_identity(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    package = session / "pkg"
    package.mkdir()
    ctrl = OverlayQcExportController(
        behavioral_exporter=MagicMock(return_value=_behavioral_result(session))
    )
    ctrl.set_session(session)
    settings = OverlayQcUiSettings(
        export_family="behavioral",
        source_kind="s8_automated",
        scope_kind="bout_span",
        bout_id="bout_1",
        start_sample_idx=100,
        end_sample_idx=160,
        hud_mode="focal_0_only",
        include_contact=False,
        include_sbs=True,
        pre_context_sec=0.25,
        post_context_sec=0.5,
    )
    identity = ctrl.begin_export_launch(
        session_root=session,
        settings=settings,
        package_dir=package,
        package_run_id="pkg",
    )
    assert identity.settings.scope_kind == "bout_span"
    assert identity.settings.bout_id == "bout_1"
    assert identity.settings.hud_mode == "focal_0_only"
    assert identity.settings.include_contact is False
    assert identity.settings.pre_context_sec == 0.25
    assert identity.package_dir == package.resolve(strict=False)


def test_launch_conflict_and_stale_callback_rejection(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    package = session / "pkg"
    package.mkdir()
    ctrl = OverlayQcExportController(
        behavioral_exporter=MagicMock(return_value=_behavioral_result(session))
    )
    ctrl.set_session(session)
    identity = ctrl.begin_export_launch(
        session_root=session,
        settings=OverlayQcUiSettings(export_family="behavioral"),
        package_dir=package,
        package_run_id="pkg",
    )
    assert ctrl.state.task_running is True
    with pytest.raises(OverlayQcExportUiError, match="already running"):
        ctrl.begin_export_launch(
            session_root=session,
            settings=OverlayQcUiSettings(export_family="behavioral"),
            package_dir=package,
            package_run_id="pkg",
        )

    # Stale success after identity cleared.
    ctrl.state.active_identity = None
    assert (
        ctrl.record_export_success(
            _behavioral_result(session), expected_identity=identity
        )
        is False
    )


def test_workspace_retarget_blocked_while_running(tmp_path: Path) -> None:
    session_a = tmp_path / "a"
    session_b = tmp_path / "b"
    session_a.mkdir()
    session_b.mkdir()
    package = session_a / "pkg"
    package.mkdir()
    ctrl = OverlayQcExportController(
        behavioral_exporter=MagicMock(return_value=_behavioral_result(session_a))
    )
    ctrl.set_session(session_a)
    ctrl.begin_export_launch(
        session_root=session_a,
        settings=OverlayQcUiSettings(export_family="behavioral"),
        package_dir=package,
        package_run_id="pkg",
    )
    with pytest.raises(OverlayQcExportUiError, match="retarget"):
        ctrl.set_session(session_b)


def test_cancellation_outcome_and_discovery_rows(tmp_path: Path) -> None:
    session = tmp_path / "session"
    exports = session / "results_exports"
    job = exports / "overlay_qc_behavioral__done"
    job.mkdir(parents=True)
    (job / "behavioral_qc.mp4").write_bytes(b"x")
    (job / "overlay_artifact_manifest.json").write_text("{}", encoding="utf-8")
    (job / "run_meta.json").write_text(
        '{"source_kind":"s8_automated","package_run_id":"pkg","frames_written":12,'
        '"plan":{"selected_layers":["s8.contact"],'
        '"prepared_interval":{"prepared_frame_idx_start_inclusive":1,'
        '"prepared_frame_idx_end_inclusive":12}}}',
        encoding="utf-8",
    )
    # Naked media export must not appear.
    (exports / "media_export__ignore").mkdir()

    ctrl = OverlayQcExportController()
    summary = ctrl.set_session(session)
    assert summary is not None
    assert any(j.job_id == job.name for j in summary.jobs)
    assert all(not j.job_id.startswith("media_export") for j in summary.jobs)
    rows = ctrl.job_table_rows()
    assert any(row[0] == job.name for row in rows)

    package = session / "pkg"
    package.mkdir()
    identity = ctrl.begin_export_launch(
        session_root=session,
        settings=OverlayQcUiSettings(export_family="behavioral"),
        package_dir=package,
        package_run_id="pkg",
    )
    assert ctrl.record_export_failure(
        "cancelled by operator", expected_identity=identity, canceled=True
    )
    assert ctrl.state.last_outcome == STATUS_CANCELED
    assert ctrl.state.task_running is False


def test_execute_pins_output_root_to_bound_session(tmp_path: Path) -> None:
    """Isolated package copies must not publish into package.session_root."""

    session = tmp_path / "isolated_session"
    foreign = tmp_path / "lab_session"
    session.mkdir()
    foreign.mkdir()
    package = session / "pkg"
    package.mkdir()
    captured: dict[str, Path] = {}

    def _behavioral(**kwargs):  # noqa: ANN003
        captured["output_root"] = Path(kwargs["output_root"])
        return _behavioral_result(session)

    def _s4(**kwargs):  # noqa: ANN003
        captured["output_root"] = Path(kwargs["output_root"])
        return _s4_result(session)

    ctrl = OverlayQcExportController(
        behavioral_exporter=_behavioral,
        s4_exporter=_s4,
    )
    ctrl.set_session(session)

    import ui.controllers.overlay_qc_export_controller as mod

    class _Probe:
        fps = 30.0
        n_frames = 100

    fake_source = MagicMock()
    fake_source.video_probe = _Probe()
    fake_source.session_root = foreign  # package would point at lab session
    original_resolve = mod.resolve_overlay_qc_source
    original_plan = mod.plan_prepared_interval
    mod.resolve_overlay_qc_source = MagicMock(return_value=fake_source)
    mod.plan_prepared_interval = MagicMock(return_value=MagicMock())
    try:
        identity = ctrl.begin_export_launch(
            session_root=session,
            settings=OverlayQcUiSettings(
                export_family="behavioral",
                prepared_frame_idx_start=0,
                prepared_frame_idx_end=5,
            ),
            package_dir=package,
            package_run_id="pkg",
        )
        ctrl.execute_export_launch(identity)
    finally:
        mod.resolve_overlay_qc_source = original_resolve
        mod.plan_prepared_interval = original_plan

    assert Path(captured["output_root"]) == session / "results_exports"


def test_execute_routes_s4_and_behavioral_exporters(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    package = session / "pkg"
    package.mkdir()
    s4 = session / "s4"
    s4.mkdir()

    behavioral_mock = MagicMock(return_value=_behavioral_result(session))
    s4_mock = MagicMock(return_value=_s4_result(session))

    # Patch resolve/plan by injecting exporters only — resolve still needs real source.
    # For unit routing we override execute helpers by wrapping controller methods.
    ctrl = OverlayQcExportController(
        behavioral_exporter=behavioral_mock,
        s4_exporter=s4_mock,
    )
    ctrl.set_session(session)

    # Behavioral path: mock resolve + plan by stubbing execute internals via side effect.
    identity_b = ctrl.begin_export_launch(
        session_root=session,
        settings=OverlayQcUiSettings(
            export_family="behavioral",
            prepared_frame_idx_start=0,
            prepared_frame_idx_end=5,
        ),
        package_dir=package,
        package_run_id="pkg",
    )

    class _Probe:
        fps = 30.0
        n_frames = 100
        width = 64
        height = 64

    fake_source = MagicMock()
    fake_source.video_probe = _Probe()
    fake_source.sync = MagicMock()

    import ui.controllers.overlay_qc_export_controller as mod

    monkey_resolve = MagicMock(return_value=fake_source)
    monkey_plan = MagicMock(return_value=MagicMock())
    original_resolve = mod.resolve_overlay_qc_source
    original_plan = mod.plan_prepared_interval
    mod.resolve_overlay_qc_source = monkey_resolve
    mod.plan_prepared_interval = monkey_plan
    try:
        result = ctrl.execute_export_launch(identity_b, progress_callback=None)
    finally:
        mod.resolve_overlay_qc_source = original_resolve
        mod.plan_prepared_interval = original_plan
    assert isinstance(result, OverlayQcExportResult)
    behavioral_mock.assert_called_once()
    assert ctrl.record_export_success(result, expected_identity=identity_b)

    ctrl2 = OverlayQcExportController(
        behavioral_exporter=behavioral_mock,
        s4_exporter=s4_mock,
    )
    ctrl2.set_session(session)
    identity_s4 = ctrl2.begin_export_launch(
        session_root=session,
        settings=OverlayQcUiSettings(
            export_family="s4_pose",
            prepared_frame_idx_start=0,
            prepared_frame_idx_end=5,
        ),
        s4_run_dir=s4,
        s4_run_id="s4",
    )
    fake_s4_source = MagicMock()
    fake_s4_source.video_probe = _Probe()
    original_s4_resolve = mod.resolve_overlay_s4_qc_source
    original_s4_plan = mod.plan_s4_prepared_interval
    mod.resolve_overlay_s4_qc_source = MagicMock(return_value=fake_s4_source)
    mod.plan_s4_prepared_interval = MagicMock(return_value=MagicMock())
    try:
        result_s4 = ctrl2.execute_export_launch(identity_s4)
    finally:
        mod.resolve_overlay_s4_qc_source = original_s4_resolve
        mod.plan_s4_prepared_interval = original_s4_plan
    assert isinstance(result_s4, OverlayS4PoseQcExportResult)
    s4_mock.assert_called_once()


def test_bout_combo_limit_and_outside_250_prefill(tmp_path: Path) -> None:
    import pandas as pd

    from ui.controllers.overlay_qc_export_controller import (
        BOUT_COMBO_LIMIT,
        OverlayQcExportController,
        load_bout_options_from_package,
        lookup_bout_option_from_package,
    )

    session = tmp_path / "session"
    package = session / "results_package" / "pkg"
    package.mkdir(parents=True)
    rows = []
    for i in range(BOUT_COMBO_LIMIT + 5):
        rows.append(
            {
                "bout_id": f"bout_{i:04d}",
                "channel": "exclusive",
                "behavior_name": "approach",
                "start_sample_idx": i * 10,
                "end_sample_idx": i * 10 + 5,
                "prepared_frame_idx_start": i * 8,
                "prepared_frame_idx_end": i * 8 + 4,
            }
        )
    pd.DataFrame(rows).to_parquet(package / "behavior_bouts.parquet")

    listed = load_bout_options_from_package(package)
    assert len(listed) == BOUT_COMBO_LIMIT
    outside_id = f"bout_{BOUT_COMBO_LIMIT + 2:04d}"
    assert all(opt.bout_id != outside_id for opt in listed)
    looked = lookup_bout_option_from_package(package, bout_id=outside_id)
    assert looked is not None
    assert looked.bout_id == outside_id

    ctrl = OverlayQcExportController()
    ctrl.set_session(session)
    ctrl.refresh_bout_options(package)
    assert ctrl.state.bout_options_truncated is True
    assert ctrl.state.bout_options_total_exclusive == BOUT_COMBO_LIMIT + 5
    ensured = ctrl.ensure_preferred_bout_option(
        package, preferred_bout_id=outside_id
    )
    assert ensured is not None
    assert ctrl.state.bout_options[0].bout_id == outside_id

    launch = OverlayQcLaunchContext(
        session_root=session,
        export_family="behavioral",
        package_dir=package,
        package_run_id="pkg",
        lock_family=True,
        package_options=(("pkg", package),),
        preferred_bout_id=outside_id,
    )
    dialog = OverlayQcExportDialog(launch=launch)
    dialog.show()
    assert dialog.bout_combo.currentData() == outside_id
    assert outside_id in dialog.bout_limit_label.text() or "first" in dialog.bout_limit_label.text().lower()
    assert dialog.start_sample_spin.value() == looked.start_sample_idx
    assert dialog.end_sample_spin.value() == looked.end_sample_idx
    dialog.close()


def test_worker_nonblocking_task_kind_and_dialog_prefill(tmp_path: Path) -> None:
    assert GuiTaskKind.OVERLAY_QC_EXPORT.value == "overlay_qc_export"
    session = tmp_path / "session"
    session.mkdir()
    package = session / "results_package" / "results_package__demo"
    package.mkdir(parents=True)
    s4 = session / "pose_finalization" / "pose_finalization__demo"
    s4.mkdir(parents=True)
    launch = OverlayQcLaunchContext(
        session_root=session,
        export_family="behavioral",
        package_dir=package,
        package_run_id="results_package__demo",
        lock_family=True,
        package_options=(("results_package__demo", package),),
        s4_options=(("pose_finalization__demo", s4),),
        preferred_prepared_start=5,
        preferred_prepared_end=15,
    )
    dialog = OverlayQcExportDialog(launch=launch)
    dialog.show()
    assert dialog.family_combo.currentData() == "behavioral"
    assert dialog.family_combo.isEnabled() is False
    assert dialog.prep_start_spin.value() == 5
    assert dialog.prep_end_spin.value() == 15
    assert dialog.task_runner.is_running is False
    dialog.close()


def test_actionable_errors_and_cancel_mapping() -> None:
    assert "draft" in actionable_overlay_error(RuntimeError("working draft")).lower()
    assert "digest" in actionable_overlay_error(
        RuntimeError("digest mismatch for sync")
    ).lower()
    assert "ffmpeg" in actionable_overlay_error(RuntimeError("ffmpeg failed")).lower()
    assert "canceled" in actionable_overlay_error(
        OverlayExportCancelled("cancelled by caller")
    ).lower()


def test_stale_session_mismatch_rejects_success(tmp_path: Path) -> None:
    session_a = tmp_path / "a"
    session_b = tmp_path / "b"
    session_a.mkdir()
    session_b.mkdir()
    package = session_a / "pkg"
    package.mkdir()
    ctrl = OverlayQcExportController(
        behavioral_exporter=MagicMock(return_value=_behavioral_result(session_a))
    )
    ctrl.set_session(session_a)
    identity = ctrl.begin_export_launch(
        session_root=session_a,
        settings=OverlayQcUiSettings(export_family="behavioral"),
        package_dir=package,
        package_run_id="pkg",
    )
    # Simulate session retarget race: identity still active but bound session changed.
    ctrl.state.bound_session_root = session_b.resolve(strict=False)
    assert (
        ctrl.record_export_success(
            _behavioral_result(session_a), expected_identity=identity
        )
        is False
    )
