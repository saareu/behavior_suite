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
from overlay.export_s2 import OverlayS2PoseQcExportResult
from overlay.provenance import OverlayArtifactManifest
from ui.controllers.overlay_qc_export_controller import (
    OverlayQcExportController,
    OverlayQcExportUiError,
)
from ui.controllers.overlay_qc_export_support import (
    OverlayQcLaunchContext,
    OverlayQcUiSettings,
    actionable_overlay_error,
    plan_s2_pose_overlay_launch,
    plan_s3_pose_overlay_launch,
    plan_s8_behavioral_overlay_launch,
)
from ui.dialogs.overlay_qc_export_dialog import OverlayQcExportDialog
from ui.pages.behavioral_arbitration_page import BehavioralArbitrationPage
from ui.pages.behavioral_review_page import BehavioralReviewPage
from ui.pages.pose_finalization_page import PoseFinalizationPage
from ui.pages.pose_inference_page import PoseInferencePage
from ui.pages.results_packaging_page import ResultsPackagingPage
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


def _s2_result(session: Path) -> OverlayS2PoseQcExportResult:
    job = session / "results_exports" / "overlay_qc_s2_pose__test"
    job.mkdir(parents=True, exist_ok=True)
    video = job / "s2_pose_qc.mp4"
    video.write_bytes(b"x")
    manifest_path = job / "overlay_artifact_manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    return OverlayS2PoseQcExportResult(
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


def _s9_option(
    tmp_path: Path,
    *,
    run_id: str,
    source_s8_run_id: str | None,
) -> MagicMock:
    run = MagicMock()
    run.run_id = run_id
    run.run_dir = tmp_path / "results_package" / run_id
    run.run_dir.mkdir(parents=True, exist_ok=True)
    run.source_s8_run_id = source_s8_run_id
    return run


def test_s8_page_overlay_button_visible_and_launches(tmp_path: Path, monkeypatch) -> None:
    page = BehavioralArbitrationPage()
    assert page.overlay_qc_button.text() == "Export QC Overlay Video…"
    assert page.overlay_qc_button.isVisibleTo(page) or page.overlay_qc_button.isVisible()

    session = tmp_path / "session"
    session.mkdir()
    page.controller.state.session_root = session
    matching = _s9_option(
        session, run_id="results_package__match", source_s8_run_id="behavioral_inference__s8a"
    )
    selected = MagicMock()
    selected.run_id = "behavioral_inference__s8a"
    monkeypatch.setattr(page.controller, "selected_s8_run", lambda: selected)
    monkeypatch.setattr(
        "results_packaging.run_discovery.summarize_results_package_project",
        lambda _root: MagicMock(eligible_runs=(matching,)),
    )
    opened: list[OverlayQcLaunchContext] = []

    def _fake_open(_parent, *, launch, controller=None):
        opened.append(launch)
        return MagicMock()

    monkeypatch.setattr(
        "ui.dialogs.overlay_qc_export_dialog.open_overlay_qc_export_dialog",
        _fake_open,
    )
    monkeypatch.setattr(
        "ui.pages.behavioral_arbitration_page.QMessageBox.warning",
        staticmethod(lambda *a, **k: 0),
    )
    monkeypatch.setattr(
        "ui.pages.behavioral_arbitration_page.QMessageBox.information",
        staticmethod(lambda *a, **k: 0),
    )
    page._open_overlay_qc_export()
    assert len(opened) == 1
    launch = opened[0]
    assert launch.export_family == "behavioral"
    assert launch.lock_family is True
    assert launch.session_root == session.resolve()
    assert launch.package_run_id == "results_package__match"
    assert launch.package_dir == matching.run_dir
    assert launch.review_run_id is None
    assert launch.review_run_dir is None
    assert launch.review_options == ()
    assert launch.lock_source_kind is True
    assert launch.locked_source_kind == "s8_automated"


def test_plan_s8_behavioral_overlay_launch_prefill_and_eligibility(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    match_a = _s9_option(
        session, run_id="results_package__a", source_s8_run_id="s8_run_a"
    )
    match_b = _s9_option(
        session, run_id="results_package__b", source_s8_run_id="s8_run_a"
    )
    other = _s9_option(
        session, run_id="results_package__other", source_s8_run_id="s8_run_other"
    )

    blocked = plan_s8_behavioral_overlay_launch(
        session_root=session,
        selected_s8_run_id="s8_run_a",
        eligible_s9_runs=(),
    )
    assert blocked.blocked is True
    assert blocked.launch is None
    assert blocked.block_message is not None
    assert "S9" in blocked.block_message
    assert "pins this S8" in blocked.block_message

    unambiguous = plan_s8_behavioral_overlay_launch(
        session_root=session,
        selected_s8_run_id="s8_run_a",
        eligible_s9_runs=(match_a, other),
    )
    assert unambiguous.blocked is False
    assert unambiguous.launch is not None
    assert unambiguous.launch.export_family == "behavioral"
    assert unambiguous.launch.lock_family is True
    assert unambiguous.launch.lock_source_kind is True
    assert unambiguous.launch.locked_source_kind == "s8_automated"
    assert unambiguous.launch.package_run_id == "results_package__a"
    assert unambiguous.launch.package_options == (
        ("results_package__a", match_a.run_dir),
    )
    assert unambiguous.launch.review_options == ()
    assert unambiguous.launch.review_run_id is None
    assert unambiguous.launch.review_run_dir is None

    ambiguous = plan_s8_behavioral_overlay_launch(
        session_root=session,
        selected_s8_run_id="s8_run_a",
        eligible_s9_runs=(match_a, match_b, other),
    )
    assert ambiguous.launch is not None
    assert ambiguous.launch.package_run_id is None
    assert ambiguous.launch.package_dir is None
    assert ambiguous.launch.package_options == (
        ("results_package__a", match_a.run_dir),
        ("results_package__b", match_b.run_dir),
    )
    # Unrelated S9 for a different S8 must not appear when an S8 is selected.
    assert all(run_id != "results_package__other" for run_id, _ in ambiguous.launch.package_options)

    no_selection = plan_s8_behavioral_overlay_launch(
        session_root=session,
        selected_s8_run_id=None,
        eligible_s9_runs=(match_a, other),
    )
    assert no_selection.launch is not None
    assert no_selection.launch.package_run_id is None
    assert {run_id for run_id, _ in no_selection.launch.package_options} == {
        "results_package__a",
        "results_package__other",
    }

    empty_session = plan_s8_behavioral_overlay_launch(
        session_root=session,
        selected_s8_run_id=None,
        eligible_s9_runs=(),
    )
    assert empty_session.blocked is True
    assert empty_session.block_message is not None
    assert "eligible S9" in empty_session.block_message


def test_s8_launch_dialog_defaults_both_focal_and_contact_sbs(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    package = session / "results_package" / "results_package__demo"
    package.mkdir(parents=True)
    plan = plan_s8_behavioral_overlay_launch(
        session_root=session,
        selected_s8_run_id="s8_demo",
        eligible_s9_runs=(
            _s9_option(
                session,
                run_id="results_package__demo",
                source_s8_run_id="s8_demo",
            ),
        ),
    )
    assert plan.launch is not None
    dialog = OverlayQcExportDialog(launch=plan.launch)
    dialog.show()
    assert dialog.family_combo.currentData() == "behavioral"
    assert dialog.family_combo.isEnabled() is False
    assert dialog.source_kind_combo.currentData() == "s8_automated"
    assert dialog.source_kind_combo.isEnabled() is False
    assert dialog.source_kind_combo.count() == 1
    assert dialog.source_kind_combo.findData("s10_reviewed") < 0
    assert dialog.review_combo.isEnabled() is False
    assert dialog.hud_combo.currentData() == "both"
    assert dialog.include_contact.isChecked() is True
    assert dialog.include_sbs.isChecked() is True
    settings = dialog._read_settings()
    assert settings.hud_mode == "both"
    assert settings.include_contact is True
    assert settings.include_sbs is True
    assert settings.source_kind == "s8_automated"
    dialog.close()


def test_s8_entry_cannot_switch_to_reviewed_s10(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    match_a = _s9_option(
        session, run_id="results_package__a", source_s8_run_id="s8_run_a"
    )
    match_b = _s9_option(
        session, run_id="results_package__b", source_s8_run_id="s8_run_a"
    )
    other = _s9_option(
        session, run_id="results_package__other", source_s8_run_id="s8_run_other"
    )
    plan = plan_s8_behavioral_overlay_launch(
        session_root=session,
        selected_s8_run_id="s8_run_a",
        eligible_s9_runs=(match_a, match_b, other),
    )
    assert plan.launch is not None
    assert plan.launch.lock_source_kind is True
    assert plan.launch.package_run_id is None
    assert plan.launch.package_options == (
        ("results_package__a", match_a.run_dir),
        ("results_package__b", match_b.run_dir),
    )

    dialog = OverlayQcExportDialog(launch=plan.launch)
    dialog.show()
    assert dialog.source_kind_combo.findData("s10_reviewed") < 0
    assert dialog.source_kind_combo.isEnabled() is False
    assert dialog.review_combo.isEnabled() is False
    # Explicit package choice among S8-filtered options remains available.
    assert dialog.package_combo.isEnabled() is True
    package_ids = {
        None
        if dialog.package_combo.itemData(i) is None
        else dialog.package_combo.itemData(i)[0]
        for i in range(dialog.package_combo.count())
    }
    assert "results_package__a" in package_ids
    assert "results_package__b" in package_ids
    assert "results_package__other" not in package_ids

    # Even if the combo were forced, settings stay locked to automated S8.
    dialog.source_kind_combo.blockSignals(True)
    dialog.source_kind_combo.clear()
    dialog.source_kind_combo.addItem("Published S10 reviewed", "s10_reviewed")
    dialog.source_kind_combo.setCurrentIndex(0)
    dialog.source_kind_combo.blockSignals(False)
    settings = dialog._read_settings()
    assert settings.source_kind == "s8_automated"
    dialog.close()


def test_s9_and_s10_entry_retain_source_selection(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    package = session / "results_package" / "results_package__demo"
    package.mkdir(parents=True)
    review = session / "behavioral_review" / "behavioral_review__pub"
    review.mkdir(parents=True)

    s9_launch = OverlayQcLaunchContext(
        session_root=session,
        export_family="behavioral",
        package_dir=package,
        package_run_id="results_package__demo",
        lock_family=True,
        package_options=(("results_package__demo", package),),
        preferred_prepared_start=0,
        preferred_prepared_end=29,
    )
    assert s9_launch.lock_source_kind is False
    s9_dialog = OverlayQcExportDialog(launch=s9_launch)
    s9_dialog.show()
    assert s9_dialog.source_kind_combo.isEnabled() is True
    assert s9_dialog.source_kind_combo.findData("s8_automated") >= 0
    assert s9_dialog.source_kind_combo.findData("s10_reviewed") >= 0
    s9_dialog.source_kind_combo.setCurrentIndex(
        s9_dialog.source_kind_combo.findData("s10_reviewed")
    )
    assert s9_dialog.source_kind_combo.currentData() == "s10_reviewed"
    assert s9_dialog.review_combo.isEnabled() is True
    assert s9_dialog._read_settings().source_kind == "s10_reviewed"
    s9_dialog.close()

    s10_launch = OverlayQcLaunchContext(
        session_root=session,
        export_family="behavioral",
        package_dir=package,
        package_run_id="results_package__demo",
        review_run_dir=review,
        review_run_id="behavioral_review__pub",
        review_is_draft=False,
        lock_family=False,
        package_options=(("results_package__demo", package),),
        review_options=(
            ("behavioral_review__pub", review, False, "results_package__demo"),
        ),
    )
    assert s10_launch.lock_source_kind is False
    s10_dialog = OverlayQcExportDialog(launch=s10_launch)
    s10_dialog.show()
    assert s10_dialog.source_kind_combo.isEnabled() is True
    assert s10_dialog.source_kind_combo.currentData() == "s10_reviewed"
    assert s10_dialog.source_kind_combo.findData("s8_automated") >= 0
    assert s10_dialog.review_combo.isEnabled() is True
    s10_dialog.source_kind_combo.setCurrentIndex(
        s10_dialog.source_kind_combo.findData("s8_automated")
    )
    assert s10_dialog.source_kind_combo.currentData() == "s8_automated"
    assert s10_dialog._read_settings().source_kind == "s8_automated"
    s10_dialog.close()


def test_s8_opener_blocks_without_eligible_s9(tmp_path: Path, monkeypatch) -> None:
    page = BehavioralArbitrationPage()
    session = tmp_path / "session"
    session.mkdir()
    page.controller.state.session_root = session
    selected = MagicMock()
    selected.run_id = "s8_missing_pkg"
    monkeypatch.setattr(page.controller, "selected_s8_run", lambda: selected)
    monkeypatch.setattr(
        "results_packaging.run_discovery.summarize_results_package_project",
        lambda _root: MagicMock(eligible_runs=()),
    )
    messages: list[tuple[str, str]] = []

    def _info(_parent, title, text):
        messages.append((str(title), str(text)))
        return 0

    monkeypatch.setattr(
        "ui.pages.behavioral_arbitration_page.QMessageBox.information",
        staticmethod(_info),
    )
    opened: list[object] = []
    monkeypatch.setattr(
        "ui.dialogs.overlay_qc_export_dialog.open_overlay_qc_export_dialog",
        lambda *a, **k: opened.append(k.get("launch")),
    )
    page._open_overlay_qc_export()
    assert opened == []
    assert messages
    assert "S9" in messages[0][1]


def test_existing_s4_s9_s10_overlay_entry_points_unchanged() -> None:
    s4 = PoseFinalizationPage()
    s9 = ResultsPackagingPage()
    s10 = BehavioralReviewPage()
    assert s4.overlay_qc_button.text() == "Export QC Overlay Video…"
    assert s9.overlay_qc_button.text() == "Export QC Overlay Video…"
    assert s10.overlay_qc_button.text() == "Open QC Overlay Export…"
    assert callable(s4._open_overlay_qc_export)
    assert callable(s9._open_overlay_qc_export)
    assert callable(s10._open_overlay_qc_export)


def test_s2_settings_and_explicit_selection_gate(tmp_path: Path) -> None:
    s2 = OverlayQcUiSettings(
        export_family="s2_pose",
        prepared_frame_idx_start=0,
        prepared_frame_idx_end=29,
        draw_skeleton=True,
        show_confidence=True,
        include_frame_counter=True,
    ).validated()
    assert s2.export_family == "s2_pose"
    assert s2.show_confidence is True

    session = tmp_path / "session"
    session.mkdir()
    s2_dir = session / "pose_inference" / "bottomup__demo"
    s2_dir.mkdir(parents=True)
    ctrl = OverlayQcExportController(
        s2_exporter=MagicMock(return_value=_s2_result(session)),
    )
    ctrl.set_session(session)
    reason = ctrl.cannot_export_reason(
        session_root=session,
        settings=OverlayQcUiSettings(export_family="s2_pose"),
        s2_run_dir=None,
        s2_run_id=None,
    )
    assert reason is not None
    assert "S2" in reason

    identity = ctrl.begin_export_launch(
        session_root=session,
        settings=s2,
        s2_run_dir=s2_dir,
        s2_run_id="bottomup__demo",
    )
    assert identity.export_family == "s2_pose"
    assert identity.s2_run_id == "bottomup__demo"
    assert identity.s2_run_dir == s2_dir.resolve(strict=False)


def test_s2_dialog_family_launch_and_confidence_option(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    s2_a = session / "pose_inference" / "bottomup__a"
    s2_b = session / "pose_inference" / "bottomup__b"
    s2_a.mkdir(parents=True)
    s2_b.mkdir(parents=True)
    launch = plan_s2_pose_overlay_launch(
        session_root=session,
        selected_s2_run_id="bottomup__a",
        selected_s2_run_dir=s2_a,
        eligible_s2_runs=(("bottomup__a", s2_a), ("bottomup__b", s2_b)),
    )
    assert launch.export_family == "s2_pose"
    assert launch.lock_family is True
    assert launch.s2_run_id == "bottomup__a"
    assert launch.skeleton_nodes_only_disclosure

    dialog = OverlayQcExportDialog(launch=launch)
    dialog.show()
    assert dialog.family_combo.currentData() == "s2_pose"
    assert dialog.family_combo.isEnabled() is False
    assert dialog.s2_combo.isEnabled() is True
    assert dialog.s4_combo.isEnabled() is False
    assert dialog.draw_skeleton.isEnabled() is True
    assert dialog.show_confidence.isEnabled() is True
    dialog.show_confidence.setChecked(True)
    settings = dialog._read_settings()
    assert settings.export_family == "s2_pose"
    assert settings.show_confidence is True
    dialog.close()


def test_s2_page_entry_button_present() -> None:
    from ui.controllers.pose_inference_controller import PoseInferenceController

    page = PoseInferencePage(PoseInferenceController())
    assert page.overlay_qc_button.text() == "Export QC Overlay Video…"
    assert callable(page._open_overlay_qc_export)


def test_s2_opener_prefills_only_when_unambiguous(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    only = session / "pose_inference" / "bottomup__only"
    only.mkdir(parents=True)
    launch = plan_s2_pose_overlay_launch(
        session_root=session,
        selected_s2_run_id=None,
        selected_s2_run_dir=None,
        eligible_s2_runs=(("bottomup__only", only),),
    )
    assert launch.s2_run_id == "bottomup__only"

    a = session / "pose_inference" / "bottomup__a"
    b = session / "pose_inference" / "bottomup__b"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    ambiguous = plan_s2_pose_overlay_launch(
        session_root=session,
        selected_s2_run_id=None,
        selected_s2_run_dir=None,
        eligible_s2_runs=(("bottomup__a", a), ("bottomup__b", b)),
    )
    assert ambiguous.s2_run_id is None
    assert len(ambiguous.s2_options) == 2


def test_s3_dialog_family_launch_and_event_toggles(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    s3_a = session / "tracking_correction" / "s3__a"
    s3_b = session / "tracking_correction" / "s3__b"
    s3_a.mkdir(parents=True)
    s3_b.mkdir(parents=True)
    plan = plan_s3_pose_overlay_launch(
        session_root=session,
        selected_s3_run_id="s3__a",
        selected_s3_run_dir=s3_a,
        selected_acceptance_state="accepted",
        accepted_s3_runs=(("s3__a", s3_a), ("s3__b", s3_b)),
    )
    assert plan.blocked is False
    assert plan.launch is not None
    launch = plan.launch
    assert launch.export_family == "s3_pose"
    assert launch.lock_family is True
    assert launch.s3_run_id == "s3__a"

    dialog = OverlayQcExportDialog(launch=launch)
    dialog.show()
    assert dialog.family_combo.currentData() == "s3_pose"
    assert dialog.family_combo.isEnabled() is False
    assert dialog.s3_combo.isEnabled() is True
    assert dialog.s2_combo.isEnabled() is False
    assert dialog.draw_skeleton.isEnabled() is True
    assert dialog.show_machine_corrections.isEnabled() is True
    assert dialog.show_manual_corrections.isEnabled() is True
    assert dialog.show_confidence.isEnabled() is False
    dialog.show_machine_corrections.setChecked(False)
    dialog.show_manual_corrections.setChecked(True)
    settings = dialog._read_settings()
    assert settings.export_family == "s3_pose"
    assert settings.show_machine_corrections is False
    assert settings.show_manual_corrections is True
    dialog.close()


def test_s3_unaccepted_run_blocks_with_actionable_message(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    s3 = session / "tracking_correction" / "s3__working"
    s3.mkdir(parents=True)
    plan = plan_s3_pose_overlay_launch(
        session_root=session,
        selected_s3_run_id="s3__working",
        selected_s3_run_dir=s3,
        selected_acceptance_state="not_accepted",
        accepted_s3_runs=(),
    )
    assert plan.blocked is True
    assert plan.launch is None
    assert plan.block_title == "S3 run not accepted"
    assert "accepted" in (plan.block_message or "").lower()

    # Page state claiming "accepted" is not enough when discovery omitted the run.
    claimed = session / "tracking_correction" / "s3__stale"
    claimed.mkdir(parents=True)
    stale = plan_s3_pose_overlay_launch(
        session_root=session,
        selected_s3_run_id="s3__stale",
        selected_s3_run_dir=claimed,
        selected_acceptance_state="accepted",
        accepted_s3_runs=(),
    )
    assert stale.blocked is True
    assert stale.launch is None


def test_s3_page_entry_button_present() -> None:
    from ui.controllers.tracking_review_controller import TrackingReviewController
    from ui.pages.tracking_review_page import TrackingReviewPage

    page = TrackingReviewPage(TrackingReviewController())
    assert page.export_qc_overlay_button.text() == "Export QC Overlay Video…"
    assert callable(page._open_overlay_qc_export)


def test_s5_dialog_family_launch_and_option_toggles(tmp_path: Path) -> None:
    from ui.controllers.overlay_qc_export_support import (
        plan_s5_segmentation_overlay_launch,
    )

    session = tmp_path / "session"
    session.mkdir()
    s5_a = session / "blob_features" / "blob_features__a"
    s5_b = session / "blob_features" / "blob_features__b"
    s5_a.mkdir(parents=True)
    s5_b.mkdir(parents=True)
    plan = plan_s5_segmentation_overlay_launch(
        session_root=session,
        selected_s5_run_id="blob_features__a",
        selected_s5_run_dir=s5_a,
        eligible_s5_runs=(("blob_features__a", s5_a), ("blob_features__b", s5_b)),
    )
    assert plan.blocked is False
    assert plan.launch is not None
    launch = plan.launch
    assert launch.export_family == "s5_segmentation"
    assert launch.lock_family is True
    assert launch.s5_run_id == "blob_features__a"

    dialog = OverlayQcExportDialog(launch=launch)
    dialog.show()
    assert dialog.family_combo.currentData() == "s5_segmentation"
    assert dialog.family_combo.isEnabled() is False
    assert dialog.s5_combo.isEnabled() is True
    assert dialog.s3_combo.isEnabled() is False
    assert dialog.show_contours.isEnabled() is True
    assert dialog.show_masks.isEnabled() is True
    assert dialog.show_scalar_text.isEnabled() is True
    assert dialog.draw_skeleton.isEnabled() is False
    dialog.show_contours.setChecked(True)
    dialog.show_masks.setChecked(True)
    dialog.show_scalar_text.setChecked(False)
    dialog.mask_opacity_spin.setValue(0.25)
    settings = dialog._read_settings()
    assert settings.export_family == "s5_segmentation"
    assert settings.show_contours is True
    assert settings.show_masks is True
    assert settings.mask_opacity == pytest.approx(0.25)
    dialog.close()


def test_s5_ineligible_run_blocks_with_actionable_message(tmp_path: Path) -> None:
    from ui.controllers.overlay_qc_export_support import (
        plan_s5_segmentation_overlay_launch,
    )

    session = tmp_path / "session"
    session.mkdir()
    s5 = session / "blob_features" / "blob_features__stale"
    s5.mkdir(parents=True)
    plan = plan_s5_segmentation_overlay_launch(
        session_root=session,
        selected_s5_run_id="blob_features__stale",
        selected_s5_run_dir=s5,
        eligible_s5_runs=(),
    )
    assert plan.blocked is True
    assert plan.launch is None
    assert plan.block_title == "S5 run not eligible"
    assert "eligible" in (plan.block_message or "").lower()

    none = plan_s5_segmentation_overlay_launch(
        session_root=session,
        selected_s5_run_id=None,
        selected_s5_run_dir=None,
        eligible_s5_runs=(),
    )
    assert none.blocked is True
    assert none.block_title == "No eligible S5 run"


def test_s5_page_entry_button_present() -> None:
    from ui.controllers.blob_features_controller import BlobFeaturesController
    from ui.pages.blob_features_page import BlobFeaturesPage

    page = BlobFeaturesPage(BlobFeaturesController())
    assert page.export_qc_overlay_button.text() == "Export QC Overlay Video…"
    assert callable(page._open_overlay_qc_export)


def test_mixed_settings_and_compatible_selection_gates(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    controller = OverlayQcExportController()
    controller.set_session(session)

    ok = OverlayQcUiSettings(
        export_family="mixed_pose_behavior",
        pose_source_kind="s4_finalized",
        source_kind="s8_automated",
        prepared_frame_idx_start=10,
        prepared_frame_idx_end=20,
        hud_mode="both",
        include_contact=True,
        include_sbs=True,
        draw_skeleton=True,
    ).validated()
    assert ok.export_family == "mixed_pose_behavior"
    assert ok.pose_source_kind == "s4_finalized"

    with pytest.raises(ValueError, match="S3\\+S10"):
        OverlayQcUiSettings(
            export_family="mixed_pose_behavior",
            pose_source_kind="s3_accepted",
            source_kind="s10_reviewed",
        ).validated()

    reason = controller.cannot_export_reason(
        session_root=session,
        settings=OverlayQcUiSettings(
            export_family="mixed_pose_behavior",
            pose_source_kind="s4_finalized",
            source_kind="s8_automated",
        ),
        package_dir=None,
        package_run_id=None,
        s4_run_dir=session / "pose_finalization" / "s4",
        s4_run_id="s4",
    )
    assert reason is not None
    assert "S9 package" in reason

    identity = controller.begin_export_launch(
        session_root=session,
        settings=OverlayQcUiSettings(
            export_family="mixed_pose_behavior",
            pose_source_kind="s4_finalized",
            source_kind="s8_automated",
            prepared_frame_idx_start=0,
            prepared_frame_idx_end=5,
        ),
        package_dir=session / "results_package__a",
        package_run_id="results_package__a",
        s4_run_dir=session / "pose_finalization" / "s4",
        s4_run_id="s4",
    )
    assert identity.export_family == "mixed_pose_behavior"
    assert identity.source_kind == "s8_automated"
    assert identity.s4_run_id == "s4"
    assert identity.package_run_id == "results_package__a"
    controller.record_export_failure("cleanup", expected_identity=identity)


def test_mixed_dialog_family_selection_and_workspace_safety(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    s4 = session / "pose_finalization" / "s4_a"
    s3 = session / "tracking_correction" / "s3_a"
    pkg = session / "results_package__a"
    s4.mkdir(parents=True)
    s3.mkdir(parents=True)
    pkg.mkdir(parents=True)
    launch = OverlayQcLaunchContext(
        session_root=session,
        export_family="mixed_pose_behavior",
        lock_family=False,
        package_dir=pkg,
        package_run_id="results_package__a",
        s4_run_dir=s4,
        s4_run_id="s4_a",
        s3_run_dir=s3,
        s3_run_id="s3_a",
        package_options=(("results_package__a", pkg),),
        s4_options=(("s4_a", s4),),
        s3_options=(("s3_a", s3),),
        preferred_prepared_start=5,
        preferred_prepared_end=15,
    )
    dialog = OverlayQcExportDialog(launch=launch)
    dialog.show()
    idx = dialog.family_combo.findData("mixed_pose_behavior")
    assert idx >= 0
    dialog.family_combo.setCurrentIndex(idx)
    assert dialog.pose_source_combo.isEnabled() is True
    assert dialog.draw_skeleton.isEnabled() is True
    assert dialog.hud_combo.isEnabled() is True
    assert dialog.include_contact.isEnabled() is True
    assert dialog.include_sbs.isEnabled() is True
    assert dialog.s4_provenance_label.isVisible() is True
    dialog.pose_source_combo.setCurrentIndex(
        dialog.pose_source_combo.findData("s3_accepted")
    )
    assert dialog.source_kind_combo.currentData() == "s8_automated"
    assert dialog.s3_combo.isEnabled() is True
    assert dialog.s4_combo.isEnabled() is False
    assert dialog.s4_provenance_label.isVisible() is False
    settings = dialog._read_settings().validated()
    assert settings.export_family == "mixed_pose_behavior"
    assert settings.pose_source_kind == "s3_accepted"
    assert settings.source_kind == "s8_automated"
    assert settings.prepared_frame_idx_start == 5
    assert settings.hud_mode == "both"

    # S4 mixed may select published S10; S3+S10 remains forced off.
    dialog.pose_source_combo.setCurrentIndex(
        dialog.pose_source_combo.findData("s4_finalized")
    )
    dialog.source_kind_combo.setCurrentIndex(
        dialog.source_kind_combo.findData("s10_reviewed")
    )
    assert dialog.source_kind_combo.currentData() == "s10_reviewed"
    assert dialog.review_combo.isEnabled() is True
    dialog.pose_source_combo.setCurrentIndex(
        dialog.pose_source_combo.findData("s3_accepted")
    )
    assert dialog.source_kind_combo.currentData() == "s8_automated"
    assert dialog.review_combo.isEnabled() is False

    # Missing pose selection is rejected with an explicit operator message.
    missing = dialog.controller.cannot_export_reason(
        session_root=session,
        settings=OverlayQcUiSettings(
            export_family="mixed_pose_behavior",
            pose_source_kind="s4_finalized",
            source_kind="s8_automated",
        ),
        package_dir=pkg,
        package_run_id="results_package__a",
        s4_run_dir=None,
        s4_run_id=None,
    )
    assert missing is not None
    assert "never auto-picks newest" in missing

    # Workspace safety: retarget blocked while a task is marked running.
    dialog.controller.state.task_running = True
    with pytest.raises(OverlayQcExportUiError, match="while a QC overlay export"):
        dialog.controller.set_session(tmp_path / "other")
    dialog.controller.state.task_running = False
    dialog.close()


def test_mixed_preselection_is_caller_attributed(tmp_path: Path) -> None:
    session = tmp_path / "session"
    session.mkdir()
    s4_a = session / "pose_finalization" / "s4_a"
    s4_b = session / "pose_finalization" / "s4_b"
    pkg = session / "results_package__a"
    for path in (s4_a, s4_b, pkg):
        path.mkdir(parents=True)
    # Explicit caller preselection of s4_b — never newest/fallback to s4_a.
    launch = OverlayQcLaunchContext(
        session_root=session,
        export_family="mixed_pose_behavior",
        lock_family=True,
        pose_source_kind="s4_finalized",
        package_dir=pkg,
        package_run_id="results_package__a",
        s4_run_dir=s4_b,
        s4_run_id="s4_b",
        package_options=(("results_package__a", pkg),),
        s4_options=(("s4_a", s4_a), ("s4_b", s4_b)),
        preferred_prepared_start=2,
        preferred_prepared_end=7,
    )
    dialog = OverlayQcExportDialog(launch=launch)
    dialog.show()
    assert dialog.family_combo.currentData() == "mixed_pose_behavior"
    assert dialog.family_combo.isEnabled() is False  # lock_family
    assert dialog.pose_source_combo.currentData() == "s4_finalized"
    assert dialog.s4_combo.currentData()[0] == "s4_b"
    assert dialog.s4_combo.currentData()[1] == s4_b
    settings = dialog._read_settings().validated()
    assert settings.pose_source_kind == "s4_finalized"
    assert settings.prepared_frame_idx_start == 2
    assert settings.prepared_frame_idx_end == 7
    dialog.close()
