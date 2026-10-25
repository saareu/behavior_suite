"""Subsystem 05 GUI: readiness, preview, invalidation, bookmarks, navigation."""

from __future__ import annotations

import json
import os
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
from PySide6.QtWidgets import QApplication

from blob_features import BlobSegmentationSettings, load_blob_features_config
from preprocess.sync_writer import build_prepared_sync, write_prepared_sync_npz
from ui.controllers.blob_features_controller import (
    BlobFeaturesController,
    BlobFeaturesUiError,
    BlobFeaturesUiSettings,
)
from ui.controllers.preprocess_setup_controller import PreprocessSetupController
from ui.main_window import MainWindow
from ui.pages.blob_features_page import BlobFeaturesPage

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_APPLICATION = QApplication.instance() or QApplication([])


def _write_eligible_s1(
    root: Path,
    *,
    frame_count: int = 6,
    fps: float = 20.0,
    size: tuple[int, int] = (64, 48),
) -> Path:
    session = root / "session"
    preprocess = session / "preprocess"
    preprocess.mkdir(parents=True)
    width, height = size
    video_path = preprocess / "prepared_video.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    assert writer.isOpened()
    background = np.full((height, width, 3), 220, dtype=np.uint8)
    for frame_idx in range(frame_count):
        frame = background.copy()
        # Two dark blobs for most frames; force fragmentation on frame 3.
        if frame_idx == 3:
            frame[8:20, 8:20] = 5
            frame[8:20, 28:40] = 5
            frame[8:20, 48:60] = 5
        else:
            frame[10:30, 8:28] = 5
            frame[10:30, 36:56] = 5
        writer.write(frame)
    writer.release()
    assert cv2.imwrite(str(preprocess / "cropped_background.png"), background)
    (preprocess / "prepare_meta.json").write_text(
        json.dumps(
            {
                "schema_version": "prepare_meta_v1",
                "prepared_video": {
                    "width": width,
                    "height": height,
                    "opencv_reported_frame_count": frame_count,
                    "opencv_readable_frame_count": frame_count,
                    "frame_count_used_for_sleap": frame_count,
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
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
    write_prepared_sync_npz(preprocess / "prepared_sync.npz", sync)
    return session


def _soft_settings() -> BlobFeaturesUiSettings:
    """Suite defaults with area thresholds lowered for synthetic fixtures."""

    config = load_blob_features_config()
    soft = BlobSegmentationSettings(
        blob_threshold=-1,
        blob_blur_kernel_size=5,
        blob_kernel_size=5,
        blob_tail_open_kernel_size=3,
        blob_min_area=20,
        blob_max_area=0,
        blob_prune_min_radius=1,
        blob_prune_regrow_iters=1,
        blob_prune_max_regrow_dist=2,
        blob_post_prune_min_area=20,
        blob_post_prune_max_area=0,
        blob_final_smooth_kernel_size=3,
        abs_dark_threshold=250,
    )
    return BlobFeaturesUiSettings(
        segmentation=soft,
        temporal=config.temporal,
        execution=config.execution,
    ).validated()


def _wait_until(predicate, timeout_sec: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while not predicate():
        _APPLICATION.processEvents()
        if time.monotonic() >= deadline:
            raise AssertionError("Timed out waiting for Qt state change")
        time.sleep(0.01)
    _APPLICATION.processEvents()


def test_controller_readiness_requires_eligible_s1(tmp_path: Path) -> None:
    controller = BlobFeaturesController()
    with pytest.raises(BlobFeaturesUiError):
        controller.require_eligible()
    session = _write_eligible_s1(tmp_path)
    eligible = controller.set_session(session)
    assert eligible.frame_count == 6
    assert controller.state.current_frame_bgr is not None
    assert controller.state.background_bgr is not None


def test_frame_navigation_uses_prepared_indices(tmp_path: Path) -> None:
    controller = BlobFeaturesController()
    session = _write_eligible_s1(tmp_path)
    controller.set_session(session)
    assert controller.set_frame_index(2) == 2
    assert controller.current_timestamp_sec() == pytest.approx(2 / 20.0)
    assert controller.step_frame(1) == 3
    assert controller.step_frame(-10) == 0
    with pytest.raises(BlobFeaturesUiError):
        controller.set_frame_index(99)


def test_settings_load_from_suite_defaults() -> None:
    controller = BlobFeaturesController()
    settings = controller.reset_working_settings_to_defaults()
    suite = load_blob_features_config()
    assert settings.segmentation.blob_threshold == suite.segmentation.blob_threshold
    assert settings.segmentation.blob_min_area == suite.segmentation.blob_min_area
    assert settings.execution.n_workers == suite.execution.n_workers


def test_invalid_settings_rejected() -> None:
    controller = BlobFeaturesController()
    bad = BlobFeaturesUiSettings(
        segmentation=BlobSegmentationSettings(blob_blur_kernel_size=8)
    )
    with pytest.raises(BlobFeaturesUiError):
        controller.set_working_settings(bad)


def test_preview_computation_and_invalidation(tmp_path: Path) -> None:
    controller = BlobFeaturesController()
    session = _write_eligible_s1(tmp_path)
    controller.set_session(session)
    controller.set_working_settings(_soft_settings())
    identity = controller.begin_preview_request()
    result = controller.compute_single_frame_preview(identity)
    assert controller.accept_single_preview(result, expected_request_id=identity.request_id)
    assert controller.state.preview_valid is True
    assert result.preview.cc_count_raw == 2

    # Changing scientific settings invalidates preview without touching extraction snapshot.
    controller.record_extraction_settings_snapshot()
    extraction = controller.state.last_extraction_settings
    assert extraction is not None
    soft = _soft_settings()
    updated = BlobFeaturesUiSettings(
        segmentation=replace(soft.segmentation, blob_min_area=25),
        temporal=soft.temporal,
        execution=soft.execution,
    )
    controller.set_working_settings(updated)
    assert controller.state.preview_valid is False
    assert controller.state.last_single_preview is None
    assert controller.state.last_extraction_settings == extraction


def test_source_switch_invalidates_preview(tmp_path: Path) -> None:
    controller = BlobFeaturesController()
    first = _write_eligible_s1(tmp_path / "a")
    second = _write_eligible_s1(tmp_path / "b")
    controller.set_session(first)
    controller.set_working_settings(_soft_settings())
    identity = controller.begin_preview_request()
    result = controller.compute_single_frame_preview(identity)
    assert controller.accept_single_preview(result, expected_request_id=identity.request_id)
    controller.set_session(second)
    assert controller.state.preview_valid is False
    assert controller.state.last_single_preview is None


def test_stale_preview_result_rejected(tmp_path: Path) -> None:
    controller = BlobFeaturesController()
    session = _write_eligible_s1(tmp_path)
    controller.set_session(session)
    controller.set_working_settings(_soft_settings())
    identity = controller.begin_preview_request()
    result = controller.compute_single_frame_preview(identity)
    # Mutate settings after the worker snapshot was taken.
    soft = _soft_settings()
    controller.set_working_settings(
        BlobFeaturesUiSettings(
            segmentation=replace(soft.segmentation, blob_min_area=30),
            temporal=soft.temporal,
            execution=soft.execution,
        )
    )
    assert (
        controller.accept_single_preview(result, expected_request_id=identity.request_id)
        is False
    )


def test_bookmark_persistence_and_multi_preview(tmp_path: Path) -> None:
    controller = BlobFeaturesController()
    session = _write_eligible_s1(tmp_path)
    controller.set_session(session)
    controller.set_working_settings(_soft_settings())
    controller.set_frame_index(1)
    controller.add_current_bookmark("first")
    controller.set_frame_index(3)
    controller.add_current_bookmark("fragmented")
    assert [b.prepared_frame_idx for b in controller.bookmark_list()] == [1, 3]

    identity = controller.begin_preview_request([1, 3])
    multi = controller.compute_multi_frame_preview(identity)
    assert controller.accept_multi_preview(multi, expected_request_id=identity.request_id)
    assert len(multi.items) == 2
    assert multi.items[0].preview.cc_count_raw == 2
    assert multi.items[1].preview.cc_count_raw == 3
    assert np.isnan(multi.items[1].preview.cc_count)

    # Reloading the session restores bookmarks.
    controller2 = BlobFeaturesController()
    controller2.set_session(session)
    assert [b.prepared_frame_idx for b in controller2.bookmark_list()] == [1, 3]
    assert controller2.bookmark_list()[0].note == "first"


def test_missing_s1_error_handling(tmp_path: Path) -> None:
    controller = BlobFeaturesController()
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(BlobFeaturesUiError):
        controller.set_session(empty)


def test_page_readiness_and_preview_action(tmp_path: Path) -> None:
    session = _write_eligible_s1(tmp_path)
    page = BlobFeaturesPage(BlobFeaturesController())
    page.controller.set_working_settings(_soft_settings())
    page.set_session(session)
    assert page.controller.state.eligible is not None
    assert "Eligible S1" in page.eligibility_label.text()
    page.preview_button.click()
    _wait_until(lambda: page.controller.state.preview_valid is True)
    assert page.controller.state.last_single_preview is not None
    assert page.cc_raw_label.text() != "—"


def test_main_window_exposes_s5_navigation(tmp_path: Path) -> None:
    session = _write_eligible_s1(tmp_path)
    window = MainWindow(PreprocessSetupController())
    assert window.s5_button.isEnabled() is True
    window.open_blob_features(session)
    assert window.pages.currentWidget() is window.blob_features_page
    assert window.blob_features_page.controller.state.eligible is not None


def test_group_navigation_preserves_s1_s4_and_shares_session(tmp_path: Path) -> None:
    from ui.main_window import GROUP_DATA_PREPARATION, GROUP_FEATURE_PROCESSING

    session = _write_eligible_s1(tmp_path)
    window = MainWindow(PreprocessSetupController())
    window.show()
    _APPLICATION.processEvents()
    window.open_pose_inference(session)
    assert window._active_group == GROUP_DATA_PREPARATION
    assert window.s1_button.isHidden() is False
    assert window.s5_button.isHidden() is True
    assert window.pages.currentWidget() is window.pose_page

    window.set_application_group(GROUP_FEATURE_PROCESSING)
    assert window._active_group == GROUP_FEATURE_PROCESSING
    assert window.s5_button.isHidden() is False
    assert window.s1_button.isHidden() is True
    assert window.s6_button.isHidden() is False
    assert window.s6_button.isEnabled() is True
    assert window.pages.currentWidget() is window.blob_features_page
    assert window._session_root == session.resolve()
    assert window.blob_features_page.controller.state.eligible is not None

    window.set_application_group(GROUP_DATA_PREPARATION)
    assert window.pages.currentWidget() is window.wizard
    assert window.s4_button.isHidden() is False
    # Session root remains shared across groups.
    assert window._session_root == session.resolve()


def test_s5_independent_of_s2_s4_completion(tmp_path: Path) -> None:
    session = _write_eligible_s1(tmp_path)
    assert not (session / "pose_inference").exists()
    window = MainWindow(PreprocessSetupController())
    window.set_application_group("feature_processing")
    window.open_blob_features(session)
    assert window.blob_features_page.controller.state.eligible is not None


def test_extraction_gui_launch_and_settings_snapshot(tmp_path: Path) -> None:
    session = _write_eligible_s1(tmp_path)
    controller = BlobFeaturesController()
    controller.set_session(session)
    controller.set_working_settings(_soft_settings())
    request, identity = controller.build_request(timestamp="20260101T120000")
    assert request.config is not None
    snapshot = request.config.segmentation.blob_min_area
    # Mutate working settings after snapshot — must not affect the request.
    soft = _soft_settings()
    controller.set_working_settings(
        BlobFeaturesUiSettings(
            segmentation=replace(soft.segmentation, blob_min_area=99),
            temporal=soft.temporal,
            execution=soft.execution,
        )
    )
    assert request.config.segmentation.blob_min_area == snapshot
    assert identity.settings.segmentation.blob_min_area == snapshot

    page = BlobFeaturesPage(BlobFeaturesController())
    page.controller.set_working_settings(_soft_settings())
    page.set_session(session)
    assert page.run_button.isEnabled() is True
    page.run_button.click()
    _wait_until(lambda: page.controller.state.last_result is not None, timeout_sec=60.0)
    result = page.controller.state.last_result
    assert result is not None
    assert result.success is True
    assert result.blob_features_path is not None
    assert result.blob_features_path.is_file()
    rows = page.controller.qc_display_rows()
    assert any(name == "Status" for name, _ in rows)
    assert page.controller.state.summary is not None
    assert page.controller.state.summary.completed_runs


def test_working_settings_and_bookmarks_persist_per_s1_identity(tmp_path: Path) -> None:
    first = _write_eligible_s1(tmp_path / "a")
    second = _write_eligible_s1(tmp_path / "b")
    controller = BlobFeaturesController()
    controller.set_session(first)
    soft = _soft_settings()
    updated = BlobFeaturesUiSettings(
        segmentation=replace(soft.segmentation, blob_min_area=55),
        temporal=soft.temporal,
        execution=soft.execution,
    )
    controller.set_working_settings(updated)
    controller.set_frame_index(2)
    controller.add_current_bookmark("a-only")

    controller.set_session(second)
    # Second session must not inherit the first session's scientific settings.
    assert controller.state.working_settings.segmentation.blob_min_area != 55
    # Second session starts without first session's bookmarks.
    assert controller.bookmark_list() == ()
    controller.set_frame_index(1)
    controller.add_current_bookmark("b-only")

    again = BlobFeaturesController()
    again.set_session(first)
    assert again.state.working_settings.segmentation.blob_min_area == 55
    assert [b.prepared_frame_idx for b in again.bookmark_list()] == [2]
    again.set_session(second)
    assert [b.prepared_frame_idx for b in again.bookmark_list()] == [1]


def test_source_switch_during_extraction_rejects_stale_result(tmp_path: Path) -> None:
    first = _write_eligible_s1(tmp_path / "a")
    second = _write_eligible_s1(tmp_path / "b")
    controller = BlobFeaturesController()
    controller.set_session(first)
    controller.set_working_settings(_soft_settings())
    request, identity = controller.build_request(timestamp="20260101T130000")
    # Primary protection: session mutation is blocked while the task flag is set.
    with pytest.raises(BlobFeaturesUiError, match="Cannot change S5 session"):
        controller.set_session(second)
    # Secondary protection: if the GUI thread somehow advances to a different
    # source before accept, the stale worker result is discarded.
    controller.state.task_running = False
    controller.set_session(second)
    result = controller.execute_request(request)
    assert (
        controller.record_success(result, expected_identity=identity) is False
    )


def test_completed_run_discovery_in_controller(tmp_path: Path) -> None:
    from blob_features import (
        BlobFeaturesRequest,
        run_blob_feature_extraction,
    )

    session = _write_eligible_s1(tmp_path)
    run_blob_feature_extraction(
        BlobFeaturesRequest(
            session_root=session,
            config=_soft_settings().to_config(),
            run_purpose="test",
            timestamp="20260101T140000",
        )
    )
    controller = BlobFeaturesController()
    controller.set_session(session)
    assert controller.state.summary is not None
    assert controller.state.summary.completed_runs
    assert controller.selected_s5_run() is not None
    assert controller.qc_display_rows()


def test_batch_bookmark_preview_caches_all_frames_without_jumping(
    tmp_path: Path,
) -> None:
    controller = BlobFeaturesController()
    session = _write_eligible_s1(tmp_path)
    controller.set_session(session)
    controller.set_working_settings(_soft_settings())
    controller.set_frame_index(1)
    controller.add_current_bookmark("a")
    controller.set_frame_index(3)
    controller.add_current_bookmark("b")
    controller.set_frame_index(2)
    assert controller.state.current_frame_idx == 2

    identity = controller.begin_preview_request([1, 3])
    multi = controller.compute_multi_frame_preview(identity)
    assert controller.accept_multi_preview(multi, expected_request_id=identity.request_id)
    assert len(multi.items) == 2
    # Selected frame must remain unchanged after batch accept.
    assert controller.state.current_frame_idx == 2
    assert controller.current_frame_has_valid_preview() is False

    controller.set_frame_index(1)
    assert controller.current_frame_has_valid_preview() is True
    assert controller.preview_for_current_frame() is not None
    assert controller.preview_for_current_frame().preview.prepared_frame_idx == 1

    controller.set_frame_index(3)
    assert controller.current_frame_has_valid_preview() is True
    assert controller.preview_for_current_frame().preview.cc_count_raw == 3


def test_navigating_back_restores_cached_preview(tmp_path: Path) -> None:
    controller = BlobFeaturesController()
    session = _write_eligible_s1(tmp_path)
    controller.set_session(session)
    controller.set_working_settings(_soft_settings())
    identity = controller.begin_preview_request()
    result = controller.compute_single_frame_preview(identity)
    assert controller.accept_single_preview(result, expected_request_id=identity.request_id)
    first = controller.preview_for_current_frame()
    assert first is not None
    controller.set_frame_index(4)
    assert controller.current_frame_has_valid_preview() is False
    controller.set_frame_index(0)
    restored = controller.preview_for_current_frame()
    assert restored is not None
    assert restored.preview.cc_count_raw == first.preview.cc_count_raw


def test_processed_views_require_preview_without_original_fallback(
    tmp_path: Path,
) -> None:
    page = BlobFeaturesPage(BlobFeaturesController())
    session = _write_eligible_s1(tmp_path)
    page.controller.set_working_settings(_soft_settings())
    page.set_session(session)
    page.view_mode.setCurrentIndex(
        page.view_mode.findData("binary_mask")
    )
    page._display_current_view()
    # Page is not shown in offscreen tests; use isHidden() rather than isVisible().
    assert page.preview_required_label.isHidden() is False
    assert page.frame_view._frame is None
    assert page.controller.current_frame_has_valid_preview() is False
    assert page.controller.state.current_frame_bgr is not None
    # Original remains available when explicitly selected.
    page.view_mode.setCurrentIndex(page.view_mode.findData("original"))
    page._display_current_view()
    assert page.preview_required_label.isHidden() is True
    assert page.frame_view._frame is not None


def test_preview_cache_invalidates_on_settings_and_source_change(
    tmp_path: Path,
) -> None:
    controller = BlobFeaturesController()
    first = _write_eligible_s1(tmp_path / "a")
    second = _write_eligible_s1(tmp_path / "b")
    controller.set_session(first)
    controller.set_working_settings(_soft_settings())
    identity = controller.begin_preview_request()
    result = controller.compute_single_frame_preview(identity)
    assert controller.accept_single_preview(result, expected_request_id=identity.request_id)
    assert len(controller.state.preview_cache) == 1

    soft = _soft_settings()
    controller.set_working_settings(
        BlobFeaturesUiSettings(
            segmentation=replace(soft.segmentation, blob_min_area=40),
            temporal=soft.temporal,
            execution=soft.execution,
        )
    )
    assert len(controller.state.preview_cache) == 0
    assert controller.current_frame_has_valid_preview() is False

    identity2 = controller.begin_preview_request()
    result2 = controller.compute_single_frame_preview(identity2)
    assert controller.accept_single_preview(
        result2, expected_request_id=identity2.request_id
    )
    controller.set_session(second)
    assert len(controller.state.preview_cache) == 0


def test_scientific_defaults_use_updated_min_areas() -> None:
    controller = BlobFeaturesController()
    settings = controller.reset_working_settings_to_defaults()
    assert settings.segmentation.blob_min_area == 2900
    assert settings.segmentation.blob_post_prune_min_area == 2900
    assert settings.segmentation.blob_min_area != settings.segmentation.blob_max_area


def test_min_area_parameters_remain_distinct(tmp_path: Path) -> None:
    controller = BlobFeaturesController()
    session = _write_eligible_s1(tmp_path)
    controller.set_session(session)
    soft = _soft_settings()
    distinct = BlobFeaturesUiSettings(
        segmentation=replace(
            soft.segmentation,
            blob_min_area=111,
            blob_post_prune_min_area=222,
        ),
        temporal=soft.temporal,
        execution=soft.execution,
    )
    controller.set_working_settings(distinct)
    seg = controller.state.working_settings.segmentation
    assert seg.blob_min_area == 111
    assert seg.blob_post_prune_min_area == 222

    page = BlobFeaturesPage(BlobFeaturesController())
    page.controller.set_working_settings(distinct)
    page.set_session(session)
    assert page.blob_min_area.value() == 111
    assert page.post_min_area.value() == 222
    # Changing one control must not silently rewrite the other.
    page.blob_min_area.setValue(150)
    page._settings_widgets_changed()
    assert page.controller.state.working_settings.segmentation.blob_min_area == 150
    assert page.controller.state.working_settings.segmentation.blob_post_prune_min_area == 222


def test_group_navigation_allowed_during_s5_extraction(tmp_path: Path) -> None:
    from ui.main_window import GROUP_DATA_PREPARATION, GROUP_FEATURE_PROCESSING

    session = _write_eligible_s1(tmp_path)
    window = MainWindow(PreprocessSetupController())
    window.open_blob_features(session)
    page = window.blob_features_page
    page.controller.set_working_settings(_soft_settings())
    request, identity = page.controller.build_request(timestamp="20260101T150000")
    assert page.controller.state.active_extraction_identity is not None
    assert page.has_running_task is True
    window._update_navigation()
    assert window.group_data_button.isEnabled() is True
    assert window.s1_button.isEnabled() is True
    assert window.s5_button.isEnabled() is True

    window.set_application_group(GROUP_DATA_PREPARATION)
    assert window._active_group == GROUP_DATA_PREPARATION
    assert window.pages.currentWidget() is window.wizard
    assert page.controller.state.active_extraction_identity is identity
    assert page.controller.state.task_running is True

    window.set_application_group(GROUP_FEATURE_PROCESSING)
    assert window.pages.currentWidget() is page
    assert page.controller.state.active_extraction_identity is identity
    # Returning must not start a duplicate extraction or clear identity.
    assert page.run_button.isEnabled() is False
    page.controller.record_failure("cancelled-for-test", expected_identity=identity)
    _ = request


def test_source_mutation_blocked_while_s5_task_active(tmp_path: Path) -> None:
    first = _write_eligible_s1(tmp_path / "a")
    second = _write_eligible_s1(tmp_path / "b")
    controller = BlobFeaturesController()
    controller.set_session(first)
    controller.set_working_settings(_soft_settings())
    _request, _identity = controller.build_request(timestamp="20260101T160000")
    with pytest.raises(BlobFeaturesUiError, match="Cannot change S5 session"):
        controller.set_session(second)
    assert controller.state.session_root == first.resolve()


def test_area_filtering_labels_grouped(tmp_path: Path) -> None:
    from PySide6.QtWidgets import QGroupBox

    page = BlobFeaturesPage(BlobFeaturesController())
    session = _write_eligible_s1(tmp_path)
    page.set_session(session)
    groups = [
        box for box in page.findChildren(QGroupBox) if box.title() == "Blob Area Filtering"
    ]
    assert len(groups) == 1
    assert page.blob_min_area.parent() is groups[0]
    assert page.post_min_area.parent() is groups[0]
    assert "Initial filtering" in page.blob_min_area.toolTip() or page.blob_min_area.toolTip()
    assert "pruning" in page.post_min_area.toolTip().lower()
