"""Subsystem 05 interactive video segmentation, preview, and extraction workspace."""

from __future__ import annotations

import contextlib
import logging
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from blob_features import BlobFeaturesResult, BlobSegmentationSettings
from ui.controllers.blob_features_controller import (
    BlobFeaturesController,
    BlobFeaturesUiError,
    BlobFeaturesUiSettings,
    ExtractionRequestIdentity,
    MultiFramePreviewResult,
    SingleFramePreviewResult,
    ensure_bgr,
)
from ui.tasks import (
    GuiTaskContext,
    GuiTaskKind,
    GuiTaskProgressEvent,
    GuiTaskRunner,
    TaskAlreadyRunningError,
    format_pipeline_progress,
    format_task_duration,
)
from ui.widgets.video_frame_view import VideoFrameView

_LOGGER = logging.getLogger(__name__)

_VIEW_ORIGINAL = "original"
_VIEW_FOREGROUND = "foreground"
_VIEW_BINARY = "binary_mask"
_VIEW_CONTOURS = "contours"
_VIEW_BACKGROUND = "background"


class BlobFeaturesPage(QWidget):
    """S5 workspace: eligible S1 video, segmentation preview, and full extraction."""

    status_message = Signal(str)
    unexpected_error = Signal(str)
    task_running_changed = Signal(bool)
    task_finished = Signal()

    def __init__(
        self,
        controller: BlobFeaturesController | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller or BlobFeaturesController()
        self.task_runner = GuiTaskRunner(self)
        self._syncing_widgets = False
        self._last_running_state: bool | None = None
        self._multi_preview_index = 0
        self._started_at: float | None = None
        self._active_extraction_identity: ExtractionRequestIdentity | None = None
        self._build_ui()
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.setInterval(500)
        self.elapsed_timer.timeout.connect(self._update_elapsed)
        self.refresh_from_state()

    @property
    def has_running_task(self) -> bool:
        return self.task_runner.is_running or self.controller.state.task_running

    def set_session(self, session_root: Path | str) -> None:
        """Open a project/session with eligible S1 artifacts."""

        try:
            self.controller.set_session(session_root)
        except BlobFeaturesUiError as exc:
            QMessageBox.warning(self, "Cannot open S5 session", str(exc))
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return
        self.refresh_from_state()
        self.status_message.emit(f"Opened session for S5: {session_root}")

    def refresh_from_state(self) -> None:
        state = self.controller.state
        self._syncing_widgets = True
        try:
            self.session_path.setText(
                str(state.session_root) if state.session_root is not None else ""
            )
            self._sync_settings_widgets(state.working_settings)
            self._refresh_eligibility_labels()
            self._refresh_frame_controls()
            self._refresh_bookmarks()
            self._refresh_preview_panel()
            self._populate_runs_table()
            self._refresh_qc_panel()
            self._update_enabled()
            self._emit_running_changed()
        finally:
            self._syncing_widgets = False

    def _build_ui(self) -> None:
        title = QLabel("Subsystem 5 — Video / Blob Feature Extraction")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        scope = QLabel(
            "Inspect prepared-video frames from an eligible Subsystem 1 run, "
            "adjust legacy segmentation settings, preview masks/topology, then "
            "run full-video blob feature extraction to 120 Hz parquet. "
            "S2–S4 completion is not required. Cancellation is not supported; "
            "partial parquet outputs are never published as completed."
        )
        scope.setWordWrap(True)

        session_group = QGroupBox("Eligible S1 source")
        session_form = QFormLayout(session_group)
        self.session_path = QLineEdit()
        self.session_browse_button = QPushButton("Browse…")
        self.session_browse_button.clicked.connect(self._browse_session)
        self.open_session_button = QPushButton("Open session")
        self.open_session_button.clicked.connect(self._open_typed_session)
        session_row = QHBoxLayout()
        session_row.addWidget(self.session_path, 1)
        session_row.addWidget(self.session_browse_button)
        session_row.addWidget(self.open_session_button)
        self.eligibility_label = QLabel("Select a project/session with eligible S1.")
        self.eligibility_label.setWordWrap(True)
        self.eligibility_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.refresh_button = QPushButton("Refresh discovery")
        self.refresh_button.clicked.connect(self._refresh_discovery)
        session_form.addRow("Session/project", session_row)
        session_form.addRow("Eligibility", self.eligibility_label)
        session_form.addRow(self.refresh_button)

        self.frame_view = VideoFrameView()
        viewer_group = QGroupBox("Prepared video viewer")
        viewer_layout = QVBoxLayout(viewer_group)
        self.frame_meta_label = QLabel("No frame loaded.")
        self.frame_meta_label.setWordWrap(True)
        nav = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.next_button = QPushButton("Next")
        self.prev_button.clicked.connect(lambda: self._step_frame(-1))
        self.next_button.clicked.connect(lambda: self._step_frame(1))
        self.frame_spin = QSpinBox()
        self.frame_spin.setMinimum(0)
        self.frame_spin.valueChanged.connect(self._frame_spin_changed)
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self._frame_slider_changed)
        nav.addWidget(self.prev_button)
        nav.addWidget(self.next_button)
        nav.addWidget(QLabel("Frame"))
        nav.addWidget(self.frame_spin)
        nav.addWidget(self.frame_slider, 1)
        self.view_mode = QComboBox()
        self.view_mode.addItem("Original frame", _VIEW_ORIGINAL)
        self.view_mode.addItem("Background", _VIEW_BACKGROUND)
        self.view_mode.addItem("Foreground (bg − frame)", _VIEW_FOREGROUND)
        self.view_mode.addItem("Binary mask", _VIEW_BINARY)
        self.view_mode.addItem("Contours overlay", _VIEW_CONTOURS)
        self.view_mode.currentIndexChanged.connect(self._view_mode_changed)
        self.preview_required_label = QLabel("Preview required for this frame")
        self.preview_required_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_required_label.setWordWrap(True)
        self.preview_required_label.setStyleSheet(
            "color: #5c4a00; background: #fff6d5; padding: 10px; border: 1px solid #e0c35a;"
        )
        self.preview_required_label.setVisible(False)
        viewer_layout.addWidget(self.frame_meta_label)
        viewer_layout.addLayout(nav)
        viewer_layout.addWidget(self.view_mode)
        viewer_layout.addWidget(self.preview_required_label)
        viewer_layout.addWidget(self.frame_view, 1)

        diagnostics = QGroupBox("Segmentation diagnostics")
        diag_form = QFormLayout(diagnostics)
        self.cc_raw_label = QLabel("—")
        self.cc_effective_label = QLabel("—")
        self.min_dist_label = QLabel("—")
        self.warnings_label = QLabel("—")
        self.warnings_label.setWordWrap(True)
        self.preview_status_label = QLabel("No preview yet. Click Preview Frame.")
        self.preview_status_label.setWordWrap(True)
        diag_form.addRow("cc_count_raw", self.cc_raw_label)
        diag_form.addRow("cc_count (effective)", self.cc_effective_label)
        diag_form.addRow("min_inter_contour_dist", self.min_dist_label)
        diag_form.addRow("Warnings", self.warnings_label)
        diag_form.addRow("Preview state", self.preview_status_label)

        settings_group = self._build_settings_group()
        bookmarks_group = self._build_bookmarks_group()

        actions = QHBoxLayout()
        self.preview_button = QPushButton("Preview Frame")
        self.preview_button.clicked.connect(self._preview_current_frame)
        self.preview_bookmarks_button = QPushButton("Preview Bookmarks")
        self.preview_bookmarks_button.clicked.connect(self._preview_bookmarks)
        self.reset_defaults_button = QPushButton("Reset settings to defaults")
        self.reset_defaults_button.clicked.connect(self._reset_defaults)
        self.run_button = QPushButton("Run Full Extraction")
        self.run_button.clicked.connect(self._run_extraction)
        self.open_output_button = QPushButton("Open S5 run folder")
        self.open_output_button.clicked.connect(self._open_output_folder)
        actions.addWidget(self.preview_button)
        actions.addWidget(self.preview_bookmarks_button)
        actions.addWidget(self.reset_defaults_button)
        actions.addWidget(self.run_button)
        actions.addWidget(self.open_output_button)
        actions.addStretch(1)

        self.busy_indicator = QProgressBar()
        self.busy_indicator.setRange(0, 0)
        self.busy_indicator.setVisible(False)

        status_group = QGroupBox("Processing status")
        status_layout = QVBoxLayout(status_group)
        self.stage_label = QLabel("Not started")
        self.stage_label.setWordWrap(True)
        self.elapsed_label = QLabel("Elapsed: —")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        status_layout.addWidget(self.stage_label)
        status_layout.addWidget(self.elapsed_label)
        status_layout.addWidget(self.progress)
        status_layout.addWidget(self.busy_indicator)

        qc_group = QGroupBox("QC summary")
        qc_layout = QVBoxLayout(qc_group)
        self.qc_table = QTableWidget(0, 2)
        self.qc_table.setHorizontalHeaderLabels(("Metric", "Value"))
        self.qc_table.verticalHeader().setVisible(False)
        self.qc_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.qc_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        self.qc_table.horizontalHeader().setSectionResizeMode(
            1, QHeaderView.ResizeMode.Stretch
        )
        self.qc_table.setMaximumHeight(160)
        qc_layout.addWidget(self.qc_table)

        runs_group = QGroupBox("Completed S5 runs (newest first)")
        runs_layout = QVBoxLayout(runs_group)
        self.runs_table = QTableWidget(0, 4)
        self.runs_table.setHorizontalHeaderLabels(
            ("Run ID", "Status", "Classification", "Warnings")
        )
        self.runs_table.verticalHeader().setVisible(False)
        self.runs_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.runs_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.runs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.runs_table.itemSelectionChanged.connect(self._s5_selection_changed)
        self.runs_table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.Stretch
        )
        self.runs_table.setMaximumHeight(140)
        runs_layout.addWidget(self.runs_table)

        self.multi_preview_group = QGroupBox("Bookmark batch preview")
        multi_layout = QVBoxLayout(self.multi_preview_group)
        multi_nav = QHBoxLayout()
        self.multi_prev_button = QPushButton("Prev result")
        self.multi_next_button = QPushButton("Next result")
        self.multi_prev_button.clicked.connect(lambda: self._step_multi_preview(-1))
        self.multi_next_button.clicked.connect(lambda: self._step_multi_preview(1))
        self.multi_summary_label = QLabel("No batch preview.")
        self.multi_summary_label.setWordWrap(True)
        multi_nav.addWidget(self.multi_prev_button)
        multi_nav.addWidget(self.multi_next_button)
        multi_nav.addStretch(1)
        multi_layout.addLayout(multi_nav)
        multi_layout.addWidget(self.multi_summary_label)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.addWidget(settings_group)
        side_layout.addWidget(bookmarks_group)
        side_layout.addWidget(diagnostics)
        side_layout.addWidget(status_group)
        side_layout.addWidget(qc_group)
        side_layout.addWidget(runs_group)
        side_layout.addWidget(self.multi_preview_group)
        side_layout.addStretch(1)
        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setMinimumWidth(360)
        side_scroll.setMaximumWidth(480)
        side_scroll.setWidget(side)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(side_scroll)
        splitter.addWidget(viewer_group)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)

        self.error_label = QLabel()
        self.error_label.setWordWrap(True)
        self.error_label.setStyleSheet("color: #b00020;")

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(scope)
        layout.addWidget(session_group)
        layout.addLayout(actions)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.error_label)

    def _build_settings_group(self) -> QGroupBox:
        group = QGroupBox("Segmentation settings")
        layout = QVBoxLayout(group)
        common = QFormLayout()
        self.blob_threshold = self._int_spin(-2, 255)
        self.blob_blur = self._int_spin(1, 99, step=2)
        self.blob_kernel = self._int_spin(1, 99, step=2)
        self.abs_dark = self._int_spin(0, 255)
        common.addRow("blob_threshold (−2 Multi-Otsu, −1 Otsu, ≥0 fixed)", self.blob_threshold)
        common.addRow("blob_blur_kernel_size (odd)", self.blob_blur)
        common.addRow("blob_kernel_size", self.blob_kernel)
        common.addRow("abs_dark_threshold", self.abs_dark)
        layout.addLayout(common)

        area_group = QGroupBox("Blob Area Filtering")
        area_form = QFormLayout(area_group)
        self.blob_min_area = self._int_spin(1, 10_000_000)
        self.post_min_area = self._int_spin(1, 10_000_000)
        self.blob_min_area.setToolTip(
            "Minimum connected-component area kept after the first cleanup pass "
            "(before appendage pruning). Distinct from the post-prune threshold."
        )
        self.post_min_area.setToolTip(
            "Minimum connected-component area kept after appendage pruning / "
            "regrowth. Applied later than initial filtering; values are not linked."
        )
        area_form.addRow("Minimum area — Initial filtering", self.blob_min_area)
        area_form.addRow("Minimum area — After pruning", self.post_min_area)
        layout.addWidget(area_group)

        self.advanced_toggle = QToolButton()
        self.advanced_toggle.setText("Advanced scientific settings")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setArrowType(Qt.ArrowType.RightArrow)
        self.advanced_toggle.toggled.connect(self._toggle_advanced)
        self.advanced_widget = QWidget()
        advanced = QFormLayout(self.advanced_widget)
        self.blob_tail_open = self._int_spin(1, 99, step=2)
        self.blob_max_area = self._int_spin(0, 10_000_000)
        self.prune_radius = self._int_spin(1, 1000)
        self.prune_iters = self._int_spin(0, 1000)
        self.prune_dist = self._int_spin(0, 1000)
        self.post_max_area = self._int_spin(0, 10_000_000)
        self.final_smooth = self._int_spin(1, 99, step=2)
        self.gap_fill = self._int_spin(0, 10_000)
        advanced.addRow("blob_tail_open_kernel_size", self.blob_tail_open)
        advanced.addRow("blob_max_area (0=unlimited)", self.blob_max_area)
        advanced.addRow("blob_prune_min_radius", self.prune_radius)
        advanced.addRow("blob_prune_regrow_iters", self.prune_iters)
        advanced.addRow("blob_prune_max_regrow_dist", self.prune_dist)
        advanced.addRow("blob_post_prune_max_area (0=unlimited)", self.post_max_area)
        advanced.addRow("blob_final_smooth_kernel_size", self.final_smooth)
        advanced.addRow("cc_count_gap_fill_max_gap (extraction only)", self.gap_fill)
        self.advanced_widget.setVisible(False)
        layout.addWidget(self.advanced_toggle)
        layout.addWidget(self.advanced_widget)

        performance = QGroupBox("Performance (execution only)")
        performance.setCheckable(True)
        performance.setChecked(False)
        perf_form = QFormLayout(performance)
        self.n_workers = self._int_spin(1, 256)
        self.max_in_flight = self._int_spin(1, 10_000)
        perf_form.addRow("n_workers", self.n_workers)
        perf_form.addRow("max_in_flight", self.max_in_flight)
        layout.addWidget(performance)

        for widget in (
            self.blob_threshold,
            self.blob_blur,
            self.blob_kernel,
            self.blob_min_area,
            self.abs_dark,
            self.blob_tail_open,
            self.blob_max_area,
            self.prune_radius,
            self.prune_iters,
            self.prune_dist,
            self.post_min_area,
            self.post_max_area,
            self.final_smooth,
            self.gap_fill,
            self.n_workers,
            self.max_in_flight,
        ):
            widget.valueChanged.connect(self._settings_widgets_changed)
        return group

    def _build_bookmarks_group(self) -> QGroupBox:
        group = QGroupBox("Validation-frame bookmarks")
        layout = QVBoxLayout(group)
        note = QLabel(
            "Diagnostic prepared-frame references for the selected S1 source. "
            "Not annotations or ground truth."
        )
        note.setWordWrap(True)
        self.bookmark_list = QListWidget()
        self.bookmark_list.itemDoubleClicked.connect(self._bookmark_activated)
        self.bookmark_note = QLineEdit()
        self.bookmark_note.setPlaceholderText("Optional short note")
        row = QHBoxLayout()
        self.add_bookmark_button = QPushButton("Bookmark current")
        self.remove_bookmark_button = QPushButton("Remove selected")
        self.goto_bookmark_button = QPushButton("Go to selected")
        self.add_bookmark_button.clicked.connect(self._add_bookmark)
        self.remove_bookmark_button.clicked.connect(self._remove_bookmark)
        self.goto_bookmark_button.clicked.connect(self._goto_bookmark)
        row.addWidget(self.add_bookmark_button)
        row.addWidget(self.remove_bookmark_button)
        row.addWidget(self.goto_bookmark_button)
        layout.addWidget(note)
        layout.addWidget(self.bookmark_list)
        layout.addWidget(self.bookmark_note)
        layout.addLayout(row)
        return group

    def _int_spin(self, minimum: int, maximum: int, *, step: int = 1) -> QSpinBox:
        spin = QSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        return spin

    def _toggle_advanced(self, checked: bool) -> None:
        self.advanced_widget.setVisible(checked)
        self.advanced_toggle.setArrowType(
            Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow
        )

    def _browse_session(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        start = self.session_path.text().strip()
        chosen = QFileDialog.getExistingDirectory(
            self, "Open project/session directory", start
        )
        if chosen:
            self.set_session(chosen)

    def _open_typed_session(self) -> None:
        text = self.session_path.text().strip()
        if not text:
            QMessageBox.information(self, "Session required", "Enter a session path.")
            return
        self.set_session(text)

    def _settings_from_widgets(self) -> BlobFeaturesUiSettings:
        from blob_features import BlobExecutionSettings, BlobTemporalSettings

        segmentation = BlobSegmentationSettings(
            blob_threshold=int(self.blob_threshold.value()),
            blob_blur_kernel_size=int(self.blob_blur.value()),
            blob_kernel_size=int(self.blob_kernel.value()),
            blob_tail_open_kernel_size=int(self.blob_tail_open.value()),
            blob_min_area=int(self.blob_min_area.value()),
            blob_max_area=int(self.blob_max_area.value()),
            blob_prune_min_radius=int(self.prune_radius.value()),
            blob_prune_regrow_iters=int(self.prune_iters.value()),
            blob_prune_max_regrow_dist=int(self.prune_dist.value()),
            blob_post_prune_min_area=int(self.post_min_area.value()),
            blob_post_prune_max_area=int(self.post_max_area.value()),
            blob_final_smooth_kernel_size=int(self.final_smooth.value()),
            abs_dark_threshold=int(self.abs_dark.value()),
        )
        temporal = BlobTemporalSettings(
            cc_count_gap_fill_max_gap=int(self.gap_fill.value())
        )
        execution = BlobExecutionSettings(
            n_workers=int(self.n_workers.value()),
            max_in_flight=int(self.max_in_flight.value()),
        )
        return BlobFeaturesUiSettings(
            segmentation=segmentation,
            temporal=temporal,
            execution=execution,
        )

    def _sync_settings_widgets(self, settings: BlobFeaturesUiSettings) -> None:
        seg = settings.segmentation
        temporal = settings.temporal
        execution = settings.execution
        widgets_values = (
            (self.blob_threshold, seg.blob_threshold),
            (self.blob_blur, seg.blob_blur_kernel_size),
            (self.blob_kernel, seg.blob_kernel_size),
            (self.blob_min_area, seg.blob_min_area),
            (self.abs_dark, seg.abs_dark_threshold),
            (self.blob_tail_open, seg.blob_tail_open_kernel_size),
            (self.blob_max_area, seg.blob_max_area),
            (self.prune_radius, seg.blob_prune_min_radius),
            (self.prune_iters, seg.blob_prune_regrow_iters),
            (self.prune_dist, seg.blob_prune_max_regrow_dist),
            (self.post_min_area, seg.blob_post_prune_min_area),
            (self.post_max_area, seg.blob_post_prune_max_area),
            (self.final_smooth, seg.blob_final_smooth_kernel_size),
            (self.gap_fill, temporal.cc_count_gap_fill_max_gap),
            (self.n_workers, execution.n_workers),
            (self.max_in_flight, execution.max_in_flight),
        )
        for widget, value in widgets_values:
            widget.blockSignals(True)
            widget.setValue(int(value))
            widget.blockSignals(False)

    def _settings_widgets_changed(self, *_args: object) -> None:
        if self._syncing_widgets:
            return
        try:
            settings = self._settings_from_widgets().validated()
        except Exception as exc:  # noqa: BLE001 - surface as UI validation
            self.error_label.setText(str(exc))
            return
        try:
            self.controller.set_working_settings(settings)
        except BlobFeaturesUiError as exc:
            self.error_label.setText(str(exc))
            return
        self.error_label.setText("")
        self._refresh_preview_panel()
        self._update_enabled()

    def _reset_defaults(self) -> None:
        try:
            self.controller.reset_working_settings_to_defaults()
        except BlobFeaturesUiError as exc:
            QMessageBox.warning(self, "Cannot reset settings", str(exc))
            return
        self.refresh_from_state()
        self.status_message.emit("Segmentation settings restored to suite defaults.")

    def _step_frame(self, delta: int) -> None:
        try:
            self.controller.step_frame(delta)
        except BlobFeaturesUiError as exc:
            self.status_message.emit(str(exc))
            return
        self._refresh_frame_controls()
        self._refresh_preview_panel()
        self._update_enabled()

    def _frame_spin_changed(self, value: int) -> None:
        if self._syncing_widgets:
            return
        try:
            self.controller.set_frame_index(int(value))
        except BlobFeaturesUiError as exc:
            self.status_message.emit(str(exc))
            self._refresh_frame_controls()
            return
        self._refresh_frame_controls()
        self._refresh_preview_panel()

    def _frame_slider_changed(self, value: int) -> None:
        if self._syncing_widgets:
            return
        self.frame_spin.setValue(int(value))

    def _view_mode_changed(self, *_args: object) -> None:
        self._display_current_view()

    def _preview_current_frame(self) -> None:
        self._settings_widgets_changed()
        if self.has_running_task:
            self.status_message.emit("A preview task is already running.")
            return
        try:
            identity = self.controller.begin_preview_request()
        except BlobFeaturesUiError as exc:
            QMessageBox.warning(self, "Cannot preview", str(exc))
            self.error_label.setText(str(exc))
            return
        request_id = identity.request_id
        self._set_busy(True)
        self.status_message.emit(
            f"Previewing prepared frame {identity.prepared_frame_indices[0]}…"
        )

        def _run() -> SingleFramePreviewResult:
            return self.controller.compute_single_frame_preview(identity)

        try:
            self.task_runner.start(
                _run,
                task_name="S5 frame preview",
                on_success=lambda result: self._on_single_preview_success(
                    result, request_id
                ),
                on_error=lambda exc: self._on_preview_error(exc, request_id),
                on_finished=self._on_preview_finished,
            )
        except TaskAlreadyRunningError:
            self.controller.reject_preview(
                "Preview already running.", expected_request_id=request_id
            )
            self._set_busy(False)
            self.status_message.emit("A preview task is already running.")

    def _preview_bookmarks(self) -> None:
        self._settings_widgets_changed()
        bookmarks = self.controller.bookmark_list()
        if not bookmarks:
            QMessageBox.information(
                self, "No bookmarks", "Bookmark at least one prepared frame first."
            )
            return
        if self.has_running_task:
            self.status_message.emit("A preview task is already running.")
            return
        indices = [item.prepared_frame_idx for item in bookmarks]
        selected_before = int(self.controller.state.current_frame_idx)
        try:
            identity = self.controller.begin_preview_request(indices)
        except BlobFeaturesUiError as exc:
            QMessageBox.warning(self, "Cannot preview bookmarks", str(exc))
            self.error_label.setText(str(exc))
            return
        request_id = identity.request_id
        self._set_busy(True)
        self.progress.setRange(0, max(1, len(indices)))
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.busy_indicator.setVisible(False)
        self.status_message.emit(
            f"Previewing {len(indices)} bookmarked frame(s)…"
        )

        context = GuiTaskContext(
            task_id=uuid4().hex,
            task_kind=GuiTaskKind.BLOB_FEATURES_PREVIEW,
            project_dir=identity.session_root,
            raw_video_path=None,
            generation=0,
        )

        def _run(report: object) -> MultiFramePreviewResult:
            callback = report if callable(report) else None
            return self.controller.compute_multi_frame_preview(
                identity,
                progress_callback=callback,  # type: ignore[arg-type]
            )

        try:
            self.task_runner.start_progressive(
                _run,
                task_name="S5 bookmark preview",
                context=context,
                on_progress=self._on_bookmark_preview_progress,
                on_success=lambda result: self._on_multi_preview_success(
                    result, request_id, selected_before
                ),
                on_error=lambda exc: self._on_preview_error(exc, request_id),
                on_finished=self._on_preview_finished,
            )
        except TaskAlreadyRunningError:
            self.controller.reject_preview(
                "Preview already running.", expected_request_id=request_id
            )
            self.progress.setVisible(False)
            self._set_busy(False)

    def _on_single_preview_success(
        self, result: object, request_id: str
    ) -> None:
        if not isinstance(result, SingleFramePreviewResult):
            self.unexpected_error.emit("Unexpected single-frame preview result type.")
            return
        accepted = self.controller.accept_single_preview(
            result, expected_request_id=request_id
        )
        if not accepted:
            self.status_message.emit(
                "Discarded stale preview result (settings or source changed)."
            )
            self._refresh_preview_panel()
            return
        self.error_label.setText("")
        self._refresh_preview_panel()
        self.status_message.emit(
            f"Preview ready for frame {result.preview.prepared_frame_idx}."
        )

    def _on_bookmark_preview_progress(self, event: GuiTaskProgressEvent) -> None:
        if event.total_units is not None and event.total_units > 0:
            self.progress.setRange(0, int(event.total_units))
            if event.completed_units is not None:
                self.progress.setValue(int(event.completed_units))
        self.stage_label.setText(event.message)
        self.status_message.emit(event.message)

    def _on_multi_preview_success(
        self,
        result: object,
        request_id: str,
        selected_frame_idx: int,
    ) -> None:
        if not isinstance(result, MultiFramePreviewResult):
            self.unexpected_error.emit("Unexpected multi-frame preview result type.")
            return
        accepted = self.controller.accept_multi_preview(
            result, expected_request_id=request_id
        )
        if not accepted:
            self.status_message.emit(
                "Discarded stale bookmark preview (settings or source changed)."
            )
            self._refresh_preview_panel()
            return
        # Never jump to the first bookmark; restore the pre-batch selection.
        with contextlib.suppress(BlobFeaturesUiError):
            self.controller.set_frame_index(int(selected_frame_idx))
        self._multi_preview_index = self._index_of_multi_item(
            result, int(selected_frame_idx)
        )
        self.error_label.setText("")
        self._refresh_multi_preview_summary()
        self._refresh_frame_controls()
        self._refresh_preview_panel()
        ok_count = len(result.items)
        err_count = len(result.errors)
        message = f"Bookmark preview ready ({ok_count} frame(s) cached)."
        if err_count:
            details = "; ".join(
                f"frame {idx}: {err}" for idx, err in result.errors[:5]
            )
            message = f"{message} {err_count} error(s): {details}"
            self.error_label.setText(message)
        self.status_message.emit(message)
        self.stage_label.setText(message)

    def _index_of_multi_item(
        self, result: MultiFramePreviewResult, frame_idx: int
    ) -> int:
        for index, item in enumerate(result.items):
            if item.prepared_frame_idx == frame_idx:
                return index
        return 0

    def _refresh_multi_preview_summary(self) -> None:
        result = self.controller.state.last_multi_preview
        if result is None or not result.items:
            self.multi_summary_label.setText("No batch preview.")
            return
        index = max(0, min(self._multi_preview_index, len(result.items) - 1))
        item = result.items[index]
        summary = self.controller.diagnostics_summary(item.preview)
        err_note = ""
        if result.errors:
            err_note = f"\nBatch errors: {len(result.errors)}"
        self.multi_summary_label.setText(
            f"Cached {len(result.items)} bookmarked frame(s). "
            f"Showing result {index + 1}/{len(result.items)} · "
            f"frame {item.prepared_frame_idx}\n"
            f"cc_count_raw={summary['cc_count_raw']} · "
            f"cc_count={summary['cc_count']}\n"
            f"warnings: {summary['warnings']}{err_note}\n"
            "Navigate bookmarked frames to view their cached previews; "
            "selection is not forced to the first bookmark."
        )

    def _on_preview_error(self, exc: BaseException, request_id: str) -> None:
        _LOGGER.exception("S5 preview failed: %s", exc)
        self.controller.reject_preview(exc, expected_request_id=request_id)
        message = str(exc)
        self.error_label.setText(message)
        self._refresh_preview_panel()
        if isinstance(exc, (BlobFeaturesUiError,)):
            self.status_message.emit(message)
            return
        from blob_features import BlobFeaturesError
        from preprocess.exceptions import VideoProbeError

        if isinstance(exc, (BlobFeaturesError, VideoProbeError)):
            self.status_message.emit(message)
            return
        self.unexpected_error.emit(message)

    def _on_preview_finished(self) -> None:
        self.progress.setVisible(False)
        self._set_busy(False)
        self.controller.state.task_running = False
        self._update_enabled()
        self._emit_running_changed()
        self.task_finished.emit()

    def _refresh_discovery(self) -> None:
        if not self._guard_idle("refresh discovery"):
            return
        try:
            self.controller.refresh_discovery()
        except BlobFeaturesUiError as exc:
            QMessageBox.warning(self, "Discovery failed", str(exc))
            self.status_message.emit(str(exc))
            return
        self.refresh_from_state()
        self.status_message.emit("S5 run discovery refreshed.")

    def _run_extraction(self) -> None:
        if not self._guard_idle("start full extraction"):
            return
        self._settings_widgets_changed()
        try:
            request, identity = self.controller.build_request()
        except BlobFeaturesUiError as exc:
            QMessageBox.warning(self, "Cannot run extraction", str(exc))
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return

        self._active_extraction_identity = identity
        self._started_at = time.perf_counter()
        self.stage_label.setText("Running blob feature extraction…")
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.busy_indicator.setVisible(False)
        self.elapsed_timer.start()
        self._update_enabled()
        self._emit_running_changed()
        self.status_message.emit(
            "S5 full extraction started (settings snapshotted; cancellation unsupported)."
        )

        context = GuiTaskContext(
            task_id=uuid4().hex,
            task_kind=GuiTaskKind.BLOB_FEATURES_EXTRACTION,
            project_dir=identity.session_root,
            raw_video_path=None,
            generation=0,
        )

        def _operation(report: object) -> BlobFeaturesResult:
            return self.controller.execute_request(
                request, progress_callback=report  # type: ignore[arg-type]
            )

        try:
            self.task_runner.start_progressive(
                _operation,
                task_name="S5 blob feature extraction",
                context=context,
                on_progress=lambda event: self._on_extraction_progress(event, identity),
                on_success=lambda result: self._on_extraction_success(result, identity),
                on_error=lambda exc: self._on_extraction_error(exc, identity),
                on_finished=lambda: self._on_extraction_finished(identity),
            )
        except TaskAlreadyRunningError:
            self.controller.record_failure(
                "Extraction already running.", expected_identity=identity
            )
            self.progress.setVisible(False)
            self.elapsed_timer.stop()
            self._update_enabled()
            self._emit_running_changed()
            QMessageBox.warning(
                self,
                "Task already running",
                "A Subsystem 5 task is already active.",
            )

    def _on_extraction_progress(
        self,
        event: object,
        identity: ExtractionRequestIdentity,
    ) -> None:
        if self._active_extraction_identity is None:
            return
        if identity.request_id != self._active_extraction_identity.request_id:
            return
        if not isinstance(event, GuiTaskProgressEvent):
            return
        self.stage_label.setText(format_pipeline_progress(event))
        if (
            not event.is_indeterminate
            and event.total_units is not None
            and event.completed_units is not None
        ):
            self.progress.setRange(0, int(event.total_units))
            self.progress.setValue(int(event.completed_units))
        else:
            self.progress.setRange(0, 0)

    def _on_extraction_success(
        self, result: object, identity: ExtractionRequestIdentity
    ) -> None:
        if not isinstance(result, BlobFeaturesResult):
            self.unexpected_error.emit(f"Unexpected S5 result type: {type(result)!r}")
            return
        accepted = self.controller.record_success(result, expected_identity=identity)
        if not accepted:
            self.status_message.emit(
                "Discarded stale extraction result (S1 source changed during the run)."
            )
            self.refresh_from_state()
            return
        self.stage_label.setText(f"Finished: {result.status}")
        self.refresh_from_state()
        warnings = f" ({len(result.warnings)} warnings)" if result.warnings else ""
        self.status_message.emit(
            f"S5 {result.status}{warnings}. Output: {result.blob_features_path}"
        )

    def _on_extraction_error(
        self, exc: BaseException, identity: ExtractionRequestIdentity
    ) -> None:
        _LOGGER.exception("S5 extraction failed: %s", exc)
        message = str(exc)
        accepted = self.controller.record_failure(message, expected_identity=identity)
        if not accepted:
            self.status_message.emit("Ignored stale extraction failure after source change.")
            return
        self.stage_label.setText(f"Failed: {message}")
        with contextlib.suppress(BlobFeaturesUiError):
            self.controller.refresh_discovery()
        self.refresh_from_state()
        from blob_features import BlobFeaturesError

        if isinstance(exc, (BlobFeaturesUiError, BlobFeaturesError)):
            QMessageBox.warning(self, "Blob feature extraction failed", message)
            self.status_message.emit(f"S5 failed: {message}")
        else:
            self.unexpected_error.emit(message)
            self.status_message.emit(f"S5 failed: {message}")

    def _on_extraction_finished(
        self, identity: ExtractionRequestIdentity
    ) -> None:
        if (
            self._active_extraction_identity is not None
            and identity.request_id == self._active_extraction_identity.request_id
        ):
            self._active_extraction_identity = None
        self.controller.state.task_running = False
        self.progress.setVisible(False)
        self.elapsed_timer.stop()
        self._started_at = None
        self._update_enabled()
        self._emit_running_changed()
        self.task_finished.emit()

    def _open_output_folder(self) -> None:
        try:
            opened = self.controller.open_selected_s5_folder()
        except BlobFeaturesUiError as exc:
            QMessageBox.information(self, "No S5 output", str(exc))
            self.status_message.emit(str(exc))
            return
        if not opened:
            self.status_message.emit("Could not open the S5 run folder.")
        else:
            self.status_message.emit("Opened S5 run folder.")

    def _populate_runs_table(self) -> None:
        summary = self.controller.state.summary
        selected = self.controller.state.selected_s5_run_id
        self.runs_table.blockSignals(True)
        self.runs_table.setRowCount(0)
        selected_row = -1
        if summary is not None:
            for row, run in enumerate(summary.runs):
                self.runs_table.insertRow(row)
                values = (
                    run.run_id,
                    run.status or "—",
                    run.classification,
                    str(run.warning_count),
                )
                for col, value in enumerate(values):
                    item = QTableWidgetItem(value)
                    item.setData(Qt.ItemDataRole.UserRole, run.run_id)
                    self.runs_table.setItem(row, col, item)
                if selected is not None and run.run_id == selected:
                    selected_row = row
        self.runs_table.blockSignals(False)
        if selected_row >= 0:
            self.runs_table.selectRow(selected_row)

    def _s5_selection_changed(self) -> None:
        items = self.runs_table.selectedItems()
        if not items:
            return
        run_id = items[0].data(Qt.ItemDataRole.UserRole)
        try:
            self.controller.select_s5_run(str(run_id) if run_id else None)
        except BlobFeaturesUiError as exc:
            self.status_message.emit(str(exc))
            return
        self._refresh_qc_panel()
        self._update_enabled()

    def _refresh_qc_panel(self) -> None:
        rows = self.controller.qc_display_rows()
        self.qc_table.setRowCount(len(rows))
        for index, (name, value) in enumerate(rows):
            self.qc_table.setItem(index, 0, QTableWidgetItem(name))
            self.qc_table.setItem(index, 1, QTableWidgetItem(value))
        state = self.controller.state
        if state.run_error and not rows:
            self.stage_label.setText(f"Failed: {state.run_error}")
        elif not self.has_running_task:
            self.stage_label.setText(state.run_stage)

    def _update_elapsed(self) -> None:
        if self._started_at is None:
            self.elapsed_label.setText("Elapsed: —")
            return
        elapsed = time.perf_counter() - self._started_at
        self.elapsed_label.setText(f"Elapsed: {format_task_duration(elapsed)}")

    def _guard_idle(self, action: str) -> bool:
        if self.has_running_task:
            self.status_message.emit(f"Cannot {action} while a task is active.")
            return False
        return True

    def _add_bookmark(self) -> None:
        try:
            self.controller.add_current_bookmark(self.bookmark_note.text())
        except BlobFeaturesUiError as exc:
            QMessageBox.warning(self, "Cannot bookmark", str(exc))
            return
        self.bookmark_note.clear()
        self._refresh_bookmarks()
        self.status_message.emit("Bookmarked current prepared frame.")

    def _remove_bookmark(self) -> None:
        item = self.bookmark_list.currentItem()
        if item is None:
            return
        frame_idx = int(item.data(Qt.ItemDataRole.UserRole))
        try:
            self.controller.remove_bookmark_at(frame_idx)
        except BlobFeaturesUiError as exc:
            QMessageBox.warning(self, "Cannot remove bookmark", str(exc))
            return
        self._refresh_bookmarks()

    def _goto_bookmark(self) -> None:
        item = self.bookmark_list.currentItem()
        if item is None:
            return
        self._activate_bookmark_frame(int(item.data(Qt.ItemDataRole.UserRole)))

    def _bookmark_activated(self, item: QListWidgetItem) -> None:
        self._activate_bookmark_frame(int(item.data(Qt.ItemDataRole.UserRole)))

    def _activate_bookmark_frame(self, frame_idx: int) -> None:
        try:
            self.controller.go_to_bookmark(frame_idx)
        except BlobFeaturesUiError as exc:
            self.status_message.emit(str(exc))
            return
        self._refresh_frame_controls()
        self._refresh_preview_panel()

    def _step_multi_preview(self, delta: int) -> None:
        result = self.controller.state.last_multi_preview
        if result is None or not result.items:
            return
        nxt = int(self._multi_preview_index) + int(delta)
        nxt = max(0, min(len(result.items) - 1, nxt))
        self._apply_multi_preview_item(nxt)

    def _apply_multi_preview_item(self, index: int) -> None:
        result = self.controller.state.last_multi_preview
        if result is None or not result.items:
            self.multi_summary_label.setText("No batch preview.")
            return
        self._multi_preview_index = index
        item = result.items[index]
        with contextlib.suppress(BlobFeaturesUiError):
            self.controller.set_frame_index(item.prepared_frame_idx)
        self._refresh_multi_preview_summary()
        self._refresh_frame_controls()
        self._refresh_preview_panel()

    def _refresh_eligibility_labels(self) -> None:
        state = self.controller.state
        if state.eligible is not None:
            eligible = state.eligible
            self.eligibility_label.setText(
                f"Eligible S1 · {eligible.frame_count} prepared frames · "
                f"{eligible.frame_width}×{eligible.frame_height} · "
                f"video={eligible.prepared_video.name}"
            )
            return
        if state.eligibility_error:
            self.eligibility_label.setText(state.eligibility_error)
        else:
            self.eligibility_label.setText(
                "Select a project/session with eligible S1 (prepared video, "
                "background, sync, prepare_meta)."
            )

    def _refresh_frame_controls(self) -> None:
        state = self.controller.state
        self._syncing_widgets = True
        try:
            if state.eligible is None:
                self.frame_spin.setMaximum(0)
                self.frame_slider.setMaximum(0)
                self.frame_spin.setValue(0)
                self.frame_slider.setValue(0)
                self.frame_meta_label.setText("No eligible S1 session loaded.")
            else:
                max_idx = max(0, state.eligible.frame_count - 1)
                self.frame_spin.setMaximum(max_idx)
                self.frame_slider.setMaximum(max_idx)
                self.frame_spin.setValue(int(state.current_frame_idx))
                self.frame_slider.setValue(int(state.current_frame_idx))
                ts = self.controller.current_timestamp_sec()
                ts_text = f"{ts:.6f} s" if ts is not None else "—"
                self.frame_meta_label.setText(
                    f"Prepared frame {state.current_frame_idx} / {max_idx} · "
                    f"timestamp {ts_text} (prepared-video coordinates; not resampled to 120 Hz)"
                )
        finally:
            self._syncing_widgets = False
        self._display_current_view()

    def _refresh_bookmarks(self) -> None:
        self.bookmark_list.clear()
        for bookmark in self.controller.bookmark_list():
            label = f"frame {bookmark.prepared_frame_idx}"
            if bookmark.note:
                label = f"{label} — {bookmark.note}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, int(bookmark.prepared_frame_idx))
            self.bookmark_list.addItem(item)

    def _refresh_preview_panel(self) -> None:
        state = self.controller.state
        summary = self.controller.diagnostics_summary()
        self.cc_raw_label.setText(summary["cc_count_raw"])
        self.cc_effective_label.setText(summary["cc_count"])
        self.min_dist_label.setText(summary["min_inter_contour_dist"])
        self.warnings_label.setText(summary["warnings"])
        has_current = self.controller.current_frame_has_valid_preview()
        if state.preview_error:
            self.preview_status_label.setText(f"Error: {state.preview_error}")
        elif has_current:
            self.preview_status_label.setText(
                "Preview valid for this frame under the current settings snapshot."
            )
        elif self.controller.preview_is_stale():
            self.preview_status_label.setText(
                "Preview stale or missing for this frame — click Preview Frame "
                "or Preview Bookmarks."
            )
        else:
            self.preview_status_label.setText(
                "No preview for this frame yet — click Preview Frame."
            )
        if state.last_multi_preview is None:
            self.multi_summary_label.setText("No batch preview.")
        elif state.last_multi_preview.items:
            self._refresh_multi_preview_summary()
        self._display_current_view()

    def _display_current_view(self) -> None:
        state = self.controller.state
        mode = self.view_mode.currentData()
        image: np.ndarray | None = None
        requires_preview = mode in (
            _VIEW_FOREGROUND,
            _VIEW_BINARY,
            _VIEW_CONTOURS,
        )
        cached = self.controller.preview_for_current_frame()
        preview_required = False
        if mode == _VIEW_BACKGROUND and state.background_bgr is not None:
            image = state.background_bgr
        elif mode == _VIEW_ORIGINAL and state.current_frame_bgr is not None:
            image = state.current_frame_bgr
        elif requires_preview and cached is not None:
            if mode == _VIEW_FOREGROUND:
                image = cached.preview.foreground_gray
            elif mode == _VIEW_BINARY:
                image = cached.preview.mask
            else:
                image = cached.contour_overlay_bgr
        elif requires_preview:
            # Never silently substitute the original frame for processed views.
            preview_required = True
            image = None
        self.preview_required_label.setVisible(preview_required)
        if image is None:
            self.frame_view.set_frame(None)
            return
        try:
            self.frame_view.set_frame(ensure_bgr(image))
        except BlobFeaturesUiError as exc:
            self.error_label.setText(str(exc))

    def _set_busy(self, busy: bool) -> None:
        extraction = self.controller.state.active_extraction_identity is not None
        # Keep the indeterminate bar for short previews; extraction uses progress.
        self.busy_indicator.setVisible(busy and not extraction)
        self._update_enabled()
        self._emit_running_changed()

    def _update_enabled(self) -> None:
        ready = self.controller.state.eligible is not None
        busy = self.has_running_task
        preview_busy = self.controller.state.active_preview_request_id is not None
        # During full extraction, keep frame navigation available so users can
        # inspect the source while work continues. Session mutation and new tasks
        # remain locked.
        frame_locked = preview_busy
        self.preview_button.setEnabled(ready and not busy)
        self.preview_bookmarks_button.setEnabled(ready and not busy)
        self.prev_button.setEnabled(ready and not frame_locked)
        self.next_button.setEnabled(ready and not frame_locked)
        self.frame_spin.setEnabled(ready and not frame_locked)
        self.frame_slider.setEnabled(ready and not frame_locked)
        self.add_bookmark_button.setEnabled(ready and not busy)
        self.remove_bookmark_button.setEnabled(ready and not busy)
        self.goto_bookmark_button.setEnabled(ready and not frame_locked)
        self.open_session_button.setEnabled(not busy)
        self.session_browse_button.setEnabled(not busy)
        self.refresh_button.setEnabled(not busy)
        self.reset_defaults_button.setEnabled(not busy)
        can_run = self.controller.can_run_extraction() and not busy
        self.run_button.setEnabled(can_run)
        has_output = (
            self.controller.selected_s5_run() is not None
            or self.controller.state.last_result is not None
        )
        self.open_output_button.setEnabled(has_output and not busy)
        multi = self.controller.state.last_multi_preview
        has_multi = multi is not None and bool(multi.items)
        self.multi_prev_button.setEnabled(has_multi and not frame_locked)
        self.multi_next_button.setEnabled(has_multi and not frame_locked)

    def _emit_running_changed(self) -> None:
        running = self.has_running_task
        if running != self._last_running_state:
            self._last_running_state = running
            self.task_running_changed.emit(running)
