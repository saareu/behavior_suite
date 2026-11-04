"""Shared QC Overlay Export dialog (Increment C2-C).

Hosts one GuiTaskRunner for overlay QC exports so S9 CSV/media and S10 PI-report
workflows stay independent. Pages open this dialog with an explicit launch context.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from overlay.contracts import OverlayExportCancelled
from overlay.discovery import (
    EXPORT_FAMILY_BEHAVIORAL,
    EXPORT_FAMILY_MIXED,
    EXPORT_FAMILY_S2_POSE,
    EXPORT_FAMILY_S3_POSE,
    EXPORT_FAMILY_S4_POSE,
    EXPORT_FAMILY_S5_SEGMENTATION,
)
from ui.controllers.overlay_qc_export_controller import (
    BOUT_COMBO_LIMIT,
    OverlayQcExportController,
    OverlayQcExportUiError,
)
from ui.controllers.overlay_qc_export_support import (
    OverlayQcLaunchContext,
    OverlayQcRequestIdentity,
    OverlayQcUiSettings,
    actionable_overlay_error,
)
from ui.tasks import (
    GuiTaskContext,
    GuiTaskKind,
    GuiTaskProgressEvent,
    GuiTaskRunner,
    TaskAlreadyRunningError,
    format_task_duration,
)

_JOB_COLUMNS = (
    "Job ID",
    "Family",
    "Status",
    "Source",
    "Prepared range",
    "Layers",
    "Frames",
    "Video",
    "Path",
)


class OverlayQcExportDialog(QDialog):
    """Modal QC overlay exporter with discovery, progress, and cancel."""

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        controller: OverlayQcExportController | None = None,
        launch: OverlayQcLaunchContext | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("QC Overlay Video Export")
        self.resize(920, 720)
        self.controller = controller or OverlayQcExportController()
        self.task_runner = GuiTaskRunner(self)
        self._launch = launch
        self._active_identity: OverlayQcRequestIdentity | None = None
        self._syncing = False
        self._build_ui()
        if launch is not None:
            self.apply_launch_context(launch)

    def apply_launch_context(self, launch: OverlayQcLaunchContext) -> None:
        self._launch = launch
        self.controller.set_session(launch.session_root)
        self._populate_source_combos(launch)
        self._configure_source_kind_combo(launch)
        self._syncing = True
        try:
            family_idx = self.family_combo.findData(launch.export_family)
            if family_idx >= 0:
                self.family_combo.setCurrentIndex(family_idx)
            self.family_combo.setEnabled(not launch.lock_family)
            if (
                launch.export_family
                in {EXPORT_FAMILY_S4_POSE, EXPORT_FAMILY_MIXED}
                and launch.s4_run_id
            ):
                self._select_combo_by_run_id(self.s4_combo, launch.s4_run_id)
            if launch.export_family == EXPORT_FAMILY_S2_POSE and launch.s2_run_id:
                self._select_combo_by_run_id(self.s2_combo, launch.s2_run_id)
            if (
                launch.export_family
                in {EXPORT_FAMILY_S3_POSE, EXPORT_FAMILY_MIXED}
                and launch.s3_run_id
            ):
                self._select_combo_by_run_id(self.s3_combo, launch.s3_run_id)
            if (
                launch.export_family == EXPORT_FAMILY_S5_SEGMENTATION
                and launch.s5_run_id
            ):
                self._select_combo_by_run_id(self.s5_combo, launch.s5_run_id)
            if launch.package_run_id:
                self._select_combo_by_run_id(self.package_combo, launch.package_run_id)
            if launch.review_run_id:
                self._select_combo_by_run_id(self.review_combo, launch.review_run_id)
            source_kind = self._resolved_launch_source_kind(launch)
            if launch.export_family in {
                EXPORT_FAMILY_BEHAVIORAL,
                EXPORT_FAMILY_MIXED,
            }:
                idx = self.source_kind_combo.findData(source_kind)
                if idx >= 0:
                    self.source_kind_combo.setCurrentIndex(idx)
            if (
                launch.export_family == EXPORT_FAMILY_MIXED
                and launch.pose_source_kind
            ):
                pose_idx = self.pose_source_combo.findData(launch.pose_source_kind)
                if pose_idx >= 0:
                    self.pose_source_combo.setCurrentIndex(pose_idx)
            if launch.preferred_prepared_start is not None:
                self.prep_start_spin.setValue(int(launch.preferred_prepared_start))
            if launch.preferred_prepared_end is not None:
                self.prep_end_spin.setValue(int(launch.preferred_prepared_end))
            if (
                launch.preferred_bout_id
                or launch.preferred_start_sample_idx is not None
            ):
                self.scope_combo.setCurrentIndex(self.scope_combo.findData("bout_span"))
            self._refresh_bouts()
            self._apply_preferred_bout(launch)
        finally:
            self._syncing = False
        self._on_family_changed()
        self._on_scope_changed()
        self._apply_settings_from_widgets()
        self._refresh_jobs_table()
        self._update_enabled()

    @staticmethod
    def _resolved_launch_source_kind(launch: OverlayQcLaunchContext) -> str:
        if launch.lock_source_kind:
            locked = launch.locked_source_kind or "s8_automated"
            return str(locked)
        if launch.review_run_id and launch.review_is_draft is False:
            return "s10_reviewed"
        return "s8_automated"

    def _configure_source_kind_combo(self, launch: OverlayQcLaunchContext) -> None:
        """Offer only the locked source when entry context forbids switching."""

        current = self.source_kind_combo.currentData()
        self.source_kind_combo.blockSignals(True)
        self.source_kind_combo.clear()
        if launch.lock_source_kind:
            locked = str(launch.locked_source_kind or "s8_automated")
            label = (
                "Automated S8 via S9 package"
                if locked == "s8_automated"
                else "Published S10 reviewed"
            )
            self.source_kind_combo.addItem(label, locked)
        else:
            self.source_kind_combo.addItem(
                "Automated S8 via S9 package", "s8_automated"
            )
            self.source_kind_combo.addItem("Published S10 reviewed", "s10_reviewed")
            if current is not None:
                idx = self.source_kind_combo.findData(current)
                if idx >= 0:
                    self.source_kind_combo.setCurrentIndex(idx)
        self.source_kind_combo.blockSignals(False)

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Export a short prepared-rate QC overlay MP4. Sources must be selected "
            "explicitly (never auto-picks newest). Full-session export is not available."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        settings_group = QGroupBox("Export settings")
        form = QFormLayout(settings_group)

        self.family_combo = QComboBox()
        self.family_combo.addItem("Behavioral QC (S8/S10)", EXPORT_FAMILY_BEHAVIORAL)
        self.family_combo.addItem("S4 finalized-pose QC", EXPORT_FAMILY_S4_POSE)
        self.family_combo.addItem("S2 inferred-pose QC", EXPORT_FAMILY_S2_POSE)
        self.family_combo.addItem("S3 accepted tracked-pose QC", EXPORT_FAMILY_S3_POSE)
        self.family_combo.addItem(
            "S5 blob-segmentation QC", EXPORT_FAMILY_S5_SEGMENTATION
        )
        self.family_combo.addItem(
            "Mixed pose + behavioral QC", EXPORT_FAMILY_MIXED
        )
        self.family_combo.currentIndexChanged.connect(self._on_family_changed)
        form.addRow("Export type", self.family_combo)

        self.pose_source_combo = QComboBox()
        self.pose_source_combo.addItem("S4 finalized pose", "s4_finalized")
        self.pose_source_combo.addItem("S3 accepted tracked pose", "s3_accepted")
        self.pose_source_combo.currentIndexChanged.connect(self._on_pose_source_changed)
        form.addRow("Mixed pose source", self.pose_source_combo)

        self.source_kind_combo = QComboBox()
        self.source_kind_combo.addItem("Automated S8 via S9 package", "s8_automated")
        self.source_kind_combo.addItem("Published S10 reviewed", "s10_reviewed")
        self.source_kind_combo.currentIndexChanged.connect(self._on_source_kind_changed)
        form.addRow("Behavioral source", self.source_kind_combo)

        self.s4_provenance_label = QLabel(
            "S4 provenance styling: observed / temporal_resample / gap_fill; "
            "missing and protected blanks stay undrawn."
        )
        self.s4_provenance_label.setWordWrap(True)
        form.addRow("", self.s4_provenance_label)

        self.package_combo = QComboBox()
        self.package_combo.currentIndexChanged.connect(self._on_package_changed)
        form.addRow("S9 package", self.package_combo)

        self.review_combo = QComboBox()
        self.review_combo.currentIndexChanged.connect(self._apply_settings_from_widgets)
        form.addRow("S10 review", self.review_combo)

        self.s4_combo = QComboBox()
        self.s4_combo.currentIndexChanged.connect(self._apply_settings_from_widgets)
        form.addRow("S4 run", self.s4_combo)

        self.s2_combo = QComboBox()
        self.s2_combo.currentIndexChanged.connect(self._apply_settings_from_widgets)
        form.addRow("S2 run", self.s2_combo)

        self.s3_combo = QComboBox()
        self.s3_combo.currentIndexChanged.connect(self._apply_settings_from_widgets)
        form.addRow("S3 run", self.s3_combo)

        self.s5_combo = QComboBox()
        self.s5_combo.currentIndexChanged.connect(self._apply_settings_from_widgets)
        form.addRow("S5 run", self.s5_combo)

        self.scope_combo = QComboBox()
        self.scope_combo.addItem("Prepared-frame interval", "prepared_interval")
        self.scope_combo.addItem("Selected behavior bout", "bout_span")
        self.scope_combo.currentIndexChanged.connect(self._on_scope_changed)
        form.addRow("Scope", self.scope_combo)

        self.prep_start_spin = QSpinBox()
        self.prep_start_spin.setRange(0, 10_000_000)
        self.prep_end_spin = QSpinBox()
        self.prep_end_spin.setRange(0, 10_000_000)
        self.prep_end_spin.setValue(29)
        prep_row = QHBoxLayout()
        prep_row.addWidget(self.prep_start_spin)
        prep_row.addWidget(QLabel("to"))
        prep_row.addWidget(self.prep_end_spin)
        form.addRow("Prepared frames (inclusive)", prep_row)

        self.bout_combo = QComboBox()
        self.bout_combo.currentIndexChanged.connect(self._on_bout_changed)
        form.addRow("Bout", self.bout_combo)
        self.bout_limit_label = QLabel(
            f"Combo lists the first {BOUT_COMBO_LIMIT} exclusive packaged bouts "
            "(by dense start). Dense endpoints stay editable; S10 selections "
            "outside that window are injected into the list."
        )
        self.bout_limit_label.setWordWrap(True)
        form.addRow("", self.bout_limit_label)

        self.start_sample_spin = QSpinBox()
        self.start_sample_spin.setRange(0, 10_000_000)
        self.end_sample_spin = QSpinBox()
        self.end_sample_spin.setRange(0, 10_000_000)
        dense_row = QHBoxLayout()
        dense_row.addWidget(self.start_sample_spin)
        dense_row.addWidget(QLabel("to"))
        dense_row.addWidget(self.end_sample_spin)
        form.addRow("Dense samples (inclusive)", dense_row)

        self.hud_combo = QComboBox()
        self.hud_combo.addItem("Both directions (default)", "both")
        self.hud_combo.addItem("Focal 0 only", "focal_0_only")
        self.hud_combo.addItem("Focal 1 only", "focal_1_only")
        form.addRow("Behavioral perspective", self.hud_combo)

        self.include_contact = QCheckBox("Contact LED")
        self.include_contact.setChecked(True)
        self.include_sbs = QCheckBox("Side-by-side LED")
        self.include_sbs.setChecked(True)
        self.include_counter = QCheckBox("Frame/time counter")
        self.include_counter.setChecked(True)
        self.draw_skeleton = QCheckBox("Draw skeleton (when topology available)")
        self.draw_skeleton.setChecked(True)
        self.show_confidence = QCheckBox(
            "Confidence styling (SLEAP score fill; missing ≠ zero)"
        )
        self.show_confidence.setChecked(False)
        self.show_machine_corrections = QCheckBox(
            "Automatic correction event indicators (machine_corrections.json)"
        )
        self.show_machine_corrections.setChecked(True)
        self.show_manual_corrections = QCheckBox(
            "Manual correction event indicators (manual_corrections.json)"
        )
        self.show_manual_corrections.setChecked(True)
        self.show_contours = QCheckBox("Draw S5 segmentation contours (default)")
        self.show_contours.setChecked(True)
        self.show_masks = QCheckBox("Draw translucent S5 segmentation masks")
        self.show_masks.setChecked(False)
        self.show_scalar_text = QCheckBox(
            "Show replay scalar text (cc_count / min distance)"
        )
        self.show_scalar_text.setChecked(False)
        self.mask_opacity_spin = QDoubleSpinBox()
        self.mask_opacity_spin.setRange(0.0, 1.0)
        self.mask_opacity_spin.setSingleStep(0.05)
        self.mask_opacity_spin.setDecimals(2)
        self.mask_opacity_spin.setValue(0.35)
        self.nodes_only_label = QLabel("")
        self.nodes_only_label.setWordWrap(True)
        self.nodes_only_label.setStyleSheet("color: #666;")
        form.addRow(self.include_contact)
        form.addRow(self.include_sbs)
        form.addRow(self.include_counter)
        form.addRow(self.draw_skeleton)
        form.addRow(self.show_confidence)
        form.addRow(self.show_machine_corrections)
        form.addRow(self.show_manual_corrections)
        form.addRow(self.show_contours)
        form.addRow(self.show_masks)
        form.addRow(self.show_scalar_text)
        form.addRow("Mask opacity", self.mask_opacity_spin)
        form.addRow(self.nodes_only_label)

        self.pre_context_spin = QDoubleSpinBox()
        self.pre_context_spin.setRange(0.0, 30.0)
        self.pre_context_spin.setDecimals(2)
        self.post_context_spin = QDoubleSpinBox()
        self.post_context_spin.setRange(0.0, 30.0)
        self.post_context_spin.setDecimals(2)
        pad_row = QHBoxLayout()
        pad_row.addWidget(self.pre_context_spin)
        pad_row.addWidget(QLabel("/"))
        pad_row.addWidget(self.post_context_spin)
        form.addRow("Pre/post context (sec)", pad_row)

        self.source_summary_label = QLabel("No QC overlay source selected.")
        self.source_summary_label.setWordWrap(True)
        form.addRow("Source", self.source_summary_label)
        self.output_label = QLabel("")
        self.output_label.setWordWrap(True)
        self.output_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        form.addRow("Output", self.output_label)
        layout.addWidget(settings_group)

        actions = QHBoxLayout()
        self.export_button = QPushButton("Export QC video")
        self.export_button.clicked.connect(self._export)
        self.cancel_button = QPushButton("Cancel export")
        self.cancel_button.clicked.connect(self._cancel)
        self.refresh_button = QPushButton("Refresh jobs")
        self.refresh_button.clicked.connect(self._refresh_jobs_table)
        self.open_folder_button = QPushButton("Open output folder")
        self.open_folder_button.clicked.connect(self._open_folder)
        self.open_video_button = QPushButton("Open MP4")
        self.open_video_button.clicked.connect(self._open_video)
        actions.addWidget(self.export_button)
        actions.addWidget(self.cancel_button)
        actions.addWidget(self.refresh_button)
        actions.addWidget(self.open_folder_button)
        actions.addWidget(self.open_video_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        status_group = QGroupBox("Progress")
        status_layout = QVBoxLayout(status_group)
        self.stage_label = QLabel("Idle")
        self.stage_label.setWordWrap(True)
        self.elapsed_label = QLabel("Elapsed: —")
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        status_layout.addWidget(self.stage_label)
        status_layout.addWidget(self.elapsed_label)
        status_layout.addWidget(self.progress)
        layout.addWidget(status_group)

        jobs_group = QGroupBox("Existing QC overlay jobs (this session)")
        jobs_layout = QVBoxLayout(jobs_group)
        self.jobs_table = QTableWidget(0, len(_JOB_COLUMNS))
        self.jobs_table.setHorizontalHeaderLabels(list(_JOB_COLUMNS))
        self.jobs_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.jobs_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.jobs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.jobs_table.horizontalHeader().setStretchLastSection(True)
        self.jobs_table.itemSelectionChanged.connect(self._job_selection_changed)
        self.jobs_table.setMaximumHeight(220)
        jobs_layout.addWidget(self.jobs_table)
        layout.addWidget(jobs_group)

        close_row = QHBoxLayout()
        close_row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.reject)
        close_row.addWidget(close_btn)
        layout.addLayout(close_row)

        for widget in (
            self.prep_start_spin,
            self.prep_end_spin,
            self.start_sample_spin,
            self.end_sample_spin,
            self.pre_context_spin,
            self.post_context_spin,
            self.hud_combo,
            self.include_contact,
            self.include_sbs,
            self.include_counter,
            self.draw_skeleton,
            self.show_confidence,
            self.show_machine_corrections,
            self.show_manual_corrections,
            self.show_contours,
            self.show_masks,
            self.show_scalar_text,
            self.mask_opacity_spin,
        ):
            if isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self._apply_settings_from_widgets)
            elif isinstance(widget, QCheckBox):
                widget.stateChanged.connect(self._apply_settings_from_widgets)
            else:
                widget.valueChanged.connect(self._apply_settings_from_widgets)

    @staticmethod
    def _select_combo_by_run_id(combo: QComboBox, run_id: str) -> None:
        for idx in range(combo.count()):
            data = combo.itemData(idx)
            if data is None:
                continue
            if isinstance(data, tuple) and data and str(data[0]) == str(run_id):
                combo.setCurrentIndex(idx)
                return
            if str(data) == str(run_id):
                combo.setCurrentIndex(idx)
                return

    def _populate_source_combos(self, launch: OverlayQcLaunchContext) -> None:
        self.package_combo.blockSignals(True)
        self.review_combo.blockSignals(True)
        self.s4_combo.blockSignals(True)
        self.s2_combo.blockSignals(True)
        self.s3_combo.blockSignals(True)
        self.s5_combo.blockSignals(True)
        self.package_combo.clear()
        self.review_combo.clear()
        self.s4_combo.clear()
        self.s2_combo.clear()
        self.package_combo.addItem("(select S9 package)", None)
        for run_id, path in launch.package_options:
            self.package_combo.addItem(run_id, (run_id, path))
        if (
            launch.package_run_id
            and launch.package_dir is not None
            and self.package_combo.findData(
                (launch.package_run_id, launch.package_dir)
            )
            < 0
        ):
            self.package_combo.addItem(
                launch.package_run_id, (launch.package_run_id, launch.package_dir)
            )
        self.review_combo.addItem("(none / automated only)", None)
        for run_id, path, is_draft, parent in launch.review_options:
            label = f"{run_id}{' [draft]' if is_draft else ''}"
            self.review_combo.addItem(label, (run_id, path, is_draft, parent))
        if launch.review_run_id and launch.review_run_dir is not None:
            data = (
                launch.review_run_id,
                launch.review_run_dir,
                bool(launch.review_is_draft),
                launch.package_run_id,
            )
            if self.review_combo.findData(data) < 0:
                self.review_combo.addItem(
                    f"{launch.review_run_id}"
                    f"{' [draft]' if launch.review_is_draft else ''}",
                    data,
                )
        self.s4_combo.addItem("(select S4 run)", None)
        for run_id, path in launch.s4_options:
            self.s4_combo.addItem(run_id, (run_id, path))
        if (
            launch.s4_run_id
            and launch.s4_run_dir is not None
            and self.s4_combo.findData((launch.s4_run_id, launch.s4_run_dir)) < 0
        ):
            self.s4_combo.addItem(
                launch.s4_run_id, (launch.s4_run_id, launch.s4_run_dir)
            )
        self.s2_combo.addItem("(select S2 run)", None)
        for run_id, path in launch.s2_options:
            self.s2_combo.addItem(run_id, (run_id, path))
        if (
            launch.s2_run_id
            and launch.s2_run_dir is not None
            and self.s2_combo.findData((launch.s2_run_id, launch.s2_run_dir)) < 0
        ):
            self.s2_combo.addItem(
                launch.s2_run_id, (launch.s2_run_id, launch.s2_run_dir)
            )
        if launch.skeleton_nodes_only_disclosure:
            self.nodes_only_label.setText(str(launch.skeleton_nodes_only_disclosure))
        else:
            self.nodes_only_label.setText("")
        self.s3_combo.blockSignals(True)
        self.s3_combo.clear()
        self.s3_combo.addItem("(select accepted S3 run)", None)
        for run_id, path in launch.s3_options:
            self.s3_combo.addItem(run_id, (run_id, path))
        if (
            launch.s3_run_id
            and launch.s3_run_dir is not None
            and self.s3_combo.findData((launch.s3_run_id, launch.s3_run_dir)) < 0
        ):
            self.s3_combo.addItem(
                launch.s3_run_id, (launch.s3_run_id, launch.s3_run_dir)
            )
        self.s5_combo.clear()
        self.s5_combo.addItem("(select eligible S5 run)", None)
        for run_id, path in launch.s5_options:
            self.s5_combo.addItem(run_id, (run_id, path))
        if (
            launch.s5_run_id
            and launch.s5_run_dir is not None
            and self.s5_combo.findData((launch.s5_run_id, launch.s5_run_dir)) < 0
        ):
            self.s5_combo.addItem(
                launch.s5_run_id, (launch.s5_run_id, launch.s5_run_dir)
            )
        self.package_combo.blockSignals(False)
        self.review_combo.blockSignals(False)
        self.s4_combo.blockSignals(False)
        self.s2_combo.blockSignals(False)
        self.s3_combo.blockSignals(False)
        self.s5_combo.blockSignals(False)

    def _on_family_changed(self) -> None:
        if self.family_combo.currentData() != EXPORT_FAMILY_BEHAVIORAL:
            self.scope_combo.setCurrentIndex(
                self.scope_combo.findData("prepared_interval")
            )
        family = self.family_combo.currentData()
        if family != EXPORT_FAMILY_S2_POSE:
            self.show_confidence.setChecked(False)
        if family != EXPORT_FAMILY_S3_POSE:
            self.show_machine_corrections.setChecked(True)
            self.show_manual_corrections.setChecked(True)
        if family == EXPORT_FAMILY_S5_SEGMENTATION:
            self.show_contours.setChecked(True)
            self.show_masks.setChecked(False)
            self.show_scalar_text.setChecked(False)
            self.mask_opacity_spin.setValue(0.35)
        if family == EXPORT_FAMILY_MIXED:
            # Mixed defaults: both-direction HUD + skeleton; do not alter
            # single-family export defaults when leaving mixed.
            self.hud_combo.setCurrentIndex(self.hud_combo.findData("both"))
            self.draw_skeleton.setChecked(True)
            self._enforce_mixed_behavior_compatibility()
        self._apply_settings_from_widgets()
        self._update_enabled()

    def _on_pose_source_changed(self) -> None:
        self._enforce_mixed_behavior_compatibility()
        self._apply_settings_from_widgets()
        self._update_enabled()

    def _enforce_mixed_behavior_compatibility(self) -> None:
        """Keep mixed pose/behavior pairs inside supported combinations only."""

        if self.family_combo.currentData() != EXPORT_FAMILY_MIXED:
            return
        pose_kind = self.pose_source_combo.currentData()
        if pose_kind == "s3_accepted":
            # S3+S10 deferred — force automated S8 without silent S10 keep.
            idx = self.source_kind_combo.findData("s8_automated")
            if idx >= 0:
                self.source_kind_combo.setCurrentIndex(idx)

    def _on_source_kind_changed(self) -> None:
        self._enforce_mixed_behavior_compatibility()
        self._apply_settings_from_widgets()
        self._update_enabled()

    def _on_scope_changed(self) -> None:
        self._apply_settings_from_widgets()
        self._update_enabled()

    def _on_package_changed(self) -> None:
        self._refresh_bouts()
        self._apply_settings_from_widgets()

    def _refresh_bouts(self) -> None:
        package = self._selected_package()
        self.bout_combo.blockSignals(True)
        self.bout_combo.clear()
        self.bout_combo.addItem("(select bout)", None)
        if package is not None:
            options = self.controller.refresh_bout_options(package[1])
            for opt in options:
                self.bout_combo.addItem(opt.label, opt.bout_id)
            total = self.controller.state.bout_options_total_exclusive
            if self.controller.state.bout_options_truncated and total is not None:
                self.bout_limit_label.setText(
                    f"Showing first {len(options)} of {total} exclusive packaged "
                    f"bouts (cap {BOUT_COMBO_LIMIT}, by dense start). Dense endpoints "
                    "remain editable; S10 selections outside this window are injected."
                )
            else:
                count = f" ({total})" if total is not None else ""
                self.bout_limit_label.setText(
                    f"Exclusive packaged bouts available in combo{count}. "
                    "Dense endpoints remain editable."
                )
        else:
            self.bout_limit_label.setText(
                f"Combo lists the first {BOUT_COMBO_LIMIT} exclusive packaged bouts "
                "(by dense start). Dense endpoints stay editable; S10 selections "
                "outside that window are injected into the list."
            )
        self.bout_combo.blockSignals(False)

    def _apply_preferred_bout(self, launch: OverlayQcLaunchContext) -> None:
        package = self._selected_package()
        package_dir = None if package is None else package[1]
        match = self.controller.ensure_preferred_bout_option(
            package_dir,
            preferred_bout_id=launch.preferred_bout_id,
            preferred_start_sample_idx=launch.preferred_start_sample_idx,
            preferred_end_sample_idx=launch.preferred_end_sample_idx,
        )
        if match is not None:
            # Re-render combo so an injected outside-250 bout is selectable.
            self.bout_combo.blockSignals(True)
            self.bout_combo.clear()
            self.bout_combo.addItem("(select bout)", None)
            for opt in self.controller.state.bout_options:
                self.bout_combo.addItem(opt.label, opt.bout_id)
            idx = self.bout_combo.findData(match.bout_id)
            if idx >= 0:
                self.bout_combo.setCurrentIndex(idx)
            self.bout_combo.blockSignals(False)
            self.start_sample_spin.setValue(match.start_sample_idx)
            self.end_sample_spin.setValue(match.end_sample_idx)
            if match.prepared_frame_idx_start is not None:
                self.prep_start_spin.setValue(match.prepared_frame_idx_start)
            if match.prepared_frame_idx_end is not None:
                self.prep_end_spin.setValue(match.prepared_frame_idx_end)
            return

        if launch.preferred_bout_id:
            idx = self.bout_combo.findData(launch.preferred_bout_id)
            if idx >= 0:
                self.bout_combo.setCurrentIndex(idx)
                return
        if launch.preferred_start_sample_idx is not None:
            self.start_sample_spin.setValue(int(launch.preferred_start_sample_idx))
            if launch.preferred_end_sample_idx is not None:
                self.end_sample_spin.setValue(int(launch.preferred_end_sample_idx))

    def _on_bout_changed(self) -> None:
        bout_id = self.bout_combo.currentData()
        if bout_id is None:
            return
        match = next(
            (b for b in self.controller.state.bout_options if b.bout_id == bout_id),
            None,
        )
        if match is None:
            return
        self.start_sample_spin.setValue(match.start_sample_idx)
        self.end_sample_spin.setValue(match.end_sample_idx)
        if match.prepared_frame_idx_start is not None:
            self.prep_start_spin.setValue(match.prepared_frame_idx_start)
        if match.prepared_frame_idx_end is not None:
            self.prep_end_spin.setValue(match.prepared_frame_idx_end)
        self._apply_settings_from_widgets()

    def _read_settings(self) -> OverlayQcUiSettings:
        family = str(self.family_combo.currentData())
        bout_id = self.bout_combo.currentData()
        packaged_start = None
        packaged_end = None
        if bout_id:
            match = next(
                (b for b in self.controller.state.bout_options if b.bout_id == bout_id),
                None,
            )
            if match is not None:
                packaged_start = match.prepared_frame_idx_start
                packaged_end = match.prepared_frame_idx_end
        source_kind = str(self.source_kind_combo.currentData())
        if self._launch is not None and self._launch.lock_source_kind:
            source_kind = str(self._launch.locked_source_kind or "s8_automated")
        pose_source_kind = str(self.pose_source_combo.currentData() or "s4_finalized")
        return OverlayQcUiSettings(
            export_family=family,  # type: ignore[arg-type]
            source_kind=source_kind,  # type: ignore[arg-type]
            pose_source_kind=pose_source_kind,  # type: ignore[arg-type]
            scope_kind=str(self.scope_combo.currentData()),  # type: ignore[arg-type]
            prepared_frame_idx_start=int(self.prep_start_spin.value()),
            prepared_frame_idx_end=int(self.prep_end_spin.value()),
            bout_id=None if bout_id is None else str(bout_id),
            start_sample_idx=int(self.start_sample_spin.value()),
            end_sample_idx=int(self.end_sample_spin.value()),
            packaged_prepared_start=packaged_start,
            packaged_prepared_end=packaged_end,
            hud_mode=str(self.hud_combo.currentData()),
            include_contact=self.include_contact.isChecked(),
            include_sbs=self.include_sbs.isChecked(),
            include_frame_counter=self.include_counter.isChecked(),
            draw_skeleton=self.draw_skeleton.isChecked(),
            show_confidence=self.show_confidence.isChecked(),
            show_machine_corrections=self.show_machine_corrections.isChecked(),
            show_manual_corrections=self.show_manual_corrections.isChecked(),
            show_contours=self.show_contours.isChecked(),
            show_masks=self.show_masks.isChecked(),
            show_scalar_text=self.show_scalar_text.isChecked(),
            mask_opacity=float(self.mask_opacity_spin.value()),
            pre_context_sec=float(self.pre_context_spin.value()),
            post_context_sec=float(self.post_context_spin.value()),
        )

    def _apply_settings_from_widgets(self) -> None:
        if self._syncing:
            return
        try:
            settings = self._read_settings().validated()
            self.controller.update_settings(settings)
        except (ValueError, OverlayQcExportUiError):
            return
        package = self._selected_package()
        review = self._selected_review()
        s4 = self._selected_s4()
        s2 = self._selected_s2()
        s3 = self._selected_s3()
        s5 = self._selected_s5()
        self.source_summary_label.setText(
            self.controller.describe_source(
                settings=self.controller.state.settings,
                package_run_id=None if package is None else package[0],
                review_run_id=None if review is None else review[0],
                s4_run_id=None if s4 is None else s4[0],
                s2_run_id=None if s2 is None else s2[0],
                s3_run_id=None if s3 is None else s3[0],
                s5_run_id=None if s5 is None else s5[0],
            )
        )
        self.output_label.setText(self.controller.state.output_destination or "—")
        self._update_enabled()

    def _selected_package(self) -> tuple[str, Path] | None:
        data = self.package_combo.currentData()
        if data is None:
            return None
        run_id, path = data
        return str(run_id), Path(path)

    def _selected_review(self) -> tuple[str, Path, bool, str | None] | None:
        data = self.review_combo.currentData()
        if data is None:
            return None
        run_id, path, is_draft, parent = data
        return str(run_id), Path(path), bool(is_draft), parent

    def _selected_s4(self) -> tuple[str, Path] | None:
        data = self.s4_combo.currentData()
        if data is None:
            return None
        run_id, path = data
        return str(run_id), Path(path)

    def _selected_s2(self) -> tuple[str, Path] | None:
        data = self.s2_combo.currentData()
        if data is None:
            return None
        run_id, path = data
        return str(run_id), Path(path)

    def _selected_s3(self) -> tuple[str, Path] | None:
        data = self.s3_combo.currentData()
        if data is None:
            return None
        run_id, path = data
        return str(run_id), Path(path)

    def _selected_s5(self) -> tuple[str, Path] | None:
        data = self.s5_combo.currentData()
        if data is None:
            return None
        run_id, path = data
        return str(run_id), Path(path)

    def _export(self) -> None:
        if self.task_runner.is_running or self.controller.state.task_running:
            QMessageBox.warning(self, "Busy", "A QC overlay export is already running.")
            return
        launch = self._launch
        session = (
            None
            if launch is None
            else Path(launch.session_root)
        )
        if session is None:
            session = self.controller.state.bound_session_root
        try:
            settings = self._read_settings().validated()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid settings", str(exc))
            return
        package = self._selected_package()
        review = self._selected_review()
        s4 = self._selected_s4()
        s2 = self._selected_s2()
        s3 = self._selected_s3()
        s5 = self._selected_s5()
        try:
            identity = self.controller.begin_export_launch(
                session_root=session,  # type: ignore[arg-type]
                settings=settings,
                package_dir=None if package is None else package[1],
                package_run_id=None if package is None else package[0],
                review_run_dir=None if review is None else review[1],
                review_run_id=None if review is None else review[0],
                review_is_draft=None if review is None else review[2],
                s4_run_dir=None if s4 is None else s4[1],
                s4_run_id=None if s4 is None else s4[0],
                s2_run_dir=None if s2 is None else s2[1],
                s2_run_id=None if s2 is None else s2[0],
                s3_run_dir=None if s3 is None else s3[1],
                s3_run_id=None if s3 is None else s3[0],
                s5_run_dir=None if s5 is None else s5[1],
                s5_run_id=None if s5 is None else s5[0],
            )
        except OverlayQcExportUiError as exc:
            QMessageBox.warning(self, "Cannot export QC overlay", str(exc))
            return

        self._active_identity = identity
        self.stage_label.setText(self.controller.state.run_stage)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self._update_enabled()

        context = GuiTaskContext(
            task_id=uuid4().hex,
            task_kind=GuiTaskKind.OVERLAY_QC_EXPORT,
            project_dir=session,
            raw_video_path=None,
            generation=0,
        )

        def operation(report: object) -> object:
            return self.controller.execute_export_launch(
                identity,
                progress_callback=report,  # type: ignore[arg-type]
                cancellation_requested=context.is_cancellation_requested,
            )

        try:
            self.task_runner.start_progressive(
                operation,
                task_name="QC overlay export",
                context=context,
                on_progress=lambda event: self._on_progress(event, identity),
                on_success=lambda result: self._on_success(result, identity),
                on_error=lambda exc: self._on_error(exc, identity),
                on_finished=lambda: self._on_finished(identity),
            )
        except TaskAlreadyRunningError as exc:
            self.controller.record_export_failure(
                str(exc), expected_identity=identity
            )
            self._active_identity = None
            QMessageBox.warning(self, "QC overlay busy", str(exc))
            self._update_enabled()

    def _cancel(self) -> None:
        if not self.task_runner.is_running:
            return
        self.task_runner.request_cancellation()
        self.stage_label.setText("Cancel requested…")

    def _on_progress(
        self, event: GuiTaskProgressEvent, identity: OverlayQcRequestIdentity
    ) -> None:
        if (
            self._active_identity is None
            or identity.request_id != self._active_identity.request_id
        ):
            return
        self.stage_label.setText(event.message)
        self.elapsed_label.setText(f"Elapsed: {format_task_duration(event.elapsed_sec)}")
        if event.is_indeterminate or event.total_units is None:
            self.progress.setRange(0, 0)
        else:
            self.progress.setRange(0, int(event.total_units))
            self.progress.setValue(int(event.completed_units or 0))

    def _on_success(self, result: object, identity: OverlayQcRequestIdentity) -> None:
        from overlay.export import OverlayQcExportResult, OverlayS4PoseQcExportResult

        if not isinstance(result, (OverlayQcExportResult, OverlayS4PoseQcExportResult)):
            self.controller.record_export_failure(
                "Unexpected QC overlay result type.",
                expected_identity=identity,
            )
            return
        accepted = self.controller.record_export_success(
            result, expected_identity=identity
        )
        if not accepted:
            return
        self.controller.refresh_discovery(identity.session_root)
        self._refresh_jobs_table()
        self.stage_label.setText(self.controller.state.run_stage)
        QMessageBox.information(
            self,
            "QC overlay export completed",
            f"Wrote {result.video_path}\n({result.frames_written} frames)",
        )

    def _on_error(self, exc: BaseException, identity: OverlayQcRequestIdentity) -> None:
        canceled = isinstance(exc, OverlayExportCancelled)
        message = actionable_overlay_error(exc)
        accepted = self.controller.record_export_failure(
            message, expected_identity=identity, canceled=canceled
        )
        if not accepted:
            return
        self.controller.refresh_discovery(identity.session_root)
        self._refresh_jobs_table()
        self.stage_label.setText(self.controller.state.run_stage)
        title = "QC overlay canceled" if canceled else "QC overlay failed"
        QMessageBox.warning(self, title, message)

    def _on_finished(self, identity: OverlayQcRequestIdentity) -> None:
        if (
            self._active_identity is not None
            and identity.request_id == self._active_identity.request_id
        ):
            self._active_identity = None
        self.progress.setVisible(False)
        self._update_enabled()

    def _refresh_jobs_table(self) -> None:
        self.controller.refresh_discovery()
        rows = self.controller.job_table_rows()
        selected = self.controller.state.selected_job_id
        self.jobs_table.blockSignals(True)
        self.jobs_table.setRowCount(len(rows))
        selected_row = -1
        for r, row in enumerate(rows):
            for c, value in enumerate(row):
                self.jobs_table.setItem(r, c, QTableWidgetItem(value))
            if selected is not None and row[0] == selected:
                selected_row = r
        if selected_row >= 0:
            self.jobs_table.selectRow(selected_row)
        self.jobs_table.blockSignals(False)
        self.output_label.setText(self.controller.state.output_destination or "—")
        self._update_enabled()

    def _job_selection_changed(self) -> None:
        items = self.jobs_table.selectedItems()
        if not items:
            self.controller.select_job(None)
            self._update_enabled()
            return
        job_id_item = self.jobs_table.item(items[0].row(), 0)
        text = None if job_id_item is None else job_id_item.text()
        if text == "(last attempt)":
            text = None
        self.controller.select_job(text)
        self._update_enabled()

    def _open_folder(self) -> None:
        try:
            self.controller.open_selected_job_folder()
        except OverlayQcExportUiError as exc:
            QMessageBox.warning(self, "Open folder", str(exc))

    def _open_video(self) -> None:
        try:
            self.controller.open_selected_video()
        except OverlayQcExportUiError as exc:
            QMessageBox.warning(self, "Open MP4", str(exc))

    def _update_enabled(self) -> None:
        running = self.task_runner.is_running or self.controller.state.task_running
        self.export_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.refresh_button.setEnabled(not running)
        job = self.controller.selected_job()
        completed = job is not None and job.is_completed
        self.open_folder_button.setEnabled(
            completed or self.controller.state.last_result is not None
        )
        self.open_video_button.setEnabled(
            completed or self.controller.state.last_result is not None
        )
        if running:
            for widget in (
                self.family_combo,
                self.pose_source_combo,
                self.source_kind_combo,
                self.package_combo,
                self.review_combo,
                self.s4_combo,
                self.s2_combo,
                self.s3_combo,
                self.s5_combo,
                self.scope_combo,
                self.prep_start_spin,
                self.prep_end_spin,
                self.bout_combo,
                self.start_sample_spin,
                self.end_sample_spin,
                self.hud_combo,
                self.include_contact,
                self.include_sbs,
                self.include_counter,
                self.draw_skeleton,
                self.show_confidence,
                self.show_machine_corrections,
                self.show_manual_corrections,
                self.show_contours,
                self.show_masks,
                self.show_scalar_text,
                self.mask_opacity_spin,
                self.pre_context_spin,
                self.post_context_spin,
            ):
                widget.setEnabled(False)
            return

        lock = bool(self._launch.lock_family) if self._launch else False
        source_locked = (
            bool(self._launch.lock_source_kind) if self._launch is not None else False
        )
        family = self.family_combo.currentData()
        behavioral = family == EXPORT_FAMILY_BEHAVIORAL
        mixed = family == EXPORT_FAMILY_MIXED
        s4_family = family == EXPORT_FAMILY_S4_POSE
        s2_family = family == EXPORT_FAMILY_S2_POSE
        s3_family = family == EXPORT_FAMILY_S3_POSE
        s5_family = family == EXPORT_FAMILY_S5_SEGMENTATION
        bout = self.scope_combo.currentData() == "bout_span"
        reviewed = self.source_kind_combo.currentData() == "s10_reviewed"
        pose_kind = self.pose_source_combo.currentData()
        mixed_s4 = mixed and pose_kind == "s4_finalized"
        mixed_s3 = mixed and pose_kind == "s3_accepted"
        behavioral_controls = behavioral or mixed
        # S3 mixed locks behavior to S8; S4 mixed may choose S8 or S10.
        mixed_behavior_switchable = mixed_s4 and not source_locked
        self.family_combo.setEnabled(not lock)
        self.pose_source_combo.setEnabled(mixed)
        self.source_kind_combo.setEnabled(
            (behavioral and not source_locked) or mixed_behavior_switchable
        )
        self.package_combo.setEnabled(behavioral_controls)
        self.review_combo.setEnabled(
            ((behavioral and reviewed) or (mixed_s4 and reviewed)) and not source_locked
        )
        self.s4_combo.setEnabled(s4_family or mixed_s4)
        self.s2_combo.setEnabled(s2_family)
        self.s3_combo.setEnabled(s3_family or mixed_s3)
        self.s5_combo.setEnabled(s5_family)
        self.scope_combo.setEnabled(behavioral)
        self.hud_combo.setEnabled(behavioral_controls)
        self.include_contact.setEnabled(behavioral_controls)
        self.include_sbs.setEnabled(behavioral_controls)
        self.draw_skeleton.setEnabled(
            s4_family or s2_family or s3_family or mixed
        )
        self.show_confidence.setEnabled(s2_family)
        self.show_machine_corrections.setEnabled(s3_family)
        self.show_manual_corrections.setEnabled(s3_family)
        self.show_contours.setEnabled(s5_family)
        self.show_masks.setEnabled(s5_family)
        self.show_scalar_text.setEnabled(s5_family)
        self.mask_opacity_spin.setEnabled(s5_family and self.show_masks.isChecked())
        self.s4_provenance_label.setVisible(s4_family or mixed_s4)
        self.include_counter.setEnabled(True)
        self.pre_context_spin.setEnabled(True)
        self.post_context_spin.setEnabled(True)
        self.bout_combo.setEnabled(bout and behavioral)
        self.start_sample_spin.setEnabled(bout and behavioral)
        self.end_sample_spin.setEnabled(bout and behavioral)
        self.prep_start_spin.setEnabled(not bout or not behavioral)
        self.prep_end_spin.setEnabled(not bout or not behavioral)

    def reject(self) -> None:
        if self.task_runner.is_running:
            QMessageBox.warning(
                self,
                "Export running",
                "Cancel the QC overlay export before closing this dialog.",
            )
            return
        super().reject()


def open_overlay_qc_export_dialog(
    parent: QWidget | None,
    *,
    launch: OverlayQcLaunchContext,
    controller: OverlayQcExportController | None = None,
) -> OverlayQcExportDialog:
    """Construct, show, and return the shared QC overlay export dialog."""

    dialog = OverlayQcExportDialog(parent, controller=controller, launch=launch)
    dialog.exec()
    return dialog


__all__ = [
    "OverlayQcExportDialog",
    "open_overlay_qc_export_dialog",
]
