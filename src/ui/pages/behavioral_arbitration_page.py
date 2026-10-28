"""Subsystem 08 — Behavioral Arbitration GUI page (Group 3 beside S7)."""

from __future__ import annotations

import contextlib
import time
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from behavioral_arbitration import (
    BehavioralArbitrationError,
    BehavioralArbitrationRunResult,
)
from ui.controllers.behavioral_arbitration_controller import (
    ArbitrationRequestIdentity,
    BehavioralArbitrationController,
    BehavioralArbitrationUiError,
    BehavioralArbitrationUiSettings,
    DiscoveryRequestIdentity,
    DiscoverySnapshot,
    replace_scientific,
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


class BehavioralArbitrationPage(QWidget):
    """Group 3 page for S8 source selection, settings, run, QC, and discovery."""

    status_message = Signal(str)
    unexpected_error = Signal(str)
    task_running_changed = Signal(bool)
    task_finished = Signal()

    def __init__(
        self,
        controller: BehavioralArbitrationController | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller or BehavioralArbitrationController()
        self.task_runner = GuiTaskRunner(self)
        self.discovery_runner = GuiTaskRunner(self)
        self._syncing_widgets = False
        self._active_discovery_identity: DiscoveryRequestIdentity | None = None
        self._active_arbitration_identity: ArbitrationRequestIdentity | None = None
        self._started_at: float | None = None
        self._last_running_state = False
        self._spin_map: dict[str, QDoubleSpinBox | QSpinBox] = {}
        self._build_ui()
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.setInterval(250)
        self.elapsed_timer.timeout.connect(self._update_elapsed)
        self.refresh_from_state()

    @property
    def has_running_task(self) -> bool:
        return (
            self.task_runner.is_running
            or self.discovery_runner.is_running
            or self.controller.state.task_running
            or self.controller.state.discovery_running
        )

    def set_session(self, session_root: Path | str) -> None:
        try:
            identity = self.controller.prepare_session(session_root)
        except BehavioralArbitrationUiError as exc:
            QMessageBox.warning(self, "Cannot open session", str(exc))
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return
        self.refresh_from_state()
        self.status_message.emit(
            f"S8 session opened: {Path(session_root).name}; discovering…"
        )
        self._start_discovery_task(identity)

    def refresh_from_state(self) -> None:
        state = self.controller.state
        self._syncing_widgets = True
        try:
            self.session_path.setText(
                str(state.session_root) if state.session_root is not None else ""
            )
            self._populate_s7_combo()
            self._sync_settings_widgets(state.working_settings)
            self._refresh_source_panel()
            self._populate_runs_table()
            self._refresh_summary_panels()
            if state.task_running and self._started_at is None:
                self._started_at = time.perf_counter()
                self.elapsed_timer.start()
                self.progress.setVisible(True)
                self.progress.setRange(0, 0)
                self.stage_label.setText(state.run_stage)
            elif not state.task_running and not self.task_runner.is_running:
                if state.run_error:
                    self.stage_label.setText(f"Failed: {state.run_error}")
                else:
                    self.stage_label.setText(state.run_stage)
        finally:
            self._syncing_widgets = False
        self._update_enabled()
        self._emit_running_changed()

    def _build_ui(self) -> None:
        title = QLabel("Subsystem 8 — Behavioral Arbitration")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        scope = QLabel(
            "Arbitrate eligible Subsystem 7 predictions with the linked Subsystem 6 "
            "features into an exclusive ethogram, parallel contact/side-by-side "
            "channels, and bout tables (Pipeline 100). Does not train models, edit "
            "bouts, or generate overlays. Cancellation is not supported; partial "
            "artifacts are never published as completed."
        )
        scope.setWordWrap(True)

        session_group = QGroupBox("Session and eligible S7 source")
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
        self.s7_combo = QComboBox()
        self.s7_combo.currentIndexChanged.connect(self._s7_selection_changed)
        self.s7_status_label = QLabel("No S7 selected.")
        self.s7_status_label.setWordWrap(True)
        self.s7_status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.source_status_label = QLabel(
            "Select a project/session with an S8-ready S7 run."
        )
        self.source_status_label.setWordWrap(True)
        self.refresh_button = QPushButton("Refresh discovery")
        self.refresh_button.clicked.connect(self._refresh_discovery)
        session_form.addRow("Session/project", session_row)
        session_form.addRow("S7 run (S8 readiness assessed)", self.s7_combo)
        session_form.addRow("S7 details", self.s7_status_label)
        session_form.addRow("Readiness", self.source_status_label)
        session_form.addRow(self.refresh_button)

        s6_group = QGroupBox("Linked S6 source and alignment")
        s6_form = QFormLayout(s6_group)
        self.linked_s6_label = QLabel("—")
        self.linked_s6_label.setWordWrap(True)
        self.linked_s6_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        s6_form.addRow("Linked S6 (exact; no substitution)", self.linked_s6_label)

        settings_group = QGroupBox("Scientific configuration (frozen at run)")
        settings_layout = QVBoxLayout(settings_group)
        settings_hint = QLabel(
            "Defaults match the versioned suite YAML. Stage 8 contact morphological "
            "closing (max_gap_to_fill) is distinct from Stage 10 general gap filling "
            "(gap_fill_seconds). Units shown per control."
        )
        settings_hint.setWordWrap(True)
        settings_layout.addWidget(settings_hint)

        settings_layout.addWidget(
            self._spin_group(
                "Hard-distance gates (normalized)",
                [
                    ("hard_gap_nose_to_nose_distance_norm", "N–N hard gap", 0.0, 10.0, False),
                    ("hard_gap_nose_to_body_distance_norm", "N–body hard gap", 0.0, 10.0, False),
                    ("hard_gap_nose_to_tail_distance_norm", "N–tail hard gap", 0.0, 10.0, False),
                    ("hard_gap_side_by_side_distance_norm", "SBS hard gap", 0.0, 10.0, False),
                    (
                        "hard_gap_contact_spine_to_spine_distance_norm",
                        "Contact spine–spine hard gap",
                        0.0,
                        10.0,
                        False,
                    ),
                ],
            )
        )
        settings_layout.addWidget(
            self._spin_group(
                "Hierarchy and decay",
                [
                    ("decay_rate", "Decay rate", 0.0, 1.0, False),
                    ("conf_frames", "Confidence frames", 1, 120, True),
                    ("engaged_min_threshold", "Engaged min threshold", 0.0, 1.0, False),
                    ("hierarchy_layer1_floor", "Layer-1 significance floor", 0.0, 1.0, False),
                ],
            )
        )
        settings_layout.addWidget(
            self._spin_group(
                "Adaptive approach / retract",
                [
                    ("adaptive_approach_engaged_distance", "Approach engaged distance", 0.0, 20.0, False),
                    ("adaptive_retract_scale", "Retract engaged distance scale", 0.0, 10.0, False),
                    ("adaptive_strict_threshold", "Strict threshold", 0.0, 1.0, False),
                    ("adaptive_envelope_threshold", "Envelope threshold", 0.0, 1.0, False),
                    ("adaptive_window_frames", "Window frames", 1, 120, True),
                ],
            )
        )
        contact_box = self._spin_group(
            "Contact optical / Schmitt / morphology (Stage 8 only)",
            [
                ("contact_smoothing_window", "Smoothing window (frames)", 1, 21, True),
                ("contact_optical_assist_min", "Optical assist min", 0.0, 1.0, False),
                ("contact_prob_low", "Schmitt low threshold", 0.0, 1.0, False),
                (
                    "contact_max_gap_to_fill",
                    "Stage 8 morphological closing max_gap_to_fill (frames; 0=disabled)",
                    0,
                    120,
                    True,
                ),
                ("contact_min_bout_duration", "Stage 8 min bout duration (frames)", 1, 120, True),
            ],
        )
        settings_layout.addWidget(contact_box)
        settings_layout.addWidget(
            self._spin_group(
                "Gap filling and minimum durations (Stage 10 — independent of Stage 8)",
                [
                    ("gap_fill_seconds", "Stage 10 gap_fill_seconds", 0.0, 2.0, False),
                    ("min_bout_seconds", "Stage 10 min_bout_seconds", 0.0, 2.0, False),
                ],
            )
        )
        settings_layout.addWidget(
            self._spin_group(
                "Stitching and rescue (frames)",
                [
                    ("max_sniff_engaged_gap_frames", "Max sniff–engaged gap", 0, 2000, True),
                    ("max_dynamic_engaged_gap_frames", "Max dynamic engaged gap", 0, 2000, True),
                    ("max_idle_retract_gap_frames", "Max idle–retract gap", 0, 2000, True),
                    ("rescue_approach_prob_hint", "Rescue approach prob hint", 0.0, 1.0, False),
                    ("rescue_retract_prob_hint", "Rescue retract prob hint", 0.0, 1.0, False),
                    ("stitch_look_ahead_frames", "Dynamic stitch look-ahead", 1, 120, True),
                ],
            )
        )
        settings_layout.addWidget(
            self._spin_group(
                "Reciprocity and engaged hysteresis",
                [
                    (
                        "min_mutually_disengaged_frames",
                        "Min mutually_disengaged (frames)",
                        1,
                        2000,
                        True,
                    ),
                    (
                        "final_engaged_hysteresis_frames",
                        "Final engaged hysteresis (frames)",
                        1,
                        2000,
                        True,
                    ),
                ],
            )
        )
        exec_row = QHBoxLayout()
        self.emit_diagnostics_check = QCheckBox(
            "Emit arbitration_diagnostics.parquet (execution only)"
        )
        exec_row.addWidget(self.emit_diagnostics_check)
        exec_row.addStretch(1)
        settings_layout.addLayout(exec_row)
        self.effective_settings_label = QLabel("")
        self.effective_settings_label.setWordWrap(True)
        self.effective_settings_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        settings_layout.addWidget(self.effective_settings_label)

        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run Behavioral Arbitration")
        self.run_button.clicked.connect(self._run_arbitration)
        self.open_output_button = QPushButton("Open S8 run folder")
        self.open_output_button.clicked.connect(self._open_output_folder)
        self.apply_settings_button = QPushButton("Apply settings")
        self.apply_settings_button.clicked.connect(self._apply_settings)
        run_row.addWidget(self.run_button)
        run_row.addWidget(self.open_output_button)
        run_row.addWidget(self.apply_settings_button)
        run_row.addStretch(1)

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

        ethogram_group = QGroupBox("Framewise behavioral summary (exclusive ethogram)")
        ethogram_layout = QVBoxLayout(ethogram_group)
        self.ethogram_table = self._metric_table()
        ethogram_layout.addWidget(self.ethogram_table)

        bout_group = QGroupBox("Bout summary (exclusive + parallel channels)")
        bout_layout = QVBoxLayout(bout_group)
        self.bout_table = self._metric_table()
        bout_layout.addWidget(self.bout_table)

        qc_group = QGroupBox("QC and provenance")
        qc_layout = QVBoxLayout(qc_group)
        self.qc_table = self._metric_table(max_height=280)
        qc_layout.addWidget(self.qc_table)

        runs_group = QGroupBox("Completed S8 runs (newest first)")
        runs_layout = QVBoxLayout(runs_group)
        self.runs_table = QTableWidget(0, 7)
        self.runs_table.setHorizontalHeaderLabels(
            (
                "Run ID",
                "Status",
                "Current eligibility",
                "Source S7",
                "Source S6",
                "Warnings",
                "Output",
            )
        )
        self.runs_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.runs_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.runs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.runs_table.verticalHeader().setVisible(False)
        header = self.runs_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.runs_table.itemSelectionChanged.connect(self._s8_selection_changed)
        runs_layout.addWidget(self.runs_table)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.addWidget(title)
        layout.addWidget(scope)
        layout.addWidget(session_group)
        layout.addWidget(s6_group)
        layout.addWidget(settings_group)
        layout.addLayout(run_row)
        layout.addWidget(status_group)
        layout.addWidget(ethogram_group)
        layout.addWidget(bout_group)
        layout.addWidget(qc_group)
        layout.addWidget(runs_group, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _metric_table(self, *, max_height: int = 220) -> QTableWidget:
        table = QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(("Metric", "Value"))
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(
            0, QHeaderView.ResizeMode.ResizeToContents
        )
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        table.setMaximumHeight(max_height)
        return table

    def _spin_group(
        self,
        title: str,
        specs: list[tuple[str, str, float, float, bool]],
    ) -> QGroupBox:
        box = QGroupBox(title)
        form = QFormLayout(box)
        for key, label, lo, hi, integer in specs:
            if integer:
                spin: QDoubleSpinBox | QSpinBox = QSpinBox()
                spin.setRange(int(lo), int(hi))
            else:
                spin = QDoubleSpinBox()
                spin.setDecimals(6)
                spin.setRange(float(lo), float(hi))
                spin.setSingleStep(0.01)
            self._spin_map[key] = spin
            form.addRow(label, spin)
        return box

    def _browse_session(self) -> None:
        if self.controller.state.task_running or self.task_runner.is_running:
            self.status_message.emit(
                "Cannot change session while behavioral arbitration is active."
            )
            return
        start = self.session_path.text().strip()
        chosen = QFileDialog.getExistingDirectory(
            self, "Open project/session directory", start
        )
        if not chosen:
            return
        self.set_session(chosen)

    def _open_typed_session(self) -> None:
        if self.controller.state.task_running or self.task_runner.is_running:
            self.status_message.emit(
                "Cannot change session while behavioral arbitration is active."
            )
            return
        text = self.session_path.text().strip()
        if not text:
            QMessageBox.information(self, "Session required", "Enter a session path.")
            return
        self.set_session(text)

    def _refresh_discovery(self) -> None:
        if not self._guard_idle("refresh discovery"):
            return
        if self.controller.state.session_root is None:
            QMessageBox.information(
                self, "Session required", "Open a project/session first."
            )
            return
        try:
            identity = self.controller.begin_discovery()
        except BehavioralArbitrationUiError as exc:
            QMessageBox.warning(self, "Discovery failed", str(exc))
            self.status_message.emit(str(exc))
            return
        self.refresh_from_state()
        self.status_message.emit("S8 discovery started in background…")
        self._start_discovery_task(identity)

    def _start_discovery_task(self, identity: DiscoveryRequestIdentity) -> None:
        self._active_discovery_identity = identity
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.stage_label.setText("Discovering S7/S8 artifacts…")
        self._update_enabled()
        self._emit_running_changed()
        if self.discovery_runner.is_running:
            return
        context = GuiTaskContext(
            task_id=uuid4().hex,
            task_kind=GuiTaskKind.BEHAVIORAL_ARBITRATION_DISCOVERY,
            project_dir=identity.session_root,
            raw_video_path=None,
            generation=identity.generation,
        )

        def _operation() -> DiscoverySnapshot:
            return self.controller.collect_discovery_snapshot(identity)

        try:
            self.discovery_runner.start(
                _operation,
                task_name="S8 discovery",
                context=context,
                on_success=lambda result: self._on_discovery_success(result, identity),
                on_error=lambda exc: self._on_discovery_error(exc, identity),
                on_finished=lambda: self._on_discovery_finished(identity),
            )
        except TaskAlreadyRunningError:
            return

    def _on_discovery_success(
        self, result: object, identity: DiscoveryRequestIdentity
    ) -> None:
        if not isinstance(result, DiscoverySnapshot):
            self.unexpected_error.emit(
                f"Unexpected S8 discovery result type: {type(result)!r}"
            )
            return
        accepted = self.controller.apply_discovery_snapshot(result)
        if not accepted:
            self.status_message.emit(
                "Ignored stale S8 discovery result after session change."
            )
            return
        self.refresh_from_state()
        self.status_message.emit(
            f"S8 discovery complete for {identity.session_root.name}."
        )

    def _on_discovery_error(
        self, exc: BaseException, identity: DiscoveryRequestIdentity
    ) -> None:
        message = str(exc)
        accepted = self.controller.record_discovery_failure(
            message, expected_identity=identity
        )
        if not accepted:
            return
        self.refresh_from_state()
        if isinstance(exc, BehavioralArbitrationUiError):
            QMessageBox.warning(self, "Discovery failed", message)
            self.status_message.emit(f"S8 discovery failed: {message}")
        else:
            self.unexpected_error.emit(message)

    def _on_discovery_finished(self, identity: DiscoveryRequestIdentity) -> None:
        latest = self.controller.state.active_discovery_identity
        if (
            self.controller.state.discovery_running
            and latest is not None
            and latest.request_id != identity.request_id
        ):
            self._start_discovery_task(latest)
            return
        if (
            self._active_discovery_identity is not None
            and identity.request_id == self._active_discovery_identity.request_id
        ):
            self._active_discovery_identity = None
        if not self.controller.state.task_running and not self.discovery_runner.is_running:
            self.progress.setVisible(False)
            if self.controller.state.run_stage == "Not started":
                self.stage_label.setText(self.controller.state.run_stage)
        self._update_enabled()
        self._emit_running_changed()
        self.task_finished.emit()

    def _apply_settings(self) -> None:
        if not self._guard_idle("apply settings"):
            return
        try:
            settings = self._read_settings_from_widgets()
            self.controller.update_settings(settings)
        except BehavioralArbitrationUiError as exc:
            QMessageBox.warning(self, "Invalid settings", str(exc))
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return
        self.refresh_from_state()
        self.status_message.emit(
            "S8 settings applied (immutable snapshot taken at run time)."
        )

    def _populate_s7_combo(self) -> None:
        self.s7_combo.blockSignals(True)
        self.s7_combo.clear()
        self.s7_combo.addItem("(none)", None)
        selected = self.controller.state.selected_s7_run_id
        index = 0
        for i, opt in enumerate(self.controller.state.s7_options, start=1):
            self.s7_combo.addItem(opt.label, opt.run_id)
            if selected is not None and opt.run_id == selected:
                index = i
        self.s7_combo.setCurrentIndex(index)
        self.s7_combo.blockSignals(False)

    def _refresh_source_panel(self) -> None:
        self.source_status_label.setText(self.controller.source_status_text())
        source = self.controller.selected_s7_source()
        if source is None:
            self.s7_status_label.setText("No S7 selected.")
            self.linked_s6_label.setText("—")
            return
        reasons = "; ".join(source.reasons) if source.reasons else "—"
        missing = (
            ", ".join(source.missing_detectors) if source.missing_detectors else "none"
        )
        missing_s6 = (
            ", ".join(source.missing_s6_columns) if source.missing_s6_columns else "none"
        )
        self.s7_status_label.setText(
            f"Historical status={source.historical_status or '—'}; "
            f"S7 classification={source.s7_classification}; "
            f"S8-ready={source.s8_ready}; mice={list(source.mouse_ids) or '—'}; "
            f"frames={source.n_frames if source.n_frames is not None else '—'}; "
            f"dyadic={', '.join(source.detectors_dyadic) or '—'}; "
            f"focal={', '.join(source.detectors_focal) or '—'}; "
            f"missing detectors={missing}; missing S6 cols={missing_s6}; "
            f"lineage_aligned={source.lineage_aligned}; reasons={reasons}"
        )
        self.linked_s6_label.setText(
            f"run_id={source.linked_s6_run_id or '—'}; "
            f"path={source.linked_s6_run_dir or '—'}; "
            f"alignment={'OK' if source.lineage_aligned else 'not confirmed / failed'}"
        )
        effective = self.controller.settings_display_rows()
        self.effective_settings_label.setText(
            "Effective highlights: "
            + "; ".join(f"{k}={v}" for k, v in effective[:8])
        )

    def _populate_runs_table(self) -> None:
        summary = self.controller.state.summary
        runs = list(summary.runs) if summary is not None else []
        self.runs_table.blockSignals(True)
        self.runs_table.setRowCount(len(runs))
        selected = self.controller.state.selected_s8_run_id
        select_row = -1
        for row, run in enumerate(runs):
            values = (
                run.run_id,
                run.status or "—",
                run.classification,
                run.source_s7_run_id or "—",
                run.source_s6_run_id or "—",
                str(run.warning_count),
                str(run.behavior_states_path or run.run_dir),
            )
            for col, value in enumerate(values):
                self.runs_table.setItem(row, col, QTableWidgetItem(value))
            if selected is not None and run.run_id == selected:
                select_row = row
        if select_row >= 0:
            self.runs_table.selectRow(select_row)
        self.runs_table.blockSignals(False)

    def _refresh_summary_panels(self) -> None:
        self._fill_metric_table(self.ethogram_table, self.controller.ethogram_summary_rows())
        self._fill_metric_table(self.bout_table, self.controller.bout_summary_rows())
        self._fill_metric_table(self.qc_table, self.controller.qc_display_rows())

    def _fill_metric_table(
        self, table: QTableWidget, rows: tuple[tuple[str, str], ...]
    ) -> None:
        table.setRowCount(len(rows))
        for index, (name, value) in enumerate(rows):
            table.setItem(index, 0, QTableWidgetItem(name))
            table.setItem(index, 1, QTableWidgetItem(value))

    def _sync_settings_widgets(self, settings: BehavioralArbitrationUiSettings) -> None:
        sci = settings.scientific
        contact = sci.contact_post_processing
        values: dict[str, float | int] = {
            "hard_gap_nose_to_nose_distance_norm": sci.hard_gap_nose_to_nose_distance_norm or 0.0,
            "hard_gap_nose_to_body_distance_norm": sci.hard_gap_nose_to_body_distance_norm or 0.0,
            "hard_gap_nose_to_tail_distance_norm": sci.hard_gap_nose_to_tail_distance_norm or 0.0,
            "hard_gap_side_by_side_distance_norm": sci.hard_gap_side_by_side_distance_norm or 0.0,
            "hard_gap_contact_spine_to_spine_distance_norm": (
                sci.hard_gap_contact_spine_to_spine_distance_norm or 0.0
            ),
            "decay_rate": sci.decay_rate,
            "conf_frames": sci.conf_frames,
            "engaged_min_threshold": sci.engaged_min_threshold,
            "hierarchy_layer1_floor": sci.hierarchy.layer1_significance_floor,
            "adaptive_approach_engaged_distance": sci.adaptive.approach_engaged_distance,
            "adaptive_retract_scale": sci.adaptive.retract_engaged_distance_scale,
            "adaptive_strict_threshold": sci.adaptive.strict_threshold,
            "adaptive_envelope_threshold": sci.adaptive.envelope_threshold,
            "adaptive_window_frames": sci.adaptive.window_frames,
            "contact_smoothing_window": contact.smoothing_window,
            "contact_optical_assist_min": contact.optical_assist_min,
            "contact_prob_low": contact.prob_threshold_low,
            "contact_max_gap_to_fill": contact.max_gap_to_fill,
            "contact_min_bout_duration": contact.min_bout_duration,
            "gap_fill_seconds": sci.temporal.gap_fill_seconds,
            "min_bout_seconds": sci.temporal.min_bout_seconds,
            "max_sniff_engaged_gap_frames": sci.max_sniff_engaged_gap_frames,
            "max_dynamic_engaged_gap_frames": sci.max_dynamic_engaged_gap_frames,
            "max_idle_retract_gap_frames": sci.max_idle_retract_gap_frames,
            "rescue_approach_prob_hint": sci.rescue.approach_prob_hint,
            "rescue_retract_prob_hint": sci.rescue.retract_prob_hint,
            "stitch_look_ahead_frames": sci.stitch_dynamic.look_ahead_frames,
            "min_mutually_disengaged_frames": sci.min_mutually_disengaged_frames,
            "final_engaged_hysteresis_frames": sci.final_engaged_hysteresis_frames,
        }
        for key, value in values.items():
            spin = self._spin_map.get(key)
            if spin is None:
                continue
            spin.blockSignals(True)
            spin.setValue(value)  # type: ignore[arg-type]
            spin.blockSignals(False)
        self.emit_diagnostics_check.blockSignals(True)
        self.emit_diagnostics_check.setChecked(
            settings.execution.emit_arbitration_diagnostics
        )
        self.emit_diagnostics_check.blockSignals(False)

    def _read_settings_from_widgets(self) -> BehavioralArbitrationUiSettings:
        base = self.controller.state.working_settings
        v = self._spin_value
        settings = replace_scientific(
            base,
            decay_rate=v("decay_rate"),
            conf_frames=int(v("conf_frames")),
            engaged_min_threshold=v("engaged_min_threshold"),
            hard_gap_nose_to_nose_distance_norm=v("hard_gap_nose_to_nose_distance_norm"),
            hard_gap_nose_to_body_distance_norm=v("hard_gap_nose_to_body_distance_norm"),
            hard_gap_nose_to_tail_distance_norm=v("hard_gap_nose_to_tail_distance_norm"),
            hard_gap_side_by_side_distance_norm=v("hard_gap_side_by_side_distance_norm"),
            hard_gap_contact_spine_to_spine_distance_norm=v(
                "hard_gap_contact_spine_to_spine_distance_norm"
            ),
            max_sniff_engaged_gap_frames=int(v("max_sniff_engaged_gap_frames")),
            max_dynamic_engaged_gap_frames=int(v("max_dynamic_engaged_gap_frames")),
            max_idle_retract_gap_frames=int(v("max_idle_retract_gap_frames")),
            min_mutually_disengaged_frames=int(v("min_mutually_disengaged_frames")),
            final_engaged_hysteresis_frames=int(v("final_engaged_hysteresis_frames")),
            hierarchy={"layer1_significance_floor": v("hierarchy_layer1_floor")},
            adaptive={
                "approach_engaged_distance": v("adaptive_approach_engaged_distance"),
                "retract_engaged_distance_scale": v("adaptive_retract_scale"),
                "strict_threshold": v("adaptive_strict_threshold"),
                "envelope_threshold": v("adaptive_envelope_threshold"),
                "window_frames": int(v("adaptive_window_frames")),
            },
            contact={
                "smoothing_window": int(v("contact_smoothing_window")),
                "optical_assist_min": v("contact_optical_assist_min"),
                "prob_threshold_low": v("contact_prob_low"),
                "max_gap_to_fill": int(v("contact_max_gap_to_fill")),
                "min_bout_duration": int(v("contact_min_bout_duration")),
            },
            temporal={
                "gap_fill_seconds": v("gap_fill_seconds"),
                "min_bout_seconds": v("min_bout_seconds"),
            },
            rescue={
                "approach_prob_hint": v("rescue_approach_prob_hint"),
                "retract_prob_hint": v("rescue_retract_prob_hint"),
            },
            stitch_dynamic={"look_ahead_frames": int(v("stitch_look_ahead_frames"))},
        )
        execution = replace(
            base.execution,
            emit_arbitration_diagnostics=self.emit_diagnostics_check.isChecked(),
        ).validated()
        return replace(settings, execution=execution).validated()

    def _spin_value(self, key: str) -> float:
        spin = self._spin_map[key]
        return float(spin.value())

    def _s7_selection_changed(self) -> None:
        if self._syncing_widgets:
            return
        if not self._guard_idle("change S7 source"):
            self.refresh_from_state()
            return
        run_id = self.s7_combo.currentData()
        try:
            self.controller.select_s7_source(
                str(run_id) if run_id is not None else None, manual=True
            )
        except BehavioralArbitrationUiError as exc:
            QMessageBox.warning(self, "Cannot select S7", str(exc))
            self.status_message.emit(str(exc))
        self.refresh_from_state()

    def _s8_selection_changed(self) -> None:
        if self._syncing_widgets:
            return
        row = self.runs_table.currentRow()
        if row < 0:
            return
        item = self.runs_table.item(row, 0)
        if item is None:
            return
        try:
            self.controller.select_s8_run(item.text())
        except BehavioralArbitrationUiError as exc:
            self.status_message.emit(str(exc))
            return
        self._refresh_summary_panels()
        self._update_enabled()

    def _run_arbitration(self) -> None:
        if not self._guard_idle("start behavioral arbitration"):
            return
        # Immediate accept: snapshot settings and mark launching before any
        # joint S7/S6 revalidation (that work belongs on the worker thread).
        self.run_button.setEnabled(False)
        self.stage_label.setText("Launching arbitration…")
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.status_message.emit("S8 arbitration request accepted; launching…")
        try:
            settings = self._read_settings_from_widgets()
            self.controller.update_settings(settings)
            identity = self.controller.begin_arbitration_launch(settings)
        except BehavioralArbitrationUiError as exc:
            self.progress.setVisible(False)
            QMessageBox.warning(self, "Cannot run S8 arbitration", str(exc))
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return

        self._active_arbitration_identity = identity
        self._started_at = time.perf_counter()
        self.elapsed_timer.start()
        self._update_enabled()
        self._emit_running_changed()
        self.status_message.emit(
            "S8 arbitration started (settings snapshotted; cancellation unsupported)."
        )

        context = GuiTaskContext(
            task_id=uuid4().hex,
            task_kind=GuiTaskKind.BEHAVIORAL_ARBITRATION,
            project_dir=identity.session_root,
            raw_video_path=None,
            generation=0,
        )

        def _operation(report: object) -> BehavioralArbitrationRunResult:
            return self.controller.execute_arbitration_launch(
                identity, progress_callback=report  # type: ignore[arg-type]
            )

        try:
            self.task_runner.start_progressive(
                _operation,
                task_name="S8 behavioral arbitration",
                context=context,
                on_progress=lambda event: self._on_arbitration_progress(event, identity),
                on_success=lambda result: self._on_arbitration_success(result, identity),
                on_error=lambda exc: self._on_arbitration_error(exc, identity),
                on_finished=lambda: self._on_arbitration_finished(identity),
            )
        except TaskAlreadyRunningError:
            self.controller.record_failure(
                "Arbitration already running.", expected_identity=identity
            )
            self.progress.setVisible(False)
            self.elapsed_timer.stop()
            self._update_enabled()
            self._emit_running_changed()
            QMessageBox.warning(
                self,
                "Task already running",
                "A Subsystem 8 arbitration task is already active.",
            )

    def _on_arbitration_progress(
        self,
        event: object,
        identity: ArbitrationRequestIdentity,
    ) -> None:
        if self._active_arbitration_identity is None:
            return
        if identity.request_id != self._active_arbitration_identity.request_id:
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

    def _on_arbitration_success(
        self, result: object, identity: ArbitrationRequestIdentity
    ) -> None:
        if not isinstance(result, BehavioralArbitrationRunResult):
            self.unexpected_error.emit(f"Unexpected S8 result type: {type(result)!r}")
            return
        accepted = self.controller.record_success(
            result, expected_identity=identity, refresh_discovery=False
        )
        if not accepted:
            self.status_message.emit(
                "Ignored stale S8 arbitration result after session/source change."
            )
            return
        self.stage_label.setText(f"Finished: {result.status}")
        self.refresh_from_state()
        warnings = f" ({len(result.warnings)} warnings)" if result.warnings else ""
        self.status_message.emit(
            f"S8 {result.status}{warnings}. Output: {result.behavior_states_path}"
        )
        # Refresh S7/S8 discovery asynchronously so the GUI does not block on
        # nested digest / eligibility walks after a long publish.
        if self.controller.state.session_root is not None:
            with contextlib.suppress(BehavioralArbitrationUiError):
                discovery_identity = self.controller.begin_discovery()
                self.controller.state.selected_s8_run_id = result.run_id
                self._start_discovery_task(discovery_identity)

    def _on_arbitration_error(
        self, exc: BaseException, identity: ArbitrationRequestIdentity
    ) -> None:
        message = str(exc)
        accepted = self.controller.record_failure(
            message, expected_identity=identity
        )
        if not accepted:
            return
        self.stage_label.setText(f"Failed: {message}")
        self.refresh_from_state()
        if self.controller.state.session_root is not None:
            with contextlib.suppress(BehavioralArbitrationUiError):
                discovery_identity = self.controller.begin_discovery()
                self._start_discovery_task(discovery_identity)
        if isinstance(exc, (BehavioralArbitrationUiError, BehavioralArbitrationError)):
            QMessageBox.warning(self, "Behavioral arbitration failed", message)
            self.status_message.emit(f"S8 failed: {message}")
        else:
            self.unexpected_error.emit(message)

    def _on_arbitration_finished(self, identity: ArbitrationRequestIdentity) -> None:
        if (
            self._active_arbitration_identity is not None
            and identity.request_id == self._active_arbitration_identity.request_id
        ):
            self._active_arbitration_identity = None
        self.controller.state.task_running = False
        self.elapsed_timer.stop()
        self._started_at = None
        # Keep indeterminate progress if post-success discovery was started.
        if not (
            self.controller.state.discovery_running or self.discovery_runner.is_running
        ):
            self.progress.setVisible(False)
        self._update_enabled()
        self._emit_running_changed()
        self.task_finished.emit()

    def _open_output_folder(self) -> None:
        try:
            opened = self.controller.open_selected_s8_folder()
        except BehavioralArbitrationUiError as exc:
            QMessageBox.information(self, "No S8 output", str(exc))
            self.status_message.emit(str(exc))
            return
        if not opened:
            self.status_message.emit("Could not open the S8 run folder.")
        else:
            self.status_message.emit("Opened S8 run folder.")

    def _update_elapsed(self) -> None:
        if self._started_at is None:
            self.elapsed_label.setText("Elapsed: —")
            return
        elapsed = time.perf_counter() - self._started_at
        self.elapsed_label.setText(f"Elapsed: {format_task_duration(elapsed)}")

    def _update_enabled(self) -> None:
        busy = self.has_running_task
        arbitration_busy = (
            self.controller.state.task_running or self.task_runner.is_running
        )
        self.session_browse_button.setEnabled(not arbitration_busy)
        self.open_session_button.setEnabled(not arbitration_busy)
        self.refresh_button.setEnabled(not busy)
        self.s7_combo.setEnabled(not busy)
        self.apply_settings_button.setEnabled(not busy)
        self.emit_diagnostics_check.setEnabled(not busy)
        for spin in self._spin_map.values():
            spin.setEnabled(not busy)
        can_run = self.controller.can_run_arbitration() and not busy
        self.run_button.setEnabled(can_run)
        has_output = (
            self.controller.selected_s8_run() is not None
            or self.controller.state.last_result is not None
        )
        self.open_output_button.setEnabled(has_output and not busy)

    def _guard_idle(self, action: str) -> bool:
        if self.has_running_task:
            self.status_message.emit(f"Cannot {action} while an S8 task is active.")
            return False
        return True

    def _emit_running_changed(self) -> None:
        running = self.has_running_task
        if running != self._last_running_state:
            self._last_running_state = running
            self.task_running_changed.emit(running)
