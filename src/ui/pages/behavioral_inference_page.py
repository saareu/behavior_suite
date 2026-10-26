"""Subsystem 07 Behavioral Inference GUI: eligible S6 → dyadic/focal predictions."""

from __future__ import annotations

import contextlib
import time
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
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from behavioral_inference import (
    COMPLETED_WITH_WARNINGS,
    ELIGIBLE_DOWNSTREAM,
    FAILED_OR_INCOMPLETE,
    INVALID_ARTIFACT,
    LOGICAL_DYADIC_DETECTORS,
    LOGICAL_FOCAL_DETECTORS,
    MISSING_REQUIRED_ARTIFACTS,
    STALE_MODELS,
    STALE_SOURCE,
    BehavioralInferenceError,
    BehavioralInferenceRunResult,
)
from ui.controllers.behavioral_inference_controller import (
    BehavioralInferenceController,
    BehavioralInferenceUiError,
    BehavioralInferenceUiSettings,
    DiscoveryRequestIdentity,
    DiscoverySnapshot,
    InferenceRequestIdentity,
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


class BehavioralInferencePage(QWidget):
    """S7 workspace: S6 source, registry, detectors, inference, QC, discovery."""

    status_message = Signal(str)
    unexpected_error = Signal(str)
    task_running_changed = Signal(bool)
    task_finished = Signal()

    def __init__(
        self,
        controller: BehavioralInferenceController | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.controller = controller or BehavioralInferenceController()
        self.task_runner = GuiTaskRunner(self)
        self.discovery_runner = GuiTaskRunner(self)
        self._syncing_widgets = False
        self._started_at: float | None = None
        self._last_running_state: bool | None = None
        self._active_inference_identity: InferenceRequestIdentity | None = None
        self._active_discovery_identity: DiscoveryRequestIdentity | None = None
        self._dyadic_checks: dict[str, QCheckBox] = {}
        self._focal_checks: dict[str, QCheckBox] = {}
        self._build_ui()
        self.elapsed_timer = QTimer(self)
        self.elapsed_timer.setInterval(500)
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
        """Open a project/session and refresh S6/S7 discovery in the background."""

        if self.controller.state.task_running or self.task_runner.is_running:
            QMessageBox.warning(
                self,
                "Cannot open S7 session",
                "Cannot change S7 session while behavioral inference is active.",
            )
            return
        try:
            identity = self.controller.prepare_session(session_root)
        except BehavioralInferenceUiError as exc:
            QMessageBox.warning(self, "Cannot open S7 session", str(exc))
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return
        self.refresh_from_state()
        self.status_message.emit(
            f"Opened session for S7: {session_root} (discovering in background…)"
        )
        self._start_discovery_task(identity)

    def refresh_from_state(self) -> None:
        state = self.controller.state
        self._syncing_widgets = True
        try:
            self.session_path.setText(
                str(state.session_root) if state.session_root is not None else ""
            )
            self._sync_settings_widgets(state.working_settings)
            self._populate_s6_combo()
            self._refresh_source_panel()
            self._refresh_registry_panel()
            self._refresh_detector_panel()
            self._refresh_threshold_panel()
            self._populate_runs_table()
            self._refresh_qc_panel()
            self._update_enabled()
            self._emit_running_changed()
        finally:
            self._syncing_widgets = False

    def _build_ui(self) -> None:
        title = QLabel("Subsystem 7 — Behavioral Inference")
        title.setStyleSheet("font-size: 20px; font-weight: 600;")
        scope = QLabel(
            "Score eligible Subsystem 6 dyadic features with configured dyadic and "
            "focal detectors in one workflow. Publishes framewise predictions, wide "
            "views, QC, and model provenance for downstream S8. Does not train models, "
            "arbitrate detectors, or generate bouts. Cancellation is not supported; "
            "partial prediction artifacts are never published as completed."
        )
        scope.setWordWrap(True)

        session_group = QGroupBox("Session and eligible S6 source")
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

        self.s6_combo = QComboBox()
        self.s6_combo.currentIndexChanged.connect(self._s6_selection_changed)
        self.s6_status_label = QLabel("No S6 selected.")
        self.s6_status_label.setWordWrap(True)
        self.s6_status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.source_status_label = QLabel(
            "Select a project/session with an eligible S6 run."
        )
        self.source_status_label.setWordWrap(True)
        self.refresh_button = QPushButton("Refresh discovery")
        self.refresh_button.clicked.connect(self._refresh_discovery)
        session_form.addRow("Session/project", session_row)
        session_form.addRow("Eligible S6 run", self.s6_combo)
        session_form.addRow("S6 details", self.s6_status_label)
        session_form.addRow("Readiness", self.source_status_label)
        session_form.addRow(self.refresh_button)

        registry_group = QGroupBox("Model registry (required for inference)")
        registry_form = QFormLayout(registry_group)
        self.registry_path = QLineEdit()
        self.registry_browse_button = QPushButton("Browse…")
        self.registry_browse_button.clicked.connect(self._browse_registry)
        self.registry_clear_button = QPushButton("Clear")
        self.registry_clear_button.clicked.connect(self._clear_registry)
        registry_row = QHBoxLayout()
        registry_row.addWidget(self.registry_path, 1)
        registry_row.addWidget(self.registry_browse_button)
        registry_row.addWidget(self.registry_clear_button)
        self.registry_status_label = QLabel(
            "Configure a trusted local model-registry root. Discovery uses specs "
            "and digests only — models are not loaded solely to populate the list."
        )
        self.registry_status_label.setWordWrap(True)
        self.runtime_deps_label = QLabel("")
        self.runtime_deps_label.setWordWrap(True)
        registry_form.addRow("Registry root", registry_row)
        registry_form.addRow(self.registry_status_label)
        registry_form.addRow(self.runtime_deps_label)

        detector_group = QGroupBox("Behavioral detectors (dyadic + focal)")
        detector_layout = QVBoxLayout(detector_group)
        detector_hint = QLabel(
            "Select logical detectors. Specialist binaries appear under their "
            "parent detector and are not selectable as separate behaviors. "
            "Unsupported/obsolete models are listed but not executable."
        )
        detector_hint.setWordWrap(True)
        detector_layout.addWidget(detector_hint)

        checks_row = QHBoxLayout()
        dyadic_box = QGroupBox("Dyadic")
        dyadic_form = QVBoxLayout(dyadic_box)
        for name in LOGICAL_DYADIC_DETECTORS:
            check = QCheckBox(name)
            check.stateChanged.connect(self._detector_checks_changed)
            self._dyadic_checks[name] = check
            dyadic_form.addWidget(check)
            if name == "contact":
                dep = QLabel("  └ specialists: contact_intact, contact_occluded")
                dep.setStyleSheet("color: #555;")
                dyadic_form.addWidget(dep)
        dyadic_form.addStretch(1)
        focal_box = QGroupBox("Focal")
        focal_form = QVBoxLayout(focal_box)
        for name in LOGICAL_FOCAL_DETECTORS:
            check = QCheckBox(name)
            check.stateChanged.connect(self._detector_checks_changed)
            self._focal_checks[name] = check
            focal_form.addWidget(check)
            if name == "nose_to_body":
                dep = QLabel(
                    "  └ specialists: nose_body_intact, nose_body_occ "
                    "(or supported single-stage fallback)"
                )
                dep.setStyleSheet("color: #555;")
                focal_form.addWidget(dep)
        focal_form.addStretch(1)
        checks_row.addWidget(dyadic_box)
        checks_row.addWidget(focal_box)
        detector_layout.addLayout(checks_row)

        detector_buttons = QHBoxLayout()
        self.select_compatible_button = QPushButton("Select all compatible")
        self.select_compatible_button.clicked.connect(self._select_compatible)
        self.clear_detectors_button = QPushButton("Clear detectors")
        self.clear_detectors_button.clicked.connect(self._clear_detectors)
        detector_buttons.addWidget(self.select_compatible_button)
        detector_buttons.addWidget(self.clear_detectors_button)
        detector_buttons.addStretch(1)
        detector_layout.addLayout(detector_buttons)

        self.detector_table = QTableWidget(0, 6)
        self.detector_table.setHorizontalHeaderLabels(
            ("Detector", "Family", "Selected", "Ready", "Status", "Detail")
        )
        self.detector_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.detector_table.verticalHeader().setVisible(False)
        det_header = self.detector_table.horizontalHeader()
        det_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        det_header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)
        self.detector_table.setMaximumHeight(180)
        detector_layout.addWidget(self.detector_table)

        settings_group = QGroupBox("Thresholds and inference settings")
        settings_form = QFormLayout(settings_group)
        self.parity_combo = QComboBox()
        self.parity_combo.addItem(
            "legacy_parity — Pipeline 70/90 match; nose-to-body 5-frame rolling median",
            "legacy_parity",
        )
        self.parity_combo.addItem(
            "raw_framewise — no nose-to-body smoothing (changes scientific output)",
            "raw_framewise",
        )
        self.settings_warning = QLabel(
            "Changing smoothing mode or thresholds can change scientific predictions."
        )
        self.settings_warning.setWordWrap(True)
        self.settings_warning.setStyleSheet("color: #8a5a00;")
        self.exp_name_edit = QLineEdit()
        self.exp_name_edit.setPlaceholderText("(default: session folder name)")
        self.threshold_table = QTableWidget(0, 3)
        self.threshold_table.setHorizontalHeaderLabels(("Detector key", "Value", "Source"))
        self.threshold_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.threshold_table.verticalHeader().setVisible(False)
        th_header = self.threshold_table.horizontalHeader()
        th_header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        th_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.threshold_table.setMaximumHeight(160)
        override_row = QHBoxLayout()
        self.override_key_edit = QLineEdit()
        self.override_key_edit.setPlaceholderText("e.g. approach or contact.intact")
        self.override_value_spin = QDoubleSpinBox()
        self.override_value_spin.setDecimals(6)
        self.override_value_spin.setRange(0.0, 1.0)
        self.override_value_spin.setSingleStep(0.01)
        self.apply_override_button = QPushButton("Apply override")
        self.apply_override_button.clicked.connect(self._apply_override)
        self.clear_overrides_button = QPushButton("Clear overrides")
        self.clear_overrides_button.clicked.connect(self._clear_overrides)
        override_row.addWidget(self.override_key_edit, 1)
        override_row.addWidget(self.override_value_spin)
        override_row.addWidget(self.apply_override_button)
        override_row.addWidget(self.clear_overrides_button)
        settings_form.addRow("Smoothing / parity", self.parity_combo)
        settings_form.addRow(self.settings_warning)
        settings_form.addRow("exp_name", self.exp_name_edit)
        settings_form.addRow(self.threshold_table)
        settings_form.addRow("User override", override_row)

        run_row = QHBoxLayout()
        self.run_button = QPushButton("Run Behavioral Inference")
        self.run_button.clicked.connect(self._run_inference)
        self.open_output_button = QPushButton("Open S7 run folder")
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

        qc_group = QGroupBox("QC and prediction summary")
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
        self.qc_table.setMaximumHeight(260)
        qc_layout.addWidget(self.qc_table)

        runs_group = QGroupBox("Completed S7 runs (newest first)")
        runs_layout = QVBoxLayout(runs_group)
        self.runs_table = QTableWidget(0, 8)
        self.runs_table.setHorizontalHeaderLabels(
            (
                "Run ID",
                "Status",
                "Current eligibility",
                "Source S6",
                "Dyadic",
                "Focal",
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
        self.runs_table.itemSelectionChanged.connect(self._s7_selection_changed)
        runs_layout.addWidget(self.runs_table)

        body = QWidget()
        layout = QVBoxLayout(body)
        layout.addWidget(title)
        layout.addWidget(scope)
        layout.addWidget(session_group)
        layout.addWidget(registry_group)
        layout.addWidget(detector_group)
        layout.addWidget(settings_group)
        layout.addLayout(run_row)
        layout.addWidget(status_group)
        layout.addWidget(qc_group)
        layout.addWidget(runs_group, 1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(body)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _browse_session(self) -> None:
        if self.controller.state.task_running or self.task_runner.is_running:
            self.status_message.emit(
                "Cannot change session while behavioral inference is active."
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
                "Cannot change session while behavioral inference is active."
            )
            return
        text = self.session_path.text().strip()
        if not text:
            QMessageBox.information(self, "Session required", "Enter a session path.")
            return
        self.set_session(text)

    def _browse_registry(self) -> None:
        if not self._guard_idle("change model registry"):
            return
        start = (
            self.registry_path.text().strip()
            or self.controller.suggested_registry_browse_start()
        )
        chosen = QFileDialog.getExistingDirectory(
            self, "Select trusted model registry root", start
        )
        if not chosen:
            return
        self.registry_path.setText(chosen)
        self._apply_settings()

    def _clear_registry(self) -> None:
        if not self._guard_idle("clear model registry"):
            return
        self.registry_path.clear()
        self._apply_settings()

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
        except BehavioralInferenceUiError as exc:
            QMessageBox.warning(self, "Discovery failed", str(exc))
            self.status_message.emit(str(exc))
            return
        self.refresh_from_state()
        self.status_message.emit("S7 discovery started in background…")
        self._start_discovery_task(identity)

    def _start_discovery_task(self, identity: DiscoveryRequestIdentity) -> None:
        """Start discovery, or defer until the in-flight discovery worker exits."""

        self._active_discovery_identity = identity
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.stage_label.setText("Discovering S6/S7 artifacts…")
        self._update_enabled()
        self._emit_running_changed()

        if self.discovery_runner.is_running:
            # Generation already advanced; finished handler restarts for the latest id.
            return

        context = GuiTaskContext(
            task_id=uuid4().hex,
            task_kind=GuiTaskKind.BEHAVIORAL_INFERENCE_DISCOVERY,
            project_dir=identity.session_root,
            raw_video_path=None,
            generation=identity.generation,
        )

        def _operation() -> DiscoverySnapshot:
            return self.controller.collect_discovery_snapshot(identity)

        try:
            self.discovery_runner.start(
                _operation,
                task_name="S7 discovery",
                context=context,
                on_success=lambda result: self._on_discovery_success(result, identity),
                on_error=lambda exc: self._on_discovery_error(exc, identity),
                on_finished=lambda: self._on_discovery_finished(identity),
            )
        except TaskAlreadyRunningError:
            # Another discovery is finishing; keep the latest identity pending.
            return

    def _on_discovery_success(
        self, result: object, identity: DiscoveryRequestIdentity
    ) -> None:
        if not isinstance(result, DiscoverySnapshot):
            self.unexpected_error.emit(
                f"Unexpected S7 discovery result type: {type(result)!r}"
            )
            return
        accepted = self.controller.apply_discovery_snapshot(result)
        if not accepted:
            self.status_message.emit(
                "Ignored stale S7 discovery result after session change."
            )
            return
        self.refresh_from_state()
        self.status_message.emit(
            f"S7 discovery complete for {identity.session_root.name}."
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
        if isinstance(exc, BehavioralInferenceUiError):
            QMessageBox.warning(self, "Discovery failed", message)
            self.status_message.emit(f"S7 discovery failed: {message}")
        else:
            self.unexpected_error.emit(message)

    def _on_discovery_finished(self, identity: DiscoveryRequestIdentity) -> None:
        latest = self.controller.state.active_discovery_identity
        if (
            self.controller.state.discovery_running
            and latest is not None
            and latest.request_id != identity.request_id
        ):
            # Session changed while this worker ran; start discovery for the latest id.
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
        except BehavioralInferenceUiError as exc:
            QMessageBox.warning(self, "Invalid settings", str(exc))
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return
        self.refresh_from_state()
        self.status_message.emit(
            "S7 settings applied (thresholds/parity/registry snapshotted at run time)."
        )

    def _select_compatible(self) -> None:
        if not self._guard_idle("change detector selection"):
            return
        try:
            # Ensure registry/detectors are current before selecting.
            settings = self._read_settings_from_widgets()
            self.controller.update_settings(settings)
            self.controller.select_all_compatible_detectors()
        except BehavioralInferenceUiError as exc:
            QMessageBox.warning(self, "Cannot select detectors", str(exc))
            self.status_message.emit(str(exc))
        self.refresh_from_state()

    def _clear_detectors(self) -> None:
        if not self._guard_idle("change detector selection"):
            return
        try:
            self.controller.set_selected_detectors(dyadic=(), focal=())
        except BehavioralInferenceUiError as exc:
            QMessageBox.warning(self, "Cannot clear detectors", str(exc))
            self.status_message.emit(str(exc))
        self.refresh_from_state()

    def _apply_override(self) -> None:
        if not self._guard_idle("set threshold override"):
            return
        key = self.override_key_edit.text().strip()
        if not key:
            QMessageBox.information(self, "Override key", "Enter a detector key.")
            return
        try:
            self.controller.set_threshold_override(key, self.override_value_spin.value())
        except BehavioralInferenceUiError as exc:
            QMessageBox.warning(self, "Invalid override", str(exc))
            self.status_message.emit(str(exc))
        self.refresh_from_state()

    def _clear_overrides(self) -> None:
        if not self._guard_idle("clear threshold overrides"):
            return
        try:
            self.controller.clear_threshold_overrides()
        except BehavioralInferenceUiError as exc:
            QMessageBox.warning(self, "Cannot clear overrides", str(exc))
            self.status_message.emit(str(exc))
        self.refresh_from_state()

    def _populate_s6_combo(self) -> None:
        state = self.controller.state
        self.s6_combo.blockSignals(True)
        self.s6_combo.clear()
        if not state.s6_options:
            self.s6_combo.addItem("(none eligible)", None)
            self.s6_status_label.setText("No currently eligible S6 run.")
        else:
            for opt in state.s6_options:
                self.s6_combo.addItem(opt.label, opt.run_id)
            selected = state.selected_s6_run_id
            if selected is not None:
                index = self.s6_combo.findData(selected)
                if index >= 0:
                    self.s6_combo.setCurrentIndex(index)
            current = self.controller.selected_s6_source()
            if current is not None:
                manual = " (manual)" if state.s6_selection_manual else " (default)"
                samples = (
                    str(current.sample_count)
                    if current.sample_count is not None
                    else "—"
                )
                tracks = (
                    str(current.track_count) if current.track_count is not None else "—"
                )
                failures = (
                    "; ".join(current.structural_failures)
                    if current.structural_failures
                    else "none"
                )
                self.s6_status_label.setText(
                    f"{current.run_id}\n"
                    f"path={current.run_dir}\n"
                    f"status={current.status or '—'}, "
                    f"eligibility={current.classification}{manual}\n"
                    f"session={current.session_identity or '—'}\n"
                    f"samples={samples}, tracks={tracks}\n"
                    f"structural/provenance failures: {failures}"
                )
        self.s6_combo.blockSignals(False)

    def _refresh_source_panel(self) -> None:
        self.source_status_label.setText(self.controller.source_status_text())
        warnings = self.controller.state.readiness_warnings
        if warnings:
            self.source_status_label.setText(
                self.source_status_label.text()
                + "\nWarnings: "
                + "; ".join(warnings)
            )

    def _refresh_registry_panel(self) -> None:
        state = self.controller.state
        status = state.registry_status or "—"
        notes = "; ".join(state.registry_notes) if state.registry_notes else ""
        readiness = state.readiness_kind or "—"
        self.registry_status_label.setText(
            f"Registry status: {status}. Readiness: {readiness}. {notes} "
            "List population does not load estimators. "
            "Runtime libraries and model binaries are separate requirements."
        )
        self.runtime_deps_label.setText(
            state.runtime_deps_detail
            or f"Python environment: {state.python_executable}"
        )

    def _refresh_detector_panel(self) -> None:
        settings = self.controller.state.working_settings
        for name, check in self._dyadic_checks.items():
            check.blockSignals(True)
            check.setChecked(name in settings.selected_detectors_dyadic)
            check.blockSignals(False)
        for name, check in self._focal_checks.items():
            check.blockSignals(True)
            check.setChecked(name in settings.selected_detectors_focal)
            check.blockSignals(False)

        rows = self.controller.state.detector_availability
        self.detector_table.setRowCount(len(rows))
        for index, item in enumerate(rows):
            values = (
                item.name,
                item.family,
                "yes" if item.selected else "no",
                "yes" if item.ready else "no",
                item.status,
                item.detail,
            )
            for col, value in enumerate(values):
                self.detector_table.setItem(index, col, QTableWidgetItem(value))

    def _refresh_threshold_panel(self) -> None:
        rows = self.controller.threshold_display_rows()
        self.threshold_table.setRowCount(len(rows))
        for index, (key, value, source) in enumerate(rows):
            self.threshold_table.setItem(index, 0, QTableWidgetItem(key))
            self.threshold_table.setItem(index, 1, QTableWidgetItem(value))
            self.threshold_table.setItem(index, 2, QTableWidgetItem(source))

    def _populate_runs_table(self) -> None:
        summary = self.controller.state.summary
        runs = summary.runs if summary is not None else ()
        self.runs_table.setRowCount(len(runs))
        selected_id = self.controller.state.selected_s7_run_id
        selected_row = -1
        for row, run in enumerate(runs):
            values = (
                run.run_id,
                run.status or "—",
                _eligibility_label(run.classification),
                run.source_s6_run_id or "—",
                ", ".join(run.detectors_dyadic) if run.detectors_dyadic else "—",
                ", ".join(run.detectors_focal) if run.detectors_focal else "—",
                str(run.warning_count),
                str(run.predictions_path) if run.predictions_path else "—",
            )
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, run.run_id)
                self.runs_table.setItem(row, col, item)
            if run.run_id == selected_id:
                selected_row = row
        if selected_row >= 0:
            self.runs_table.selectRow(selected_row)

    def _sync_settings_widgets(self, settings: BehavioralInferenceUiSettings) -> None:
        index = self.parity_combo.findData(settings.parity_mode)
        if index >= 0:
            self.parity_combo.setCurrentIndex(index)
        self.registry_path.setText(
            str(settings.model_registry_root)
            if settings.model_registry_root is not None
            else ""
        )
        self.exp_name_edit.setText(settings.exp_name or "")

    def _read_settings_from_widgets(self) -> BehavioralInferenceUiSettings:
        current = self.controller.state.working_settings
        registry_text = self.registry_path.text().strip()
        exp_text = self.exp_name_edit.text().strip()
        parity = self.parity_combo.currentData() or "legacy_parity"
        dyadic = tuple(
            name for name, check in self._dyadic_checks.items() if check.isChecked()
        )
        focal = tuple(
            name for name, check in self._focal_checks.items() if check.isChecked()
        )
        return BehavioralInferenceUiSettings(
            parity_mode=str(parity),
            model_iteration=current.model_iteration,
            model_registry_root=Path(registry_text) if registry_text else None,
            include_focal=current.include_focal,
            include_obsolete=current.include_obsolete,
            selected_detectors_dyadic=dyadic,
            selected_detectors_focal=focal,
            working_thresholds=dict(current.working_thresholds),
            user_threshold_overrides=dict(current.user_threshold_overrides),
            allow_nose_to_body_single_stage_fallback=(
                current.allow_nose_to_body_single_stage_fallback
            ),
            require_exactly_two_mice=current.require_exactly_two_mice,
            emit_wide_views=current.emit_wide_views,
            fail_fast_on_detector_error=current.fail_fast_on_detector_error,
            exp_name=exp_text or None,
        )

    def _s6_selection_changed(self) -> None:
        if self._syncing_widgets or self.has_running_task:
            return
        run_id = self.s6_combo.currentData()
        try:
            self.controller.select_s6_source(run_id, manual=True)
        except BehavioralInferenceUiError as exc:
            self.source_status_label.setText(str(exc))
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return
        self._refresh_source_panel()
        self._populate_s6_combo()
        self._update_enabled()

    def _detector_checks_changed(self) -> None:
        if self._syncing_widgets or self.has_running_task:
            return
        dyadic = tuple(
            name for name, check in self._dyadic_checks.items() if check.isChecked()
        )
        focal = tuple(
            name for name, check in self._focal_checks.items() if check.isChecked()
        )
        try:
            self.controller.set_selected_detectors(dyadic=dyadic, focal=focal)
        except BehavioralInferenceUiError as exc:
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return
        self._refresh_detector_panel()
        self._refresh_source_panel()
        self._update_enabled()

    def _s7_selection_changed(self) -> None:
        items = self.runs_table.selectedItems()
        if not items:
            return
        run_id = items[0].data(Qt.ItemDataRole.UserRole)
        try:
            self.controller.select_s7_run(str(run_id) if run_id else None)
        except BehavioralInferenceUiError as exc:
            self.status_message.emit(str(exc))
            return
        self._refresh_qc_panel()
        self._update_enabled()

    def _run_inference(self) -> None:
        if not self._guard_idle("start behavioral inference"):
            return
        try:
            settings = self._read_settings_from_widgets()
            self.controller.update_settings(settings)
            request, identity = self.controller.build_request()
        except BehavioralInferenceUiError as exc:
            QMessageBox.warning(self, "Cannot run S7 inference", str(exc))
            self.status_message.emit(str(exc))
            self.refresh_from_state()
            return

        self._active_inference_identity = identity
        self._started_at = time.perf_counter()
        self.stage_label.setText("Running behavioral inference…")
        self.progress.setRange(0, 0)
        self.progress.setVisible(True)
        self.elapsed_timer.start()
        self._update_enabled()
        self._emit_running_changed()
        self.status_message.emit(
            "S7 inference started (settings snapshotted; cancellation unsupported)."
        )

        context = GuiTaskContext(
            task_id=uuid4().hex,
            task_kind=GuiTaskKind.BEHAVIORAL_INFERENCE,
            project_dir=identity.session_root,
            raw_video_path=None,
            generation=0,
        )

        def _operation(report: object) -> BehavioralInferenceRunResult:
            return self.controller.execute_request(
                request, progress_callback=report  # type: ignore[arg-type]
            )

        try:
            self.task_runner.start_progressive(
                _operation,
                task_name="S7 behavioral inference",
                context=context,
                on_progress=lambda event: self._on_inference_progress(event, identity),
                on_success=lambda result: self._on_inference_success(result, identity),
                on_error=lambda exc: self._on_inference_error(exc, identity),
                on_finished=lambda: self._on_inference_finished(identity),
            )
        except TaskAlreadyRunningError:
            self.controller.record_failure(
                "Inference already running.", expected_identity=identity
            )
            self.progress.setVisible(False)
            self.elapsed_timer.stop()
            self._update_enabled()
            self._emit_running_changed()
            QMessageBox.warning(
                self,
                "Task already running",
                "A Subsystem 7 inference task is already active.",
            )

    def _on_inference_progress(
        self,
        event: object,
        identity: InferenceRequestIdentity,
    ) -> None:
        if self._active_inference_identity is None:
            return
        if identity.request_id != self._active_inference_identity.request_id:
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

    def _on_inference_success(
        self, result: object, identity: InferenceRequestIdentity
    ) -> None:
        if not isinstance(result, BehavioralInferenceRunResult):
            self.unexpected_error.emit(f"Unexpected S7 result type: {type(result)!r}")
            return
        accepted = self.controller.record_success(
            result, expected_identity=identity
        )
        if not accepted:
            self.status_message.emit(
                "Ignored stale S7 inference result after session/source change."
            )
            return
        self.stage_label.setText(f"Finished: {result.status}")
        self.refresh_from_state()
        warnings = f" ({len(result.warnings)} warnings)" if result.warnings else ""
        self.status_message.emit(
            f"S7 {result.status}{warnings}. Output: {result.predictions_path}"
        )

    def _on_inference_error(
        self, exc: BaseException, identity: InferenceRequestIdentity
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
            with contextlib.suppress(BehavioralInferenceUiError):
                identity = self.controller.begin_discovery()
                self._start_discovery_task(identity)
        if isinstance(exc, (BehavioralInferenceUiError, BehavioralInferenceError)):
            QMessageBox.warning(self, "Behavioral inference failed", message)
            self.status_message.emit(f"S7 failed: {message}")
        else:
            self.unexpected_error.emit(message)

    def _on_inference_finished(self, identity: InferenceRequestIdentity) -> None:
        if (
            self._active_inference_identity is not None
            and identity.request_id == self._active_inference_identity.request_id
        ):
            self._active_inference_identity = None
        self.controller.state.task_running = False
        self.progress.setVisible(False)
        self.elapsed_timer.stop()
        self._started_at = None
        self._update_enabled()
        self._emit_running_changed()
        self.task_finished.emit()

    def _open_output_folder(self) -> None:
        try:
            opened = self.controller.open_selected_s7_folder()
        except BehavioralInferenceUiError as exc:
            QMessageBox.information(self, "No S7 output", str(exc))
            self.status_message.emit(str(exc))
            return
        if not opened:
            self.status_message.emit("Could not open the S7 run folder.")
        else:
            self.status_message.emit("Opened S7 run folder.")

    def _refresh_qc_panel(self) -> None:
        rows = self.controller.qc_display_rows()
        self.qc_table.setRowCount(len(rows))
        for index, (name, value) in enumerate(rows):
            self.qc_table.setItem(index, 0, QTableWidgetItem(name))
            self.qc_table.setItem(index, 1, QTableWidgetItem(value))
        state = self.controller.state
        if state.run_error:
            self.stage_label.setText(f"Failed: {state.run_error}")
        elif not self.has_running_task:
            self.stage_label.setText(state.run_stage)

    def _update_elapsed(self) -> None:
        if self._started_at is None:
            self.elapsed_label.setText("Elapsed: —")
            return
        elapsed = time.perf_counter() - self._started_at
        self.elapsed_label.setText(f"Elapsed: {format_task_duration(elapsed)}")

    def _update_enabled(self) -> None:
        busy = self.has_running_task
        inference_busy = (
            self.controller.state.task_running or self.task_runner.is_running
        )
        # Session changes may supersede in-flight discovery; block only inference.
        self.session_browse_button.setEnabled(not inference_busy)
        self.open_session_button.setEnabled(not inference_busy)
        self.refresh_button.setEnabled(not busy)
        self.s6_combo.setEnabled(not busy)
        self.registry_path.setEnabled(not busy)
        self.registry_browse_button.setEnabled(not busy)
        self.registry_clear_button.setEnabled(not busy)
        self.parity_combo.setEnabled(not busy)
        self.exp_name_edit.setEnabled(not busy)
        self.apply_settings_button.setEnabled(not busy)
        self.select_compatible_button.setEnabled(not busy)
        self.clear_detectors_button.setEnabled(not busy)
        self.apply_override_button.setEnabled(not busy)
        self.clear_overrides_button.setEnabled(not busy)
        self.override_key_edit.setEnabled(not busy)
        self.override_value_spin.setEnabled(not busy)
        for check in self._dyadic_checks.values():
            check.setEnabled(not busy)
        for check in self._focal_checks.values():
            check.setEnabled(not busy)
        can_run = self.controller.can_run_inference() and not busy
        self.run_button.setEnabled(can_run)
        has_output = (
            self.controller.selected_s7_run() is not None
            or self.controller.state.last_result is not None
        )
        self.open_output_button.setEnabled(has_output and not busy)

    def _guard_idle(self, action: str) -> bool:
        if self.has_running_task:
            self.status_message.emit(f"Cannot {action} while an S7 task is active.")
            return False
        return True

    def _emit_running_changed(self) -> None:
        running = self.has_running_task
        if running != self._last_running_state:
            self._last_running_state = running
            self.task_running_changed.emit(running)


def _eligibility_label(classification: str) -> str:
    mapping = {
        ELIGIBLE_DOWNSTREAM: "eligible",
        COMPLETED_WITH_WARNINGS: "eligible (warnings)",
        STALE_SOURCE: "stale source",
        STALE_MODELS: "stale models",
        FAILED_OR_INCOMPLETE: "failed/incomplete",
        MISSING_REQUIRED_ARTIFACTS: "missing artifacts",
        INVALID_ARTIFACT: "invalid artifact",
    }
    return mapping.get(classification, classification)
