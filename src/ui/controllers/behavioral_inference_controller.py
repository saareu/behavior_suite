"""Headless coordination for Subsystem 07 Behavioral Inference GUI workflow.

Owns session discovery, eligible S6 source selection, model-registry / detector
readiness, threshold / parity settings, completed-run inspection, and dispatch
of ``run_behavioral_inference_publication``. Scientific scoring stays in
``behavioral_inference`` — never duplicated in the GUI.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

from behavioral_inference import (
    COMPLETED_WITH_WARNINGS,
    CONTACT_SPECIALISTS,
    ELIGIBLE_DOWNSTREAM,
    FAILED_OR_INCOMPLETE,
    INVALID_ARTIFACT,
    LOGICAL_DYADIC_DETECTORS,
    LOGICAL_FOCAL_DETECTORS,
    MISSING_REQUIRED_ARTIFACTS,
    MODEL_MANIFEST_FILENAME,
    NOSE_TO_BODY_SPECIALISTS,
    QC_FILENAME,
    READINESS_DISCOVERING,
    READINESS_MISSING_RUNTIME_DEPS,
    READINESS_MODELS_INCOMPATIBLE,
    READINESS_MODELS_MISSING,
    READINESS_NO_DETECTORS,
    READINESS_NO_SESSION,
    READINESS_OTHER,
    READINESS_READY,
    READINESS_REGISTRY_NOT_CONFIGURED,
    READINESS_REGISTRY_UNAVAILABLE,
    READINESS_S6_INELIGIBLE,
    READINESS_THRESHOLDS_UNRESOLVED,
    REGISTRY_PARTIAL,
    REGISTRY_UNAVAILABLE,
    S7_RUNTIME_INSTALL_HINT,
    STALE_MODELS,
    STALE_SOURCE,
    BehavioralInferenceConfig,
    BehavioralInferenceError,
    BehavioralInferenceProjectSummary,
    BehavioralInferenceRequest,
    BehavioralInferenceRunResult,
    BehavioralInferenceRunSummary,
    InferenceError,
    ModelRegistryError,
    ProvenanceError,
    PublicationError,
    ThresholdConfigError,
    discover_model_registry,
    find_latest_eligible_s6_for_s7,
    is_s7_run_eligible_for_downstream,
    load_behavioral_inference_config,
    probe_runtime_dependencies,
    required_bundle_names,
    resolve_nose_to_body_mode,
    resolve_working_threshold,
    run_behavioral_inference_publication,
    summarize_behavioral_inference_project,
    validate_s7_run_for_downstream,
)
from behavioral_inference.config import AdapterSettings
from behavioral_inference.gui_persistence import (
    load_persisted_registry_root,
    persist_registry_root,
    suggest_lab_model_registry,
)
from combined_features import (
    CombinedFeaturesError,
    ModelRegistrySettings,
)
from combined_features.run_discovery import (
    CombinedFeaturesProjectSummary,
    CombinedFeaturesRunSummary,
    summarize_combined_features_project,
)
from combined_features.runner import (
    is_s6_run_eligible_for_downstream,
    validate_s6_run_for_downstream,
)
from preprocess.models import OperationProgress
from ui.controllers.run_preprocess_controller import open_folder_in_platform_explorer

ProgressCallback = Callable[[OperationProgress], None]
FolderOpener = Callable[[Path], bool]
BehavioralInferenceRunner = Callable[..., BehavioralInferenceRunResult]


class SettingsStore(Protocol):
    """QSettings-compatible persistence used for the model-registry path."""

    def value(self, key: str, defaultValue: object = None) -> object: ...

    def setValue(self, key: str, value: object) -> None: ...


_THRESHOLD_DISPLAY_KEYS: tuple[str, ...] = (
    "contact_intact",
    "contact_occluded",
    "nose_to_nose",
    "mutually_disengaged",
    "side_by_side",
    "nose_body_intact",
    "nose_body_occ",
    "nose_to_tail",
    "approach",
    "retract",
)


class BehavioralInferenceUiError(ValueError):
    """Expected, actionable Subsystem 07 UI validation error."""


@dataclass(frozen=True, slots=True)
class BehavioralInferenceUiSettings:
    """GUI-exposed working settings snapshotted into each inference request."""

    parity_mode: str = "legacy_parity"
    model_iteration: str = "v1"
    model_registry_root: Path | None = None
    include_focal: bool = True
    include_obsolete: bool = False
    selected_detectors_dyadic: tuple[str, ...] = LOGICAL_DYADIC_DETECTORS
    selected_detectors_focal: tuple[str, ...] = LOGICAL_FOCAL_DETECTORS
    working_thresholds: Mapping[str, Any] = field(default_factory=dict)
    user_threshold_overrides: Mapping[str, Any] = field(default_factory=dict)
    allow_nose_to_body_single_stage_fallback: bool = True
    require_exactly_two_mice: bool = True
    emit_wide_views: bool = True
    fail_fast_on_detector_error: bool = True
    exp_name: str | None = None

    @classmethod
    def from_config(
        cls,
        config: BehavioralInferenceConfig,
        *,
        exp_name: str | None = None,
    ) -> BehavioralInferenceUiSettings:
        validated = config.validated()
        return cls(
            parity_mode=validated.parity_mode,
            model_iteration=validated.model_iteration,
            model_registry_root=validated.model_registry.root,
            include_focal=validated.model_registry.include_focal,
            include_obsolete=validated.model_registry.include_obsolete,
            selected_detectors_dyadic=validated.active_detectors_dyadic,
            selected_detectors_focal=validated.active_detectors_focal,
            working_thresholds=dict(validated.working_thresholds),
            user_threshold_overrides=dict(validated.user_threshold_overrides),
            allow_nose_to_body_single_stage_fallback=(
                validated.adapters.allow_nose_to_body_single_stage_fallback
            ),
            require_exactly_two_mice=validated.adapters.require_exactly_two_mice,
            emit_wide_views=validated.adapters.emit_wide_views,
            fail_fast_on_detector_error=validated.adapters.fail_fast_on_detector_error,
            exp_name=exp_name,
        )

    def to_config(self) -> BehavioralInferenceConfig:
        adapters = AdapterSettings(
            allow_nose_to_body_single_stage_fallback=(
                self.allow_nose_to_body_single_stage_fallback
            ),
            require_exactly_two_mice=self.require_exactly_two_mice,
            emit_wide_views=self.emit_wide_views,
            fail_fast_on_detector_error=self.fail_fast_on_detector_error,
        )
        return BehavioralInferenceConfig(
            model_iteration=str(self.model_iteration),
            parity_mode=str(self.parity_mode),
            model_registry=ModelRegistrySettings(
                root=self.model_registry_root,
                include_focal=bool(self.include_focal),
                include_obsolete=bool(self.include_obsolete),
            ),
            active_detectors_dyadic=tuple(self.selected_detectors_dyadic),
            active_detectors_focal=tuple(self.selected_detectors_focal),
            working_thresholds=dict(self.working_thresholds),
            user_threshold_overrides=dict(self.user_threshold_overrides),
            adapters=adapters,
        ).validated()

    def validated(self) -> BehavioralInferenceUiSettings:
        return self.from_config(self.to_config(), exp_name=self.exp_name)


@dataclass(frozen=True, slots=True)
class InferenceRequestIdentity:
    """Identity used to reject stale asynchronous inference results."""

    request_id: str
    session_root: Path
    s6_run_id: str
    s6_run_dir: Path
    settings: BehavioralInferenceUiSettings


@dataclass(frozen=True, slots=True)
class DiscoveryRequestIdentity:
    """Identity used to reject stale asynchronous discovery results."""

    request_id: str
    generation: int
    session_root: Path


@dataclass(frozen=True, slots=True)
class DiscoverySnapshot:
    """Payload collected off the GUI thread for one discovery generation."""

    identity: DiscoveryRequestIdentity
    s6_project: CombinedFeaturesProjectSummary
    s7_summary: BehavioralInferenceProjectSummary


@dataclass(frozen=True, slots=True)
class S6SourceOption:
    """One selectable eligible S6 run for inference input."""

    run_id: str
    run_dir: Path
    status: str | None
    classification: str
    label: str
    sample_count: int | None = None
    track_count: int | None = None
    session_identity: str | None = None
    structural_failures: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DetectorAvailability:
    """Logical detector readiness derived from registry discovery (no model load)."""

    name: str
    family: str
    selected: bool
    ready: bool
    status: str
    specialist_deps: tuple[str, ...]
    detail: str


@dataclass(slots=True)
class BehavioralInferenceUiState:
    """Mutable S7 workspace state (working settings vs inference snapshots)."""

    session_root: Path | None = None
    s6_options: tuple[S6SourceOption, ...] = ()
    selected_s6_run_id: str | None = None
    s6_selection_manual: bool = False
    s6_project: CombinedFeaturesProjectSummary | None = None
    summary: BehavioralInferenceProjectSummary | None = None
    selected_s7_run_id: str | None = None
    readiness_kind: str = READINESS_NO_SESSION
    readiness_error: str | None = None
    readiness_warnings: tuple[str, ...] = ()
    working_settings: BehavioralInferenceUiSettings = field(
        default_factory=BehavioralInferenceUiSettings
    )
    last_inference_settings: BehavioralInferenceUiSettings | None = None
    registry_status: str | None = None
    registry_notes: tuple[str, ...] = ()
    detector_availability: tuple[DetectorAvailability, ...] = ()
    runtime_deps_ok: bool = False
    runtime_deps_detail: str = ""
    python_executable: str = field(default_factory=lambda: sys.executable)
    task_running: bool = False
    discovery_running: bool = False
    discovery_generation: int = 0
    active_discovery_identity: DiscoveryRequestIdentity | None = None
    active_inference_identity: InferenceRequestIdentity | None = None
    run_stage: str = "Not started"
    run_error: str | None = None
    last_result: BehavioralInferenceRunResult | None = None
    unexpected_error_detail: str | None = None


class BehavioralInferenceController:
    """Session discovery, S6 source selection, settings, inference dispatch."""

    def __init__(
        self,
        *,
        config_loader: Callable[[], BehavioralInferenceConfig] | None = None,
        folder_opener: FolderOpener = open_folder_in_platform_explorer,
        runner: BehavioralInferenceRunner = run_behavioral_inference_publication,
        settings: SettingsStore | None = None,
    ) -> None:
        self.state = BehavioralInferenceUiState()
        self._config_loader = config_loader or (
            lambda: load_behavioral_inference_config(None)
        )
        self._folder_opener = folder_opener
        self._runner = runner
        self._settings = settings
        self.reset_working_settings_to_defaults()

    def reset_working_settings_to_defaults(self) -> BehavioralInferenceUiSettings:
        try:
            settings = BehavioralInferenceUiSettings.from_config(self._config_loader())
        except (BehavioralInferenceError, ThresholdConfigError) as exc:
            raise BehavioralInferenceUiError(str(exc)) from exc
        persisted = load_persisted_registry_root(self._settings)
        if settings.model_registry_root is None and persisted is not None:
            settings = replace(settings, model_registry_root=persisted)
        self.state.working_settings = settings
        self._refresh_registry_and_detectors()
        self._recompute_readiness()
        return settings

    def suggested_registry_browse_start(self) -> str:
        """Directory to seed the registry file dialog."""

        current = self.state.working_settings.model_registry_root
        if current is not None:
            return str(current)
        suggested = suggest_lab_model_registry()
        return str(suggested) if suggested is not None else ""

    def set_session(self, session_root: Path | str) -> BehavioralInferenceProjectSummary:
        """Open a project/session and refresh S6 / S7 discovery (synchronous)."""

        self.prepare_session(session_root)
        return self.refresh_discovery()

    def prepare_session(self, session_root: Path | str) -> DiscoveryRequestIdentity:
        """Bind a session and bump discovery generation without scanning yet."""

        root = Path(session_root).expanduser().resolve(strict=False)
        if not root.is_dir():
            raise BehavioralInferenceUiError(f"Session directory does not exist: {root}")
        previous = self.state.session_root
        if (
            self.state.task_running
            and previous is not None
            and previous.resolve(strict=False) != root
        ):
            raise BehavioralInferenceUiError(
                "Cannot change S7 session while behavioral inference is active."
            )
        self.state.session_root = root
        self.state.readiness_error = None
        self.state.readiness_kind = READINESS_DISCOVERING
        self.state.run_error = None
        self.state.run_stage = "Not started"
        if previous is None or previous.resolve(strict=False) != root:
            self.state.last_result = None
            self.state.last_inference_settings = None
            self.state.selected_s7_run_id = None
            self.state.selected_s6_run_id = None
            self.state.s6_selection_manual = False
            self.state.s6_options = ()
            self.state.s6_project = None
            self.state.summary = None
        return self.begin_discovery()

    def clear_session(self) -> None:
        if self.state.task_running or self.state.discovery_running:
            raise BehavioralInferenceUiError(
                "Cannot clear the S7 session while a task is active."
            )
        working = self.state.working_settings
        self.state = BehavioralInferenceUiState(working_settings=working)
        self._refresh_registry_and_detectors()
        self._recompute_readiness()

    def begin_discovery(self) -> DiscoveryRequestIdentity:
        """Mark discovery in progress and return an identity for async apply."""

        root = self.state.session_root
        if root is None:
            raise BehavioralInferenceUiError("Select a project/session first.")
        self.state.discovery_generation += 1
        identity = DiscoveryRequestIdentity(
            request_id=str(uuid4()),
            generation=self.state.discovery_generation,
            session_root=root.resolve(strict=False),
        )
        self.state.active_discovery_identity = identity
        self.state.discovery_running = True
        self.state.readiness_kind = READINESS_DISCOVERING
        self.state.readiness_error = (
            "Discovering eligible S6 runs and validating model/source digests…"
        )
        return identity

    def collect_discovery_snapshot(
        self, identity: DiscoveryRequestIdentity
    ) -> DiscoverySnapshot:
        """Heavy filesystem / hashing work safe to run off the GUI thread."""

        root = identity.session_root
        try:
            s6_project = summarize_combined_features_project(root)
            s7_summary = summarize_behavioral_inference_project(root)
        except (CombinedFeaturesError, PublicationError, OSError) as exc:
            raise BehavioralInferenceUiError(str(exc)) from exc
        return DiscoverySnapshot(
            identity=identity,
            s6_project=s6_project,
            s7_summary=s7_summary,
        )

    def apply_discovery_snapshot(self, snapshot: DiscoverySnapshot) -> bool:
        """Apply a discovery result only when its generation is still current."""

        if not self._is_current_discovery_result(snapshot.identity):
            return False
        self.state.discovery_running = False
        self.state.active_discovery_identity = None
        self.state.s6_project = snapshot.s6_project
        self.state.s6_options = tuple(
            _s6_option_from_summary(run) for run in snapshot.s6_project.eligible_runs
        )
        self.state.summary = snapshot.s7_summary
        self._reconcile_s6_selection()
        self._reconcile_s7_selection()
        self._refresh_registry_and_detectors()
        self._recompute_readiness()
        return True

    def record_discovery_failure(
        self, message: str, *, expected_identity: DiscoveryRequestIdentity | None
    ) -> bool:
        """Record a discovery failure when the request is still current."""

        if not self._is_current_discovery_result(expected_identity):
            return False
        self.state.discovery_running = False
        self.state.active_discovery_identity = None
        self.state.readiness_kind = READINESS_OTHER
        self.state.readiness_error = message
        return True

    def refresh_discovery(self) -> BehavioralInferenceProjectSummary:
        """Re-scan eligible S6 inputs and completed S7 outputs (synchronous)."""

        identity = self.begin_discovery()
        snapshot = self.collect_discovery_snapshot(identity)
        applied = self.apply_discovery_snapshot(snapshot)
        if not applied or self.state.summary is None:
            raise BehavioralInferenceUiError(
                "Discovery result was superseded by a newer session selection."
            )
        return self.state.summary

    def _is_current_discovery_result(
        self, expected: DiscoveryRequestIdentity | None
    ) -> bool:
        if expected is None:
            return False
        active = self.state.active_discovery_identity
        if active is None:
            return False
        if expected.request_id != active.request_id:
            return False
        if expected.generation != self.state.discovery_generation:
            return False
        current_root = self.state.session_root
        if current_root is None:
            return False
        return (
            current_root.resolve(strict=False)
            == expected.session_root.resolve(strict=False)
        )

    def select_s6_source(
        self, run_id: str | None, *, manual: bool = True
    ) -> S6SourceOption | None:
        """Select an eligible S6 input run (manual selections are preserved)."""

        if self.state.task_running or self.state.discovery_running:
            raise BehavioralInferenceUiError(
                "Cannot change S6 source while a Subsystem 7 task is active."
            )
        if run_id is None or not str(run_id).strip():
            self.state.selected_s6_run_id = None
            if manual:
                self.state.s6_selection_manual = True
            self._recompute_readiness()
            return None
        match = next((opt for opt in self.state.s6_options if opt.run_id == run_id), None)
        if match is None:
            raise BehavioralInferenceUiError(
                f"Eligible S6 run is not available in the current session: {run_id}"
            )
        self.state.selected_s6_run_id = match.run_id
        if manual:
            self.state.s6_selection_manual = True
        self._recompute_readiness()
        return match

    def select_s7_run(self, run_id: str | None) -> BehavioralInferenceRunSummary | None:
        """Select a discovered S7 run for QC / reopen inspection."""

        if run_id is None or not str(run_id).strip():
            self.state.selected_s7_run_id = None
            return None
        summary = self.state.summary
        if summary is None:
            raise BehavioralInferenceUiError("Refresh discovery before selecting an S7 run.")
        match = next((run for run in summary.runs if run.run_id == run_id), None)
        if match is None:
            raise BehavioralInferenceUiError(f"S7 run was not found: {run_id}")
        self.state.selected_s7_run_id = match.run_id
        return match

    def selected_s6_source(self) -> S6SourceOption | None:
        selected = self.state.selected_s6_run_id
        if selected is None:
            return None
        return next((opt for opt in self.state.s6_options if opt.run_id == selected), None)

    def selected_s7_run(self) -> BehavioralInferenceRunSummary | None:
        summary = self.state.summary
        selected = self.state.selected_s7_run_id
        if summary is None or selected is None:
            return None
        return next((run for run in summary.runs if run.run_id == selected), None)

    def update_settings(
        self, settings: BehavioralInferenceUiSettings
    ) -> BehavioralInferenceUiSettings:
        """Validate and store working settings (snapshotted at run time)."""

        if self.state.task_running or self.state.discovery_running:
            raise BehavioralInferenceUiError(
                "Cannot change S7 settings while a Subsystem 7 task is active."
            )
        validated = self.validate_settings(settings)
        previous_root = self.state.working_settings.model_registry_root
        self.state.working_settings = validated
        if previous_root != validated.model_registry_root:
            persist_registry_root(self._settings, validated.model_registry_root)
        self._refresh_registry_and_detectors()
        self._recompute_readiness()
        return validated

    def validate_settings(
        self, settings: BehavioralInferenceUiSettings | None = None
    ) -> BehavioralInferenceUiSettings:
        candidate = settings if settings is not None else self.state.working_settings
        try:
            return candidate.validated()
        except (BehavioralInferenceError, ThresholdConfigError) as exc:
            raise BehavioralInferenceUiError(str(exc)) from exc

    def set_model_registry_root(
        self, root: Path | str | None
    ) -> BehavioralInferenceUiSettings:
        """Update the required model-registry path."""

        path: Path | None
        if root is None or str(root).strip() == "":
            path = None
        else:
            path = Path(root).expanduser().resolve(strict=False)
        return self.update_settings(
            replace(self.state.working_settings, model_registry_root=path)
        )

    def set_selected_detectors(
        self,
        *,
        dyadic: Sequence[str] | None = None,
        focal: Sequence[str] | None = None,
    ) -> BehavioralInferenceUiSettings:
        settings = self.state.working_settings
        new_dyadic = (
            tuple(str(x) for x in dyadic)
            if dyadic is not None
            else settings.selected_detectors_dyadic
        )
        new_focal = (
            tuple(str(x) for x in focal)
            if focal is not None
            else settings.selected_detectors_focal
        )
        return self.update_settings(
            replace(
                settings,
                selected_detectors_dyadic=new_dyadic,
                selected_detectors_focal=new_focal,
            )
        )

    def select_all_compatible_detectors(self) -> BehavioralInferenceUiSettings:
        """Select every logical detector whose required bundles are currently ready."""

        ready_dyadic = tuple(
            d.name
            for d in self.state.detector_availability
            if d.family == "dyadic" and d.ready
        )
        ready_focal = tuple(
            d.name
            for d in self.state.detector_availability
            if d.family == "focal" and d.ready
        )
        return self.set_selected_detectors(dyadic=ready_dyadic, focal=ready_focal)

    def set_threshold_override(
        self, detector_key: str, value: float | None
    ) -> BehavioralInferenceUiSettings:
        """Set or clear one user threshold override (nested keys use dotted form)."""

        key = str(detector_key).strip()
        overrides = dict(self.state.working_settings.user_threshold_overrides)
        if value is None:
            overrides.pop(key, None)
            # Also clear nested parent.child if keyed that way.
            if "." in key:
                parent, child = key.split(".", 1)
                parent_val = overrides.get(parent)
                if isinstance(parent_val, dict):
                    parent_val = dict(parent_val)
                    parent_val.pop(child, None)
                    if parent_val:
                        overrides[parent] = parent_val
                    else:
                        overrides.pop(parent, None)
        else:
            overrides[key] = float(value)
        return self.update_settings(
            replace(self.state.working_settings, user_threshold_overrides=overrides)
        )

    def clear_threshold_overrides(self) -> BehavioralInferenceUiSettings:
        return self.update_settings(
            replace(self.state.working_settings, user_threshold_overrides={})
        )

    def can_run_inference(self) -> bool:
        return self.cannot_run_reason() is None

    def cannot_run_reason(self) -> str | None:
        if self.state.task_running:
            return "A Subsystem 7 inference task is already running."
        if self.state.discovery_running:
            return "S7 discovery is still running; wait for readiness to finish."
        if self.state.session_root is None:
            return "Select a project/session first."
        if self.state.readiness_kind != READINESS_READY:
            return self.state.readiness_error or "Subsystem 7 is not ready for inference."
        dyadic = self.state.working_settings.selected_detectors_dyadic
        focal = self.state.working_settings.selected_detectors_focal
        if not dyadic and not focal:
            return "Select at least one dyadic or focal detector."
        return None

    def source_status_text(self) -> str:
        """Human-readable S6 source readiness summary."""

        kind = self.state.readiness_kind
        if kind == READINESS_NO_SESSION or self.state.session_root is None:
            return "Select a project/session with an eligible S6 run."
        if kind == READINESS_DISCOVERING or self.state.discovery_running:
            return self.state.readiness_error or "Discovering S6/S7 artifacts…"
        if not self.state.s6_options:
            reason = (
                self.state.s6_project.no_pair_reason
                if self.state.s6_project is not None
                else None
            )
            return (
                "No currently eligible S6 run was found for this session."
                + (f" ({reason})" if reason else "")
                + " Complete Subsystem 6, then Refresh discovery."
            )
        source = self.selected_s6_source()
        if source is None:
            return self.state.readiness_error or "Select an eligible S6 run."
        manual = " (manual)" if self.state.s6_selection_manual else " (default)"
        samples = (
            f"{source.sample_count} samples"
            if source.sample_count is not None
            else "sample count unknown"
        )
        tracks = (
            f"{source.track_count} tracks"
            if source.track_count is not None
            else "track count unknown"
        )
        failures = (
            f" Structural/provenance notes: {'; '.join(source.structural_failures)}."
            if source.structural_failures
            else ""
        )
        if kind == READINESS_READY and self.state.readiness_error is None:
            ready = "Ready for inference (runtime installed and models configured)."
        else:
            ready = f"Blocked [{kind}]: {self.state.readiness_error}"
        return (
            f"S6={source.run_id}: status={source.status or '—'}, "
            f"eligibility={source.classification}{manual}; {samples}, {tracks}. "
            f"{ready}{failures}"
        )

    def threshold_display_rows(self) -> tuple[tuple[str, str, str], ...]:
        """Rows of (detector_key, value, source) for the thresholds panel."""

        try:
            config = self.state.working_settings.to_config()
        except (BehavioralInferenceError, ThresholdConfigError):
            return ()
        rows: list[tuple[str, str, str]] = []
        for key in _THRESHOLD_DISPLAY_KEYS:
            try:
                resolved = resolve_working_threshold(key, config)
            except ThresholdConfigError as exc:
                rows.append((key, "—", f"unresolved: {exc}"))
                continue
            rows.append((key, f"{resolved.value:.6g}", resolved.source))
        return tuple(rows)

    def build_request(
        self, *, timestamp: str | None = None
    ) -> tuple[BehavioralInferenceRequest, InferenceRequestIdentity]:
        """Validate readiness, snapshot settings, and build a publication request."""

        reason = self.cannot_run_reason()
        if reason is not None:
            raise BehavioralInferenceUiError(reason)
        source = self.selected_s6_source()
        root = self.state.session_root
        if source is None or root is None:
            raise BehavioralInferenceUiError("Select an eligible S6 run first.")
        settings = self.validate_settings()
        # Definitive S6 eligibility check immediately before execution.
        try:
            if not is_s6_run_eligible_for_downstream(source.run_dir):
                raise BehavioralInferenceUiError(
                    f"Selected S6 run is no longer eligible for downstream use: "
                    f"{source.run_id}"
                )
            validate_s6_run_for_downstream(source.run_dir)
        except CombinedFeaturesError as exc:
            self.state.readiness_error = str(exc)
            raise BehavioralInferenceUiError(str(exc)) from exc

        config_snapshot = settings.to_config()
        # Final lightweight registry / threshold gate before worker load.
        self._assert_execution_readiness(config_snapshot, source)

        request = BehavioralInferenceRequest(
            session_root=root,
            s6_run_dir=source.run_dir,
            config=config_snapshot,
            selected_detectors_dyadic=settings.selected_detectors_dyadic,
            selected_detectors_focal=settings.selected_detectors_focal,
            run_purpose="gui",
            timestamp=timestamp,
            exp_name=settings.exp_name,
        )
        identity = InferenceRequestIdentity(
            request_id=str(uuid4()),
            session_root=root.resolve(strict=False),
            s6_run_id=source.run_id,
            s6_run_dir=source.run_dir.resolve(strict=False),
            settings=settings,
        )
        self.state.active_inference_identity = identity
        self.state.task_running = True
        self.state.run_error = None
        self.state.run_stage = "Running behavioral inference…"
        self.state.readiness_error = None
        return request, identity

    def execute_request(
        self,
        request: BehavioralInferenceRequest,
        progress_callback: ProgressCallback | None = None,
    ) -> BehavioralInferenceRunResult:
        """Run one validated request (worker-thread safe; no Qt updates)."""

        return self._runner(request, progress_callback=progress_callback)

    def record_success(
        self,
        result: BehavioralInferenceRunResult,
        *,
        expected_identity: InferenceRequestIdentity | None = None,
    ) -> bool:
        """Apply a successful result only when the request identity is still current."""

        self.state.task_running = False
        if not self._is_current_inference_result(expected_identity):
            return False
        self.state.last_result = result
        self.state.run_stage = result.status
        self.state.run_error = None
        self.state.last_inference_settings = (
            expected_identity.settings
            if expected_identity is not None
            else self.state.working_settings
        )
        self.state.selected_s7_run_id = result.run_id
        self.state.active_inference_identity = None
        try:
            self.refresh_discovery()
            self.state.selected_s7_run_id = result.run_id
        except BehavioralInferenceUiError:
            pass
        return True

    def record_failure(
        self,
        message: str,
        *,
        expected_identity: InferenceRequestIdentity | None = None,
    ) -> bool:
        """Record a failed run without marking it completed."""

        self.state.task_running = False
        if (
            expected_identity is not None
            and self.state.active_inference_identity is not None
            and expected_identity.request_id
            != self.state.active_inference_identity.request_id
        ):
            return False
        self.state.last_result = None
        self.state.run_stage = "Failed"
        self.state.run_error = message
        self.state.active_inference_identity = None
        return True

    def open_selected_s7_folder(self) -> bool:
        run = self.selected_s7_run()
        if run is None and self.state.last_result is not None:
            return bool(self._folder_opener(self.state.last_result.run_dir))
        if run is None:
            raise BehavioralInferenceUiError("Select a completed S7 run first.")
        return bool(self._folder_opener(run.run_dir))

    def qc_display_rows(self) -> tuple[tuple[str, str], ...]:
        """Compact QC / prediction-summary rows for the results panel."""

        result = self.state.last_result
        run = self.selected_s7_run()
        meta_qc: Mapping[str, Any] | None = None
        model_manifest: Mapping[str, Any] | None = None
        status = None
        output = None
        classification = None
        source_s6 = None
        detectors_dyadic: tuple[str, ...] = ()
        detectors_focal: tuple[str, ...] = ()

        use_last = result is not None and (run is None or run.run_id == result.run_id)
        if use_last and result is not None:
            status = result.status
            output = (
                str(result.predictions_path)
                if result.predictions_path is not None
                else str(result.run_dir)
            )
            meta_qc = result.qc
            model_manifest = result.model_manifest
            source_s6 = result.source_s6_run_id
            if run is not None:
                classification = run.classification
                source_s6 = run.source_s6_run_id or source_s6
                detectors_dyadic = run.detectors_dyadic
                detectors_focal = run.detectors_focal
        elif run is not None:
            status = run.status
            output = (
                str(run.predictions_path) if run.predictions_path else str(run.run_dir)
            )
            classification = run.classification
            source_s6 = run.source_s6_run_id
            detectors_dyadic = run.detectors_dyadic
            detectors_focal = run.detectors_focal
            meta_qc = _load_json_mapping(run.run_dir / QC_FILENAME)
            model_manifest = _load_json_mapping(run.run_dir / MODEL_MANIFEST_FILENAME)
        else:
            return ()

        rows: list[tuple[str, str]] = [
            ("Historical status", status or "—"),
            (
                "Current eligibility",
                _eligibility_label(classification) if classification else "—",
            ),
            ("Source S6", source_s6 or "—"),
            (
                "Detectors (dyadic)",
                ", ".join(detectors_dyadic) if detectors_dyadic else "—",
            ),
            (
                "Detectors (focal)",
                ", ".join(detectors_focal) if detectors_focal else "—",
            ),
            ("Output", output or "—"),
        ]

        if use_last and result is not None:
            rows.append(
                (
                    "Wide views",
                    (
                        f"dyadic={'yes' if result.dyadic_wide_path else 'no'}; "
                        f"focal={'yes' if result.focal_wide_path else 'no'}"
                    ),
                )
            )
            if result.warnings:
                rows.append(("Run warnings", "; ".join(result.warnings)))

        if isinstance(model_manifest, Mapping):
            rows.append(
                (
                    "Model registry / manifest",
                    str(
                        model_manifest.get("registry_root")
                        or model_manifest.get("schema_version")
                        or "present"
                    ),
                )
            )
            iteration = model_manifest.get("model_iteration")
            if iteration is not None:
                rows.append(("Model iteration", str(iteration)))

        if isinstance(meta_qc, Mapping):
            metrics = meta_qc.get("metrics")
            if isinstance(metrics, Mapping):
                for key, label in (
                    ("n_unique_frames", "Sample / frame count"),
                    ("frame_coverage_n", "Frame coverage"),
                    ("n_input_invalid_predictions", "Invalid-routing count"),
                    ("n_nonfinite_prob_effective", "Non-finite score warnings"),
                    ("dyadic_wide_rows", "Dyadic wide rows"),
                    ("focal_wide_rows", "Focal wide rows"),
                ):
                    value = metrics.get(key)
                    if value is not None:
                        rows.append((label, _format_metric(value)))
                pos_stats = metrics.get("detector_positive_stats")
                if isinstance(pos_stats, Mapping):
                    for detector, stats in pos_stats.items():
                        if not isinstance(stats, Mapping):
                            continue
                        n_pos = stats.get("n_positive")
                        frac = stats.get("positive_fraction")
                        n_inv = stats.get("n_input_invalid")
                        grain = "dyadic" if detector in LOGICAL_DYADIC_DETECTORS or (
                            str(detector).startswith("contact")
                        ) else "focal"
                        # Logical fused contact + standalone logical names only.
                        if detector in LOGICAL_DYADIC_DETECTORS or detector in LOGICAL_FOCAL_DETECTORS:
                            rows.append(
                                (
                                    f"Positives [{grain}] {detector}",
                                    (
                                        f"n={n_pos}, fraction={_format_metric(frac)}, "
                                        f"invalid={n_inv}"
                                    ),
                                )
                            )
                coverage = metrics.get("detector_coverage")
                if isinstance(coverage, Mapping):
                    rows.append(("Detector coverage", _format_metric(coverage)))
                invalid_reasons = metrics.get("invalid_reason_counts")
                if isinstance(invalid_reasons, Mapping) and invalid_reasons:
                    rows.append(
                        ("Invalid-route reasons", _format_metric(invalid_reasons))
                    )
            structural = meta_qc.get("structural_ok")
            if structural is not None:
                rows.append(("Structural QC", "OK" if structural else "FAILED"))
            warnings = meta_qc.get("warnings")
            if isinstance(warnings, list) and warnings:
                rows.append(("QC warnings", "; ".join(str(w) for w in warnings)))
            failures = meta_qc.get("failures")
            if isinstance(failures, list) and failures:
                rows.append(("QC failures", "; ".join(str(f) for f in failures)))

        # S8 handoff readiness (artifact eligibility ≠ required-detector coverage).
        handoff = self.s8_handoff_rows(run_dir=_run_dir_for_qc(result, run))
        rows.extend(handoff)
        return tuple(rows)

    def s8_handoff_rows(
        self, *, run_dir: Path | None = None
    ) -> tuple[tuple[str, str], ...]:
        """Distinguish S7 artifact eligibility from S8 detector/view coverage."""

        path = run_dir
        if path is None:
            return ()
        rows: list[tuple[str, str]] = []
        try:
            eligible = is_s7_run_eligible_for_downstream(path)
            detail = validate_s7_run_for_downstream(path)
        except (PublicationError, ProvenanceError, OSError) as exc:
            rows.append(("S8 handoff (S7 eligible)", f"no — {exc}"))
            return tuple(rows)

        rows.append(("S8 handoff (S7 eligible)", "yes" if eligible else "no"))
        coverage = detail.get("detector_coverage")
        if isinstance(coverage, Mapping):
            selected = coverage.get("detectors_selected") or coverage
            rows.append(
                (
                    "S8 required-view note",
                    "S7 eligibility does not require all detectors; "
                    f"coverage={_format_metric(selected)}",
                )
            )
        else:
            rows.append(
                (
                    "S8 required-view note",
                    "S7 eligibility does not require all detectors; check coverage.",
                )
            )
        return tuple(rows)

    def _reconcile_s6_selection(self) -> None:
        """Default to newest eligible S6 unless the user chose manually."""

        ids = {opt.run_id for opt in self.state.s6_options}
        if self.state.selected_s6_run_id not in ids:
            # Manual selection that is no longer eligible is cleared (not silently
            # replaced with another run while still marked manual).
            if self.state.s6_selection_manual and self.state.selected_s6_run_id is not None:
                self.state.selected_s6_run_id = None
                self.state.readiness_error = (
                    "Previously selected S6 run is no longer eligible. "
                    "Choose another eligible run."
                )
                return
            self.state.selected_s6_run_id = None
            self.state.s6_selection_manual = False

        if self.state.s6_selection_manual:
            return
        if self.state.session_root is None:
            return
        latest = find_latest_eligible_s6_for_s7(self.state.session_root)
        if latest is None:
            self.state.selected_s6_run_id = None
            return
        self.state.selected_s6_run_id = latest.run_id
        self.state.readiness_error = None

    def _reconcile_s7_selection(self) -> None:
        summary = self.state.summary
        if summary is None or not summary.runs:
            if self.state.last_result is None:
                self.state.selected_s7_run_id = None
            return
        current = self.state.selected_s7_run_id
        if current is not None and any(run.run_id == current for run in summary.runs):
            return
        eligible = summary.eligible_runs
        if eligible:
            self.state.selected_s7_run_id = eligible[0].run_id
            return
        completed = summary.completed_runs
        self.state.selected_s7_run_id = (
            completed[0].run_id if completed else summary.runs[0].run_id
        )

    def _refresh_registry_and_detectors(self) -> None:
        self.state.python_executable = sys.executable
        deps = probe_runtime_dependencies()
        self.state.runtime_deps_ok = bool(deps.available)
        exe = self.state.python_executable
        if deps.available:
            parts = [
                f"{name}={ver}"
                for name, ver in deps.versions.items()
                if ver
            ]
            self.state.runtime_deps_detail = (
                f"Runtime deps available in {exe}: "
                + (", ".join(parts) if parts else "ok")
                + ". Model bundles are configured separately via the registry path."
            )
        else:
            missing = ", ".join(deps.missing) if deps.missing else "unknown"
            self.state.runtime_deps_detail = (
                f"Runtime deps missing ({missing}) in {exe}. "
                f"{S7_RUNTIME_INSTALL_HINT}"
            )

        settings = self.state.working_settings
        if settings.model_registry_root is None:
            suggested = suggest_lab_model_registry()
            hint = (
                f" Browse to a trusted models directory"
                f"{f' (lab suggestion: {suggested})' if suggested is not None else ''}."
            )
            self.state.registry_status = REGISTRY_UNAVAILABLE
            self.state.registry_notes = (
                "model_registry.root is not configured." + hint,
            )
            self.state.detector_availability = ()
            return

        try:
            registry_settings = settings.to_config().model_registry
        except (BehavioralInferenceError, ThresholdConfigError) as exc:
            self.state.registry_status = REGISTRY_UNAVAILABLE
            self.state.registry_notes = (str(exc),)
            self.state.detector_availability = ()
            return

        root = registry_settings.root
        if root is not None and not Path(root).is_dir():
            self.state.registry_status = REGISTRY_UNAVAILABLE
            self.state.registry_notes = (
                f"Configured registry path is missing or not a directory: {root}. "
                "Choose an existing trusted model-registry root.",
            )
            self.state.detector_availability = ()
            return

        try:
            snapshot = discover_model_registry(registry_settings)
        except ModelRegistryError as exc:
            self.state.registry_status = REGISTRY_UNAVAILABLE
            self.state.registry_notes = (str(exc),)
            self.state.detector_availability = ()
            return

        self.state.registry_status = snapshot.status
        notes = list(snapshot.notes)
        loadable = snapshot.loadable_bundles()
        if snapshot.status == REGISTRY_UNAVAILABLE:
            notes.append(
                "Registry path is set but no usable model layout was discovered."
            )
        elif not loadable:
            notes.append(
                "Registry configured but no loadable model bundles were found. "
                "Confirm dyadic/v1 and focal/v1 bundle directories."
            )
        elif snapshot.status == REGISTRY_PARTIAL:
            notes.append(
                "Some expected model bundles are missing or incomplete."
            )
        self.state.registry_notes = tuple(notes)
        self.state.detector_availability = _build_detector_availability(
            snapshot=snapshot,
            settings=settings,
        )

    def _set_readiness(
        self,
        kind: str,
        error: str | None,
        warnings: Sequence[str] = (),
    ) -> None:
        self.state.readiness_kind = kind
        self.state.readiness_error = error
        self.state.readiness_warnings = tuple(warnings)

    def _recompute_readiness(self) -> None:
        warnings: list[str] = []
        if self.state.discovery_running:
            self._set_readiness(
                READINESS_DISCOVERING,
                self.state.readiness_error
                or "Discovering eligible S6 runs and validating digests…",
            )
            return
        if self.state.session_root is None:
            self._set_readiness(
                READINESS_NO_SESSION,
                "Select a project/session first.",
            )
            return

        if not self.state.runtime_deps_ok:
            self._set_readiness(
                READINESS_MISSING_RUNTIME_DEPS,
                self.state.runtime_deps_detail,
                warnings,
            )
            return

        source = self.selected_s6_source()
        if source is None:
            self._set_readiness(
                READINESS_S6_INELIGIBLE,
                (
                    "Select an eligible S6 run."
                    if self.state.s6_options
                    else "No currently eligible S6 run is available. "
                    "Complete Subsystem 6 before Behavioral Inference."
                ),
            )
            return

        if source.classification not in (ELIGIBLE_DOWNSTREAM, COMPLETED_WITH_WARNINGS):
            self._set_readiness(
                READINESS_S6_INELIGIBLE,
                f"Selected S6 run is not currently eligible ({source.classification}).",
            )
            return

        try:
            if not is_s6_run_eligible_for_downstream(source.run_dir):
                self._set_readiness(
                    READINESS_S6_INELIGIBLE,
                    f"Selected S6 run failed live eligibility: {source.run_id}",
                )
                return
        except CombinedFeaturesError as exc:
            self._set_readiness(READINESS_S6_INELIGIBLE, str(exc))
            return

        if source.track_count is not None and source.track_count != 2:
            if self.state.working_settings.require_exactly_two_mice:
                self._set_readiness(
                    READINESS_S6_INELIGIBLE,
                    f"MVP requires exactly two tracked mice (found {source.track_count}).",
                )
                return
            warnings.append(
                f"Track count is {source.track_count}; two-mouse MVP check disabled."
            )

        if self.state.working_settings.model_registry_root is None:
            suggested = suggest_lab_model_registry()
            hint = (
                f" Suggested local lab path when present: {suggested}."
                if suggested is not None
                else ""
            )
            self._set_readiness(
                READINESS_REGISTRY_NOT_CONFIGURED,
                "Registry not configured. Browse to a trusted local model-registry "
                "root; the path is saved for later launches." + hint,
                warnings,
            )
            return

        if self.state.registry_status == REGISTRY_UNAVAILABLE:
            note = "; ".join(self.state.registry_notes) if self.state.registry_notes else ""
            kind = (
                READINESS_MODELS_MISSING
                if self.state.working_settings.model_registry_root is not None
                else READINESS_REGISTRY_UNAVAILABLE
            )
            self._set_readiness(
                kind,
                "Registry configured but model files are missing or unusable."
                + (f" {note}" if note else ""),
                warnings,
            )
            return

        dyadic = self.state.working_settings.selected_detectors_dyadic
        focal = self.state.working_settings.selected_detectors_focal
        if not dyadic and not focal:
            self._set_readiness(
                READINESS_NO_DETECTORS,
                "Select at least one dyadic or focal detector.",
                warnings,
            )
            return

        not_ready = [
            d
            for d in self.state.detector_availability
            if d.selected and not d.ready
        ]
        if not_ready:
            details = "; ".join(f"{d.name}: {d.detail}" for d in not_ready)
            missing_like = any(
                "missing" in d.detail.lower() or d.status in {"incomplete", "missing"}
                for d in not_ready
            )
            incompatible_like = any(
                "unsupported" in d.detail.lower()
                or "obsolete" in d.detail.lower()
                or "digest" in d.detail.lower()
                for d in not_ready
            )
            kind = (
                READINESS_MODELS_MISSING
                if missing_like and not incompatible_like
                else READINESS_MODELS_INCOMPATIBLE
            )
            self._set_readiness(
                kind,
                "Models discovered but selected detector(s) are not ready: "
                f"{details}",
                warnings,
            )
            return

        try:
            config = self.state.working_settings.to_config()
            for key in _threshold_keys_for_selection(dyadic, focal):
                resolve_working_threshold(key, config)
        except (BehavioralInferenceError, ThresholdConfigError) as exc:
            self._set_readiness(READINESS_THRESHOLDS_UNRESOLVED, str(exc), warnings)
            return

        if self.state.working_settings.parity_mode == "raw_framewise":
            warnings.append(
                "Smoothing mode is raw_framewise — nose-to-body rolling median is "
                "disabled; scientific predictions differ from legacy parity."
            )
        if self.state.working_settings.user_threshold_overrides:
            warnings.append(
                "User threshold overrides are active and will change scientific "
                "predictions relative to the inference configuration defaults."
            )

        self._set_readiness(READINESS_READY, None, warnings)

    def _assert_execution_readiness(
        self,
        config: BehavioralInferenceConfig,
        source: S6SourceOption,
    ) -> None:
        """Definitive pre-execution checks (still without loading every estimator)."""

        if not self.state.runtime_deps_ok:
            raise BehavioralInferenceUiError(self.state.runtime_deps_detail)
        snapshot = discover_model_registry(config.model_registry)
        if snapshot.status == REGISTRY_UNAVAILABLE:
            raise BehavioralInferenceUiError(
                "Model registry became unavailable before execution."
            )
        if "nose_to_body" in config.active_detectors_focal:
            try:
                mode = resolve_nose_to_body_mode(
                    snapshot,
                    allow_fallback=(
                        config.adapters.allow_nose_to_body_single_stage_fallback
                    ),
                )
            except InferenceError as exc:
                raise BehavioralInferenceUiError(str(exc)) from exc
        else:
            mode = "two_stage"
        needed = required_bundle_names(
            config.active_detectors_dyadic,
            config.active_detectors_focal,
            nose_to_body_mode=mode,
        )
        by_name = snapshot.by_name()
        missing: list[str] = []
        for _family, name in needed:
            info = by_name.get(name)
            if info is None or not info.is_loadable:
                missing.append(name)
        if missing:
            raise BehavioralInferenceUiError(
                "Required model bundles are not loadable: " + ", ".join(missing)
            )
        for key in _threshold_keys_for_selection(
            config.active_detectors_dyadic, config.active_detectors_focal
        ):
            resolve_working_threshold(key, config)
        # Keep source referenced for clearer errors.
        _ = source.run_id

    def _is_current_inference_result(
        self, expected: InferenceRequestIdentity | None
    ) -> bool:
        if expected is None:
            return self.state.active_inference_identity is None
        active = self.state.active_inference_identity
        if active is None:
            return False
        if expected.request_id != active.request_id:
            return False
        current_root = self.state.session_root
        if current_root is None:
            return False
        return (
            current_root.resolve(strict=False)
            == expected.session_root.resolve(strict=False)
            and self.state.selected_s6_run_id == expected.s6_run_id
        )


def _s6_option_from_summary(run: CombinedFeaturesRunSummary) -> S6SourceOption:
    sample_count, track_count, session_id, failures = _s6_display_meta(run.run_dir)
    return S6SourceOption(
        run_id=run.run_id,
        run_dir=run.run_dir,
        status=run.status,
        classification=run.classification,
        label=f"{run.run_id}  [{_eligibility_label(run.classification)}]",
        sample_count=sample_count,
        track_count=track_count,
        session_identity=session_id,
        structural_failures=failures,
    )


def _s6_display_meta(
    run_dir: Path,
) -> tuple[int | None, int | None, str | None, tuple[str, ...]]:
    meta = _load_json_mapping(run_dir / "run_meta.json")
    qc = _load_json_mapping(run_dir / "qc.json")
    sample_count: int | None = None
    track_count: int | None = None
    session_id: str | None = None
    failures: list[str] = []
    if isinstance(meta, Mapping):
        session_id = _optional_text(meta.get("session_root"))
        summary = meta.get("summary")
        if isinstance(summary, Mapping):
            sample_count = _optional_int(summary.get("sample_count"))
        adapter = meta.get("adapter")
        if isinstance(adapter, Mapping):
            track_count = _optional_int(adapter.get("n_tracks"))
            if sample_count is None:
                sample_count = _optional_int(adapter.get("n_samples"))
        if meta.get("error"):
            failures.append(str(meta["error"]))
    if isinstance(qc, Mapping):
        metrics = qc.get("metrics") if isinstance(qc.get("metrics"), Mapping) else qc
        if isinstance(metrics, Mapping):
            if sample_count is None:
                sample_count = _optional_int(
                    metrics.get("n_samples") or metrics.get("sample_count")
                )
            if track_count is None:
                track_count = _optional_int(metrics.get("n_tracks"))
        qc_failures = qc.get("failures")
        if isinstance(qc_failures, list):
            failures.extend(str(x) for x in qc_failures)
        if qc.get("structural_ok") is False:
            failures.append("structural_ok=false")
    return sample_count, track_count, session_id, tuple(failures)


def _build_detector_availability(
    *,
    snapshot: Any,
    settings: BehavioralInferenceUiSettings,
) -> tuple[DetectorAvailability, ...]:
    by_name = snapshot.by_name()
    selected_dyadic = set(settings.selected_detectors_dyadic)
    selected_focal = set(settings.selected_detectors_focal)
    rows: list[DetectorAvailability] = []

    # Resolve nose-to-body mode when possible (soft — missing bundles → not ready).
    nose_mode = "two_stage"
    try:
        nose_mode = resolve_nose_to_body_mode(
            snapshot,
            allow_fallback=settings.allow_nose_to_body_single_stage_fallback,
        )
    except Exception:  # noqa: BLE001 — soft readiness probe
        nose_mode = "two_stage"

    try:
        config = settings.to_config()
    except (BehavioralInferenceError, ThresholdConfigError) as exc:
        detail = str(exc)
        for name in LOGICAL_DYADIC_DETECTORS:
            rows.append(
                DetectorAvailability(
                    name=name,
                    family="dyadic",
                    selected=name in selected_dyadic,
                    ready=False,
                    status="config_error",
                    specialist_deps=_specialists_for("dyadic", name, nose_mode),
                    detail=detail,
                )
            )
        for name in LOGICAL_FOCAL_DETECTORS:
            rows.append(
                DetectorAvailability(
                    name=name,
                    family="focal",
                    selected=name in selected_focal,
                    ready=False,
                    status="config_error",
                    specialist_deps=_specialists_for("focal", name, nose_mode),
                    detail=detail,
                )
            )
        return tuple(rows)

    for name in LOGICAL_DYADIC_DETECTORS:
        rows.append(
            _availability_for_logical(
                name=name,
                family="dyadic",
                selected=name in selected_dyadic,
                by_name=by_name,
                nose_mode=nose_mode,
                config=config,
            )
        )
    for name in LOGICAL_FOCAL_DETECTORS:
        rows.append(
            _availability_for_logical(
                name=name,
                family="focal",
                selected=name in selected_focal,
                by_name=by_name,
                nose_mode=nose_mode,
                config=config,
            )
        )

    # Surface obsolete / unsupported bundles that are not logical detectors.
    for bundle in snapshot.bundles:
        if bundle.detector_name in LOGICAL_DYADIC_DETECTORS:
            continue
        if bundle.detector_name in LOGICAL_FOCAL_DETECTORS:
            continue
        if bundle.detector_name in CONTACT_SPECIALISTS:
            continue
        if bundle.detector_name in NOSE_TO_BODY_SPECIALISTS:
            continue
        if bundle.support in ("obsolete", "unsupported"):
            rows.append(
                DetectorAvailability(
                    name=bundle.detector_name,
                    family=bundle.family,
                    selected=False,
                    ready=False,
                    status=bundle.support,
                    specialist_deps=(),
                    detail=(
                        "Identified in registry but not executable as an active "
                        f"logical detector ({bundle.support})."
                    ),
                )
            )
    return tuple(rows)


def _availability_for_logical(
    *,
    name: str,
    family: str,
    selected: bool,
    by_name: Mapping[str, Any],
    nose_mode: str,
    config: BehavioralInferenceConfig,
) -> DetectorAvailability:
    deps = _specialists_for(family, name, nose_mode)
    needed: tuple[str, ...]
    if family == "dyadic" and name == "contact":
        needed = CONTACT_SPECIALISTS
    elif family == "focal" and name == "nose_to_body":
        needed = (
            NOSE_TO_BODY_SPECIALISTS
            if nose_mode == "two_stage"
            else ("nose_to_body",)
        )
    else:
        needed = (name,)

    missing: list[str] = []
    status_bits: list[str] = []
    for bundle_name in needed:
        info = by_name.get(bundle_name)
        if info is None:
            missing.append(bundle_name)
            status_bits.append(f"{bundle_name}=absent")
        elif not info.is_loadable:
            missing.append(bundle_name)
            status_bits.append(f"{bundle_name}={info.support}")
        else:
            status_bits.append(f"{bundle_name}=ready")

    threshold_ok = True
    threshold_detail = ""
    try:
        for key in _threshold_keys_for_selection(
            (name,) if family == "dyadic" else (),
            (name,) if family == "focal" else (),
        ):
            resolve_working_threshold(key, config)
    except ThresholdConfigError as exc:
        threshold_ok = False
        threshold_detail = str(exc)

    ready = not missing and threshold_ok
    if missing:
        status = "missing_bundles"
        detail = "Missing/unloadable: " + ", ".join(missing)
    elif not threshold_ok:
        status = "threshold_unresolved"
        detail = threshold_detail
    else:
        status = "ready"
        detail = "; ".join(status_bits) if status_bits else "ready"
        if name == "nose_to_body":
            detail = f"mode={nose_mode}; {detail}"
    return DetectorAvailability(
        name=name,
        family=family,
        selected=selected,
        ready=ready,
        status=status,
        specialist_deps=deps,
        detail=detail,
    )


def _specialists_for(family: str, name: str, nose_mode: str) -> tuple[str, ...]:
    if family == "dyadic" and name == "contact":
        return CONTACT_SPECIALISTS
    if family == "focal" and name == "nose_to_body":
        if nose_mode == "single_stage_fallback":
            return ("nose_to_body",)
        return NOSE_TO_BODY_SPECIALISTS
    return ()


def _threshold_keys_for_selection(
    dyadic: Sequence[str], focal: Sequence[str]
) -> tuple[str, ...]:
    keys: list[str] = []
    for name in dyadic:
        if name == "contact":
            keys.extend(["contact_intact", "contact_occluded"])
        else:
            keys.append(name)
    for name in focal:
        if name == "nose_to_body":
            keys.extend(["nose_body_intact", "nose_body_occ"])
        else:
            keys.append(name)
    return tuple(keys)


def _eligibility_label(classification: str | None) -> str:
    if classification is None:
        return "—"
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


def _load_json_mapping(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeError):
        return None
    return payload if isinstance(payload, Mapping) else None


def _format_metric(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return ", ".join(str(v) for v in value)
    if isinstance(value, Mapping):
        return "; ".join(f"{k}={v}" for k, v in value.items())
    return str(value)


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _run_dir_for_qc(
    result: BehavioralInferenceRunResult | None,
    run: BehavioralInferenceRunSummary | None,
) -> Path | None:
    if run is not None:
        return run.run_dir
    if result is not None:
        return result.run_dir
    return None
