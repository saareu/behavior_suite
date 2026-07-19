"""Headless view-model/controller for the Subsystem 02 desktop workflow."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import traceback
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

from pose_inference.run_discovery import (
    COMPLETE_REVIEWABLE,
    PoseInferenceProjectSummary,
    PoseInferenceRunSummary,
    summarize_pose_inference_project,
)
from pose_inference.runner import (
    PoseInferenceError,
    PoseInferenceModelSpec,
    PoseInferencePreflightError,
    PoseInferencePreflightFailure,
    PoseInferencePreflightResult,
    PoseInferencePreflightStage,
    PoseInferenceRequest,
    PoseInferenceResult,
    load_inference_profile,
    preflight_pose_inference,
    run_pose_inference,
    validate_s1_handoff,
)
from ui.controllers.run_preprocess_controller import open_folder_in_platform_explorer


class PoseInferenceUiError(ValueError):
    """Expected, actionable Subsystem 02 UI validation error."""


class InferenceMode(StrEnum):
    """Inference modes exposed by the MVP screen."""

    BOTTOMUP = "bottomup"
    TOPDOWN = "topdown"


class S1UiStatus(StrEnum):
    """Lightweight S1 handoff states shown before backend preflight."""

    NOT_SELECTED = "not_selected"
    READY = "ready"
    INCOMPLETE = "incomplete"
    METADATA_UNREADABLE = "metadata_unreadable"
    INVALID = "invalid"


@dataclass(frozen=True, slots=True)
class S3PoseInput:
    """Narrow navigation handoff for a future Subsystem 03 screen."""

    session_root: Path
    selected_run_dir: Path
    pose_slp_path: Path
    pose_parquet_path: Path
    overlay_path: Path
    inference_mode: str | None
    qc_outcome: str | None


@dataclass(frozen=True, slots=True)
class ProfileSummary:
    """Small profile-derived summary; the profile remains authoritative."""

    path: Path
    profile_id: str | None
    inference_mode: str
    device: str | None
    batch_size: int | None
    expected_animals: int | None
    tracking_enabled: bool


@dataclass(slots=True)
class PoseInferenceUiState:
    """Compact mutable state with no pose arrays or video handles."""

    session_root: Path | None = None
    summary: PoseInferenceProjectSummary | None = None
    s1_status: S1UiStatus = S1UiStatus.NOT_SELECTED
    s1_message: str = "Select a project/session."
    prepared_video_path: Path | None = None
    prepared_frame_count: int | None = None
    timing_status: str | None = None
    inference_mode: InferenceMode = InferenceMode.BOTTOMUP
    bottomup_model_path: str = ""
    centroid_model_path: str = ""
    centered_instance_model_path: str = ""
    bottomup_profile_path: str = ""
    topdown_profile_path: str = ""
    sleap_executable_path: str = ""
    run_purpose: str = "development"
    selected_run_id: str | None = None
    task_running: bool = False
    task_generation: int = 0
    active_task_token: int | None = None
    run_stage: str = "Not started"
    run_error: str | None = None
    last_result: PoseInferenceResult | None = None
    active_run_dir: Path | None = None
    unexpected_error_detail: str | None = None
    s3_handoff: S3PoseInput | None = None
    profile_summary: ProfileSummary | None = None
    last_preflight: PoseInferencePreflightResult | None = None
    preflight_report: str | None = None
    bottomup_default_profile_explicit: bool = False
    topdown_default_profile_explicit: bool = False
    unverified_network_paths: set[str] = field(default_factory=set)
    notices: list[str] = field(default_factory=list)


class SettingsStore(Protocol):
    """Small QSettings-compatible protocol used for dependency injection."""

    def value(self, key: str, default_value: object = None) -> object: ...

    def setValue(self, key: str, value: object) -> None: ...


DiscoveryFunction = Callable[[Path | str], PoseInferenceProjectSummary]
RunnerFunction = Callable[[PoseInferenceRequest], PoseInferenceResult]
PreflightFunction = Callable[[PoseInferenceRequest], PoseInferencePreflightResult]
PathOpener = Callable[[Path], bool]
ProgressCallback = Callable[[str, Path | None], None]


def open_file_in_default_application(path: Path) -> bool:
    """Open one existing file with the platform default application."""

    target = Path(path).resolve()
    if not target.is_file():
        return False
    if sys.platform == "win32":
        os.startfile(target)  # type: ignore[attr-defined]
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(target)])
    else:
        subprocess.Popen(["xdg-open", str(target)])
    return True


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _positive_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        return None
    return value


def _is_network_path_text(value: str) -> bool:
    normalized = value.strip().replace("/", "\\")
    return normalized.startswith("\\\\")


def _same_path(first: Path, second: Path) -> bool:
    return first.resolve(strict=False) == second.resolve(strict=False)


class PoseInferenceController:
    """Coordinate discovery, configuration, execution, and S3 input selection."""

    _SETTINGS_PREFIX = "pose_inference/"
    _MAX_DISPLAYED_INTERVALS_PER_WARNING = 10

    def __init__(
        self,
        state: PoseInferenceUiState | None = None,
        *,
        discovery: DiscoveryFunction = summarize_pose_inference_project,
        runner: RunnerFunction = run_pose_inference,
        preflight: PreflightFunction = preflight_pose_inference,
        settings: SettingsStore | None = None,
        folder_opener: PathOpener = open_folder_in_platform_explorer,
        file_opener: PathOpener = open_file_in_default_application,
    ) -> None:
        self.state = state or PoseInferenceUiState()
        self._discovery = discovery
        self._runner = runner
        self._preflight = preflight
        self._settings = settings
        self._folder_opener = folder_opener
        self._file_opener = file_opener
        self.state.bottomup_profile_path = str(self.default_profile_path(InferenceMode.BOTTOMUP))
        self.state.topdown_profile_path = str(self.default_profile_path(InferenceMode.TOPDOWN))
        self.load_persisted_paths()
        self.refresh_profile_summary()

    def _ensure_idle(self, action: str) -> None:
        if self.state.task_running:
            raise PoseInferenceUiError(
                f"Cannot {action} while a Subsystem 2 task is active."
            )

    @staticmethod
    def default_profile_path(mode: InferenceMode | str) -> Path:
        """Return the repository profile matching one explicit inference mode."""

        normalized = InferenceMode(mode)
        filename = (
            "sleapnn_topdown_default_profile.yaml"
            if normalized is InferenceMode.TOPDOWN
            else "sleapnn_default_profile.yaml"
        )
        return Path(__file__).resolve().parents[3] / "configs" / "subsystem_02" / filename

    @property
    def runs(self) -> tuple[PoseInferenceRunSummary, ...]:
        """Return discovery-ordered runs without loading pose artifacts."""

        return self.state.summary.runs if self.state.summary is not None else ()

    @property
    def selected_run(self) -> PoseInferenceRunSummary | None:
        """Return the selected discovered run, if it still exists."""

        selected_id = self.state.selected_run_id
        return next((run for run in self.runs if run.run_id == selected_id), None)

    @property
    def can_submit(self) -> bool:
        """Return whether basic UI state permits authoritative backend preflight."""

        if self.state.s1_status is not S1UiStatus.READY or self.state.task_running:
            return False
        try:
            self.build_request()
        except PoseInferenceUiError:
            return False
        return True

    def set_session(self, session_root: Path | str) -> PoseInferenceProjectSummary | None:
        """Select and discover one session, preserving it across subsystem navigation."""

        self._ensure_idle("change the selected session")
        raw = str(session_root).strip()
        if not raw:
            self._clear_session("Select a project/session.")
            return None
        root = Path(raw).expanduser().resolve(strict=False)
        self.state.session_root = root
        self.state.selected_run_id = None
        self.state.s3_handoff = None
        return self.refresh_discovery()

    def refresh_discovery(self) -> PoseInferenceProjectSummary | None:
        """Refresh only compact S1/S2 metadata and retain a valid run selection."""

        self._ensure_idle("refresh run discovery")
        root = self.state.session_root
        if root is None:
            self._clear_session("Select a project/session.")
            return None
        if not root.is_dir():
            self.state.summary = None
            self.state.s1_status = S1UiStatus.INVALID
            self.state.s1_message = f"Session directory does not exist or is unavailable: {root}"
            self.state.prepared_video_path = root / "preprocess" / "prepared_video.mp4"
            return None
        previous_selection = self.state.selected_run_id
        try:
            summary = self._discovery(root)
        except (OSError, ValueError) as exc:
            self.state.summary = None
            self.state.s1_status = S1UiStatus.INVALID
            self.state.s1_message = f"Could not inspect the session: {exc}"
            return None
        self.state.summary = summary
        self.state.prepared_video_path = summary.s1_handoff.prepared_video
        if not summary.s1_handoff.is_complete:
            missing = ", ".join(summary.s1_handoff.missing_files)
            self.state.s1_status = S1UiStatus.INCOMPLETE
            self.state.s1_message = f"S1 handoff is incomplete. Missing: {missing}."
            self.state.prepared_frame_count = None
            self.state.timing_status = None
        else:
            self._inspect_s1_metadata(root)
        run_ids = {run.run_id for run in summary.runs}
        self.state.selected_run_id = (
            previous_selection if previous_selection in run_ids else None
        )
        return summary

    def _inspect_s1_metadata(self, root: Path) -> None:
        try:
            handoff = validate_s1_handoff(root)
            metadata = handoff.prepare_meta_payload
            prepared_video = _mapping(metadata.get("prepared_video"))
            sync = _mapping(metadata.get("sync"))
            frame_count = _positive_int(
                prepared_video.get("frame_count_used_for_sleap")
            ) or _positive_int(prepared_video.get("opencv_readable_frame_count"))
            if frame_count is None:
                frame_count = _positive_int(sync.get("frame_count_used_for_sleap"))
            external = _mapping(metadata.get("external_time"))
            if external.get("provided") is True:
                timing = str(external.get("validation_status") or "provided")
                source = external.get("source_path")
                self.state.timing_status = f"external timing: {timing}" + (
                    f" ({source})" if source else ""
                )
            else:
                encoding = _mapping(metadata.get("encoding"))
                source = encoding.get("fps_header_source")
                self.state.timing_status = (
                    f"video/FPS timing ({source})" if source else "video/FPS timing"
                )
            self.state.prepared_frame_count = frame_count
            self.state.s1_status = S1UiStatus.READY
            self.state.s1_message = (
                "S1 handoff metadata is readable. Full sync/video/runtime checks run "
                "during backend preflight."
            )
        except PoseInferenceError as exc:
            self.state.prepared_frame_count = None
            self.state.timing_status = None
            self.state.s1_status = S1UiStatus.METADATA_UNREADABLE
            self.state.s1_message = f"S1 metadata is unreadable or invalid: {exc}"

    def _clear_session(self, message: str) -> None:
        self.state.session_root = None
        self.state.summary = None
        self.state.s1_status = S1UiStatus.NOT_SELECTED
        self.state.s1_message = message
        self.state.prepared_video_path = None
        self.state.prepared_frame_count = None
        self.state.timing_status = None
        self.state.selected_run_id = None

    def set_mode(self, mode: InferenceMode | str) -> None:
        """Switch modes and refresh the profile-driven shared setting summary."""

        self._ensure_idle("change inference mode")
        try:
            self.state.inference_mode = InferenceMode(mode)
        except ValueError as exc:
            raise PoseInferenceUiError(f"Unsupported inference mode: {mode!s}") from exc
        self.state.run_error = None
        self.refresh_profile_summary()

    def active_profile_path(self) -> str:
        """Return the editable profile path for the active mode."""

        return (
            self.state.topdown_profile_path
            if self.state.inference_mode is InferenceMode.TOPDOWN
            else self.state.bottomup_profile_path
        )

    def set_active_profile_path(
        self,
        path: str,
        *,
        explicitly_selected_default: bool = False,
    ) -> None:
        """Store the profile path separately per inference mode."""

        self._ensure_idle("change the inference profile")
        cleaned = path.strip()
        default = str(self.default_profile_path(self.state.inference_mode))
        if self.state.inference_mode is InferenceMode.TOPDOWN:
            self.state.topdown_profile_path = cleaned
            if explicitly_selected_default:
                self.state.topdown_default_profile_explicit = True
            elif cleaned != default:
                self.state.topdown_default_profile_explicit = False
        else:
            self.state.bottomup_profile_path = cleaned
            if explicitly_selected_default:
                self.state.bottomup_default_profile_explicit = True
            elif cleaned != default:
                self.state.bottomup_default_profile_explicit = False
        self.refresh_profile_summary()

    def refresh_profile_summary(self) -> ProfileSummary | None:
        """Parse the active profile for display; backend validation remains authoritative."""

        raw = self.active_profile_path().strip()
        if not raw:
            self.state.profile_summary = None
            return None
        if _is_network_path_text(raw) and raw in self.state.unverified_network_paths:
            self.state.profile_summary = None
            return None
        try:
            path = Path(raw).expanduser().resolve()
            payload = load_inference_profile(path)
            configured_mode = str(payload.get("inference_mode", "bottomup"))
            tracking = _mapping(payload.get("tracking"))
            summary = ProfileSummary(
                path=path,
                profile_id=(str(payload["profile_id"]) if payload.get("profile_id") else None),
                inference_mode=configured_mode,
                device=(str(payload["device"]) if payload.get("device") is not None else None),
                batch_size=_positive_int(payload.get("batch_size")),
                expected_animals=_positive_int(payload.get("expected_animals")),
                tracking_enabled=bool(tracking.get("enabled")),
            )
        except (PoseInferenceError, OSError, ValueError):
            self.state.profile_summary = None
            return None
        self.state.profile_summary = summary
        return summary

    def validate_configuration(self) -> None:
        """Reject only basic missing or ambiguous UI combinations before preflight."""

        if self.state.session_root is None:
            raise PoseInferenceUiError("Select a project/session before running inference.")
        if self.state.s1_status is not S1UiStatus.READY:
            raise PoseInferenceUiError(self.state.s1_message)
        profile_path = self.active_profile_path().strip()
        if not profile_path:
            raise PoseInferenceUiError("Select an inference profile YAML.")
        if self.state.inference_mode is InferenceMode.BOTTOMUP:
            if not self.state.bottomup_model_path.strip():
                raise PoseInferenceUiError("Select a bottom-up model path.")
        else:
            if not self.state.centroid_model_path.strip():
                raise PoseInferenceUiError("Select a top-down centroid model path.")
            if not self.state.centered_instance_model_path.strip():
                raise PoseInferenceUiError(
                    "Select a top-down centered-instance model path."
                )

    def build_request(self) -> PoseInferenceRequest:
        """Build the exact typed request passed to the authoritative backend."""

        self.validate_configuration()
        assert self.state.session_root is not None
        mode = self.state.inference_mode
        model_spec = PoseInferenceModelSpec(
            inference_mode=mode.value,
            bottomup_model_path=(
                Path(self.state.bottomup_model_path).expanduser()
                if mode is InferenceMode.BOTTOMUP
                else None
            ),
            centroid_model_path=(
                Path(self.state.centroid_model_path).expanduser()
                if mode is InferenceMode.TOPDOWN
                else None
            ),
            centered_instance_model_path=(
                Path(self.state.centered_instance_model_path).expanduser()
                if mode is InferenceMode.TOPDOWN
                else None
            ),
        )
        executable = self.state.sleap_executable_path.strip()
        return PoseInferenceRequest(
            session_root=self.state.session_root,
            model_path=None,
            profile_path=Path(self.active_profile_path()).expanduser(),
            run_purpose=self.state.run_purpose.strip() or "development",
            model_spec=model_spec,
            sleap_executable=Path(executable).expanduser() if executable else None,
        )

    def begin_run(self) -> PoseInferenceRequest:
        """Validate UI state and mark one asynchronous run as active."""

        self._ensure_idle("start another pose inference run")
        request = self.build_request()
        self.persist_paths()
        self._begin_task("Validating backend preflight")
        return request

    def begin_preflight(self) -> PoseInferenceRequest:
        """Mark an authoritative validation-only task active and return its request."""

        self._ensure_idle("validate another configuration")
        request = self.build_request()
        self.persist_paths()
        self._begin_task("Validating configuration")
        return request

    def _begin_task(self, stage: str) -> int:
        if self.state.task_running:
            raise PoseInferenceUiError("A Subsystem 2 task is already running.")
        self.state.task_generation += 1
        self.state.active_task_token = self.state.task_generation
        self.state.task_running = True
        self.state.run_stage = stage
        self.state.run_error = None
        self.state.last_result = None
        self.state.last_preflight = None
        self.state.preflight_report = None
        self.state.active_run_dir = None
        self.state.unexpected_error_detail = None
        return self.state.task_generation

    @property
    def active_task_token(self) -> int:
        token = self.state.active_task_token
        if token is None:
            raise PoseInferenceUiError("No Subsystem 2 task is active.")
        return token

    def execute_request(
        self,
        request: PoseInferenceRequest,
        progress_callback: ProgressCallback | None = None,
    ) -> PoseInferenceResult:
        """Execute one request in a worker; this method does not mutate UI state."""

        if self._runner is run_pose_inference:
            return run_pose_inference(request, progress_callback=progress_callback)
        if progress_callback is not None:
            progress_callback("running inference", None)
        return self._runner(request)

    def execute_preflight(
        self,
        request: PoseInferenceRequest,
    ) -> PoseInferencePreflightResult:
        """Execute authoritative read-only backend preflight in a worker."""

        return self._preflight(request)

    def _accepts_task_token(self, token: int | None) -> bool:
        return token is None or token == self.state.active_task_token

    def apply_progress(
        self,
        stage: str,
        run_dir: Path | None,
        *,
        token: int | None = None,
    ) -> bool:
        """Apply one backend stage update after Qt queues it to the GUI thread."""

        if not self._accepts_task_token(token):
            return False
        self.state.run_stage = stage
        if run_dir is not None:
            self.state.active_run_dir = Path(run_dir)
        return True

    def apply_result(
        self,
        result: PoseInferenceResult,
        *,
        token: int | None = None,
    ) -> bool:
        """Store a worker result, refresh discovery, and select its run directory."""

        if not self._accepts_task_token(token):
            return False
        if not isinstance(result, PoseInferenceResult):
            raise TypeError("Pose inference worker returned an unexpected result type.")
        had_active_task = self.state.task_running
        self.state.last_result = result
        self.state.active_run_dir = result.run_dir
        self.state.run_stage = "Complete" if result.success else "Failed"
        self.state.run_error = None if result.success else (
            f"Pose inference failed with status '{result.status}'. See processing_log.txt."
        )
        if not had_active_task:
            self.refresh_discovery()
            if any(run.run_id == result.run_id for run in self.runs):
                self.state.selected_run_id = result.run_id
        return True

    def apply_preflight_result(
        self,
        result: PoseInferencePreflightResult,
        *,
        token: int | None = None,
    ) -> bool:
        """Store a matching validation-only result without discovering or writing runs."""

        if not self._accepts_task_token(token):
            return False
        if not isinstance(result, PoseInferencePreflightResult):
            raise TypeError("Pose preflight worker returned an unexpected result type.")
        self.state.last_preflight = result
        self.state.run_stage = "Configuration valid"
        self.state.run_error = None
        self.state.preflight_report = self._format_preflight_report(result)
        return True

    def apply_error(self, exc: BaseException, *, token: int | None = None) -> bool:
        """Convert backend/preflight failures into readable state on the GUI thread."""

        if not self._accepts_task_token(token):
            return False
        had_active_task = self.state.task_running
        self.state.run_stage = "Failed"
        if isinstance(exc, (PoseInferenceError, OSError, ValueError)):
            self.state.run_error = str(exc)
        else:
            self.state.run_error = "An unexpected error occurred while running pose inference."
            self.state.unexpected_error_detail = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
        if isinstance(exc, PoseInferencePreflightError):
            self.state.preflight_report = self._format_preflight_failure(exc.report)
        if not had_active_task:
            self.refresh_discovery()
        return True

    def finish_run(self, *, token: int | None = None) -> bool:
        """Clear active state after a worker thread exits."""

        if not self._accepts_task_token(token):
            return False
        self.state.task_running = False
        self.state.active_task_token = None
        result = self.state.last_result
        self.refresh_discovery()
        if result is not None and any(run.run_id == result.run_id for run in self.runs):
            self.state.selected_run_id = result.run_id
        return True

    def abort_task_submission(self, token: int, message: str) -> None:
        """Roll back only a just-created task that could not be submitted."""

        if self._accepts_task_token(token):
            self.state.task_running = False
            self.state.active_task_token = None
            self.state.run_stage = "Not started"
            self.state.run_error = message

    def select_run(self, run_id: str | None) -> PoseInferenceRunSummary | None:
        """Select a discovered run without opening pose or parquet artifacts."""

        self._ensure_idle("change the selected run")
        if run_id is None:
            self.state.selected_run_id = None
            return None
        selected = next((run for run in self.runs if run.run_id == run_id), None)
        if selected is None:
            raise PoseInferenceUiError(f"Run is no longer available: {run_id}")
        self.state.selected_run_id = run_id
        return selected

    def selected_run_can_feed_s3(self) -> bool:
        """Allow technical pass and review-recommended runs, never incomplete runs."""

        run = self.selected_run
        return bool(
            run is not None
            and run.classification == COMPLETE_REVIEWABLE
            and run.pose_qc_outcome in {"pass", "review_recommended"}
            and all(
                run.artifact_presence.get(key, False)
                for key in ("pose_slp", "pose_parquet", "overlay_mp4")
            )
        )

    @staticmethod
    def _validate_readable_file(path: Path, label: str) -> None:
        if not path.is_file():
            raise PoseInferenceUiError(
                f"The selected run is stale or incomplete: {label} is missing."
            )
        try:
            with path.open("rb") as stream:
                stream.read(1)
        except OSError as exc:
            raise PoseInferenceUiError(
                f"The selected run cannot continue to S3 because {label} is unreadable: {exc}"
            ) from exc

    def _validate_s3_artifacts(self, run: PoseInferenceRunSummary) -> None:
        run_dir = run.run_dir
        if not run_dir.is_dir():
            raise PoseInferenceUiError(
                "The selected run directory was deleted or is unavailable."
            )
        try:
            next(run_dir.iterdir(), None)
        except OSError as exc:
            raise PoseInferenceUiError(
                f"The selected run directory is unreadable: {exc}"
            ) from exc
        for filename in ("pose.slp", "pose.parquet", "overlay.mp4"):
            self._validate_readable_file(run_dir / filename, filename)

    @staticmethod
    def _local_path_unavailable(path_text: str, *, require_file: bool) -> bool:
        if _is_network_path_text(path_text):
            return False
        path = Path(path_text).expanduser()
        return not (path.is_file() if require_file else path.exists())

    def rerun_unavailable_reason(
        self,
        run: PoseInferenceRunSummary | None = None,
    ) -> str | None:
        """Explain why authoritative metadata cannot populate a safe rerun."""

        if self.state.task_running:
            return "Rerun settings are unavailable while a Subsystem 2 task is active."
        selected = run or self.selected_run
        if selected is None:
            return "Select a run to copy its settings."
        try:
            mode = InferenceMode(selected.inference_mode)
        except (TypeError, ValueError):
            return (
                "Run metadata has an unsupported or missing inference mode: "
                f"{selected.inference_mode!r}."
            )
        if mode is InferenceMode.BOTTOMUP:
            model_paths = (selected.bottomup_model_path or selected.model_path,)
        else:
            model_paths = (
                selected.centroid_model_path,
                selected.centered_instance_model_path,
            )
        if any(not path for path in model_paths):
            needed = (
                "bottom-up model path"
                if mode is InferenceMode.BOTTOMUP
                else "centroid and centered-instance model paths"
            )
            return f"Run metadata does not contain the authoritative {needed}."
        for model_path in model_paths:
            assert model_path is not None
            if self._local_path_unavailable(model_path, require_file=False):
                return f"A copied model path is no longer available: {model_path}"

        profile_path = selected.profile_path
        explicit_default = (
            self.state.topdown_default_profile_explicit
            if mode is InferenceMode.TOPDOWN
            else self.state.bottomup_default_profile_explicit
        )
        if not profile_path and not explicit_default:
            return (
                "Run metadata does not contain a profile path. Explicitly choose the "
                "current mode default before copying settings."
            )
        effective_profile = profile_path or str(self.default_profile_path(mode))
        if self._local_path_unavailable(effective_profile, require_file=True):
            return f"The copied inference profile is no longer available: {effective_profile}"
        if not _is_network_path_text(effective_profile):
            try:
                profile = load_inference_profile(Path(effective_profile))
                configured_mode = str(
                    profile.get("inference_mode", "bottomup")
                ).strip().lower().replace("-", "")
            except (PoseInferenceError, OSError, ValueError) as exc:
                return f"The copied inference profile is invalid: {exc}"
            if configured_mode != mode.value:
                return (
                    f"The copied profile mode '{configured_mode}' does not match "
                    f"the run mode '{mode.value}'."
                )
        return None

    def normalized_flagged_intervals(
        self,
        run: PoseInferenceRunSummary | None = None,
    ) -> tuple[dict[str, list[dict[str, Any]]], tuple[str, ...]]:
        """Return display-safe interval metadata plus ignored-entry diagnostics."""

        selected = run or self.selected_run
        if selected is None:
            return {}, ()
        raw = selected.flagged_intervals
        if raw is None:
            return {}, ()
        if not isinstance(raw, Mapping):
            return {}, ("flagged_intervals is not a mapping and was ignored",)
        normalized: dict[str, list[dict[str, Any]]] = {}
        diagnostics: list[str] = []
        for warning_type, entries in raw.items():
            warning = str(warning_type)
            if not isinstance(entries, list | tuple):
                diagnostics.append(
                    f"flagged_intervals.{warning} is not a list/tuple and was ignored"
                )
                continue
            valid: list[dict[str, Any]] = []
            for index, entry in enumerate(entries):
                if not isinstance(entry, Mapping):
                    diagnostics.append(
                        f"flagged_intervals.{warning}[{index}] is not a mapping and was ignored"
                    )
                    continue
                valid.append(dict(entry))
                if len(valid) == self._MAX_DISPLAYED_INTERVALS_PER_WARNING:
                    if len(entries) > index + 1:
                        diagnostics.append(
                            f"flagged_intervals.{warning} display was capped at "
                            f"{self._MAX_DISPLAYED_INTERVALS_PER_WARNING} entries"
                        )
                    break
            normalized[warning] = valid
        return normalized, tuple(diagnostics)

    def flagged_interval_display_lines(
        self,
        run: PoseInferenceRunSummary | None = None,
    ) -> tuple[str, ...]:
        intervals, _diagnostics = self.normalized_flagged_intervals(run)
        lines: list[str] = []
        for warning, entries in intervals.items():
            for entry in entries:
                video_index = entry.get("video_index", 0)
                lines.append(
                    f"{warning}: video_index={video_index}; inclusive frames "
                    f"{entry.get('start_frame', '?')}–{entry.get('end_frame', '?')}; "
                    f"frame_count={entry.get('frame_count', '?')}; timestamp span "
                    f"(time_span_sec)={entry.get('time_span_sec', '?')}"
                )
        return tuple(lines)

    @staticmethod
    def _format_preflight_report(result: PoseInferencePreflightResult) -> str:
        command = subprocess.list2cmdline(list(result.command))
        stages = PoseInferenceController._format_preflight_stages(result.stages)
        return (
            "Configuration validation passed.\n"
            f"\nValidation stages:\n{stages}\n\n"
            f"S1 handoff: valid ({result.s1_handoff.prepared_video})\n"
            f"Model roles/profile mode: valid ({result.inference_mode}; "
            f"profile={result.profile_id or result.profile_path.name})\n"
            f"SLEAP-NN executable: {result.sleap_runtime.executable_path}\n"
            f"SLEAP-NN version: {result.sleap_runtime.version}\n"
            f"Resolved command (not executed): {command}"
        )

    @staticmethod
    def _format_preflight_stages(
        stages: tuple[PoseInferencePreflightStage, ...],
    ) -> str:
        if not stages:
            return "- Stage details were not provided by the preflight implementation."
        return "\n".join(
            f"- {stage.name}: {stage.status}"
            + (f" — {stage.detail}" if stage.detail else "")
            for stage in stages
        )

    @staticmethod
    def _format_preflight_failure(report: PoseInferencePreflightFailure) -> str:
        if report.found_sleap_version is not None:
            heading = (
                "Configuration inputs validated up to runtime check.\n"
                "Local SLEAP-NN runtime is incompatible:\n"
                f"found {report.found_sleap_version}; Behavior Suite currently "
                "requires 0.3.x."
            )
        else:
            heading = "Configuration validation failed."
        executable = (
            f"\nResolved SLEAP executable: {report.resolved_executable_path}"
            if report.resolved_executable_path is not None
            else ""
        )
        stages = PoseInferenceController._format_preflight_stages(report.stages)
        return (
            f"{heading}{executable}\n\nValidation stages:\n{stages}\n\n"
            f"Full actionable error: {report.error}"
        )

    def build_s3_handoff(self) -> S3PoseInput:
        """Build a transient S3 navigation input without scientific approval semantics."""

        self._ensure_idle("continue to Subsystem 3")
        cached_run = self.selected_run
        root = self.state.session_root
        if root is None or cached_run is None:
            raise PoseInferenceUiError(
                "Select a technically complete S2 run before continuing to S3."
            )
        selected_id = cached_run.run_id
        selected_dir = cached_run.run_dir.resolve(strict=False)
        self.refresh_discovery()
        run = next(
            (
                candidate
                for candidate in self.runs
                if candidate.run_id == selected_id
                and candidate.run_dir.resolve(strict=False) == selected_dir
            ),
            None,
        )
        if run is None:
            raise PoseInferenceUiError(
                "The selected run was deleted or changed. Refresh and select an available run."
            )
        self.state.selected_run_id = run.run_id
        if run.classification != COMPLETE_REVIEWABLE:
            raise PoseInferenceUiError(
                "The selected run is not technically complete and reviewable."
            )
        if run.pose_qc_outcome not in {"pass", "review_recommended"}:
            raise PoseInferenceUiError(
                "The selected run has failed, missing, or contradictory pose QC metadata."
            )
        self._validate_s3_artifacts(run)
        handoff = S3PoseInput(
            session_root=root,
            selected_run_dir=run.run_dir,
            pose_slp_path=run.run_dir / "pose.slp",
            pose_parquet_path=run.run_dir / "pose.parquet",
            overlay_path=run.run_dir / "overlay.mp4",
            inference_mode=run.inference_mode,
            qc_outcome=run.pose_qc_outcome,
        )
        self.state.s3_handoff = handoff
        return handoff

    def copy_selected_run_settings(self) -> None:
        """Copy discoverable paths only; backend preflight validates their current roles."""

        self._ensure_idle("copy settings for a rerun")
        run = self.selected_run
        if run is None:
            raise PoseInferenceUiError("Select a run to copy its settings.")
        reason = self.rerun_unavailable_reason(run)
        if reason is not None:
            raise PoseInferenceUiError(reason)
        try:
            mode = InferenceMode(run.inference_mode)
        except (TypeError, ValueError) as exc:
            raise PoseInferenceUiError(
                f"Run metadata has an unsupported inference mode: {run.inference_mode!r}."
            ) from exc
        self.state.bottomup_model_path = ""
        self.state.centroid_model_path = ""
        self.state.centered_instance_model_path = ""
        self.state.sleap_executable_path = run.sleap_executable_path or ""
        if mode is InferenceMode.TOPDOWN:
            self.state.centroid_model_path = run.centroid_model_path or ""
            self.state.centered_instance_model_path = run.centered_instance_model_path or ""
        else:
            self.state.bottomup_model_path = run.bottomup_model_path or run.model_path or ""
        self.state.inference_mode = mode
        if run.profile_path:
            if mode is InferenceMode.TOPDOWN:
                self.state.topdown_profile_path = run.profile_path
            else:
                self.state.bottomup_profile_path = run.profile_path
        else:
            default = str(self.default_profile_path(mode))
            if mode is InferenceMode.TOPDOWN:
                self.state.topdown_profile_path = default
            else:
                self.state.bottomup_profile_path = default
        self.state.run_purpose = f"rerun-{run.run_id}"
        self.refresh_profile_summary()

    def open_selected_run_folder(self) -> Path:
        """Open the selected run directory with an injectable platform action."""

        self._ensure_idle("open a run folder")
        run = self.selected_run
        if run is None or not run.run_dir.is_dir():
            raise PoseInferenceUiError("The selected run folder is unavailable.")
        if self._folder_opener(run.run_dir) is False:
            raise PoseInferenceUiError("Could not open the selected run folder.")
        return run.run_dir

    def open_selected_overlay(self) -> Path:
        """Open the selected overlay only when discovery found the artifact."""

        self._ensure_idle("open an overlay")
        run = self.selected_run
        overlay = run.run_dir / "overlay.mp4" if run is not None else None
        if (
            run is None
            or not run.artifact_presence.get("overlay_mp4", False)
            or overlay is None
            or not overlay.is_file()
        ):
            raise PoseInferenceUiError("The selected run has no available overlay.mp4.")
        if self._file_opener(overlay) is False:
            raise PoseInferenceUiError("Could not open the selected overlay.")
        return overlay

    def technical_details(self, run: PoseInferenceRunSummary | None = None) -> str:
        """Render compact, copyable discovery metadata for technical review."""

        selected = run or self.selected_run
        if selected is None:
            return "Select a run to inspect technical details."
        models = {
            "model_id": selected.model_id,
            "model_path": selected.model_path,
            "centroid": {
                "id": selected.centroid_model_id,
                "path": selected.centroid_model_path,
            },
            "centered_instance": {
                "id": selected.centered_instance_model_id,
                "path": selected.centered_instance_model_path,
            },
        }
        intervals, interval_diagnostics = self.normalized_flagged_intervals(selected)
        payload = {
            "run_id": selected.run_id,
            "status": selected.status,
            "classification": selected.classification,
            "dry_run": selected.dry_run,
            "inference_mode": selected.inference_mode,
            "models": models,
            "profile": {"id": selected.profile_id, "path": selected.profile_path},
            "sleap_runtime": {
                "executable_path": selected.sleap_executable_path,
                "version": selected.sleap_executable_version,
            },
            "command": list(selected.command),
            "parquet_timing": selected.parquet_timing,
            "qc": {
                "outcome": selected.pose_qc_outcome,
                "thresholds": selected.qc_thresholds,
                "diagnostic_findings": list(selected.diagnostic_findings),
                "review_recommendation_reasons": list(
                    selected.review_recommendation_reasons
                ),
                "flagged_intervals": intervals,
                "flagged_interval_labels": list(
                    self.flagged_interval_display_lines(selected)
                ),
            },
            "artifact_paths": {
                key: str(selected.run_dir / filename)
                for key, filename in (
                    ("pose_slp", "pose.slp"),
                    ("pose_parquet", "pose.parquet"),
                    ("overlay_mp4", "overlay.mp4"),
                    ("pose_meta", "pose_meta.json"),
                    ("settings_used", "settings_used.yaml"),
                    ("job_manifest", "job_manifest.yaml"),
                    ("processing_log", "processing_log.txt"),
                )
            },
            "metadata_errors": [
                *selected.metadata_errors,
                *interval_diagnostics,
            ],
        }
        return json.dumps(payload, indent=2, ensure_ascii=False, default=str)

    def configuration_summary(self) -> str:
        """Return a concise submission preview without building a command."""

        mode = self.state.inference_mode
        models = (
            f"model={self.state.bottomup_model_path or 'not selected'}"
            if mode is InferenceMode.BOTTOMUP
            else (
                f"centroid={self.state.centroid_model_path or 'not selected'}; "
                "centered-instance="
                f"{self.state.centered_instance_model_path or 'not selected'}"
            )
        )
        return (
            f"Mode: {mode.value}; {models}; profile={self.active_profile_path() or 'not selected'}; "
            f"purpose={self.state.run_purpose.strip() or 'development'}"
        )

    def load_persisted_paths(self) -> None:
        """Load saved text without synchronously probing disconnected network paths."""

        if self._settings is None:
            return
        path_fields = (
            ("last_bottomup_model", "bottomup_model_path", False),
            ("last_centroid_model", "centroid_model_path", False),
            ("last_centered_instance_model", "centered_instance_model_path", False),
            ("last_bottomup_profile", "bottomup_profile_path", True),
            ("last_topdown_profile", "topdown_profile_path", True),
            ("last_sleap_executable", "sleap_executable_path", True),
        )
        for key, attribute, require_file in path_fields:
            value = self._settings.value(self._SETTINGS_PREFIX + key, "")
            raw = str(value).strip() if value is not None else ""
            if not raw:
                continue
            if _is_network_path_text(raw):
                setattr(self.state, attribute, raw)
                self.state.unverified_network_paths.add(raw)
                self.state.notices.append(
                    f"Saved network path is loaded but unverified: {raw}"
                )
                continue
            path = Path(raw).expanduser()
            valid = path.is_file() if require_file else path.exists()
            resolved = str(path.resolve(strict=False))
            setattr(self.state, attribute, resolved)
            if valid:
                continue
            self.state.notices.append(f"Saved path is currently unavailable: {resolved}")

    def persist_paths(self) -> None:
        """Persist useful convenience paths, never run completion or S3 decisions."""

        if self._settings is None:
            return
        bottomup_profile = self.state.bottomup_profile_path
        topdown_profile = self.state.topdown_profile_path
        if Path(bottomup_profile).resolve(strict=False) == self.default_profile_path(
            InferenceMode.BOTTOMUP
        ).resolve(strict=False):
            bottomup_profile = ""
        if Path(topdown_profile).resolve(strict=False) == self.default_profile_path(
            InferenceMode.TOPDOWN
        ).resolve(strict=False):
            topdown_profile = ""
        values = {
            "last_bottomup_model": self.state.bottomup_model_path,
            "last_centroid_model": self.state.centroid_model_path,
            "last_centered_instance_model": self.state.centered_instance_model_path,
            "last_bottomup_profile": bottomup_profile,
            "last_topdown_profile": topdown_profile,
            "last_sleap_executable": self.state.sleap_executable_path,
        }
        for key, value in values.items():
            self._settings.setValue(self._SETTINGS_PREFIX + key, value.strip())
