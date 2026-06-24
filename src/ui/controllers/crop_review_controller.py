"""Headless coordination for crop review and explicit acceptance."""

from __future__ import annotations

import math
import traceback
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from pydantic import ValidationError

from preprocess.cage_detection import CageDetectionResult, detect_cage_crop_plan
from preprocess.config import (
    CageDetectConfig,
    CanonicalResolutionConfig,
    PrepareConfig,
    PreprocessConfig,
)
from preprocess.crop_plan import CropMode, CropPlan
from preprocess.exceptions import (
    CageDetectionCancelledError,
    CropPlanError,
    PreprocessError,
    VideoProbeError,
)
from preprocess.manual_crop import make_manual_crop_plan
from preprocess.models import OperationProgress
from preprocess.pre_crop import ResolvedPreCrop
from ui.state import CropReviewMode, PreprocessSetupState

FrameReader = Callable[[Path, int], np.ndarray]
PreviewBuilder = Callable[[np.ndarray, CropPlan, str], np.ndarray]
AutomaticDetector = Callable[..., CageDetectionResult]
ManualPlanBuilder = Callable[..., CropPlan]

_POINT_LABELS = ("top-left", "top-right", "bottom-right", "bottom-left")
_PREVIEW_MASK_COORDINATE_SHIFT = 8


class CropReviewValidationError(ValueError):
    """Expected user-facing crop-review failure."""


@dataclass(frozen=True, slots=True)
class AutomaticDetectionRequest:
    """Immutable inputs captured before automatic detection enters a worker."""

    raw_video_path: Path
    config: PreprocessConfig
    pre_crop: ResolvedPreCrop
    representative_frame: np.ndarray


@dataclass(frozen=True, slots=True)
class CropReviewComputation:
    """Worker result applied to GUI state only after returning to the GUI thread."""

    crop_plan: CropPlan
    prepared_preview: np.ndarray
    detector_diagnostics: dict[str, object] | None


def read_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    """Decode one requested raw frame without decoding the complete video."""

    path = Path(video_path).expanduser()
    if not path.is_file():
        raise VideoProbeError(f"Raw video does not exist: {path}")
    if isinstance(frame_index, bool) or not isinstance(frame_index, int) or frame_index < 0:
        raise VideoProbeError("Representative frame index must be a non-negative integer.")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoProbeError(f"OpenCV could not open raw video: {path}")
    try:
        if frame_index and not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            raise VideoProbeError(f"Could not seek to raw frame {frame_index}.")
        success, frame = capture.read()
    finally:
        capture.release()
    if not success or frame is None or frame.ndim not in {2, 3} or frame.size == 0:
        raise VideoProbeError(f"Raw frame {frame_index} is unreadable.")
    return np.array(frame, copy=True)


def build_crop_preview(
    raw_frame: np.ndarray,
    crop_plan: CropPlan,
    perspective_interpolation: str,
) -> np.ndarray:
    """Transform and clip one raw frame using validated CropPlan geometry.

    A direct homography can map prepared padding pixels back to valid raw pixels
    outside the accepted quadrilateral. Stage A instead inserts black
    rectification/canonical padding. Clipping to the transformed crop footprint
    preserves the CropPlan transform while matching that visible-content rule.
    """

    frame = np.asarray(raw_frame)
    if frame.ndim not in {2, 3} or frame.size == 0:
        raise CropReviewValidationError("Preview source frame is empty or invalid.")
    interpolation = {
        "linear": cv2.INTER_LINEAR,
        "cubic": cv2.INTER_CUBIC,
    }.get(perspective_interpolation)
    if interpolation is None:
        raise CropReviewValidationError(
            f"Unsupported perspective interpolation: {perspective_interpolation}"
        )
    try:
        preview = cv2.warpPerspective(
            frame,
            crop_plan.H_raw_to_prepared_3x3,
            crop_plan.prepared_size_wh,
            flags=interpolation,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    except cv2.error as exc:
        raise CropReviewValidationError(
            f"Could not generate prepared crop preview: {exc}"
        ) from exc
    width, height = crop_plan.prepared_size_wh
    if preview.shape[:2] != (height, width):
        raise CropReviewValidationError(
            "Prepared crop preview dimensions do not match the CropPlan output size."
        )
    homogeneous_quad = np.column_stack(
        [
            crop_plan.quad_raw_tl_tr_br_bl,
            np.ones(4, dtype=np.float64),
        ]
    )
    transformed_quad = (crop_plan.H_raw_to_prepared_3x3 @ homogeneous_quad.T).T
    denominators = transformed_quad[:, 2]
    if np.any(np.abs(denominators) <= np.finfo(np.float64).eps):
        raise CropReviewValidationError(
            "CropPlan maps a preview crop corner to infinity."
        )
    prepared_quad = transformed_quad[:, :2] / denominators[:, np.newaxis]
    if not np.all(np.isfinite(prepared_quad)):
        raise CropReviewValidationError(
            "CropPlan maps preview crop corners to non-finite coordinates."
        )
    fixed_point_scale = 1 << _PREVIEW_MASK_COORDINATE_SHIFT
    prepared_quad_fixed = np.rint(prepared_quad * fixed_point_scale).astype(np.int32)
    footprint = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(
        footprint,
        prepared_quad_fixed,
        color=255,
        lineType=cv2.LINE_8,
        shift=_PREVIEW_MASK_COORDINATE_SHIFT,
    )
    clipped_preview = np.array(preview, copy=True)
    clipped_preview[footprint == 0] = 0
    return clipped_preview


def _copy_plan_with_acceptance(crop_plan: CropPlan, accepted: bool) -> CropPlan:
    metadata = crop_plan.to_metadata_dict()
    metadata["accepted_by_user"] = accepted
    return CropPlan.model_validate(metadata)


class CropReviewController:
    """Coordinate candidate generation, preview, invalidation, and acceptance."""

    def __init__(
        self,
        state: PreprocessSetupState,
        *,
        detector: AutomaticDetector = detect_cage_crop_plan,
        manual_plan_builder: ManualPlanBuilder = make_manual_crop_plan,
        frame_reader: FrameReader = read_video_frame,
        preview_builder: PreviewBuilder = build_crop_preview,
    ) -> None:
        self.state = state
        self._detector = detector
        self._manual_plan_builder = manual_plan_builder
        self._frame_reader = frame_reader
        self._preview_builder = preview_builder
        self._representative_frame: np.ndarray | None = None
        self._representative_frame_index: int | None = None
        self._prepared_preview: np.ndarray | None = None
        self._preview_crop_plan: CropPlan | None = None
        self._manual_points: list[tuple[float, float]] = []
        self._upstream_context = self._context_key()
        self._seen_crop_review_revision = state.crop_review_revision

    @property
    def representative_frame(self) -> np.ndarray | None:
        """Return the current raw review frame held in controller memory."""

        return self._representative_frame

    @property
    def representative_frame_index(self) -> int | None:
        """Return the raw decode index used for the review frame."""

        return self._representative_frame_index

    @property
    def prepared_preview(self) -> np.ndarray | None:
        """Return the current non-artifact prepared crop preview."""

        return self._prepared_preview

    @property
    def manual_points(self) -> tuple[tuple[float, float], ...]:
        """Return manual raw-frame points in their exact click order."""

        return tuple(self._manual_points)

    @property
    def next_manual_point_label(self) -> str:
        """Return the next prescribed point label for the page."""

        if len(self._manual_points) >= len(_POINT_LABELS):
            return "all four points selected"
        return _POINT_LABELS[len(self._manual_points)]

    def synchronize_upstream_context(self) -> None:
        """Drop page-local frames and points after upstream identity changes."""

        context = self._context_key()
        if (
            context == self._upstream_context
            and self._seen_crop_review_revision == self.state.crop_review_revision
        ):
            return
        self._upstream_context = context
        self._representative_frame = None
        self._representative_frame_index = None
        self._manual_points.clear()
        self._clear_preview()
        self.state.invalidate_crop_review(
            "Upstream crop inputs changed; review and acceptance are required."
        )
        self._seen_crop_review_revision = self.state.crop_review_revision

    def load_representative_frame(self) -> np.ndarray:
        """Load start_frame when readable, otherwise deterministically load frame 0."""

        path, _config, _pre_crop = self._require_context()
        probe = self.state.raw_probe
        start_frame = self.state.start_frame if self.state.start_frame is not None else 0
        known_count = None
        if probe is not None:
            known_count = (
                probe.frame_count_opencv_readable
                if probe.frame_count_opencv_readable is not None
                else probe.frame_count_opencv_reported
            )
        start_is_valid = start_frame >= 0 and (
            known_count is None or known_count <= 0 or start_frame < known_count
        )
        attempted: list[int] = []
        if start_is_valid:
            attempted.append(start_frame)
        if 0 not in attempted:
            attempted.append(0)

        failures: list[str] = []
        for frame_index in attempted:
            try:
                frame = self._frame_reader(path, frame_index)
                self._validate_frame_shape(frame)
            except (PreprocessError, OSError, ValueError, cv2.error) as exc:
                failures.append(f"frame {frame_index}: {exc}")
                continue
            self._representative_frame = np.array(frame, copy=True)
            self._representative_frame_index = frame_index
            self._invalidate_candidate(
                "Representative frame loaded; crop review is required."
            )
            self.state.last_validation_error = None
            return self._representative_frame

        message = "Representative raw frame is unreadable"
        if failures:
            message += ": " + "; ".join(failures)
        self.state.invalidate_crop_review("Representative frame could not be loaded.")
        raise self._new_validation_error(message)

    def begin_automatic_detection(self) -> AutomaticDetectionRequest:
        """Capture typed worker inputs and invalidate all prior crop decisions."""

        path, config, pre_crop = self._require_context()
        if self._representative_frame is None:
            self.load_representative_frame()
        assert self._representative_frame is not None
        self._manual_points.clear()
        self.state.crop_mode = CropReviewMode.AUTOMATIC
        self._invalidate_candidate("Automatic cage detection is running.")
        frame = np.array(self._representative_frame, copy=True)
        frame.setflags(write=False)
        return AutomaticDetectionRequest(
            raw_video_path=path,
            config=config,
            pre_crop=pre_crop,
            representative_frame=frame,
        )

    def compute_automatic_detection(
        self,
        request: AutomaticDetectionRequest,
        *,
        progress_callback: Callable[[OperationProgress], None] | None = None,
        cancellation_requested: Callable[[], bool] | None = None,
    ) -> CropReviewComputation:
        """Run core detection and preview generation without mutating GUI state."""

        if progress_callback is None and cancellation_requested is None:
            result = self._detector(
                request.raw_video_path,
                request.config,
                request.pre_crop,
            )
        else:
            result = self._detector(
                request.raw_video_path,
                request.config,
                request.pre_crop,
                progress_callback=progress_callback,
                cancellation_requested=cancellation_requested,
            )
        if cancellation_requested is not None and cancellation_requested():
            raise CageDetectionCancelledError("Automatic cage detection was cancelled.")
        crop_plan = _copy_plan_with_acceptance(result.crop_plan, False)
        if crop_plan.mode is not CropMode.AUTOMATIC:
            raise CropReviewValidationError(
                "Automatic cage detection did not return an automatic CropPlan."
            )
        if progress_callback is not None:
            progress_callback(
                OperationProgress(
                    phase="Preparing preview",
                    message="Preparing preview",
                    completed_units=None,
                    total_units=None,
                    is_indeterminate=True,
                )
            )
        preview = self._preview_builder(
            request.representative_frame,
            crop_plan,
            request.config.prepare.perspective_interpolation,
        )
        if cancellation_requested is not None and cancellation_requested():
            raise CageDetectionCancelledError("Automatic cage detection was cancelled.")
        return CropReviewComputation(
            crop_plan=crop_plan,
            prepared_preview=preview,
            detector_diagnostics=dict(result.detector_diagnostics),
        )

    def apply_automatic_detection(
        self,
        result: CropReviewComputation,
    ) -> CropPlan:
        """Store a completed automatic result on the GUI thread."""

        self._apply_candidate(result)
        self.state.crop_mode = CropReviewMode.AUTOMATIC
        self.state.crop_review_status = "automatic_candidate_ready"
        return result.crop_plan

    def run_automatic_detection(self) -> CropPlan:
        """Synchronous headless convenience wrapper used by controller tests."""

        try:
            request = self.begin_automatic_detection()
            result = self.compute_automatic_detection(request)
        except (PreprocessError, ValidationError, ValueError, cv2.error) as exc:
            self.record_automatic_detection_failure(exc)
            if isinstance(exc, CropReviewValidationError):
                raise
            raise CropReviewValidationError(
                f"Automatic cage detection failed: {exc}"
            ) from exc
        return self.apply_automatic_detection(result)

    def record_automatic_detection_failure(self, exc: BaseException) -> None:
        """Clear stale plans after a worker-delivered detection failure."""

        self._invalidate_candidate("Automatic cage detection failed.")
        if isinstance(exc, CropReviewValidationError):
            message = str(exc)
        elif isinstance(
            exc,
            (PreprocessError, ValidationError, OSError, ValueError, cv2.error),
        ):
            message = f"Automatic cage detection failed: {exc}"
        else:
            message = "An unexpected error occurred during automatic cage detection."
            self.state.unexpected_error_detail = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
        self.state.last_validation_error = message

    def record_automatic_detection_cancellation(self) -> None:
        """Clear any candidate without presenting cooperative cancellation as failure."""

        self._invalidate_candidate("Automatic cage detection cancelled.")
        self.state.last_validation_error = None

    def update_detector_settings(
        self,
        *,
        sample_step: int | None = None,
        pad_px: int | None = None,
        threshold: int | None = None,
        roi_margin_px: int | None = None,
        perspective_interpolation: str | None = None,
        canonical_enabled: bool | None = None,
        canonical_width: int | None = None,
        canonical_height: int | None = None,
        pre_crop_expansion_percent: float | None = None,
        dilate_kernel_size: int | None = None,
        erode_kernel_size: int | None = None,
        rim_close_kernel_size: int | None = None,
        minimum_cage_width_fraction: float | None = None,
        minimum_cage_height_fraction: float | None = None,
        minimum_contour_area: float | None = None,
        fit_tolerance_px: int | None = None,
    ) -> PreprocessConfig:
        """Create and store a fully validated typed config copy for retry."""

        config = self.state.preprocess_config
        if config is None:
            raise self._new_validation_error("Load a preprocess configuration first.")
        self._invalidate_candidate("Detector settings changed; retry detection.")

        cage_updates = {
            name: value
            for name, value in {
                "sample_step": sample_step,
                "pad_px": pad_px,
                "threshold": threshold,
                "pre_crop_expansion_percent": pre_crop_expansion_percent,
                "dilate_kernel_size": dilate_kernel_size,
                "erode_kernel_size": erode_kernel_size,
                "rim_close_kernel_size": rim_close_kernel_size,
                "minimum_cage_width_fraction": minimum_cage_width_fraction,
                "minimum_cage_height_fraction": minimum_cage_height_fraction,
                "minimum_contour_area": minimum_contour_area,
                "fit_tolerance_px": fit_tolerance_px,
            }.items()
            if value is not None
        }
        prepare_updates = {
            name: value
            for name, value in {
                "roi_margin_px": roi_margin_px,
                "perspective_interpolation": perspective_interpolation,
            }.items()
            if value is not None
        }
        canonical_updates = {
            name: value
            for name, value in {
                "enabled": canonical_enabled,
                "width": canonical_width,
                "height": canonical_height,
            }.items()
            if value is not None
        }
        try:
            cage = CageDetectConfig.model_validate(
                {**config.cage_detect.model_dump(), **cage_updates}
            )
            canonical = CanonicalResolutionConfig.model_validate(
                {
                    **config.prepare.canonical_resolution.model_dump(),
                    **canonical_updates,
                }
            )
            prepare = PrepareConfig.model_validate(
                {
                    **config.prepare.model_dump(),
                    **prepare_updates,
                    "canonical_resolution": canonical,
                }
            )
            updated = PreprocessConfig.model_validate(
                {
                    **config.model_dump(),
                    "cage_detect": cage,
                    "prepare": prepare,
                }
            )
        except (ValidationError, ValueError) as exc:
            raise self._new_validation_error(f"Invalid detector settings: {exc}") from exc
        self.state.store_preprocess_config(updated)
        self.state.crop_mode = CropReviewMode.AUTOMATIC
        self.state.last_validation_error = None
        return updated

    def set_crop_mode(self, mode: CropReviewMode | str) -> CropReviewMode:
        """Select automatic or manual review and invalidate prior acceptance."""

        try:
            selected = CropReviewMode(mode)
        except ValueError as exc:
            raise self._new_validation_error(f"Unsupported crop mode: {mode}") from exc
        if selected is not self.state.crop_mode:
            self._manual_points.clear()
            self._invalidate_candidate("Crop mode changed; select and review a new crop.")
        self.state.crop_mode = selected
        return selected

    def add_manual_point(self, x: float, y: float) -> CropPlan | None:
        """Append one raw-coordinate point and delegate four-point geometry to core."""

        self.state.crop_mode = CropReviewMode.MANUAL
        self._invalidate_candidate("Manual points changed; crop acceptance is required.")
        if len(self._manual_points) >= 4:
            raise self._new_validation_error(
                "Exactly four manual points are already selected; clear them to retry."
            )
        try:
            point = (float(x), float(y))
        except (TypeError, ValueError) as exc:
            raise self._new_validation_error("Manual crop points must be numeric.") from exc
        if not all(math.isfinite(value) for value in point):
            raise self._new_validation_error("Manual crop points must be finite.")

        _path, config, pre_crop = self._require_context()
        if self._representative_frame is None:
            self.load_representative_frame()
        assert self._representative_frame is not None
        height, width = self._representative_frame.shape[:2]
        if not (0 <= point[0] < width and 0 <= point[1] < height):
            raise self._new_validation_error(
                "Manual crop point must lie within the raw image."
            )
        roi = pre_crop.roi
        if not (
            roi.x <= point[0] < roi.x + roi.width
            and roi.y <= point[1] < roi.y + roi.height
        ):
            raise self._new_validation_error(
                "Manual crop point must lie within the selected pre-crop ROI."
            )

        self._manual_points.append(point)
        if len(self._manual_points) < 4:
            self.state.crop_review_status = "manual_points_pending"
            self.state.last_validation_error = None
            return None

        try:
            plan = self._manual_plan_builder(
                raw_frame_shape=(height, width),
                points_tl_tr_br_bl=np.asarray(self._manual_points, dtype=np.float64),
                pre_crop_roi=(roi.x, roi.y, roi.width, roi.height),
                canonical_resolution=config.prepare.canonical_resolution,
            )
            plan = _copy_plan_with_acceptance(plan, False)
            if plan.mode is not CropMode.MANUAL:
                raise CropPlanError(
                    "Manual crop helper did not return a manual CropPlan."
                )
        except (CropPlanError, ValidationError, ValueError, cv2.error) as exc:
            self._invalidate_candidate("Manual crop is invalid.")
            raise self._new_validation_error(f"Manual crop is invalid: {exc}") from exc
        try:
            preview = self._preview_builder(
                self._representative_frame,
                plan,
                config.prepare.perspective_interpolation,
            )
        except (CropReviewValidationError, ValueError, cv2.error) as exc:
            self._invalidate_candidate("Prepared crop preview generation failed.")
            raise self._new_validation_error(
                f"Prepared crop preview generation failed: {exc}"
            ) from exc

        result = CropReviewComputation(
            crop_plan=plan,
            prepared_preview=preview,
            detector_diagnostics=None,
        )
        self._apply_candidate(result)
        self.state.crop_mode = CropReviewMode.MANUAL
        self.state.crop_review_status = "manual_candidate_ready"
        return plan

    def clear_manual_points(self) -> None:
        """Reset manual collection and invalidate any accepted manual crop."""

        self._manual_points.clear()
        self.state.crop_mode = CropReviewMode.MANUAL
        self._invalidate_candidate("Manual points cleared; select four points.")

    def accept_crop(self) -> CropPlan:
        """Create and store a separately validated explicitly accepted CropPlan."""

        candidate = self.state.candidate_crop_plan
        if candidate is None:
            raise self._new_validation_error("A valid candidate crop is required.")
        if candidate.accepted_by_user:
            raise self._new_validation_error("Candidate crop must remain unaccepted.")
        if self._prepared_preview is None or self._preview_crop_plan is not candidate:
            raise self._new_validation_error(
                "A matching prepared crop preview is required before acceptance."
            )
        try:
            accepted = _copy_plan_with_acceptance(candidate, True)
        except (CropPlanError, ValidationError) as exc:
            raise self._new_validation_error(
                f"Accepted CropPlan validation failed: {exc}"
            ) from exc
        self.state.store_accepted_crop_plan(accepted)
        self.state.crop_review_status = "Crop accepted. Ready for encode settings."
        self.state.last_validation_error = None
        return accepted

    def can_advance(self) -> bool:
        """Return whether an explicitly accepted CropPlan enables navigation."""

        accepted = self.state.accepted_crop_plan
        return accepted is not None and accepted.accepted_by_user

    def _apply_candidate(self, result: CropReviewComputation) -> None:
        if result.crop_plan.accepted_by_user:
            raise self._new_validation_error("Candidate CropPlan must be unaccepted.")
        width, height = result.crop_plan.prepared_size_wh
        if result.prepared_preview.shape[:2] != (height, width):
            raise self._new_validation_error(
                "Prepared crop preview dimensions do not match the candidate CropPlan."
            )
        self.state.candidate_crop_plan = result.crop_plan
        self.state.accepted_crop_plan = None
        self.state.detector_diagnostics = result.detector_diagnostics
        self._prepared_preview = np.array(result.prepared_preview, copy=True)
        self._preview_crop_plan = result.crop_plan
        self.state.last_validation_error = None

    def _invalidate_candidate(self, status: str) -> None:
        self.state.invalidate_crop_review(status)
        self._seen_crop_review_revision = self.state.crop_review_revision
        self._clear_preview()

    def _clear_preview(self) -> None:
        self._prepared_preview = None
        self._preview_crop_plan = None

    def _require_context(self) -> tuple[Path, PreprocessConfig, ResolvedPreCrop]:
        path = self.state.raw_video_path
        if path is None:
            raise self._new_validation_error("Raw video is missing.")
        config = self.state.preprocess_config
        if config is None:
            raise self._new_validation_error("Load a preprocess configuration first.")
        pre_crop = self.state.resolved_pre_crop
        if pre_crop is None or not self.state.trim_pre_crop_valid:
            raise self._new_validation_error("Resolve trim and pre-crop before crop review.")
        return path, config, pre_crop

    def _validate_frame_shape(self, frame: np.ndarray) -> None:
        array = np.asarray(frame)
        if array.ndim not in {2, 3} or array.size == 0:
            raise CropReviewValidationError("Representative raw frame is invalid.")
        probe = self.state.raw_probe
        if probe is not None and array.shape[:2] != (probe.height, probe.width):
            raise CropReviewValidationError(
                "Representative frame dimensions do not match the raw-video probe."
            )

    def _context_key(self) -> tuple[object, ...]:
        pre_crop = self.state.resolved_pre_crop
        roi = None
        if pre_crop is not None:
            roi = (
                pre_crop.roi.x,
                pre_crop.roi.y,
                pre_crop.roi.width,
                pre_crop.roi.height,
                pre_crop.raw_size_wh,
            )
        return (
            self.state.raw_video_path,
            self.state.start_frame,
            self.state.end_frame_exclusive,
            roi,
        )

    def _new_validation_error(self, message: str) -> CropReviewValidationError:
        self.state.last_validation_error = message
        return CropReviewValidationError(message)
