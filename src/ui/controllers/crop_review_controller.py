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
    MaskConfig,
    MaskPolygonConfig,
    MaskRectangleConfig,
    MaskShapeConfig,
    PrepareConfig,
    PreprocessConfig,
)
from preprocess.crop_plan import CropMode, CropPlan
from preprocess.exceptions import (
    CageDetectionCancelledError,
    CropPlanError,
    PreprocessError,
    VideoPreparationError,
)
from preprocess.manual_crop import (
    make_axis_aligned_rectangle_crop_plan,
    make_manual_crop_plan,
)
from preprocess.masking import apply_static_mask_to_frame, validate_static_mask
from preprocess.models import OperationProgress
from preprocess.pre_crop import ResolvedPreCrop
from preprocess.video_probe import read_raw_frame_at_index
from ui.state import CropReviewMode, PreprocessSetupState

FrameReader = Callable[[Path, int], np.ndarray]
PreviewBuilder = Callable[..., np.ndarray]
AutomaticDetector = Callable[..., CageDetectionResult]
ManualPlanBuilder = Callable[..., CropPlan]
ManualRectanglePlanBuilder = Callable[..., CropPlan]

_POINT_LABELS = ("top-left", "top-right", "bottom-right", "bottom-left")
_PREVIEW_MASK_COORDINATE_SHIFT = 8
DETECTOR_SETTING_FIELDS = tuple(CageDetectConfig.model_fields)


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


@dataclass(frozen=True, slots=True)
class DetectorSettingsResetResult:
    """Summary of one detector-default reset operation."""

    changed_fields: tuple[str, ...]
    invalidated_candidate: bool
    invalidated_accepted: bool
    preserved_manual_acceptance: bool
    config: PreprocessConfig

    @property
    def already_at_defaults(self) -> bool:
        """Return whether the reset was a no-op."""

        return not self.changed_fields


def read_video_frame(video_path: Path, frame_index: int) -> np.ndarray:
    """Decode one requested raw frame using the shared raw-frame helper."""

    return read_raw_frame_at_index(video_path, frame_index)


def build_crop_preview(
    raw_frame: np.ndarray,
    crop_plan: CropPlan,
    perspective_interpolation: str,
    mask_config: MaskConfig | None = None,
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
    if mask_config is not None:
        return apply_static_mask_to_frame(clipped_preview, mask_config)
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
        manual_rectangle_plan_builder: ManualRectanglePlanBuilder = (
            make_axis_aligned_rectangle_crop_plan
        ),
        frame_reader: FrameReader = read_video_frame,
        preview_builder: PreviewBuilder = build_crop_preview,
    ) -> None:
        self.state = state
        self._detector = detector
        self._manual_plan_builder = manual_plan_builder
        self._manual_rectangle_plan_builder = manual_rectangle_plan_builder
        self._frame_reader = frame_reader
        self._preview_builder = preview_builder
        self._representative_frame: np.ndarray | None = None
        self._representative_frame_index: int | None = None
        self._prepared_preview: np.ndarray | None = None
        self._prepared_preview_unmasked: np.ndarray | None = None
        self._preview_crop_plan: CropPlan | None = None
        self._manual_points: list[tuple[float, float]] = []
        self._manual_rectangle_xywh: tuple[int, int, int, int] | None = None
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
        """Return the current non-artifact prepared crop preview with mask applied."""

        if self._prepared_preview_unmasked is None:
            return None
        config = self.state.preprocess_config
        if config is None:
            return np.array(self._prepared_preview_unmasked, copy=True)
        return apply_static_mask_to_frame(
            self._prepared_preview_unmasked,
            config.mask,
        )

    @property
    def manual_points(self) -> tuple[tuple[float, float], ...]:
        """Return manual raw-frame points in their exact click order."""

        return tuple(self._manual_points)

    @property
    def manual_rectangle_xywh(self) -> tuple[int, int, int, int] | None:
        """Return the current manual raw-frame rectangle selection, if any."""

        return self._manual_rectangle_xywh

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
        self._manual_rectangle_xywh = None
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
            known_count = probe.frame_count_opencv_readable
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
        self._manual_rectangle_xywh = None
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

    def prepared_preview_size_wh(self) -> tuple[int, int] | None:
        """Return the current prepared-preview dimensions when a candidate exists."""

        if self._prepared_preview_unmasked is not None:
            height, width = self._prepared_preview_unmasked.shape[:2]
            return width, height
        candidate = self.state.candidate_crop_plan
        if candidate is not None:
            return candidate.prepared_size_wh
        accepted = self.state.accepted_crop_plan
        if accepted is not None:
            return accepted.prepared_size_wh
        return None

    def crop_content_size_wh(self) -> tuple[int, int] | None:
        """Return authoritative current crop-content dimensions, if available."""

        crop_plan = self.state.candidate_crop_plan or self.state.accepted_crop_plan
        if crop_plan is None:
            return None
        return crop_plan.native_size_wh or crop_plan.prepared_size_wh

    def set_static_mask_enabled(self, enabled: bool) -> bool:
        """Enable or disable static prepared-pixel mask rendering."""

        config = self._require_config()
        mask = MaskConfig.model_validate(
            {
                **config.mask.model_dump(mode="python"),
                "enabled": enabled,
            }
        )
        return self._store_mask_config(mask)

    def add_static_mask_rectangle(
        self,
        rectangle_xywh: tuple[int, int, int, int],
    ) -> int:
        """Append one prepared-coordinate rectangle mask and return its index."""

        x, y, width, height = rectangle_xywh
        shape = MaskRectangleConfig(
            type="rectangle",
            x=x,
            y=y,
            width=width,
            height=height,
        )
        return self._append_static_mask_shape(shape)

    def add_static_mask_polygon(self, vertices: tuple[tuple[int, int], ...]) -> int:
        """Append one prepared-coordinate polygon mask and return its index."""

        shape = MaskPolygonConfig(type="polygon", vertices=vertices)
        return self._append_static_mask_shape(shape)

    def replace_static_mask_shape(self, index: int, shape: MaskShapeConfig) -> bool:
        """Replace one static mask shape without changing shape order."""

        config = self._require_config()
        shapes = list(config.mask.shapes)
        if index < 0 or index >= len(shapes):
            raise self._new_validation_error("Selected mask shape does not exist.")
        shapes[index] = shape
        return self._store_mask_config(
            MaskConfig.model_validate(
                {
                    **config.mask.model_dump(mode="python"),
                    "shapes": shapes,
                }
            )
        )

    def delete_static_mask_shape(self, index: int) -> bool:
        """Delete one selected static mask shape."""

        config = self._require_config()
        shapes = list(config.mask.shapes)
        if index < 0 or index >= len(shapes):
            raise self._new_validation_error("Selected mask shape does not exist.")
        del shapes[index]
        return self._store_mask_config(
            MaskConfig.model_validate(
                {
                    **config.mask.model_dump(mode="python"),
                    "shapes": shapes,
                }
            )
        )

    def clear_static_mask_shapes(self) -> bool:
        """Remove all static mask shapes while preserving enabled state."""

        config = self._require_config()
        return self._store_mask_config(
            MaskConfig.model_validate(
                {
                    **config.mask.model_dump(mode="python"),
                    "shapes": [],
                }
            )
        )

    def detector_settings_modified_fields(self) -> tuple[str, ...]:
        """Return cage-detection fields whose values differ from loaded defaults."""

        config = self.state.preprocess_config
        if config is None:
            return ()
        default_config = self._detector_defaults_config()
        current = config.cage_detect.model_dump(mode="python")
        defaults = default_config.cage_detect.model_dump(mode="python")
        return tuple(
            field
            for field in DETECTOR_SETTING_FIELDS
            if current[field] != defaults[field]
        )

    def reset_detector_settings_to_defaults(self) -> DetectorSettingsResetResult:
        """Reset only cage-detection settings from loaded preprocess defaults."""

        config = self.state.preprocess_config
        if config is None:
            raise self._new_validation_error("Load a preprocess configuration first.")
        default_config = self._detector_defaults_config()
        changed_fields = self.detector_settings_modified_fields()
        if not changed_fields:
            self.state.last_validation_error = None
            return DetectorSettingsResetResult(
                changed_fields=(),
                invalidated_candidate=False,
                invalidated_accepted=False,
                preserved_manual_acceptance=False,
                config=config,
            )

        updated = PreprocessConfig.model_validate(
            {
                **config.model_dump(mode="python"),
                "cage_detect": default_config.cage_detect,
            }
        )
        self.state.store_preprocess_config(updated)

        invalidated_candidate = self._clear_automatic_candidate()
        invalidated_accepted = self._clear_automatic_acceptance()
        preserved_manual_acceptance = (
            self.state.accepted_crop_plan is not None
            and self.state.accepted_crop_plan.mode is CropMode.MANUAL
        )
        if invalidated_candidate or invalidated_accepted:
            self.state.crop_review_revision += 1
            self._seen_crop_review_revision = self.state.crop_review_revision
        if invalidated_candidate or invalidated_accepted:
            self.state.crop_review_status = (
                "Detector settings reset to defaults; rerun automatic detection."
            )
        elif preserved_manual_acceptance:
            self.state.crop_review_status = (
                "Detector settings reset to defaults; accepted manual crop preserved."
            )
        else:
            self.state.crop_review_status = (
                "Detector settings reset to defaults; automatic detection must be rerun."
            )
        self.state.last_validation_error = None
        return DetectorSettingsResetResult(
            changed_fields=changed_fields,
            invalidated_candidate=invalidated_candidate,
            invalidated_accepted=invalidated_accepted,
            preserved_manual_acceptance=preserved_manual_acceptance,
            config=updated,
        )

    def set_crop_mode(self, mode: CropReviewMode | str) -> CropReviewMode:
        """Select automatic/manual review and invalidate prior acceptance."""

        try:
            selected = CropReviewMode(mode)
        except ValueError as exc:
            raise self._new_validation_error(f"Unsupported crop mode: {mode}") from exc
        if selected is not self.state.crop_mode:
            self._manual_points.clear()
            self._manual_rectangle_xywh = None
            self._invalidate_candidate("Crop mode changed; select and review a new crop.")
        self.state.crop_mode = selected
        return selected

    def add_manual_point(self, x: float, y: float) -> CropPlan | None:
        """Append one raw-coordinate point and delegate four-point geometry to core."""

        self.state.crop_mode = CropReviewMode.MANUAL
        self._manual_rectangle_xywh = None
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

    def set_manual_rectangle(
        self,
        rectangle_xywh: tuple[int, int, int, int],
    ) -> CropPlan:
        """Set one axis-aligned manual rectangle and build its CropPlan."""

        self.state.crop_mode = CropReviewMode.MANUAL_RECTANGLE
        self._manual_points.clear()
        self._invalidate_candidate(
            "Manual rectangle changed; crop acceptance is required."
        )
        _path, config, pre_crop = self._require_context()
        if self._representative_frame is None:
            self.load_representative_frame()
        assert self._representative_frame is not None
        height, width = self._representative_frame.shape[:2]
        clamped = self._clamp_manual_rectangle(rectangle_xywh, width, height, pre_crop)
        self._manual_rectangle_xywh = clamped
        roi = pre_crop.roi

        try:
            plan = self._manual_rectangle_plan_builder(
                raw_frame_shape=(height, width),
                rectangle_xywh=clamped,
                pre_crop_roi=(roi.x, roi.y, roi.width, roi.height),
                canonical_resolution=config.prepare.canonical_resolution,
            )
            plan = _copy_plan_with_acceptance(plan, False)
            if plan.mode is not CropMode.MANUAL:
                raise CropPlanError(
                    "Manual rectangle helper did not return a manual CropPlan."
                )
        except (CropPlanError, ValidationError, ValueError, cv2.error) as exc:
            self._clear_preview()
            self.state.crop_review_status = "manual_rectangle_invalid"
            raise self._new_validation_error(
                f"Manual rectangle crop is invalid: {exc}"
            ) from exc
        try:
            preview = self._preview_builder(
                self._representative_frame,
                plan,
                config.prepare.perspective_interpolation,
            )
        except (CropReviewValidationError, ValueError, cv2.error) as exc:
            self._invalidate_candidate("Prepared crop preview generation failed.")
            self._manual_rectangle_xywh = clamped
            raise self._new_validation_error(
                f"Prepared crop preview generation failed: {exc}"
            ) from exc

        result = CropReviewComputation(
            crop_plan=plan,
            prepared_preview=preview,
            detector_diagnostics=None,
        )
        self._apply_candidate(result)
        self.state.crop_mode = CropReviewMode.MANUAL_RECTANGLE
        self.state.crop_review_status = "manual_rectangle_candidate_ready"
        return plan

    def clear_manual_rectangle(self) -> None:
        """Reset manual rectangle selection and invalidate accepted rectangle crop."""

        self._manual_rectangle_xywh = None
        self.state.crop_mode = CropReviewMode.MANUAL_RECTANGLE
        self._invalidate_candidate("Manual rectangle cleared; draw a rectangle.")

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
        self._prepared_preview_unmasked = np.array(result.prepared_preview, copy=True)
        self._prepared_preview = self.prepared_preview
        self._preview_crop_plan = result.crop_plan
        self.state.last_validation_error = None

    def _invalidate_candidate(self, status: str) -> None:
        self.state.invalidate_crop_review(status)
        self._seen_crop_review_revision = self.state.crop_review_revision
        self._clear_preview()

    def _detector_defaults_config(self) -> PreprocessConfig:
        default_config = (
            self.state.default_preprocess_config
            or self.state.original_preprocess_config
            or PreprocessConfig()
        )
        return PreprocessConfig.model_validate(default_config.model_dump(mode="python"))

    def _clear_automatic_candidate(self) -> bool:
        candidate = self.state.candidate_crop_plan
        if candidate is None or candidate.mode is not CropMode.AUTOMATIC:
            return False
        self.state.candidate_crop_plan = None
        self.state.detector_diagnostics = None
        if self._preview_crop_plan is candidate:
            self._clear_preview()
        return True

    def _clear_automatic_acceptance(self) -> bool:
        accepted = self.state.accepted_crop_plan
        if accepted is None or accepted.mode is not CropMode.AUTOMATIC:
            return False
        self.state.accepted_crop_plan = None
        return True

    def _clear_preview(self) -> None:
        self._prepared_preview = None
        self._prepared_preview_unmasked = None
        self._preview_crop_plan = None

    def _require_config(self) -> PreprocessConfig:
        config = self.state.preprocess_config
        if config is None:
            raise self._new_validation_error("Load a preprocess configuration first.")
        return config

    def _append_static_mask_shape(self, shape: MaskShapeConfig) -> int:
        config = self._require_config()
        shapes = [*config.mask.shapes, shape]
        self._store_mask_config(
            MaskConfig.model_validate(
                {
                    **config.mask.model_dump(mode="python"),
                    "enabled": True,
                    "shapes": shapes,
                }
            )
        )
        return len(shapes) - 1

    def _store_mask_config(self, mask: MaskConfig) -> bool:
        config = self._require_config()
        preview_size = self.prepared_preview_size_wh()
        if preview_size is not None:
            try:
                validate_static_mask(mask, preview_size)
            except (ValueError, VideoPreparationError) as exc:
                raise self._new_validation_error(f"Invalid static mask: {exc}") from exc
        if mask == config.mask:
            self.state.last_validation_error = None
            return False
        updated = PreprocessConfig.model_validate(
            {
                **config.model_dump(mode="python"),
                "mask": mask,
            }
        )
        self.state.store_preprocess_config(updated)
        self._prepared_preview = self.prepared_preview
        self.state.last_validation_error = None
        return True

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

    def _clamp_manual_rectangle(
        self,
        rectangle_xywh: tuple[int, int, int, int],
        frame_width: int,
        frame_height: int,
        pre_crop: ResolvedPreCrop,
    ) -> tuple[int, int, int, int]:
        try:
            value_count = len(rectangle_xywh)
        except TypeError as exc:
            raise self._new_validation_error(
                "Manual rectangle must contain x, y, width, and height."
            ) from exc
        if value_count != 4:
            raise self._new_validation_error(
                "Manual rectangle must contain x, y, width, and height."
            )
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in rectangle_xywh
        ):
            raise self._new_validation_error("Manual rectangle values must be integers.")

        x, y, width, height = rectangle_xywh
        if width <= 0 or height <= 0:
            raise self._new_validation_error(
                "Manual rectangle width and height must be positive."
            )
        raw_left = max(0, x)
        raw_top = max(0, y)
        raw_right = min(frame_width, x + width)
        raw_bottom = min(frame_height, y + height)
        roi = pre_crop.roi
        left = max(raw_left, roi.x)
        top = max(raw_top, roi.y)
        right = min(raw_right, roi.x + roi.width)
        bottom = min(raw_bottom, roi.y + roi.height)
        clamped_width = right - left
        clamped_height = bottom - top
        if clamped_width <= 0 or clamped_height <= 0:
            raise self._new_validation_error(
                "Manual rectangle must overlap the selected pre-crop ROI with "
                "positive width and height."
            )
        return left, top, clamped_width, clamped_height

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
            roi,
        )

    def _new_validation_error(self, message: str) -> CropReviewValidationError:
        self.state.last_validation_error = message
        return CropReviewValidationError(message)
