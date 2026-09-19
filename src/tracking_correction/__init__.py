"""Subsystem 03 tracking-correction package."""

from tracking_correction.contracts import (
    ACCEPTANCE_ACCEPTED,
    ACCEPTANCE_NOT_ACCEPTED,
    ACCEPTANCE_SUPERSEDED,
    CURRENT_BACKEND_ID,
    CURRENT_PROFILE_ID,
    BackendResult,
    TrackingCorrectionError,
    TrackingCorrectionRequest,
    TrackingCorrectionResult,
    is_currently_accepted,
)
from tracking_correction.manual_edits import ManualEditResult
from tracking_correction.review import (
    AcceptanceResult,
    CorrectionEpisode,
    PoseDisplaySource,
    TrackingReviewSession,
    TrackLegendEntry,
    accept_tracking_result,
    load_review_session,
    track_legend_entries,
)
from tracking_correction.run_discovery import (
    TrackingCorrectionProjectSummary,
    find_reviewable_s3_run_for_s2,
    summarize_tracking_correction_project,
)
from tracking_correction.runner import (
    project_root_from_s2_run_dir,
    run_tracking_correction,
    selected_profile_id,
    validate_s2_handoff,
)

__all__ = [
    "ACCEPTANCE_ACCEPTED",
    "ACCEPTANCE_NOT_ACCEPTED",
    "ACCEPTANCE_SUPERSEDED",
    "CURRENT_BACKEND_ID",
    "CURRENT_PROFILE_ID",
    "AcceptanceResult",
    "BackendResult",
    "CorrectionEpisode",
    "ManualEditResult",
    "PoseDisplaySource",
    "TrackLegendEntry",
    "TrackingCorrectionError",
    "TrackingCorrectionProjectSummary",
    "TrackingCorrectionRequest",
    "TrackingCorrectionResult",
    "TrackingReviewSession",
    "accept_tracking_result",
    "find_reviewable_s3_run_for_s2",
    "is_currently_accepted",
    "load_review_session",
    "project_root_from_s2_run_dir",
    "run_tracking_correction",
    "selected_profile_id",
    "summarize_tracking_correction_project",
    "track_legend_entries",
    "validate_s2_handoff",
]
