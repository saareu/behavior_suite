"""PI-facing immutable HTML behavioral report export (Increments B1/B2).

Resolves automated S9 or published S10 sources, builds exclusive gallery groups,
materializes local H.264 previews via S9 frame-accurate encode, and publishes an
offline ``results_exports/pi_report__*`` bundle. B2 adds cooperative cancel,
source digest recheck, selected-bout scope, and GUI integration hooks.

Working drafts are rejected. Contact/SBS are excluded from the default exclusive
view. No overlays or GIF generation.
"""

from behavioral_review.pi_report.contracts import (
    PI_REPORT_DATA_SCHEMA_VERSION,
    PI_REPORT_MANIFEST_SCHEMA_VERSION,
    PI_REPORT_RUN_PREFIX,
    PI_SOURCE_AUTOMATED,
    PI_SOURCE_REVIEWED,
    REPORT_SCOPE_COMPLETE,
    REPORT_SCOPE_SELECTED,
    RESULTS_EXPORTS_DIRNAME,
    STATUS_CANCELED,
    PiReportCancelledError,
    PiReportError,
    PiReportMediaError,
    PiReportPublicationError,
    PiReportSourceError,
    PiReportValidationError,
)
from behavioral_review.pi_report.discovery import (
    PiReportProjectSummary,
    PiReportSummary,
    summarize_pi_report_project,
)
from behavioral_review.pi_report.publication import (
    PiReportResult,
    PiReportSettings,
    publish_pi_report,
    validate_report_directory,
    write_pi_report_zip,
)
from behavioral_review.pi_report.source import (
    PiReportSource,
    assert_not_draft_source,
    resolve_pi_report_source,
    verify_pinned_source_integrity,
)

__all__ = [
    "PI_REPORT_DATA_SCHEMA_VERSION",
    "PI_REPORT_MANIFEST_SCHEMA_VERSION",
    "PI_REPORT_RUN_PREFIX",
    "PI_SOURCE_AUTOMATED",
    "PI_SOURCE_REVIEWED",
    "REPORT_SCOPE_COMPLETE",
    "REPORT_SCOPE_SELECTED",
    "RESULTS_EXPORTS_DIRNAME",
    "STATUS_CANCELED",
    "PiReportCancelledError",
    "PiReportError",
    "PiReportMediaError",
    "PiReportProjectSummary",
    "PiReportPublicationError",
    "PiReportResult",
    "PiReportSettings",
    "PiReportSource",
    "PiReportSourceError",
    "PiReportSummary",
    "PiReportValidationError",
    "assert_not_draft_source",
    "publish_pi_report",
    "resolve_pi_report_source",
    "summarize_pi_report_project",
    "validate_report_directory",
    "verify_pinned_source_integrity",
    "write_pi_report_zip",
]
