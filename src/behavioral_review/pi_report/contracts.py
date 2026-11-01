"""Frozen contracts for PI-facing HTML behavioral report export (B1)."""

from __future__ import annotations

from behavioral_review.contracts import (
    GALLERY_DEFAULT_PREVIEW_PAD_SEC,
    SOURCE_TYPE_AUTOMATED,
    SOURCE_TYPE_REVIEWED,
    BehavioralReviewError,
)
from results_packaging.media_csv_export.contracts import (
    DEFAULT_CODEC,
    DEFAULT_CRF,
    DEFAULT_PIXEL_FORMAT,
    RESULTS_EXPORTS_DIRNAME,
)

# Re-export source vocabulary used in manifests.
PI_SOURCE_AUTOMATED = SOURCE_TYPE_AUTOMATED
PI_SOURCE_REVIEWED = SOURCE_TYPE_REVIEWED

PI_REPORT_RUN_PREFIX = "pi_report"
PI_REPORT_JOB_KIND = "pi_report"

PI_REPORT_MANIFEST_SCHEMA_VERSION = "pi_report_manifest_v1"
PI_REPORT_DATA_SCHEMA_VERSION = "pi_report_data_v1"
PI_REPORT_QC_SCHEMA_VERSION = "pi_report_qc_v1"
PI_REPORT_RUN_META_SCHEMA_VERSION = "pi_report_run_meta_v1"
PI_REPORT_SETTINGS_SCHEMA_VERSION = "pi_report_settings_v1"

REPORT_MANIFEST_FILENAME = "report_manifest.json"
REPORT_DATA_FILENAME = "report_data.json"
INDEX_HTML_FILENAME = "index.html"
README_FILENAME = "README.txt"
QC_FILENAME = "qc.json"
RUN_META_FILENAME = "run_meta.json"
SETTINGS_USED_FILENAME = "settings_used.yaml"
PROCESSING_LOG_FILENAME = "processing_log.txt"
MEDIA_DIRNAME = "media"
ASSETS_DIRNAME = "assets"
APP_CSS_FILENAME = "app.css"
APP_JS_FILENAME = "app.js"
REPORT_EMBED_JS_FILENAME = "report_embed.js"
STAGING_DIRNAME = "_pi_report_staging"

DEFAULT_PRE_CONTEXT_SEC = float(GALLERY_DEFAULT_PREVIEW_PAD_SEC)
DEFAULT_POST_CONTEXT_SEC = float(GALLERY_DEFAULT_PREVIEW_PAD_SEC)
DEFAULT_MOVFLAGS = "+faststart"

MEDIA_ENCODE_SUCCEEDED = "succeeded"
MEDIA_ENCODE_FAILED = "failed"
MEDIA_ENCODE_OMITTED = "omitted"

# Report scope labels (never treat a selected-bout subset as a complete export).
REPORT_SCOPE_COMPLETE = "all_exclusive_bouts"
REPORT_SCOPE_SELECTED = "selected_bouts"

# Terminal job statuses surfaced by GUI discovery / run_meta.
STATUS_CANCELED = "canceled"

REQUIRED_REPORT_FILES: tuple[str, ...] = (
    INDEX_HTML_FILENAME,
    REPORT_DATA_FILENAME,
    REPORT_MANIFEST_FILENAME,
    QC_FILENAME,
    RUN_META_FILENAME,
    SETTINGS_USED_FILENAME,
    PROCESSING_LOG_FILENAME,
    f"{ASSETS_DIRNAME}/{APP_CSS_FILENAME}",
    f"{ASSETS_DIRNAME}/{APP_JS_FILENAME}",
    f"{ASSETS_DIRNAME}/{REPORT_EMBED_JS_FILENAME}",
    README_FILENAME,
)


class PiReportError(BehavioralReviewError):
    """Base error for PI report export."""


class PiReportSourceError(PiReportError):
    """Invalid or forbidden report source (e.g. working draft)."""


class PiReportPublicationError(PiReportError):
    """Report staging / finalize / immutability failure."""


class PiReportMediaError(PiReportError):
    """Preview clip encode / probe failure."""


class PiReportValidationError(PiReportError):
    """Report QC / integrity validation failure."""


class PiReportCancelledError(PiReportPublicationError):
    """Cooperative cancellation before final publication."""


__all__ = [
    "ASSETS_DIRNAME",
    "APP_CSS_FILENAME",
    "APP_JS_FILENAME",
    "DEFAULT_CODEC",
    "DEFAULT_CRF",
    "DEFAULT_MOVFLAGS",
    "DEFAULT_PIXEL_FORMAT",
    "DEFAULT_POST_CONTEXT_SEC",
    "DEFAULT_PRE_CONTEXT_SEC",
    "INDEX_HTML_FILENAME",
    "MEDIA_DIRNAME",
    "MEDIA_ENCODE_FAILED",
    "MEDIA_ENCODE_OMITTED",
    "MEDIA_ENCODE_SUCCEEDED",
    "PI_REPORT_DATA_SCHEMA_VERSION",
    "PI_REPORT_JOB_KIND",
    "PI_REPORT_MANIFEST_SCHEMA_VERSION",
    "PI_REPORT_QC_SCHEMA_VERSION",
    "PI_REPORT_RUN_META_SCHEMA_VERSION",
    "PI_REPORT_RUN_PREFIX",
    "PI_REPORT_SETTINGS_SCHEMA_VERSION",
    "PI_SOURCE_AUTOMATED",
    "PI_SOURCE_REVIEWED",
    "PROCESSING_LOG_FILENAME",
    "QC_FILENAME",
    "README_FILENAME",
    "REPORT_DATA_FILENAME",
    "REPORT_EMBED_JS_FILENAME",
    "REPORT_MANIFEST_FILENAME",
    "REPORT_SCOPE_COMPLETE",
    "REPORT_SCOPE_SELECTED",
    "REQUIRED_REPORT_FILES",
    "RESULTS_EXPORTS_DIRNAME",
    "RUN_META_FILENAME",
    "SETTINGS_USED_FILENAME",
    "STATUS_CANCELED",
    "STAGING_DIRNAME",
    "PiReportCancelledError",
    "PiReportError",
    "PiReportMediaError",
    "PiReportPublicationError",
    "PiReportSourceError",
    "PiReportValidationError",
]
