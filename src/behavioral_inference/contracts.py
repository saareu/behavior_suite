"""Subsystem 07 (behavioral inference) contracts.

Defines registry / inference schema stamps, active and unsupported detector
inventories, expected lab artifact digests, prediction grains, run artifact
names, discovery classifications, and shared error types.
"""

from __future__ import annotations

from combined_features.contracts import (
    ACTIVE_DYADIC_DETECTORS,
    OBSOLETE_DYADIC_DETECTORS,
)

# Package / settings schema.
SETTINGS_SCHEMA_VERSION = "behavioral_inference_settings_v1"
MODEL_MANIFEST_SCHEMA_VERSION = "behavioral_inference_model_manifest_v1"
PREDICTIONS_SCHEMA_VERSION = "behavioral_predictions_v1"
INFERENCE_RESULT_SCHEMA_VERSION = "behavioral_inference_result_v1"
RUN_META_SCHEMA_VERSION = "behavioral_inference_run_meta_v1"
QC_SCHEMA_VERSION = "behavioral_inference_qc_v1"
DYADIC_WIDE_SCHEMA_VERSION = "dyadic_predictions_wide_v1"
FOCAL_WIDE_SCHEMA_VERSION = "focal_predictions_wide_v1"

# Package / run layout.
BEHAVIORAL_INFERENCE_DIRNAME = "behavioral_inference"
PREDICTIONS_FILENAME = "predictions.parquet"
SPECIALIST_PREDICTIONS_FILENAME = "specialist_predictions.parquet"
DYADIC_WIDE_FILENAME = "dyadic_predictions_wide.parquet"
FOCAL_WIDE_FILENAME = "focal_predictions_wide.parquet"
RUN_META_FILENAME = "run_meta.json"
SETTINGS_USED_FILENAME = "settings_used.yaml"
MODEL_MANIFEST_FILENAME = "model_manifest.json"
QC_FILENAME = "qc.json"
PROCESSING_LOG_FILENAME = "processing_log.txt"

S7_RUN_ARTIFACT_FILENAMES = (
    PREDICTIONS_FILENAME,
    SPECIALIST_PREDICTIONS_FILENAME,
    DYADIC_WIDE_FILENAME,
    FOCAL_WIDE_FILENAME,
    RUN_META_FILENAME,
    SETTINGS_USED_FILENAME,
    MODEL_MANIFEST_FILENAME,
    QC_FILENAME,
    PROCESSING_LOG_FILENAME,
)

# Run status lifecycle (mirror S5/S6).
STATUS_NOT_STARTED = "not_started"
STATUS_RUNNING = "running"
STATUS_COMPLETED = "completed"
STATUS_COMPLETED_WITH_WARNINGS = "completed_with_warnings"
STATUS_FAILED = "failed"

# Discovery classifications (S6 + model staleness).
ELIGIBLE_DOWNSTREAM = "eligible"
COMPLETED_WITH_WARNINGS = "completed_with_warnings"
STALE_SOURCE = "stale_source"
STALE_MODELS = "stale_models"
FAILED_OR_INCOMPLETE = "failed_or_incomplete"
MISSING_REQUIRED_ARTIFACTS = "missing_required_artifacts"
INVALID_ARTIFACT = "invalid_artifact"

# Invalid-route reason codes (stable for QC / wide views / S8).
INVALID_REASON_OCCLUSION_MASK_EXCLUDED = "occlusion_mask_excluded"
INVALID_REASON_NON_MATCHING_OCCLUSION = "non_matching_occlusion_state"

# Score-origin annotations (export-time; does not change inference formulas).
SCORE_ORIGIN_MODEL_OBSERVED = "model_observed"
SCORE_ORIGIN_ROUTING_SENTINEL = "routing_sentinel"
SCORE_ORIGIN_FUSION_ROUTED = "fusion_routed"

# Registry discovery statuses (mirror S6 unavailable semantics).
REGISTRY_OK = "ok"
REGISTRY_PARTIAL = "partial"
REGISTRY_UNAVAILABLE = "unavailable"

# Bundle support classification.
BUNDLE_SUPPORTED = "supported"
BUNDLE_OBSOLETE = "obsolete"
BUNDLE_UNSUPPORTED = "unsupported"
BUNDLE_FALLBACK = "fallback"
BUNDLE_INCOMPLETE = "incomplete"

# Serialization labels for joblib XGBoost classifiers.
SERIALIZATION_JOBLIB_XGBOOST = "joblib_xgboost"
MODEL_FILENAME = "model.joblib"
FEATURE_SPEC_FILENAME = "feature_spec.json"

# Active executable specialists / detectors for Sprint 1 loading.
ACTIVE_DYADIC_BUNDLES = ACTIVE_DYADIC_DETECTORS
ACTIVE_FOCAL_BUNDLES = (
    "nose_body_intact",
    "nose_body_occ",
    "nose_to_tail",
    "approach",
    "retract",
)
# Documented Pipeline 90 single-stage fallback (not preferred when specialists exist).
FOCAL_FALLBACK_BUNDLES = ("nose_to_body",)
# JSON-expert / non-joblib architectures — unsupported in S7 MVP.
UNSUPPORTED_FOCAL_BUNDLES = ("nose_body", "nose_tail")

# Re-export obsolete dyadic name for registry marking.
OBSOLETE_BUNDLES = OBSOLETE_DYADIC_DETECTORS

# Lab-local SHA-256 digests for model.joblib (S7-B inventory; verification aid).
EXPECTED_MODEL_SHA256: dict[str, str] = {
    "dyadic/contact_intact": (
        "ffd2aeb7bc8f8b5582c85738f9192015e8f97cbe304f281fb31fa70ac19533f2"
    ),
    "dyadic/contact_occluded": (
        "1301ef479558a9e5031349c5a0dabf3cfae29e6724fc8567871af2ca30cd1256"
    ),
    "dyadic/nose_to_nose": (
        "84ae82becf4abc908d4fb8ecb72fc95ba322121d66acc997a79cdc419d30e2fa"
    ),
    "dyadic/mutually_disengaged": (
        "5c241be8ad687cbd320f408225b44e0ad2f675de5769ff33fa79957770c49f95"
    ),
    "dyadic/side_by_side": (
        "b8b3f65de40528be05bf9d606c9fd73e3bbcb1720b502fb8a562ac0fda9877c7"
    ),
    "dyadic/contact": (
        "6a2bc6ed4c76ba651b0c0633763fa119c96572c9a29796b3955cb2c95ec722e4"
    ),
    "focal/nose_body_intact": (
        "d96e78645caecb3e4875925f2d73daba5f4316e08e79e446fb4a7f7423c9d65c"
    ),
    "focal/nose_body_occ": (
        "318f33f032bf28c6bd9d32ad78197eb728529804d388e491d61ff9272d268d2d"
    ),
    "focal/nose_to_tail": (
        "6150e208e7edf4613c3b46751a3527419dea6e05307b7af6d804fd28ac6fab74"
    ),
    "focal/approach": (
        "3878b56b008d93855ae1944e35e6cd15a74953505b4074f39b0015c2e663f094"
    ),
    "focal/retract": (
        "33817b07a60310cc03b52361b35108aa334938d95e79bd319b10dba9ddff778b"
    ),
    "focal/nose_to_body": (
        "ceb650551d575218a7859132ab46a9e569fc2edffab8b3b079f33fcc54ebacf5"
    ),
}

# Runtime dependency package names for the optional ``s7`` extra.
S7_RUNTIME_DEPENDENCIES = ("xgboost", "sklearn", "joblib")

# PowerShell expands $7 inside double quotes; prefer single-quoted extras.
# The Windows GUI installer installs both S2 and S7 into behavior_suite_gui.
S7_RUNTIME_INSTALL_HINT = (
    "Install S7 runtime into the GUI Conda env with scripts\\install_windows_gui.bat, "
    "or manually: conda run -n behavior_suite_gui python -m pip install -e '.[s2,s7]'"
)

# Distinct GUI / operator readiness kinds (not discovery classifications).
READINESS_READY = "ready"
READINESS_NO_SESSION = "no_session"
READINESS_DISCOVERING = "discovering"
READINESS_MISSING_RUNTIME_DEPS = "missing_runtime_deps"
READINESS_REGISTRY_NOT_CONFIGURED = "registry_not_configured"
READINESS_REGISTRY_UNAVAILABLE = "registry_unavailable"
READINESS_MODELS_MISSING = "models_missing"
READINESS_MODELS_INCOMPATIBLE = "models_incompatible"
READINESS_S6_INELIGIBLE = "s6_ineligible"
READINESS_NO_DETECTORS = "no_detectors"
READINESS_THRESHOLDS_UNRESOLVED = "thresholds_unresolved"
READINESS_OTHER = "other"

# Positive class label expected for binary detectors.
POSITIVE_CLASS_LABEL = 1
NEGATIVE_CLASS_LABEL = 0

# Prediction grains and output roles (Sprint 2 in-memory contract).
GRAIN_UNDIRECTED_PAIR = "undirected_pair"
GRAIN_DIRECTIONAL = "directional"
FAMILY_DYADIC = "dyadic"
FAMILY_FOCAL = "focal"
OUTPUT_ROLE_PRIMARY = "primary"
OUTPUT_ROLE_SPECIALIST = "specialist"
OUTPUT_ROLE_LOGICAL_FUSED = "logical_fused"

# Logical detector inventories (config names, not necessarily bundle names).
LOGICAL_DYADIC_DETECTORS = (
    "contact",
    "nose_to_nose",
    "mutually_disengaged",
    "side_by_side",
)
LOGICAL_FOCAL_DETECTORS = (
    "nose_to_body",
    "nose_to_tail",
    "approach",
    "retract",
)
STANDALONE_DYADIC_DETECTORS = (
    "nose_to_nose",
    "mutually_disengaged",
    "side_by_side",
)
CONTACT_SPECIALISTS = ("contact_intact", "contact_occluded")
NOSE_TO_BODY_SPECIALISTS = ("nose_body_intact", "nose_body_occ")

# Occlusion routing column required by dual-model detectors.
OCCLUSION_COLUMN = "is_total_occlusion"

# Required identity / timing columns from eligible S6 features.
REQUIRED_FEATURE_META_COLS = (
    "exp_name",
    "focal_id",
    "partner_id",
    "dense_frame_idx",
)

# Nose-to-body legacy parity smoother (Pipeline 90).
NOSE_TO_BODY_ROLLING_WINDOW = 5
NOSE_TO_BODY_ROLLING_MIN_PERIODS = 1


class BehavioralInferenceError(ValueError):
    """Base error for Subsystem 07 behavioral inference."""


class ModelRegistryError(BehavioralInferenceError):
    """Registry discovery / configuration failures."""


class ModelLoadError(BehavioralInferenceError):
    """Trusted-path model loading / serialization failures."""


class FeatureValidationError(BehavioralInferenceError):
    """Runtime feature matrix validation failures."""


class ThresholdConfigError(BehavioralInferenceError):
    """Missing or ambiguous working-threshold resolution."""


class OptionalDependencyError(BehavioralInferenceError):
    """Required optional S7 runtime libraries are unavailable."""


class InferenceError(BehavioralInferenceError):
    """Fail-fast failures during dyadic/focal inference execution."""


class DetectorSelectionError(InferenceError):
    """Invalid or silently omitted detector selection."""


class PublicationError(BehavioralInferenceError):
    """Artifact validation, QC, or atomic publication failures."""


class ProvenanceError(BehavioralInferenceError):
    """Upstream S6 or model-source provenance / staleness failures."""
