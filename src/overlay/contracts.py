"""Overlay compositor contracts for Increment C1/C2-A/C2-B.

Defines layer ids, frame-domain / plateau-policy enums, unavailable markers,
and schema stamps. Scientific schemas are not modified here.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final

OVERLAY_PACKAGE_NAME: Final[str] = "overlay"
OVERLAY_RENDERER_VERSION: Final[str] = "overlay_renderer_c2a_v1"
OVERLAY_S4_POSE_RENDERER_VERSION: Final[str] = "overlay_renderer_c2b_v1"
OVERLAY_S2_POSE_RENDERER_VERSION: Final[str] = "overlay_renderer_c3_1_v1"
OVERLAY_ARTIFACT_MANIFEST_SCHEMA: Final[str] = "overlay_artifact_manifest_v1"
OVERLAY_QC_EXPORT_DIRNAME: Final[str] = "results_exports"
OVERLAY_QC_JOB_PREFIX: Final[str] = "overlay_qc_behavioral__"
OVERLAY_QC_VIDEO_FILENAME: Final[str] = "behavioral_qc.mp4"
OVERLAY_QC_MANIFEST_FILENAME: Final[str] = "overlay_artifact_manifest.json"
OVERLAY_QC_STAGING_DIRNAME: Final[str] = "_overlay_qc_staging"
OVERLAY_QC_S4_POSE_JOB_PREFIX: Final[str] = "overlay_qc_s4_pose__"
OVERLAY_QC_S4_POSE_VIDEO_FILENAME: Final[str] = "s4_pose_qc.mp4"
OVERLAY_QC_S2_POSE_JOB_PREFIX: Final[str] = "overlay_qc_s2_pose__"
OVERLAY_QC_S2_POSE_VIDEO_FILENAME: Final[str] = "s2_pose_qc.mp4"
OVERLAY_QC_PROCESSING_LOG_FILENAME: Final[str] = "processing_log.txt"
OVERLAY_QC_RUN_META_FILENAME: Final[str] = "run_meta.json"
OVERLAY_QC_SETTINGS_FILENAME: Final[str] = "settings_used.yaml"

# SLEAP per-keypoint ``score`` is conventionally near [0, 1] but may exceed 1.
# Visual mapping assumes this range and clamps for drawing only; original scores
# are preserved in observation payloads / manifests. Score never filters nodes.
S2_SCORE_VISUAL_RANGE_MIN: Final[float] = 0.0
S2_SCORE_VISUAL_RANGE_MAX: Final[float] = 1.0
S2_SCORE_VISUAL_MAPPING_ID: Final[str] = "s2_score_fill_intensity_v1"
S2_SCORE_VISUAL_MAPPING_SUMMARY: Final[str] = (
    "Optional continuous confidence: filled disc intensity scales with SLEAP "
    "score assuming [0,1] (values outside clamped for drawing only). "
    "Missing/NaN score → hollow ring (distinct from numerical zero). "
    "Track-colored outline keeps provisional identity legible. "
    "No confidence thresholding; drawable nodes are never dropped for viz."
)

# Pixel domain for composed frames (OpenCV / existing draw helpers).
PIXEL_FORMAT_BGR: Final[str] = "bgr"


class FrameDomain(StrEnum):
    """Output / context frame domain.

    Video products default to prepared-video space. Dense 120 Hz remains the
    authoritative behavioral timeline; dense_export is reserved for opt-in C2+.
    """

    PREPARED = "prepared"
    DENSE_EXPORT = "dense_export"


class PlateauPolicy(StrEnum):
    """How a prepared frame resolves onto dense behavioral state.

    ``LAST_IN_PLATEAU`` (C1 offline default): among dense samples that share the
    prepared frame, use the chronologically latest ``sample_idx``.

    ``NEAREST_TIME``: SyncMap interactive contract — nearest dense ``time_sec``
    to plateau ``prepared_time_sec``; earlier ``sample_idx`` wins residual ties.
    Documented for contrast; not the offline prepared-rate HUD default.

    ``PLAYHEAD``: interactive live authority (C3); not used for offline C1.
    """

    LAST_IN_PLATEAU = "last_in_plateau"
    NEAREST_TIME = "nearest_time"
    PLAYHEAD = "playhead"


class BehaviorSourceKind(StrEnum):
    """Keep automated S8 and published reviewed S10 identities separate."""

    S8_AUTOMATED = "s8_automated"
    S10_REVIEWED = "s10_reviewed"


class HudDirectionMode(StrEnum):
    """Directional exclusive HUD presentation for behavioral QC export."""

    FOCAL_0_ONLY = "focal_0_only"
    FOCAL_1_ONLY = "focal_1_only"
    BOTH = "both"


class OverlayExportScopeKind(StrEnum):
    """How the prepared-frame export interval was selected."""

    PREPARED_INTERVAL = "prepared_interval"
    BOUT_SPAN = "bout_span"


class LayerId(StrEnum):
    """Independently selectable overlay layers."""

    S2_POSE_INFERRED = "s2.pose_inferred"
    S3_POSE_TRACKED = "s3.pose_tracked"
    S4_POSE_FINAL = "s4.pose_final"
    S5_SEGMENTATION = "s5.segmentation"
    S8_EXCLUSIVE_A_TO_B = "s8.exclusive.A_to_B"
    S8_EXCLUSIVE_B_TO_A = "s8.exclusive.B_to_A"
    S8_CONTACT = "s8.contact"
    S8_SBS = "s8.sbs"
    DEBUG_FRAME_COUNTER = "debug.frame_counter"


# Layers implemented enough for C1 contract / timing / compose validation.
C1_SUPPORTED_LAYERS: frozenset[LayerId] = frozenset(
    {
        LayerId.S2_POSE_INFERRED,
        LayerId.S3_POSE_TRACKED,
        LayerId.S4_POSE_FINAL,
        LayerId.S8_EXCLUSIVE_A_TO_B,
        LayerId.S8_EXCLUSIVE_B_TO_A,
        LayerId.S8_CONTACT,
        LayerId.S8_SBS,
        LayerId.DEBUG_FRAME_COUNTER,
    }
)

# Behavioral-only QC default layers (C2-A; no pose tables required).
C2A_BEHAVIORAL_QC_LAYERS: frozenset[LayerId] = frozenset(
    {
        LayerId.S8_EXCLUSIVE_A_TO_B,
        LayerId.S8_EXCLUSIVE_B_TO_A,
        LayerId.S8_CONTACT,
        LayerId.S8_SBS,
        LayerId.DEBUG_FRAME_COUNTER,
    }
)

# Deferred beyond C1 (interfaces only / always unavailable).
C1_DEFERRED_LAYERS: frozenset[LayerId] = frozenset(
    {
        LayerId.S5_SEGMENTATION,
    }
)

# Faithful S4 provenance codes (do not invent others).
S4_PROVENANCE_MISSING: Final[int] = 0
S4_PROVENANCE_ORIGINAL: Final[int] = 1
S4_PROVENANCE_IDENTITY_CORRECTED: Final[int] = 2
S4_PROVENANCE_TEMPORAL_RESAMPLE: Final[int] = 3
S4_PROVENANCE_LINEAR_GAP_FILL: Final[int] = 4
S4_PROVENANCE_KALMAN_GAP_FILL: Final[int] = 5
S4_PROVENANCE_PROTECTED_MANUAL_BLANK: Final[int] = 6

S4_PROVENANCE_NAMES: Final[dict[int, str]] = {
    S4_PROVENANCE_MISSING: "missing",
    S4_PROVENANCE_ORIGINAL: "original",
    S4_PROVENANCE_IDENTITY_CORRECTED: "identity_corrected",
    S4_PROVENANCE_TEMPORAL_RESAMPLE: "temporal_resample",
    S4_PROVENANCE_LINEAR_GAP_FILL: "linear_gap_fill",
    S4_PROVENANCE_KALMAN_GAP_FILL: "kalman_gap_fill",
    S4_PROVENANCE_PROTECTED_MANUAL_BLANK: "protected_manual_blank",
}

# Observed / reconstructed / resampled visual classes (C2-B; O3 third class).
S4_OBSERVED_PROVENANCE: frozenset[int] = frozenset(
    {S4_PROVENANCE_ORIGINAL, S4_PROVENANCE_IDENTITY_CORRECTED}
)
S4_TEMPORAL_RESAMPLE_PROVENANCE: frozenset[int] = frozenset(
    {S4_PROVENANCE_TEMPORAL_RESAMPLE}
)
S4_GAP_FILL_PROVENANCE: frozenset[int] = frozenset(
    {S4_PROVENANCE_LINEAR_GAP_FILL, S4_PROVENANCE_KALMAN_GAP_FILL}
)
# Legacy umbrella (observed vs any non-observed reconstruction). Prefer the
# three-way visual-class map for drawing; keep this for adapters/tests.
S4_RECONSTRUCTED_PROVENANCE: frozenset[int] = frozenset(
    S4_TEMPORAL_RESAMPLE_PROVENANCE | S4_GAP_FILL_PROVENANCE
)
S4_UNAVAILABLE_PROVENANCE: frozenset[int] = frozenset(
    {S4_PROVENANCE_MISSING, S4_PROVENANCE_PROTECTED_MANUAL_BLANK}
)


class S4ProvenanceVisualClass(StrEnum):
    """Per-keypoint visual class for S4 QC (not a scientific category)."""

    OBSERVED = "observed"
    TEMPORAL_RESAMPLE = "temporal_resample"
    GAP_FILL = "gap_fill"
    UNAVAILABLE = "unavailable"


S4_PROVENANCE_VISUAL_CLASS_BY_CODE: Final[dict[int, S4ProvenanceVisualClass]] = {
    S4_PROVENANCE_MISSING: S4ProvenanceVisualClass.UNAVAILABLE,
    S4_PROVENANCE_ORIGINAL: S4ProvenanceVisualClass.OBSERVED,
    S4_PROVENANCE_IDENTITY_CORRECTED: S4ProvenanceVisualClass.OBSERVED,
    S4_PROVENANCE_TEMPORAL_RESAMPLE: S4ProvenanceVisualClass.TEMPORAL_RESAMPLE,
    S4_PROVENANCE_LINEAR_GAP_FILL: S4ProvenanceVisualClass.GAP_FILL,
    S4_PROVENANCE_KALMAN_GAP_FILL: S4ProvenanceVisualClass.GAP_FILL,
    S4_PROVENANCE_PROTECTED_MANUAL_BLANK: S4ProvenanceVisualClass.UNAVAILABLE,
}

# S4 pose QC default layers (C2-B; no behavioral HUD required).
C2B_S4_POSE_QC_LAYERS: frozenset[LayerId] = frozenset(
    {
        LayerId.S4_POSE_FINAL,
        LayerId.DEBUG_FRAME_COUNTER,
    }
)

# S2 inferred-pose QC default layers (C3-1; prepared-domain; no dense HUD).
C31_S2_POSE_QC_LAYERS: frozenset[LayerId] = frozenset(
    {
        LayerId.S2_POSE_INFERRED,
        LayerId.DEBUG_FRAME_COUNTER,
    }
)


class OverlayError(RuntimeError):
    """Base overlay compositor error."""


class OverlayContractError(OverlayError):
    """Invalid layer selection, manifest, or adapter contract."""


class OverlayTimingError(OverlayError):
    """Prepared↔dense timing resolution failure."""


class OverlayAdapterError(OverlayError):
    """Read-only scientific adapter failure."""


class OverlayRenderError(OverlayError):
    """Frame composition failure."""


class OverlaySourceError(OverlayError):
    """Overlay QC source selection / eligibility failure."""


class OverlayExportError(OverlayError):
    """Behavioral QC video export failure."""


class OverlayExportCancelled(OverlayExportError):
    """Cooperative cancellation requested during overlay QC export."""


def parse_layer_id(value: str | LayerId) -> LayerId:
    """Parse a layer id string into ``LayerId``."""

    if isinstance(value, LayerId):
        return value
    try:
        return LayerId(str(value))
    except ValueError as exc:
        raise OverlayContractError(f"Unknown overlay layer id: {value!r}.") from exc


def validate_layer_selection(layers: list[str] | tuple[str, ...] | list[LayerId]) -> tuple[LayerId, ...]:
    """Validate and normalize an explicit layer selection (opt-in only)."""

    if not layers:
        raise OverlayContractError("Overlay layer selection is empty (overlays are opt-in).")
    out: list[LayerId] = []
    seen: set[LayerId] = set()
    for raw in layers:
        layer = parse_layer_id(raw)
        if layer in seen:
            continue
        seen.add(layer)
        out.append(layer)
    return tuple(out)


def layers_for_hud_mode(mode: HudDirectionMode | str) -> tuple[LayerId, ...]:
    """Exclusive layer ids implied by a directional HUD mode."""

    normalized = (
        mode if isinstance(mode, HudDirectionMode) else HudDirectionMode(str(mode))
    )
    if normalized is HudDirectionMode.FOCAL_0_ONLY:
        return (LayerId.S8_EXCLUSIVE_A_TO_B,)
    if normalized is HudDirectionMode.FOCAL_1_ONLY:
        return (LayerId.S8_EXCLUSIVE_B_TO_A,)
    if normalized is HudDirectionMode.BOTH:
        return (LayerId.S8_EXCLUSIVE_A_TO_B, LayerId.S8_EXCLUSIVE_B_TO_A)
    raise OverlayContractError(f"Unsupported HUD direction mode: {normalized!r}.")


def default_behavioral_qc_layers(
    *,
    hud_mode: HudDirectionMode | str = HudDirectionMode.BOTH,
    include_contact: bool = True,
    include_sbs: bool = True,
    include_frame_counter: bool = True,
) -> tuple[LayerId, ...]:
    """Build the default C2-A behavioral QC layer selection."""

    layers: list[LayerId] = list(layers_for_hud_mode(hud_mode))
    if include_contact:
        layers.append(LayerId.S8_CONTACT)
    if include_sbs:
        layers.append(LayerId.S8_SBS)
    if include_frame_counter:
        layers.append(LayerId.DEBUG_FRAME_COUNTER)
    return validate_layer_selection(layers)


def default_s4_pose_qc_layers(
    *,
    include_frame_counter: bool = True,
) -> tuple[LayerId, ...]:
    """Build the default C2-B S4 finalized-pose QC layer selection."""

    layers: list[LayerId] = [LayerId.S4_POSE_FINAL]
    if include_frame_counter:
        layers.append(LayerId.DEBUG_FRAME_COUNTER)
    return validate_layer_selection(layers)


def default_s2_pose_qc_layers(
    *,
    include_frame_counter: bool = True,
) -> tuple[LayerId, ...]:
    """Build the default C3-1 S2 inferred-pose QC layer selection."""

    layers: list[LayerId] = [LayerId.S2_POSE_INFERRED]
    if include_frame_counter:
        layers.append(LayerId.DEBUG_FRAME_COUNTER)
    return validate_layer_selection(layers)


def s2_score_visual_intensity(score: float | None) -> float | None:
    """Map a SLEAP score to fill intensity in ``[0, 1]`` for drawing only.

    ``None`` means missing confidence (caller draws hollow ring). Numerical
    zero maps to intensity 0 (still drawn as a filled disc). Values outside
    the assumed ``[0, 1]`` range are clamped for visualization; originals are
    not mutated here.
    """

    if score is None:
        return None
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    if value != value:  # NaN
        return None
    span = float(S2_SCORE_VISUAL_RANGE_MAX - S2_SCORE_VISUAL_RANGE_MIN)
    if span <= 0:
        return 0.0
    normalized = (value - float(S2_SCORE_VISUAL_RANGE_MIN)) / span
    if normalized < 0.0:
        return 0.0
    if normalized > 1.0:
        return 1.0
    return float(normalized)


def s4_provenance_visual_class(code: int) -> S4ProvenanceVisualClass:
    """Map an exact S4 provenance code to its C2-B visual class.

    Unknown codes raise — never invent categories or confidence.
    """

    try:
        return S4_PROVENANCE_VISUAL_CLASS_BY_CODE[int(code)]
    except (KeyError, TypeError, ValueError) as exc:
        raise OverlayContractError(
            f"Unknown S4 provenance code {code!r}; refusing to invent a visual class."
        ) from exc
