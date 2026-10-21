"""Subsystem 05 configuration: segmentation vs execution settings.

Scientific defaults follow lab-tuned S5 values (minimum areas 2900). Suite YAML
lives at ``configs/subsystem_05/blob_features_default.yaml``. The legacy lab YAML
remains loadable for historical comparison but is not the active suite default.

Cable-artifact removal is excluded from the active S5 configuration. This
assumes the legacy lab default ``remove_cable_artifact: false``. Cable-related
keys present in a loaded YAML are rejected rather than silently applied.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import yaml

from blob_features.contracts import (
    CABLE_REMOVAL_ASSUMED_LEGACY_DEFAULT,
    CABLE_REMOVAL_EXCLUDED,
    CC_COUNT_GAP_FILL_MAX_GAP_DEFAULT,
    SETTINGS_SCHEMA_VERSION,
    TARGET_FPS_DEFAULT,
    BlobFeaturesError,
)

# Keys retained from legacy lab YAML for per-frame segmentation.
_SEGMENTATION_KEYS = (
    "blob_threshold",
    "blob_blur_kernel_size",
    "blob_kernel_size",
    "blob_tail_open_kernel_size",
    "blob_min_area",
    "blob_max_area",
    "blob_prune_min_radius",
    "blob_prune_regrow_iters",
    "blob_prune_max_regrow_dist",
    "blob_post_prune_min_area",
    "blob_post_prune_max_area",
    "blob_final_smooth_kernel_size",
    "abs_dark_threshold",
)

_EXECUTION_KEYS = (
    "n_workers",
    "max_in_flight",
)

_TEMPORAL_KEYS = (
    "cc_count_gap_fill_max_gap",
)

# Excluded from S5 MVP (legacy lab default is false; module also missing in-tree).
_EXCLUDED_CABLE_KEYS = (
    "remove_cable_artifact",
    "cable_tophat_kernel_size",
    "cable_open_kernel_size",
    "cable_close_kernel_size",
    "cable_dilate_kernel_size",
    "cable_min_area",
    "cable_max_area",
    "cable_inpaint_radius",
)

@dataclass(frozen=True, slots=True)
class BlobSegmentationSettings:
    """Per-frame foreground segmentation parameters (legacy scientific defaults)."""

    blob_threshold: int = -2
    blob_blur_kernel_size: int = 9
    blob_kernel_size: int = 27
    blob_tail_open_kernel_size: int = 7
    blob_min_area: int = 2900
    blob_max_area: int = 0
    blob_prune_min_radius: int = 10
    blob_prune_regrow_iters: int = 7
    blob_prune_max_regrow_dist: int = 8
    blob_post_prune_min_area: int = 2900
    blob_post_prune_max_area: int = 0
    blob_final_smooth_kernel_size: int = 3
    abs_dark_threshold: int = 10

    def validated(self) -> BlobSegmentationSettings:
        """Return a copy after coercing legacy-compatible bounds (no silent science changes)."""

        threshold = int(self.blob_threshold)
        if threshold < -2:
            raise BlobFeaturesError(
                "blob_threshold must be >= -2 (-2=Multi-Otsu, -1=Otsu, >=0 fixed)."
            )

        blur = max(1, int(self.blob_blur_kernel_size))
        if blur % 2 == 0:
            raise BlobFeaturesError(
                f"blob_blur_kernel_size must be odd (legacy Gaussian kernel); got {blur}."
            )

        kernel = max(1, int(self.blob_kernel_size))
        tail = max(1, int(self.blob_tail_open_kernel_size))
        min_area = max(1, int(self.blob_min_area))
        max_area = max(0, int(self.blob_max_area))
        prune_radius = max(1, int(self.blob_prune_min_radius))
        prune_iters = max(0, int(self.blob_prune_regrow_iters))
        prune_dist = max(0, int(self.blob_prune_max_regrow_dist))
        post_min = max(1, int(self.blob_post_prune_min_area))
        post_max = max(0, int(self.blob_post_prune_max_area))
        smooth = max(1, int(self.blob_final_smooth_kernel_size))
        abs_dark = int(self.abs_dark_threshold)
        if abs_dark < 0:
            raise BlobFeaturesError("abs_dark_threshold must be >= 0.")

        return replace(
            self,
            blob_threshold=threshold,
            blob_blur_kernel_size=blur,
            blob_kernel_size=kernel,
            blob_tail_open_kernel_size=tail,
            blob_min_area=min_area,
            blob_max_area=max_area,
            blob_prune_min_radius=prune_radius,
            blob_prune_regrow_iters=prune_iters,
            blob_prune_max_regrow_dist=prune_dist,
            blob_post_prune_min_area=post_min,
            blob_post_prune_max_area=post_max,
            blob_final_smooth_kernel_size=smooth,
            abs_dark_threshold=abs_dark,
        )


@dataclass(frozen=True, slots=True)
class BlobExecutionSettings:
    """Runtime parallelism knobs (do not affect scientific segmentation)."""

    n_workers: int = 16
    max_in_flight: int = 128

    def validated(self) -> BlobExecutionSettings:
        workers = max(1, int(self.n_workers))
        inflight = max(1, int(self.max_in_flight))
        return replace(self, n_workers=workers, max_in_flight=inflight)


@dataclass(frozen=True, slots=True)
class BlobTemporalSettings:
    """Source-rate temporal post-processing (before 120 Hz mapping)."""

    cc_count_gap_fill_max_gap: int = CC_COUNT_GAP_FILL_MAX_GAP_DEFAULT

    def validated(self) -> BlobTemporalSettings:
        return replace(
            self,
            cc_count_gap_fill_max_gap=max(0, int(self.cc_count_gap_fill_max_gap)),
        )


@dataclass(frozen=True, slots=True)
class BlobFeaturesConfig:
    """Combined S5 settings with processing and execution kept separate."""

    segmentation: BlobSegmentationSettings = field(default_factory=BlobSegmentationSettings)
    temporal: BlobTemporalSettings = field(default_factory=BlobTemporalSettings)
    execution: BlobExecutionSettings = field(default_factory=BlobExecutionSettings)
    target_fps: float = TARGET_FPS_DEFAULT
    schema_version: str = SETTINGS_SCHEMA_VERSION
    cable_removal_excluded: bool = CABLE_REMOVAL_EXCLUDED
    cable_removal_assumed_legacy_default: bool = CABLE_REMOVAL_ASSUMED_LEGACY_DEFAULT

    def validated(self) -> BlobFeaturesConfig:
        fps = float(self.target_fps)
        if not math.isfinite(fps) or fps <= 0.0:
            raise BlobFeaturesError("target_fps must be finite and positive.")
        if abs(fps - TARGET_FPS_DEFAULT) > 1e-9:
            raise BlobFeaturesError(
                f"S5 target_fps is fixed at {TARGET_FPS_DEFAULT} (got {fps})."
            )
        return replace(
            self,
            segmentation=self.segmentation.validated(),
            temporal=self.temporal.validated(),
            execution=self.execution.validated(),
            target_fps=TARGET_FPS_DEFAULT,
        )

    def to_settings_dict(self) -> dict[str, Any]:
        """Serialize for ``settings_used.yaml``."""

        return {
            "schema_version": self.schema_version,
            "segmentation": asdict(self.segmentation),
            "temporal": asdict(self.temporal),
            "execution": asdict(self.execution),
            "target_fps": float(self.target_fps),
            "cable_removal": {
                "excluded_from_s5_mvp": self.cable_removal_excluded,
                "assumed_legacy_default": self.cable_removal_assumed_legacy_default,
                "note": (
                    "Cable-artifact removal is out of S5 MVP scope. Processing "
                    "assumes the legacy lab default remove_cable_artifact=false."
                ),
            },
        }


def default_suite_config_path() -> Path:
    """Path to the suite YAML seeded from the legacy lab defaults."""

    # src/blob_features/config.py → repo root is parents[2]
    return (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "subsystem_05"
        / "blob_features_default.yaml"
    )


def legacy_lab_config_path() -> Path:
    """Path to the authoritative legacy lab YAML (reference / optional load)."""

    return (
        Path(__file__).resolve().parents[2]
        / "legacy"
        / "configs"
        / "blob_features.yaml"
    )


def load_blob_features_config(path: Path | str | None = None) -> BlobFeaturesConfig:
    """Load config from YAML, or return lab-equivalent dataclass defaults.

    When ``path`` is ``None``, prefers the suite YAML if present; otherwise uses
    frozen dataclass defaults (identical lab values).
    """

    if path is None:
        suite_path = default_suite_config_path()
        if suite_path.is_file():
            return load_blob_features_config(suite_path)
        return BlobFeaturesConfig().validated()

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise BlobFeaturesError(f"Blob features config file not found: {config_path}")

    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise BlobFeaturesError(f"Could not load blob features config: {exc}") from exc

    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise BlobFeaturesError("Blob features config root must be a mapping.")

    # Legacy file nests under ``blob_features:``; suite file may be flat or nested.
    if "blob_features" in raw and isinstance(raw["blob_features"], Mapping):
        payload = dict(raw["blob_features"])
    elif "segmentation" in raw or "execution" in raw:
        return _config_from_structured(raw).validated()
    else:
        payload = dict(raw)

    _drop_excluded_cable_keys(payload, source=config_path)
    _enforce_fixed_target_fps(payload, source=config_path)

    seg_kwargs = {k: payload[k] for k in _SEGMENTATION_KEYS if k in payload}
    temporal_kwargs = {k: payload[k] for k in _TEMPORAL_KEYS if k in payload}
    exec_kwargs = {k: payload[k] for k in _EXECUTION_KEYS if k in payload}
    known = set(_SEGMENTATION_KEYS) | set(_TEMPORAL_KEYS) | set(_EXECUTION_KEYS)
    unknown = set(payload) - known
    if unknown:
        raise BlobFeaturesError(
            "Unknown blob features config keys: " + ", ".join(sorted(unknown))
        )

    return BlobFeaturesConfig(
        segmentation=BlobSegmentationSettings(**seg_kwargs),
        temporal=BlobTemporalSettings(**temporal_kwargs),
        execution=BlobExecutionSettings(**exec_kwargs),
    ).validated()


def _config_from_structured(raw: Mapping[str, Any]) -> BlobFeaturesConfig:
    seg_raw = raw.get("segmentation", {})
    temporal_raw = raw.get("temporal", {})
    exec_raw = raw.get("execution", {})
    if not isinstance(seg_raw, Mapping) or not isinstance(exec_raw, Mapping):
        raise BlobFeaturesError(
            "Structured config requires mapping 'segmentation' and 'execution' blocks."
        )
    if temporal_raw and not isinstance(temporal_raw, Mapping):
        raise BlobFeaturesError("Structured config 'temporal' block must be a mapping.")
    seg_payload = dict(seg_raw)
    temporal_payload = dict(temporal_raw) if isinstance(temporal_raw, Mapping) else {}
    exec_payload = dict(exec_raw)
    _drop_excluded_cable_keys(seg_payload, source="segmentation")
    _drop_excluded_cable_keys(exec_payload, source="execution")

    if "target_fps" in raw:
        _enforce_fixed_target_fps({"target_fps": raw["target_fps"]}, source="structured")

    seg_fields = {f.name for f in fields(BlobSegmentationSettings)}
    temporal_fields = {f.name for f in fields(BlobTemporalSettings)}
    exec_fields = {f.name for f in fields(BlobExecutionSettings)}
    unknown_seg = set(seg_payload) - seg_fields
    unknown_temporal = set(temporal_payload) - temporal_fields
    unknown_exec = set(exec_payload) - exec_fields
    if unknown_seg or unknown_temporal or unknown_exec:
        raise BlobFeaturesError(
            "Unknown structured config keys: "
            + ", ".join(sorted(unknown_seg | unknown_temporal | unknown_exec))
        )
    return BlobFeaturesConfig(
        segmentation=BlobSegmentationSettings(**seg_payload),
        temporal=BlobTemporalSettings(**temporal_payload),
        execution=BlobExecutionSettings(**exec_payload),
        schema_version=str(raw.get("schema_version", SETTINGS_SCHEMA_VERSION)),
        target_fps=float(raw.get("target_fps", TARGET_FPS_DEFAULT)),
    )


def _enforce_fixed_target_fps(payload: dict[str, Any], *, source: object) -> None:
    """Reject non-120 target_fps overrides; drop the key after validation."""

    if "target_fps" not in payload:
        return
    fps = float(payload.pop("target_fps"))
    if not math.isfinite(fps) or abs(fps - TARGET_FPS_DEFAULT) > 1e-9:
        raise BlobFeaturesError(
            f"S5 target_fps is fixed at {TARGET_FPS_DEFAULT} "
            f"(got {fps} from {source})."
        )


def _drop_excluded_cable_keys(payload: dict[str, Any], *, source: object) -> None:
    """Remove cable keys after enforcing the assumed legacy default of false.

    Enabling ``remove_cable_artifact`` is rejected: the helper module is not in
    this checkout and cable cleanup is out of S5 MVP scope.
    """

    if "remove_cable_artifact" in payload:
        enabled = bool(payload["remove_cable_artifact"])
        if enabled != CABLE_REMOVAL_ASSUMED_LEGACY_DEFAULT:
            raise BlobFeaturesError(
                "Cable-artifact removal is excluded from the S5 MVP "
                f"(assumes legacy default remove_cable_artifact="
                f"{CABLE_REMOVAL_ASSUMED_LEGACY_DEFAULT}). "
                f"Refusing enabled cable removal from {source}."
            )
    for key in _EXCLUDED_CABLE_KEYS:
        payload.pop(key, None)
