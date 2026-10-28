"""Versioned Subsystem 08 configuration (scientific vs execution settings)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import yaml

from behavioral_arbitration.contracts import (
    SETTINGS_SCHEMA_VERSION,
    TARGET_FPS,
    ConfigError,
)


@dataclass(frozen=True, slots=True)
class PosturalApproachSettings:
    rearing_contraction_gt: float = 5.0
    behind_score_lt: float = -0.5
    cos_bearing_lt: float = -0.2
    smear_frames: int = 15
    escape_projected_vel_gt: float = 1.0

    def validated(self) -> PosturalApproachSettings:
        smear = int(self.smear_frames)
        if smear < 1:
            raise ConfigError("postural.approach.smear_frames must be >= 1")
        return replace(self, smear_frames=smear)


@dataclass(frozen=True, slots=True)
class PosturalRetractSettings:
    stationary_speed_lt: float = 15.0
    passive_effort_lt: float = 0.20
    passive_partner_speed_gt: float = 80.0
    flinch_contraction_gt: float = 3.0
    flinch_spine_vel_gt: float = -5.0
    smear_frames: int = 15
    escape_spine_vel_lt: float = -1.0
    escape_combined_vel_lt: float = -1.0
    escape_intent_lt: float = -1.0

    def validated(self) -> PosturalRetractSettings:
        smear = int(self.smear_frames)
        if smear < 1:
            raise ConfigError("postural.retract.smear_frames must be >= 1")
        return replace(self, smear_frames=smear)


@dataclass(frozen=True, slots=True)
class PosturalSettings:
    approach: PosturalApproachSettings = field(default_factory=PosturalApproachSettings)
    retract: PosturalRetractSettings = field(default_factory=PosturalRetractSettings)

    def validated(self) -> PosturalSettings:
        return replace(
            self,
            approach=self.approach.validated(),
            retract=self.retract.validated(),
        )


@dataclass(frozen=True, slots=True)
class HierarchySettings:
    layer1_significance_floor: float = 0.05
    approach_contact_penalty: float = 0.0


@dataclass(frozen=True, slots=True)
class TemporalSettings:
    gap_fill_seconds: float = 0.050
    min_bout_seconds: float = 0.100


@dataclass(frozen=True, slots=True)
class ContactPostProcessingSettings:
    smoothing_window: int = 3
    prob_threshold_high_intact: float = 0.9731
    prob_threshold_high_occluded: float = 0.4295
    prob_threshold_low: float = 0.4
    optical_assist_min: float = 0.30
    max_gap_to_fill: int = 0
    min_bout_duration: int = 6

    def validated(self) -> ContactPostProcessingSettings:
        return replace(
            self,
            smoothing_window=max(1, int(self.smoothing_window)),
            max_gap_to_fill=int(self.max_gap_to_fill),
            min_bout_duration=max(1, int(self.min_bout_duration)),
        )

    def to_settings_dict(self) -> dict[str, Any]:
        return {
            "smoothing_window": self.smoothing_window,
            "prob_threshold_high": {
                "intact": self.prob_threshold_high_intact,
                "occluded": self.prob_threshold_high_occluded,
            },
            "prob_threshold_low": self.prob_threshold_low,
            "optical_assist_min": self.optical_assist_min,
            "max_gap_to_fill": self.max_gap_to_fill,
            "min_bout_duration": self.min_bout_duration,
        }


@dataclass(frozen=True, slots=True)
class AdaptiveSettings:
    approach_engaged_distance: float = 3.0
    retract_engaged_distance_scale: float = 1.5
    strict_threshold: float = 0.5
    envelope_threshold: float = 0.25
    window_frames: int = 15


@dataclass(frozen=True, slots=True)
class RescueSettings:
    approach_prob_hint: float = 0.4
    retract_prob_hint: float = 0.4
    moving_intent_abs: float = 0.1
    moving_smooth_frames: int = 12


@dataclass(frozen=True, slots=True)
class DynamicStitchSettings:
    look_ahead_frames: int = 15
    hard_speed: float = 50.0
    hard_intent_factor: float = 2.0


@dataclass(frozen=True, slots=True)
class ScientificSettings:
    """All scientific thresholds / windows (Pipeline 100 parity values)."""

    decay_rate: float = 0.5
    conf_frames: int = 5
    engaged_min_threshold: float = 0.0469
    hard_gap_nose_to_nose_distance_norm: float | None = 0.3
    hard_gap_nose_to_body_distance_norm: float | None = 0.55
    hard_gap_nose_to_tail_distance_norm: float | None = 0.45
    hard_gap_side_by_side_distance_norm: float | None = 0.9
    hard_gap_contact_spine_to_spine_distance_norm: float | None = 1.2
    approach_hard_gap_nose_to_nose_distance_norm: float | None = 0.15
    approach_hard_gap_nose_to_body_distance_norm: float | None = 0.2
    approach_hard_gap_nose_to_tail_distance_norm: float | None = 0.15
    max_sniff_engaged_gap_frames: int = 180
    max_dynamic_engaged_gap_frames: int = 120
    max_dynamic_padding_frames: int = 120
    min_kinematic_intent_stitch: float = 0.5
    min_rolling_speed_stitch: float = 25.0
    case4_max_kinematic_intent_stitch: float = 1.0
    case4_max_rolling_speed_stitch: float = 200.0
    case4_initiation_kinematic_intent_stitch: float = 0.5
    case4_initiation_rolling_speed_stitch: float = 50.0
    max_idle_retract_gap_frames: int = 120
    far_apart_idle_distance: float = 3.5
    min_mutually_disengaged_frames: int = 120
    final_engaged_hysteresis_frames: int = 12
    contact_post_processing: ContactPostProcessingSettings = field(
        default_factory=ContactPostProcessingSettings
    )
    hierarchy: HierarchySettings = field(default_factory=HierarchySettings)
    temporal: TemporalSettings = field(default_factory=TemporalSettings)
    postural: PosturalSettings = field(default_factory=PosturalSettings)
    adaptive: AdaptiveSettings = field(default_factory=AdaptiveSettings)
    rescue: RescueSettings = field(default_factory=RescueSettings)
    stitch_dynamic: DynamicStitchSettings = field(default_factory=DynamicStitchSettings)

    def validated(self) -> ScientificSettings:
        conf = int(self.conf_frames)
        if conf < 1:
            raise ConfigError("scientific.conf_frames must be >= 1")
        return replace(
            self,
            conf_frames=conf,
            contact_post_processing=self.contact_post_processing.validated(),
            postural=self.postural.validated(),
        )

    def to_settings_dict(self) -> dict[str, Any]:
        return {
            "decay_rate": self.decay_rate,
            "conf_frames": self.conf_frames,
            "engaged_min_threshold": self.engaged_min_threshold,
            "hard_gap_nose_to_nose_distance_norm": (
                self.hard_gap_nose_to_nose_distance_norm
            ),
            "hard_gap_nose_to_body_distance_norm": (
                self.hard_gap_nose_to_body_distance_norm
            ),
            "hard_gap_nose_to_tail_distance_norm": (
                self.hard_gap_nose_to_tail_distance_norm
            ),
            "hard_gap_side_by_side_distance_norm": (
                self.hard_gap_side_by_side_distance_norm
            ),
            "hard_gap_contact_spine_to_spine_distance_norm": (
                self.hard_gap_contact_spine_to_spine_distance_norm
            ),
            "approach_hard_gap_nose_to_nose_distance_norm": (
                self.approach_hard_gap_nose_to_nose_distance_norm
            ),
            "approach_hard_gap_nose_to_body_distance_norm": (
                self.approach_hard_gap_nose_to_body_distance_norm
            ),
            "approach_hard_gap_nose_to_tail_distance_norm": (
                self.approach_hard_gap_nose_to_tail_distance_norm
            ),
            "max_sniff_engaged_gap_frames": self.max_sniff_engaged_gap_frames,
            "max_dynamic_engaged_gap_frames": self.max_dynamic_engaged_gap_frames,
            "max_dynamic_padding_frames": self.max_dynamic_padding_frames,
            "min_kinematic_intent_stitch": self.min_kinematic_intent_stitch,
            "min_rolling_speed_stitch": self.min_rolling_speed_stitch,
            "case4_max_kinematic_intent_stitch": self.case4_max_kinematic_intent_stitch,
            "case4_max_rolling_speed_stitch": self.case4_max_rolling_speed_stitch,
            "case4_initiation_kinematic_intent_stitch": (
                self.case4_initiation_kinematic_intent_stitch
            ),
            "case4_initiation_rolling_speed_stitch": (
                self.case4_initiation_rolling_speed_stitch
            ),
            "max_idle_retract_gap_frames": self.max_idle_retract_gap_frames,
            "far_apart_idle_distance": self.far_apart_idle_distance,
            "min_mutually_disengaged_frames": self.min_mutually_disengaged_frames,
            "final_engaged_hysteresis_frames": self.final_engaged_hysteresis_frames,
            "post_processing": {"contact": self.contact_post_processing.to_settings_dict()},
            "hierarchy": {
                "layer1_significance_floor": self.hierarchy.layer1_significance_floor,
                "approach_contact_penalty": self.hierarchy.approach_contact_penalty,
            },
            "temporal": {
                "gap_fill_seconds": self.temporal.gap_fill_seconds,
                "min_bout_seconds": self.temporal.min_bout_seconds,
            },
            "postural": {
                "approach": {
                    "rearing_contraction_gt": self.postural.approach.rearing_contraction_gt,
                    "behind_score_lt": self.postural.approach.behind_score_lt,
                    "cos_bearing_lt": self.postural.approach.cos_bearing_lt,
                    "smear_frames": self.postural.approach.smear_frames,
                    "escape_projected_vel_gt": (
                        self.postural.approach.escape_projected_vel_gt
                    ),
                },
                "retract": {
                    "stationary_speed_lt": self.postural.retract.stationary_speed_lt,
                    "passive_effort_lt": self.postural.retract.passive_effort_lt,
                    "passive_partner_speed_gt": (
                        self.postural.retract.passive_partner_speed_gt
                    ),
                    "flinch_contraction_gt": self.postural.retract.flinch_contraction_gt,
                    "flinch_spine_vel_gt": self.postural.retract.flinch_spine_vel_gt,
                    "smear_frames": self.postural.retract.smear_frames,
                    "escape_spine_vel_lt": self.postural.retract.escape_spine_vel_lt,
                    "escape_combined_vel_lt": (
                        self.postural.retract.escape_combined_vel_lt
                    ),
                    "escape_intent_lt": self.postural.retract.escape_intent_lt,
                },
            },
            "adaptive": {
                "approach_engaged_distance": self.adaptive.approach_engaged_distance,
                "retract_engaged_distance_scale": (
                    self.adaptive.retract_engaged_distance_scale
                ),
                "strict_threshold": self.adaptive.strict_threshold,
                "envelope_threshold": self.adaptive.envelope_threshold,
                "window_frames": self.adaptive.window_frames,
            },
            "rescue": {
                "approach_prob_hint": self.rescue.approach_prob_hint,
                "retract_prob_hint": self.rescue.retract_prob_hint,
                "moving_intent_abs": self.rescue.moving_intent_abs,
                "moving_smooth_frames": self.rescue.moving_smooth_frames,
            },
            "stitch": {
                "dynamic": {
                    "look_ahead_frames": self.stitch_dynamic.look_ahead_frames,
                    "hard_speed": self.stitch_dynamic.hard_speed,
                    "hard_intent_factor": self.stitch_dynamic.hard_intent_factor,
                }
            },
        }


@dataclass(frozen=True, slots=True)
class ExecutionSettings:
    emit_stage_diagnostics: bool = True
    emit_arbitration_diagnostics: bool = False
    generate_overlay_video: bool = False
    apply_pipeline_110_emit_filters: bool = False

    def validated(self) -> ExecutionSettings:
        if self.apply_pipeline_110_emit_filters:
            raise ConfigError(
                "execution.apply_pipeline_110_emit_filters must remain false in S8 "
                "(Pipeline 110 emit floors are downstream-only)."
            )
        return replace(
            self,
            emit_stage_diagnostics=bool(self.emit_stage_diagnostics),
            emit_arbitration_diagnostics=bool(self.emit_arbitration_diagnostics),
            generate_overlay_video=bool(self.generate_overlay_video),
            apply_pipeline_110_emit_filters=False,
        )

    def to_settings_dict(self) -> dict[str, Any]:
        return {
            "emit_stage_diagnostics": self.emit_stage_diagnostics,
            "emit_arbitration_diagnostics": self.emit_arbitration_diagnostics,
            "generate_overlay_video": self.generate_overlay_video,
            "apply_pipeline_110_emit_filters": self.apply_pipeline_110_emit_filters,
        }


@dataclass(frozen=True, slots=True)
class BehavioralArbitrationConfig:
    """Immutable S8 settings snapshot."""

    fps: float = TARGET_FPS
    require_exactly_two_mice: bool = True
    detector_coverage_profile: str = "full_ethogram"
    working_thresholds: dict[str, Any] = field(default_factory=dict)
    scientific: ScientificSettings = field(default_factory=ScientificSettings)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)

    def validated(self) -> BehavioralArbitrationConfig:
        fps = float(self.fps)
        if abs(fps - TARGET_FPS) > 1e-9:
            raise ConfigError(
                f"S8 MVP requires fps={TARGET_FPS}; got {fps}."
            )
        profile = str(self.detector_coverage_profile).strip() or "full_ethogram"
        if profile != "full_ethogram":
            raise ConfigError(
                f"Unsupported detector_coverage_profile '{profile}'. "
                "S8 MVP supports only 'full_ethogram'."
            )
        if not self.require_exactly_two_mice:
            raise ConfigError(
                "S8 MVP requires require_exactly_two_mice=true (K=2)."
            )
        return replace(
            self,
            fps=fps,
            require_exactly_two_mice=True,
            detector_coverage_profile=profile,
            working_thresholds=dict(self.working_thresholds),
            scientific=self.scientific.validated(),
            execution=self.execution.validated(),
        )

    def to_settings_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SETTINGS_SCHEMA_VERSION,
            "fps": self.fps,
            "require_exactly_two_mice": self.require_exactly_two_mice,
            "detector_coverage_profile": self.detector_coverage_profile,
            "working_thresholds": dict(self.working_thresholds),
            "scientific": self.scientific.to_settings_dict(),
            "execution": self.execution.to_settings_dict(),
        }


def default_suite_config_path() -> Path:
    """Return the tracked default YAML path under configs/subsystem_08/."""

    return (
        Path(__file__).resolve().parents[2]
        / "configs"
        / "subsystem_08"
        / "behavioral_arbitration_default.yaml"
    )


def load_behavioral_arbitration_config(
    path: Path | str | None = None,
) -> BehavioralArbitrationConfig:
    """Load S8 settings from YAML, or the suite default when ``path`` is None."""

    file_path = (
        default_suite_config_path()
        if path is None
        else Path(path).expanduser().resolve()
    )
    if not file_path.is_file():
        raise ConfigError(f"S8 config YAML is missing: {file_path}")
    try:
        with file_path.open("r", encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"Could not load S8 config: {file_path}: {exc}") from exc
    if payload is None:
        return BehavioralArbitrationConfig().validated()
    if not isinstance(payload, Mapping):
        raise ConfigError(f"S8 config root must be a mapping: {file_path}")
    return config_from_mapping(payload)


def config_from_mapping(payload: Mapping[str, Any]) -> BehavioralArbitrationConfig:
    """Build a validated config from a settings mapping."""

    scientific_raw = payload.get("scientific", payload)
    if not isinstance(scientific_raw, Mapping):
        raise ConfigError("scientific settings must be a mapping when provided.")
    execution_raw = payload.get("execution", {})
    if execution_raw is None:
        execution_raw = {}
    if not isinstance(execution_raw, Mapping):
        raise ConfigError("execution settings must be a mapping when provided.")

    thresholds = payload.get("working_thresholds", {})
    if thresholds is None:
        thresholds = {}
    if not isinstance(thresholds, Mapping):
        raise ConfigError("working_thresholds must be a mapping when provided.")

    return BehavioralArbitrationConfig(
        fps=float(payload.get("fps", TARGET_FPS)),
        require_exactly_two_mice=bool(payload.get("require_exactly_two_mice", True)),
        detector_coverage_profile=str(
            payload.get("detector_coverage_profile", "full_ethogram")
        ),
        working_thresholds=dict(thresholds),
        scientific=_scientific_from_mapping(scientific_raw),
        execution=ExecutionSettings(
            emit_stage_diagnostics=bool(
                execution_raw.get("emit_stage_diagnostics", True)
            ),
            emit_arbitration_diagnostics=bool(
                execution_raw.get("emit_arbitration_diagnostics", False)
            ),
            generate_overlay_video=bool(
                execution_raw.get("generate_overlay_video", False)
            ),
            apply_pipeline_110_emit_filters=bool(
                execution_raw.get("apply_pipeline_110_emit_filters", False)
            ),
        ),
    ).validated()


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _scientific_from_mapping(raw: Mapping[str, Any]) -> ScientificSettings:
    contact_raw = {}
    post = raw.get("post_processing")
    if isinstance(post, Mapping):
        contact = post.get("contact")
        if isinstance(contact, Mapping):
            contact_raw = dict(contact)
    high = contact_raw.get("prob_threshold_high", {})
    if not isinstance(high, Mapping):
        high = {}

    postural_raw = raw.get("postural", {})
    if not isinstance(postural_raw, Mapping):
        postural_raw = {}
    approach_raw = postural_raw.get("approach", {})
    retract_raw = postural_raw.get("retract", {})
    if not isinstance(approach_raw, Mapping):
        approach_raw = {}
    if not isinstance(retract_raw, Mapping):
        retract_raw = {}

    hierarchy_raw = raw.get("hierarchy", {})
    if not isinstance(hierarchy_raw, Mapping):
        hierarchy_raw = {}
    temporal_raw = raw.get("temporal", {})
    if not isinstance(temporal_raw, Mapping):
        temporal_raw = {}
    adaptive_raw = raw.get("adaptive", {})
    if not isinstance(adaptive_raw, Mapping):
        adaptive_raw = {}
    rescue_raw = raw.get("rescue", {})
    if not isinstance(rescue_raw, Mapping):
        rescue_raw = {}
    stitch_raw = raw.get("stitch", {})
    if not isinstance(stitch_raw, Mapping):
        stitch_raw = {}
    dynamic_raw = stitch_raw.get("dynamic", {}) if isinstance(stitch_raw, Mapping) else {}
    if not isinstance(dynamic_raw, Mapping):
        dynamic_raw = {}

    return ScientificSettings(
        decay_rate=float(raw.get("decay_rate", 0.5)),
        conf_frames=int(raw.get("conf_frames", 5)),
        engaged_min_threshold=float(raw.get("engaged_min_threshold", 0.0469)),
        hard_gap_nose_to_nose_distance_norm=_optional_float(
            raw.get("hard_gap_nose_to_nose_distance_norm", 0.3)
        ),
        hard_gap_nose_to_body_distance_norm=_optional_float(
            raw.get("hard_gap_nose_to_body_distance_norm", 0.55)
        ),
        hard_gap_nose_to_tail_distance_norm=_optional_float(
            raw.get("hard_gap_nose_to_tail_distance_norm", 0.45)
        ),
        hard_gap_side_by_side_distance_norm=_optional_float(
            raw.get("hard_gap_side_by_side_distance_norm", 0.9)
        ),
        hard_gap_contact_spine_to_spine_distance_norm=_optional_float(
            raw.get("hard_gap_contact_spine_to_spine_distance_norm", 1.2)
        ),
        approach_hard_gap_nose_to_nose_distance_norm=_optional_float(
            raw.get("approach_hard_gap_nose_to_nose_distance_norm", 0.15)
        ),
        approach_hard_gap_nose_to_body_distance_norm=_optional_float(
            raw.get("approach_hard_gap_nose_to_body_distance_norm", 0.2)
        ),
        approach_hard_gap_nose_to_tail_distance_norm=_optional_float(
            raw.get("approach_hard_gap_nose_to_tail_distance_norm", 0.15)
        ),
        max_sniff_engaged_gap_frames=int(raw.get("max_sniff_engaged_gap_frames", 180)),
        max_dynamic_engaged_gap_frames=int(
            raw.get("max_dynamic_engaged_gap_frames", 120)
        ),
        max_dynamic_padding_frames=int(raw.get("max_dynamic_padding_frames", 120)),
        min_kinematic_intent_stitch=float(raw.get("min_kinematic_intent_stitch", 0.5)),
        min_rolling_speed_stitch=float(raw.get("min_rolling_speed_stitch", 25.0)),
        case4_max_kinematic_intent_stitch=float(
            raw.get("case4_max_kinematic_intent_stitch", 1.0)
        ),
        case4_max_rolling_speed_stitch=float(
            raw.get("case4_max_rolling_speed_stitch", 200.0)
        ),
        case4_initiation_kinematic_intent_stitch=float(
            raw.get("case4_initiation_kinematic_intent_stitch", 0.5)
        ),
        case4_initiation_rolling_speed_stitch=float(
            raw.get("case4_initiation_rolling_speed_stitch", 50.0)
        ),
        max_idle_retract_gap_frames=int(raw.get("max_idle_retract_gap_frames", 120)),
        far_apart_idle_distance=float(raw.get("far_apart_idle_distance", 3.5)),
        min_mutually_disengaged_frames=int(
            raw.get("min_mutually_disengaged_frames", 120)
        ),
        final_engaged_hysteresis_frames=int(
            raw.get("final_engaged_hysteresis_frames", 12)
        ),
        contact_post_processing=ContactPostProcessingSettings(
            smoothing_window=int(contact_raw.get("smoothing_window", 3)),
            prob_threshold_high_intact=float(high.get("intact", 0.9731)),
            prob_threshold_high_occluded=float(high.get("occluded", 0.4295)),
            prob_threshold_low=float(contact_raw.get("prob_threshold_low", 0.4)),
            optical_assist_min=float(contact_raw.get("optical_assist_min", 0.30)),
            max_gap_to_fill=int(contact_raw.get("max_gap_to_fill", 0)),
            min_bout_duration=int(contact_raw.get("min_bout_duration", 6)),
        ),
        hierarchy=HierarchySettings(
            layer1_significance_floor=float(
                hierarchy_raw.get("layer1_significance_floor", 0.05)
            ),
            approach_contact_penalty=float(
                hierarchy_raw.get("approach_contact_penalty", 0.0)
            ),
        ),
        temporal=TemporalSettings(
            gap_fill_seconds=float(temporal_raw.get("gap_fill_seconds", 0.050)),
            min_bout_seconds=float(temporal_raw.get("min_bout_seconds", 0.100)),
        ),
        postural=PosturalSettings(
            approach=PosturalApproachSettings(
                rearing_contraction_gt=float(
                    approach_raw.get("rearing_contraction_gt", 5.0)
                ),
                behind_score_lt=float(approach_raw.get("behind_score_lt", -0.5)),
                cos_bearing_lt=float(approach_raw.get("cos_bearing_lt", -0.2)),
                smear_frames=int(approach_raw.get("smear_frames", 15)),
                escape_projected_vel_gt=float(
                    approach_raw.get("escape_projected_vel_gt", 1.0)
                ),
            ),
            retract=PosturalRetractSettings(
                stationary_speed_lt=float(
                    retract_raw.get("stationary_speed_lt", 15.0)
                ),
                passive_effort_lt=float(retract_raw.get("passive_effort_lt", 0.20)),
                passive_partner_speed_gt=float(
                    retract_raw.get("passive_partner_speed_gt", 80.0)
                ),
                flinch_contraction_gt=float(
                    retract_raw.get("flinch_contraction_gt", 3.0)
                ),
                flinch_spine_vel_gt=float(
                    retract_raw.get("flinch_spine_vel_gt", -5.0)
                ),
                smear_frames=int(retract_raw.get("smear_frames", 15)),
                escape_spine_vel_lt=float(
                    retract_raw.get("escape_spine_vel_lt", -1.0)
                ),
                escape_combined_vel_lt=float(
                    retract_raw.get("escape_combined_vel_lt", -1.0)
                ),
                escape_intent_lt=float(retract_raw.get("escape_intent_lt", -1.0)),
            ),
        ),
        adaptive=AdaptiveSettings(
            approach_engaged_distance=float(
                adaptive_raw.get("approach_engaged_distance", 3.0)
            ),
            retract_engaged_distance_scale=float(
                adaptive_raw.get("retract_engaged_distance_scale", 1.5)
            ),
            strict_threshold=float(adaptive_raw.get("strict_threshold", 0.5)),
            envelope_threshold=float(adaptive_raw.get("envelope_threshold", 0.25)),
            window_frames=int(adaptive_raw.get("window_frames", 15)),
        ),
        rescue=RescueSettings(
            approach_prob_hint=float(rescue_raw.get("approach_prob_hint", 0.4)),
            retract_prob_hint=float(rescue_raw.get("retract_prob_hint", 0.4)),
            moving_intent_abs=float(rescue_raw.get("moving_intent_abs", 0.1)),
            moving_smooth_frames=int(rescue_raw.get("moving_smooth_frames", 12)),
        ),
        stitch_dynamic=DynamicStitchSettings(
            look_ahead_frames=int(dynamic_raw.get("look_ahead_frames", 15)),
            hard_speed=float(dynamic_raw.get("hard_speed", 50)),
            hard_intent_factor=float(dynamic_raw.get("hard_intent_factor", 2.0)),
        ),
    )
