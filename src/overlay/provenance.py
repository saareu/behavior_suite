"""``overlay_artifact_manifest_v1`` construction and serialization.

No scientific schema changes — this is overlay-only provenance.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from behavioral_review.sync import SyncMap
from overlay.adapters.base import SourceIdentity
from overlay.compositor import ComposedFrame
from overlay.contracts import (
    OVERLAY_ARTIFACT_MANIFEST_SCHEMA,
    OVERLAY_RENDERER_VERSION,
    FrameDomain,
    LayerId,
    OverlayContractError,
    PlateauPolicy,
)
from overlay.style import DrawConfig
from overlay.timing import PreparedFrameTiming, timing_mapping_identity


@dataclass(frozen=True, slots=True)
class PreparedVideoIdentity:
    """Prepared-video identity pinned into the overlay manifest."""

    path: Path | None
    digest_sha256: str | None = None
    fps: float | None = None
    frame_count: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": None if self.path is None else str(self.path),
            "digest_sha256": self.digest_sha256,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "extra": dict(self.extra),
        }


@dataclass(frozen=True, slots=True)
class OverlayArtifactManifest:
    """Serializable overlay provenance document."""

    schema_version: str
    renderer_version: str
    source_runs: tuple[dict[str, Any], ...]
    source_digests: dict[str, str | None]
    prepared_video: dict[str, Any]
    timing_mapping: dict[str, Any]
    plateau_policy: str
    layer_selections: tuple[str, ...]
    unavailable_layers: tuple[str, ...]
    output_frame_domain: str
    output_fps: float | None
    render_settings: dict[str, Any]
    pixel_format: str
    notes: tuple[str, ...] = ()
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "renderer_version": self.renderer_version,
            "source_runs": list(self.source_runs),
            "source_digests": dict(self.source_digests),
            "prepared_video": dict(self.prepared_video),
            "timing_mapping": dict(self.timing_mapping),
            "plateau_policy": self.plateau_policy,
            "layer_selections": list(self.layer_selections),
            "unavailable_layers": list(self.unavailable_layers),
            "output_frame_domain": self.output_frame_domain,
            "output_fps": self.output_fps,
            "render_settings": dict(self.render_settings),
            "pixel_format": self.pixel_format,
            "notes": list(self.notes),
            "extra": dict(self.extra),
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def write_json(self, path: Path | str, *, indent: int = 2) -> Path:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(self.to_json(indent=indent) + "\n", encoding="utf-8")
        return out


def build_overlay_artifact_manifest(
    *,
    sync: SyncMap,
    prepared_video: PreparedVideoIdentity,
    source_identities: Sequence[SourceIdentity],
    selected_layers: Sequence[LayerId | str],
    plateau_policy: PlateauPolicy,
    style: DrawConfig,
    composed: ComposedFrame | None = None,
    timing: PreparedFrameTiming | None = None,
    output_frame_domain: FrameDomain = FrameDomain.PREPARED,
    output_fps: float | None = None,
    notes: Sequence[str] = (),
    extra: Mapping[str, Any] | None = None,
) -> OverlayArtifactManifest:
    """Build ``overlay_artifact_manifest_v1`` from composition inputs."""

    layers = tuple(
        x.value if isinstance(x, LayerId) else str(x) for x in selected_layers
    )
    if not layers:
        raise OverlayContractError("Manifest requires non-empty layer selections.")

    source_runs = tuple(s.to_dict() for s in source_identities)
    digests: dict[str, str | None] = {}
    for src in source_identities:
        key = src.run_id or src.subsystem
        if src.artifact_path is not None:
            key = f"{key}:{Path(src.artifact_path).name}"
        digests[key] = src.digest_sha256

    unavailable: tuple[str, ...] = ()
    if composed is not None:
        unavailable = tuple(x.value for x in composed.unavailable_layers)

    fps = output_fps
    if fps is None:
        fps = prepared_video.fps

    timing_id = timing_mapping_identity(sync)
    if timing is not None:
        timing_id = {
            **timing_id,
            "example_prepared_frame_idx": timing.prepared_frame_idx,
            "example_dense_sample_idx": timing.dense_sample_idx,
            "example_plateau_policy": timing.plateau_policy.value,
        }

    return OverlayArtifactManifest(
        schema_version=OVERLAY_ARTIFACT_MANIFEST_SCHEMA,
        renderer_version=OVERLAY_RENDERER_VERSION,
        source_runs=source_runs,
        source_digests=digests,
        prepared_video=prepared_video.to_dict(),
        timing_mapping=timing_id,
        plateau_policy=plateau_policy.value,
        layer_selections=layers,
        unavailable_layers=unavailable,
        output_frame_domain=output_frame_domain.value,
        output_fps=fps,
        render_settings=style.to_dict(),
        pixel_format=style.pixel_format,
        notes=tuple(notes),
        extra=dict(extra or {}),
    )


def parse_overlay_artifact_manifest(data: Mapping[str, Any]) -> OverlayArtifactManifest:
    """Validate minimal required keys and rebuild a manifest object."""

    required = (
        "schema_version",
        "renderer_version",
        "source_runs",
        "source_digests",
        "prepared_video",
        "timing_mapping",
        "plateau_policy",
        "layer_selections",
        "output_frame_domain",
        "render_settings",
    )
    missing = [k for k in required if k not in data]
    if missing:
        raise OverlayContractError(
            f"overlay manifest missing keys: {', '.join(missing)}"
        )
    if data["schema_version"] != OVERLAY_ARTIFACT_MANIFEST_SCHEMA:
        raise OverlayContractError(
            f"Unsupported overlay manifest schema: {data['schema_version']!r}"
        )
    return OverlayArtifactManifest(
        schema_version=str(data["schema_version"]),
        renderer_version=str(data["renderer_version"]),
        source_runs=tuple(dict(x) for x in data["source_runs"]),
        source_digests={str(k): v for k, v in dict(data["source_digests"]).items()},
        prepared_video=dict(data["prepared_video"]),
        timing_mapping=dict(data["timing_mapping"]),
        plateau_policy=str(data["plateau_policy"]),
        layer_selections=tuple(str(x) for x in data["layer_selections"]),
        unavailable_layers=tuple(str(x) for x in data.get("unavailable_layers", ())),
        output_frame_domain=str(data["output_frame_domain"]),
        output_fps=(
            None if data.get("output_fps") is None else float(data["output_fps"])
        ),
        render_settings=dict(data["render_settings"]),
        pixel_format=str(data.get("pixel_format") or "bgr"),
        notes=tuple(str(x) for x in data.get("notes", ())),
        extra=dict(data.get("extra") or {}),
    )
