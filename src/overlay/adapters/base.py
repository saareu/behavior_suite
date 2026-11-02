"""Read-only adapter contracts for scientific overlay sources."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from overlay.context import OverlayObservation
from overlay.contracts import BehaviorSourceKind, LayerId, OverlayAdapterError
from overlay.timing import PreparedFrameTiming


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Pinned source subsystem / run identity for provenance."""

    subsystem: str
    run_id: str | None
    run_dir: Path | None = None
    artifact_path: Path | None = None
    digest_sha256: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subsystem": self.subsystem,
            "run_id": self.run_id,
            "run_dir": None if self.run_dir is None else str(self.run_dir),
            "artifact_path": (
                None if self.artifact_path is None else str(self.artifact_path)
            ),
            "digest_sha256": self.digest_sha256,
            "extra": dict(self.extra),
        }


@runtime_checkable
class OverlayStageAdapter(Protocol):
    """Read-only stage adapter: load tables; never recompute science."""

    @property
    def layer_id(self) -> LayerId: ...

    @property
    def source_identity(self) -> SourceIdentity: ...

    @property
    def available(self) -> bool: ...

    def observe(
        self,
        timing: PreparedFrameTiming,
        *,
        prepared_frame_idx: int,
    ) -> OverlayObservation: ...


@dataclass(frozen=True, slots=True)
class UnavailableAdapter:
    """Explicit stub for stages deferred or missing in C1."""

    layer_id: LayerId
    source_identity: SourceIdentity
    reason: str = "Adapter not implemented for this increment."

    @property
    def available(self) -> bool:
        return False

    def observe(
        self,
        timing: PreparedFrameTiming,
        *,
        prepared_frame_idx: int,
    ) -> OverlayObservation:
        _ = timing, prepared_frame_idx
        return OverlayObservation(
            layer_id=self.layer_id,
            available=False,
            reason=self.reason,
            payload={},
        )


def require_columns(
    frame_columns: Iterable[object], required: set[str], *, label: str
) -> None:
    present = {str(c) for c in frame_columns}
    missing = sorted(required - present)
    if missing:
        raise OverlayAdapterError(
            f"{label} is missing required columns: {', '.join(missing)}"
        )


def behavior_source_kind_label(kind: BehaviorSourceKind) -> str:
    if kind is BehaviorSourceKind.S8_AUTOMATED:
        return "s8_automated"
    if kind is BehaviorSourceKind.S10_REVIEWED:
        return "s10_reviewed"
    raise OverlayAdapterError(f"Unknown behavior source kind: {kind!r}")
