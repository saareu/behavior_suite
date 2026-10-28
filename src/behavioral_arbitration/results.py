"""In-memory Sprint 1–3 arbitration result contracts (+ published run summary)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from behavioral_arbitration.contracts import (
    SPRINT1_RESULT_SCHEMA_VERSION,
    SPRINT2_RESULT_SCHEMA_VERSION,
    SPRINT3_RESULT_SCHEMA_VERSION,
)


@dataclass(frozen=True, slots=True)
class StageDiagnostic:
    """Compact per-stage change record."""

    stage: str
    rule_code: str
    detector: str | None
    n_rows_changed: int
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "rule_code": self.rule_code,
            "detector": self.detector,
            "n_rows_changed": int(self.n_rows_changed),
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class Sprint1ArbitrationResult:
    """Headless Sprint 1 output: directional frames after initial gates.

    Does **not** include exclusive ``behavior_label`` or bout tables.
    Stable enough for Sprint 2 hierarchy / temporal stages to continue from
    ``frames`` + ``diagnostics`` + provenance fields.
    """

    frames: pd.DataFrame
    diagnostics: tuple[StageDiagnostic, ...]
    source_s7_run_id: str
    source_s7_run_dir: Path
    source_s6_run_id: str
    source_s6_run_dir: Path
    session_root: Path
    exp_name: str
    fps: float
    n_frames: int
    n_directional_rows: int
    mouse_ids: tuple[int, ...]
    detectors_dyadic: tuple[str, ...]
    detectors_focal: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    settings_used: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SPRINT1_RESULT_SCHEMA_VERSION

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source_s7_run_id": self.source_s7_run_id,
            "source_s7_run_dir": str(self.source_s7_run_dir),
            "source_s6_run_id": self.source_s6_run_id,
            "source_s6_run_dir": str(self.source_s6_run_dir),
            "session_root": str(self.session_root),
            "exp_name": self.exp_name,
            "fps": self.fps,
            "n_frames": self.n_frames,
            "n_directional_rows": self.n_directional_rows,
            "mouse_ids": list(self.mouse_ids),
            "detectors_dyadic": list(self.detectors_dyadic),
            "detectors_focal": list(self.detectors_focal),
            "warnings": list(self.warnings),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "settings_used": dict(self.settings_used),
        }


@dataclass(frozen=True, slots=True)
class Sprint2ArbitrationResult:
    """Headless Sprint 2 output after hierarchy / adaptive / contact / pass-2.

    Includes provisional ``behavior_label`` (engaged residual or unresolved).
    Does **not** publish bouts or Parquet artifacts.
    """

    frames: pd.DataFrame
    diagnostics: tuple[StageDiagnostic, ...]
    source_s7_run_id: str
    source_s7_run_dir: Path
    source_s6_run_id: str
    source_s6_run_dir: Path
    session_root: Path
    exp_name: str
    fps: float
    n_frames: int
    n_directional_rows: int
    mouse_ids: tuple[int, ...]
    detectors_dyadic: tuple[str, ...]
    detectors_focal: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    settings_used: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SPRINT2_RESULT_SCHEMA_VERSION
    sprint1_schema_version: str = SPRINT1_RESULT_SCHEMA_VERSION

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sprint1_schema_version": self.sprint1_schema_version,
            "source_s7_run_id": self.source_s7_run_id,
            "source_s7_run_dir": str(self.source_s7_run_dir),
            "source_s6_run_id": self.source_s6_run_id,
            "source_s6_run_dir": str(self.source_s6_run_dir),
            "session_root": str(self.session_root),
            "exp_name": self.exp_name,
            "fps": self.fps,
            "n_frames": self.n_frames,
            "n_directional_rows": self.n_directional_rows,
            "mouse_ids": list(self.mouse_ids),
            "detectors_dyadic": list(self.detectors_dyadic),
            "detectors_focal": list(self.detectors_focal),
            "warnings": list(self.warnings),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "settings_used": dict(self.settings_used),
        }


@dataclass(frozen=True, slots=True)
class Sprint3ArbitrationResult:
    """Headless Sprint 3 output after temporal stages + finalize (no publish)."""

    frames: pd.DataFrame
    diagnostics: tuple[StageDiagnostic, ...]
    source_s7_run_id: str
    source_s7_run_dir: Path
    source_s6_run_id: str
    source_s6_run_dir: Path
    session_root: Path
    exp_name: str
    fps: float
    n_frames: int
    n_directional_rows: int
    mouse_ids: tuple[int, ...]
    detectors_dyadic: tuple[str, ...]
    detectors_focal: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    settings_used: dict[str, Any] = field(default_factory=dict)
    schema_version: str = SPRINT3_RESULT_SCHEMA_VERSION
    sprint1_schema_version: str = SPRINT1_RESULT_SCHEMA_VERSION
    sprint2_schema_version: str = SPRINT2_RESULT_SCHEMA_VERSION

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "sprint1_schema_version": self.sprint1_schema_version,
            "sprint2_schema_version": self.sprint2_schema_version,
            "source_s7_run_id": self.source_s7_run_id,
            "source_s7_run_dir": str(self.source_s7_run_dir),
            "source_s6_run_id": self.source_s6_run_id,
            "source_s6_run_dir": str(self.source_s6_run_dir),
            "session_root": str(self.session_root),
            "exp_name": self.exp_name,
            "fps": self.fps,
            "n_frames": self.n_frames,
            "n_directional_rows": self.n_directional_rows,
            "mouse_ids": list(self.mouse_ids),
            "detectors_dyadic": list(self.detectors_dyadic),
            "detectors_focal": list(self.detectors_focal),
            "warnings": list(self.warnings),
            "diagnostics": [d.to_dict() for d in self.diagnostics],
            "settings_used": dict(self.settings_used),
        }


@dataclass(frozen=True, slots=True)
class BehavioralArbitrationRunResult:
    """Summary returned after an S8 run directory is recorded."""

    success: bool
    status: str
    run_id: str
    run_dir: Path
    run_meta_path: Path
    settings_used_path: Path
    processing_log_path: Path
    qc_path: Path
    behavior_states_path: Path | None
    behavior_bouts_path: Path | None
    diagnostics_path: Path | None
    session_root: Path
    source_s7_run_id: str | None
    source_s6_run_id: str | None
    warnings: tuple[str, ...] = ()
    error_message: str | None = None
    qc: Mapping[str, Any] | None = None

    def to_metadata(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "status": self.status,
            "run_id": self.run_id,
            "run_dir": str(self.run_dir),
            "run_meta_path": str(self.run_meta_path),
            "settings_used_path": str(self.settings_used_path),
            "processing_log_path": str(self.processing_log_path),
            "qc_path": str(self.qc_path),
            "behavior_states_path": (
                str(self.behavior_states_path)
                if self.behavior_states_path is not None
                else None
            ),
            "behavior_bouts_path": (
                str(self.behavior_bouts_path)
                if self.behavior_bouts_path is not None
                else None
            ),
            "diagnostics_path": (
                str(self.diagnostics_path)
                if self.diagnostics_path is not None
                else None
            ),
            "session_root": str(self.session_root),
            "source_s7_run_id": self.source_s7_run_id,
            "source_s6_run_id": self.source_s6_run_id,
            "warnings": list(self.warnings),
            "error_message": self.error_message,
            "qc": dict(self.qc) if self.qc is not None else None,
        }
