"""Joint S4 / S5 eligibility, source lineage, and 120 Hz grid alignment.

Sprint 1 selects and validates an eligible S4+S5 pair for the same session.
Feature calculation, blob merge, and S6 run export are later sprints.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from blob_features.contracts import (
    BLOB_FEATURES_FILENAME,
    BlobFeaturesError,
)
from blob_features.contracts import (
    RUN_META_FILENAME as S5_RUN_META_FILENAME,
)
from blob_features.contracts import (
    TARGET_FPS_DEFAULT as S5_TARGET_FPS,
)
from blob_features.export import load_blob_features_table
from blob_features.run_discovery import (
    BlobFeaturesRunSummary,
    find_latest_eligible_s5_run,
    summarize_blob_features_project,
)
from blob_features.runner import validate_s5_run_for_downstream
from combined_features.contracts import (
    JOINT_TARGET_FPS,
    JOINT_TIME_ATOL_SEC,
    REASON_FPS_MISMATCH,
    REASON_INDEX_INTEGRITY,
    REASON_LINEAGE_UNPROVABLE,
    REASON_MISSING_UPSTREAM,
    REASON_S4_INELIGIBLE,
    REASON_S4_STRUCTURE,
    REASON_S5_INELIGIBLE,
    REASON_SAMPLE_COUNT_MISMATCH,
    REASON_SESSION_MISMATCH,
    REASON_SYNC_DIGEST_MISMATCH,
    REASON_SYNC_DIGEST_UNAVAILABLE,
    REASON_SYNC_MISSING,
    REASON_SYNC_PATH_MISMATCH,
    REASON_TIME_GRID_MISMATCH,
    CombinedFeaturesError,
)
from pose_finalization.contracts import (
    FINAL_POSE_FILENAME,
    PoseFinalizationError,
)
from pose_finalization.contracts import (
    RUN_META_FILENAME as S4_RUN_META_FILENAME,
)
from pose_finalization.contracts import (
    TARGET_FPS_DEFAULT as S4_TARGET_FPS,
)
from pose_finalization.export import load_final_pose_table
from pose_finalization.run_discovery import (
    PoseFinalizationRunSummary,
    find_latest_eligible_s4_run,
    summarize_pose_finalization_project,
)
from pose_finalization.runner import validate_s4_run_for_downstream
from pose_finalization.timing import build_uniform_target_timeline, load_source_timing


@dataclass(frozen=True, slots=True)
class JointAlignmentReport:
    """Machine-readable result of S4↔S5 session / sync / timeline checks."""

    eligible: bool
    reason_code: str | None
    message: str
    session_root: Path
    s4_run_dir: Path | None
    s5_run_dir: Path | None
    prepared_sync: Path | None
    prepared_sync_sha256: str | None
    sample_count: int | None
    target_fps: float
    time_atol_sec: float
    checks: Mapping[str, bool]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "reason_code": self.reason_code,
            "message": self.message,
            "session_root": str(self.session_root),
            "s4_run_dir": str(self.s4_run_dir) if self.s4_run_dir is not None else None,
            "s5_run_dir": str(self.s5_run_dir) if self.s5_run_dir is not None else None,
            "prepared_sync": (
                str(self.prepared_sync) if self.prepared_sync is not None else None
            ),
            "prepared_sync_sha256": self.prepared_sync_sha256,
            "sample_count": self.sample_count,
            "target_fps": self.target_fps,
            "time_atol_sec": self.time_atol_sec,
            "checks": dict(self.checks),
        }


@dataclass(frozen=True, slots=True)
class JointEligiblePair:
    """Validated S4 + S5 pair ready for the Sprint 1 pose adapter."""

    session_root: Path
    s4_run_dir: Path
    s4_run_id: str
    s5_run_dir: Path
    s5_run_id: str
    final_pose_path: Path
    blob_features_path: Path
    prepared_sync: Path
    prepared_sync_sha256: str
    sample_count: int
    alignment: JointAlignmentReport
    s4_meta: Mapping[str, Any]
    s5_meta: Mapping[str, Any]
    final_pose: pd.DataFrame
    blob_features: pd.DataFrame


def select_default_joint_pair(
    session_root: Path | str,
) -> JointEligiblePair:
    """Select newest eligible S4 and S5 runs and enforce joint alignment.

    Headless policy (implementation contract §2.2 / §2.5): on alignment failure
    of the newest pair, raise with an explicit reason — do not silently search
    alternate eligible pairs.
    """

    root = Path(session_root).expanduser().resolve()
    if not root.is_dir():
        raise CombinedFeaturesError(
            f"Session root does not exist: {root}",
            reason_code=REASON_MISSING_UPSTREAM,
        )

    s4 = find_latest_eligible_s4_run(root)
    if s4 is None:
        raise CombinedFeaturesError(
            "No currently eligible S4 pose-finalization run was found for this session.",
            reason_code=REASON_S4_INELIGIBLE,
        )
    s5 = find_latest_eligible_s5_run(root)
    if s5 is None:
        raise CombinedFeaturesError(
            "No currently eligible S5 blob-features run was found for this session.",
            reason_code=REASON_S5_INELIGIBLE,
        )
    return validate_joint_s4_s5_pair(
        session_root=root,
        s4_run_dir=s4.run_dir,
        s5_run_dir=s5.run_dir,
    )


def validate_joint_s4_s5_pair(
    *,
    session_root: Path | str,
    s4_run_dir: Path | str,
    s5_run_dir: Path | str,
) -> JointEligiblePair:
    """Revalidate upstream eligibility and enforce shared S1 lineage + timeline."""

    root = Path(session_root).expanduser().resolve()
    s4_dir = Path(s4_run_dir).expanduser().resolve()
    s5_dir = Path(s5_run_dir).expanduser().resolve()

    try:
        validate_s4_run_for_downstream(s4_dir)
    except PoseFinalizationError as exc:
        raise CombinedFeaturesError(
            f"S4 run is not eligible for downstream use: {exc}",
            reason_code=REASON_S4_INELIGIBLE,
        ) from exc

    try:
        validate_s5_run_for_downstream(s5_dir)
    except BlobFeaturesError as exc:
        raise CombinedFeaturesError(
            f"S5 run is not eligible for downstream use: {exc}",
            reason_code=REASON_S5_INELIGIBLE,
        ) from exc

    s4_meta = _load_json_object(s4_dir / S4_RUN_META_FILENAME)
    s5_meta = _load_json_object(s5_dir / S5_RUN_META_FILENAME)
    final_pose_path = s4_dir / FINAL_POSE_FILENAME
    blob_path = s5_dir / BLOB_FEATURES_FILENAME
    if not final_pose_path.is_file():
        raise CombinedFeaturesError(
            f"S4 final_pose.parquet is missing: {final_pose_path}",
            reason_code=REASON_S4_INELIGIBLE,
        )
    if not blob_path.is_file():
        raise CombinedFeaturesError(
            f"S5 blob_features.parquet is missing: {blob_path}",
            reason_code=REASON_S5_INELIGIBLE,
        )

    final_pose = load_final_pose_table(final_pose_path)
    blob_features = load_blob_features_table(blob_path)

    alignment = evaluate_joint_alignment(
        session_root=root,
        s4_run_dir=s4_dir,
        s5_run_dir=s5_dir,
        s4_meta=s4_meta,
        s5_meta=s5_meta,
        final_pose=final_pose,
        blob_features=blob_features,
    )
    if not alignment.eligible:
        raise CombinedFeaturesError(
            alignment.message,
            reason_code=alignment.reason_code,
        )

    assert alignment.prepared_sync is not None
    assert alignment.prepared_sync_sha256 is not None
    assert alignment.sample_count is not None

    return JointEligiblePair(
        session_root=root,
        s4_run_dir=s4_dir,
        s4_run_id=str(s4_meta.get("run_id") or s4_dir.name),
        s5_run_dir=s5_dir,
        s5_run_id=str(s5_meta.get("run_id") or s5_dir.name),
        final_pose_path=final_pose_path,
        blob_features_path=blob_path,
        prepared_sync=alignment.prepared_sync,
        prepared_sync_sha256=alignment.prepared_sync_sha256,
        sample_count=alignment.sample_count,
        alignment=alignment,
        s4_meta=s4_meta,
        s5_meta=s5_meta,
        final_pose=final_pose,
        blob_features=blob_features,
    )


def evaluate_joint_alignment(
    *,
    session_root: Path | str,
    s4_run_dir: Path | str | None,
    s5_run_dir: Path | str | None,
    s4_meta: Mapping[str, Any],
    s5_meta: Mapping[str, Any],
    final_pose: pd.DataFrame,
    blob_features: pd.DataFrame,
) -> JointAlignmentReport:
    """Run joint session / sync-identity / timeline checks without raising."""

    root = Path(session_root).expanduser().resolve()
    s4_dir = Path(s4_run_dir).expanduser().resolve() if s4_run_dir is not None else None
    s5_dir = Path(s5_run_dir).expanduser().resolve() if s5_run_dir is not None else None
    checks: dict[str, bool] = {
        "session": False,
        "sync_path": False,
        "sync_exists": False,
        "sync_digest": False,
        "target_fps": False,
        "sample_count": False,
        "index_integrity": False,
        "time_grid": False,
        "s4_structure": False,
    }

    def _fail(
        reason: str,
        message: str,
        *,
        prepared_sync: Path | None = None,
        prepared_sync_sha256: str | None = None,
        sample_count: int | None = None,
    ) -> JointAlignmentReport:
        return JointAlignmentReport(
            eligible=False,
            reason_code=reason,
            message=message,
            session_root=root,
            s4_run_dir=s4_dir,
            s5_run_dir=s5_dir,
            prepared_sync=prepared_sync,
            prepared_sync_sha256=prepared_sync_sha256,
            sample_count=sample_count,
            target_fps=JOINT_TARGET_FPS,
            time_atol_sec=JOINT_TIME_ATOL_SEC,
            checks=checks,
        )

    source_s3 = s4_meta.get("source_s3")
    source_s1 = s5_meta.get("source_s1")
    if not isinstance(source_s3, Mapping):
        return _fail(
            REASON_LINEAGE_UNPROVABLE,
            "S4 run_meta.json is missing source_s3; shared S1 lineage cannot be proven.",
        )
    if not isinstance(source_s1, Mapping):
        return _fail(
            REASON_LINEAGE_UNPROVABLE,
            "S5 run_meta.json is missing source_s1; shared S1 lineage cannot be proven.",
        )

    s4_session = _optional_path(source_s3.get("session_root"))
    s5_session = _optional_path(source_s1.get("session_root"))
    if s4_session is None or s5_session is None:
        return _fail(
            REASON_LINEAGE_UNPROVABLE,
            "S4/S5 session_root provenance is incomplete; shared lineage cannot be proven.",
        )
    if s4_session != root or s5_session != root:
        return _fail(
            REASON_SESSION_MISMATCH,
            "S4 and S5 session_root provenance do not both resolve to the requested session.",
        )
    checks["session"] = True

    s4_sync = _optional_path(source_s3.get("prepared_sync"))
    s5_sync = _optional_path(source_s1.get("prepared_sync"))
    if s4_sync is None or s5_sync is None:
        return _fail(
            REASON_LINEAGE_UNPROVABLE,
            "S4/S5 prepared_sync path provenance is incomplete; shared lineage cannot be proven.",
        )
    if s4_sync != s5_sync:
        return _fail(
            REASON_SYNC_PATH_MISMATCH,
            "S4 and S5 prepared_sync paths do not resolve to the same file "
            f"(s4={s4_sync}, s5={s5_sync}).",
            prepared_sync=s4_sync,
        )
    checks["sync_path"] = True

    if not s4_sync.is_file():
        return _fail(
            REASON_SYNC_MISSING,
            f"Shared prepared_sync.npz is missing: {s4_sync}",
            prepared_sync=s4_sync,
        )
    checks["sync_exists"] = True

    s5_digest = _optional_text(source_s1.get("prepared_sync_sha256"))
    if s5_digest is None:
        return _fail(
            REASON_SYNC_DIGEST_UNAVAILABLE,
            "S5 run_meta.json does not record prepared_sync_sha256; "
            "sync identity cannot be verified against S4's recorded sync path.",
            prepared_sync=s4_sync,
        )
    live_digest = _sha256(s4_sync)
    if live_digest.lower() != s5_digest.lower():
        return _fail(
            REASON_SYNC_DIGEST_MISMATCH,
            "Live SHA256 of S4's prepared_sync path does not match S5's recorded "
            "prepared_sync_sha256 (S1 re-prepare or mismatched lineage).",
            prepared_sync=s4_sync,
            prepared_sync_sha256=live_digest,
        )
    checks["sync_digest"] = True

    if not _target_fps_ok(s4_meta, s5_meta):
        return _fail(
            REASON_FPS_MISMATCH,
            "S4/S5 recorded target FPS is not the lab 120 Hz contract clock.",
            prepared_sync=s4_sync,
            prepared_sync_sha256=live_digest,
        )
    checks["target_fps"] = True

    try:
        s4_times, s4_samples = extract_s4_sample_timeline(final_pose)
    except CombinedFeaturesError as exc:
        return _fail(
            exc.reason_code or REASON_S4_STRUCTURE,
            str(exc),
            prepared_sync=s4_sync,
            prepared_sync_sha256=live_digest,
        )
    checks["s4_structure"] = True

    try:
        s5_times, s5_samples, s5_orig = extract_s5_sample_timeline(blob_features)
    except CombinedFeaturesError as exc:
        return _fail(
            exc.reason_code or REASON_INDEX_INTEGRITY,
            str(exc),
            prepared_sync=s4_sync,
            prepared_sync_sha256=live_digest,
            sample_count=int(s4_samples.size),
        )

    if int(s4_samples.size) != int(s5_samples.size):
        return _fail(
            REASON_SAMPLE_COUNT_MISMATCH,
            "S4 unique sample count does not match S5 row count "
            f"({int(s4_samples.size)} != {int(s5_samples.size)}).",
            prepared_sync=s4_sync,
            prepared_sync_sha256=live_digest,
            sample_count=int(s4_samples.size),
        )

    expected_from_sync = _expected_target_sample_count(s4_sync)
    if expected_from_sync is not None and int(s4_samples.size) != expected_from_sync:
        return _fail(
            REASON_SAMPLE_COUNT_MISMATCH,
            "Aligned sample count does not match the timeline reconstructed from "
            f"prepared_sync.npz ({int(s4_samples.size)} != {expected_from_sync}).",
            prepared_sync=s4_sync,
            prepared_sync_sha256=live_digest,
            sample_count=int(s4_samples.size),
        )
    checks["sample_count"] = True

    expected_idx = np.arange(int(s4_samples.size), dtype=np.int64)
    if not np.array_equal(s4_samples, expected_idx) or not np.array_equal(
        s5_samples, expected_idx
    ):
        return _fail(
            REASON_INDEX_INTEGRITY,
            "S4/S5 sample_idx grids are not unique contiguous [0 .. N-1].",
            prepared_sync=s4_sync,
            prepared_sync_sha256=live_digest,
            sample_count=int(s4_samples.size),
        )
    if s5_orig.shape != expected_idx.shape:
        return _fail(
            REASON_INDEX_INTEGRITY,
            "S5 orig_frame_idx length does not match the sample grid.",
            prepared_sync=s4_sync,
            prepared_sync_sha256=live_digest,
            sample_count=int(s4_samples.size),
        )
    checks["index_integrity"] = True

    if s4_times.shape != s5_times.shape or not np.allclose(
        s4_times, s5_times, rtol=0.0, atol=JOINT_TIME_ATOL_SEC
    ):
        return _fail(
            REASON_TIME_GRID_MISMATCH,
            "S4 and S5 time_sec grids do not match within atol=1e-12.",
            prepared_sync=s4_sync,
            prepared_sync_sha256=live_digest,
            sample_count=int(s4_samples.size),
        )
    checks["time_grid"] = True

    return JointAlignmentReport(
        eligible=True,
        reason_code=None,
        message="S4 and S5 share session, prepared_sync identity, and 120 Hz timeline.",
        session_root=root,
        s4_run_dir=s4_dir,
        s5_run_dir=s5_dir,
        prepared_sync=s4_sync,
        prepared_sync_sha256=live_digest,
        sample_count=int(s4_samples.size),
        target_fps=JOINT_TARGET_FPS,
        time_atol_sec=JOINT_TIME_ATOL_SEC,
        checks=checks,
    )


def extract_s4_sample_timeline(
    final_pose: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return unique ``(time_sec, sample_idx)`` for long-form S4 pose rows.

    Validates contiguous unique sample indices and consistent ``time_sec`` per
    sample. Also rejects duplicate ``(sample_idx, track, node)`` keys.
    """

    missing = [name for name in ("sample_idx", "time_sec", "track", "node") if name not in final_pose]
    if missing:
        raise CombinedFeaturesError(
            "S4 final_pose is missing required columns: " + ", ".join(missing),
            reason_code=REASON_S4_STRUCTURE,
        )
    if final_pose.empty:
        raise CombinedFeaturesError(
            "S4 final_pose.parquet is empty.",
            reason_code=REASON_S4_STRUCTURE,
        )

    key_cols = ["sample_idx", "track", "node"]
    duplicated = final_pose.duplicated(subset=key_cols, keep=False)
    if bool(duplicated.any()):
        raise CombinedFeaturesError(
            "S4 final_pose has duplicate (sample_idx, track, node) keys.",
            reason_code=REASON_S4_STRUCTURE,
        )

    # One time_sec per sample_idx.
    grouped = final_pose.groupby("sample_idx", sort=True)["time_sec"]
    nunique = grouped.nunique(dropna=False)
    if int((nunique > 1).sum()) > 0:
        raise CombinedFeaturesError(
            "S4 final_pose has conflicting time_sec values for the same sample_idx.",
            reason_code=REASON_S4_STRUCTURE,
        )
    sample_idx = np.asarray(sorted(final_pose["sample_idx"].unique()), dtype=np.int64)
    if sample_idx.size == 0:
        raise CombinedFeaturesError(
            "S4 final_pose has no sample_idx values.",
            reason_code=REASON_S4_STRUCTURE,
        )
    if int(sample_idx.min()) != 0 or int(sample_idx.max()) != int(sample_idx.size) - 1:
        raise CombinedFeaturesError(
            "S4 sample_idx must be unique contiguous integers [0 .. N-1].",
            reason_code=REASON_INDEX_INTEGRITY,
        )
    if sample_idx.size != len(np.unique(sample_idx)):
        raise CombinedFeaturesError(
            "S4 sample_idx values are not unique.",
            reason_code=REASON_INDEX_INTEGRITY,
        )

    time_by_sample = grouped.first()
    times = np.asarray(time_by_sample.loc[sample_idx].to_numpy(), dtype=np.float64)
    if not np.all(np.isfinite(times)):
        raise CombinedFeaturesError(
            "S4 time_sec contains non-finite values.",
            reason_code=REASON_S4_STRUCTURE,
        )
    return times, sample_idx


def extract_s5_sample_timeline(
    blob_features: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(time_sec, sample_idx, orig_frame_idx)`` for one-row-per-sample S5."""

    required = ("sample_idx", "dense_frame_idx", "frame_idx", "time_sec", "orig_frame_idx")
    missing = [name for name in required if name not in blob_features.columns]
    if missing:
        raise CombinedFeaturesError(
            "S5 blob_features is missing required columns: " + ", ".join(missing),
            reason_code=REASON_INDEX_INTEGRITY,
        )
    if blob_features.empty:
        raise CombinedFeaturesError(
            "S5 blob_features.parquet is empty.",
            reason_code=REASON_INDEX_INTEGRITY,
        )

    sample_idx = np.asarray(blob_features["sample_idx"], dtype=np.int64)
    dense = np.asarray(blob_features["dense_frame_idx"], dtype=np.int64)
    frame = np.asarray(blob_features["frame_idx"], dtype=np.int64)
    times = np.asarray(blob_features["time_sec"], dtype=np.float64)
    orig = np.asarray(blob_features["orig_frame_idx"], dtype=np.int64)

    expected = np.arange(len(blob_features), dtype=np.int64)
    if not (
        np.array_equal(sample_idx, expected)
        and np.array_equal(dense, expected)
        and np.array_equal(frame, expected)
    ):
        raise CombinedFeaturesError(
            "S5 sample_idx/dense_frame_idx/frame_idx must equal unique [0 .. N-1].",
            reason_code=REASON_INDEX_INTEGRITY,
        )
    if not np.all(np.isfinite(times)):
        raise CombinedFeaturesError(
            "S5 time_sec contains non-finite values.",
            reason_code=REASON_INDEX_INTEGRITY,
        )
    return times, sample_idx, orig


def list_eligible_joint_pairs(
    session_root: Path | str,
) -> tuple[tuple[PoseFinalizationRunSummary, BlobFeaturesRunSummary, JointAlignmentReport], ...]:
    """Enumerate eligible S4×S5 pairs with alignment reports (GUI override helper)."""

    root = Path(session_root).expanduser().resolve()
    s4_summary = summarize_pose_finalization_project(root)
    s5_summary = summarize_blob_features_project(root)
    pairs: list[
        tuple[PoseFinalizationRunSummary, BlobFeaturesRunSummary, JointAlignmentReport]
    ] = []
    for s4 in s4_summary.eligible_runs:
        for s5 in s5_summary.eligible_runs:
            try:
                s4_meta = _load_json_object(s4.run_dir / S4_RUN_META_FILENAME)
                s5_meta = _load_json_object(s5.run_dir / S5_RUN_META_FILENAME)
                final_pose = load_final_pose_table(s4.run_dir / FINAL_POSE_FILENAME)
                blob = load_blob_features_table(s5.run_dir / BLOB_FEATURES_FILENAME)
            except (CombinedFeaturesError, PoseFinalizationError, BlobFeaturesError, OSError) as exc:
                report = JointAlignmentReport(
                    eligible=False,
                    reason_code=REASON_MISSING_UPSTREAM,
                    message=str(exc),
                    session_root=root,
                    s4_run_dir=s4.run_dir,
                    s5_run_dir=s5.run_dir,
                    prepared_sync=None,
                    prepared_sync_sha256=None,
                    sample_count=None,
                    target_fps=JOINT_TARGET_FPS,
                    time_atol_sec=JOINT_TIME_ATOL_SEC,
                    checks={},
                )
                pairs.append((s4, s5, report))
                continue
            report = evaluate_joint_alignment(
                session_root=root,
                s4_run_dir=s4.run_dir,
                s5_run_dir=s5.run_dir,
                s4_meta=s4_meta,
                s5_meta=s5_meta,
                final_pose=final_pose,
                blob_features=blob,
            )
            pairs.append((s4, s5, report))
    return tuple(pairs)


def _target_fps_ok(s4_meta: Mapping[str, Any], s5_meta: Mapping[str, Any]) -> bool:
    expected = float(JOINT_TARGET_FPS)
    if abs(float(S4_TARGET_FPS) - expected) > 0.0:
        return False
    if abs(float(S5_TARGET_FPS) - expected) > 0.0:
        return False

    dense = s4_meta.get("dense_summary")
    if isinstance(dense, Mapping) and dense.get("target_fps") is not None:
        try:
            if abs(float(dense["target_fps"]) - expected) > 1e-9:
                return False
        except (TypeError, ValueError):
            return False

    source_s1 = s5_meta.get("source_s1")
    if isinstance(source_s1, Mapping) and source_s1.get("target_fps") is not None:
        try:
            if abs(float(source_s1["target_fps"]) - expected) > 1e-9:
                return False
        except (TypeError, ValueError):
            return False
    return True


def _expected_target_sample_count(prepared_sync: Path) -> int | None:
    try:
        source = load_source_timing(prepared_sync)
        targets = build_uniform_target_timeline(
            source.prepared_time_sec,
            target_fps=JOINT_TARGET_FPS,
        )
    except (PoseFinalizationError, OSError, ValueError):
        return None
    return int(targets.size)


def _load_json_object(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise CombinedFeaturesError(
            f"Required metadata file is missing: {path}",
            reason_code=REASON_MISSING_UPSTREAM,
        )
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise CombinedFeaturesError(
            f"Could not load JSON metadata: {path}: {exc}",
            reason_code=REASON_MISSING_UPSTREAM,
        ) from exc
    if not isinstance(payload, dict):
        raise CombinedFeaturesError(
            f"JSON root must be an object: {path}",
            reason_code=REASON_MISSING_UPSTREAM,
        )
    return payload


def _optional_path(value: object) -> Path | None:
    text = _optional_text(value)
    if text is None:
        return None
    return Path(text).expanduser().resolve()


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "JointAlignmentReport",
    "JointEligiblePair",
    "evaluate_joint_alignment",
    "extract_s4_sample_timeline",
    "extract_s5_sample_timeline",
    "list_eligible_joint_pairs",
    "select_default_joint_pair",
    "validate_joint_s4_s5_pair",
]
