"""Discover and validate accepted Subsystem 03 results for S4.

S3 ``complete_reviewable`` means automatic correction finished and the run is
open for review. It is never treated as acceptance. S4 eligibility requires
``acceptance_state == "accepted"``, a present ``tracked_pose.parquet``, and a
matching recorded SHA256 from the S3 acceptance metadata block.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pose_finalization.contracts import PoseFinalizationError
from tracking_correction.contracts import (
    ACCEPTANCE_ACCEPTED,
    FINAL_TRACKED_POSE_FILENAME,
    RUN_META_FILENAME,
    is_currently_accepted,
)
from tracking_correction.run_discovery import summarize_tracking_correction_project

_TRACKING_CORRECTION_DIRNAME = "tracking_correction"
_PREPROCESS_DIRNAME = "preprocess"
_PREPARED_SYNC_FILENAME = "prepared_sync.npz"


@dataclass(frozen=True, slots=True)
class AcceptedS3Run:
    """Validated accepted S3 handoff usable as an S4 input."""

    run_dir: Path
    run_id: str
    session_root: Path
    tracked_pose_path: Path
    run_meta_path: Path
    prepared_sync_path: Path
    acceptance_state: str
    tracked_pose_sha256: str
    accepted_at: str | None
    profile_id: str | None
    created_at: str | None
    completed_at: str | None


def validate_accepted_s3_run(s3_run_dir: Path | str) -> AcceptedS3Run:
    """Validate one S3 run directory as an accepted S4 input.

    Read-only with respect to S1/S2/S3 artifacts.
    """

    run_dir = Path(s3_run_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise PoseFinalizationError(f"S3 run directory does not exist: {run_dir}")

    run_meta_path = run_dir / RUN_META_FILENAME
    if not run_meta_path.is_file():
        raise PoseFinalizationError(f"S3 run_meta.json is missing: {run_meta_path}")

    meta = _load_json_object(run_meta_path)
    acceptance_state = meta.get("acceptance_state")
    if not is_currently_accepted(acceptance_state):
        state_text = acceptance_state if isinstance(acceptance_state, str) else "missing"
        raise PoseFinalizationError(
            "S3 run is not currently accepted "
            f"(acceptance_state={state_text!r}); "
            "complete_reviewable is not sufficient for S4."
        )

    tracked_pose_path = run_dir / FINAL_TRACKED_POSE_FILENAME
    if not tracked_pose_path.is_file():
        raise PoseFinalizationError(
            "S3 run is marked accepted but tracked_pose.parquet is missing: "
            f"{tracked_pose_path}"
        )

    acceptance = meta.get("acceptance")
    if not isinstance(acceptance, Mapping):
        raise PoseFinalizationError(
            "S3 run_meta.json acceptance block is missing or malformed."
        )
    recorded_sha = _recorded_tracked_pose_sha256(acceptance)
    if recorded_sha is None:
        raise PoseFinalizationError(
            "S3 acceptance metadata does not record tracked_pose_sha256."
        )

    actual_sha = _sha256(tracked_pose_path)
    if actual_sha != recorded_sha:
        raise PoseFinalizationError(
            "Accepted tracked_pose.parquet SHA256 does not match the recorded "
            "S3 acceptance digest."
        )

    session_root = _resolve_session_root(run_dir, meta)
    prepared_sync_path = _resolve_prepared_sync(session_root, run_dir, meta)
    if not prepared_sync_path.is_file():
        raise PoseFinalizationError(
            f"Authoritative S1 prepared_sync.npz was not found: {prepared_sync_path}"
        )

    return AcceptedS3Run(
        run_dir=run_dir,
        run_id=_optional_text(meta.get("run_id")) or run_dir.name,
        session_root=session_root,
        tracked_pose_path=tracked_pose_path,
        run_meta_path=run_meta_path,
        prepared_sync_path=prepared_sync_path,
        acceptance_state=ACCEPTANCE_ACCEPTED,
        tracked_pose_sha256=actual_sha,
        accepted_at=_optional_text(acceptance.get("accepted_at")),
        profile_id=_optional_text(meta.get("profile_id")),
        created_at=_optional_text(meta.get("created_at")),
        completed_at=_optional_text(meta.get("completed_at")),
    )


def list_accepted_s3_runs(session_root: Path | str) -> tuple[AcceptedS3Run, ...]:
    """Return validated accepted S3 runs under a session, newest first.

    Candidates are discovered via the existing S3 project summary. Each
    candidate is independently validated; reviewable-but-unaccepted runs are
    skipped rather than treated as eligible.
    """

    root = Path(session_root).expanduser().resolve()
    if not root.is_dir():
        raise PoseFinalizationError(f"Session root does not exist: {root}")

    summary = summarize_tracking_correction_project(root)
    accepted: list[AcceptedS3Run] = []
    for run in summary.runs:
        try:
            accepted.append(validate_accepted_s3_run(run.run_dir))
        except PoseFinalizationError:
            continue
    accepted.sort(key=_accepted_sort_key, reverse=True)
    return tuple(accepted)


def find_latest_accepted_s3_run(session_root: Path | str) -> AcceptedS3Run | None:
    """Return the newest validated accepted S3 run for a session, if any."""

    runs = list_accepted_s3_runs(session_root)
    return runs[0] if runs else None


def resolve_accepted_s3_run(
    session_root: Path | str | None = None,
    *,
    s3_run_dir: Path | str | None = None,
) -> AcceptedS3Run:
    """Resolve an accepted S3 run from an explicit path or session discovery.

    Explicit ``s3_run_dir`` always wins and is validated directly. Otherwise
    ``session_root`` is required and the latest eligible accepted run is used.
    """

    if s3_run_dir is not None:
        return validate_accepted_s3_run(s3_run_dir)

    if session_root is None:
        raise PoseFinalizationError(
            "Either s3_run_dir or session_root is required to resolve an accepted S3 run."
        )

    latest = find_latest_accepted_s3_run(session_root)
    if latest is None:
        raise PoseFinalizationError(
            "No eligible accepted S3 run was found under the session "
            f"tracking_correction directory: {Path(session_root) / _TRACKING_CORRECTION_DIRNAME}"
        )
    return latest


def _recorded_tracked_pose_sha256(acceptance: Mapping[str, Any]) -> str | None:
    for key in ("tracked_pose_sha256", "working_tracked_pose_sha256"):
        value = _optional_text(acceptance.get(key))
        if value is not None:
            return value.lower()
    return None


def _resolve_session_root(run_dir: Path, meta: Mapping[str, Any]) -> Path:
    if run_dir.parent.name == _TRACKING_CORRECTION_DIRNAME:
        layout_root = run_dir.parent.parent
        if layout_root.is_dir():
            return layout_root.resolve()

    input_block = meta.get("input")
    if isinstance(input_block, Mapping):
        recorded = _optional_text(input_block.get("session_root"))
        if recorded is not None:
            candidate = Path(recorded).expanduser()
            if candidate.is_dir():
                return candidate.resolve()

    raise PoseFinalizationError(
        "Could not resolve session_root for accepted S3 run "
        f"{run_dir}; expected …/tracking_correction/<run_id> layout."
    )


def _resolve_prepared_sync(
    session_root: Path,
    run_dir: Path,
    meta: Mapping[str, Any],
) -> Path:
    layout_sync = session_root / _PREPROCESS_DIRNAME / _PREPARED_SYNC_FILENAME
    if layout_sync.is_file():
        return layout_sync.resolve()

    input_block = meta.get("input")
    if isinstance(input_block, Mapping):
        recorded = _optional_text(input_block.get("prepared_sync"))
        if recorded is not None:
            candidate = Path(recorded).expanduser()
            if not candidate.is_absolute():
                candidate = run_dir / candidate
            if candidate.is_file():
                return candidate.resolve()

    return layout_sync.resolve()


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)
    except (OSError, json.JSONDecodeError) as exc:
        raise PoseFinalizationError(
            f"S3 run_meta.json is missing or malformed: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise PoseFinalizationError(
            f"S3 run_meta.json must be a JSON object: {path}"
        )
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_text(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text or None


def _accepted_sort_key(run: AcceptedS3Run) -> tuple[datetime, str]:
    for value in (run.completed_at, run.created_at, run.accepted_at):
        parsed = _parse_datetime(value)
        if parsed is not None:
            return (parsed, run.run_id)
    parsed = _parse_run_id_timestamp(run.run_id)
    if parsed is not None:
        return (parsed, run.run_id)
    try:
        return (datetime.fromtimestamp(run.run_dir.stat().st_mtime), run.run_id)
    except OSError:
        return (datetime.min, run.run_id)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is not None:
        return parsed.replace(tzinfo=None)
    return parsed


def _parse_run_id_timestamp(run_id: str) -> datetime | None:
    if "__" not in run_id:
        return None
    timestamp = run_id.rsplit("__", maxsplit=1)[-1]
    if re.fullmatch(r"\d{8}T\d{6}", timestamp) is None:
        return None
    return datetime.strptime(timestamp, "%Y%m%dT%H%M%S")
