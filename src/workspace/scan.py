"""Headless discovery of existing project/session trees (D0-B).

Scanning is read-only: it never creates catalogs, never moves science trees, and
never launches scientific processing. Subsystem run presence is detected via the
existing S1–S10 discovery interfaces where practical.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from workspace.models import (
    SessionFingerprint,
    normalize_path,
    probe_has_preprocess,
    probe_path_status,
    utc_now_iso,
)

SubsystemKey = Literal["s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10"]

SUBSYSTEM_DIRNAMES: dict[SubsystemKey, str] = {
    "s1": "preprocess",
    "s2": "pose_inference",
    "s3": "tracking_correction",
    "s4": "pose_finalization",
    "s5": "blob_features",
    "s6": "combined_features",
    "s7": "behavioral_inference",
    "s8": "behavioral_arbitration",
    "s9": "results_package",
    "s10": "behavioral_review",
}

DOWNSTREAM_SUBSYSTEMS: tuple[SubsystemKey, ...] = (
    "s2",
    "s3",
    "s4",
    "s5",
    "s6",
    "s7",
    "s8",
    "s9",
    "s10",
)

ScanKind = Literal[
    "legacy_self_project",
    "parent_project",
    "ambiguous",
    "empty",
]


class DiscoveryIssueCode(StrEnum):
    """Structured discovery / association warnings (non-fatal)."""

    MISSING_PREPROCESS = "missing_preprocess"
    INCOMPLETE_S1 = "incomplete_s1"
    PARTIAL_DOWNSTREAM = "partial_downstream"
    AMBIGUOUS_NESTING = "ambiguous_nesting"
    NO_RECOGNIZED_LAYOUT = "no_recognized_layout"
    DISCOVERY_ERROR = "discovery_error"
    PATH_MISSING = "path_missing"


@dataclass(frozen=True)
class DiscoveryIssue:
    code: DiscoveryIssueCode
    message: str
    path: Path | None = None


@dataclass(frozen=True)
class DiscoveredRunRef:
    """Explicit per-subsystem run identity retained from existing scanners."""

    subsystem: SubsystemKey
    run_id: str
    run_dir: Path
    status: str | None = None
    classification: str | None = None
    completed_at: str | None = None


@dataclass(frozen=True)
class SessionDiscovery:
    """Structured discovery result for one session directory."""

    session_root: Path
    display_name: str
    path_status: str
    has_preprocess_dir: bool
    s1_handoff_complete: bool
    s1_missing_files: tuple[str, ...]
    fingerprint: SessionFingerprint
    runs_by_subsystem: Mapping[SubsystemKey, tuple[DiscoveredRunRef, ...]]
    present_subsystems: tuple[SubsystemKey, ...]
    missing_or_incomplete_stages: tuple[str, ...]
    issues: tuple[DiscoveryIssue, ...] = ()
    discovered_at_utc: str = field(default_factory=utc_now_iso)

    @property
    def run_counts(self) -> dict[str, int]:
        return {key: len(runs) for key, runs in self.runs_by_subsystem.items()}


@dataclass(frozen=True)
class ProjectDiscovery:
    """Structured discovery result for a candidate project root."""

    root_path: Path
    display_name: str
    kind: ScanKind
    sessions: tuple[SessionDiscovery, ...]
    issues: tuple[DiscoveryIssue, ...] = ()
    discovered_at_utc: str = field(default_factory=utc_now_iso)

    @property
    def is_legacy_self_project(self) -> bool:
        return self.kind == "legacy_self_project"


def _iso_mtime(path: Path) -> str | None:
    if not path.is_file() and not path.is_dir():
        return None
    try:
        stamp = datetime.fromtimestamp(path.stat().st_mtime, UTC).replace(microsecond=0)
    except OSError:
        return None
    return stamp.isoformat().replace("+00:00", "Z")


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _load_json_mapping(path: Path) -> Mapping[str, Any] | None:
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return raw if isinstance(raw, Mapping) else None


def _find_recorded_prepared_video_sha256(session_root: Path) -> str | None:
    """Best-effort read of an already-recorded prepared-video digest (no hashing)."""

    prepare_meta = _load_json_mapping(session_root / "preprocess" / "prepare_meta.json")
    if prepare_meta is not None:
        for key in ("prepared_video_sha256", "sha256", "content_sha256"):
            direct = _optional_text(prepare_meta.get(key))
            if direct is not None:
                return direct
        prepared = prepare_meta.get("prepared_video")
        if isinstance(prepared, Mapping):
            for key in ("sha256", "prepared_video_sha256", "content_sha256"):
                nested = _optional_text(prepared.get(key))
                if nested is not None:
                    return nested

    # Prefer small run_meta / package_manifest fields already written by science.
    search_roots = (
        session_root / "blob_features",
        session_root / "results_package",
        session_root / "combined_features",
        session_root / "pose_finalization",
    )
    for root in search_roots:
        if not root.is_dir():
            continue
        for child in sorted(root.iterdir()):
            if not child.is_dir():
                continue
            for name in ("run_meta.json", "package_manifest.json"):
                payload = _load_json_mapping(child / name)
                if payload is None:
                    continue
                for key in ("prepared_video_sha256",):
                    value = _optional_text(payload.get(key))
                    if value is not None:
                        return value
                source = payload.get("source")
                if isinstance(source, Mapping):
                    value = _optional_text(source.get("prepared_video_sha256"))
                    if value is not None:
                        return value
                source_s1 = payload.get("source_s1")
                if isinstance(source_s1, Mapping):
                    value = _optional_text(source_s1.get("prepared_video_sha256"))
                    if value is not None:
                        return value
                deps = payload.get("dependency_snapshot")
                if isinstance(deps, Mapping):
                    extra = deps.get("extra")
                    if isinstance(extra, Mapping):
                        value = _optional_text(extra.get("prepared_video_sha256_recorded"))
                        if value is not None:
                            return value
    return None


def build_session_fingerprint(session_root: Path | str) -> SessionFingerprint:
    """Build a lightweight, rebuildable fingerprint without hashing large media."""

    root = normalize_path(session_root)
    has_preprocess = probe_has_preprocess(root)
    prepare_meta = root / "preprocess" / "prepare_meta.json"
    return SessionFingerprint(
        preprocess_present=has_preprocess,
        prepared_video_sha256=_find_recorded_prepared_video_sha256(root)
        if has_preprocess
        else None,
        prepare_meta_mtime=_iso_mtime(prepare_meta) if has_preprocess else None,
    )


def _run_ref(
    *,
    subsystem: SubsystemKey,
    run_id: str,
    run_dir: Path,
    status: str | None = None,
    classification: str | None = None,
    completed_at: str | None = None,
) -> DiscoveredRunRef:
    return DiscoveredRunRef(
        subsystem=subsystem,
        run_id=run_id,
        run_dir=normalize_path(run_dir),
        status=status,
        classification=classification,
        completed_at=completed_at,
    )


def _safe_summarize(
    label: str,
    session_root: Path,
    fn: Callable[[Path], Any],
) -> tuple[Any | None, DiscoveryIssue | None]:
    try:
        return fn(session_root), None
    except Exception as exc:  # noqa: BLE001 - discovery must not abort the scan
        return None, DiscoveryIssue(
            code=DiscoveryIssueCode.DISCOVERY_ERROR,
            message=f"{label} discovery failed: {exc}",
            path=session_root,
        )


def discover_session_artifacts(session_root: Path | str) -> SessionDiscovery:
    """Discover S1 handoff + S2–S10 runs under an existing session directory.

    Never modifies the tree and never starts processing.
    """

    root = normalize_path(session_root)
    path_status = probe_path_status(root)
    issues: list[DiscoveryIssue] = []
    empty_runs: dict[SubsystemKey, tuple[DiscoveredRunRef, ...]] = {
        key: () for key in DOWNSTREAM_SUBSYSTEMS
    }

    if path_status.value != "ok":
        issues.append(
            DiscoveryIssue(
                code=DiscoveryIssueCode.PATH_MISSING,
                message=f"Session path is missing: {root}",
                path=root,
            )
        )
        return SessionDiscovery(
            session_root=root,
            display_name=root.name or "session",
            path_status=path_status.value,
            has_preprocess_dir=False,
            s1_handoff_complete=False,
            s1_missing_files=(),
            fingerprint=SessionFingerprint(preprocess_present=False),
            runs_by_subsystem=empty_runs,
            present_subsystems=(),
            missing_or_incomplete_stages=("s1", *DOWNSTREAM_SUBSYSTEMS),
            issues=tuple(issues),
        )

    has_preprocess = probe_has_preprocess(root)
    fingerprint = build_session_fingerprint(root)

    s1_complete = False
    s1_missing: tuple[str, ...] = ()
    from pose_inference.run_discovery import summarize_pose_inference_project

    s2_summary, s2_issue = _safe_summarize(
        "S2", root, summarize_pose_inference_project
    )
    if s2_issue is not None:
        issues.append(s2_issue)
    if s2_summary is not None:
        s1_complete = bool(s2_summary.s1_handoff.is_complete)
        s1_missing = tuple(s2_summary.s1_handoff.missing_files)
        empty_runs["s2"] = tuple(
            _run_ref(
                subsystem="s2",
                run_id=run.run_id,
                run_dir=run.run_dir,
                status=run.status,
                classification=run.classification,
                completed_at=run.completed_at,
            )
            for run in s2_summary.runs
        )
    elif has_preprocess:
        # Fallback presence-only S1 check when S2 discovery fails.
        required = (
            "prepared_video.mp4",
            "prepare_meta.json",
            "prepared_sync.npz",
        )
        missing = tuple(
            name for name in required if not (root / "preprocess" / name).is_file()
        )
        s1_complete = not missing
        s1_missing = missing

    if not has_preprocess:
        issues.append(
            DiscoveryIssue(
                code=DiscoveryIssueCode.MISSING_PREPROCESS,
                message=f"Session is missing preprocess/: {root / 'preprocess'}",
                path=root,
            )
        )
    elif not s1_complete:
        issues.append(
            DiscoveryIssue(
                code=DiscoveryIssueCode.INCOMPLETE_S1,
                message=(
                    "S1 handoff incomplete; missing: "
                    + (", ".join(s1_missing) if s1_missing else "unknown")
                ),
                path=root / "preprocess",
            )
        )

    def _ingest(
        key: SubsystemKey,
        label: str,
        summarize: Callable[[Path], Any],
        *,
        run_id_attr: str = "run_id",
    ) -> None:
        summary, issue = _safe_summarize(label, root, summarize)
        if issue is not None:
            issues.append(issue)
            return
        if summary is None:
            return
        runs = getattr(summary, "runs", ())
        empty_runs[key] = tuple(
            _run_ref(
                subsystem=key,
                run_id=str(getattr(run, run_id_attr)),
                run_dir=run.run_dir,
                status=getattr(run, "status", None),
                classification=getattr(run, "classification", None),
                completed_at=getattr(run, "completed_at", None),
            )
            for run in runs
        )

    # S2 already ingested above when summarize succeeded.
    from behavioral_arbitration.run_discovery import (
        summarize_behavioral_arbitration_project,
    )
    from behavioral_inference.run_discovery import summarize_behavioral_inference_project
    from behavioral_review.discovery import summarize_behavioral_review_project
    from blob_features.run_discovery import summarize_blob_features_project
    from combined_features.run_discovery import summarize_combined_features_project
    from pose_finalization.run_discovery import summarize_pose_finalization_project
    from results_packaging.run_discovery import summarize_results_package_project
    from tracking_correction.run_discovery import summarize_tracking_correction_project

    _ingest("s3", "S3", summarize_tracking_correction_project)
    _ingest("s4", "S4", summarize_pose_finalization_project)
    _ingest("s5", "S5", summarize_blob_features_project)
    _ingest("s6", "S6", summarize_combined_features_project)
    _ingest("s7", "S7", summarize_behavioral_inference_project)
    _ingest("s8", "S8", summarize_behavioral_arbitration_project)
    _ingest("s9", "S9", summarize_results_package_project)
    _ingest(
        "s10",
        "S10",
        summarize_behavioral_review_project,
        run_id_attr="review_run_id",
    )

    present: list[SubsystemKey] = []
    if has_preprocess:
        present.append("s1")
    for key in DOWNSTREAM_SUBSYSTEMS:
        dirname = SUBSYSTEM_DIRNAMES[key]
        if (root / dirname).is_dir() or empty_runs[key]:
            present.append(key)

    missing_stages: list[str] = []
    if not has_preprocess or not s1_complete:
        missing_stages.append("s1")
    for key in DOWNSTREAM_SUBSYSTEMS:
        dirname = SUBSYSTEM_DIRNAMES[key]
        dir_present = (root / dirname).is_dir()
        if not dir_present and not empty_runs[key]:
            missing_stages.append(key)
        elif dir_present and not empty_runs[key]:
            missing_stages.append(f"{key}:no_runs")
            issues.append(
                DiscoveryIssue(
                    code=DiscoveryIssueCode.PARTIAL_DOWNSTREAM,
                    message=f"{key} directory present but no runs discovered.",
                    path=root / dirname,
                )
            )

    return SessionDiscovery(
        session_root=root,
        display_name=root.name or "session",
        path_status=path_status.value,
        has_preprocess_dir=has_preprocess,
        s1_handoff_complete=s1_complete,
        s1_missing_files=s1_missing,
        fingerprint=fingerprint,
        runs_by_subsystem=empty_runs,
        present_subsystems=tuple(present),
        missing_or_incomplete_stages=tuple(missing_stages),
        issues=tuple(issues),
    )


def _child_session_dirs(root: Path) -> list[Path]:
    children: list[Path] = []
    try:
        entries = sorted(root.iterdir(), key=lambda p: p.name.lower())
    except OSError:
        return children
    for child in entries:
        if not child.is_dir():
            continue
        if child.name.startswith("."):
            continue
        if child.name in set(SUBSYSTEM_DIRNAMES.values()) | {"results_exports", "subsessions"}:
            continue
        if probe_has_preprocess(child):
            children.append(normalize_path(child))
            continue
        # Incomplete child: subsystem dirs without preprocess still count as sessions.
        if any((child / SUBSYSTEM_DIRNAMES[key]).is_dir() for key in DOWNSTREAM_SUBSYSTEMS):
            children.append(normalize_path(child))
    return children


def _has_local_subsystem_tree(root: Path) -> bool:
    return any((root / SUBSYSTEM_DIRNAMES[key]).is_dir() for key in DOWNSTREAM_SUBSYSTEMS)


def scan_candidate(path: Path | str, *, discover_runs: bool = True) -> ProjectDiscovery:
    """Scan a filesystem candidate without modifying directories.

    Rules (architecture §4.2):
    - ``path/preprocess/`` → legacy self-project (project ≡ session)
    - else children with ``preprocess/`` (or incomplete subsystem trees) → parent project
    - both / neither → ambiguous or empty
    """

    root = normalize_path(path)
    issues: list[DiscoveryIssue] = []
    if probe_path_status(root).value != "ok":
        issues.append(
            DiscoveryIssue(
                code=DiscoveryIssueCode.PATH_MISSING,
                message=f"Candidate path is missing: {root}",
                path=root,
            )
        )
        return ProjectDiscovery(
            root_path=root,
            display_name=root.name or "project",
            kind="empty",
            sessions=(),
            issues=tuple(issues),
        )

    has_preprocess = probe_has_preprocess(root)
    children = _child_session_dirs(root)
    local_subsystems = _has_local_subsystem_tree(root)

    kind: ScanKind
    session_roots: list[Path]
    if has_preprocess and children:
        kind = "ambiguous"
        session_roots = [root, *children]
        issues.append(
            DiscoveryIssue(
                code=DiscoveryIssueCode.AMBIGUOUS_NESTING,
                message=(
                    "Path has both a local preprocess/ and child session directories; "
                    "operator must confirm associations."
                ),
                path=root,
            )
        )
    elif has_preprocess:
        kind = "legacy_self_project"
        session_roots = [root]
    elif children:
        kind = "parent_project"
        session_roots = children
    elif local_subsystems:
        # Incomplete legacy session: downstream dirs without preprocess.
        kind = "legacy_self_project"
        session_roots = [root]
        issues.append(
            DiscoveryIssue(
                code=DiscoveryIssueCode.MISSING_PREPROCESS,
                message=(
                    "Legacy session-as-project fallback: subsystem directories present "
                    "without preprocess/."
                ),
                path=root,
            )
        )
    else:
        kind = "empty"
        session_roots = []
        issues.append(
            DiscoveryIssue(
                code=DiscoveryIssueCode.NO_RECOGNIZED_LAYOUT,
                message="No preprocess/ or child session directories discovered.",
                path=root,
            )
        )

    sessions: list[SessionDiscovery] = []
    for session_root in session_roots:
        if discover_runs:
            sessions.append(discover_session_artifacts(session_root))
        else:
            has_prep = probe_has_preprocess(session_root)
            sessions.append(
                SessionDiscovery(
                    session_root=session_root,
                    display_name=session_root.name or "session",
                    path_status=probe_path_status(session_root).value,
                    has_preprocess_dir=has_prep,
                    s1_handoff_complete=False,
                    s1_missing_files=(),
                    fingerprint=build_session_fingerprint(session_root),
                    runs_by_subsystem={key: () for key in DOWNSTREAM_SUBSYSTEMS},
                    present_subsystems=("s1",) if has_prep else (),
                    missing_or_incomplete_stages=(),
                )
            )

    # Bubble session-level issues that affect project association.
    for session in sessions:
        for issue in session.issues:
            if issue.code in {
                DiscoveryIssueCode.AMBIGUOUS_NESTING,
                DiscoveryIssueCode.MISSING_PREPROCESS,
            }:
                issues.append(issue)

    return ProjectDiscovery(
        root_path=root,
        display_name=root.name or "project",
        kind=kind,
        sessions=tuple(sessions),
        issues=tuple(issues),
    )
