"""Classify Hala Resident-Intruder sessions for resume → legacy_framewise_v1.

Read-only inventory. Writes one JSON report (and optional per-session files).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from behavioral_arbitration.run_discovery import find_latest_eligible_s8_run
from behavioral_inference.run_discovery import (
    find_latest_eligible_s6_for_s7,
    find_latest_eligible_s7_run,
)
from pose_finalization.accepted_s3 import find_latest_accepted_s3_run
from results_packaging.legacy_framewise_csv.contracts import LEGACY_FRAMEWISE_CSV_FILENAME
from results_packaging.media_csv_export.contracts import RESULTS_EXPORTS_DIRNAME
from results_packaging.media_csv_export.discovery import summarize_export_project
from results_packaging.run_discovery import (
    find_latest_eligible_s9_run,
    summarize_results_package_project,
)

DEFAULT_PARENT = Path(
    r"\\192.114.21.31\Engelhard-Lab-Nas\Miniscope\Miniscope\Hala\Resident Intruder SLEAP"
)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "tmp_hala_ri_batch" / "INVENTORY_REPORT.json"

Action = str  # skip_done | export_legacy_csv | resume_from_sN | blocked


@dataclass(frozen=True)
class SessionStatus:
    session_name: str
    session_root: str
    next_action: Action
    reason: str
    legacy_csv_paths: tuple[str, ...]
    eligible_s9_run_id: str | None
    eligible_s9_run_dir: str | None
    eligible_s8_run_id: str | None
    eligible_s7_run_id: str | None
    eligible_s6_run_id: str | None
    accepted_s3_run_id: str | None
    s9_run_count: int
    s9_eligible_count: int
    s9_classifications: tuple[str, ...]
    export_job_count: int
    extras: dict[str, Any]


def _list_legacy_csv_paths(session_root: Path) -> list[Path]:
    exports = session_root / RESULTS_EXPORTS_DIRNAME
    if not exports.is_dir():
        return []
    return sorted(exports.rglob(LEGACY_FRAMEWISE_CSV_FILENAME))


def _completed_legacy_jobs(session_root: Path) -> list[dict[str, Any]]:
    """Detect completed legacy CSV jobs via filesystem (avoids export-job sort bugs)."""

    out: list[dict[str, Any]] = []
    for csv_path in _list_legacy_csv_paths(session_root):
        job_dir = csv_path.parent
        meta_path = job_dir / "run_meta.json"
        status = None
        n_rows = None
        source_package_run_id = None
        if meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                meta = {}
            if isinstance(meta, dict):
                status = meta.get("status")
                n_rows = meta.get("n_rows")
                source_package_run_id = meta.get("source_package_run_id") or meta.get(
                    "package_run_id"
                )
        if status not in (None, "completed", "completed_with_warnings"):
            continue
        # Treat present CSV + missing/completed meta as done.
        if status is None:
            status = "completed"
        out.append(
            {
                "export_job_id": job_dir.name,
                "job_dir": str(job_dir),
                "status": status,
                "csv_path": str(csv_path),
                "n_rows": n_rows,
                "source_package_run_id": source_package_run_id,
            }
        )
    return out


def _safe_export_summary(session_root: Path) -> dict[str, Any]:
    try:
        summary = summarize_export_project(session_root)
    except Exception as exc:  # noqa: BLE001
        return {
            "error": str(exc),
            "job_count": len(list((session_root / RESULTS_EXPORTS_DIRNAME).glob("*")))
            if (session_root / RESULTS_EXPORTS_DIRNAME).is_dir()
            else 0,
            "job_kinds_present": [],
        }
    return {
        "error": None,
        "job_count": len(summary.jobs),
        "job_kinds_present": sorted({j.job_kind or "?" for j in summary.jobs}),
    }


def classify_session(session_root: Path) -> SessionStatus:
    name = session_root.name
    root = session_root.expanduser().resolve()
    legacy_paths = _list_legacy_csv_paths(root)
    completed_legacy = _completed_legacy_jobs(root)

    s9_summary = None
    s9_err: str | None = None
    try:
        s9_summary = summarize_results_package_project(root)
    except Exception as exc:  # noqa: BLE001 — inventory must continue
        s9_err = str(exc)

    eligible_s9 = None
    try:
        eligible_s9 = find_latest_eligible_s9_run(root)
    except Exception as exc:  # noqa: BLE001
        s9_err = s9_err or str(exc)

    eligible_s8 = None
    try:
        eligible_s8 = find_latest_eligible_s8_run(root)
    except Exception:
        eligible_s8 = None

    eligible_s7 = None
    try:
        eligible_s7 = find_latest_eligible_s7_run(root)
    except Exception:
        eligible_s7 = None

    eligible_s6 = None
    try:
        eligible_s6 = find_latest_eligible_s6_for_s7(root)
    except Exception:
        eligible_s6 = None

    accepted_s3 = None
    try:
        accepted_s3 = find_latest_accepted_s3_run(root)
    except Exception:
        accepted_s3 = None

    classifications = tuple(
        run.classification for run in (s9_summary.runs if s9_summary else ())
    )
    export_info = (
        _safe_export_summary(root)
        if (root / RESULTS_EXPORTS_DIRNAME).is_dir()
        else {"error": None, "job_count": 0, "job_kinds_present": []}
    )
    extras: dict[str, Any] = {
        "completed_legacy_jobs": completed_legacy,
        "s9_error": s9_err,
        "eligible_s8_available": (
            None if s9_summary is None else s9_summary.eligible_s8_available
        ),
        "no_s8_reason": None if s9_summary is None else s9_summary.no_s8_reason,
        "has_preprocess": (root / "preprocess").is_dir(),
        "job_kinds_present": export_info["job_kinds_present"],
        "export_summary_error": export_info["error"],
    }

    if completed_legacy:
        action: Action = "skip_done"
        reason = (
            f"Found {len(completed_legacy)} completed "
            f"{LEGACY_FRAMEWISE_CSV_FILENAME} export(s)."
        )
    elif eligible_s9 is not None:
        action = "export_legacy_csv"
        reason = f"Eligible S9 package available: {eligible_s9.run_id}"
    elif eligible_s8 is not None:
        action = "resume_from_s9"
        reason = f"Eligible S8 present ({eligible_s8.run_id}); need S9 package."
    elif eligible_s7 is not None:
        action = "resume_from_s8"
        reason = f"Eligible S7 present ({eligible_s7.run_id}); need S8 arbitration."
    elif eligible_s6 is not None:
        action = "resume_from_s7"
        reason = f"Eligible S6 present ({eligible_s6.run_id}); need S7 inference."
    elif accepted_s3 is not None:
        action = "resume_from_s4"
        reason = (
            f"Accepted S3 present ({accepted_s3.run_id}) but no eligible S6 for S7."
        )
    else:
        action = "blocked"
        reason = (
            "No completed legacy CSV, no eligible S9/S8/S7/S6, and no accepted S3."
        )
        if s9_err:
            reason = f"{reason} S9 discovery error: {s9_err}"

    return SessionStatus(
        session_name=name,
        session_root=str(root),
        next_action=action,
        reason=reason,
        legacy_csv_paths=tuple(str(p) for p in legacy_paths),
        eligible_s9_run_id=None if eligible_s9 is None else eligible_s9.run_id,
        eligible_s9_run_dir=None if eligible_s9 is None else str(eligible_s9.run_dir),
        eligible_s8_run_id=None if eligible_s8 is None else eligible_s8.run_id,
        eligible_s7_run_id=None if eligible_s7 is None else eligible_s7.run_id,
        eligible_s6_run_id=None if eligible_s6 is None else eligible_s6.run_id,
        accepted_s3_run_id=None if accepted_s3 is None else accepted_s3.run_id,
        s9_run_count=0 if s9_summary is None else len(s9_summary.runs),
        s9_eligible_count=0 if s9_summary is None else len(s9_summary.eligible_runs),
        s9_classifications=classifications,
        export_job_count=int(export_info["job_count"]),
        extras=extras,
    )


def _discover_sessions(parent: Path) -> list[Path]:
    if not parent.is_dir():
        raise FileNotFoundError(f"Parent path not accessible: {parent}")
    sessions = [
        p
        for p in sorted(parent.iterdir(), key=lambda x: x.name)
        if p.is_dir() and not p.name.startswith(".") and not p.name.startswith("_")
    ]
    return sessions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parent",
        type=Path,
        default=DEFAULT_PARENT,
        help="Parent folder containing session directories",
    )
    parser.add_argument(
        "--session-root",
        type=Path,
        action="append",
        default=None,
        help="Classify a single session (repeatable). Overrides --parent listing.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Inventory JSON output path",
    )
    parser.add_argument(
        "--per-session-dir",
        type=Path,
        default=None,
        help="Optional directory for one JSON file per session",
    )
    args = parser.parse_args(argv)

    if args.session_root:
        sessions = [Path(p).expanduser().resolve() for p in args.session_root]
    else:
        sessions = _discover_sessions(Path(args.parent))

    results: list[SessionStatus] = []
    for session in sessions:
        print(f"Scanning {session} …", flush=True)
        status = classify_session(session)
        results.append(status)
        print(f"  → {status.next_action}: {status.reason}", flush=True)
        if args.per_session_dir is not None:
            args.per_session_dir.mkdir(parents=True, exist_ok=True)
            out_one = args.per_session_dir / f"{status.session_name}.json"
            out_one.write_text(
                json.dumps(asdict(status), indent=2),
                encoding="utf-8",
            )

    by_action: dict[str, list[str]] = {}
    for row in results:
        by_action.setdefault(row.next_action, []).append(row.session_name)

    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "parent": str(Path(args.parent).expanduser().resolve())
        if not args.session_root
        else None,
        "n_sessions": len(results),
        "by_action": {k: sorted(v) for k, v in sorted(by_action.items())},
        "sessions": [asdict(r) for r in results],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}", flush=True)
    print(json.dumps(report["by_action"], indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
