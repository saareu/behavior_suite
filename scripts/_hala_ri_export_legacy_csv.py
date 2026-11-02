"""Export automated legacy_framewise_v1 CSV for one session (or inventory batch).

Uses the latest currently eligible S9 package. Does not allow stale sources.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from results_packaging.legacy_framewise_csv import (
    BEHAVIOR_SOURCE_AUTOMATED,
    LEGACY_FRAMEWISE_CSV_FILENAME,
    LegacyFramewiseExportRequest,
    run_legacy_framewise_csv_export,
    validate_legacy_framewise_csv_bytes,
)
from results_packaging.run_discovery import find_latest_eligible_s9_run

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = REPO_ROOT / "tmp_hala_ri_batch" / "exports"


def _dense_continuity(csv_path: Path) -> dict[str, Any]:
    df = pd.read_csv(csv_path)
    dense = df["dense_frame_idx"].astype(int)
    expected = set(range(int(dense.min()), int(dense.max()) + 1))
    actual = set(dense.tolist())
    return {
        "n_rows": int(len(df)),
        "dense_min": int(dense.min()),
        "dense_max": int(dense.max()),
        "n_unique_dense": int(dense.nunique()),
        "n_duplicate_dense": int(dense.duplicated().sum()),
        "n_missing_dense": int(len(expected - actual)),
        "contiguous_from_zero": bool(
            dense.min() == 0
            and dense.max() == len(df) - 1
            and dense.nunique() == len(df)
            and not dense.duplicated().any()
        ),
    }


def _already_done(session_root: Path) -> Path | None:
    exports = session_root / "results_exports"
    if not exports.is_dir():
        return None
    hits = sorted(exports.rglob(LEGACY_FRAMEWISE_CSV_FILENAME))
    for path in reversed(hits):
        meta = path.parent / "run_meta.json"
        if not meta.is_file():
            continue
        try:
            payload = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if payload.get("status") in ("completed", "completed_with_warnings"):
            return path
    return hits[-1] if hits else None


def export_one(
    session_root: Path,
    *,
    package_dir: Path | None = None,
    skip_if_done: bool = True,
    timestamp: str | None = None,
) -> dict[str, Any]:
    root = session_root.expanduser().resolve()
    report: dict[str, Any] = {
        "session_name": root.name,
        "session_root": str(root),
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
    }

    if skip_if_done:
        existing = _already_done(root)
        if existing is not None:
            report.update(
                {
                    "success": True,
                    "skipped": True,
                    "reason": "legacy_framewise_v1.csv already present",
                    "csv_path": str(existing),
                    "job_dir": str(existing.parent),
                }
            )
            if existing.is_file():
                report["format_qc"] = validate_legacy_framewise_csv_bytes(
                    existing.read_bytes()
                )
                report["dense_continuity"] = _dense_continuity(existing)
            return report

    resolved_package = package_dir
    if resolved_package is None:
        eligible = find_latest_eligible_s9_run(root)
        if eligible is None:
            report.update(
                {
                    "success": False,
                    "skipped": False,
                    "error_message": "No currently eligible S9 package",
                }
            )
            return report
        resolved_package = eligible.run_dir
        report["package_run_id"] = eligible.run_id
    else:
        resolved_package = Path(resolved_package).expanduser().resolve()
        report["package_run_id"] = resolved_package.name

    report["package_dir"] = str(resolved_package)
    print(
        f"Exporting legacy_framewise_v1 for {root.name} "
        f"from {resolved_package.name} …",
        flush=True,
    )

    result = run_legacy_framewise_csv_export(
        LegacyFramewiseExportRequest(
            session_root=root,
            package_dir=resolved_package,
            behavior_source_kind=BEHAVIOR_SOURCE_AUTOMATED,
            timestamp=timestamp,
            allow_stale_source=False,
        )
    )
    report.update(
        {
            "success": bool(result.success),
            "skipped": False,
            "status": result.status,
            "error_message": result.error_message,
            "job_dir": str(result.job_dir) if result.job_dir else None,
            "csv_path": str(result.csv_path) if result.csv_path else None,
            "n_rows": result.n_rows,
            "warnings": list(result.warnings),
            "diagnostics": dict(result.diagnostics),
        }
    )
    if result.csv_path and result.csv_path.is_file():
        report["format_qc"] = validate_legacy_framewise_csv_bytes(
            result.csv_path.read_bytes()
        )
        report["dense_continuity"] = _dense_continuity(result.csv_path)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session-root",
        type=Path,
        action="append",
        required=False,
        help="Session root to export (repeatable)",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        default=None,
        help="Inventory JSON; export sessions with next_action=export_legacy_csv",
    )
    parser.add_argument(
        "--package-dir",
        type=Path,
        default=None,
        help="Optional explicit S9 package (single-session only)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Export even if a completed legacy CSV already exists",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=DEFAULT_REPORT_DIR,
        help="Directory for per-session export reports",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Optional immutable export timestamp suffix",
    )
    args = parser.parse_args(argv)

    sessions: list[Path] = []
    package_by_session: dict[str, Path] = {}
    if args.inventory is not None:
        inv = json.loads(Path(args.inventory).read_text(encoding="utf-8"))
        for row in inv.get("sessions", []):
            if row.get("next_action") != "export_legacy_csv":
                continue
            root = Path(row["session_root"])
            sessions.append(root)
            if row.get("eligible_s9_run_dir"):
                package_by_session[root.name] = Path(row["eligible_s9_run_dir"])
    if args.session_root:
        sessions.extend(Path(p) for p in args.session_root)

    # De-dupe while preserving order.
    seen: set[str] = set()
    unique: list[Path] = []
    for s in sessions:
        key = str(s.expanduser().resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(s)

    if not unique:
        print("No sessions to export.", flush=True)
        return 2

    if args.package_dir is not None and len(unique) != 1:
        print("--package-dir requires exactly one --session-root", flush=True)
        return 2

    args.report_dir.mkdir(parents=True, exist_ok=True)
    reports: list[dict[str, Any]] = []
    failures = 0
    for session in unique:
        pkg = args.package_dir
        if pkg is None:
            pkg = package_by_session.get(session.name)
        try:
            report = export_one(
                session,
                package_dir=pkg,
                skip_if_done=not args.force,
                timestamp=args.timestamp,
            )
        except Exception as exc:  # noqa: BLE001 — per-session isolation
            report = {
                "session_name": session.name,
                "session_root": str(session),
                "success": False,
                "error_message": str(exc),
            }
        reports.append(report)
        out = args.report_dir / f"{session.name}_export.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(
            f"  {session.name}: success={report.get('success')} "
            f"skipped={report.get('skipped')} "
            f"csv={report.get('csv_path')}",
            flush=True,
        )
        if not report.get("success"):
            failures += 1

    batch_path = args.report_dir / "EXPORT_BATCH_REPORT.json"
    batch_path.write_text(
        json.dumps(
            {
                "generated_at": datetime.now()
                .astimezone()
                .isoformat(timespec="seconds"),
                "n_sessions": len(reports),
                "n_failures": failures,
                "reports": reports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {batch_path}", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
