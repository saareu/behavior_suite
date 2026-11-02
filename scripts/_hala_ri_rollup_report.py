"""Merge inventory + export + resume reports into FINAL_REPORT.json."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BATCH_DIR = REPO_ROOT / "tmp_hala_ri_batch"
DEFAULT_OUT = DEFAULT_BATCH_DIR / "FINAL_REPORT.json"


def _load_json(path: Path) -> Any | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _session_rows(batch_dir: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    inventory = _load_json(batch_dir / "INVENTORY_REPORT.json")
    if isinstance(inventory, dict):
        for sess in inventory.get("sessions", []):
            name = sess.get("session_name")
            if not name:
                continue
            rows[name] = {
                "session_name": name,
                "session_root": sess.get("session_root"),
                "inventory_next_action": sess.get("next_action"),
                "inventory_reason": sess.get("reason"),
                "eligible_s9_run_id": sess.get("eligible_s9_run_id"),
            }

    exports_dir = batch_dir / "exports"
    if exports_dir.is_dir():
        for path in sorted(exports_dir.glob("*_export.json")):
            payload = _load_json(path)
            if not isinstance(payload, dict):
                continue
            name = payload.get("session_name") or path.name.replace("_export.json", "")
            row = rows.setdefault(name, {"session_name": name})
            row["export"] = payload

    resume_dir = batch_dir / "resume"
    if resume_dir.is_dir():
        for path in sorted(resume_dir.glob("*_resume.json")):
            payload = _load_json(path)
            if not isinstance(payload, dict):
                continue
            name = payload.get("session_name") or path.name.replace("_resume.json", "")
            row = rows.setdefault(name, {"session_name": name})
            row["resume"] = payload

    return rows


def _finalize_row(row: dict[str, Any]) -> dict[str, Any]:
    export = row.get("export") or {}
    resume = row.get("resume") or {}
    csv_path = (
        resume.get("csv_path")
        or export.get("csv_path")
        or None
    )
    success = False
    if resume:
        success = bool(resume.get("success"))
    elif export:
        success = bool(export.get("success"))
    elif row.get("inventory_next_action") == "skip_done":
        success = True

    dense = None
    format_qc = None
    if isinstance(export, dict):
        dense = export.get("dense_continuity")
        format_qc = export.get("format_qc")
    if resume and isinstance(resume.get("steps"), list):
        for step in resume["steps"]:
            if step.get("stage") == "legacy_csv":
                dense = step.get("dense_continuity") or dense
                format_qc = step.get("format_qc") or format_qc
                csv_path = step.get("csv_path") or csv_path

    return {
        **row,
        "final_success": success,
        "final_csv_path": csv_path,
        "dense_continuity": dense,
        "format_qc": format_qc,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, default=DEFAULT_BATCH_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)

    rows = [_finalize_row(r) for r in _session_rows(args.batch_dir).values()]
    rows.sort(key=lambda r: str(r.get("session_name") or ""))
    n_ok = sum(1 for r in rows if r.get("final_success"))
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "batch_dir": str(args.batch_dir.resolve()),
        "n_sessions": len(rows),
        "n_success": n_ok,
        "n_failed": len(rows) - n_ok,
        "sessions": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}", flush=True)
    print(
        json.dumps(
            {
                "n_sessions": report["n_sessions"],
                "n_success": report["n_success"],
                "n_failed": report["n_failed"],
                "ok": [r["session_name"] for r in rows if r.get("final_success")],
                "failed": [
                    r["session_name"] for r in rows if not r.get("final_success")
                ],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0 if report["n_failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
