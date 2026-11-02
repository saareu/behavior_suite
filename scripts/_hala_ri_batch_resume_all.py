"""Sequentially resume all inventory sessions that need S7+ through legacy CSV."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INVENTORY = REPO_ROOT / "tmp_hala_ri_batch" / "INVENTORY_REPORT.json"
DEFAULT_REPORT_DIR = REPO_ROOT / "tmp_hala_ri_batch" / "resume"


def _load_resume_module():
    path = Path(__file__).with_name("_hala_ri_resume_to_legacy_csv.py")
    spec = spec_from_file_location("hala_ri_resume_to_legacy_csv", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--registry-root", type=Path, default=None)
    parser.add_argument(
        "--actions",
        type=str,
        default="resume_from_s7,resume_from_s8,resume_from_s9,export_legacy_csv",
        help="Comma-separated inventory next_action values to process",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Optional comma-separated session names to include",
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the batch after the first failed session",
    )
    args = parser.parse_args(argv)

    inventory = json.loads(args.inventory.read_text(encoding="utf-8"))
    wanted_actions = {a.strip() for a in args.actions.split(",") if a.strip()}
    only = (
        {n.strip() for n in args.only.split(",") if n.strip()}
        if args.only
        else None
    )

    targets: list[dict[str, Any]] = []
    for row in inventory.get("sessions", []):
        name = row.get("session_name")
        action = row.get("next_action")
        if action not in wanted_actions:
            continue
        if only is not None and name not in only:
            continue
        targets.append(row)

    if not targets:
        print("No matching sessions to resume.", flush=True)
        return 2

    mod = _load_resume_module()
    registry = mod._resolve_registry(args.registry_root)
    print(f"Registry: {registry}", flush=True)
    print(
        f"Resuming {len(targets)} session(s): "
        f"{', '.join(t['session_name'] for t in targets)}",
        flush=True,
    )

    results: list[dict[str, Any]] = []
    failures = 0
    for row in targets:
        name = row["session_name"]
        root = Path(row["session_root"])
        print(f"\n===== RESUME {name} ({row['next_action']}) =====", flush=True)
        try:
            report = mod.resume_session(
                root,
                registry_root=registry,
                report_dir=args.report_dir,
                run_purpose="hala_ri_batch",
            )
        except Exception as exc:  # noqa: BLE001
            report = {
                "session_name": name,
                "session_root": str(root),
                "success": False,
                "error_message": str(exc),
            }
            args.report_dir.mkdir(parents=True, exist_ok=True)
            (args.report_dir / f"{name}_resume.json").write_text(
                json.dumps(report, indent=2),
                encoding="utf-8",
            )
        results.append(
            {
                "session_name": name,
                "success": bool(report.get("success")),
                "error_message": report.get("error_message"),
                "csv_path": report.get("csv_path"),
            }
        )
        print(
            f"===== DONE {name}: success={report.get('success')} "
            f"csv={report.get('csv_path')} =====",
            flush=True,
        )
        if not report.get("success"):
            failures += 1
            if args.stop_on_error:
                break

    batch = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "n_targets": len(targets),
        "n_failures": failures,
        "results": results,
    }
    out = args.report_dir / "RESUME_BATCH_REPORT.json"
    args.report_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(batch, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
