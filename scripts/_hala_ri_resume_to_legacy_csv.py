"""Resume one session from the first missing eligible stage through legacy CSV.

Intended for Cohort B (e.g. 9522): S7 → S8 → S9 → automated legacy_framewise_v1.
Uses the lab model registry from QSettings / --registry-root / repo models/.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Any

from behavioral_arbitration.run_discovery import find_latest_eligible_s8_run
from behavioral_arbitration.runner import (
    BehavioralArbitrationRequest,
    run_behavioral_arbitration_publication,
)
from behavioral_inference.config import (
    BehavioralInferenceConfig,
    load_behavioral_inference_config,
)
from behavioral_inference.gui_persistence import (
    load_persisted_registry_root,
    suggest_lab_model_registry,
)
from behavioral_inference.run_discovery import (
    find_latest_eligible_s6_for_s7,
    find_latest_eligible_s7_run,
)
from behavioral_inference.runner import (
    BehavioralInferenceRequest,
    run_behavioral_inference_publication,
)
from combined_features.config import ModelRegistrySettings
from pose_finalization.accepted_s3 import find_latest_accepted_s3_run
from results_packaging.run_discovery import find_latest_eligible_s9_run
from results_packaging.runner import run_publication_from_session

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT_DIR = REPO_ROOT / "tmp_hala_ri_batch" / "resume"


def _progress(phase: str, message: str, **_: Any) -> None:
    print(f"[{phase}] {message}", flush=True)


def _resolve_registry(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Model registry not found: {path}")
        return path
    try:
        from PySide6.QtCore import QSettings

        persisted = load_persisted_registry_root(
            QSettings("BehaviorSuite", "BehaviorSuite")
        )
        if persisted is not None and Path(persisted).is_dir():
            return Path(persisted)
    except Exception:  # noqa: BLE001
        pass
    suggested = suggest_lab_model_registry(REPO_ROOT)
    if suggested is not None:
        return suggested
    raise FileNotFoundError(
        "No model registry configured. Pass --registry-root or set S7 GUI path."
    )


def _s7_config(registry_root: Path) -> BehavioralInferenceConfig:
    base = load_behavioral_inference_config(None)
    return replace(
        base,
        model_registry=ModelRegistrySettings(
            root=registry_root,
            dyadic_subdir=base.model_registry.dyadic_subdir,
            focal_subdir=base.model_registry.focal_subdir,
            include_obsolete=base.model_registry.include_obsolete,
            include_focal=base.model_registry.include_focal,
        ),
    ).validated()


def resume_session(
    session_root: Path,
    *,
    registry_root: Path,
    report_dir: Path,
    run_purpose: str = "hala_ri_batch",
) -> dict[str, Any]:
    root = session_root.expanduser().resolve()
    steps: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "session_name": root.name,
        "session_root": str(root),
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "registry_root": str(registry_root),
        "steps": steps,
    }

    accepted = find_latest_accepted_s3_run(root)
    steps.append(
        {
            "stage": "s3_accept_check",
            "ok": accepted is not None,
            "run_id": None if accepted is None else accepted.run_id,
        }
    )
    if accepted is None:
        report["success"] = False
        report["error_message"] = "No accepted S3 tracked_pose; cannot resume S7+."
        return report

    # S7
    s7 = find_latest_eligible_s7_run(root)
    if s7 is None:
        s6 = find_latest_eligible_s6_for_s7(root)
        if s6 is None:
            report["success"] = False
            report["error_message"] = "No eligible S6 for S7."
            return report
        print(f"Running S7 from {s6.run_id} …", flush=True)
        s7_result = run_behavioral_inference_publication(
            BehavioralInferenceRequest(
                session_root=root,
                s6_run_dir=s6.run_dir,
                config=_s7_config(registry_root),
                run_purpose=run_purpose,
            ),
            progress_callback=lambda p: _progress(
                getattr(p, "phase", "s7"),
                getattr(p, "message", ""),
            ),
        )
        steps.append(
            {
                "stage": "s7",
                "ok": bool(s7_result.success),
                "run_id": s7_result.run_id,
                "status": s7_result.status,
                "error_message": s7_result.error_message,
                "run_dir": str(s7_result.run_dir),
            }
        )
        if not s7_result.success:
            report["success"] = False
            report["error_message"] = s7_result.error_message
            return report
        s7_dir = s7_result.run_dir
    else:
        steps.append(
            {
                "stage": "s7",
                "ok": True,
                "skipped": True,
                "run_id": s7.run_id,
                "run_dir": str(s7.run_dir),
            }
        )
        s7_dir = s7.run_dir

    # S8
    s8 = find_latest_eligible_s8_run(root)
    if s8 is None:
        print(f"Running S8 from {Path(s7_dir).name} …", flush=True)
        s8_result = run_behavioral_arbitration_publication(
            BehavioralArbitrationRequest(
                session_root=root,
                s7_run_dir=s7_dir,
                run_purpose=run_purpose,
            ),
            progress_callback=lambda p: _progress(
                getattr(p, "phase", "s8"),
                getattr(p, "message", ""),
            ),
        )
        steps.append(
            {
                "stage": "s8",
                "ok": bool(s8_result.success),
                "run_id": s8_result.run_id,
                "status": s8_result.status,
                "error_message": s8_result.error_message,
                "run_dir": str(s8_result.run_dir),
            }
        )
        if not s8_result.success:
            report["success"] = False
            report["error_message"] = s8_result.error_message
            return report
        s8_dir = s8_result.run_dir
    else:
        steps.append(
            {
                "stage": "s8",
                "ok": True,
                "skipped": True,
                "run_id": s8.run_id,
                "run_dir": str(s8.run_dir),
            }
        )
        s8_dir = s8.run_dir

    # S9
    s9 = find_latest_eligible_s9_run(root)
    if s9 is None:
        print(f"Running S9 from {Path(s8_dir).name} …", flush=True)
        s9_result = run_publication_from_session(
            root,
            s8_run_dir=s8_dir,
            progress_callback=lambda p: _progress(
                getattr(p, "phase", "s9"),
                getattr(p, "message", ""),
            ),
        )
        steps.append(
            {
                "stage": "s9",
                "ok": bool(s9_result.success),
                "run_id": s9_result.run_id,
                "status": s9_result.status,
                "error_message": s9_result.error_message,
                "run_dir": str(s9_result.run_dir),
            }
        )
        if not s9_result.success:
            report["success"] = False
            report["error_message"] = s9_result.error_message
            return report
        package_dir = s9_result.run_dir
    else:
        steps.append(
            {
                "stage": "s9",
                "ok": True,
                "skipped": True,
                "run_id": s9.run_id,
                "run_dir": str(s9.run_dir),
            }
        )
        package_dir = s9.run_dir

    # Import export helper without packaging scripts as a module.
    from importlib.util import module_from_spec, spec_from_file_location

    export_path = Path(__file__).with_name("_hala_ri_export_legacy_csv.py")
    spec = spec_from_file_location("hala_ri_export_legacy_csv", export_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {export_path}")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    export_report = mod.export_one(
        root,
        package_dir=package_dir,
        skip_if_done=True,
    )
    steps.append({"stage": "legacy_csv", **export_report})
    report["success"] = bool(export_report.get("success"))
    report["error_message"] = export_report.get("error_message")
    report["csv_path"] = export_report.get("csv_path")
    report["finished_at"] = datetime.now().astimezone().isoformat(timespec="seconds")

    report_dir.mkdir(parents=True, exist_ok=True)
    out = report_dir / f"{root.name}_resume.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out}", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-root", type=Path, required=True)
    parser.add_argument("--registry-root", type=Path, default=None)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--run-purpose", type=str, default="hala_ri_batch")
    args = parser.parse_args(argv)

    registry = _resolve_registry(args.registry_root)
    print(f"Using model registry: {registry}", flush=True)
    report = resume_session(
        args.session_root,
        registry_root=registry,
        report_dir=args.report_dir,
        run_purpose=args.run_purpose,
    )
    print(json.dumps({k: report[k] for k in ("success", "error_message", "csv_path") if k in report}, indent=2))
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
