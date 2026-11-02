"""E-session legacy_framewise_v1 export smoke (schema/format + dense continuity)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

from results_packaging.legacy_framewise_csv import (
    LEGACY_FRAMEWISE_COLUMNS,
    LEGACY_FRAMEWISE_CSV_FILENAME,
    LegacyFramewiseExportRequest,
    run_legacy_framewise_csv_export,
    validate_legacy_framewise_csv_bytes,
)

SESSION = Path(r"C:\behavior_suite\test_cases\E_full_session_integration")
PACKAGE = SESSION / "results_package" / "results_package__20261028T181714"
REF = Path(r"C:\behavior_suite_legacy_csv\tmp_fixtures\9722.csv")
OUT_ROOT = Path(r"C:\behavior_suite_legacy_csv\tmp_legacy_framewise_e_session")
EVIDENCE = OUT_ROOT / "E_SESSION_LEGACY_FRAMEWISE_REPORT.json"

# Prefer a published S10 review when present.
_REVIEW_CANDIDATES = sorted(
    (SESSION / "behavioral_review").glob("behavioral_review__*")
) if (SESSION / "behavioral_review").is_dir() else []


def _dense_continuity(csv_path: Path) -> dict:
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
        "missing_dense_examples": sorted(expected - actual)[:10],
        "contiguous_from_zero": bool(
            dense.min() == 0
            and dense.max() == len(df) - 1
            and dense.nunique() == len(df)
            and not dense.duplicated().any()
        ),
        "frame_min": int(df["frame"].min()),
        "frame_max": int(df["frame"].max()),
        "n_unique_original_frames": int(df["frame"].nunique()),
        "n_duplicate_original_frame_rows": int(len(df) - df["frame"].nunique()),
        "nonzero_trim_offset": bool(int(df["frame"].min()) != 0),
        "focal0_empty": int(
            df["behaviour_label_focal_0"].isna().sum()
            + (df["behaviour_label_focal_0"].astype(str).str.strip() == "").sum()
        ),
        "focal1_empty": int(
            df["behaviour_label_focal_1"].isna().sum()
            + (df["behaviour_label_focal_1"].astype(str).str.strip() == "").sum()
        ),
        "sbs_values": sorted(
            int(v) for v in df["pred_side_by_side"].dropna().unique().tolist()
        ),
    }


def _format_compare(csv_path: Path) -> dict:
    payload = csv_path.read_bytes()
    qc = validate_legacy_framewise_csv_bytes(payload)
    out: dict = {"format_qc": qc, "crlf": b"\r\n" in payload}
    if REF.is_file():
        ref_header = next(
            csv.reader([REF.read_text(encoding="utf-8").splitlines()[0]])
        )
        out["reference_header_equal"] = ref_header == list(LEGACY_FRAMEWISE_COLUMNS)
        out["reference_path"] = str(REF)
        out["reference_n_cols"] = len(ref_header)
        # Spot-check first data row pose formatting style (.0 or empty).
        lines = payload.decode("utf-8").split("\r\n")
        if len(lines) > 1:
            row = next(csv.reader([lines[1]]))
            pose = row[5:]
            out["first_row_pose_all_dot_zero_or_empty"] = all(
                (v == "" or v.endswith(".0")) for v in pose
            )
            out["first_row_sbs_binary"] = row[4] in {"0", "1"}
    return out


def _run_one(*, kind: str, timestamp: str, review_run_dir: Path | None = None) -> dict:
    request = LegacyFramewiseExportRequest(
        session_root=SESSION,
        package_dir=PACKAGE if kind == "automated_s8" else None,
        review_run_dir=review_run_dir,
        behavior_source_kind=kind,
        timestamp=timestamp,
        output_root=OUT_ROOT,
        allow_stale_source=True,
    )
    result = run_legacy_framewise_csv_export(request)
    report: dict = {
        "success": result.success,
        "status": result.status,
        "error_message": result.error_message,
        "job_dir": str(result.job_dir),
        "n_rows": result.n_rows,
        "csv_path": str(result.csv_path) if result.csv_path else None,
        "warnings": list(result.warnings),
        "diagnostics": dict(result.diagnostics),
        "behavior_source_kind": kind,
    }
    if result.csv_path and result.csv_path.is_file():
        report["dense_continuity"] = _dense_continuity(result.csv_path)
        report["format_compare"] = _format_compare(result.csv_path)
        report["header"] = list(LEGACY_FRAMEWISE_COLUMNS)
        report["filename"] = LEGACY_FRAMEWISE_CSV_FILENAME
        manifest = result.job_dir / "export_job_manifest.json"
        if manifest.is_file():
            meta = json.loads(manifest.read_text(encoding="utf-8"))
            report["pose_source"] = meta.get("pose_source")
            report["pinned_source_digests"] = meta.get("pinned_source_digests")
    return report


def main() -> int:
    if not PACKAGE.is_dir():
        print(f"SKIP: package missing: {PACKAGE}")
        return 2
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # Fresh timestamps so reruns do not collide with prior immutable jobs.
    auto = _run_one(kind="automated_s8", timestamp="20261102THARDEN_S9")
    reviewed = None
    published = None
    for cand in reversed(_REVIEW_CANDIDATES):
        meta_path = cand / "run_meta.json"
        if not meta_path.is_file():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        state = str(meta.get("publication_state") or "").lower()
        if state and state != "draft" and (cand / "reviewed_behavior_states.parquet").is_file():
            published = cand
            break
    if published is not None:
        reviewed = _run_one(
            kind="published_s10",
            timestamp="20261102THARDEN_S10",
            review_run_dir=published,
        )
    else:
        reviewed = {
            "success": False,
            "skipped": True,
            "reason": "No published S10 review with reviewed_behavior_states.parquet",
        }

    report = {
        "automated_s8": auto,
        "published_s10": reviewed,
        "note": (
            "E-session values are not expected to match 9722.csv recording values; "
            "schema/format and dense continuity are compared. "
            "Reported '0–36383' confusion is original-frame min (raw_decode), "
            "not dense max."
        ),
    }
    EVIDENCE.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    ok = bool(auto.get("success")) and (
        reviewed.get("skipped") or reviewed.get("success")
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
