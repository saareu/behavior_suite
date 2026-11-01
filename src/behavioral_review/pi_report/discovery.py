"""Discover finalized PI report directories under results_exports/."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from behavioral_review.pi_report.contracts import (
    PI_REPORT_RUN_PREFIX,
    REPORT_MANIFEST_FILENAME,
    RESULTS_EXPORTS_DIRNAME,
    RUN_META_FILENAME,
)
from results_packaging.export import load_json_object


@dataclass(frozen=True, slots=True)
class PiReportSummary:
    """Discovery summary for one finalized (or failed) PI report job."""

    report_id: str
    report_dir: Path
    status: str | None
    source_type: str | None
    package_run_id: str | None
    review_run_id: str | None
    n_bouts: int | None
    created_at_utc: str | None
    completed_at_utc: str | None
    has_manifest: bool
    has_index_html: bool
    report_scope: str | None = None
    has_zip: bool = False
    zip_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "report_id": self.report_id,
            "report_dir": str(self.report_dir),
            "status": self.status,
            "source_type": self.source_type,
            "package_run_id": self.package_run_id,
            "review_run_id": self.review_run_id,
            "n_bouts": self.n_bouts,
            "created_at_utc": self.created_at_utc,
            "completed_at_utc": self.completed_at_utc,
            "has_manifest": self.has_manifest,
            "has_index_html": self.has_index_html,
            "report_scope": self.report_scope,
            "has_zip": self.has_zip,
            "zip_path": str(self.zip_path) if self.zip_path is not None else None,
        }


@dataclass(frozen=True, slots=True)
class PiReportProjectSummary:
    session_root: Path
    exports_root: Path
    reports: tuple[PiReportSummary, ...]

    @property
    def completed_reports(self) -> tuple[PiReportSummary, ...]:
        return tuple(
            r
            for r in self.reports
            if r.status in {"completed", "completed_with_warnings"} and r.has_manifest
        )


def summarize_pi_report_project(
    session_root: Path | str,
    *,
    exports_root: Path | str | None = None,
) -> PiReportProjectSummary:
    root = Path(session_root).expanduser().resolve()
    exports = (
        Path(exports_root).expanduser().resolve()
        if exports_root is not None
        else (root / RESULTS_EXPORTS_DIRNAME)
    )
    reports: list[PiReportSummary] = []
    if exports.is_dir():
        for child in sorted(exports.iterdir()):
            if not child.is_dir():
                continue
            if not child.name.startswith(f"{PI_REPORT_RUN_PREFIX}__"):
                continue
            if child.name.startswith("_"):
                continue
            reports.append(_summarize_report(child))
    return PiReportProjectSummary(
        session_root=root,
        exports_root=exports,
        reports=tuple(reports),
    )


def _summarize_report(report_dir: Path) -> PiReportSummary:
    meta_path = report_dir / RUN_META_FILENAME
    manifest_path = report_dir / REPORT_MANIFEST_FILENAME
    status = None
    source_type = None
    package_run_id = None
    review_run_id = None
    n_bouts = None
    created_at = None
    completed_at = None
    report_scope = None
    if meta_path.is_file():
        try:
            meta = load_json_object(meta_path)
            status = meta.get("status")
            source_type = meta.get("source_type")
            package_run_id = meta.get("package_run_id")
            review_run_id = meta.get("review_run_id")
            created_at = meta.get("created_at_utc")
            completed_at = meta.get("completed_at_utc")
            counts_raw = meta.get("counts")
            counts: dict[str, Any] = counts_raw if isinstance(counts_raw, dict) else {}
            n_bouts = counts.get("n_bouts")
            report_scope = counts.get("report_scope")
            settings_raw = meta.get("settings")
            settings: dict[str, Any] = (
                settings_raw if isinstance(settings_raw, dict) else {}
            )
            report_scope = report_scope or settings.get("report_scope")
        except Exception:  # noqa: BLE001
            pass
    if manifest_path.is_file():
        try:
            manifest = load_json_object(manifest_path)
            source_type = source_type or manifest.get("source_type")
            package_run_id = package_run_id or manifest.get("source_package_run_id")
            review_run_id = review_run_id or manifest.get("source_review_run_id")
            n_bouts = n_bouts if n_bouts is not None else manifest.get("n_bouts")
            report_scope = report_scope or manifest.get("report_scope")
        except Exception:  # noqa: BLE001
            pass
    zip_candidate = report_dir.parent / f"{report_dir.name}.zip"
    has_zip = zip_candidate.is_file()
    return PiReportSummary(
        report_id=report_dir.name,
        report_dir=report_dir,
        status=None if status is None else str(status),
        source_type=None if source_type is None else str(source_type),
        package_run_id=None if package_run_id is None else str(package_run_id),
        review_run_id=None if review_run_id is None else str(review_run_id),
        n_bouts=int(n_bouts) if isinstance(n_bouts, int) else None,
        created_at_utc=None if created_at is None else str(created_at),
        completed_at_utc=None if completed_at is None else str(completed_at),
        has_manifest=manifest_path.is_file(),
        has_index_html=(report_dir / "index.html").is_file(),
        report_scope=None if report_scope is None else str(report_scope),
        has_zip=has_zip,
        zip_path=zip_candidate if has_zip else None,
    )


__all__ = [
    "PiReportProjectSummary",
    "PiReportSummary",
    "summarize_pi_report_project",
]
