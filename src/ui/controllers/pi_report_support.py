"""PI report GUI settings, request identity, and discovery table helpers (B2)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from behavioral_review.gallery import GalleryQuery
from behavioral_review.pi_report import (
    PI_SOURCE_AUTOMATED,
    PI_SOURCE_REVIEWED,
    REPORT_SCOPE_COMPLETE,
    REPORT_SCOPE_SELECTED,
    PiReportSettings,
)
from behavioral_review.pi_report.contracts import (
    DEFAULT_POST_CONTEXT_SEC,
    DEFAULT_PRE_CONTEXT_SEC,
)

PiReportSourceKind = Literal["automated", "reviewed"]
PiReportBoutScope = Literal["all_exclusive", "selected_bouts"]


@dataclass(frozen=True, slots=True)
class PiReportUiSettings:
    """GUI working settings snapshotted into each PI report export request."""

    source_kind: PiReportSourceKind = "automated"
    bout_scope: PiReportBoutScope = "all_exclusive"
    selected_bout_ids: tuple[str, ...] = ()
    pre_context_sec: float = DEFAULT_PRE_CONTEXT_SEC
    post_context_sec: float = DEFAULT_POST_CONTEXT_SEC
    write_zip: bool = True
    # C3-6: overlays off by default; annotated media replaces naked when enabled.
    overlays_enabled: bool = False
    overlay_include_behavior: bool = True
    overlay_include_pose: bool = False
    overlay_pose_run_dir: str | None = None
    overlay_include_contact: bool = True
    overlay_include_sbs: bool = True
    overlay_draw_skeleton: bool = True
    overlay_hud_mode: str = "both"

    def validated(self) -> PiReportUiSettings:
        kind = str(self.source_kind).strip().lower()
        if kind not in {PI_SOURCE_AUTOMATED, PI_SOURCE_REVIEWED}:
            raise ValueError(
                f"PI report source_kind must be automated or reviewed, got {kind!r}."
            )
        scope = str(self.bout_scope).strip().lower()
        if scope not in {"all_exclusive", "selected_bouts"}:
            raise ValueError(f"Unsupported PI report bout_scope {scope!r}.")
        if float(self.pre_context_sec) < 0 or float(self.post_context_sec) < 0:
            raise ValueError("PI report context padding seconds must be >= 0.")
        selected = tuple(
            str(b).strip() for b in self.selected_bout_ids if str(b).strip()
        )
        if scope == "selected_bouts" and not selected:
            raise ValueError(
                "Selected-bout PI report requires at least one eligible bout id."
            )
        overlays = bool(self.overlays_enabled)
        include_behavior = bool(self.overlay_include_behavior)
        include_pose = bool(self.overlay_include_pose)
        pose_dir: str | None = None
        if overlays:
            if kind != PI_SOURCE_REVIEWED:
                raise ValueError(
                    "PI report overlays require Published S10 review source "
                    "(refusing automated S8 label burn-in)."
                )
            if not include_behavior and not include_pose:
                raise ValueError(
                    "Overlays require reviewed behavioral labels and/or S4 pose."
                )
            if include_pose:
                if self.overlay_pose_run_dir is None or not str(
                    self.overlay_pose_run_dir
                ).strip():
                    raise ValueError(
                        "S4 pose overlay requires an explicit pose run selection."
                    )
                pose_dir = str(
                    Path(str(self.overlay_pose_run_dir).strip()).expanduser().resolve()
                )
            elif self.overlay_pose_run_dir is not None and str(
                self.overlay_pose_run_dir
            ).strip():
                raise ValueError(
                    "overlay_pose_run_dir is only valid when overlay_include_pose=True."
                )
            hud = str(self.overlay_hud_mode or "both").strip().lower() or "both"
            if hud not in {"both", "focal_0_only", "focal_1_only"}:
                raise ValueError(
                    "overlay_hud_mode must be both, focal_0_only, or focal_1_only."
                )
        else:
            hud = str(self.overlay_hud_mode or "both").strip().lower() or "both"
            if self.overlay_pose_run_dir is not None and str(
                self.overlay_pose_run_dir
            ).strip():
                # Ignore retained pose selection when overlays are off.
                pose_dir = None
        return replace(
            self,
            source_kind=kind,  # type: ignore[arg-type]
            bout_scope=scope,  # type: ignore[arg-type]
            selected_bout_ids=selected,
            pre_context_sec=float(self.pre_context_sec),
            post_context_sec=float(self.post_context_sec),
            write_zip=bool(self.write_zip),
            overlays_enabled=overlays,
            overlay_include_behavior=include_behavior,
            overlay_include_pose=include_pose,
            overlay_pose_run_dir=pose_dir,
            overlay_include_contact=bool(self.overlay_include_contact),
            overlay_include_sbs=bool(self.overlay_include_sbs),
            overlay_draw_skeleton=bool(self.overlay_draw_skeleton),
            overlay_hud_mode=hud,
        )

    def to_gallery_query(self) -> GalleryQuery | None:
        settings = self.validated()
        if settings.bout_scope != "selected_bouts":
            return None
        return GalleryQuery(bout_ids=frozenset(settings.selected_bout_ids))

    def to_pi_settings(self) -> PiReportSettings:
        settings = self.validated()
        return PiReportSettings(
            pre_context_sec=settings.pre_context_sec,
            post_context_sec=settings.post_context_sec,
            write_zip=settings.write_zip,
            overlays_enabled=settings.overlays_enabled,
            overlay_include_behavior=settings.overlay_include_behavior,
            overlay_include_pose=settings.overlay_include_pose,
            overlay_pose_run_dir=settings.overlay_pose_run_dir,
            overlay_include_contact=settings.overlay_include_contact,
            overlay_include_sbs=settings.overlay_include_sbs,
            overlay_draw_skeleton=settings.overlay_draw_skeleton,
            overlay_hud_mode=settings.overlay_hud_mode,
        )

    def to_reviewed_overlay_media_settings(self) -> Any:
        """Map GUI overlay toggles onto the reviewed media sibling exporter."""

        from overlay.contracts import PoseSourceKind
        from overlay.export_s10_media import ReviewedOverlayMediaExportSettings
        from results_packaging.media_csv_export.config import SelectionSpec
        from results_packaging.media_csv_export.contracts import (
            SELECTION_ALL_PACKAGE_BOUTS,
            SELECTION_BOUT_ID_LIST,
        )

        settings = self.validated()
        if not settings.overlays_enabled:
            raise ValueError(
                "to_reviewed_overlay_media_settings requires overlays_enabled=True."
            )
        if settings.bout_scope == "selected_bouts":
            selection = SelectionSpec(
                mode=SELECTION_BOUT_ID_LIST,
                bout_ids=settings.selected_bout_ids,
            )
        else:
            selection = SelectionSpec(mode=SELECTION_ALL_PACKAGE_BOUTS)
        return ReviewedOverlayMediaExportSettings(
            selection=selection,
            pre_context_sec=settings.pre_context_sec,
            post_context_sec=settings.post_context_sec,
            include_behavior=settings.overlay_include_behavior,
            include_pose=settings.overlay_include_pose,
            pose_source_kind=(
                PoseSourceKind.S4_FINALIZED.value
                if settings.overlay_include_pose
                else None
            ),
            pose_run_dir=settings.overlay_pose_run_dir,
            hud_mode=settings.overlay_hud_mode,
            include_contact=settings.overlay_include_contact,
            include_sbs=settings.overlay_include_sbs,
            draw_skeleton=settings.overlay_draw_skeleton,
        ).validated()

    def report_scope_label(self) -> str:
        settings = self.validated()
        if settings.bout_scope == "selected_bouts":
            return REPORT_SCOPE_SELECTED
        return REPORT_SCOPE_COMPLETE

    def scope_summary(self) -> str:
        settings = self.validated()
        if settings.bout_scope == "selected_bouts":
            return (
                f"Selected exclusive bouts only ({len(settings.selected_bout_ids)} "
                "included; not a complete exclusive export)."
            )
        return "All exclusive bouts (complete exclusive report scope)."


@dataclass(frozen=True, slots=True)
class PiReportRequestIdentity:
    """Identity used to reject stale asynchronous PI report results."""

    request_id: str
    session_root: Path
    source_kind: str
    package_run_id: str
    package_dir: Path
    review_run_id: str | None
    review_run_dir: Path | None
    report_id: str
    output_dir: Path
    settings: PiReportUiSettings
    pinned_source_digests: dict[str, str | None]

    @staticmethod
    def new(
        *,
        session_root: Path,
        source_kind: str,
        package_run_id: str,
        package_dir: Path,
        review_run_id: str | None,
        review_run_dir: Path | None,
        report_id: str,
        output_dir: Path,
        settings: PiReportUiSettings,
        pinned_source_digests: dict[str, str | None],
    ) -> PiReportRequestIdentity:
        return PiReportRequestIdentity(
            request_id=str(uuid4()),
            session_root=session_root.resolve(strict=False),
            source_kind=str(source_kind),
            package_run_id=str(package_run_id),
            package_dir=package_dir.resolve(strict=False),
            review_run_id=None if review_run_id is None else str(review_run_id),
            review_run_dir=(
                review_run_dir.resolve(strict=False)
                if review_run_dir is not None
                else None
            ),
            report_id=str(report_id),
            output_dir=output_dir.resolve(strict=False),
            settings=settings.validated(),
            pinned_source_digests=dict(pinned_source_digests),
        )


def pi_report_table_rows(reports: tuple[Any, ...]) -> tuple[tuple[str, ...], ...]:
    rows: list[tuple[str, ...]] = []
    for report in reports:
        rows.append(
            (
                str(getattr(report, "report_id", "—")),
                str(getattr(report, "status", None) or "—"),
                str(getattr(report, "source_type", None) or "—"),
                str(getattr(report, "report_scope", None) or "—"),
                (
                    str(report.n_bouts)
                    if getattr(report, "n_bouts", None) is not None
                    else "—"
                ),
                "yes" if getattr(report, "has_zip", False) else "no",
                str(getattr(report, "package_run_id", None) or "—"),
                str(getattr(report, "review_run_id", None) or "—"),
                str(getattr(report, "report_dir", "—")),
            )
        )
    return tuple(rows)


def source_identity_summary(
    *,
    source_kind: str,
    package_run_id: str,
    review_run_id: str | None,
    settings: PiReportUiSettings,
) -> str:
    validated = settings.validated()
    parts = [
        f"source={source_kind}",
        f"package={package_run_id}",
    ]
    if source_kind == PI_SOURCE_REVIEWED:
        parts.append(f"review={review_run_id or '—'}")
    parts.append(f"scope={validated.report_scope_label()}")
    if validated.bout_scope == "selected_bouts":
        parts.append(f"n_selected={len(validated.selected_bout_ids)}")
    parts.append(f"pad={validated.pre_context_sec:g}/{validated.post_context_sec:g}s")
    parts.append(f"zip={'yes' if validated.write_zip else 'no'}")
    if validated.overlays_enabled:
        layers: list[str] = []
        if validated.overlay_include_behavior:
            layers.append("s10_reviewed")
        if validated.overlay_include_pose:
            layers.append("s4_finalized")
        parts.append(f"overlays={'+'.join(layers) if layers else 'on'}")
        if validated.overlay_pose_run_dir:
            parts.append(f"pose={Path(validated.overlay_pose_run_dir).name}")
    else:
        parts.append("overlays=off")
    return "; ".join(parts)


__all__ = [
    "PiReportBoutScope",
    "PiReportRequestIdentity",
    "PiReportSourceKind",
    "PiReportUiSettings",
    "pi_report_table_rows",
    "source_identity_summary",
]
