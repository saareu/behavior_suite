"""Resolve and validate PI report sources (automated S9 or published S10)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from behavioral_review.contracts import (
    CLASS_DRAFT,
    GALLERY_SOURCE_AUTOMATED,
    GALLERY_SOURCE_PUBLISHED,
    GALLERY_SOURCE_WORKING,
    PUBLICATION_STATE_DRAFT,
    SOURCE_TYPE_AUTOMATED,
)
from behavioral_review.discovery import (
    classify_analysis_source,
    reopen_reviewed_run,
    validate_s10_run_for_downstream,
)
from behavioral_review.gallery import (
    GalleryIndex,
    GalleryQuery,
    build_gallery_from_automated_bouts,
    build_gallery_from_published_review,
    gallery_source_kind_from_analysis,
)
from behavioral_review.pi_report.contracts import (
    DEFAULT_POST_CONTEXT_SEC,
    DEFAULT_PRE_CONTEXT_SEC,
    PI_SOURCE_AUTOMATED,
    PI_SOURCE_REVIEWED,
    PiReportSourceError,
)
from behavioral_review.source import ReviewSourcePackage, open_review_source
from behavioral_review.sync import SyncMap
from results_packaging.export import load_json_object
from results_packaging.media_csv_export.video import VideoProbe, probe_video


@dataclass(frozen=True, slots=True)
class PiReportSource:
    """Pinned exclusive-bout report source (never mixes automated + reviewed)."""

    source_type: str
    gallery_source_kind: str
    package_run_id: str
    package_dir: Path
    review_run_id: str | None
    review_run_dir: Path | None
    session_root: Path | None
    package: ReviewSourcePackage
    sync: SyncMap
    gallery: GalleryIndex
    video_path: Path
    video_probe: VideoProbe
    video_sha256: str | None
    source_digests: dict[str, str | None]
    provenance: dict[str, Any]


def resolve_pi_report_source(
    *,
    source_type: str,
    package_dir: Path | str | None = None,
    review_run_dir: Path | str | None = None,
    query: GalleryQuery | None = None,
    pre_context_sec: float = DEFAULT_PRE_CONTEXT_SEC,
    post_context_sec: float = DEFAULT_POST_CONTEXT_SEC,
    require_video: bool = True,
) -> PiReportSource:
    """Open an explicit automated package or published review for PI export.

    Working drafts are never accepted as final-report sources.
    """

    normalized = classify_analysis_source(source_type)
    gallery_kind = gallery_source_kind_from_analysis(normalized)
    if gallery_kind == GALLERY_SOURCE_WORKING:
        raise PiReportSourceError(
            "Working drafts cannot be used as PI report sources; "
            "publish the review first or export from an automated S9 package."
        )

    if normalized == SOURCE_TYPE_AUTOMATED:
        if package_dir is None:
            raise PiReportSourceError(
                "Automated PI report requires an explicit package_dir."
            )
        if review_run_dir is not None:
            raise PiReportSourceError(
                "Automated PI report must not receive a review_run_dir "
                "(refusing silent merge of automated and reviewed sources)."
            )
        return _resolve_automated(
            package_dir=Path(package_dir),
            query=query,
            pre_context_sec=pre_context_sec,
            post_context_sec=post_context_sec,
            require_video=require_video,
        )

    if review_run_dir is None:
        raise PiReportSourceError(
            "Reviewed PI report requires an explicit published review_run_dir."
        )
    return _resolve_published(
        review_run_dir=Path(review_run_dir),
        package_dir=Path(package_dir) if package_dir is not None else None,
        query=query,
        pre_context_sec=pre_context_sec,
        post_context_sec=post_context_sec,
        require_video=require_video,
    )


def verify_pinned_source_integrity(source: PiReportSource) -> None:
    """Recheck digests/eligibility against the pinned source before finalize.

    Rejects changed package/review inputs. Does not mutate any science trees.
    """

    if source.source_type == PI_SOURCE_REVIEWED:
        if source.review_run_dir is None:
            raise PiReportSourceError(
                "Reviewed PI report source is missing review_run_dir for integrity check."
            )
        assert_not_draft_source(source.review_run_dir)
        validate_s10_run_for_downstream(source.review_run_dir)

    live_package = open_review_source(source.package_dir, require_video=True)
    live_digests: dict[str, str | None] = {
        "package_manifest_sha256": live_package.package_manifest_sha256,
        "sync_index_sha256": live_package.sync_index_sha256,
        "behavior_bouts_sha256": live_package.behavior_bouts_sha256,
        "prepared_video_sha256": live_package.source_video.sha256,
    }
    for key, expected in source.source_digests.items():
        if expected is None:
            continue
        if key not in live_digests:
            # Review-only digest keys are checked via re-open below.
            continue
        live = live_digests.get(key)
        if live is None or str(live) != str(expected):
            raise PiReportSourceError(
                f"PI report source digest mismatch for {key}: "
                f"pinned={expected!r} live={live!r}."
            )

    if (
        source.video_sha256 is not None
        and live_package.source_video.sha256 is not None
        and str(live_package.source_video.sha256) != str(source.video_sha256)
    ):
        raise PiReportSourceError(
            "PI report prepared-video digest changed during export."
        )

    if source.source_type == PI_SOURCE_REVIEWED and source.review_run_dir is not None:
        reopened = reopen_reviewed_run(source.review_run_dir)
        manifest = reopened["review_manifest"]
        for key, manifest_key in (
            ("parent_package_manifest_sha256", "parent_package_manifest_sha256"),
            ("parent_sync_index_sha256", "parent_sync_index_sha256"),
            ("parent_behavior_bouts_sha256", "parent_behavior_bouts_sha256"),
        ):
            expected = source.source_digests.get(key)
            if expected is None:
                continue
            live = manifest.get(manifest_key)
            if live is None or str(live) != str(expected):
                raise PiReportSourceError(
                    f"PI report reviewed-source digest mismatch for {key}."
                )
        expected_cursor = source.source_digests.get("publication_cursor_event_seq")
        if expected_cursor is not None:
            live_cursor = manifest.get("publication_cursor_event_seq")
            if str(live_cursor) != str(expected_cursor):
                raise PiReportSourceError(
                    "PI report reviewed publication cursor changed during export."
                )


def assert_not_draft_source(path: Path | str) -> None:
    """Fail closed when a path looks like an unfinished S10 working draft."""

    run_dir = Path(path).expanduser().resolve()
    working_manifest = run_dir / "working" / "working_manifest.json"
    meta_path = run_dir / "run_meta.json"
    publication_state = None
    status = None
    if working_manifest.is_file():
        try:
            working = load_json_object(working_manifest)
            publication_state = working.get("publication_state")
        except Exception:  # noqa: BLE001 - treat unreadable draft metadata as draft
            raise PiReportSourceError(
                f"Refusing PI report source with unreadable working draft metadata: "
                f"{working_manifest}"
            ) from None
    if meta_path.is_file():
        try:
            meta = load_json_object(meta_path)
            status = meta.get("status")
            publication_state = publication_state or meta.get("publication_state")
        except Exception:  # noqa: BLE001
            pass
    if str(publication_state or "").strip().lower() == PUBLICATION_STATE_DRAFT:
        raise PiReportSourceError(
            f"Refusing draft PI report source (publication_state=draft): {run_dir}"
        )
    if str(status or "").strip().lower() in {"running", ""} and working_manifest.is_file():
        # Draft-like incomplete review without science seal.
        science = run_dir / "reviewed_behavior_bouts.parquet"
        if not science.is_file():
            raise PiReportSourceError(
                f"Refusing unfinished working-draft review as PI report source: "
                f"{run_dir} (classification={CLASS_DRAFT})."
            )


def _resolve_automated(
    *,
    package_dir: Path,
    query: GalleryQuery | None,
    pre_context_sec: float,
    post_context_sec: float,
    require_video: bool,
) -> PiReportSource:
    package = open_review_source(package_dir, require_video=require_video)
    video_path, probe = _require_video(package)
    sync = SyncMap.from_sync_index(package.sync_index)
    gallery = build_gallery_from_automated_bouts(
        package.behavior_bouts,
        sync,
        package_run_id=package.package_run_id,
        package_dir=str(package.package_dir),
        query=query,
        source_fps=probe.fps,
        source_n_frames=probe.n_frames,
        pre_context_sec=pre_context_sec,
        post_context_sec=post_context_sec,
    )
    digests = {
        "package_manifest_sha256": package.package_manifest_sha256,
        "sync_index_sha256": package.sync_index_sha256,
        "behavior_bouts_sha256": package.behavior_bouts_sha256,
        "prepared_video_sha256": package.source_video.sha256,
    }
    provenance = {
        "source_type": PI_SOURCE_AUTOMATED,
        "gallery_source_kind": GALLERY_SOURCE_AUTOMATED,
        "package_run_id": package.package_run_id,
        "package_dir_name": package.package_dir.name,
        "review_run_id": None,
        "session_root_name": (
            package.session_root.name if package.session_root is not None else None
        ),
        "n_exclusive_bouts": gallery.n_cards,
        "n_package_bouts": package.n_bouts,
        "default_channel_view": "exclusive",
        "contact_sbs_in_default_view": False,
    }
    return PiReportSource(
        source_type=PI_SOURCE_AUTOMATED,
        gallery_source_kind=GALLERY_SOURCE_AUTOMATED,
        package_run_id=package.package_run_id,
        package_dir=package.package_dir,
        review_run_id=None,
        review_run_dir=None,
        session_root=package.session_root,
        package=package,
        sync=sync,
        gallery=gallery,
        video_path=video_path,
        video_probe=probe,
        video_sha256=package.source_video.sha256,
        source_digests=digests,
        provenance=provenance,
    )


def _resolve_published(
    *,
    review_run_dir: Path,
    package_dir: Path | None,
    query: GalleryQuery | None,
    pre_context_sec: float,
    post_context_sec: float,
    require_video: bool,
) -> PiReportSource:
    assert_not_draft_source(review_run_dir)
    validated = validate_s10_run_for_downstream(review_run_dir)
    reopened = reopen_reviewed_run(review_run_dir)
    parent_dir = Path(validated["parent_package_dir"]).resolve()
    if package_dir is not None and Path(package_dir).resolve() != parent_dir:
        raise PiReportSourceError(
            "Reviewed PI report package_dir does not match the published parent "
            f"package ({Path(package_dir).resolve()} vs {parent_dir})."
        )
    package = open_review_source(parent_dir, require_video=require_video)
    video_path, probe = _require_video(package)
    sync = SyncMap.from_sync_index(package.sync_index)
    reviewed_bouts: pd.DataFrame = reopened["reviewed_behavior_bouts"]
    status: pd.DataFrame = reopened["review_bout_status"]
    gallery = build_gallery_from_published_review(
        reviewed_bouts,
        sync,
        package_run_id=package.package_run_id,
        review_run_id=str(validated["review_run_id"]),
        review_bout_status=status,
        automated_bouts=package.behavior_bouts,
        package_dir=str(package.package_dir),
        review_run_dir=str(Path(review_run_dir).resolve()),
        query=query,
        source_fps=probe.fps,
        source_n_frames=probe.n_frames,
        pre_context_sec=pre_context_sec,
        post_context_sec=post_context_sec,
    )
    manifest = reopened["review_manifest"]
    digests = {
        "package_manifest_sha256": package.package_manifest_sha256,
        "sync_index_sha256": package.sync_index_sha256,
        "behavior_bouts_sha256": package.behavior_bouts_sha256,
        "prepared_video_sha256": package.source_video.sha256,
        "parent_package_manifest_sha256": manifest.get(
            "parent_package_manifest_sha256"
        ),
        "parent_sync_index_sha256": manifest.get("parent_sync_index_sha256"),
        "parent_behavior_bouts_sha256": manifest.get("parent_behavior_bouts_sha256"),
        "publication_cursor_event_seq": manifest.get("publication_cursor_event_seq"),
    }
    # Include published science table digests when recorded.
    artifact_digests = manifest.get("artifact_digests")
    if isinstance(artifact_digests, dict):
        for key in (
            "reviewed_behavior_bouts",
            "review_bout_status",
            "reviewed_behavior_states",
        ):
            if key in artifact_digests:
                digests[f"review_{key}_sha256"] = artifact_digests.get(key)

    provenance = {
        "source_type": PI_SOURCE_REVIEWED,
        "gallery_source_kind": GALLERY_SOURCE_PUBLISHED,
        "package_run_id": package.package_run_id,
        "package_dir_name": package.package_dir.name,
        "review_run_id": str(validated["review_run_id"]),
        "review_run_dir_name": Path(review_run_dir).resolve().name,
        "session_root_name": (
            package.session_root.name if package.session_root is not None else None
        ),
        "publication_cursor_event_seq": manifest.get("publication_cursor_event_seq"),
        "n_exclusive_bouts": gallery.n_cards,
        "n_reviewed_bouts": int(len(reviewed_bouts)),
        "default_channel_view": "exclusive",
        "contact_sbs_in_default_view": False,
        "status_vocabulary": ["unreviewed", "accepted", "rejected", "voided"],
    }
    return PiReportSource(
        source_type=PI_SOURCE_REVIEWED,
        gallery_source_kind=GALLERY_SOURCE_PUBLISHED,
        package_run_id=package.package_run_id,
        package_dir=package.package_dir,
        review_run_id=str(validated["review_run_id"]),
        review_run_dir=Path(review_run_dir).resolve(),
        session_root=package.session_root,
        package=package,
        sync=sync,
        gallery=gallery,
        video_path=video_path,
        video_probe=probe,
        video_sha256=package.source_video.sha256,
        source_digests={k: (None if v is None else str(v)) for k, v in digests.items()},
        provenance=provenance,
    )


def _require_video(package: ReviewSourcePackage) -> tuple[Path, VideoProbe]:
    if not package.source_video.available or package.source_video.path is None:
        raise PiReportSourceError(
            "PI report requires an available prepared video pinned by the "
            "source package."
        )
    path = Path(package.source_video.path).expanduser().resolve()
    if not path.is_file():
        raise PiReportSourceError(f"Prepared video missing for PI report: {path}")
    try:
        probe = probe_video(path)
    except Exception as exc:  # noqa: BLE001 - normalize to source error
        raise PiReportSourceError(f"Could not probe prepared video: {exc}") from exc
    return path, probe


__all__ = [
    "PiReportSource",
    "assert_not_draft_source",
    "resolve_pi_report_source",
    "verify_pinned_source_integrity",
]
