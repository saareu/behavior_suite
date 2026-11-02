"""Resolve and pin overlay QC sources (eligible S8/S9 automated or published S10).

Working drafts are never accepted. Scientific trees are read-only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from behavioral_review.contracts import (
    GALLERY_SOURCE_AUTOMATED,
    GALLERY_SOURCE_PUBLISHED,
    GALLERY_SOURCE_WORKING,
    REVIEWED_BEHAVIOR_STATES_FILENAME,
    SOURCE_TYPE_AUTOMATED,
    SOURCE_TYPE_REVIEWED,
)
from behavioral_review.discovery import (
    classify_analysis_source,
    reopen_reviewed_run,
    validate_s10_run_for_downstream,
)
from behavioral_review.pi_report.source import assert_not_draft_source
from behavioral_review.source import ReviewSourcePackage, open_review_source
from behavioral_review.sync import SyncMap
from overlay.contracts import BehaviorSourceKind, OverlaySourceError
from project.file_digest import sha256_file
from results_packaging.media_csv_export.video import VideoProbe, probe_video


@dataclass(frozen=True, slots=True)
class OverlayQcSource:
    """Pinned exclusive behavioral overlay source (never mixes automated + reviewed)."""

    source_kind: BehaviorSourceKind
    package_run_id: str
    package_dir: Path
    review_run_id: str | None
    review_run_dir: Path | None
    session_root: Path | None
    package: ReviewSourcePackage
    sync: SyncMap
    states_path: Path
    states_sha256: str
    video_path: Path
    video_probe: VideoProbe
    video_sha256: str | None
    source_digests: dict[str, str | None]
    behavior_bouts: pd.DataFrame
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def gallery_source_kind(self) -> str:
        if self.source_kind is BehaviorSourceKind.S8_AUTOMATED:
            return GALLERY_SOURCE_AUTOMATED
        return GALLERY_SOURCE_PUBLISHED


def resolve_overlay_qc_source(
    *,
    source_kind: BehaviorSourceKind | str,
    package_dir: Path | str | None = None,
    review_run_dir: Path | str | None = None,
) -> OverlayQcSource:
    """Open an explicitly selected eligible automated or published reviewed source.

    Working drafts are rejected. Digests and identities are pinned at open time.
    """

    kind = _normalize_source_kind(source_kind)
    if kind is BehaviorSourceKind.S8_AUTOMATED:
        if package_dir is None:
            raise OverlaySourceError(
                "Automated overlay QC requires an explicit package_dir."
            )
        if review_run_dir is not None:
            raise OverlaySourceError(
                "Automated overlay QC must not receive a review_run_dir "
                "(refusing silent merge of automated and reviewed sources)."
            )
        return _resolve_automated(Path(package_dir))

    if review_run_dir is None:
        raise OverlaySourceError(
            "Reviewed overlay QC requires an explicit published review_run_dir."
        )
    return _resolve_reviewed(
        review_run_dir=Path(review_run_dir),
        package_dir=Path(package_dir) if package_dir is not None else None,
    )


def verify_overlay_qc_source_integrity(source: OverlayQcSource) -> None:
    """Recheck digests and eligibility against the pinned source before publication."""

    if source.source_kind is BehaviorSourceKind.S10_REVIEWED:
        if source.review_run_dir is None:
            raise OverlaySourceError(
                "Reviewed overlay QC source is missing review_run_dir for integrity check."
            )
        assert_not_draft_source(source.review_run_dir)
        validate_s10_run_for_downstream(source.review_run_dir)

    live_package = open_review_source(
        source.package_dir, require_video=True, require_s8_states=True
    )
    live_digests: dict[str, str | None] = {
        "package_manifest_sha256": live_package.package_manifest_sha256,
        "sync_index_sha256": live_package.sync_index_sha256,
        "behavior_bouts_sha256": live_package.behavior_bouts_sha256,
        "prepared_video_sha256": live_package.source_video.sha256,
    }
    if live_package.source_s8.states_path is not None and live_package.source_s8.states_path.is_file():
        live_digests["s8_behavior_states_sha256"] = sha256_file(
            live_package.source_s8.states_path
        )

    for key, expected in source.source_digests.items():
        if expected is None:
            continue
        if key not in live_digests:
            continue
        live = live_digests.get(key)
        if live is None or str(live) != str(expected):
            raise OverlaySourceError(
                f"Overlay QC source digest mismatch for {key}: "
                f"pinned={expected!r} live={live!r}."
            )

    if (
        source.video_sha256 is not None
        and live_package.source_video.sha256 is not None
        and str(live_package.source_video.sha256) != str(source.video_sha256)
    ):
        raise OverlaySourceError(
            "Overlay QC prepared-video digest changed during export."
        )

    if not source.states_path.is_file():
        raise OverlaySourceError(
            f"Pinned behavioral states path missing during integrity check: "
            f"{source.states_path}"
        )
    live_states_sha = sha256_file(source.states_path)
    if live_states_sha != source.states_sha256:
        raise OverlaySourceError(
            "Overlay QC behavioral states digest changed during export."
        )

    if source.source_kind is BehaviorSourceKind.S10_REVIEWED and source.review_run_dir is not None:
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
                raise OverlaySourceError(
                    f"Overlay QC reviewed-source digest mismatch for {key}."
                )


def _normalize_source_kind(value: BehaviorSourceKind | str) -> BehaviorSourceKind:
    if isinstance(value, BehaviorSourceKind):
        return value
    text = str(value).strip().lower()
    if text in {
        BehaviorSourceKind.S8_AUTOMATED.value,
        SOURCE_TYPE_AUTOMATED,
        "automated",
        "s8",
        "s9",
    }:
        return BehaviorSourceKind.S8_AUTOMATED
    if text in {
        BehaviorSourceKind.S10_REVIEWED.value,
        SOURCE_TYPE_REVIEWED,
        "reviewed",
        "published",
        "s10",
    }:
        return BehaviorSourceKind.S10_REVIEWED
    if text in {"draft", "working", GALLERY_SOURCE_WORKING}:
        raise OverlaySourceError(
            "Working drafts cannot be used as overlay QC sources; "
            "publish the review first or export from an eligible automated S9 package."
        )
    # Reuse S10 classifier language for operator-facing aliases.
    try:
        classified = classify_analysis_source(text)
    except Exception as exc:  # noqa: BLE001 - normalize unknown aliases
        raise OverlaySourceError(
            f"Unsupported overlay QC source kind: {value!r}."
        ) from exc
    if classified == SOURCE_TYPE_AUTOMATED:
        return BehaviorSourceKind.S8_AUTOMATED
    if classified == SOURCE_TYPE_REVIEWED:
        return BehaviorSourceKind.S10_REVIEWED
    raise OverlaySourceError(f"Unsupported overlay QC source kind: {value!r}.")


def _resolve_automated(package_dir: Path) -> OverlayQcSource:
    package = open_review_source(
        package_dir, require_video=True, require_s8_states=True
    )
    states_path = _require_s8_states(package)
    states_sha = sha256_file(states_path)
    video_path, probe = _require_video(package)
    sync = SyncMap.from_sync_index(package.sync_index)
    digests = {
        "package_manifest_sha256": package.package_manifest_sha256,
        "sync_index_sha256": package.sync_index_sha256,
        "behavior_bouts_sha256": package.behavior_bouts_sha256,
        "prepared_video_sha256": package.source_video.sha256,
        "s8_behavior_states_sha256": states_sha,
        "s8_behavior_states_sha256_recorded": package.source_s8.states_sha256_recorded,
    }
    provenance = {
        "source_kind": BehaviorSourceKind.S8_AUTOMATED.value,
        "gallery_source_kind": GALLERY_SOURCE_AUTOMATED,
        "package_run_id": package.package_run_id,
        "package_dir_name": package.package_dir.name,
        "s8_run_id": package.source_s8.run_id,
        "review_run_id": None,
        "session_root_name": (
            package.session_root.name if package.session_root is not None else None
        ),
    }
    return OverlayQcSource(
        source_kind=BehaviorSourceKind.S8_AUTOMATED,
        package_run_id=package.package_run_id,
        package_dir=package.package_dir,
        review_run_id=None,
        review_run_dir=None,
        session_root=package.session_root,
        package=package,
        sync=sync,
        states_path=states_path,
        states_sha256=states_sha,
        video_path=video_path,
        video_probe=probe,
        video_sha256=package.source_video.sha256,
        source_digests={k: (None if v is None else str(v)) for k, v in digests.items()},
        behavior_bouts=package.behavior_bouts,
        provenance=provenance,
    )


def _resolve_reviewed(
    *,
    review_run_dir: Path,
    package_dir: Path | None,
) -> OverlayQcSource:
    assert_not_draft_source(review_run_dir)
    validated = validate_s10_run_for_downstream(review_run_dir)
    reopened = reopen_reviewed_run(review_run_dir)
    parent_dir = Path(validated["parent_package_dir"]).resolve()
    if package_dir is not None and Path(package_dir).resolve() != parent_dir:
        raise OverlaySourceError(
            "Reviewed overlay QC package_dir does not match the published parent "
            f"package ({Path(package_dir).resolve()} vs {parent_dir})."
        )
    package = open_review_source(parent_dir, require_video=True, require_s8_states=False)
    states_path = Path(review_run_dir).resolve() / REVIEWED_BEHAVIOR_STATES_FILENAME
    if not states_path.is_file():
        raise OverlaySourceError(
            f"Published review is missing reviewed_behavior_states.parquet: {states_path}"
        )
    states_sha = sha256_file(states_path)
    video_path, probe = _require_video(package)
    sync = SyncMap.from_sync_index(package.sync_index)
    reviewed_bouts: pd.DataFrame = reopened["reviewed_behavior_bouts"]
    manifest = reopened["review_manifest"]
    digests: dict[str, str | None] = {
        "package_manifest_sha256": package.package_manifest_sha256,
        "sync_index_sha256": package.sync_index_sha256,
        "behavior_bouts_sha256": package.behavior_bouts_sha256,
        "prepared_video_sha256": package.source_video.sha256,
        "reviewed_behavior_states_sha256": states_sha,
        "parent_package_manifest_sha256": manifest.get(
            "parent_package_manifest_sha256"
        ),
        "parent_sync_index_sha256": manifest.get("parent_sync_index_sha256"),
        "parent_behavior_bouts_sha256": manifest.get("parent_behavior_bouts_sha256"),
        "publication_cursor_event_seq": (
            None
            if manifest.get("publication_cursor_event_seq") is None
            else str(manifest.get("publication_cursor_event_seq"))
        ),
    }
    artifact_digests = manifest.get("artifact_digests")
    if isinstance(artifact_digests, dict):
        for key in (
            "reviewed_behavior_bouts",
            "review_bout_status",
            "reviewed_behavior_states",
        ):
            if key in artifact_digests:
                digests[f"review_{key}_sha256"] = (
                    None
                    if artifact_digests.get(key) is None
                    else str(artifact_digests.get(key))
                )
    provenance = {
        "source_kind": BehaviorSourceKind.S10_REVIEWED.value,
        "gallery_source_kind": GALLERY_SOURCE_PUBLISHED,
        "package_run_id": package.package_run_id,
        "package_dir_name": package.package_dir.name,
        "review_run_id": str(validated["review_run_id"]),
        "review_run_dir_name": Path(review_run_dir).resolve().name,
        "session_root_name": (
            package.session_root.name if package.session_root is not None else None
        ),
        "publication_cursor_event_seq": manifest.get("publication_cursor_event_seq"),
    }
    return OverlayQcSource(
        source_kind=BehaviorSourceKind.S10_REVIEWED,
        package_run_id=package.package_run_id,
        package_dir=package.package_dir,
        review_run_id=str(validated["review_run_id"]),
        review_run_dir=Path(review_run_dir).resolve(),
        session_root=package.session_root,
        package=package,
        sync=sync,
        states_path=states_path,
        states_sha256=states_sha,
        video_path=video_path,
        video_probe=probe,
        video_sha256=package.source_video.sha256,
        source_digests=digests,
        behavior_bouts=reviewed_bouts,
        provenance=provenance,
    )


def _require_s8_states(package: ReviewSourcePackage) -> Path:
    if not package.s8_states_available or package.source_s8.states_path is None:
        raise OverlaySourceError(
            "Eligible automated overlay QC requires resolvable S8 behavior_states."
        )
    path = Path(package.source_s8.states_path).expanduser().resolve()
    if not path.is_file():
        raise OverlaySourceError(f"S8 behavior_states.parquet missing: {path}")
    return path


def _require_video(package: ReviewSourcePackage) -> tuple[Path, VideoProbe]:
    if not package.source_video.available or package.source_video.path is None:
        raise OverlaySourceError(
            "Overlay QC requires an available prepared video pinned by the source package."
        )
    path = Path(package.source_video.path).expanduser().resolve()
    if not path.is_file():
        raise OverlaySourceError(f"Prepared video missing for overlay QC: {path}")
    try:
        probe = probe_video(path)
    except Exception as exc:  # noqa: BLE001 - normalize to source error
        raise OverlaySourceError(f"Could not probe prepared video: {exc}") from exc
    return path, probe
