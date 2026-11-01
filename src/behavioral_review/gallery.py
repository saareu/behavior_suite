"""Behavior-grouped bout gallery index (S10 Gallery A1/A2).

Builds view-only cards over automated S9, working-draft, or published S10 bout
tables. Never merges distinct source kinds into one row set, never writes
journals or science artifacts during index construction, and never requires
eager GIF/MP4 generation. Gallery A2 review actions use EditingSession APIs.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from behavioral_review.contracts import (
    BOUT_ID_KIND_PACKAGE,
    BOUT_ID_KIND_REVIEWED,
    BOUT_ID_KIND_SOURCE_S8,
    BOUT_STATUS_VALUES,
    CHANNEL_EXCLUSIVE,
    GALLERY_CARD_SCHEMA_VERSION,
    GALLERY_INDEX_SCHEMA_VERSION,
    GALLERY_SOURCE_AUTOMATED,
    GALLERY_SOURCE_PUBLISHED,
    GALLERY_SOURCE_WORKING,
    SOURCE_TYPE_AUTOMATED,
    SOURCE_TYPE_REVIEWED,
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    STATUS_UNREVIEWED,
    STATUS_VOIDED,
    InspectionError,
)
from behavioral_review.regenerate import interval_key_from_row
from behavioral_review.sync import SyncMap
from results_packaging.media_csv_export.clip_bounds import (
    ClipBounds,
    compute_clip_bounds,
)

# Re-export status vocabulary for callers / tests.
_STATUS_ORDER = (
    STATUS_UNREVIEWED,
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    STATUS_VOIDED,
)

DEFAULT_PREVIEW_PAD_SEC = 0.5
DEFAULT_GALLERY_SORT = "time"


@dataclass(frozen=True, slots=True)
class GalleryPreviewPlan:
    """Prepared-domain decode window for one gallery card (not science endpoints)."""

    prepared_frame_idx_start: int
    prepared_frame_idx_end: int
    clip_frame_idx_start: int
    clip_frame_idx_end: int
    clip_frame_count: int
    pre_context_sec: float
    post_context_sec: float
    source_fps: float
    source_n_frames: int
    first_visible_prepared_frame_idx: int
    last_visible_prepared_frame_idx: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "prepared_frame_idx_start": self.prepared_frame_idx_start,
            "prepared_frame_idx_end": self.prepared_frame_idx_end,
            "clip_frame_idx_start": self.clip_frame_idx_start,
            "clip_frame_idx_end": self.clip_frame_idx_end,
            "clip_frame_count": self.clip_frame_count,
            "pre_context_sec": self.pre_context_sec,
            "post_context_sec": self.post_context_sec,
            "source_fps": self.source_fps,
            "source_n_frames": self.source_n_frames,
            "first_visible_prepared_frame_idx": self.first_visible_prepared_frame_idx,
            "last_visible_prepared_frame_idx": self.last_visible_prepared_frame_idx,
        }


@dataclass(frozen=True, slots=True)
class GalleryCard:
    """One exclusive-bout preview card (view over science tables)."""

    schema_version: str
    card_id: str
    bout_id: str
    source_kind: str
    channel: str
    identity_grain: str
    automated_label: str | None
    effective_label: str
    corrected: bool
    focal_id: int | None
    partner_id: int | None
    start_sample_idx: int
    end_sample_idx: int
    bout_duration_frames: int
    bout_duration_sec: float
    prepared_frame_idx_start: int
    prepared_frame_idx_end: int
    status: str
    void_reason: str | None
    source_s8_bout_id: str | None
    package_bout_id: str | None
    reviewed_bout_id: str | None
    preview_plan: GalleryPreviewPlan

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "card_id": self.card_id,
            "bout_id": self.bout_id,
            "source_kind": self.source_kind,
            "channel": self.channel,
            "identity_grain": self.identity_grain,
            "automated_label": self.automated_label,
            "effective_label": self.effective_label,
            "corrected": self.corrected,
            "focal_id": self.focal_id,
            "partner_id": self.partner_id,
            "start_sample_idx": self.start_sample_idx,
            "end_sample_idx": self.end_sample_idx,
            "bout_duration_frames": self.bout_duration_frames,
            "bout_duration_sec": self.bout_duration_sec,
            "prepared_frame_idx_start": self.prepared_frame_idx_start,
            "prepared_frame_idx_end": self.prepared_frame_idx_end,
            "status": self.status,
            "void_reason": self.void_reason,
            "source_s8_bout_id": self.source_s8_bout_id,
            "package_bout_id": self.package_bout_id,
            "reviewed_bout_id": self.reviewed_bout_id,
            "preview_plan": self.preview_plan.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class GalleryGroup:
    """Exclusive-label group with deterministic card order."""

    label: str
    cards: tuple[GalleryCard, ...]
    n_cards: int
    status_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "n_cards": self.n_cards,
            "status_counts": dict(self.status_counts),
            "card_ids": [c.card_id for c in self.cards],
        }


@dataclass(frozen=True, slots=True)
class GalleryQuery:
    """Display-only gallery filters (never mutate science tables)."""

    focal_ids: frozenset[int] | None = None
    labels: frozenset[str] | None = None
    statuses: frozenset[str] | None = None
    sort: str = DEFAULT_GALLERY_SORT

    def to_dict(self) -> dict[str, Any]:
        return {
            "focal_ids": sorted(self.focal_ids) if self.focal_ids is not None else None,
            "labels": sorted(self.labels) if self.labels is not None else None,
            "statuses": sorted(self.statuses) if self.statuses is not None else None,
            "sort": self.sort,
        }


@dataclass(frozen=True, slots=True)
class GallerySourceRef:
    """Pinned gallery source identity (automated / working_draft / published)."""

    source_kind: str
    package_run_id: str
    review_run_id: str | None = None
    package_dir: str | None = None
    review_run_dir: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_kind": self.source_kind,
            "package_run_id": self.package_run_id,
            "review_run_id": self.review_run_id,
            "package_dir": self.package_dir,
            "review_run_dir": self.review_run_dir,
        }


@dataclass(frozen=True, slots=True)
class GalleryIndex:
    """In-memory exclusive-bout gallery view for one pinned source."""

    schema_version: str
    source: GallerySourceRef
    cards: tuple[GalleryCard, ...]
    groups: tuple[GalleryGroup, ...]
    query: GalleryQuery
    n_cards: int
    n_groups: int
    available_labels: tuple[str, ...]
    available_focal_ids: tuple[int, ...]
    status_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source.to_dict(),
            "n_cards": self.n_cards,
            "n_groups": self.n_groups,
            "available_labels": list(self.available_labels),
            "available_focal_ids": list(self.available_focal_ids),
            "status_counts": dict(self.status_counts),
            "query": self.query.to_dict(),
            "groups": [g.to_dict() for g in self.groups],
        }

    def card_by_id(self, card_id: str) -> GalleryCard:
        for card in self.cards:
            if card.card_id == card_id:
                return card
        raise InspectionError(f"Unknown gallery card_id={card_id!r}.")


def infer_prepared_timing(
    sync: SyncMap,
    *,
    source_fps: float | None = None,
    source_n_frames: int | None = None,
) -> tuple[float, int]:
    """Derive prepared fps / frame count from sync (decode aids only)."""

    prepared = sync.prepared_frame_idx
    if prepared.size == 0:
        raise InspectionError("Cannot build gallery preview plans without sync rows.")
    n_frames = (
        int(source_n_frames)
        if source_n_frames is not None
        else int(prepared.max()) + 1
    )
    if n_frames <= 0:
        raise InspectionError(f"Invalid prepared frame count: {n_frames}.")
    if source_fps is not None and float(source_fps) > 0:
        return float(source_fps), n_frames
    # Prefer unique prepared times for fps inference.
    uniq = pd.Series(prepared).drop_duplicates().to_numpy()
    times = sync.prepared_time_sec
    if uniq.size >= 2:
        i0 = int(uniq[0])
        i1 = int(uniq[1])
        # Map prepared indices back to first occurrence times.
        t0 = float(times[prepared == i0][0])
        t1 = float(times[prepared == i1][0])
        dt = t1 - t0
        if dt > 0:
            return float(1.0 / dt), n_frames
    # Fallback: assume 1 prepared frame per dense sample / TARGET is wrong —
    # use prepared_time span over unique frames.
    if uniq.size >= 1 and float(times.max()) > float(times.min()):
        span = float(times.max()) - float(times.min())
        denom = max(1, int(uniq.size) - 1)
        return float(denom / span), n_frames
    return 30.0, n_frames


def build_preview_plan(
    *,
    prepared_frame_idx_start: int,
    prepared_frame_idx_end: int,
    source_fps: float,
    source_n_frames: int,
    pre_context_sec: float = DEFAULT_PREVIEW_PAD_SEC,
    post_context_sec: float = DEFAULT_PREVIEW_PAD_SEC,
) -> GalleryPreviewPlan:
    """Build a clamped prepared-domain preview window for looping decode."""

    bounds = compute_clip_bounds(
        prepared_frame_idx_start=int(prepared_frame_idx_start),
        prepared_frame_idx_end=int(prepared_frame_idx_end),
        source_fps=float(source_fps),
        source_n_frames=int(source_n_frames),
        pre_context_sec=float(pre_context_sec),
        post_context_sec=float(post_context_sec),
    )
    return _plan_from_bounds(bounds)


def build_gallery_from_automated_bouts(
    behavior_bouts: pd.DataFrame,
    sync: SyncMap,
    *,
    package_run_id: str,
    package_dir: str | None = None,
    query: GalleryQuery | None = None,
    source_fps: float | None = None,
    source_n_frames: int | None = None,
    pre_context_sec: float = DEFAULT_PREVIEW_PAD_SEC,
    post_context_sec: float = DEFAULT_PREVIEW_PAD_SEC,
    status_by_bout_id: Mapping[str, str] | None = None,
) -> GalleryIndex:
    """Build gallery cards from packaged S9 exclusive bouts (automated source).

    Status defaults to ``unreviewed``. Optional ``status_by_bout_id`` may attach
    an independent overlay without rewriting bout geometry or labels. Does not
    consume reviewed bout tables.
    """

    fps, n_frames = infer_prepared_timing(
        sync, source_fps=source_fps, source_n_frames=source_n_frames
    )
    exclusive = _exclusive_rows(behavior_bouts)
    cards: list[GalleryCard] = []
    for _, row in exclusive.iterrows():
        bout_id = str(row["bout_id"])
        label = str(row["behavior_name"])
        prep_start, prep_end = _prepared_endpoints(row, sync)
        status = STATUS_UNREVIEWED
        if status_by_bout_id is not None:
            status = str(
                status_by_bout_id.get(bout_id)
                or status_by_bout_id.get(str(row.get("source_s8_bout_id") or ""))
                or STATUS_UNREVIEWED
            )
            if status not in BOUT_STATUS_VALUES:
                status = STATUS_UNREVIEWED
        plan = build_preview_plan(
            prepared_frame_idx_start=prep_start,
            prepared_frame_idx_end=prep_end,
            source_fps=fps,
            source_n_frames=n_frames,
            pre_context_sec=pre_context_sec,
            post_context_sec=post_context_sec,
        )
        cards.append(
            GalleryCard(
                schema_version=GALLERY_CARD_SCHEMA_VERSION,
                card_id=_card_id(GALLERY_SOURCE_AUTOMATED, bout_id),
                bout_id=bout_id,
                source_kind=GALLERY_SOURCE_AUTOMATED,
                channel=CHANNEL_EXCLUSIVE,
                identity_grain=str(row.get("identity_grain") or "directional"),
                automated_label=label,
                effective_label=label,
                corrected=False,
                focal_id=_optional_int(row.get("focal_id")),
                partner_id=_optional_int(row.get("partner_id")),
                start_sample_idx=int(row["start_sample_idx"]),
                end_sample_idx=int(row["end_sample_idx"]),
                bout_duration_frames=int(row["bout_duration_frames"]),
                bout_duration_sec=float(row["bout_duration_sec"]),
                prepared_frame_idx_start=prep_start,
                prepared_frame_idx_end=prep_end,
                status=status,
                void_reason=None,
                source_s8_bout_id=_optional_str(row.get("source_s8_bout_id")),
                package_bout_id=bout_id,
                reviewed_bout_id=None,
                preview_plan=plan,
            )
        )
    source = GallerySourceRef(
        source_kind=GALLERY_SOURCE_AUTOMATED,
        package_run_id=str(package_run_id),
        review_run_id=None,
        package_dir=package_dir,
        review_run_dir=None,
    )
    return _finalize_index(cards, source=source, query=query or GalleryQuery())


def build_gallery_from_published_review(
    reviewed_bouts: pd.DataFrame,
    sync: SyncMap,
    *,
    package_run_id: str,
    review_run_id: str,
    review_bout_status: pd.DataFrame | None = None,
    automated_bouts: pd.DataFrame | None = None,
    package_dir: str | None = None,
    review_run_dir: str | None = None,
    query: GalleryQuery | None = None,
    source_fps: float | None = None,
    source_n_frames: int | None = None,
    pre_context_sec: float = DEFAULT_PREVIEW_PAD_SEC,
    post_context_sec: float = DEFAULT_PREVIEW_PAD_SEC,
) -> GalleryIndex:
    """Build gallery cards from published reviewed exclusive bouts.

    Does not merge automated bout rows into the card list. Automated labels are
    joined by lineage only for the corrected badge. Status comes from the
    published status table (default unreviewed when absent).
    """

    return _build_gallery_from_reviewed_table(
        reviewed_bouts,
        sync,
        source_kind=GALLERY_SOURCE_PUBLISHED,
        package_run_id=package_run_id,
        review_run_id=review_run_id,
        review_bout_status=review_bout_status,
        automated_bouts=automated_bouts,
        package_dir=package_dir,
        review_run_dir=review_run_dir,
        query=query,
        source_fps=source_fps,
        source_n_frames=source_n_frames,
        pre_context_sec=pre_context_sec,
        post_context_sec=post_context_sec,
    )


def build_gallery_from_working_draft(
    reviewed_bouts: pd.DataFrame,
    sync: SyncMap,
    *,
    package_run_id: str,
    review_run_id: str,
    review_bout_status: pd.DataFrame | None = None,
    automated_bouts: pd.DataFrame | None = None,
    package_dir: str | None = None,
    review_run_dir: str | None = None,
    query: GalleryQuery | None = None,
    source_fps: float | None = None,
    source_n_frames: int | None = None,
    pre_context_sec: float = DEFAULT_PREVIEW_PAD_SEC,
    post_context_sec: float = DEFAULT_PREVIEW_PAD_SEC,
) -> GalleryIndex:
    """Build gallery cards from the current working-draft exclusive projection.

    Uses regenerated reviewed bout geometry and the live status overlay. Does
    not fall back to stale automated bout intervals after corrections. Does not
    merge automated rows into the card list; automated labels join by lineage
    for the corrected badge only.
    """

    return _build_gallery_from_reviewed_table(
        reviewed_bouts,
        sync,
        source_kind=GALLERY_SOURCE_WORKING,
        package_run_id=package_run_id,
        review_run_id=review_run_id,
        review_bout_status=review_bout_status,
        automated_bouts=automated_bouts,
        package_dir=package_dir,
        review_run_dir=review_run_dir,
        query=query,
        source_fps=source_fps,
        source_n_frames=source_n_frames,
        pre_context_sec=pre_context_sec,
        post_context_sec=post_context_sec,
    )


def apply_gallery_query(
    cards: Sequence[GalleryCard],
    query: GalleryQuery,
) -> tuple[GalleryCard, ...]:
    """Filter + deterministically sort cards (display only)."""

    filtered: list[GalleryCard] = []
    for card in cards:
        if query.focal_ids is not None and card.focal_id not in query.focal_ids:
            continue
        if query.labels is not None and card.effective_label not in query.labels:
            continue
        if query.statuses is not None and card.status not in query.statuses:
            continue
        filtered.append(card)
    return tuple(_sort_cards(filtered, sort=query.sort))


def group_cards_by_effective_label(
    cards: Sequence[GalleryCard],
) -> tuple[GalleryGroup, ...]:
    """Group exclusive cards by effective label with deterministic label order."""

    buckets: dict[str, list[GalleryCard]] = {}
    for card in cards:
        buckets.setdefault(card.effective_label, []).append(card)
    groups: list[GalleryGroup] = []
    for label in sorted(buckets.keys()):
        members = tuple(_sort_cards(buckets[label], sort=DEFAULT_GALLERY_SORT))
        counts = {s: 0 for s in _STATUS_ORDER}
        for card in members:
            counts[card.status] = counts.get(card.status, 0) + 1
        groups.append(
            GalleryGroup(
                label=label,
                cards=members,
                n_cards=len(members),
                status_counts=counts,
            )
        )
    return tuple(groups)


def gallery_source_kind_from_analysis(source_type: str) -> str:
    """Map analysis source vocabulary onto gallery source kinds."""

    value = str(source_type or "").strip().lower()
    if value in {
        GALLERY_SOURCE_AUTOMATED,
        SOURCE_TYPE_AUTOMATED,
        "s9",
        "package",
        "results_package",
    }:
        return GALLERY_SOURCE_AUTOMATED
    if value in {
        GALLERY_SOURCE_WORKING,
        "working",
        "draft",
        "working_draft",
    }:
        return GALLERY_SOURCE_WORKING
    if value in {
        GALLERY_SOURCE_PUBLISHED,
        SOURCE_TYPE_REVIEWED,
        "s10",
        "reviewed",
        "published",
        "behavioral_review",
    }:
        return GALLERY_SOURCE_PUBLISHED
    raise InspectionError(
        "Unknown gallery source_type="
        f"{source_type!r}; expected automated, working_draft, or published."
    )


def deep_link_target_for_card(card: GalleryCard) -> dict[str, Any]:
    """Resolve timeline deep-link fields without applying corrections.

    Prefer packaged bout id when present (BoutBrowser identity). Always includes
    dense endpoints and directional focal/partner for playhead + direction.
    """

    seek_bout_id = card.package_bout_id or (
        card.bout_id if card.source_kind == GALLERY_SOURCE_AUTOMATED else None
    )
    return {
        "card_id": card.card_id,
        "source_kind": card.source_kind,
        "seek_bout_id": seek_bout_id,
        "reviewed_bout_id": card.reviewed_bout_id,
        "package_bout_id": card.package_bout_id,
        "source_s8_bout_id": card.source_s8_bout_id,
        "start_sample_idx": card.start_sample_idx,
        "end_sample_idx": card.end_sample_idx,
        "focal_id": card.focal_id,
        "partner_id": card.partner_id,
        "channel": card.channel,
        "effective_label": card.effective_label,
        "status": card.status,
    }


def gallery_card_is_review_eligible(card: GalleryCard) -> bool:
    """True when a card may receive default exclusive gallery Accept/Reject."""

    return (
        card.source_kind == GALLERY_SOURCE_WORKING
        and card.channel == CHANNEL_EXCLUSIVE
    )


def gallery_status_target_for_card(card: GalleryCard) -> tuple[str, str]:
    """Resolve ``(target_bout_id, target_bout_id_kind)`` for status-only writes.

    Prefer stable lineage ids so status survives review-id regeneration when the
    automated interval is unchanged. Fall back to reviewed bout id after
    geometry changes that break lineage.
    """

    if not gallery_card_is_review_eligible(card):
        raise InspectionError(
            "Gallery review actions apply only to working-draft exclusive bouts "
            f"(got source={card.source_kind!r}, channel={card.channel!r})."
        )
    if card.source_s8_bout_id:
        return str(card.source_s8_bout_id), BOUT_ID_KIND_SOURCE_S8
    if card.package_bout_id:
        return str(card.package_bout_id), BOUT_ID_KIND_PACKAGE
    reviewed = card.reviewed_bout_id or card.bout_id
    return str(reviewed), BOUT_ID_KIND_REVIEWED


def filter_eligible_gallery_review_cards(
    cards: Sequence[GalleryCard],
    *,
    selected_card_ids: Sequence[str] | None = None,
) -> tuple[GalleryCard, ...]:
    """Return explicit selected (or all provided) exclusive working-draft cards.

    Selection is never inferred from filters: callers must pass the explicit
    selected card ids for bulk operations. Contact/SBS cards are excluded.
    """

    by_id = {c.card_id: c for c in cards}
    if selected_card_ids is None:
        ordered = list(cards)
    else:
        ordered = []
        for card_id in selected_card_ids:
            card = by_id.get(str(card_id))
            if card is not None:
                ordered.append(card)
    return tuple(c for c in ordered if gallery_card_is_review_eligible(c))


def _build_gallery_from_reviewed_table(
    reviewed_bouts: pd.DataFrame,
    sync: SyncMap,
    *,
    source_kind: str,
    package_run_id: str,
    review_run_id: str,
    review_bout_status: pd.DataFrame | None = None,
    automated_bouts: pd.DataFrame | None = None,
    package_dir: str | None = None,
    review_run_dir: str | None = None,
    query: GalleryQuery | None = None,
    source_fps: float | None = None,
    source_n_frames: int | None = None,
    pre_context_sec: float = DEFAULT_PREVIEW_PAD_SEC,
    post_context_sec: float = DEFAULT_PREVIEW_PAD_SEC,
) -> GalleryIndex:
    if source_kind not in {GALLERY_SOURCE_PUBLISHED, GALLERY_SOURCE_WORKING}:
        raise InspectionError(
            f"Reviewed gallery builder does not support source_kind={source_kind!r}."
        )
    fps, n_frames = infer_prepared_timing(
        sync, source_fps=source_fps, source_n_frames=source_n_frames
    )
    exclusive = _exclusive_rows(reviewed_bouts)
    auto_by_package, auto_by_source = _automated_label_indexes(automated_bouts)
    status_map = _status_map_from_frame(review_bout_status)
    cards: list[GalleryCard] = []
    for _, row in exclusive.iterrows():
        bout_id = str(row["bout_id"])
        effective = str(row["behavior_name"])
        package_bout_id = _optional_str(row.get("source_package_bout_id"))
        source_s8 = _optional_str(row.get("source_s8_bout_id"))
        automated = None
        if package_bout_id and package_bout_id in auto_by_package:
            automated = auto_by_package[package_bout_id]
        elif source_s8 and source_s8 in auto_by_source:
            automated = auto_by_source[source_s8]
        status, void_reason = _lookup_reviewed_status(
            status_map,
            reviewed_bout_id=bout_id,
            package_bout_id=package_bout_id,
            source_s8_bout_id=source_s8,
            interval_key=_interval_key_lookup(row),
        )
        prep_start, prep_end = _prepared_endpoints(row, sync)
        plan = build_preview_plan(
            prepared_frame_idx_start=prep_start,
            prepared_frame_idx_end=prep_end,
            source_fps=fps,
            source_n_frames=n_frames,
            pre_context_sec=pre_context_sec,
            post_context_sec=post_context_sec,
        )
        cards.append(
            GalleryCard(
                schema_version=GALLERY_CARD_SCHEMA_VERSION,
                card_id=_card_id(source_kind, bout_id),
                bout_id=bout_id,
                source_kind=source_kind,
                channel=CHANNEL_EXCLUSIVE,
                identity_grain=str(row.get("identity_grain") or "directional"),
                automated_label=automated,
                effective_label=effective,
                corrected=bool(automated is not None and automated != effective),
                focal_id=_optional_int(row.get("focal_id")),
                partner_id=_optional_int(row.get("partner_id")),
                start_sample_idx=int(row["start_sample_idx"]),
                end_sample_idx=int(row["end_sample_idx"]),
                bout_duration_frames=int(row["bout_duration_frames"]),
                bout_duration_sec=float(row["bout_duration_sec"]),
                prepared_frame_idx_start=prep_start,
                prepared_frame_idx_end=prep_end,
                status=status,
                void_reason=void_reason,
                source_s8_bout_id=source_s8,
                package_bout_id=package_bout_id,
                reviewed_bout_id=bout_id,
                preview_plan=plan,
            )
        )
    source = GallerySourceRef(
        source_kind=source_kind,
        package_run_id=str(package_run_id),
        review_run_id=str(review_run_id),
        package_dir=package_dir,
        review_run_dir=review_run_dir,
    )
    return _finalize_index(cards, source=source, query=query or GalleryQuery())


def _finalize_index(
    cards: Sequence[GalleryCard],
    *,
    source: GallerySourceRef,
    query: GalleryQuery,
) -> GalleryIndex:
    filtered = apply_gallery_query(cards, query)
    groups = group_cards_by_effective_label(filtered)
    flat = tuple(card for group in groups for card in group.cards)
    labels = tuple(sorted({c.effective_label for c in cards}))
    focals = tuple(
        sorted({int(c.focal_id) for c in cards if c.focal_id is not None})
    )
    counts = {s: 0 for s in _STATUS_ORDER}
    for card in flat:
        counts[card.status] = counts.get(card.status, 0) + 1
    return GalleryIndex(
        schema_version=GALLERY_INDEX_SCHEMA_VERSION,
        source=source,
        cards=flat,
        groups=groups,
        query=query,
        n_cards=len(flat),
        n_groups=len(groups),
        available_labels=labels,
        available_focal_ids=focals,
        status_counts=counts,
    )


def _exclusive_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    if "channel" not in frame.columns:
        raise InspectionError("Bout table missing channel column.")
    exclusive = frame[frame["channel"].astype(str) == CHANNEL_EXCLUSIVE].copy()
    return exclusive.reset_index(drop=True)


def _prepared_endpoints(row: pd.Series, sync: SyncMap) -> tuple[int, int]:
    start = int(row["start_sample_idx"])
    end = int(row["end_sample_idx"])
    if "prepared_frame_idx_start" in row.index and pd.notna(
        row.get("prepared_frame_idx_start")
    ):
        prep_start = int(row["prepared_frame_idx_start"])
        prep_end = int(row["prepared_frame_idx_end"])
    else:
        prep_start = int(sync.mapping_at_sample(start).prepared_frame_idx)
        prep_end = int(sync.mapping_at_sample(end).prepared_frame_idx)
    # Integrity: sync mapping must agree when packaged endpoints present.
    mapped_start = int(sync.mapping_at_sample(start).prepared_frame_idx)
    mapped_end = int(sync.mapping_at_sample(end).prepared_frame_idx)
    if prep_start != mapped_start or prep_end != mapped_end:
        raise InspectionError(
            f"Prepared endpoints disagree with sync for bout "
            f"{row.get('bout_id')}: packaged=[{prep_start},{prep_end}] "
            f"sync=[{mapped_start},{mapped_end}]."
        )
    return prep_start, prep_end


def _plan_from_bounds(bounds: ClipBounds) -> GalleryPreviewPlan:
    return GalleryPreviewPlan(
        prepared_frame_idx_start=bounds.prepared_frame_idx_start,
        prepared_frame_idx_end=bounds.prepared_frame_idx_end,
        clip_frame_idx_start=bounds.clip_frame_idx_start,
        clip_frame_idx_end=bounds.clip_frame_idx_end,
        clip_frame_count=bounds.clip_frame_count,
        pre_context_sec=bounds.pre_context_sec,
        post_context_sec=bounds.post_context_sec,
        source_fps=bounds.source_fps,
        source_n_frames=bounds.source_n_frames,
        first_visible_prepared_frame_idx=bounds.clip_frame_idx_start,
        last_visible_prepared_frame_idx=bounds.clip_frame_idx_end,
    )


def _sort_cards(cards: Sequence[GalleryCard], *, sort: str) -> list[GalleryCard]:
    key = str(sort or DEFAULT_GALLERY_SORT).strip().lower()
    if key == "duration":
        return sorted(
            cards,
            key=lambda c: (
                c.bout_duration_frames,
                c.start_sample_idx,
                c.end_sample_idx,
                c.bout_id,
            ),
        )
    if key == "status":
        order = {s: i for i, s in enumerate(_STATUS_ORDER)}
        return sorted(
            cards,
            key=lambda c: (
                order.get(c.status, 99),
                c.start_sample_idx,
                c.end_sample_idx,
                c.bout_id,
            ),
        )
    if key == "label":
        return sorted(
            cards,
            key=lambda c: (
                c.effective_label,
                c.start_sample_idx,
                c.end_sample_idx,
                c.bout_id,
            ),
        )
    # Default: time then stable bout id.
    return sorted(
        cards,
        key=lambda c: (c.start_sample_idx, c.end_sample_idx, c.bout_id),
    )


def _card_id(source_kind: str, bout_id: str) -> str:
    return f"{source_kind}:{bout_id}"


def _optional_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if pd.isna(value):
        return None
    return int(value)


def _optional_str(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _automated_label_indexes(
    automated_bouts: pd.DataFrame | None,
) -> tuple[dict[str, str], dict[str, str]]:
    by_package: dict[str, str] = {}
    by_source: dict[str, str] = {}
    if automated_bouts is None or automated_bouts.empty:
        return by_package, by_source
    exclusive = _exclusive_rows(automated_bouts)
    for _, row in exclusive.iterrows():
        by_package[str(row["bout_id"])] = str(row["behavior_name"])
        src = _optional_str(row.get("source_s8_bout_id"))
        if src is not None:
            by_source[src] = str(row["behavior_name"])
    return by_package, by_source


def _status_map_from_frame(
    frame: pd.DataFrame | None,
) -> dict[str, tuple[str, str | None]]:
    out: dict[str, tuple[str, str | None]] = {}
    if frame is None or frame.empty:
        return out
    current = frame
    if "active" in current.columns:
        current = current[current["active"].fillna(True).astype(bool)]
    for _, row in current.iterrows():
        status = str(row.get("status") or STATUS_UNREVIEWED)
        if status not in BOUT_STATUS_VALUES:
            status = STATUS_UNREVIEWED
        void_reason = _optional_str(row.get("void_reason"))
        for key_col in (
            "reviewed_bout_id",
            "package_bout_id",
            "source_s8_bout_id",
            "bout_id",
            "target_bout_id",
            "status_row_id",
        ):
            if key_col in row.index:
                key = _optional_str(row.get(key_col))
                if key is not None:
                    out[f"{key_col}:{key}"] = (status, void_reason)
                    out[key] = (status, void_reason)
        interval_key = row.get("interval_key") if "interval_key" in row.index else None
        encoded = _encode_interval_key(interval_key)
        if encoded is not None:
            out[encoded] = (status, void_reason)
    return out


def _lookup_reviewed_status(
    status_map: Mapping[str, tuple[str, str | None]],
    *,
    reviewed_bout_id: str,
    package_bout_id: str | None,
    source_s8_bout_id: str | None,
    interval_key: str | None = None,
) -> tuple[str, str | None]:
    for key in (
        f"reviewed_bout_id:{reviewed_bout_id}",
        reviewed_bout_id,
        f"package_bout_id:{package_bout_id}" if package_bout_id else None,
        package_bout_id,
        f"source_s8_bout_id:{source_s8_bout_id}" if source_s8_bout_id else None,
        source_s8_bout_id,
        interval_key,
    ):
        if key and key in status_map:
            return status_map[key]
    return STATUS_UNREVIEWED, None


def _encode_interval_key(value: Any) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, tuple):
        payload = list(value)
    elif isinstance(value, list):
        payload = value
    else:
        return None
    return "interval_key:" + json.dumps(payload, sort_keys=False)


def _interval_key_lookup(row: pd.Series) -> str | None:
    try:
        return _encode_interval_key(interval_key_from_row(row))
    except Exception:  # noqa: BLE001
        return None

