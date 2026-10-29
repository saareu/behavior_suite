"""Structural QC and nonblocking review summaries for S10 publication."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from behavioral_arbitration.bouts import extract_behavior_bouts
from behavioral_review.contracts import (
    BOUT_STATUS_VALUES,
    CHANNEL_CONTACT,
    CHANNEL_EXCLUSIVE,
    CHANNEL_SIDE_BY_SIDE,
    EXCLUSIVE_ETHOGRAM_LABELS,
    REVIEW_QC_SCHEMA_VERSION,
    REVIEW_TARGET_FPS,
    STATUS_ACCEPTED,
    STATUS_REJECTED,
    STATUS_VOIDED,
    PublicationError,
)
from behavioral_review.snapshot import PublicationSnapshot
from behavioral_review.source import ReviewSourcePackage


@dataclass
class ReviewQcSummary:
    """Structural blockers + nonblocking review warnings/summaries."""

    schema_version: str = REVIEW_QC_SCHEMA_VERSION
    passed: bool = False
    has_warnings: bool = False
    blockers: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    summaries: dict[str, Any] = field(default_factory=dict)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "passed": self.passed,
            "has_warnings": self.has_warnings,
            "blocker_count": len(self.blockers),
            "warning_count": len(self.warnings),
            "blockers": list(self.blockers),
            "warnings": list(self.warnings),
            "summaries": dict(self.summaries),
        }


def evaluate_review_qc(
    *,
    snapshot: PublicationSnapshot,
    source: ReviewSourcePackage,
    eligibility_warnings: list[str] | None = None,
    artifact_digests: dict[str, str] | None = None,
) -> ReviewQcSummary:
    """Run structural blockers and derive review summaries (no biological gates)."""

    blockers: list[str] = []
    warnings: list[str] = list(eligibility_warnings or [])

    states = snapshot.reviewed_states
    bouts = snapshot.reviewed_bouts
    statuses = snapshot.status_frame()

    blockers.extend(_check_states_structure(states, source))
    blockers.extend(_check_bouts_structure(bouts, states, source, snapshot.review_run_id))
    blockers.extend(_check_status_integrity(statuses, bouts, source.behavior_bouts))
    blockers.extend(_check_collapse_parity(states, bouts, source))

    summaries = build_review_summaries(
        snapshot=snapshot,
        source=source,
        statuses=statuses,
    )
    if artifact_digests:
        summaries["artifact_digests"] = dict(artifact_digests)

    # Nonblocking review warnings.
    if summaries.get("n_voided_decisions", 0) > 0:
        warnings.append(
            f"{summaries['n_voided_decisions']} review decision(s) voided after "
            "geometry-changing edits."
        )
    unreviewed = int(summaries.get("n_unreviewed_bouts", 0) or 0)
    total_bouts = int(len(bouts)) if bouts is not None else 0
    if total_bouts and unreviewed / total_bouts >= 0.9:
        warnings.append(
            f"High fraction of reviewed bouts remain unreviewed "
            f"({unreviewed}/{total_bouts}); acceptance is not mandatory."
        )
    if source.package_status == "completed_with_warnings":
        warnings.append("Parent S9 package was completed_with_warnings.")
    if not source.source_s7_run_id:
        warnings.append("Optional S7 evidence pointer unavailable on parent package.")

    passed = not blockers
    return ReviewQcSummary(
        passed=passed,
        has_warnings=bool(warnings),
        blockers=blockers,
        warnings=warnings,
        summaries=summaries,
    )


def assert_structural_qc(qc: ReviewQcSummary) -> None:
    if not qc.passed:
        raise PublicationError(
            "Reviewed publication structural QC failed: " + "; ".join(qc.blockers)
        )


def build_review_summaries(
    *,
    snapshot: PublicationSnapshot,
    source: ReviewSourcePackage,
    statuses: pd.DataFrame,
) -> dict[str, Any]:
    states = snapshot.reviewed_states
    bouts = snapshot.reviewed_bouts
    baseline = None
    # Prefer comparing against source S8 states path already loaded into snapshot
    # via apply; use packaged bouts for bout-level deltas.
    auto_bouts = source.behavior_bouts

    corrected_samples = 0
    by_channel: dict[str, int] = {
        CHANNEL_EXCLUSIVE: 0,
        CHANNEL_CONTACT: 0,
        CHANNEL_SIDE_BY_SIDE: 0,
    }
    for ch, spans in (snapshot.changed_intervals or {}).items():
        n = sum(int(e) - int(s) + 1 for s, e in spans)
        by_channel[str(ch)] = by_channel.get(str(ch), 0) + n
        corrected_samples += n

    auto_label_dist = _label_distribution_from_bouts(auto_bouts, channel=CHANNEL_EXCLUSIVE)
    rev_label_dist = _label_distribution_from_bouts(bouts, channel=CHANNEL_EXCLUSIVE)

    bout_delta = _bout_change_counts(auto_bouts, bouts)

    status_counts: Counter[str] = Counter()
    if statuses is not None and not statuses.empty:
        # Effective current decisions: active rows only when column present.
        current = statuses
        if "active" in current.columns:
            current = current[current["active"].fillna(False).astype(bool)]
        status_counts.update(str(s) for s in current["status"].tolist())

    n_reviewed_bouts = int(len(bouts)) if bouts is not None else 0
    decided = (
        int(status_counts.get(STATUS_ACCEPTED, 0))
        + int(status_counts.get(STATUS_REJECTED, 0))
    )
    n_unreviewed = max(0, n_reviewed_bouts - decided)

    reviewers = sorted(
        {
            str(e.reviewer_id)
            for e in snapshot.history_events
            if str(e.reviewer_id or "").strip()
        }
    )

    return {
        "n_dense_samples": int(source.n_dense_samples),
        "n_reviewed_state_rows": int(len(states)) if states is not None else 0,
        "n_reviewed_bouts": n_reviewed_bouts,
        "n_automated_bouts": int(len(auto_bouts)) if auto_bouts is not None else 0,
        "corrected_sample_count": int(corrected_samples),
        "corrected_intervals_by_channel": {
            k: [{"start": s, "end": e} for s, e in spans]
            for k, spans in (snapshot.changed_intervals or {}).items()
        },
        "corrections_by_channel_sample_counts": by_channel,
        "automated_exclusive_label_distribution": auto_label_dist,
        "reviewed_exclusive_label_distribution": rev_label_dist,
        "bouts_created": bout_delta["created"],
        "bouts_deleted": bout_delta["deleted"],
        "bouts_changed": bout_delta["changed"],
        "bouts_unchanged": bout_delta["unchanged"],
        "n_accepted_decisions": int(status_counts.get(STATUS_ACCEPTED, 0)),
        "n_rejected_decisions": int(status_counts.get(STATUS_REJECTED, 0)),
        "n_voided_decisions": int(status_counts.get(STATUS_VOIDED, 0)),
        "n_unreviewed_bouts": int(n_unreviewed),
        "n_events_applied": int(snapshot.n_events_applied),
        "n_events_history": int(snapshot.n_events_history),
        "n_superseded_events": int(snapshot.n_superseded),
        "n_undone_beyond_cursor": int(snapshot.n_undone_beyond_cursor),
        "publication_cursor_event_seq": int(snapshot.publication_cursor_event_seq),
        "reviewer_ids": reviewers,
        "remaining_unreviewed_content": {
            "n_unreviewed_bouts": int(n_unreviewed),
            "note": (
                "Unreviewed content is informational only; full bout acceptance "
                "is not required for publication."
            ),
        },
        # Keep baseline unused reference quiet for linters if not needed.
        "baseline_compared": baseline is not None,
    }


def _check_states_structure(
    states: pd.DataFrame, source: ReviewSourcePackage
) -> list[str]:
    blockers: list[str] = []
    if states is None or states.empty:
        blockers.append("Reviewed states table is empty.")
        return blockers
    required = {
        "dense_frame_idx",
        "focal_id",
        "partner_id",
        "behavior_label",
        "pred_contact",
        "pred_side_by_side",
        "review_run_id",
        "parent_package_run_id",
        "schema_version",
    }
    missing = sorted(required - set(states.columns))
    if missing:
        blockers.append(f"Reviewed states missing columns: {missing}.")
        return blockers

    n_dense = int(source.n_dense_samples)
    samples = states["dense_frame_idx"].astype("int64")
    if int(samples.min()) < 0 or int(samples.max()) >= n_dense:
        blockers.append("Reviewed states dense_frame_idx outside parent dense range.")

    # Directional cardinality / uniqueness.
    key = ["dense_frame_idx", "focal_id", "partner_id"]
    if states.duplicated(subset=key).any():
        blockers.append(
            "Reviewed states violate exclusive directional uniqueness "
            "(duplicate dense_frame_idx,focal_id,partner_id)."
        )
    # Expect two directional rows per dense sample for K=2 dyads when baseline did.
    counts = states.groupby("dense_frame_idx", sort=False).size()
    if not counts.empty and int(counts.nunique()) != 1:
        blockers.append(
            "Reviewed states directional cardinality is inconsistent across samples."
        )
    elif not counts.empty and int(counts.iloc[0]) < 1:
        blockers.append("Reviewed states have zero rows per dense sample.")

    labels = states["behavior_label"].astype(str)
    illegal = sorted(set(labels) - set(EXCLUSIVE_ETHOGRAM_LABELS))
    if illegal:
        blockers.append(f"Invalid exclusive label vocabulary values: {illegal}.")

    for col in ("pred_contact", "pred_side_by_side"):
        vals = set(int(v) for v in states[col].dropna().astype("int64").unique())
        if not vals.issubset({0, 1}):
            blockers.append(f"{col} must be binary {{0,1}}; got {sorted(vals)}.")

    return blockers


def _check_bouts_structure(
    bouts: pd.DataFrame,
    states: pd.DataFrame,
    source: ReviewSourcePackage,
    review_run_id: str,
) -> list[str]:
    blockers: list[str] = []
    if bouts is None:
        blockers.append("Reviewed bouts table is missing.")
        return blockers
    required = {
        "bout_id",
        "channel",
        "identity_grain",
        "behavior_name",
        "start_sample_idx",
        "end_sample_idx",
        "bout_duration_frames",
        "bout_duration_sec",
        "review_run_id",
        "parent_package_run_id",
        "schema_version",
    }
    missing = sorted(required - set(bouts.columns))
    if missing:
        blockers.append(f"Reviewed bouts missing columns: {missing}.")
        return blockers

    if bouts["bout_id"].duplicated().any():
        blockers.append("Reviewed bout identifiers are not unique.")
    if (bouts["review_run_id"].astype(str) != str(review_run_id)).any():
        blockers.append("Reviewed bouts carry mismatched review_run_id stamps.")

    for _, row in bouts.iterrows():
        start = int(row["start_sample_idx"])
        end = int(row["end_sample_idx"])
        frames = int(row["bout_duration_frames"])
        if end < start:
            blockers.append(
                f"Bout {row['bout_id']} has end_sample_idx < start_sample_idx."
            )
            continue
        expected = end - start + 1
        if frames != expected:
            blockers.append(
                f"Bout {row['bout_id']} inclusive frame count mismatch "
                f"({frames} != {expected})."
            )
        expected_sec = float(frames) / float(REVIEW_TARGET_FPS)
        if abs(float(row["bout_duration_sec"]) - expected_sec) > 1e-9:
            blockers.append(
                f"Bout {row['bout_id']} duration_sec is not frames/120."
            )
    # Cap detailed row scanning message volume.
    if len(blockers) > 20:
        blockers = blockers[:20] + [f"... and {len(blockers) - 20} more bout errors."]
    return blockers


def _check_status_integrity(
    statuses: pd.DataFrame,
    reviewed_bouts: pd.DataFrame,
    automated_bouts: pd.DataFrame,
) -> list[str]:
    blockers: list[str] = []
    if statuses is None or statuses.empty:
        return blockers
    illegal = sorted(set(statuses["status"].astype(str)) - set(BOUT_STATUS_VALUES))
    if illegal:
        blockers.append(f"Invalid review status values: {illegal}.")

    reviewed_ids = set()
    if reviewed_bouts is not None and not reviewed_bouts.empty:
        reviewed_ids = set(reviewed_bouts["bout_id"].astype(str))
    source_ids = set()
    if (
        automated_bouts is not None
        and not automated_bouts.empty
        and "source_s8_bout_id" in automated_bouts.columns
    ):
        source_ids = {
            str(x) for x in automated_bouts["source_s8_bout_id"].dropna().tolist()
        }

    for _, row in statuses.iterrows():
        rid = row.get("reviewed_bout_id")
        sid = row.get("source_s8_bout_id")
        # Historical / voided references to pre-edit reviewed ids may exist;
        # only block active non-voided decisions on missing reviewed ids.
        if (
            pd.notna(rid)
            and str(rid)
            and str(rid) not in reviewed_ids
            and str(row.get("status")) in (STATUS_ACCEPTED, STATUS_REJECTED)
            and bool(row.get("active", True))
        ):
            blockers.append(
                f"Active status references missing reviewed_bout_id {rid!r}."
            )
        if (
            pd.notna(sid)
            and str(sid)
            and source_ids
            and str(sid) not in source_ids
            and str(row.get("status")) in (STATUS_ACCEPTED, STATUS_REJECTED)
            and bool(row.get("active", True))
        ):
            blockers.append(
                f"Active status references unknown source_s8_bout_id {sid!r}."
            )
    return blockers


def _check_collapse_parity(
    states: pd.DataFrame,
    bouts: pd.DataFrame,
    source: ReviewSourcePackage,
) -> list[str]:
    """Re-collapse reviewed states and compare interval keys to published bouts."""

    blockers: list[str] = []
    if states is None or states.empty or bouts is None:
        return blockers
    work = states.copy()
    if "exp_name" not in work.columns:
        work["exp_name"] = source.package_run_id
    rebuilt = extract_behavior_bouts(
        work,
        fps=float(REVIEW_TARGET_FPS),
        source_s7_run_id=str(source.source_s7_run_id or ""),
        source_s6_run_id=str(source.source_s6_run_id or ""),
        run_id=str(source.source_s8.run_id or snapshot_safe_run_id(source)),
    )
    pub_keys = _bout_interval_multiset(bouts)
    reb_keys = _bout_interval_multiset(rebuilt)
    if pub_keys != reb_keys:
        blockers.append(
            "Reviewed bouts are inconsistent with pure S8 collapse of reviewed states "
            f"(published={len(pub_keys)} rebuilt={len(reb_keys)} key multiset mismatch)."
        )
    return blockers


def snapshot_safe_run_id(source: ReviewSourcePackage) -> str:
    return str(source.source_s8.run_id or source.package_run_id)


def _bout_interval_multiset(bouts: pd.DataFrame) -> Counter[tuple[Any, ...]]:
    keys: list[tuple[Any, ...]] = []
    if bouts is None or bouts.empty:
        return Counter()
    for _, row in bouts.iterrows():
        keys.append(
            (
                str(row["channel"]),
                str(row["identity_grain"]),
                _opt_int(row.get("focal_id")),
                _opt_int(row.get("partner_id")),
                _opt_int(row.get("mouse_a_id")),
                _opt_int(row.get("mouse_b_id")),
                str(row["behavior_name"]),
                int(row["start_sample_idx"]),
                int(row["end_sample_idx"]),
                int(row["bout_duration_frames"]),
            )
        )
    return Counter(keys)


def _label_distribution_from_bouts(
    bouts: pd.DataFrame | None, *, channel: str
) -> dict[str, int]:
    if bouts is None or bouts.empty:
        return {}
    subset = bouts[bouts["channel"].astype(str) == channel]
    return {
        str(k): int(v)
        for k, v in subset["behavior_name"].astype(str).value_counts().to_dict().items()
    }


def _bout_change_counts(
    automated: pd.DataFrame | None, reviewed: pd.DataFrame | None
) -> dict[str, int]:
    auto_set = set(_bout_interval_multiset(automated))
    rev_set = set(_bout_interval_multiset(reviewed))
    return {
        "created": len(rev_set - auto_set),
        "deleted": len(auto_set - rev_set),
        "unchanged": len(auto_set & rev_set),
        "changed": len(auto_set ^ rev_set),
    }


def _opt_int(value: Any) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
