"""Optional export-profile views for Subsystem 09 (selection only).

``review_emit_v1`` produces a filtered bout index for human review. It never
mutates the complete packaged ``behavior_bouts`` table and never rewrites bout
boundaries. Seconds→frame conversion uses explicit 120 Hz (deliberate
difference from legacy Pipeline 110 ``source_video.fps``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from results_packaging.contracts import (
    BOUT_INDEX_REQUIRED_COLUMNS,
    EXPORT_PROFILE_MANIFEST_SCHEMA_VERSION,
    PACKAGE_BOUTS_SORT_COLUMNS,
    REVIEW_EMIT_EXCLUSION_REASON_PRIORITY,
    REVIEW_EMIT_PROFILE_ID,
    REVIEW_EMIT_V1_INCLUDE_CONTACT,
    REVIEW_EMIT_V1_INCLUDE_ENGAGED,
    REVIEW_EMIT_V1_INCLUDE_SIDE_BY_SIDE,
    REVIEW_EMIT_V1_MIN_EMIT_BOUT_FRAMES,
    REVIEW_EMIT_V1_MIN_EMIT_BOUT_SECONDS,
    REVIEW_EMIT_V1_TARGET_FPS,
    TARGET_FPS,
    ExportProfileError,
)


@dataclass(frozen=True, slots=True)
class ReviewEmitV1Params:
    """Explicit parameters for the optional ``review_emit_v1`` profile."""

    min_emit_bout_frames: int = REVIEW_EMIT_V1_MIN_EMIT_BOUT_FRAMES
    min_emit_bout_seconds: float = REVIEW_EMIT_V1_MIN_EMIT_BOUT_SECONDS
    target_fps: float = REVIEW_EMIT_V1_TARGET_FPS
    include_contact: bool = REVIEW_EMIT_V1_INCLUDE_CONTACT
    include_engaged: bool = REVIEW_EMIT_V1_INCLUDE_ENGAGED
    include_side_by_side: bool = REVIEW_EMIT_V1_INCLUDE_SIDE_BY_SIDE
    channels_allowlist: tuple[str, ...] | None = None
    behaviors_allowlist: tuple[str, ...] | None = None

    def validated(self) -> ReviewEmitV1Params:
        if int(self.min_emit_bout_frames) < 1:
            raise ExportProfileError(
                "min_emit_bout_frames must be >= 1 "
                f"(got {self.min_emit_bout_frames!r})."
            )
        if float(self.min_emit_bout_seconds) < 0.0:
            raise ExportProfileError(
                "min_emit_bout_seconds must be >= 0 "
                f"(got {self.min_emit_bout_seconds!r})."
            )
        if abs(float(self.target_fps) - TARGET_FPS) > 1e-9:
            raise ExportProfileError(
                "review_emit_v1 target_fps must be the 120 Hz behavioral grid "
                f"({TARGET_FPS}); got {self.target_fps!r}. "
                "Source-video FPS must not be used to count dense S8 samples."
            )
        return self

    def min_required_frames(self) -> int:
        """Combine frame and seconds floors via ``max`` (legacy-inspired).

        ``min_required = max(min_emit_bout_frames, ceil(min_emit_bout_seconds * 120))``
        with explicit frame and second units. Uses target_fps=120 only.
        """

        params = self.validated()
        from_seconds = int(
            math.ceil(float(params.min_emit_bout_seconds) * float(params.target_fps))
        )
        return int(max(int(params.min_emit_bout_frames), from_seconds))

    def to_dict(self) -> dict[str, Any]:
        params = self.validated()
        return {
            "profile_id": REVIEW_EMIT_PROFILE_ID,
            "min_emit_bout_frames": int(params.min_emit_bout_frames),
            "min_emit_bout_seconds": float(params.min_emit_bout_seconds),
            "target_fps": float(params.target_fps),
            "include_contact": bool(params.include_contact),
            "include_engaged": bool(params.include_engaged),
            "include_side_by_side": bool(params.include_side_by_side),
            "channels_allowlist": (
                list(params.channels_allowlist)
                if params.channels_allowlist is not None
                else None
            ),
            "behaviors_allowlist": (
                list(params.behaviors_allowlist)
                if params.behaviors_allowlist is not None
                else None
            ),
            "min_required_frames": self.min_required_frames(),
            "duration_selection_rule": (
                "min_required_frames = max(min_emit_bout_frames, "
                "ceil(min_emit_bout_seconds * target_fps)); "
                "keep iff bout_duration_frames >= min_required_frames "
                "and channel/behavior policy passes"
            ),
            "deliberate_difference_from_pipeline_110_source_fps": True,
            "not_numerically_identical_to_pipeline_110": True,
        }


@dataclass(frozen=True, slots=True)
class ExportProfileResult:
    """In-memory optional export-profile view (not written in Sprint 2)."""

    profile_id: str
    bout_index: pd.DataFrame
    exclusion_table: pd.DataFrame
    parameters: dict[str, Any]
    n_included: int
    n_excluded: int
    exclusion_reason_counts: dict[str, int]
    warnings: tuple[str, ...] = ()
    schema_version: str = EXPORT_PROFILE_MANIFEST_SCHEMA_VERSION
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_manifest_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "parameters": dict(self.parameters),
            "n_included": int(self.n_included),
            "n_excluded": int(self.n_excluded),
            "exclusion_reason_counts": dict(self.exclusion_reason_counts),
            "included_source_s8_bout_ids": (
                self.bout_index["source_s8_bout_id"].astype(str).tolist()
                if not self.bout_index.empty
                else []
            ),
            "excluded_source_s8_bout_ids": (
                self.exclusion_table["source_s8_bout_id"].astype(str).tolist()
                if not self.exclusion_table.empty
                else []
            ),
            "exclusion_reason_semantics": (
                "Primary excluded_reason is the first matching rule in "
                f"{list(REVIEW_EMIT_EXCLUSION_REASON_PRIORITY)}. "
                "All matching reasons are also listed in matched_reasons "
                "(pipe-separated) for audit. Absent bout_index rows are excluded."
            ),
            "warnings": list(self.warnings),
            "diagnostics": dict(self.diagnostics),
            "published": False,
        }


def review_emit_v1_params_from_mapping(
    data: Mapping[str, Any] | None,
) -> ReviewEmitV1Params:
    """Build validated review_emit_v1 params; reject source-video FPS keys."""

    raw = dict(data or {})
    forbidden = {
        "source_video_fps",
        "source_fps",
        "video_fps",
        "fps_source_video",
    }
    present = sorted(forbidden.intersection(raw))
    if present:
        raise ExportProfileError(
            "review_emit_v1 rejects source-video FPS parameters for dense-sample "
            f"filtering: {present}. Use target_fps=120.0 only."
        )
    channels = raw.get("channels_allowlist", raw.get("channels"))
    behaviors = raw.get("behaviors_allowlist", raw.get("behaviors"))
    return ReviewEmitV1Params(
        min_emit_bout_frames=int(
            raw.get("min_emit_bout_frames", REVIEW_EMIT_V1_MIN_EMIT_BOUT_FRAMES)
        ),
        min_emit_bout_seconds=float(
            raw.get("min_emit_bout_seconds", REVIEW_EMIT_V1_MIN_EMIT_BOUT_SECONDS)
        ),
        target_fps=float(raw.get("target_fps", REVIEW_EMIT_V1_TARGET_FPS)),
        include_contact=bool(raw.get("include_contact", REVIEW_EMIT_V1_INCLUDE_CONTACT)),
        include_engaged=bool(raw.get("include_engaged", REVIEW_EMIT_V1_INCLUDE_ENGAGED)),
        include_side_by_side=bool(
            raw.get("include_side_by_side", REVIEW_EMIT_V1_INCLUDE_SIDE_BY_SIDE)
        ),
        channels_allowlist=_optional_str_tuple(channels),
        behaviors_allowlist=_optional_str_tuple(behaviors),
    ).validated()


def apply_review_emit_v1(
    packaged_bouts: pd.DataFrame,
    *,
    params: ReviewEmitV1Params | Mapping[str, Any] | None = None,
) -> ExportProfileResult:
    """Select a review bout index from the complete packaged bout table.

    Selection only: no boundary rewrite, no resegmentation, no re-collapse.
    """

    if packaged_bouts is None:
        raise ExportProfileError("packaged_bouts is required for review_emit_v1.")
    resolved = (
        params
        if isinstance(params, ReviewEmitV1Params)
        else review_emit_v1_params_from_mapping(params)
    ).validated()
    param_dict = resolved.to_dict()
    min_required = int(resolved.min_required_frames())

    if packaged_bouts.empty:
        empty_index = _empty_bout_index()
        empty_excl = _empty_exclusion_table()
        return ExportProfileResult(
            profile_id=REVIEW_EMIT_PROFILE_ID,
            bout_index=empty_index,
            exclusion_table=empty_excl,
            parameters=param_dict,
            n_included=0,
            n_excluded=0,
            exclusion_reason_counts={},
            diagnostics={"min_required_frames": min_required},
        )

    required = {
        "bout_id",
        "source_s8_bout_id",
        "channel",
        "behavior_name",
        "bout_duration_frames",
        "start_sample_idx",
        "end_sample_idx",
    }
    missing = sorted(required - set(packaged_bouts.columns))
    if missing:
        raise ExportProfileError(f"packaged_bouts missing columns: {missing}")

    # Work in deterministic package order (do not depend on caller row order).
    work = packaged_bouts.copy()
    sort_cols = [c for c in PACKAGE_BOUTS_SORT_COLUMNS if c in work.columns]
    work = work.sort_values(by=sort_cols, kind="mergesort", na_position="last")
    work = work.reset_index(drop=True)

    included_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {}

    for rec in work.to_dict(orient="records"):
        matched = _exclusion_reasons(rec, resolved, min_required=min_required)
        if matched:
            primary = matched[0]
            reason_counts[primary] = int(reason_counts.get(primary, 0)) + 1
            excluded_rows.append(
                {
                    "bout_id": str(rec["bout_id"]),
                    "source_s8_bout_id": str(rec["source_s8_bout_id"]),
                    "channel": str(rec["channel"]),
                    "behavior_name": str(rec["behavior_name"]),
                    "bout_duration_frames": int(rec["bout_duration_frames"]),
                    "excluded_reason": primary,
                    "matched_reasons": "|".join(matched),
                    "min_required_frames": min_required,
                }
            )
            continue
        included_rows.append(_bout_index_row(rec, min_required=min_required))

    bout_index = (
        pd.DataFrame(included_rows)
        if included_rows
        else _empty_bout_index()
    )
    if not bout_index.empty:
        bout_index["bout_index_id"] = [
            f"{REVIEW_EMIT_PROFILE_ID}__{i:08d}" for i in range(len(bout_index))
        ]
        bout_index = bout_index.loc[:, list(BOUT_INDEX_REQUIRED_COLUMNS)]

    exclusion_table = (
        pd.DataFrame(excluded_rows) if excluded_rows else _empty_exclusion_table()
    )

    # Invariant: selection did not alter boundaries on included rows.
    if not bout_index.empty:
        src = work.set_index("bout_id")
        for _, row in bout_index.iterrows():
            src_row = src.loc[str(row["bout_id"])]
            if int(row["start_sample_idx"]) != int(src_row["start_sample_idx"]):
                raise ExportProfileError("review_emit_v1 altered start_sample_idx.")
            if int(row["end_sample_idx"]) != int(src_row["end_sample_idx"]):
                raise ExportProfileError("review_emit_v1 altered end_sample_idx.")
            if int(row["bout_duration_frames"]) != int(src_row["bout_duration_frames"]):
                raise ExportProfileError(
                    "review_emit_v1 altered bout_duration_frames."
                )

    warnings: list[str] = []
    if int(reason_counts.get("below_emit_floor", 0)):
        warnings.append(
            f"review_emit_v1 excluded {reason_counts['below_emit_floor']} bouts "
            f"below emit floor (min_required_frames={min_required})."
        )
    if int(reason_counts.get("channel_contact_excluded", 0)):
        warnings.append(
            f"review_emit_v1 excluded {reason_counts['channel_contact_excluded']} "
            "contact bouts (include_contact=false)."
        )

    return ExportProfileResult(
        profile_id=REVIEW_EMIT_PROFILE_ID,
        bout_index=bout_index.reset_index(drop=True),
        exclusion_table=exclusion_table.reset_index(drop=True),
        parameters=param_dict,
        n_included=int(len(bout_index)),
        n_excluded=int(len(exclusion_table)),
        exclusion_reason_counts={k: int(v) for k, v in sorted(reason_counts.items())},
        warnings=tuple(warnings),
        diagnostics={
            "min_required_frames": min_required,
            "n_source_bouts": int(len(work)),
            "selection_only": True,
            "boundaries_unchanged": True,
            "no_recollapse": True,
        },
    )


def _exclusion_reasons(
    rec: Mapping[str, Any],
    params: ReviewEmitV1Params,
    *,
    min_required: int,
) -> list[str]:
    """Return all matching exclusion reasons in deterministic priority order."""

    channel = str(rec["channel"])
    behavior = str(rec["behavior_name"])
    duration = int(rec["bout_duration_frames"])
    matched: list[str] = []

    candidates: dict[str, bool] = {
        "channel_contact_excluded": channel == "contact" and not params.include_contact,
        "channel_side_by_side_excluded": (
            channel == "side_by_side" and not params.include_side_by_side
        ),
        "behavior_engaged_excluded": (
            channel == "exclusive"
            and behavior == "engaged"
            and not params.include_engaged
        ),
        "behavior_not_allowed": (
            params.behaviors_allowlist is not None
            and behavior not in params.behaviors_allowlist
        ),
        "channel_not_allowed": (
            params.channels_allowlist is not None
            and channel not in params.channels_allowlist
        ),
        "below_emit_floor": duration < int(min_required),
    }
    for reason in REVIEW_EMIT_EXCLUSION_REASON_PRIORITY:
        if candidates.get(reason):
            matched.append(reason)
    return matched


def _bout_index_row(rec: Mapping[str, Any], *, min_required: int) -> dict[str, Any]:
    def _opt_int(key: str) -> int | None:
        val = rec.get(key)
        if val is None or (isinstance(val, float) and pd.isna(val)) or pd.isna(val):
            return None
        return int(val)

    def _opt_float(key: str) -> float | None:
        val = rec.get(key)
        if val is None or pd.isna(val):
            return None
        return float(val)

    return {
        "bout_index_id": "",  # filled after inclusion list is complete
        "bout_id": str(rec["bout_id"]),
        "source_s8_bout_id": str(rec["source_s8_bout_id"]),
        "filter_profile_id": REVIEW_EMIT_PROFILE_ID,
        "behavior_name": str(rec["behavior_name"]),
        "channel": str(rec["channel"]),
        "identity_grain": str(rec["identity_grain"]),
        "focal_id": _opt_int("focal_id"),
        "partner_id": _opt_int("partner_id"),
        "mouse_a_id": int(rec["mouse_a_id"]),
        "mouse_b_id": int(rec["mouse_b_id"]),
        "pair_id": str(rec["pair_id"]),
        "start_sample_idx": int(rec["start_sample_idx"]),
        "end_sample_idx": int(rec["end_sample_idx"]),
        "start_time_sec": _opt_float("start_time_sec"),
        "end_time_sec": _opt_float("end_time_sec"),
        "bout_duration_frames": int(rec["bout_duration_frames"]),
        "bout_duration_sec": float(rec["bout_duration_sec"]),
        "prepared_frame_idx_start": int(rec["prepared_frame_idx_start"]),
        "prepared_frame_idx_end": int(rec["prepared_frame_idx_end"]),
        "prepared_time_sec_start": float(rec["prepared_time_sec_start"]),
        "prepared_time_sec_end": float(rec["prepared_time_sec_end"]),
        "min_required_frames": int(min_required),
    }


def _optional_str_tuple(value: Any) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Sequence):
        return tuple(str(v) for v in value)
    raise ExportProfileError(f"Allow-list must be a sequence of strings; got {value!r}.")


def _empty_bout_index() -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in BOUT_INDEX_REQUIRED_COLUMNS})


def _empty_exclusion_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "bout_id": pd.Series(dtype=object),
            "source_s8_bout_id": pd.Series(dtype=object),
            "channel": pd.Series(dtype=object),
            "behavior_name": pd.Series(dtype=object),
            "bout_duration_frames": pd.Series(dtype="int64"),
            "excluded_reason": pd.Series(dtype=object),
            "matched_reasons": pd.Series(dtype=object),
            "min_required_frames": pd.Series(dtype="int64"),
        }
    )
