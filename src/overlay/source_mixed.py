"""Resolve and pin an explicit mixed pose + behavioral QC source pair (C3-4).

Never auto-selects the newest pose or behavioral run. Never silently substitutes
S8↔S10 or S3↔S4. Rejects unsupported combinations and incompatible lineage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from behavioral_review.sync import SyncMap
from overlay.contracts import (
    MIXED_SUPPORTED_COMBINATIONS,
    BehaviorSourceKind,
    OverlaySourceError,
    PoseSourceKind,
)
from overlay.source import (
    OverlayQcSource,
    resolve_overlay_qc_source,
    verify_overlay_qc_source_integrity,
)
from overlay.source_s3 import (
    OverlayS3QcSource,
    resolve_overlay_s3_qc_source,
    verify_overlay_s3_qc_source_integrity,
)
from overlay.source_s4 import (
    OverlayS4QcSource,
    resolve_overlay_s4_qc_source,
    verify_overlay_s4_qc_source_integrity,
)
from overlay.timing import resolve_prepared_frame_timing
from results_packaging.media_csv_export.video import VideoProbe


@dataclass(frozen=True, slots=True)
class OverlayMixedQcSource:
    """Pinned compatible pose + behavioral pair for mixed QC export."""

    pose_source_kind: PoseSourceKind
    behavior_source_kind: BehaviorSourceKind
    pose_s3: OverlayS3QcSource | None
    pose_s4: OverlayS4QcSource | None
    behavior: OverlayQcSource
    session_root: Path
    prepared_video_path: Path
    prepared_video_sha256: str
    video_probe: VideoProbe
    # Behavioral SyncMap (S8/S10 package sync_index). S4 pose keeps its own
    # SyncMap on pose_s4; S3 pose uses prepared-frame indexing directly.
    sync: SyncMap
    sync_authority: str
    compatibility_notes: tuple[str, ...]
    source_digests: dict[str, str | None]
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def pose_run_id(self) -> str:
        if self.pose_source_kind is PoseSourceKind.S4_FINALIZED:
            assert self.pose_s4 is not None
            return self.pose_s4.s4_run_id
        assert self.pose_s3 is not None
        return self.pose_s3.s3_run_id

    @property
    def pose_run_dir(self) -> Path:
        if self.pose_source_kind is PoseSourceKind.S4_FINALIZED:
            assert self.pose_s4 is not None
            return self.pose_s4.s4_run_dir
        assert self.pose_s3 is not None
        return self.pose_s3.s3_run_dir

    @property
    def skeleton_edges(self) -> tuple[tuple[str, str], ...]:
        if self.pose_source_kind is PoseSourceKind.S4_FINALIZED:
            assert self.pose_s4 is not None
            return self.pose_s4.skeleton_edges
        assert self.pose_s3 is not None
        return self.pose_s3.skeleton_edges

    @property
    def skeleton_source(self) -> str | None:
        if self.pose_source_kind is PoseSourceKind.S4_FINALIZED:
            assert self.pose_s4 is not None
            return self.pose_s4.skeleton_source
        assert self.pose_s3 is not None
        return self.pose_s3.skeleton_source

    def skeleton_disclosure(self) -> dict[str, Any]:
        if self.pose_source_kind is PoseSourceKind.S4_FINALIZED:
            assert self.pose_s4 is not None
            return self.pose_s4.skeleton_disclosure()
        assert self.pose_s3 is not None
        return self.pose_s3.skeleton_disclosure()

    def pose_source_digests(self) -> dict[str, str | None]:
        if self.pose_source_kind is PoseSourceKind.S4_FINALIZED:
            assert self.pose_s4 is not None
            return dict(self.pose_s4.source_digests)
        assert self.pose_s3 is not None
        return dict(self.pose_s3.source_digests)


def resolve_overlay_mixed_qc_source(
    *,
    pose_source_kind: PoseSourceKind | str,
    behavior_source_kind: BehaviorSourceKind | str,
    s4_run_dir: Path | str | None = None,
    s3_run_dir: Path | str | None = None,
    package_dir: Path | str | None = None,
    review_run_dir: Path | str | None = None,
    prepared_interval: tuple[int, int] | None = None,
) -> OverlayMixedQcSource:
    """Open one explicitly selected compatible pose + behavioral source pair.

    Supported combinations only:
    - S4 finalized + S8 automated
    - S4 finalized + published S10 reviewed
    - S3 accepted tracked + S8 automated

    S3 + S10 is rejected. Silent S8↔S10 or S3↔S4 substitution is refused.
    """

    pose_kind = _normalize_pose_kind(pose_source_kind)
    behavior_kind = _normalize_behavior_kind(behavior_source_kind)
    if (pose_kind, behavior_kind) not in MIXED_SUPPORTED_COMBINATIONS:
        raise OverlaySourceError(
            "Unsupported mixed QC combination "
            f"{pose_kind.value}+{behavior_kind.value}. Supported: "
            "s4_finalized+s8_automated, s4_finalized+s10_reviewed, "
            "s3_accepted+s8_automated. "
            "Refusing silent substitution of pose or behavioral sources."
        )

    # Validate explicit path arguments before opening any science trees so
    # incompatible combinations fail with actionable substitution messages.
    if pose_kind is PoseSourceKind.S4_FINALIZED:
        if s4_run_dir is None:
            raise OverlaySourceError(
                "Mixed QC with S4 pose requires an explicit s4_run_dir "
                "(never auto-selects newest)."
            )
        if s3_run_dir is not None:
            raise OverlaySourceError(
                "Mixed QC with S4 pose must not receive s3_run_dir "
                "(refusing silent S3 substitution)."
            )
    else:
        if s3_run_dir is None:
            raise OverlaySourceError(
                "Mixed QC with S3 pose requires an explicit s3_run_dir "
                "(accepted tracked only; never auto-selects newest)."
            )
        if s4_run_dir is not None:
            raise OverlaySourceError(
                "Mixed QC with S3 pose must not receive s4_run_dir "
                "(refusing silent S4 finalization implication)."
            )

    if behavior_kind is BehaviorSourceKind.S8_AUTOMATED:
        if package_dir is None:
            raise OverlaySourceError(
                "Mixed QC automated behavior requires an explicit package_dir."
            )
        if review_run_dir is not None:
            raise OverlaySourceError(
                "Mixed QC automated behavior must not receive review_run_dir "
                "(refusing silent S10 substitution for S8)."
            )
    else:
        if review_run_dir is None:
            raise OverlaySourceError(
                "Mixed QC reviewed behavior requires an explicit published "
                "review_run_dir (working drafts rejected)."
            )

    pose_s3: OverlayS3QcSource | None = None
    pose_s4: OverlayS4QcSource | None = None
    if pose_kind is PoseSourceKind.S4_FINALIZED:
        assert s4_run_dir is not None  # validated above
        pose_s4 = resolve_overlay_s4_qc_source(s4_run_dir=s4_run_dir)
        pose_session = pose_s4.session_root
        pose_video = pose_s4.prepared_video_path
        pose_video_sha = pose_s4.prepared_video_sha256
        pose_sync_sha = pose_s4.prepared_sync_sha256
        pose_probe = pose_s4.video_probe
        pose_sync = pose_s4.sync
    else:
        assert s3_run_dir is not None  # validated above
        pose_s3 = resolve_overlay_s3_qc_source(s3_run_dir=s3_run_dir)
        pose_session = pose_s3.session_root
        pose_video = pose_s3.prepared_video_path
        pose_video_sha = pose_s3.prepared_video_sha256
        pose_sync_sha = pose_s3.prepared_sync_sha256
        pose_probe = pose_s3.video_probe
        pose_sync = pose_s3.sync

    behavior = resolve_overlay_qc_source(
        source_kind=behavior_kind,
        package_dir=package_dir,
        review_run_dir=review_run_dir,
    )

    notes = _assert_compatible(
        pose_kind=pose_kind,
        pose_session=pose_session,
        pose_video=pose_video,
        pose_video_sha=pose_video_sha,
        pose_sync_sha=pose_sync_sha,
        pose_sync=pose_sync,
        pose_probe=pose_probe,
        behavior=behavior,
        prepared_interval=prepared_interval,
    )

    digests: dict[str, str | None] = {
        "prepared_video_sha256": pose_video_sha,
        "pose_prepared_sync_sha256": pose_sync_sha,
        **{f"pose_{k}": v for k, v in (
            pose_s4.source_digests if pose_s4 is not None else pose_s3.source_digests  # type: ignore[union-attr]
        ).items()},
        **{f"behavior_{k}": v for k, v in behavior.source_digests.items()},
    }

    return OverlayMixedQcSource(
        pose_source_kind=pose_kind,
        behavior_source_kind=behavior_kind,
        pose_s3=pose_s3,
        pose_s4=pose_s4,
        behavior=behavior,
        session_root=pose_session.resolve(),
        prepared_video_path=pose_video.resolve(),
        prepared_video_sha256=pose_video_sha,
        video_probe=pose_probe,
        sync=behavior.sync,
        sync_authority=(
            "dual_independent:"
            "pose_own_map_or_prepared_direct+"
            "behavior_package_sync_index_last_in_plateau"
        ),
        compatibility_notes=notes,
        source_digests=digests,
        provenance={
            "export_family": "mixed_pose_behavior",
            "pose_source_kind": pose_kind.value,
            "behavior_source_kind": behavior_kind.value,
            "pose_run_id": (
                pose_s4.s4_run_id if pose_s4 is not None else pose_s3.s3_run_id  # type: ignore[union-attr]
            ),
            "pose_run_dir": str(
                pose_s4.s4_run_dir if pose_s4 is not None else pose_s3.s3_run_dir  # type: ignore[union-attr]
            ),
            "package_run_id": behavior.package_run_id,
            "review_run_id": behavior.review_run_id,
            "session_root": str(pose_session.resolve()),
            "prepared_video_path": str(pose_video.resolve()),
            "prepared_video_sha256": pose_video_sha,
            "timing_authority": (
                "dual_independent:"
                "pose_own_map_or_prepared_direct+"
                "behavior_package_sync_index_last_in_plateau"
            ),
            "pose_timing": (
                "s4_dense_via_pose_syncmap_last_in_plateau"
                if pose_kind is PoseSourceKind.S4_FINALIZED
                else "s3_prepared_frame_direct_mapping"
            ),
            "behavior_timing": "dense_via_behavior_syncmap_last_in_plateau",
            "compatibility_notes": list(notes),
            "pose_provenance": (
                dict(pose_s4.provenance)
                if pose_s4 is not None
                else dict(pose_s3.provenance)  # type: ignore[union-attr]
            ),
            "behavior_provenance": dict(behavior.provenance),
        },
    )


def verify_overlay_mixed_qc_source_integrity(source: OverlayMixedQcSource) -> None:
    """Recheck pose and behavioral digests / eligibility before publication."""

    if source.pose_source_kind is PoseSourceKind.S4_FINALIZED:
        if source.pose_s4 is None:
            raise OverlaySourceError("Mixed S4 source missing pose_s4 handle.")
        verify_overlay_s4_qc_source_integrity(source.pose_s4)
    else:
        if source.pose_s3 is None:
            raise OverlaySourceError("Mixed S3 source missing pose_s3 handle.")
        verify_overlay_s3_qc_source_integrity(source.pose_s3)
    verify_overlay_qc_source_integrity(source.behavior)

    live_video = (
        source.pose_s4.prepared_video_sha256
        if source.pose_s4 is not None
        else source.pose_s3.prepared_video_sha256  # type: ignore[union-attr]
    )
    if str(live_video) != str(source.prepared_video_sha256):
        raise OverlaySourceError(
            "Mixed QC prepared-video digest changed during export."
        )
    if source.behavior.video_sha256 is not None and str(
        source.behavior.video_sha256
    ) != str(source.prepared_video_sha256):
        raise OverlaySourceError(
            "Mixed QC behavioral prepared-video digest diverged from pose video."
        )


def _normalize_pose_kind(value: PoseSourceKind | str) -> PoseSourceKind:
    if isinstance(value, PoseSourceKind):
        return value
    text = str(value).strip().lower()
    aliases = {
        PoseSourceKind.S3_ACCEPTED.value: PoseSourceKind.S3_ACCEPTED,
        "s3": PoseSourceKind.S3_ACCEPTED,
        "s3_tracked": PoseSourceKind.S3_ACCEPTED,
        "s3_pose": PoseSourceKind.S3_ACCEPTED,
        "accepted": PoseSourceKind.S3_ACCEPTED,
        PoseSourceKind.S4_FINALIZED.value: PoseSourceKind.S4_FINALIZED,
        "s4": PoseSourceKind.S4_FINALIZED,
        "s4_pose": PoseSourceKind.S4_FINALIZED,
        "finalized": PoseSourceKind.S4_FINALIZED,
    }
    if text in aliases:
        return aliases[text]
    raise OverlaySourceError(f"Unsupported mixed pose source kind: {value!r}.")


def _normalize_behavior_kind(
    value: BehaviorSourceKind | str,
) -> BehaviorSourceKind:
    if isinstance(value, BehaviorSourceKind):
        return value
    # Reuse behavioral resolver normalization via resolve_overlay_qc_source gates;
    # keep local aliases explicit to avoid silent S8↔S10 swaps.
    text = str(value).strip().lower()
    if text in {
        BehaviorSourceKind.S8_AUTOMATED.value,
        "automated",
        "s8",
        "s9",
    }:
        return BehaviorSourceKind.S8_AUTOMATED
    if text in {
        BehaviorSourceKind.S10_REVIEWED.value,
        "reviewed",
        "published",
        "s10",
    }:
        return BehaviorSourceKind.S10_REVIEWED
    raise OverlaySourceError(
        f"Unsupported mixed behavioral source kind: {value!r}."
    )


def _assert_compatible(
    *,
    pose_kind: PoseSourceKind,
    pose_session: Path,
    pose_video: Path,
    pose_video_sha: str,
    pose_sync_sha: str,
    pose_sync: SyncMap,
    pose_probe: VideoProbe,
    behavior: OverlayQcSource,
    prepared_interval: tuple[int, int] | None,
) -> tuple[str, ...]:
    notes: list[str] = []
    pose_session_r = pose_session.resolve()
    behavior_session = (
        behavior.session_root.resolve()
        if behavior.session_root is not None
        else None
    )
    if behavior_session is None:
        raise OverlaySourceError(
            "Behavioral source has no session_root; cannot verify same D0 session "
            "identity for mixed QC."
        )
    if pose_session_r != behavior_session:
        raise OverlaySourceError(
            "Mixed QC rejects cross-session sources: "
            f"pose session={pose_session_r} behavior session={behavior_session}."
        )
    notes.append(f"session_identity_ok:{pose_session_r.name}")

    behavior_video = Path(behavior.video_path).resolve()
    pose_video_r = pose_video.resolve()
    if pose_video_r != behavior_video:
        raise OverlaySourceError(
            "Mixed QC requires the same authoritative S1 prepared video path: "
            f"pose={pose_video_r} behavior={behavior_video}."
        )
    if behavior.video_sha256 is None:
        raise OverlaySourceError(
            "Behavioral source is missing prepared_video_sha256 for mixed "
            "compatibility checks."
        )
    if str(behavior.video_sha256) != str(pose_video_sha):
        raise OverlaySourceError(
            "Mixed QC prepared-video digest mismatch between pose and behavior: "
            f"pose={pose_video_sha!r} behavior={behavior.video_sha256!r}."
        )
    notes.append("prepared_video_digest_ok")

    if abs(float(pose_probe.fps) - float(behavior.video_probe.fps)) > 1e-6:
        raise OverlaySourceError(
            "Mixed QC prepared-video FPS mismatch between pose and behavior "
            f"probes ({pose_probe.fps} vs {behavior.video_probe.fps})."
        )
    if int(pose_probe.n_frames) != int(behavior.video_probe.n_frames):
        raise OverlaySourceError(
            "Mixed QC prepared-video frame-count mismatch between pose and "
            f"behavior probes ({pose_probe.n_frames} vs "
            f"{behavior.video_probe.n_frames})."
        )
    if (
        int(pose_probe.width) != int(behavior.video_probe.width)
        or int(pose_probe.height) != int(behavior.video_probe.height)
    ):
        raise OverlaySourceError(
            "Mixed QC prepared-pixel geometry mismatch between pose and behavior "
            f"probes ({pose_probe.width}x{pose_probe.height} vs "
            f"{behavior.video_probe.width}x{behavior.video_probe.height})."
        )
    notes.append("prepared_pixel_geometry_ok")

    behavior_sync_sha = behavior.source_digests.get("sync_index_sha256")
    if behavior_sync_sha:
        notes.append(f"behavior_sync_index_sha256={behavior_sync_sha}")
    notes.append(f"pose_prepared_sync_sha256={pose_sync_sha}")

    # Timing authorities are independent. Compatibility compares session/video
    # identity and prepared-domain alignment, not equality of dense sample
    # indices across unrelated SyncMaps.
    start, end = prepared_interval or (0, min(29, int(pose_probe.n_frames) - 1))
    start = max(0, int(start))
    end = min(int(end), int(pose_probe.n_frames) - 1)
    if start > end:
        raise OverlaySourceError(
            f"Invalid prepared interval for mixed compatibility: [{start}, {end}]."
        )

    checked = 0
    behavior_covered = 0
    pose_covered = 0
    independent_dense_diffs = 0
    for prep in range(start, end + 1):
        behavior_timing = resolve_prepared_frame_timing(behavior.sync, prep)
        checked += 1
        if behavior_timing.available:
            behavior_covered += 1
        else:
            notes.append(f"behavior_timing_unavailable_at_prepared_frame={prep}")
        if pose_kind is PoseSourceKind.S4_FINALIZED:
            pose_timing = resolve_prepared_frame_timing(pose_sync, prep)
            if pose_timing.available:
                pose_covered += 1
            else:
                notes.append(f"pose_timing_unavailable_at_prepared_frame={prep}")
            if (
                behavior_timing.available
                and pose_timing.available
                and behavior_timing.dense_sample_idx != pose_timing.dense_sample_idx
            ):
                independent_dense_diffs += 1
                if independent_dense_diffs <= 3:
                    notes.append(
                        f"independent_dense_at_prep={prep}:"
                        f"pose={pose_timing.dense_sample_idx}:"
                        f"behavior={behavior_timing.dense_sample_idx}"
                    )
        # S3 pose is prepared-domain; never invent a dense index for pose.
    if behavior_covered == 0:
        raise OverlaySourceError(
            "Mixed QC behavioral SyncMap has no LAST_IN_PLATEAU coverage over "
            f"prepared interval [{start}, {end}] (same prepared video required, "
            "but behavioral dense authority is empty for the planned window)."
        )
    if pose_kind is PoseSourceKind.S4_FINALIZED and pose_covered == 0:
        raise OverlaySourceError(
            "Mixed QC S4 pose SyncMap has no LAST_IN_PLATEAU coverage over "
            f"prepared interval [{start}, {end}] (pose keeps its own dense "
            "authority; refusing an empty pose map for the planned window)."
        )
    notes.append(
        f"timing_alignment_ok:checked_prepared_frames={checked} "
        f"behavior_covered={behavior_covered} pose_covered="
        f"{pose_covered if pose_kind is PoseSourceKind.S4_FINALIZED else 'n/a_s3'} "
        f"independent_dense_diffs={independent_dense_diffs} "
        f"interval=[{start},{end}] policy=last_in_plateau"
    )
    if pose_kind is PoseSourceKind.S3_ACCEPTED:
        notes.append(
            "s3_pose_uses_prepared_frame_mapping;"
            "behavior_uses_dense_last_in_plateau_independently"
        )
    else:
        notes.append(
            "s4_pose_uses_pose_syncmap_last_in_plateau_independently;"
            "behavior_uses_behavior_syncmap_last_in_plateau_independently;"
            "dense_sample_indices_not_forced_equal"
        )
    return tuple(notes)


__all__ = [
    "OverlayMixedQcSource",
    "resolve_overlay_mixed_qc_source",
    "verify_overlay_mixed_qc_source_integrity",
]
