"""Adapt eligible S4 ``final_pose.parquet`` into legacy frame-feature tensors.

This module is independent of Pipeline 30 NPZ files. It pivots long-form S4
coordinates, maps provenance onto legacy impute-stage / quality arrays, and
copies S5 ``orig_frame_idx`` after joint alignment.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from combined_features.contracts import (
    ADAPTER_POLICY_VERSION,
    HEADSTAGE_NODE,
    IMPLANTATION_ABSENT,
    IMPLANTATION_AMBIGUOUS,
    IMPLANTATION_RESOLVED,
    IMPUTE_STAGE_KALMAN,
    IMPUTE_STAGE_LINEAR,
    IMPUTE_STAGE_NONE,
    NODE_ALIAS_PAIRS,
    REQUIRED_SCIENTIFIC_NODES,
    CombinedFeaturesError,
)
from pose_finalization.contracts import (
    FINAL_POSE_COLUMNS,
    HEADSTAGE_NODE_NAME,
    PROVENANCE_KALMAN_GAP_FILL,
    PROVENANCE_LINEAR_GAP_FILL,
    PROVENANCE_MISSING,
    PROVENANCE_PROTECTED_MANUAL_BLANK,
)


@dataclass(frozen=True, slots=True)
class ImplantationResolution:
    """Deterministic implanted-track resolution from headstage evidence.

    Sprint 2 must consume ``implanted_track_id`` only when ``status`` is
    ``resolved``. When unresolved, do **not** invent ``is_implanted`` from
    numeric track id ``0``.
    """

    status: str
    implanted_track_id: int | None
    evidence: Mapping[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "implanted_track_id": self.implanted_track_id,
            "evidence": dict(self.evidence),
        }


@dataclass(frozen=True, slots=True)
class AdaptedPose:
    """In-memory pose tensors required by legacy ``build_frame_features`` I/O."""

    tracks: tuple[int, ...]
    nodes: tuple[str, ...]
    x: np.ndarray
    y: np.ndarray
    t_120: np.ndarray
    orig_frame_idx: np.ndarray
    point_score: np.ndarray
    impute_stage: np.ndarray
    is_imputed: np.ndarray
    is_kf: np.ndarray
    is_pchip: np.ndarray
    was_refined: np.ndarray
    provenance: np.ndarray
    protected_manual_blank: np.ndarray
    was_smoothed: np.ndarray
    x_pre_smooth: np.ndarray
    y_pre_smooth: np.ndarray
    implantation: ImplantationResolution
    adapter_policy_version: str = ADAPTER_POLICY_VERSION

    @property
    def n_tracks(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_nodes(self) -> int:
        return int(self.x.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.x.shape[2])

    def directional_track_pairs(self) -> tuple[tuple[int, int], ...]:
        """Ordered ``(focal_id, partner_id)`` pairs with ``focal != partner``."""

        pairs: list[tuple[int, int]] = []
        for focal in self.tracks:
            for partner in self.tracks:
                if focal != partner:
                    pairs.append((focal, partner))
        return tuple(pairs)

    def to_frame_feature_arrays(self) -> dict[str, np.ndarray]:
        """Arrays historically loaded from ``dense_pose_120_imputed.npz``."""

        return {
            "x": self.x,
            "y": self.y,
            "t_120": self.t_120,
            "orig_frame_idx": self.orig_frame_idx,
            "impute_stage": self.impute_stage,
            "is_imputed": self.is_imputed,
            "was_refined": self.was_refined,
            "point_score": self.point_score,
        }

    def to_impute_meta(self) -> dict[str, Any]:
        """Metadata historically loaded from ``impute_meta.json``."""

        return {
            "tracks": list(self.tracks),
            "nodes": list(self.nodes),
            "adapter_policy_version": self.adapter_policy_version,
            "implantation": self.implantation.to_metadata(),
        }


def adapt_final_pose_to_frame_feature_input(
    final_pose: pd.DataFrame,
    *,
    orig_frame_idx: np.ndarray | Sequence[int],
    run_meta: Mapping[str, Any] | None = None,
) -> AdaptedPose:
    """Convert long-form S4 pose into FF-compatible tensors.

    Parameters
    ----------
    final_pose:
        Eligible ``final_pose.parquet`` table (post-smooth ``x``/``y``).
    orig_frame_idx:
        S5 nearest-neighbor prepared-frame indices aligned on ``sample_idx``.
        Required after joint eligibility; do not invent from S4 left/right.
    run_meta:
        Optional S4 ``run_meta.json`` (used for node-order hints only).
    """

    _require_final_pose_columns(final_pose)
    sample_idx = _unique_contiguous_samples(final_pose)
    n_samples = int(sample_idx.size)
    orig = np.asarray(orig_frame_idx, dtype=np.int64)
    if orig.shape != (n_samples,):
        raise CombinedFeaturesError(
            "orig_frame_idx must have shape (T,) matching unique S4 sample_idx count "
            f"({orig.shape} != ({n_samples},))."
        )

    tracks = _resolve_tracks(final_pose)
    base_nodes = _resolve_base_nodes(final_pose, run_meta=run_meta)
    nodes = _expand_node_aliases(base_nodes)
    _require_scientific_nodes(nodes)

    shape = (len(tracks), len(nodes), n_samples)
    x = np.full(shape, np.nan, dtype=np.float64)
    y = np.full(shape, np.nan, dtype=np.float64)
    x_pre = np.full(shape, np.nan, dtype=np.float64)
    y_pre = np.full(shape, np.nan, dtype=np.float64)
    point_score = np.full(shape, np.nan, dtype=np.float64)
    provenance = np.full(shape, PROVENANCE_MISSING, dtype=np.uint8)
    protected = np.zeros(shape, dtype=bool)
    was_smoothed = np.zeros(shape, dtype=bool)

    track_index = {track: i for i, track in enumerate(tracks)}
    node_index = {name: i for i, name in enumerate(nodes)}
    # Map every parquet node name onto all alias slots it fills.
    fill_targets = _node_fill_targets(base_nodes, nodes)

    # Reject duplicate scientific keys before writing.
    key = final_pose.duplicated(subset=["sample_idx", "track", "node"], keep=False)
    if bool(key.any()):
        raise CombinedFeaturesError(
            "S4 final_pose has duplicate (sample_idx, track, node) keys."
        )

    sample_values = final_pose["sample_idx"].to_numpy(dtype=np.int64)
    track_values = np.asarray(
        [_normalize_track_id(v) for v in final_pose["track"].to_numpy()],
        dtype=object,
    )
    node_values = final_pose["node"].astype(str).to_numpy()
    x_vals = final_pose["x"].to_numpy(dtype=np.float64, copy=False)
    y_vals = final_pose["y"].to_numpy(dtype=np.float64, copy=False)
    x_pre_vals = final_pose["x_pre_smooth"].to_numpy(dtype=np.float64, copy=False)
    y_pre_vals = final_pose["y_pre_smooth"].to_numpy(dtype=np.float64, copy=False)
    score_vals = final_pose["point_score"].to_numpy(dtype=np.float64, copy=False)
    prov_vals = final_pose["provenance"].to_numpy(dtype=np.uint8, copy=False)
    protected_vals = final_pose["protected_manual_blank"].to_numpy(dtype=bool, copy=False)
    smoothed_vals = final_pose["was_smoothed"].to_numpy(dtype=bool, copy=False)

    for row_i in range(len(final_pose)):
        track_id = track_values[row_i]
        if not isinstance(track_id, (int, np.integer)):
            raise CombinedFeaturesError(f"Non-integer track id after normalization: {track_id!r}")
        track_id = int(track_id)
        if track_id not in track_index:
            continue
        node_name = str(node_values[row_i])
        targets = fill_targets.get(node_name)
        if not targets:
            continue
        t = int(sample_values[row_i])
        ti = track_index[track_id]
        for node_name_out in targets:
            ni = node_index[node_name_out]
            x[ti, ni, t] = x_vals[row_i]
            y[ti, ni, t] = y_vals[row_i]
            x_pre[ti, ni, t] = x_pre_vals[row_i]
            y_pre[ti, ni, t] = y_pre_vals[row_i]
            point_score[ti, ni, t] = score_vals[row_i]
            provenance[ti, ni, t] = prov_vals[row_i]
            protected[ti, ni, t] = bool(protected_vals[row_i])
            was_smoothed[ti, ni, t] = bool(smoothed_vals[row_i])

    impute_stage, is_imputed, is_kf, is_pchip, was_refined = map_provenance_to_quality(
        provenance
    )
    t_120 = _time_sec_per_sample(final_pose, sample_idx)
    implantation = resolve_implanted_track(
        tracks=tracks,
        nodes=nodes,
        x=x,
        y=y,
        provenance=provenance,
    )

    return AdaptedPose(
        tracks=tracks,
        nodes=nodes,
        x=x,
        y=y,
        t_120=t_120,
        orig_frame_idx=orig.copy(),
        point_score=point_score,
        impute_stage=impute_stage,
        is_imputed=is_imputed,
        is_kf=is_kf,
        is_pchip=is_pchip,
        was_refined=was_refined,
        provenance=provenance,
        protected_manual_blank=protected,
        was_smoothed=was_smoothed,
        x_pre_smooth=x_pre,
        y_pre_smooth=y_pre,
        implantation=implantation,
    )


def map_provenance_to_quality(
    provenance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Map S4 provenance codes onto legacy FF quality arrays.

    Binding policy (implementation contract §1.8):

    - ``is_kf`` / impute_stage Kalman only for ``kalman_gap_fill``
    - ``is_pchip`` always false (S4 has no PCHIP)
    - ``is_imputed`` true for linear∪kalman only
    - ``was_refined`` always false
    - temporal resample / identity / smoothing / blanks are not Kalman
    """

    prov = np.asarray(provenance, dtype=np.uint8)
    impute_stage = np.full(prov.shape, IMPUTE_STAGE_NONE, dtype=np.int8)
    impute_stage[prov == PROVENANCE_LINEAR_GAP_FILL] = IMPUTE_STAGE_LINEAR
    impute_stage[prov == PROVENANCE_KALMAN_GAP_FILL] = IMPUTE_STAGE_KALMAN

    is_kf = prov == PROVENANCE_KALMAN_GAP_FILL
    is_pchip = np.zeros(prov.shape, dtype=bool)
    is_imputed = (prov == PROVENANCE_LINEAR_GAP_FILL) | (prov == PROVENANCE_KALMAN_GAP_FILL)
    was_refined = np.zeros(prov.shape, dtype=bool)
    return impute_stage, is_imputed, is_kf, is_pchip, was_refined


def resolve_implanted_track(
    *,
    tracks: Sequence[int],
    nodes: Sequence[str],
    x: np.ndarray,
    y: np.ndarray,
    provenance: np.ndarray | None = None,
) -> ImplantationResolution:
    """Resolve implanted identity from headstage observations only when unambiguous.

    Rule
    ----
    Count samples with finite headstage ``(x, y)`` per track.

    - Exactly one track has ≥1 finite headstage observation → that track is
      implanted (``status=resolved``).
    - Zero tracks → ``status=headstage_absent`` / unresolved.
    - Two or more tracks → ``status=headstage_ambiguous`` / unresolved.

    Numeric track id ``0`` is **never** used as a default. Profile
    ``implanted_track: 0`` is intentionally ignored here so Sprint 2 does not
    silently reintroduce the legacy assumption.
    """

    track_list = [int(t) for t in tracks]
    node_list = [str(n) for n in nodes]
    evidence: dict[str, Any] = {
        "method": "unique_finite_headstage_owner",
        "headstage_node": HEADSTAGE_NODE,
        "finite_observation_counts": {},
        "notes": (
            "Sprint 2 must set is_implanted from implanted_track_id only when "
            "status=='resolved'; otherwise preserve uncertainty (do not use id==0)."
        ),
    }

    if HEADSTAGE_NODE not in node_list:
        evidence["finite_observation_counts"] = {str(t): 0 for t in track_list}
        return ImplantationResolution(
            status=IMPLANTATION_ABSENT,
            implanted_track_id=None,
            evidence=evidence,
        )

    ni = node_list.index(HEADSTAGE_NODE)
    counts: dict[int, int] = {}
    for ti, track_id in enumerate(track_list):
        finite = np.isfinite(x[ti, ni, :]) & np.isfinite(y[ti, ni, :])
        if provenance is not None:
            # Protected blanks / missing provenance are not positive evidence.
            prov = np.asarray(provenance[ti, ni, :], dtype=np.uint8)
            finite = finite & (prov != PROVENANCE_PROTECTED_MANUAL_BLANK)
            finite = finite & (prov != PROVENANCE_MISSING)
        counts[track_id] = int(np.count_nonzero(finite))
    evidence["finite_observation_counts"] = {str(k): v for k, v in counts.items()}

    owners = [track_id for track_id, count in counts.items() if count > 0]
    if len(owners) == 1:
        evidence["resolution"] = IMPLANTATION_RESOLVED
        return ImplantationResolution(
            status=IMPLANTATION_RESOLVED,
            implanted_track_id=int(owners[0]),
            evidence=evidence,
        )
    if len(owners) == 0:
        return ImplantationResolution(
            status=IMPLANTATION_ABSENT,
            implanted_track_id=None,
            evidence=evidence,
        )
    return ImplantationResolution(
        status=IMPLANTATION_AMBIGUOUS,
        implanted_track_id=None,
        evidence=evidence,
    )


def _require_final_pose_columns(final_pose: pd.DataFrame) -> None:
    missing = [name for name in FINAL_POSE_COLUMNS if name not in final_pose.columns]
    if missing:
        raise CombinedFeaturesError(
            "final_pose is missing required columns: " + ", ".join(missing)
        )


def _unique_contiguous_samples(final_pose: pd.DataFrame) -> np.ndarray:
    sample_idx = np.asarray(sorted(final_pose["sample_idx"].unique()), dtype=np.int64)
    if sample_idx.size == 0:
        raise CombinedFeaturesError("final_pose has no sample_idx values.")
    expected = np.arange(int(sample_idx.size), dtype=np.int64)
    if not np.array_equal(sample_idx, expected):
        raise CombinedFeaturesError(
            "S4 sample_idx must be unique contiguous integers [0 .. N-1]."
        )
    return sample_idx


def _normalize_track_id(value: object) -> object:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return value
    return value


def _resolve_tracks(final_pose: pd.DataFrame) -> tuple[int, ...]:
    raw = [_normalize_track_id(v) for v in final_pose["track"].unique().tolist()]
    tracks: list[int] = []
    for value in raw:
        if not isinstance(value, (int, np.integer)):
            raise CombinedFeaturesError(
                f"Track ids must be integer-coercible for directional features; got {value!r}."
            )
        tracks.append(int(value))
    return tuple(sorted(set(tracks)))


def _resolve_base_nodes(
    final_pose: pd.DataFrame,
    *,
    run_meta: Mapping[str, Any] | None,
) -> tuple[str, ...]:
    observed = [str(v) for v in final_pose["node"].astype(str).unique().tolist()]
    preferred: list[str] = []
    if run_meta is not None:
        dense = run_meta.get("dense_summary")
        if isinstance(dense, Mapping):
            names = dense.get("node_names")
            if isinstance(names, list):
                preferred = [str(v) for v in names]
    if preferred:
        # Preserve skeleton order from dense_summary when present; append extras.
        ordered = [name for name in preferred if name in observed]
        extras = [name for name in observed if name not in ordered]
        return tuple(ordered + extras)
    # Stable first-occurrence order in the table.
    seen: list[str] = []
    for name in final_pose["node"].astype(str).tolist():
        if name not in seen:
            seen.append(name)
    return tuple(seen)


def _expand_node_aliases(base_nodes: Sequence[str]) -> tuple[str, ...]:
    """Ensure both sides of each alias pair appear in the FF node axis."""

    nodes = list(base_nodes)
    for canon, alias in NODE_ALIAS_PAIRS:
        has_canon = canon in nodes
        has_alias = alias in nodes
        if has_canon and not has_alias:
            nodes.insert(nodes.index(canon) + 1, alias)
        elif has_alias and not has_canon:
            nodes.insert(nodes.index(alias), canon)
    # Headstage stays if present; no alias.
    if HEADSTAGE_NODE_NAME in base_nodes and HEADSTAGE_NODE_NAME not in nodes:
        nodes.append(HEADSTAGE_NODE_NAME)
    return tuple(nodes)


def _node_fill_targets(
    base_nodes: Sequence[str],
    expanded_nodes: Sequence[str],
) -> dict[str, tuple[str, ...]]:
    """Map each source node name onto one or more expanded tensor slots."""

    targets: dict[str, list[str]] = {name: [name] for name in expanded_nodes if name in base_nodes}
    for canon, alias in NODE_ALIAS_PAIRS:
        if canon in base_nodes and alias in expanded_nodes:
            targets.setdefault(canon, [canon])
            if alias not in targets[canon]:
                targets[canon].append(alias)
        if alias in base_nodes and canon in expanded_nodes:
            targets.setdefault(alias, [alias])
            if canon not in targets[alias]:
                targets[alias].append(canon)
    if HEADSTAGE_NODE in expanded_nodes and HEADSTAGE_NODE in base_nodes:
        targets[HEADSTAGE_NODE] = [HEADSTAGE_NODE]
    return {key: tuple(vals) for key, vals in targets.items()}


def _require_scientific_nodes(nodes: Sequence[str]) -> None:
    node_set = set(nodes)
    # spine/spine_base and tail/tail_base are satisfied by either name after expansion.
    required_groups = (
        ("nose",),
        ("neck",),
        ("spine", "spine_base"),
        ("tail", "tail_base"),
    )
    missing: list[str] = []
    for group in required_groups:
        if not any(name in node_set for name in group):
            missing.append("/".join(group))
    # REQUIRED_SCIENTIFIC_NODES documents the logical set; keep it referenced.
    _ = REQUIRED_SCIENTIFIC_NODES
    if missing:
        raise CombinedFeaturesError(
            "S4 pose is missing required scientific nodes after alias resolution: "
            + ", ".join(missing)
        )


def _time_sec_per_sample(final_pose: pd.DataFrame, sample_idx: np.ndarray) -> np.ndarray:
    grouped = final_pose.groupby("sample_idx", sort=True)["time_sec"]
    if int((grouped.nunique(dropna=False) > 1).sum()) > 0:
        raise CombinedFeaturesError(
            "Conflicting time_sec values for the same sample_idx."
        )
    times = np.asarray(grouped.first().loc[sample_idx].to_numpy(), dtype=np.float64)
    if times.shape != sample_idx.shape or not np.all(np.isfinite(times)):
        raise CombinedFeaturesError("Could not build finite t_120 from S4 time_sec.")
    return times
