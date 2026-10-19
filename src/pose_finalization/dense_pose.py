"""Dense 120 Hz pose construction, manual-blank protection, and gap fill.

Loads a validated accepted S3 ``tracked_pose.parquet``, maps coordinates onto
the Sprint 1 ``TimingMap``, protects explicit manual node blanks from
``manual_corrections.json``, and applies configurable internal gap filling.

Upstream S1/S2/S3 artifacts are read-only.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from pose_finalization.contracts import (
    PROVENANCE_IDENTITY_CORRECTED,
    PROVENANCE_MISSING,
    PROVENANCE_ORIGINAL,
    PROVENANCE_PROTECTED_MANUAL_BLANK,
    PROVENANCE_TEMPORAL_RESAMPLE,
    PoseFinalizationError,
)
from pose_finalization.gap_fill import GapFillConfig, fill_coordinate_gaps
from pose_finalization.timing import TimingMap
from tracking_correction.contracts import (
    MANUAL_ACTION_BLANK_NODE,
    MANUAL_ACTION_SWAP_IDENTITIES,
    MANUAL_ACTION_SWAP_NODE,
    MANUAL_CORRECTIONS_FILENAME,
)

_FRAME_CANDIDATES = ("prepared_frame_idx", "frame_idx", "frame")


@dataclass(frozen=True, slots=True)
class ManualBlankInterval:
    """One explicit manual blank on the prepared-frame domain."""

    start_frame: int
    end_frame: int
    track: int
    node: str


@dataclass(frozen=True, slots=True)
class ManualIdentitySwapInterval:
    """Active manual ``swap_identities`` covering an inclusive frame interval."""

    start_frame: int
    end_frame: int


@dataclass(frozen=True, slots=True)
class ManualIdentitySwapNode:
    """Active manual ``swap_node`` at one prepared frame for one node."""

    frame_idx: int
    node: str


@dataclass(frozen=True, slots=True)
class ManualCorrectionsEffects:
    """Active S3 manual-correction stack effects relevant to S4 provenance.

    S3 persists only the current edit stack in ``manual_corrections.json``:
    undo pops the last record; reset clears the file. Historical undone edits
    are therefore absent and must not affect provenance.
    """

    blanks: tuple[ManualBlankInterval, ...] = ()
    identity_swap_intervals: tuple[ManualIdentitySwapInterval, ...] = ()
    identity_swap_nodes: tuple[ManualIdentitySwapNode, ...] = ()


@dataclass(frozen=True, slots=True)
class DensePoseResult:
    """In-memory dense pose on the uniform target timeline.

    Arrays are shaped ``(n_samples, n_tracks, n_nodes)`` unless noted.
    ``x_pre_impute`` / ``y_pre_impute`` preserve coordinates after temporal
    resampling but before gap filling, for later SavGol / export stages.
    """

    timing_map: TimingMap
    track_ids: np.ndarray
    node_names: np.ndarray
    x: np.ndarray
    y: np.ndarray
    x_pre_impute: np.ndarray
    y_pre_impute: np.ndarray
    provenance: np.ndarray
    protected: np.ndarray
    point_score: np.ndarray
    instance_score: np.ndarray
    left_source_frame_idx: np.ndarray
    right_source_frame_idx: np.ndarray
    interpolation_weight: np.ndarray
    is_exact_source_sample: np.ndarray
    gap_fill_config: GapFillConfig
    tracked_pose_path: Path | None = None
    manual_corrections_path: Path | None = None
    manual_blank_intervals: tuple[ManualBlankInterval, ...] = ()
    manual_identity_swap_intervals: tuple[ManualIdentitySwapInterval, ...] = ()
    manual_identity_swap_nodes: tuple[ManualIdentitySwapNode, ...] = ()

    @property
    def n_samples(self) -> int:
        return int(self.x.shape[0])

    @property
    def n_tracks(self) -> int:
        return int(self.x.shape[1])

    @property
    def n_nodes(self) -> int:
        return int(self.x.shape[2])


def load_tracked_pose_table(tracked_pose_path: Path | str) -> pd.DataFrame:
    """Load accepted tracked pose parquet (read-only)."""

    path = Path(tracked_pose_path).expanduser().resolve()
    if not path.is_file():
        raise PoseFinalizationError(f"tracked_pose.parquet is missing: {path}")
    try:
        frame = pd.read_parquet(path)
    except Exception as exc:  # noqa: BLE001 — surface as S4 error
        raise PoseFinalizationError(
            f"Could not read tracked_pose.parquet: {path}: {exc}"
        ) from exc
    if frame.empty:
        raise PoseFinalizationError(f"tracked_pose.parquet is empty: {path}")
    _require_columns(frame, ("track", "node", "x", "y"))
    return frame


def load_manual_corrections_effects(
    manual_corrections_path: Path | str | None,
) -> ManualCorrectionsEffects:
    """Load active manual blank and identity-swap effects (read-only).

    Uses the persisted S3 edit stack only. Undo/reset remove records from that
    file, so inactive historical edits are never consulted.

    Overlapping identity swaps use parity: each covering ``swap_identities`` or
    ``swap_node`` toggles the manual identity mark so a double-swap that restores
    the prior identity is not labeled corrected by the manual stack alone.
    """

    records = _load_manual_correction_records(manual_corrections_path)
    blanks: list[ManualBlankInterval] = []
    swap_intervals: list[ManualIdentitySwapInterval] = []
    swap_nodes: list[ManualIdentitySwapNode] = []

    for item in records:
        action = str(item.get("action") or "")
        if action == MANUAL_ACTION_BLANK_NODE:
            try:
                start = int(item["start_frame"])
                end = int(item.get("end_frame", start))
                track = int(item["track"])
                node = str(item["node"]).strip()
            except (KeyError, TypeError, ValueError) as exc:
                raise PoseFinalizationError(
                    f"Malformed blank_node record: {item!r}"
                ) from exc
            if end < start:
                raise PoseFinalizationError(
                    f"blank_node end_frame {end} is before start_frame {start}."
                )
            if not node:
                raise PoseFinalizationError("blank_node record is missing node name.")
            blanks.append(
                ManualBlankInterval(
                    start_frame=start,
                    end_frame=end,
                    track=track,
                    node=node,
                )
            )
        elif action == MANUAL_ACTION_SWAP_IDENTITIES:
            try:
                start = int(item["start_frame"])
                end = int(item.get("end_frame", start))
            except (KeyError, TypeError, ValueError) as exc:
                raise PoseFinalizationError(
                    f"Malformed swap_identities record: {item!r}"
                ) from exc
            if end < start:
                raise PoseFinalizationError(
                    f"swap_identities end_frame {end} is before start_frame {start}."
                )
            swap_intervals.append(
                ManualIdentitySwapInterval(start_frame=start, end_frame=end)
            )
        elif action == MANUAL_ACTION_SWAP_NODE:
            try:
                frame_idx = int(item.get("start_frame", item.get("frame_idx")))
                node = str(item["node"]).strip()
            except (KeyError, TypeError, ValueError) as exc:
                raise PoseFinalizationError(
                    f"Malformed swap_node record: {item!r}"
                ) from exc
            if not node:
                raise PoseFinalizationError("swap_node record is missing node name.")
            if frame_idx < 0:
                raise PoseFinalizationError(
                    f"swap_node frame_idx {frame_idx} must be non-negative."
                )
            swap_nodes.append(ManualIdentitySwapNode(frame_idx=frame_idx, node=node))

    return ManualCorrectionsEffects(
        blanks=tuple(blanks),
        identity_swap_intervals=tuple(swap_intervals),
        identity_swap_nodes=tuple(swap_nodes),
    )


def load_manual_blank_intervals(
    manual_corrections_path: Path | str | None,
) -> tuple[ManualBlankInterval, ...]:
    """Load explicit manual ``blank_node`` intervals (read-only).

    Only ``action == blank_node`` records are protected. Automatic blanking
    reflected solely in ``was_node_blanked`` is intentionally ignored.
    """

    return load_manual_corrections_effects(manual_corrections_path).blanks


def _load_manual_correction_records(
    manual_corrections_path: Path | str | None,
) -> list[Mapping[str, object]]:
    if manual_corrections_path is None:
        return []
    path = Path(manual_corrections_path).expanduser().resolve()
    if not path.is_file():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PoseFinalizationError(
            f"manual_corrections.json is unreadable: {path}: {exc}"
        ) from exc
    if not isinstance(payload, list):
        raise PoseFinalizationError(
            f"manual_corrections.json must be a list: {path}"
        )
    records: list[Mapping[str, object]] = []
    for item in payload:
        if isinstance(item, Mapping):
            records.append(item)
    return records


def resolve_manual_corrections_path(
    tracked_pose_path: Path | str,
    manual_corrections_path: Path | str | None = None,
) -> Path | None:
    """Resolve ``manual_corrections.json`` beside the accepted tracked pose."""

    if manual_corrections_path is not None:
        path = Path(manual_corrections_path).expanduser().resolve()
        return path if path.is_file() else path
    sibling = Path(tracked_pose_path).expanduser().resolve().parent / MANUAL_CORRECTIONS_FILENAME
    return sibling if sibling.is_file() else None


def build_source_protected_mask(
    *,
    n_source_frames: int,
    track_ids: Sequence[int],
    node_names: Sequence[str],
    blanks: Sequence[ManualBlankInterval],
) -> np.ndarray:
    """Boolean mask on the prepared-frame domain: (frame, track, node)."""

    track_index = {int(t): i for i, t in enumerate(track_ids)}
    node_index = {str(n): i for i, n in enumerate(node_names)}
    mask = np.zeros(
        (n_source_frames, len(track_ids), len(node_names)),
        dtype=bool,
    )
    for blank in blanks:
        track_i = track_index.get(int(blank.track))
        node_i = node_index.get(str(blank.node))
        if track_i is None or node_i is None:
            continue
        start = max(0, int(blank.start_frame))
        end = min(n_source_frames - 1, int(blank.end_frame))
        if end < start or start >= n_source_frames:
            continue
        mask[start : end + 1, track_i, node_i] = True
    return mask


def build_source_manual_identity_mask(
    *,
    n_source_frames: int,
    track_ids: Sequence[int],
    node_names: Sequence[str],
    effects: ManualCorrectionsEffects,
) -> np.ndarray:
    """Parity mask of active manual identity swaps on the prepared-frame domain.

    ``swap_identities`` toggles all tracks/nodes in ``[start_frame, end_frame]``.
    ``swap_node`` toggles both tracks for that node at the given frame.
    An even number of covering toggles leaves the manual mark clear.
    """

    counts = np.zeros(
        (n_source_frames, len(track_ids), len(node_names)),
        dtype=np.int32,
    )
    node_index = {str(n): i for i, n in enumerate(node_names)}

    for interval in effects.identity_swap_intervals:
        start = max(0, int(interval.start_frame))
        end = min(n_source_frames - 1, int(interval.end_frame))
        if end < start or start >= n_source_frames:
            continue
        counts[start : end + 1, :, :] += 1

    for swap in effects.identity_swap_nodes:
        node_i = node_index.get(str(swap.node))
        frame_i = int(swap.frame_idx)
        if node_i is None or frame_i < 0 or frame_i >= n_source_frames:
            continue
        counts[frame_i, :, node_i] += 1

    return (counts % 2) == 1


def construct_dense_pose(
    tracked_pose: pd.DataFrame | Path | str,
    timing_map: TimingMap,
    *,
    manual_corrections_path: Path | str | None = None,
    gap_fill_config: GapFillConfig | None = None,
    apply_gap_fill: bool = True,
) -> DensePoseResult:
    """Build dense 120 Hz pose from accepted S3 rows and a Sprint 1 timing map.

    Does not mutate the tracked pose file, manual corrections, prepared video,
    or other upstream artifacts.
    """

    cfg = gap_fill_config if gap_fill_config is not None else GapFillConfig()
    tracked_path: Path | None
    if isinstance(tracked_pose, (str, Path)):
        tracked_path = Path(tracked_pose).expanduser().resolve()
        table = load_tracked_pose_table(tracked_path)
        corrections_path = resolve_manual_corrections_path(
            tracked_path, manual_corrections_path
        )
    else:
        tracked_path = None
        table = tracked_pose
        corrections_path = (
            Path(manual_corrections_path).expanduser().resolve()
            if manual_corrections_path is not None
            else None
        )

    effects = load_manual_corrections_effects(corrections_path)
    blanks = effects.blanks
    frame_col = _frame_column(table)
    track_ids = _unique_tracks(table)
    node_names = _unique_nodes(table)
    n_src = int(timing_map.source_frame_count)
    n_tracks = len(track_ids)
    n_nodes = len(node_names)

    source = _pivot_source_arrays(
        table,
        frame_col=frame_col,
        track_ids=track_ids,
        node_names=node_names,
        n_source_frames=n_src,
    )
    protect_src = build_source_protected_mask(
        n_source_frames=n_src,
        track_ids=track_ids,
        node_names=node_names,
        blanks=blanks,
    )
    # Automatic (was_track_swapped) OR effective active manual identity swaps.
    identity_src = np.logical_or(
        source.identity_corrected,
        build_source_manual_identity_mask(
            n_source_frames=n_src,
            track_ids=track_ids,
            node_names=node_names,
            effects=effects,
        ),
    )

    n_tgt = timing_map.target_sample_count
    x = np.full((n_tgt, n_tracks, n_nodes), np.nan, dtype=np.float64)
    y = np.full((n_tgt, n_tracks, n_nodes), np.nan, dtype=np.float64)
    provenance = np.full((n_tgt, n_tracks, n_nodes), PROVENANCE_MISSING, dtype=np.uint8)
    protected = np.zeros((n_tgt, n_tracks, n_nodes), dtype=bool)
    point_score = np.full((n_tgt, n_tracks, n_nodes), np.nan, dtype=np.float64)
    instance_score = np.full((n_tgt, n_tracks, n_nodes), np.nan, dtype=np.float64)

    left_frames = np.asarray(timing_map.left_source_frame_idx, dtype=np.int64)
    right_frames = np.asarray(timing_map.right_source_frame_idx, dtype=np.int64)
    weights = np.asarray(timing_map.interpolation_weight, dtype=np.float64)
    exact = np.asarray(timing_map.is_exact_source_sample, dtype=bool)

    if left_frames.shape[0] != n_tgt:
        raise PoseFinalizationError("TimingMap sample count does not match target timeline.")

    for i in range(n_tgt):
        left = int(left_frames[i])
        right = int(right_frames[i])
        if left < 0 or right < 0 or left >= n_src or right >= n_src:
            raise PoseFinalizationError(
                f"Timing map source index out of range at target sample {i}."
            )
        w = float(weights[i])
        is_exact = bool(exact[i])

        for track_i in range(n_tracks):
            for node_i in range(n_nodes):
                left_prot = bool(protect_src[left, track_i, node_i])
                right_prot = bool(protect_src[right, track_i, node_i])
                # Conservative propagation: any contributing source blank protects
                # the target sample. Exact samples have left == right.
                if left_prot or right_prot:
                    protected[i, track_i, node_i] = True
                    provenance[i, track_i, node_i] = PROVENANCE_PROTECTED_MANUAL_BLANK
                    continue

                if is_exact:
                    if source.valid[left, track_i, node_i]:
                        x[i, track_i, node_i] = source.x[left, track_i, node_i]
                        y[i, track_i, node_i] = source.y[left, track_i, node_i]
                        point_score[i, track_i, node_i] = source.point_score[
                            left, track_i, node_i
                        ]
                        instance_score[i, track_i, node_i] = source.instance_score[
                            left, track_i, node_i
                        ]
                        if identity_src[left, track_i, node_i]:
                            provenance[i, track_i, node_i] = PROVENANCE_IDENTITY_CORRECTED
                        else:
                            provenance[i, track_i, node_i] = PROVENANCE_ORIGINAL
                    else:
                        provenance[i, track_i, node_i] = PROVENANCE_MISSING
                    continue

                # Temporal derivation requires two valid, unprotected endpoints.
                # Missing observations are never treated as interpolation anchors.
                if (
                    source.valid[left, track_i, node_i]
                    and source.valid[right, track_i, node_i]
                ):
                    x[i, track_i, node_i] = (1.0 - w) * source.x[
                        left, track_i, node_i
                    ] + w * source.x[right, track_i, node_i]
                    y[i, track_i, node_i] = (1.0 - w) * source.y[
                        left, track_i, node_i
                    ] + w * source.y[right, track_i, node_i]
                    provenance[i, track_i, node_i] = PROVENANCE_TEMPORAL_RESAMPLE
                    # Do not fabricate SLEAP confidence for derived samples.
                else:
                    provenance[i, track_i, node_i] = PROVENANCE_MISSING

    x_pre = np.array(x, dtype=np.float64, copy=True)
    y_pre = np.array(y, dtype=np.float64, copy=True)
    prov = provenance

    if apply_gap_fill:
        x, y, prov = fill_coordinate_gaps(
            x=x,
            y=y,
            provenance=provenance,
            protected=protected,
            node_names=np.asarray(node_names, dtype=object),
            config=cfg,
        )

    for array in (
        x,
        y,
        x_pre,
        y_pre,
        prov,
        protected,
        point_score,
        instance_score,
    ):
        array.setflags(write=False)

    track_arr = np.asarray(track_ids, dtype=np.int64)
    node_arr = np.asarray(node_names, dtype=object)
    track_arr.setflags(write=False)
    node_arr.setflags(write=False)

    return DensePoseResult(
        timing_map=timing_map,
        track_ids=track_arr,
        node_names=node_arr,
        x=x,
        y=y,
        x_pre_impute=x_pre,
        y_pre_impute=y_pre,
        provenance=prov,
        protected=protected,
        point_score=point_score,
        instance_score=instance_score,
        left_source_frame_idx=left_frames,
        right_source_frame_idx=right_frames,
        interpolation_weight=weights,
        is_exact_source_sample=exact,
        gap_fill_config=cfg,
        tracked_pose_path=tracked_path,
        manual_corrections_path=corrections_path,
        manual_blank_intervals=blanks,
        manual_identity_swap_intervals=effects.identity_swap_intervals,
        manual_identity_swap_nodes=effects.identity_swap_nodes,
    )


@dataclass(frozen=True, slots=True)
class _SourceArrays:
    x: np.ndarray
    y: np.ndarray
    point_score: np.ndarray
    instance_score: np.ndarray
    valid: np.ndarray
    identity_corrected: np.ndarray


def _pivot_source_arrays(
    table: pd.DataFrame,
    *,
    frame_col: str,
    track_ids: Sequence[int],
    node_names: Sequence[str],
    n_source_frames: int,
) -> _SourceArrays:
    n_tracks = len(track_ids)
    n_nodes = len(node_names)
    shape = (n_source_frames, n_tracks, n_nodes)
    x = np.full(shape, np.nan, dtype=np.float64)
    y = np.full(shape, np.nan, dtype=np.float64)
    point_score = np.full(shape, np.nan, dtype=np.float64)
    instance_score = np.full(shape, np.nan, dtype=np.float64)
    identity_corrected = np.zeros(shape, dtype=bool)

    track_index = {int(t): i for i, t in enumerate(track_ids)}
    node_index = {str(n): i for i, n in enumerate(node_names)}

    frames = pd.to_numeric(table[frame_col], errors="coerce")
    tracks = table["track"].map(_parse_track_index)
    nodes = table["node"].astype(str)
    xs = pd.to_numeric(table["x"], errors="coerce").to_numpy(dtype=np.float64)
    ys = pd.to_numeric(table["y"], errors="coerce").to_numpy(dtype=np.float64)

    if "point_score" in table.columns:
        ps = pd.to_numeric(table["point_score"], errors="coerce").to_numpy(dtype=np.float64)
    elif "score" in table.columns:
        ps = pd.to_numeric(table["score"], errors="coerce").to_numpy(dtype=np.float64)
    else:
        ps = np.full(len(table), np.nan, dtype=np.float64)

    if "instance_score" in table.columns:
        inst = pd.to_numeric(table["instance_score"], errors="coerce").to_numpy(
            dtype=np.float64
        )
    else:
        inst = np.full(len(table), np.nan, dtype=np.float64)

    if "was_track_swapped" in table.columns:
        swapped = pd.to_numeric(table["was_track_swapped"], errors="coerce").fillna(0)
        swapped_arr = swapped.to_numpy(dtype=np.float64) > 0
    else:
        swapped_arr = np.zeros(len(table), dtype=bool)

    for row_i in range(len(table)):
        frame_val = frames.iloc[row_i]
        track_val = tracks.iloc[row_i]
        if pd.isna(frame_val) or track_val is None or pd.isna(track_val):
            continue
        frame_i = int(frame_val)
        if frame_i < 0 or frame_i >= n_source_frames:
            continue
        track_i = track_index.get(int(track_val))
        node_i = node_index.get(str(nodes.iloc[row_i]))
        if track_i is None or node_i is None:
            continue
        x[frame_i, track_i, node_i] = xs[row_i]
        y[frame_i, track_i, node_i] = ys[row_i]
        point_score[frame_i, track_i, node_i] = ps[row_i]
        instance_score[frame_i, track_i, node_i] = inst[row_i]
        identity_corrected[frame_i, track_i, node_i] = bool(swapped_arr[row_i])

    valid = np.isfinite(x) & np.isfinite(y)
    return _SourceArrays(
        x=x,
        y=y,
        point_score=point_score,
        instance_score=instance_score,
        valid=valid,
        identity_corrected=identity_corrected,
    )


def _frame_column(table: pd.DataFrame) -> str:
    for name in _FRAME_CANDIDATES:
        if name in table.columns:
            return name
    raise PoseFinalizationError(
        "tracked_pose.parquet is missing prepared_frame_idx/frame_idx/frame."
    )


def _unique_tracks(table: pd.DataFrame) -> list[int]:
    parsed = table["track"].map(_parse_track_index)
    values = sorted({int(v) for v in parsed.tolist() if v is not None and not pd.isna(v)})
    if not values:
        raise PoseFinalizationError("tracked_pose.parquet has no parsable track labels.")
    return values


def _unique_nodes(table: pd.DataFrame) -> list[str]:
    names: list[str] = []
    seen: set[str] = set()
    for value in table["node"].tolist():
        if value is None:
            continue
        text = str(value).strip()
        if not text or text.lower() == "nan" or text in seen:
            continue
        seen.add(text)
        names.append(text)
    if not names:
        raise PoseFinalizationError("tracked_pose.parquet has no node names.")
    # Stable alphabetical order for deterministic dense axes while preserving names.
    return sorted(names)


def _parse_track_index(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not np.isfinite(value) or not float(value).is_integer():
            return None
        return int(value)
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
        return int(text)
    lowered = text.lower()
    if lowered.startswith("track_"):
        suffix = text.split("_", maxsplit=1)[1]
        if suffix.isdigit():
            return int(suffix)
    return None


def _require_columns(table: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [name for name in columns if name not in table.columns]
    if missing:
        raise PoseFinalizationError(
            "tracked_pose.parquet is missing required columns: "
            + ", ".join(missing)
        )
