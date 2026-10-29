"""Windowed read-only access to S8 framewise states via the S9 package pin.

Does not re-collapse bouts, re-arbitrate, or rewrite science artifacts.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from behavioral_review.contracts import (
    CHANNEL_CONTACT,
    CHANNEL_EXCLUSIVE,
    CHANNEL_SIDE_BY_SIDE,
    ETHOGRAM_ECHO_PROB_COLUMNS,
    ETHOGRAM_ECHO_VALIDITY_COLUMNS,
    ETHOGRAM_IDENTITY_COLUMNS,
    ETHOGRAM_LABEL_COLUMNS,
    ETHOGRAM_PARALLEL_COLUMNS,
    ETHOGRAM_WINDOW_SCHEMA_VERSION,
    EVIDENCE_INVALID,
    EVIDENCE_MISSING,
    EVIDENCE_UNOBSERVED,
    EVIDENCE_VALID_NEGATIVE,
    EVIDENCE_VALID_POSITIVE,
    LABEL_ENGAGED_VALUE,
    LABEL_UNDEFINED_VALUE,
    InspectionError,
)
from behavioral_review.sync import SyncMap

_SCORE_ORIGIN_ROUTING = "routing_sentinel"


@dataclass(frozen=True, slots=True)
class EthogramLaneMeta:
    """Lane identity for GUI timeline chrome (science rows stay directional)."""

    lane_id: str
    channel: str
    identity_grain: str
    focal_id: int | None
    partner_id: int | None
    mouse_a_id: int | None
    mouse_b_id: int | None
    display_name: str


@dataclass(frozen=True, slots=True)
class DirectionEthogramTrack:
    """One exclusive directional track over a dense sample window."""

    lane: EthogramLaneMeta
    sample_idx: np.ndarray
    dense_frame_idx: np.ndarray
    time_sec: np.ndarray
    behavior_label: np.ndarray
    label_reconstruction: np.ndarray
    undefined_reason: np.ndarray
    hierarchy_residual_kind: np.ndarray
    evidence_state: np.ndarray  # per-sample coarse validity badge for the winning label


@dataclass(frozen=True, slots=True)
class ParallelChannelTrack:
    """Independent contact or side-by-side track (symmetric_pair grain)."""

    lane: EthogramLaneMeta
    sample_idx: np.ndarray
    dense_frame_idx: np.ndarray
    time_sec: np.ndarray
    pred: np.ndarray
    prob: np.ndarray
    input_valid: np.ndarray
    score_origin: np.ndarray
    evidence_state: np.ndarray


@dataclass(frozen=True, slots=True)
class EthogramWindow:
    """Column-projected S8 states for ``[start_sample_idx, end_sample_idx]``."""

    schema_version: str
    start_sample_idx: int
    end_sample_idx: int
    n_dense_requested: int
    directions: tuple[DirectionEthogramTrack, ...]
    contact: ParallelChannelTrack | None
    side_by_side: ParallelChannelTrack | None
    states_available: bool
    columns_loaded: tuple[str, ...]
    warnings: tuple[str, ...] = ()
    raw_frame: pd.DataFrame | None = field(default=None, repr=False)

    @property
    def inclusive_span(self) -> int:
        return int(self.end_sample_idx - self.start_sample_idx + 1)


class EthogramAccessor:
    """Lazy windowed reader over pinned S8 ``behavior_states.parquet``."""

    def __init__(
        self,
        states_path: Path | str | None,
        *,
        sync: SyncMap,
        available: bool = True,
        cache_windows: int = 4,
    ) -> None:
        self._sync = sync
        self._path = (
            Path(states_path).expanduser().resolve() if states_path is not None else None
        )
        self._available = bool(available) and self._path is not None and self._path.is_file()
        self._cache_windows = max(0, int(cache_windows))
        self._cache: list[tuple[tuple[Any, ...], EthogramWindow]] = []
        self._parquet_columns: frozenset[str] | None = None

    @property
    def available(self) -> bool:
        return self._available

    @property
    def states_path(self) -> Path | None:
        return self._path

    def discover_directions(
        self,
        *,
        sample_idx: int | None = None,
        half_window: int = 60,
    ) -> tuple[tuple[int, int], ...]:
        """Return ordered ``(focal_id, partner_id)`` pairs present in a probe window."""

        if not self._available:
            return ()
        center = 0 if sample_idx is None else int(sample_idx)
        start = max(0, center - int(half_window))
        end = min(self._sync.n_dense - 1, center + int(half_window))
        cols = ["dense_frame_idx", "focal_id", "partner_id"]
        frame = self._read_window_frame(start, end, columns=cols)
        if frame.empty:
            return ()
        pairs = (
            frame[["focal_id", "partner_id"]]
            .drop_duplicates()
            .sort_values(["focal_id", "partner_id"], kind="mergesort")
        )
        return tuple(
            (int(r.focal_id), int(r.partner_id)) for r in pairs.itertuples(index=False)
        )

    def read_window(
        self,
        start_sample_idx: int,
        end_sample_idx: int,
        *,
        directions: Sequence[tuple[int, int]] | None = None,
        include_echo_probs: bool = True,
        include_validity: bool = True,
        keep_raw: bool = False,
    ) -> EthogramWindow:
        """Load a dense-inclusive window without reading unused S8 columns."""

        start, end = _normalize_inclusive_window(
            start_sample_idx, end_sample_idx, n_dense=self._sync.n_dense
        )
        cache_key = (
            start,
            end,
            tuple(directions) if directions is not None else None,
            include_echo_probs,
            include_validity,
            keep_raw,
        )
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        if not self._available:
            empty = EthogramWindow(
                schema_version=ETHOGRAM_WINDOW_SCHEMA_VERSION,
                start_sample_idx=start,
                end_sample_idx=end,
                n_dense_requested=end - start + 1,
                directions=(),
                contact=None,
                side_by_side=None,
                states_available=False,
                columns_loaded=(),
                warnings=("S8 behavior_states unavailable; framewise ethogram blocked.",),
            )
            self._cache_put(cache_key, empty)
            return empty

        columns = self._select_columns(
            include_echo_probs=include_echo_probs,
            include_validity=include_validity,
        )
        frame = self._read_window_frame(start, end, columns=columns)
        if directions is not None:
            wanted = {(int(a), int(b)) for a, b in directions}
            mask = [
                (int(f), int(p)) in wanted
                for f, p in zip(
                    frame["focal_id"].to_numpy(),
                    frame["partner_id"].to_numpy(),
                    strict=False,
                )
            ]
            frame = frame.loc[mask].reset_index(drop=True)

        dir_tracks = self._build_direction_tracks(frame)
        contact = self._build_parallel_track(
            frame,
            channel=CHANNEL_CONTACT,
            pred_col="pred_contact",
            prob_col="prob_contact",
            valid_col="input_valid_contact",
            origin_col="score_origin_contact",
            arb_col="arbitration_input_valid_contact",
        )
        sbs = self._build_parallel_track(
            frame,
            channel=CHANNEL_SIDE_BY_SIDE,
            pred_col="pred_side_by_side",
            prob_col="prob_side_by_side",
            valid_col="input_valid_side_by_side",
            origin_col="score_origin_side_by_side",
            arb_col="arbitration_input_valid_side_by_side",
        )
        warnings: list[str] = []
        labels = (
            frame["behavior_label"].astype(str).to_numpy()
            if "behavior_label" in frame.columns and not frame.empty
            else np.array([], dtype=object)
        )
        if labels.size and (
            LABEL_ENGAGED_VALUE in labels and LABEL_UNDEFINED_VALUE in labels
        ):
            # Both present — fine; no warning.
            pass
        window = EthogramWindow(
            schema_version=ETHOGRAM_WINDOW_SCHEMA_VERSION,
            start_sample_idx=start,
            end_sample_idx=end,
            n_dense_requested=end - start + 1,
            directions=dir_tracks,
            contact=contact,
            side_by_side=sbs,
            states_available=True,
            columns_loaded=tuple(columns),
            warnings=tuple(warnings),
            raw_frame=frame if keep_raw else None,
        )
        self._cache_put(cache_key, window)
        return window

    def labels_at_sample(
        self,
        sample_idx: int,
        *,
        directions: Sequence[tuple[int, int]] | None = None,
    ) -> dict[tuple[int, int], str]:
        window = self.read_window(
            sample_idx,
            sample_idx,
            directions=directions,
            include_echo_probs=False,
            include_validity=False,
        )
        out: dict[tuple[int, int], str] = {}
        for track in window.directions:
            if track.sample_idx.size == 0:
                continue
            # Exact sample match (window may be empty for missing rows).
            hits = np.flatnonzero(track.sample_idx == int(sample_idx))
            if hits.size:
                key = (int(track.lane.focal_id or -1), int(track.lane.partner_id or -1))
                out[key] = str(track.behavior_label[int(hits[0])])
        return out

    def _select_columns(
        self,
        *,
        include_echo_probs: bool,
        include_validity: bool,
    ) -> list[str]:
        available = self._list_parquet_columns()
        wanted: list[str] = []
        for col in (
            *ETHOGRAM_IDENTITY_COLUMNS,
            *ETHOGRAM_LABEL_COLUMNS,
            *ETHOGRAM_PARALLEL_COLUMNS,
        ):
            if col in available and col not in wanted:
                wanted.append(col)
        if include_validity:
            for col in ETHOGRAM_ECHO_VALIDITY_COLUMNS:
                if col in available and col not in wanted:
                    wanted.append(col)
        if include_echo_probs:
            for col in ETHOGRAM_ECHO_PROB_COLUMNS:
                if col in available and col not in wanted:
                    wanted.append(col)
        # dense_frame_idx is required for window filters.
        if "dense_frame_idx" not in wanted:
            raise InspectionError(
                "S8 behavior_states missing dense_frame_idx; cannot window ethogram."
            )
        if "behavior_label" not in wanted:
            raise InspectionError(
                "S8 behavior_states missing behavior_label; cannot inspect ethogram."
            )
        return wanted

    def _list_parquet_columns(self) -> frozenset[str]:
        if self._parquet_columns is not None:
            return self._parquet_columns
        if self._path is None:
            return frozenset()
        try:
            import pyarrow.parquet as pq  # type: ignore[import-untyped]

            names = frozenset(str(n) for n in pq.ParquetFile(self._path).schema_arrow.names)
        except Exception:  # noqa: BLE001 - fall back to pandas schema probe
            probe = pd.read_parquet(self._path)
            names = frozenset(str(c) for c in probe.columns)
        self._parquet_columns = names
        return names

    def _read_window_frame(
        self,
        start: int,
        end: int,
        *,
        columns: Sequence[str],
    ) -> pd.DataFrame:
        assert self._path is not None
        cols = list(columns)
        try:
            frame = pd.read_parquet(
                self._path,
                columns=cols,
                filters=[
                    ("dense_frame_idx", ">=", int(start)),
                    ("dense_frame_idx", "<=", int(end)),
                ],
            )
        except (TypeError, ValueError, OSError):
            # Older engines / single-row-group files: project columns then filter.
            frame = pd.read_parquet(self._path, columns=cols)
            dens = frame["dense_frame_idx"].astype("int64")
            frame = frame.loc[(dens >= start) & (dens <= end)]
        if frame.empty:
            return frame.reset_index(drop=True)
        # Align sample_idx when only dense_frame_idx present.
        if "sample_idx" not in frame.columns:
            frame = frame.copy()
            frame["sample_idx"] = frame["dense_frame_idx"].astype("int64")
        if "time_sec" not in frame.columns:
            frame = frame.copy()
            dens = frame["dense_frame_idx"].astype("int64").to_numpy()
            times = np.full(dens.shape[0], np.nan, dtype=np.float64)
            valid = (dens >= 0) & (dens < self._sync.n_dense)
            times[valid] = self._sync.time_sec[dens[valid]]
            frame["time_sec"] = times
        return frame.sort_values(
            ["dense_frame_idx", "focal_id", "partner_id"], kind="mergesort"
        ).reset_index(drop=True)

    def _build_direction_tracks(
        self, frame: pd.DataFrame
    ) -> tuple[DirectionEthogramTrack, ...]:
        if frame.empty:
            return ()
        tracks: list[DirectionEthogramTrack] = []
        for (focal, partner), block in frame.groupby(
            ["focal_id", "partner_id"], sort=True
        ):
            focal_i = int(focal)
            partner_i = int(partner)
            lane = EthogramLaneMeta(
                lane_id=f"exclusive:{focal_i}->{partner_i}",
                channel=CHANNEL_EXCLUSIVE,
                identity_grain="directional",
                focal_id=focal_i,
                partner_id=partner_i,
                mouse_a_id=min(focal_i, partner_i),
                mouse_b_id=max(focal_i, partner_i),
                display_name=f"Exclusive {focal_i}→{partner_i}",
            )
            labels = block["behavior_label"].astype(str).to_numpy()
            recon = (
                block["label_reconstruction"].fillna("").astype(str).to_numpy()
                if "label_reconstruction" in block.columns
                else np.full(len(block), "", dtype=object)
            )
            undef = (
                block["undefined_reason"].fillna("").astype(str).to_numpy()
                if "undefined_reason" in block.columns
                else np.full(len(block), "", dtype=object)
            )
            residual = (
                block["hierarchy_residual_kind"].fillna("").astype(str).to_numpy()
                if "hierarchy_residual_kind" in block.columns
                else np.full(len(block), "", dtype=object)
            )
            evidence = np.array(
                [_exclusive_evidence_state(lab, u) for lab, u in zip(labels, undef, strict=True)],
                dtype=object,
            )
            tracks.append(
                DirectionEthogramTrack(
                    lane=lane,
                    sample_idx=block["sample_idx"].to_numpy(dtype=np.int64),
                    dense_frame_idx=block["dense_frame_idx"].to_numpy(dtype=np.int64),
                    time_sec=block["time_sec"].to_numpy(dtype=np.float64),
                    behavior_label=labels,
                    label_reconstruction=recon,
                    undefined_reason=undef,
                    hierarchy_residual_kind=residual,
                    evidence_state=evidence,
                )
            )
        return tuple(tracks)

    def _build_parallel_track(
        self,
        frame: pd.DataFrame,
        *,
        channel: str,
        pred_col: str,
        prob_col: str,
        valid_col: str,
        origin_col: str,
        arb_col: str,
    ) -> ParallelChannelTrack | None:
        if frame.empty or pred_col not in frame.columns:
            return None
        # Deduplicate directional rows → one undirected sample row.
        work = frame.sort_values(
            ["dense_frame_idx", "focal_id", "partner_id"], kind="mergesort"
        )
        undirected = work.drop_duplicates(subset=["dense_frame_idx"], keep="first")
        mouse_a = (
            int(undirected["mouse_a"].iloc[0])
            if "mouse_a" in undirected.columns and len(undirected)
            else None
        )
        mouse_b = (
            int(undirected["mouse_b"].iloc[0])
            if "mouse_b" in undirected.columns and len(undirected)
            else None
        )
        if (mouse_a is None or mouse_b is None) and len(undirected):
            # Infer from first directional pair.
            f0 = int(undirected["focal_id"].iloc[0])
            p0 = int(undirected["partner_id"].iloc[0])
            mouse_a, mouse_b = min(f0, p0), max(f0, p0)
        lane = EthogramLaneMeta(
            lane_id=f"parallel:{channel}",
            channel=channel,
            identity_grain="symmetric_pair",
            focal_id=None,
            partner_id=None,
            mouse_a_id=mouse_a,
            mouse_b_id=mouse_b,
            display_name=channel.replace("_", " ").title(),
        )
        pred = pd.to_numeric(undirected[pred_col], errors="coerce").fillna(0).to_numpy(
            dtype=np.int8
        )
        prob = (
            pd.to_numeric(undirected[prob_col], errors="coerce").to_numpy(dtype=np.float64)
            if prob_col in undirected.columns
            else np.full(len(undirected), np.nan, dtype=np.float64)
        )
        if arb_col in undirected.columns:
            valid = undirected[arb_col].fillna(False).astype(bool).to_numpy()
        elif valid_col in undirected.columns:
            valid = undirected[valid_col].fillna(True).astype(bool).to_numpy()
        else:
            valid = np.ones(len(undirected), dtype=bool)
        origin = (
            undirected[origin_col].fillna("").astype(str).to_numpy()
            if origin_col in undirected.columns
            else np.full(len(undirected), "", dtype=object)
        )
        evidence = np.array(
            [
                _parallel_evidence_state(bool(v), str(o), int(p), float(pr))
                for v, o, p, pr in zip(valid, origin, pred, prob, strict=True)
            ],
            dtype=object,
        )
        return ParallelChannelTrack(
            lane=lane,
            sample_idx=undirected["sample_idx"].to_numpy(dtype=np.int64),
            dense_frame_idx=undirected["dense_frame_idx"].to_numpy(dtype=np.int64),
            time_sec=undirected["time_sec"].to_numpy(dtype=np.float64),
            pred=pred,
            prob=prob,
            input_valid=valid,
            score_origin=origin,
            evidence_state=evidence,
        )

    def _cache_get(self, key: tuple[Any, ...]) -> EthogramWindow | None:
        for cached_key, value in self._cache:
            if cached_key == key:
                return value
        return None

    def _cache_put(self, key: tuple[Any, ...], value: EthogramWindow) -> None:
        if self._cache_windows <= 0:
            return
        self._cache = [(k, v) for k, v in self._cache if k != key]
        self._cache.append((key, value))
        while len(self._cache) > self._cache_windows:
            self._cache.pop(0)


def run_length_segments(
    sample_idx: np.ndarray,
    values: np.ndarray,
) -> list[tuple[int, int, Any]]:
    """Lossless contiguous segments ``(start, end_inclusive, value)``.

    Splits on value change **or** dense gap ≠ 1 (preserves S8 bout geometry).
    """

    if sample_idx.size == 0:
        return []
    if sample_idx.shape[0] != values.shape[0]:
        raise InspectionError("sample_idx and values length mismatch for segmentation.")
    segments: list[tuple[int, int, Any]] = []
    seg_start = int(sample_idx[0])
    seg_val = values[0]
    prev = int(sample_idx[0])
    for i in range(1, sample_idx.shape[0]):
        cur = int(sample_idx[i])
        val = values[i]
        if cur != prev + 1 or val != seg_val:
            segments.append((seg_start, prev, seg_val))
            seg_start = cur
            seg_val = val
        prev = cur
    segments.append((seg_start, prev, seg_val))
    return segments


def _exclusive_evidence_state(label: str, undefined_reason: str) -> str:
    if label == LABEL_UNDEFINED_VALUE:
        return EVIDENCE_UNOBSERVED if undefined_reason else EVIDENCE_MISSING
    if label == LABEL_ENGAGED_VALUE:
        return EVIDENCE_VALID_POSITIVE
    return EVIDENCE_VALID_POSITIVE


def _parallel_evidence_state(
    input_valid: bool,
    score_origin: str,
    pred: int,
    prob: float,
) -> str:
    if not input_valid or score_origin == _SCORE_ORIGIN_ROUTING:
        return EVIDENCE_INVALID
    if pred == 1:
        return EVIDENCE_VALID_POSITIVE
    if np.isfinite(prob) and float(prob) == 0.0 and score_origin == _SCORE_ORIGIN_ROUTING:
        return EVIDENCE_UNOBSERVED
    return EVIDENCE_VALID_NEGATIVE


def _normalize_inclusive_window(
    start_sample_idx: int,
    end_sample_idx: int,
    *,
    n_dense: int,
) -> tuple[int, int]:
    if n_dense <= 0:
        raise InspectionError("Cannot window ethogram over empty sync_index.")
    if isinstance(start_sample_idx, bool) or not isinstance(
        start_sample_idx, (int, np.integer)
    ):
        raise InspectionError(f"start_sample_idx must be int; got {start_sample_idx!r}.")
    if isinstance(end_sample_idx, bool) or not isinstance(
        end_sample_idx, (int, np.integer)
    ):
        raise InspectionError(f"end_sample_idx must be int; got {end_sample_idx!r}.")
    start = int(start_sample_idx)
    end = int(end_sample_idx)
    if start > end:
        raise InspectionError(f"Invalid ethogram window: start ({start}) > end ({end}).")
    if start < 0 or end >= n_dense:
        raise InspectionError(
            f"Ethogram window [{start}, {end}] outside dense range [0, {n_dense - 1}]."
        )
    return start, end
