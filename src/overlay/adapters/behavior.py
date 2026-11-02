"""Read-only S8 automated and S10 reviewed behavioral state adapters."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from overlay.adapters.base import (
    SourceIdentity,
    UnavailableAdapter,
    behavior_source_kind_label,
    require_columns,
)
from overlay.context import OverlayObservation
from overlay.contracts import BehaviorSourceKind, LayerId, OverlayAdapterError
from overlay.timing import PreparedFrameTiming
from project.file_digest import sha256_file


def _optional_digest(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return sha256_file(path)


class DirectionalExclusiveAdapter:
    """Exclusive ethogram row for one direction at a resolved dense sample."""

    def __init__(
        self,
        parquet_path: Path | str | None,
        *,
        layer_id: LayerId,
        source_kind: BehaviorSourceKind,
        focal_id: int,
        partner_id: int,
        subsystem: str,
        run_id: str | None = None,
        run_dir: Path | str | None = None,
        label_column: str = "behavior_label",
        dense_sample_range: tuple[int, int] | None = None,
    ) -> None:
        if focal_id == partner_id:
            raise OverlayAdapterError("focal_id and partner_id must differ.")
        self._layer_id = layer_id
        self._source_kind = source_kind
        self._focal_id = int(focal_id)
        self._partner_id = int(partner_id)
        self._label_column = label_column
        self._dense_sample_range = dense_sample_range
        self._path = (
            Path(parquet_path).expanduser().resolve()
            if parquet_path is not None
            else None
        )
        self._run_dir = (
            Path(run_dir).expanduser().resolve() if run_dir is not None else None
        )
        self._by_sample: dict[int, pd.DataFrame] | None = None
        self._load_error: str | None = None
        if self._path is None or not self._path.is_file():
            self._load_error = (
                f"{behavior_source_kind_label(source_kind)} states unavailable"
                if self._path is None
                else f"States parquet missing: {self._path}"
            )
        else:
            try:
                frame = _read_states_window(
                    self._path,
                    columns={
                        "dense_frame_idx",
                        "focal_id",
                        "partner_id",
                        label_column,
                        "label_reconstruction",
                        "undefined_reason",
                        "hierarchy_residual_kind",
                    },
                    dense_sample_range=dense_sample_range,
                )
                require_columns(
                    frame.columns,
                    {
                        "dense_frame_idx",
                        "focal_id",
                        "partner_id",
                        label_column,
                    },
                    label=str(self._path),
                )
                self._index(frame)
            except Exception as exc:  # noqa: BLE001
                self._load_error = f"Failed to load behavioral states: {exc}"
                self._by_sample = None

        self._identity = SourceIdentity(
            subsystem=subsystem,
            run_id=run_id,
            run_dir=self._run_dir,
            artifact_path=self._path,
            digest_sha256=_optional_digest(self._path),
            extra={
                "behavior_source_kind": behavior_source_kind_label(source_kind),
                "focal_id": self._focal_id,
                "partner_id": self._partner_id,
                "channel": "exclusive",
                "dense_sample_range": (
                    None
                    if dense_sample_range is None
                    else [int(dense_sample_range[0]), int(dense_sample_range[1])]
                ),
            },
        )

    def _index(self, frame: pd.DataFrame) -> None:
        work = frame.copy()
        keys = pd.to_numeric(work["dense_frame_idx"], errors="coerce")
        work = work.assign(_sample_key=keys).dropna(subset=["_sample_key"])
        work["_sample_key"] = work["_sample_key"].astype(np.int64)
        mask = (work["focal_id"].astype(int) == self._focal_id) & (
            work["partner_id"].astype(int) == self._partner_id
        )
        work = work.loc[mask]
        self._by_sample = {
            int(k): g.drop(columns=["_sample_key"])
            for k, g in work.groupby("_sample_key", sort=False)
        }

    @property
    def layer_id(self) -> LayerId:
        return self._layer_id

    @property
    def source_identity(self) -> SourceIdentity:
        return self._identity

    @property
    def available(self) -> bool:
        return self._load_error is None and self._by_sample is not None

    def observe(
        self,
        timing: PreparedFrameTiming,
        *,
        prepared_frame_idx: int,
    ) -> OverlayObservation:
        _ = prepared_frame_idx
        if not self.available or self._by_sample is None:
            return OverlayObservation(
                layer_id=self._layer_id,
                available=False,
                reason=self._load_error or "Behavioral adapter unavailable.",
            )
        if not timing.available or timing.dense_sample_idx is None:
            return OverlayObservation(
                layer_id=self._layer_id,
                available=False,
                reason=(
                    timing.reason
                    or "No dense sample mapped; behavioral state unavailable."
                ),
            )
        group = self._by_sample.get(int(timing.dense_sample_idx))
        if group is None or group.empty:
            return OverlayObservation(
                layer_id=self._layer_id,
                available=False,
                reason=(
                    f"Missing directional row focal={self._focal_id}→"
                    f"{self._partner_id} at dense_sample_idx="
                    f"{timing.dense_sample_idx}."
                ),
            )
        row = group.iloc[0]
        label = row[self._label_column]
        if label is None or (isinstance(label, float) and np.isnan(label)):
            return OverlayObservation(
                layer_id=self._layer_id,
                available=False,
                reason="behavior_label is null/invalid; refusing to invent a label.",
                payload={
                    "behavior_source_kind": behavior_source_kind_label(
                        self._source_kind
                    ),
                    "focal_id": self._focal_id,
                    "partner_id": self._partner_id,
                    "dense_sample_idx": int(timing.dense_sample_idx),
                },
            )
        payload: dict[str, Any] = {
            "behavior_source_kind": behavior_source_kind_label(self._source_kind),
            "channel": "exclusive",
            "focal_id": self._focal_id,
            "partner_id": self._partner_id,
            "behavior_label": str(label),
            "dense_sample_idx": int(timing.dense_sample_idx),
        }
        for optional in (
            "label_reconstruction",
            "undefined_reason",
            "hierarchy_residual_kind",
        ):
            if optional in group.columns:
                payload[optional] = _stringify_optional(row[optional])
        return OverlayObservation(
            layer_id=self._layer_id,
            available=True,
            reason=None,
            payload=payload,
        )


class ParallelChannelAdapter:
    """Contact or SBS parallel channel (not exclusive ethogram)."""

    def __init__(
        self,
        parquet_path: Path | str | None,
        *,
        layer_id: LayerId,
        source_kind: BehaviorSourceKind,
        channel: str,
        pred_column: str,
        subsystem: str,
        run_id: str | None = None,
        run_dir: Path | str | None = None,
        dense_sample_range: tuple[int, int] | None = None,
    ) -> None:
        self._layer_id = layer_id
        self._source_kind = source_kind
        self._channel = channel
        self._pred_column = pred_column
        self._dense_sample_range = dense_sample_range
        self._path = (
            Path(parquet_path).expanduser().resolve()
            if parquet_path is not None
            else None
        )
        self._run_dir = (
            Path(run_dir).expanduser().resolve() if run_dir is not None else None
        )
        self._by_sample: dict[int, Any] | None = None
        self._load_error: str | None = None
        if self._path is None or not self._path.is_file():
            self._load_error = (
                f"{channel} channel unavailable"
                if self._path is None
                else f"States parquet missing: {self._path}"
            )
        else:
            try:
                frame = _read_states_window(
                    self._path,
                    columns={"dense_frame_idx", pred_column, "focal_id", "partner_id"},
                    dense_sample_range=dense_sample_range,
                )
                require_columns(
                    frame.columns,
                    {"dense_frame_idx", pred_column},
                    label=str(self._path),
                )
                self._index(frame)
            except Exception as exc:  # noqa: BLE001
                self._load_error = f"Failed to load {channel} channel: {exc}"
                self._by_sample = None

        self._identity = SourceIdentity(
            subsystem=subsystem,
            run_id=run_id,
            run_dir=self._run_dir,
            artifact_path=self._path,
            digest_sha256=_optional_digest(self._path),
            extra={
                "behavior_source_kind": behavior_source_kind_label(source_kind),
                "channel": channel,
                "dense_sample_range": (
                    None
                    if dense_sample_range is None
                    else [int(dense_sample_range[0]), int(dense_sample_range[1])]
                ),
            },
        )

    def _index(self, frame: pd.DataFrame) -> None:
        # Parallel channels are pair-symmetric; take first row per dense sample.
        work = frame.copy()
        keys = pd.to_numeric(work["dense_frame_idx"], errors="coerce")
        work = work.assign(_sample_key=keys).dropna(subset=["_sample_key"])
        work["_sample_key"] = work["_sample_key"].astype(np.int64)
        # Prefer a stable direction if present.
        if {"focal_id", "partner_id"}.issubset(work.columns):
            work = work.sort_values(
                ["_sample_key", "focal_id", "partner_id"], kind="mergesort"
            )
        self._by_sample = {
            int(k): g.iloc[0]
            for k, g in work.groupby("_sample_key", sort=False)
        }

    @property
    def layer_id(self) -> LayerId:
        return self._layer_id

    @property
    def source_identity(self) -> SourceIdentity:
        return self._identity

    @property
    def available(self) -> bool:
        return self._load_error is None and self._by_sample is not None

    def observe(
        self,
        timing: PreparedFrameTiming,
        *,
        prepared_frame_idx: int,
    ) -> OverlayObservation:
        _ = prepared_frame_idx
        if not self.available or self._by_sample is None:
            return OverlayObservation(
                layer_id=self._layer_id,
                available=False,
                reason=self._load_error or f"{self._channel} adapter unavailable.",
            )
        if not timing.available or timing.dense_sample_idx is None:
            return OverlayObservation(
                layer_id=self._layer_id,
                available=False,
                reason=(
                    timing.reason
                    or "No dense sample mapped; parallel channel unavailable."
                ),
            )
        row = self._by_sample.get(int(timing.dense_sample_idx))
        if row is None:
            return OverlayObservation(
                layer_id=self._layer_id,
                available=False,
                reason=(
                    f"Missing {self._channel} row at dense_sample_idx="
                    f"{timing.dense_sample_idx}."
                ),
            )
        pred_raw = row[self._pred_column]
        try:
            pred = int(pred_raw)
        except (TypeError, ValueError):
            return OverlayObservation(
                layer_id=self._layer_id,
                available=False,
                reason=f"Invalid {self._pred_column} value; refusing to invent.",
            )
        return OverlayObservation(
            layer_id=self._layer_id,
            available=True,
            reason=None,
            payload={
                "behavior_source_kind": behavior_source_kind_label(self._source_kind),
                "channel": self._channel,
                "pred": pred,
                "active": bool(pred),
                "dense_sample_idx": int(timing.dense_sample_idx),
            },
        )


def make_s8_exclusive_a_to_b(
    parquet_path: Path | str | None,
    *,
    run_id: str | None = None,
    run_dir: Path | str | None = None,
    focal_id: int = 0,
    partner_id: int = 1,
    dense_sample_range: tuple[int, int] | None = None,
) -> DirectionalExclusiveAdapter:
    return DirectionalExclusiveAdapter(
        parquet_path,
        layer_id=LayerId.S8_EXCLUSIVE_A_TO_B,
        source_kind=BehaviorSourceKind.S8_AUTOMATED,
        focal_id=focal_id,
        partner_id=partner_id,
        subsystem="behavioral_arbitration",
        run_id=run_id,
        run_dir=run_dir,
        dense_sample_range=dense_sample_range,
    )


def make_s8_exclusive_b_to_a(
    parquet_path: Path | str | None,
    *,
    run_id: str | None = None,
    run_dir: Path | str | None = None,
    focal_id: int = 1,
    partner_id: int = 0,
    dense_sample_range: tuple[int, int] | None = None,
) -> DirectionalExclusiveAdapter:
    return DirectionalExclusiveAdapter(
        parquet_path,
        layer_id=LayerId.S8_EXCLUSIVE_B_TO_A,
        source_kind=BehaviorSourceKind.S8_AUTOMATED,
        focal_id=focal_id,
        partner_id=partner_id,
        subsystem="behavioral_arbitration",
        run_id=run_id,
        run_dir=run_dir,
        dense_sample_range=dense_sample_range,
    )


def make_s8_contact(
    parquet_path: Path | str | None,
    *,
    run_id: str | None = None,
    run_dir: Path | str | None = None,
    dense_sample_range: tuple[int, int] | None = None,
) -> ParallelChannelAdapter:
    return ParallelChannelAdapter(
        parquet_path,
        layer_id=LayerId.S8_CONTACT,
        source_kind=BehaviorSourceKind.S8_AUTOMATED,
        channel="contact",
        pred_column="pred_contact",
        subsystem="behavioral_arbitration",
        run_id=run_id,
        run_dir=run_dir,
        dense_sample_range=dense_sample_range,
    )


def make_s8_sbs(
    parquet_path: Path | str | None,
    *,
    run_id: str | None = None,
    run_dir: Path | str | None = None,
    dense_sample_range: tuple[int, int] | None = None,
) -> ParallelChannelAdapter:
    return ParallelChannelAdapter(
        parquet_path,
        layer_id=LayerId.S8_SBS,
        source_kind=BehaviorSourceKind.S8_AUTOMATED,
        channel="side_by_side",
        pred_column="pred_side_by_side",
        subsystem="behavioral_arbitration",
        run_id=run_id,
        run_dir=run_dir,
        dense_sample_range=dense_sample_range,
    )


def make_s10_reviewed_exclusive_a_to_b(
    parquet_path: Path | str | None,
    *,
    run_id: str | None = None,
    run_dir: Path | str | None = None,
    focal_id: int = 0,
    partner_id: int = 1,
    label_column: str = "behavior_label",
    dense_sample_range: tuple[int, int] | None = None,
) -> DirectionalExclusiveAdapter:
    """Published S10 reviewed exclusive states — identity distinct from S8."""

    return DirectionalExclusiveAdapter(
        parquet_path,
        layer_id=LayerId.S8_EXCLUSIVE_A_TO_B,
        source_kind=BehaviorSourceKind.S10_REVIEWED,
        focal_id=focal_id,
        partner_id=partner_id,
        subsystem="behavioral_review",
        run_id=run_id,
        run_dir=run_dir,
        label_column=label_column,
        dense_sample_range=dense_sample_range,
    )


def make_s10_reviewed_exclusive_b_to_a(
    parquet_path: Path | str | None,
    *,
    run_id: str | None = None,
    run_dir: Path | str | None = None,
    focal_id: int = 1,
    partner_id: int = 0,
    label_column: str = "behavior_label",
    dense_sample_range: tuple[int, int] | None = None,
) -> DirectionalExclusiveAdapter:
    """Published S10 reviewed exclusive states — B→A direction."""

    return DirectionalExclusiveAdapter(
        parquet_path,
        layer_id=LayerId.S8_EXCLUSIVE_B_TO_A,
        source_kind=BehaviorSourceKind.S10_REVIEWED,
        focal_id=focal_id,
        partner_id=partner_id,
        subsystem="behavioral_review",
        run_id=run_id,
        run_dir=run_dir,
        label_column=label_column,
        dense_sample_range=dense_sample_range,
    )


def make_s10_reviewed_contact(
    parquet_path: Path | str | None,
    *,
    run_id: str | None = None,
    run_dir: Path | str | None = None,
    dense_sample_range: tuple[int, int] | None = None,
) -> ParallelChannelAdapter:
    return ParallelChannelAdapter(
        parquet_path,
        layer_id=LayerId.S8_CONTACT,
        source_kind=BehaviorSourceKind.S10_REVIEWED,
        channel="contact",
        pred_column="pred_contact",
        subsystem="behavioral_review",
        run_id=run_id,
        run_dir=run_dir,
        dense_sample_range=dense_sample_range,
    )


def make_s10_reviewed_sbs(
    parquet_path: Path | str | None,
    *,
    run_id: str | None = None,
    run_dir: Path | str | None = None,
    dense_sample_range: tuple[int, int] | None = None,
) -> ParallelChannelAdapter:
    return ParallelChannelAdapter(
        parquet_path,
        layer_id=LayerId.S8_SBS,
        source_kind=BehaviorSourceKind.S10_REVIEWED,
        channel="side_by_side",
        pred_column="pred_side_by_side",
        subsystem="behavioral_review",
        run_id=run_id,
        run_dir=run_dir,
        dense_sample_range=dense_sample_range,
    )


def make_s5_segmentation_stub() -> UnavailableAdapter:
    """S5 contours deferred past C1 (no contour generation)."""

    return UnavailableAdapter(
        layer_id=LayerId.S5_SEGMENTATION,
        source_identity=SourceIdentity(
            subsystem="blob_features",
            run_id=None,
            extra={"deferred": "C3 optional"},
        ),
        reason="S5 segmentation overlay deferred past C1; no contour generation.",
    )


def _read_states_window(
    path: Path,
    *,
    columns: set[str],
    dense_sample_range: tuple[int, int] | None,
) -> pd.DataFrame:
    """Windowed parquet read for large behavioral tables when a dense range is known."""

    available = _list_parquet_columns(path)
    cols = [c for c in columns if c in available]
    if "dense_frame_idx" not in cols:
        cols = ["dense_frame_idx", *cols]
    if dense_sample_range is None:
        return pd.read_parquet(path, columns=cols)

    start, end = int(dense_sample_range[0]), int(dense_sample_range[1])
    if start > end:
        raise OverlayAdapterError(
            f"Invalid dense_sample_range start ({start}) > end ({end})."
        )
    try:
        frame = pd.read_parquet(
            path,
            columns=cols,
            filters=[
                ("dense_frame_idx", ">=", start),
                ("dense_frame_idx", "<=", end),
            ],
        )
    except (TypeError, ValueError, OSError):
        frame = pd.read_parquet(path, columns=cols)
        dens = pd.to_numeric(frame["dense_frame_idx"], errors="coerce")
        frame = frame.loc[(dens >= start) & (dens <= end)].copy()
    return frame


def _list_parquet_columns(path: Path) -> frozenset[str]:
    try:
        import pyarrow.parquet as pq  # type: ignore[import-untyped]

        return frozenset(str(n) for n in pq.ParquetFile(path).schema_arrow.names)
    except Exception:  # noqa: BLE001 - fall back to pandas schema probe
        probe = pd.read_parquet(path)
        return frozenset(str(c) for c in probe.columns)


def _stringify_optional(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and np.isnan(value):
        return None
    text = str(value)
    if text in {"", "nan", "<NA>", "None"}:
        return None
    return text
