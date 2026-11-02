"""Read-only S2 / S3 / S4 pose adapters for overlay evidence."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from overlay.adapters.base import SourceIdentity, require_columns
from overlay.context import OverlayObservation
from overlay.contracts import (
    S4_PROVENANCE_NAMES,
    S4_UNAVAILABLE_PROVENANCE,
    LayerId,
    OverlayAdapterError,
)
from overlay.timing import PreparedFrameTiming
from project.file_digest import sha256_file


def _optional_digest(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    return sha256_file(path)


def _finite_xy(x: object, y: object) -> tuple[float, float] | None:
    try:
        xf = float(x)  # type: ignore[arg-type]
        yf = float(y)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if not (np.isfinite(xf) and np.isfinite(yf)):
        return None
    return xf, yf


class PreparedPoseAdapter:
    """S2 inferred or S3 tracking-corrected pose (prepared-frame domain)."""

    def __init__(
        self,
        parquet_path: Path | str | None,
        *,
        layer_id: LayerId,
        subsystem: str,
        run_id: str | None = None,
        run_dir: Path | str | None = None,
        legend: str,
        require_track: bool = False,
    ) -> None:
        self._layer_id = layer_id
        self._legend = legend
        self._require_track = require_track
        self._path = (
            Path(parquet_path).expanduser().resolve()
            if parquet_path is not None
            else None
        )
        self._run_dir = (
            Path(run_dir).expanduser().resolve() if run_dir is not None else None
        )
        self._frame: pd.DataFrame | None = None
        self._by_prepared: dict[int, pd.DataFrame] | None = None
        self._load_error: str | None = None
        if self._path is None or not self._path.is_file():
            self._load_error = (
                f"{subsystem} pose parquet unavailable"
                if self._path is None
                else f"{subsystem} pose parquet missing: {self._path}"
            )
        else:
            try:
                self._frame = pd.read_parquet(self._path)
                required = {"frame_idx", "node", "x", "y"}
                if require_track:
                    required.add("track")
                require_columns(self._frame.columns, required, label=str(self._path))
                self._index_by_prepared()
            except Exception as exc:  # noqa: BLE001 - surface as unavailable
                self._load_error = f"Failed to load {subsystem} pose: {exc}"
                self._frame = None
                self._by_prepared = None

        self._identity = SourceIdentity(
            subsystem=subsystem,
            run_id=run_id,
            run_dir=self._run_dir,
            artifact_path=self._path,
            digest_sha256=_optional_digest(self._path),
            extra={"legend": legend, "domain": "prepared"},
        )

    def _index_by_prepared(self) -> None:
        assert self._frame is not None
        work = self._frame.copy()
        if "prepared_frame_idx" in work.columns:
            keys = pd.to_numeric(work["prepared_frame_idx"], errors="coerce")
        else:
            keys = pd.to_numeric(work["frame_idx"], errors="coerce")
        work = work.assign(_prepared_key=keys)
        work = work.dropna(subset=["_prepared_key"])
        work["_prepared_key"] = work["_prepared_key"].astype(np.int64)
        self._by_prepared = {
            int(k): g.drop(columns=["_prepared_key"])
            for k, g in work.groupby("_prepared_key", sort=False)
        }

    @property
    def layer_id(self) -> LayerId:
        return self._layer_id

    @property
    def source_identity(self) -> SourceIdentity:
        return self._identity

    @property
    def available(self) -> bool:
        return self._load_error is None and self._by_prepared is not None

    def observe(
        self,
        timing: PreparedFrameTiming,
        *,
        prepared_frame_idx: int,
    ) -> OverlayObservation:
        _ = timing  # prepared-domain pose does not require dense timing
        if not self.available or self._by_prepared is None:
            return OverlayObservation(
                layer_id=self._layer_id,
                available=False,
                reason=self._load_error or "Pose adapter unavailable.",
            )
        group = self._by_prepared.get(int(prepared_frame_idx))
        if group is None or group.empty:
            return OverlayObservation(
                layer_id=self._layer_id,
                available=False,
                reason=(
                    f"No {self._identity.subsystem} pose rows for prepared_frame_idx="
                    f"{prepared_frame_idx}."
                ),
            )
        points = _rows_to_points(group, require_track=self._require_track)
        if not points:
            return OverlayObservation(
                layer_id=self._layer_id,
                available=False,
                reason="Pose rows present but no finite coordinates.",
            )
        return OverlayObservation(
            layer_id=self._layer_id,
            available=True,
            reason=None,
            payload={
                "legend": self._legend,
                "domain": "prepared",
                "points": points,
                "n_points": len(points),
            },
        )


def make_s2_inferred_pose_adapter(
    parquet_path: Path | str | None,
    *,
    run_id: str | None = None,
    run_dir: Path | str | None = None,
) -> PreparedPoseAdapter:
    """S2 inferred pose — do not imply identity-final or invent corrected IDs."""

    return PreparedPoseAdapter(
        parquet_path,
        layer_id=LayerId.S2_POSE_INFERRED,
        subsystem="pose_inference",
        run_id=run_id,
        run_dir=run_dir,
        legend="inferred pose",
        require_track=False,
    )


def make_s3_tracked_pose_adapter(
    parquet_path: Path | str | None,
    *,
    run_id: str | None = None,
    run_dir: Path | str | None = None,
) -> PreparedPoseAdapter:
    """S3 tracking-corrected identities from accepted tracked pose."""

    return PreparedPoseAdapter(
        parquet_path,
        layer_id=LayerId.S3_POSE_TRACKED,
        subsystem="tracking_correction",
        run_id=run_id,
        run_dir=run_dir,
        legend="tracking-corrected identities",
        require_track=True,
    )


class S4FinalPoseAdapter:
    """S4 finalized pose on the dense timeline with faithful provenance codes."""

    def __init__(
        self,
        parquet_path: Path | str | None,
        *,
        run_id: str | None = None,
        run_dir: Path | str | None = None,
    ) -> None:
        self._path = (
            Path(parquet_path).expanduser().resolve()
            if parquet_path is not None
            else None
        )
        self._run_dir = (
            Path(run_dir).expanduser().resolve() if run_dir is not None else None
        )
        self._frame: pd.DataFrame | None = None
        self._by_sample: dict[int, pd.DataFrame] | None = None
        self._load_error: str | None = None
        if self._path is None or not self._path.is_file():
            self._load_error = (
                "S4 final_pose.parquet unavailable"
                if self._path is None
                else f"S4 final_pose.parquet missing: {self._path}"
            )
        else:
            try:
                self._frame = pd.read_parquet(self._path)
                require_columns(
                    self._frame.columns,
                    {"sample_idx", "track", "node", "x", "y", "provenance"},
                    label=str(self._path),
                )
                self._index_by_sample()
            except Exception as exc:  # noqa: BLE001
                self._load_error = f"Failed to load S4 final pose: {exc}"
                self._frame = None
                self._by_sample = None

        self._identity = SourceIdentity(
            subsystem="pose_finalization",
            run_id=run_id,
            run_dir=self._run_dir,
            artifact_path=self._path,
            digest_sha256=_optional_digest(self._path),
            extra={"legend": "final pose", "domain": "dense"},
        )

    def _index_by_sample(self) -> None:
        assert self._frame is not None
        work = self._frame.copy()
        keys = pd.to_numeric(work["sample_idx"], errors="coerce")
        work = work.assign(_sample_key=keys).dropna(subset=["_sample_key"])
        work["_sample_key"] = work["_sample_key"].astype(np.int64)
        self._by_sample = {
            int(k): g.drop(columns=["_sample_key"])
            for k, g in work.groupby("_sample_key", sort=False)
        }

    @property
    def layer_id(self) -> LayerId:
        return LayerId.S4_POSE_FINAL

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
                layer_id=self.layer_id,
                available=False,
                reason=self._load_error or "S4 pose adapter unavailable.",
            )
        if not timing.available or timing.dense_sample_idx is None:
            return OverlayObservation(
                layer_id=self.layer_id,
                available=False,
                reason=timing.reason or "Dense timing unavailable for S4 pose.",
            )
        group = self._by_sample.get(int(timing.dense_sample_idx))
        if group is None or group.empty:
            return OverlayObservation(
                layer_id=self.layer_id,
                available=False,
                reason=(
                    f"No S4 pose rows for dense_sample_idx={timing.dense_sample_idx}."
                ),
            )
        points = _s4_rows_to_points(group)
        drawable = [p for p in points if p.get("drawable")]
        if not drawable:
            return OverlayObservation(
                layer_id=self.layer_id,
                available=False,
                reason="S4 rows present but no drawable (non-missing) points.",
                payload={
                    "legend": "final pose",
                    "domain": "dense",
                    "dense_sample_idx": int(timing.dense_sample_idx),
                    "points": points,
                    "n_points": 0,
                },
            )
        return OverlayObservation(
            layer_id=self.layer_id,
            available=True,
            reason=None,
            payload={
                "legend": "final pose",
                "domain": "dense",
                "dense_sample_idx": int(timing.dense_sample_idx),
                "points": points,
                "n_points": len(drawable),
            },
        )


def _rows_to_points(
    group: pd.DataFrame,
    *,
    require_track: bool,
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in group.itertuples(index=False):
        xy = _finite_xy(getattr(row, "x", None), getattr(row, "y", None))
        if xy is None:
            continue
        track_val = getattr(row, "track", None)
        if track_val is None or (isinstance(track_val, float) and np.isnan(track_val)):
            if require_track:
                continue
            # S2 may expose instance_index when track is provisional.
            inst = getattr(row, "instance_index", None)
            track = int(inst) if inst is not None and not (
                isinstance(inst, float) and np.isnan(inst)
            ) else 0
        else:
            track = int(track_val)
        score = getattr(row, "score", None)
        points.append(
            {
                "track": track,
                "node": str(getattr(row, "node", "")),
                "x": xy[0],
                "y": xy[1],
                "score": None if score is None else float(score),
            }
        )
    return points


def _s4_rows_to_points(group: pd.DataFrame) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in group.itertuples(index=False):
        try:
            prov = int(row.provenance)
        except (TypeError, ValueError) as exc:
            raise OverlayAdapterError(
                f"Invalid S4 provenance code: {getattr(row, 'provenance', None)!r}"
            ) from exc
        if prov not in S4_PROVENANCE_NAMES:
            raise OverlayAdapterError(
                f"Unknown S4 provenance code {prov}; refusing to invent a category."
            )
        name = S4_PROVENANCE_NAMES[prov]
        recorded_name = getattr(row, "provenance_name", None)
        if (
            recorded_name is not None
            and str(recorded_name) not in {"", "nan", "<NA>"}
            and str(recorded_name) != name
        ):
            raise OverlayAdapterError(
                f"S4 provenance_name {recorded_name!r} does not match code "
                f"{prov} ({name})."
            )
        xy = _finite_xy(getattr(row, "x", None), getattr(row, "y", None))
        drawable = prov not in S4_UNAVAILABLE_PROVENANCE and xy is not None
        points.append(
            {
                "track": int(row.track),
                "node": str(row.node),
                "x": None if xy is None else xy[0],
                "y": None if xy is None else xy[1],
                "provenance": prov,
                "provenance_name": name,
                "drawable": drawable,
                "point_score": _optional_float(getattr(row, "point_score", None)),
            }
        )
    return points


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        out = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None
