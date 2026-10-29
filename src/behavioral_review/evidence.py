"""Lazy S7 prediction windows and S8 arbitration provenance (read-only).

Prefers S8-echoed probabilities for lightweight inspection. Full S7 tables are
loaded only on demand. Never re-thresholds or recalculates predictions.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from behavioral_arbitration.contracts import DIAGNOSTICS_FILENAME
from behavioral_review.contracts import (
    EVIDENCE_AVAILABLE,
    EVIDENCE_INVALID,
    EVIDENCE_MISSING,
    EVIDENCE_UNAVAILABLE,
    EVIDENCE_UNOBSERVED,
    EVIDENCE_VALID_NEGATIVE,
    EVIDENCE_VALID_POSITIVE,
    EVIDENCE_WINDOW_SCHEMA_VERSION,
    InspectionError,
)
from behavioral_review.ethogram import EthogramAccessor, EthogramWindow
from behavioral_review.source import ReviewSourcePackage
from behavioral_review.sync import SyncMap

_S7_CORE_COLUMNS: tuple[str, ...] = (
    "sample_idx",
    "dense_frame_idx",
    "time_sec",
    "prediction_grain",
    "pair_id",
    "focal_id",
    "partner_id",
    "family",
    "detector_name",
    "specialist_name",
    "output_role",
    "prob_raw",
    "prob_effective",
    "threshold",
    "prediction",
    "input_valid",
    "invalid_reason",
    "score_origin",
)

_SCORE_ORIGIN_ROUTING = "routing_sentinel"


@dataclass(frozen=True, slots=True)
class DetectorEvidencePoint:
    """One detector observation at a dense sample (S8 echo or S7 long-form)."""

    detector_name: str
    source: str  # "s8_echo" | "s7"
    focal_id: int | None
    partner_id: int | None
    sample_idx: int
    prob: float | None
    threshold: float | None
    prediction: int | None
    input_valid: bool | None
    score_origin: str | None
    invalid_reason: str | None
    specialist_name: str | None
    output_role: str | None
    availability: str
    display_state: str


@dataclass(frozen=True, slots=True)
class S8EchoEvidenceWindow:
    """Lightweight evidence from S8 echoed prob/validity columns."""

    schema_version: str
    start_sample_idx: int
    end_sample_idx: int
    points: tuple[DetectorEvidencePoint, ...]
    availability: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class S7EvidenceWindow:
    """On-demand window into canonical S7 ``predictions.parquet``."""

    schema_version: str
    start_sample_idx: int
    end_sample_idx: int
    detectors: tuple[str, ...]
    frame: pd.DataFrame
    points: tuple[DetectorEvidencePoint, ...]
    availability: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ArbitrationProvenance:
    """Read-only S8 provenance for a sample or bout selection."""

    schema_version: str
    scope: str  # "sample" | "bout"
    sample_idx: int | None
    bout_id: str | None
    focal_id: int | None
    partner_id: int | None
    behavior_label: str | None
    label_reconstruction: str | None
    undefined_reason: str | None
    hierarchy_residual_kind: str | None
    pred_contact: int | None
    pred_side_by_side: int | None
    candidate_predictions: dict[str, int]
    echoed_probabilities: dict[str, float | None]
    stage_diagnostics: tuple[dict[str, Any], ...]
    stage_diagnostics_availability: str
    warnings: tuple[str, ...] = ()


@dataclass
class EvidenceAccessor:
    """Lazy evidence facade over S9-referenced S8/S7 artifacts."""

    source: ReviewSourcePackage
    sync: SyncMap
    ethogram: EthogramAccessor
    s7_predictions_path: Path | None = None
    s7_available: bool = False
    diagnostics_path: Path | None = None
    diagnostics_available: bool = False
    _s7_columns: frozenset[str] | None = field(default=None, init=False, repr=False)
    _diagnostics_cache: tuple[dict[str, Any], ...] | None = field(
        default=None, init=False, repr=False
    )

    @classmethod
    def from_source(
        cls,
        source: ReviewSourcePackage,
        sync: SyncMap,
        ethogram: EthogramAccessor,
    ) -> EvidenceAccessor:
        s7_path = _resolve_s7_predictions_path(source)
        diag_path = _resolve_s8_diagnostics_path(source)
        return cls(
            source=source,
            sync=sync,
            ethogram=ethogram,
            s7_predictions_path=s7_path,
            s7_available=bool(s7_path is not None and s7_path.is_file()),
            diagnostics_path=diag_path,
            diagnostics_available=bool(diag_path is not None and diag_path.is_file()),
        )

    def s8_echoed_window(
        self,
        start_sample_idx: int,
        end_sample_idx: int,
        *,
        directions: Sequence[tuple[int, int]] | None = None,
        detectors: Sequence[str] | None = None,
    ) -> S8EchoEvidenceWindow:
        """Prefer S8 echoed probabilities (U10) for lightweight inspection."""

        window = self.ethogram.read_window(
            start_sample_idx,
            end_sample_idx,
            directions=directions,
            include_echo_probs=True,
            include_validity=True,
            keep_raw=True,
        )
        if not window.states_available or window.raw_frame is None or window.raw_frame.empty:
            return S8EchoEvidenceWindow(
                schema_version=EVIDENCE_WINDOW_SCHEMA_VERSION,
                start_sample_idx=int(start_sample_idx),
                end_sample_idx=int(end_sample_idx),
                points=(),
                availability=EVIDENCE_UNAVAILABLE,
                warnings=("S8 echoed evidence unavailable (states missing).",),
            )
        det_list = list(detectors) if detectors is not None else _default_echo_detectors(
            window.raw_frame
        )
        points = _points_from_s8_echo(window.raw_frame, detectors=det_list)
        return S8EchoEvidenceWindow(
            schema_version=EVIDENCE_WINDOW_SCHEMA_VERSION,
            start_sample_idx=window.start_sample_idx,
            end_sample_idx=window.end_sample_idx,
            points=points,
            availability=EVIDENCE_AVAILABLE,
        )

    def s7_prediction_window(
        self,
        start_sample_idx: int,
        end_sample_idx: int,
        *,
        detectors: Sequence[str] | None = None,
        focal_id: int | None = None,
        partner_id: int | None = None,
        columns: Sequence[str] | None = None,
    ) -> S7EvidenceWindow:
        """On-demand windowed S7 long-form predictions (no re-thresholding)."""

        start, end = int(start_sample_idx), int(end_sample_idx)
        if start > end:
            raise InspectionError(f"Invalid S7 window: start ({start}) > end ({end}).")
        if not self.s7_available or self.s7_predictions_path is None:
            return S7EvidenceWindow(
                schema_version=EVIDENCE_WINDOW_SCHEMA_VERSION,
                start_sample_idx=start,
                end_sample_idx=end,
                detectors=tuple(detectors or ()),
                frame=pd.DataFrame(),
                points=(),
                availability=EVIDENCE_UNAVAILABLE,
                warnings=("S7 predictions.parquet unavailable; deep-dive evidence blocked.",),
            )

        wanted_cols = list(columns) if columns is not None else list(_S7_CORE_COLUMNS)
        available_cols = self._list_s7_columns()
        cols = [c for c in wanted_cols if c in available_cols]
        if "sample_idx" not in cols and "dense_frame_idx" not in cols:
            raise InspectionError("S7 predictions missing sample_idx/dense_frame_idx.")

        frame = _read_s7_window(
            self.s7_predictions_path,
            start=start,
            end=end,
            columns=cols,
            detectors=detectors,
            focal_id=focal_id,
            partner_id=partner_id,
        )
        points = _points_from_s7(frame)
        det_names = tuple(
            sorted({str(x) for x in frame["detector_name"].astype(str).unique()})
            if not frame.empty and "detector_name" in frame.columns
            else tuple(detectors or ())
        )
        return S7EvidenceWindow(
            schema_version=EVIDENCE_WINDOW_SCHEMA_VERSION,
            start_sample_idx=start,
            end_sample_idx=end,
            detectors=det_names,
            frame=frame,
            points=points,
            availability=EVIDENCE_AVAILABLE if not frame.empty else EVIDENCE_MISSING,
            warnings=()
            if not frame.empty
            else ("No S7 prediction rows in requested window/filter.",),
        )

    def provenance_at_sample(
        self,
        sample_idx: int,
        *,
        focal_id: int,
        partner_id: int,
    ) -> ArbitrationProvenance:
        window = self.ethogram.read_window(
            int(sample_idx),
            int(sample_idx),
            directions=((int(focal_id), int(partner_id)),),
            include_echo_probs=True,
            include_validity=True,
            keep_raw=True,
        )
        return self._provenance_from_window(
            window,
            scope="sample",
            sample_idx=int(sample_idx),
            bout_id=None,
            focal_id=int(focal_id),
            partner_id=int(partner_id),
        )

    def provenance_for_bout(
        self,
        bout_row: Mapping[str, Any] | pd.Series,
    ) -> ArbitrationProvenance:
        bout_id = str(bout_row["bout_id"])
        start = int(bout_row["start_sample_idx"])
        channel = str(bout_row["channel"])
        focal_i: int | None
        partner_i: int | None
        if channel == "exclusive":
            focal = bout_row.get("focal_id")
            partner = bout_row.get("partner_id")
            if focal is None or partner is None or pd.isna(focal) or pd.isna(partner):
                raise InspectionError(
                    f"Exclusive bout {bout_id!r} missing focal/partner for provenance."
                )
            focal_i = int(focal)
            partner_i = int(partner)
            directions = ((focal_i, partner_i),)
        else:
            # Parallel: use first available directional row at bout start.
            directions = None
            focal_i, partner_i = None, None
        window = self.ethogram.read_window(
            start,
            start,
            directions=directions,
            include_echo_probs=True,
            include_validity=True,
            keep_raw=True,
        )
        if focal_i is None and window.directions:
            lane = window.directions[0].lane
            focal_i, partner_i = lane.focal_id, lane.partner_id
        return self._provenance_from_window(
            window,
            scope="bout",
            sample_idx=start,
            bout_id=bout_id,
            focal_id=focal_i,
            partner_id=partner_i,
            bout_reconstruction_mode=(
                str(bout_row["label_reconstruction_mode"])
                if "label_reconstruction_mode" in bout_row
                and pd.notna(bout_row["label_reconstruction_mode"])
                else None
            ),
        )

    def stage_diagnostics(self) -> tuple[dict[str, Any], ...]:
        """Return optional S8 stage diagnostics or empty when unavailable."""

        availability, rows = self._load_diagnostics()
        if availability != EVIDENCE_AVAILABLE:
            return ()
        return rows

    def stage_diagnostics_availability(self) -> str:
        availability, _ = self._load_diagnostics()
        return availability

    def _provenance_from_window(
        self,
        window: EthogramWindow,
        *,
        scope: str,
        sample_idx: int | None,
        bout_id: str | None,
        focal_id: int | None,
        partner_id: int | None,
        bout_reconstruction_mode: str | None = None,
    ) -> ArbitrationProvenance:
        warnings: list[str] = []
        if not window.states_available or window.raw_frame is None or window.raw_frame.empty:
            diag_avail, _ = self._load_diagnostics()
            return ArbitrationProvenance(
                schema_version=EVIDENCE_WINDOW_SCHEMA_VERSION,
                scope=scope,
                sample_idx=sample_idx,
                bout_id=bout_id,
                focal_id=focal_id,
                partner_id=partner_id,
                behavior_label=None,
                label_reconstruction=None,
                undefined_reason=None,
                hierarchy_residual_kind=None,
                pred_contact=None,
                pred_side_by_side=None,
                candidate_predictions={},
                echoed_probabilities={},
                stage_diagnostics=(),
                stage_diagnostics_availability=diag_avail,
                warnings=("S8 states unavailable for provenance.",),
            )

        frame = window.raw_frame
        if focal_id is not None and partner_id is not None:
            hits = frame.loc[
                (frame["focal_id"].astype("int64") == int(focal_id))
                & (frame["partner_id"].astype("int64") == int(partner_id))
            ]
        else:
            hits = frame
        if hits.empty:
            warnings.append("No matching directional state row for provenance query.")
            row = None
        else:
            row = hits.iloc[0]

        candidates: dict[str, int] = {}
        probs: dict[str, float | None] = {}
        label = None
        recon = bout_reconstruction_mode
        undef = None
        residual = None
        pred_contact = None
        pred_sbs = None
        if row is not None:
            label = str(row["behavior_label"]) if "behavior_label" in row.index else None
            if recon is None and "label_reconstruction" in row.index:
                recon = str(row["label_reconstruction"]) if pd.notna(row["label_reconstruction"]) else None
            if "undefined_reason" in row.index and pd.notna(row["undefined_reason"]):
                undef = str(row["undefined_reason"]) or None
            if (
                "hierarchy_residual_kind" in row.index
                and pd.notna(row["hierarchy_residual_kind"])
            ):
                residual = str(row["hierarchy_residual_kind"]) or None
            for col in row.index:
                name = str(col)
                if name.startswith("pred_") and name not in {
                    "pred_contact",
                    "pred_side_by_side",
                    "pred_engaged",
                }:
                    val = row[col]
                    if pd.notna(val):
                        candidates[name.removeprefix("pred_")] = int(val)
                if name.startswith("prob_") and name not in {
                    "prob_contact_fused",
                    "prob_contact_smooth",
                    "prob_contact_raw",
                }:
                    val = row[col]
                    probs[name.removeprefix("prob_")] = (
                        float(val) if pd.notna(val) else None
                    )
            if "pred_contact" in row.index and pd.notna(row["pred_contact"]):
                pred_contact = int(row["pred_contact"])
            if "pred_side_by_side" in row.index and pd.notna(row["pred_side_by_side"]):
                pred_sbs = int(row["pred_side_by_side"])

        diag_avail, diag_rows = self._load_diagnostics()
        if diag_avail != EVIDENCE_AVAILABLE:
            warnings.append(
                "S8 stage diagnostics file absent; rule reconstruction explanation "
                "unavailable (not invented)."
            )
        return ArbitrationProvenance(
            schema_version=EVIDENCE_WINDOW_SCHEMA_VERSION,
            scope=scope,
            sample_idx=sample_idx,
            bout_id=bout_id,
            focal_id=focal_id,
            partner_id=partner_id,
            behavior_label=label,
            label_reconstruction=recon,
            undefined_reason=undef,
            hierarchy_residual_kind=residual,
            pred_contact=pred_contact,
            pred_side_by_side=pred_sbs,
            candidate_predictions=candidates,
            echoed_probabilities=probs,
            stage_diagnostics=diag_rows if diag_avail == EVIDENCE_AVAILABLE else (),
            stage_diagnostics_availability=diag_avail,
            warnings=tuple(warnings),
        )

    def _load_diagnostics(self) -> tuple[str, tuple[dict[str, Any], ...]]:
        if self._diagnostics_cache is not None:
            return (
                EVIDENCE_AVAILABLE
                if self.diagnostics_available
                else EVIDENCE_UNAVAILABLE,
                self._diagnostics_cache,
            )
        if not self.diagnostics_available or self.diagnostics_path is None:
            self._diagnostics_cache = ()
            return EVIDENCE_UNAVAILABLE, ()
        path = self.diagnostics_path
        try:
            if path.suffix.lower() == ".parquet":
                frame = pd.read_parquet(path)
                rows = tuple(frame.to_dict(orient="records"))
            else:
                # JSON list / object fallback (not present on E-session).
                import json

                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, list):
                    rows = tuple(dict(x) for x in payload if isinstance(x, Mapping))
                elif isinstance(payload, Mapping) and isinstance(
                    payload.get("diagnostics"), list
                ):
                    rows = tuple(
                        dict(x)
                        for x in payload["diagnostics"]
                        if isinstance(x, Mapping)
                    )
                else:
                    rows = ()
            self._diagnostics_cache = rows
            return EVIDENCE_AVAILABLE, rows
        except Exception as exc:  # noqa: BLE001
            self.diagnostics_available = False
            self._diagnostics_cache = ()
            raise InspectionError(
                f"Failed reading S8 diagnostics at {path}: {exc}"
            ) from exc

    def _list_s7_columns(self) -> frozenset[str]:
        if self._s7_columns is not None:
            return self._s7_columns
        if self.s7_predictions_path is None:
            return frozenset()
        try:
            import pyarrow.parquet as pq  # type: ignore[import-untyped]

            names = frozenset(
                str(n) for n in pq.ParquetFile(self.s7_predictions_path).schema_arrow.names
            )
        except Exception:  # noqa: BLE001
            probe = pd.read_parquet(self.s7_predictions_path)
            names = frozenset(str(c) for c in probe.columns)
        self._s7_columns = names
        return names


def classify_prediction_display(
    *,
    input_valid: bool | None,
    score_origin: str | None,
    prediction: int | None,
    available: bool,
) -> str:
    """Map validity/provenance to GUI display state (invalid ≠ confident 0)."""

    if not available:
        return EVIDENCE_MISSING
    if input_valid is False or score_origin == _SCORE_ORIGIN_ROUTING:
        return EVIDENCE_INVALID if input_valid is False else EVIDENCE_UNOBSERVED
    if prediction == 1:
        return EVIDENCE_VALID_POSITIVE
    if prediction == 0:
        return EVIDENCE_VALID_NEGATIVE
    return EVIDENCE_MISSING


def _resolve_s7_predictions_path(source: ReviewSourcePackage) -> Path | None:
    meta = source.run_meta
    source_s7 = meta.get("source_s7") if isinstance(meta.get("source_s7"), Mapping) else {}
    run_dir_text = source_s7.get("run_dir") if isinstance(source_s7, Mapping) else None
    if isinstance(run_dir_text, str) and run_dir_text.strip():
        run_dir = Path(run_dir_text).expanduser().resolve()
        preds = run_dir / "predictions.parquet"
        if preds.is_file():
            return preds
    # Fallback: session_root / behavioral_inference / run_id
    run_id = source.source_s7_run_id
    if run_id and source.session_root is not None:
        candidate = (
            source.session_root / "behavioral_inference" / run_id / "predictions.parquet"
        )
        if candidate.is_file():
            return candidate
    return None


def _resolve_s8_diagnostics_path(source: ReviewSourcePackage) -> Path | None:
    run_dir = source.source_s8.run_dir
    if run_dir is None:
        return None
    for name in (DIAGNOSTICS_FILENAME, "stage_diagnostics.json", "diagnostics.json"):
        path = run_dir / name
        if path.is_file():
            return path
    return None


def _default_echo_detectors(frame: pd.DataFrame) -> list[str]:
    dets: list[str] = []
    for col in frame.columns:
        if col.startswith("prob_") and not col.endswith(("_raw", "_fused", "_smooth")):
            dets.append(col.removeprefix("prob_"))
    # Prefer ethogram-relevant order.
    preferred = [
        "nose_to_nose",
        "nose_to_body",
        "nose_to_tail",
        "approach",
        "retract",
        "mutually_disengaged",
        "engaged",
        "contact",
        "side_by_side",
    ]
    ordered = [d for d in preferred if d in dets]
    ordered.extend(sorted(d for d in dets if d not in ordered))
    return ordered


def _points_from_s8_echo(
    frame: pd.DataFrame,
    *,
    detectors: Sequence[str],
) -> tuple[DetectorEvidencePoint, ...]:
    points: list[DetectorEvidencePoint] = []
    records = frame.to_dict(orient="records")
    for data in records:
        sample = int(data.get("sample_idx", data["dense_frame_idx"]))
        focal = int(data["focal_id"])
        partner = int(data["partner_id"])
        for det in detectors:
            prob_col = f"prob_{det}"
            pred_col = f"pred_{det}"
            valid_col = f"arbitration_input_valid_{det}"
            origin_col = f"score_origin_{det}"
            thr_col = f"threshold_{det}"
            if prob_col not in data and pred_col not in data:
                continue
            prob_val = data.get(prob_col)
            pred_val = data.get(pred_col)
            valid: bool | None
            if valid_col in data and pd.notna(data.get(valid_col)):
                valid = bool(data.get(valid_col))
            else:
                alt = data.get(f"input_valid_{det}")
                valid = bool(alt) if alt is not None and pd.notna(alt) else None
            origin = data.get(origin_col)
            origin_s = str(origin) if origin is not None and pd.notna(origin) else None
            pred_i = int(pred_val) if pred_val is not None and pd.notna(pred_val) else None
            thr = data.get(thr_col)
            display = classify_prediction_display(
                input_valid=valid,
                score_origin=origin_s,
                prediction=pred_i,
                available=True,
            )
            points.append(
                DetectorEvidencePoint(
                    detector_name=det,
                    source="s8_echo",
                    focal_id=focal,
                    partner_id=partner,
                    sample_idx=sample,
                    prob=(
                        float(prob_val)
                        if prob_val is not None and pd.notna(prob_val)
                        else None
                    ),
                    threshold=float(thr) if thr is not None and pd.notna(thr) else None,
                    prediction=pred_i,
                    input_valid=valid,
                    score_origin=origin_s,
                    invalid_reason=None,
                    specialist_name=None,
                    output_role=None,
                    availability=EVIDENCE_AVAILABLE,
                    display_state=display,
                )
            )
    return tuple(points)


def _points_from_s7(frame: pd.DataFrame) -> tuple[DetectorEvidencePoint, ...]:
    if frame.empty:
        return ()
    points: list[DetectorEvidencePoint] = []
    for data in frame.to_dict(orient="records"):
        valid_raw = data.get("input_valid")
        valid = bool(valid_raw) if valid_raw is not None and pd.notna(valid_raw) else None
        origin = data.get("score_origin")
        origin_s = str(origin) if origin is not None and pd.notna(origin) else None
        pred_raw = data.get("prediction")
        pred_i = int(pred_raw) if pred_raw is not None and pd.notna(pred_raw) else None
        display = classify_prediction_display(
            input_valid=valid,
            score_origin=origin_s,
            prediction=pred_i,
            available=True,
        )
        focal = data.get("focal_id")
        partner = data.get("partner_id")
        sample = data.get("sample_idx", data.get("dense_frame_idx"))
        prob = data.get("prob_effective")
        thr = data.get("threshold")
        invalid = data.get("invalid_reason")
        specialist = data.get("specialist_name")
        role = data.get("output_role")
        points.append(
            DetectorEvidencePoint(
                detector_name=str(data.get("detector_name")),
                source="s7",
                focal_id=int(focal) if focal is not None and pd.notna(focal) else None,
                partner_id=(
                    int(partner) if partner is not None and pd.notna(partner) else None
                ),
                sample_idx=int(sample),
                prob=float(prob) if prob is not None and pd.notna(prob) else None,
                threshold=float(thr) if thr is not None and pd.notna(thr) else None,
                prediction=pred_i,
                input_valid=valid,
                score_origin=origin_s,
                invalid_reason=(
                    str(invalid) if invalid is not None and pd.notna(invalid) else None
                ),
                specialist_name=(
                    str(specialist)
                    if specialist is not None and pd.notna(specialist)
                    else None
                ),
                output_role=str(role) if role is not None and pd.notna(role) else None,
                availability=EVIDENCE_AVAILABLE,
                display_state=display,
            )
        )
    return tuple(points)


def _read_s7_window(
    path: Path,
    *,
    start: int,
    end: int,
    columns: Sequence[str],
    detectors: Sequence[str] | None,
    focal_id: int | None,
    partner_id: int | None,
) -> pd.DataFrame:
    idx_col = "sample_idx" if "sample_idx" in columns else "dense_frame_idx"
    try:
        frame = pd.read_parquet(
            path,
            columns=list(columns),
            filters=[
                (idx_col, ">=", int(start)),
                (idx_col, "<=", int(end)),
            ],
        )
    except (TypeError, ValueError, OSError):
        frame = pd.read_parquet(path, columns=list(columns))
        dens = frame[idx_col].astype("int64")
        frame = frame.loc[(dens >= start) & (dens <= end)]
    if detectors is not None and "detector_name" in frame.columns:
        frame = frame.loc[frame["detector_name"].astype(str).isin(list(detectors))]
    if focal_id is not None and "focal_id" in frame.columns:
        frame = frame.loc[
            pd.to_numeric(frame["focal_id"], errors="coerce") == int(focal_id)
        ]
    if partner_id is not None and "partner_id" in frame.columns:
        frame = frame.loc[
            pd.to_numeric(frame["partner_id"], errors="coerce") == int(partner_id)
        ]
    return frame.reset_index(drop=True)
