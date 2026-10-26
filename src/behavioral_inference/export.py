"""Canonical + wide prediction table construction and atomic Parquet writes.

Scientific inference remains in ``engine`` / adapters. This module only
validates, reshapes, and publishes artifacts for S8 handoff.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd  # type: ignore[import-untyped]

from behavioral_inference.contracts import (
    DYADIC_WIDE_SCHEMA_VERSION,
    FAMILY_DYADIC,
    FAMILY_FOCAL,
    FOCAL_WIDE_SCHEMA_VERSION,
    OUTPUT_ROLE_LOGICAL_FUSED,
    OUTPUT_ROLE_PRIMARY,
    OUTPUT_ROLE_SPECIALIST,
    PREDICTIONS_SCHEMA_VERSION,
    PublicationError,
)
from behavioral_inference.results import (
    PREDICTION_COLUMNS,
    BehavioralInferenceResult,
)
from behavioral_inference.validate import (
    annotate_score_origin,
    assert_wide_traceable_to_canonical,
    validate_canonical_predictions,
)

# Published long-form columns (Sprint 2 names + provenance + score_origin).
PUBLISHED_PREDICTION_COLUMNS: tuple[str, ...] = (
    *PREDICTION_COLUMNS,
    "score_origin",
    "source_s6_run_id",
    "session_root",
)

# Pipeline 100 / 70 / 90 column naming for wide views.
_DYADIC_LOGICAL_PROB_ORDER: tuple[str, ...] = (
    "contact",
    "nose_to_nose",
    "mutually_disengaged",
    "side_by_side",
)
_CONTACT_SPECIALIST_ORDER: tuple[str, ...] = (
    "contact_intact",
    "contact_occluded",
)
_FOCAL_LOGICAL_ORDER: tuple[str, ...] = (
    "nose_to_body",
    "nose_to_tail",
    "approach",
    "retract",
)
_NOSE_BODY_SPECIALIST_ORDER: tuple[str, ...] = (
    "nose_body_intact",
    "nose_body_occ",
)


def build_canonical_predictions_table(
    result: BehavioralInferenceResult,
    *,
    session_root: Path | str,
    source_s6_run_id: str,
    n_unique_frames: int | None = None,
    require_exactly_two_mice: bool = True,
) -> pd.DataFrame:
    """Validate and stamp provenance on canonical long-form predictions."""

    validate_canonical_predictions(
        result,
        n_unique_frames=n_unique_frames,
        require_exactly_two_mice=require_exactly_two_mice,
    )
    out = annotate_score_origin(result.predictions)
    out["schema_version"] = PREDICTIONS_SCHEMA_VERSION
    out["source_s6_run_id"] = str(source_s6_run_id)
    out["session_root"] = str(Path(session_root).expanduser().resolve())
    return order_prediction_columns(out)


def build_specialist_predictions_table(
    result: BehavioralInferenceResult,
    *,
    session_root: Path | str,
    source_s6_run_id: str,
) -> pd.DataFrame:
    """Publish specialist / diagnostic rows with the same provenance stamp."""

    if result.diagnostics.empty:
        # Empty specialist table is allowed when no dual-model detectors ran.
        empty = pd.DataFrame(
            {
                col: pd.Series(
                    dtype=(
                        "int64"
                        if col
                        in (
                            "sample_idx",
                            "dense_frame_idx",
                            "mouse_a",
                            "mouse_b",
                            "focal_id",
                            "partner_id",
                            "prediction",
                        )
                        else (
                            "float64"
                            if col
                            in (
                                "time_sec",
                                "prob_raw",
                                "prob_effective",
                                "threshold",
                            )
                            else (
                                "bool"
                                if col == "input_valid"
                                else "object"
                            )
                        )
                    )
                )
                for col in PUBLISHED_PREDICTION_COLUMNS
            }
        )
        return empty

    out = annotate_score_origin(result.diagnostics)
    out["schema_version"] = PREDICTIONS_SCHEMA_VERSION
    out["source_s6_run_id"] = str(source_s6_run_id)
    out["session_root"] = str(Path(session_root).expanduser().resolve())
    return order_prediction_columns(out)


def order_prediction_columns(table: pd.DataFrame) -> pd.DataFrame:
    """Deterministic column order for published long-form tables."""

    missing = [c for c in PUBLISHED_PREDICTION_COLUMNS if c not in table.columns]
    if missing:
        raise PublicationError(
            "Prediction table missing required publish columns: " + ", ".join(missing)
        )
    remainder = [c for c in table.columns if c not in PUBLISHED_PREDICTION_COLUMNS]
    return table.loc[:, list(PUBLISHED_PREDICTION_COLUMNS) + remainder]


def build_dyadic_predictions_wide(
    *,
    predictions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    selected_detectors: Sequence[str],
    exp_name: str = "session",
) -> pd.DataFrame:
    """Build Pipeline-70-shaped undirected wide view from canonical rows.

    One row per ``(dense_frame_idx, mouse_a, mouse_b)``. Collapse survivors
    provide ``focal_id`` / ``partner_id`` as ``mouse_a`` / ``mouse_b`` for
    Pipeline 100 pair-key expansion — not fabricated alternate directions.
    Validity columns preserve routing semantics so S8 must not blind
    re-threshold invalid scores.
    """

    selected = [d for d in selected_detectors if d]
    if not selected:
        return pd.DataFrame()

    predictions = (
        predictions
        if "score_origin" in predictions.columns
        else annotate_score_origin(predictions)
    )
    diagnostics = (
        diagnostics
        if diagnostics.empty or "score_origin" in diagnostics.columns
        else annotate_score_origin(diagnostics)
    )

    primary = predictions[
        (predictions["family"] == FAMILY_DYADIC)
        & (predictions["output_role"].isin([OUTPUT_ROLE_PRIMARY, OUTPUT_ROLE_LOGICAL_FUSED]))
        & (predictions["detector_name"].isin(selected))
    ].copy()
    if primary.empty:
        raise PublicationError("No dyadic primary/fused predictions for wide view.")

    keys = ["dense_frame_idx", "mouse_a", "mouse_b"]
    base = (
        primary.loc[:, keys + ["sample_idx", "time_sec", "pair_id"]]
        .drop_duplicates(subset=keys, keep="first")
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )
    base["exp_name"] = exp_name
    # Collapse survivors: lower ID as focal / higher as partner (S7-B §1.2).
    base["focal_id"] = base["mouse_a"].astype("int64")
    base["partner_id"] = base["mouse_b"].astype("int64")
    base["schema_version"] = DYADIC_WIDE_SCHEMA_VERSION

    out = base.copy()
    ordered_detectors = [d for d in _DYADIC_LOGICAL_PROB_ORDER if d in selected]
    ordered_detectors.extend([d for d in selected if d not in ordered_detectors])

    for detector in ordered_detectors:
        block = primary[primary["detector_name"] == detector]
        merged = block.loc[
            :,
            keys
            + [
                "prob_effective",
                "prob_raw",
                "prediction",
                "input_valid",
                "invalid_reason",
                "score_origin",
                "threshold",
            ],
        ].rename(
            columns={
                "prob_effective": f"prob_{detector}",
                "prob_raw": f"prob_{detector}_raw",
                "prediction": f"pred_{detector}",
                "input_valid": f"input_valid_{detector}",
                "invalid_reason": f"invalid_reason_{detector}",
                "score_origin": f"score_origin_{detector}",
                "threshold": f"threshold_{detector}",
            }
        )
        out = out.merge(merged, on=keys, how="left", validate="one_to_one")

    # Contact specialists from diagnostics when contact selected.
    if "contact" in selected and not diagnostics.empty:
        for specialist in _CONTACT_SPECIALIST_ORDER:
            block = diagnostics[
                (diagnostics["specialist_name"] == specialist)
                & (diagnostics["output_role"] == OUTPUT_ROLE_SPECIALIST)
            ]
            if block.empty:
                continue
            short = specialist  # contact_intact / contact_occluded
            merged = block.loc[
                :,
                keys
                + [
                    "prob_effective",
                    "prediction",
                    "input_valid",
                    "invalid_reason",
                    "score_origin",
                ],
            ].rename(
                columns={
                    "prob_effective": f"prob_{short}",
                    "prediction": f"pred_{short}",
                    "input_valid": f"input_valid_{short}",
                    "invalid_reason": f"invalid_reason_{short}",
                    "score_origin": f"score_origin_{short}",
                }
            )
            out = out.merge(merged, on=keys, how="left", validate="one_to_one")

    assert_wide_traceable_to_canonical(
        wide=out, canonical=predictions, family=FAMILY_DYADIC
    )
    return _order_dyadic_wide_columns(out, ordered_detectors)


def build_focal_predictions_wide(
    *,
    predictions: pd.DataFrame,
    diagnostics: pd.DataFrame,
    selected_detectors: Sequence[str],
    features_df: pd.DataFrame | None = None,
    exp_name: str = "session",
) -> pd.DataFrame:
    """Build Pipeline-90-shaped directional wide view from canonical rows.

    Preserves both directions. Nose-to-body exports pre-smooth raw
    (``prob_nose_to_body_raw``) and smoothed effective (``prob_nose_to_body``)
    plus ``input_valid_nose_to_body`` so S8 must not blind re-threshold invalid
    frames where smoothing may pull a non-zero effective from neighbors.
    """

    selected = [d for d in selected_detectors if d]
    if not selected:
        return pd.DataFrame()

    predictions = (
        predictions
        if "score_origin" in predictions.columns
        else annotate_score_origin(predictions)
    )
    diagnostics = (
        diagnostics
        if diagnostics.empty or "score_origin" in diagnostics.columns
        else annotate_score_origin(diagnostics)
    )

    primary = predictions[
        (predictions["family"] == FAMILY_FOCAL)
        & (predictions["output_role"].isin([OUTPUT_ROLE_PRIMARY, OUTPUT_ROLE_LOGICAL_FUSED]))
        & (predictions["detector_name"].isin(selected))
    ].copy()
    if primary.empty:
        raise PublicationError("No focal primary/fused predictions for wide view.")

    keys = ["dense_frame_idx", "focal_id", "partner_id"]
    base = (
        primary.loc[:, keys + ["sample_idx", "time_sec"]]
        .drop_duplicates(subset=keys, keep="first")
        .sort_values(keys, kind="stable")
        .reset_index(drop=True)
    )
    base["exp_name"] = exp_name
    base["schema_version"] = FOCAL_WIDE_SCHEMA_VERSION

    if features_df is not None and "is_total_occlusion" in features_df.columns:
        occ = (
            features_df.loc[:, keys + ["is_total_occlusion"]]
            .drop_duplicates(subset=keys, keep="first")
        )
        base = base.merge(occ, on=keys, how="left", validate="one_to_one")

    out = base.copy()
    ordered_detectors = [d for d in _FOCAL_LOGICAL_ORDER if d in selected]
    ordered_detectors.extend([d for d in selected if d not in ordered_detectors])

    for detector in ordered_detectors:
        block = primary[primary["detector_name"] == detector]
        cols = {
            "prob_effective": f"prob_{detector}",
            "prob_raw": f"prob_{detector}_raw",
            "prediction": f"pred_{detector}",
            "input_valid": f"input_valid_{detector}",
            "invalid_reason": f"invalid_reason_{detector}",
            "score_origin": f"score_origin_{detector}",
            "threshold": f"threshold_{detector}",
        }
        # Pipeline 100 looks up prob_nose_to_body (effective/smoothed).
        if detector == "nose_to_body":
            cols["prob_raw"] = "prob_nose_to_body_raw"
            cols["prob_effective"] = "prob_nose_to_body"
        merged = block.loc[:, keys + list(cols.keys())].rename(columns=cols)
        out = out.merge(merged, on=keys, how="left", validate="one_to_one")

    if "nose_to_body" in selected and not diagnostics.empty:
        for specialist in _NOSE_BODY_SPECIALIST_ORDER:
            block = diagnostics[
                (diagnostics["specialist_name"] == specialist)
                & (diagnostics["output_role"] == OUTPUT_ROLE_SPECIALIST)
            ]
            if block.empty:
                continue
            merged = block.loc[
                :,
                keys
                + [
                    "prob_effective",
                    "prediction",
                    "input_valid",
                    "invalid_reason",
                    "score_origin",
                ],
            ].rename(
                columns={
                    "prob_effective": f"prob_{specialist}",
                    "prediction": f"pred_{specialist}",
                    "input_valid": f"input_valid_{specialist}",
                    "invalid_reason": f"invalid_reason_{specialist}",
                    "score_origin": f"score_origin_{specialist}",
                }
            )
            out = out.merge(merged, on=keys, how="left", validate="one_to_one")

    assert_wide_traceable_to_canonical(
        wide=out, canonical=predictions, family=FAMILY_FOCAL
    )
    return _order_focal_wide_columns(out, ordered_detectors)


def _order_dyadic_wide_columns(
    table: pd.DataFrame,
    detectors: Sequence[str],
) -> pd.DataFrame:
    leading = [
        "schema_version",
        "exp_name",
        "sample_idx",
        "dense_frame_idx",
        "time_sec",
        "pair_id",
        "mouse_a",
        "mouse_b",
        "focal_id",
        "partner_id",
    ]
    ordered: list[str] = [c for c in leading if c in table.columns]
    for detector in detectors:
        candidates = [
            f"prob_{detector}",
            f"prob_{detector}_raw",
            f"pred_{detector}",
            f"input_valid_{detector}",
            f"invalid_reason_{detector}",
            f"score_origin_{detector}",
            f"threshold_{detector}",
        ]
        if detector == "contact":
            for specialist in _CONTACT_SPECIALIST_ORDER:
                candidates.extend(
                    [
                        f"prob_{specialist}",
                        f"pred_{specialist}",
                        f"input_valid_{specialist}",
                        f"invalid_reason_{specialist}",
                        f"score_origin_{specialist}",
                    ]
                )
        for col in candidates:
            if col in table.columns and col not in ordered:
                ordered.append(col)
    remainder = [c for c in table.columns if c not in ordered]
    return table.loc[:, ordered + remainder]


def _order_focal_wide_columns(
    table: pd.DataFrame,
    detectors: Sequence[str],
) -> pd.DataFrame:
    leading = [
        "schema_version",
        "exp_name",
        "sample_idx",
        "dense_frame_idx",
        "time_sec",
        "focal_id",
        "partner_id",
        "is_total_occlusion",
    ]
    ordered: list[str] = [c for c in leading if c in table.columns]
    for detector in detectors:
        if detector == "nose_to_body":
            candidates = [
                "prob_nose_to_body",
                "prob_nose_to_body_raw",
                "pred_nose_to_body",
                "input_valid_nose_to_body",
                "invalid_reason_nose_to_body",
                "score_origin_nose_to_body",
                "threshold_nose_to_body",
            ]
            for specialist in _NOSE_BODY_SPECIALIST_ORDER:
                candidates.extend(
                    [
                        f"prob_{specialist}",
                        f"pred_{specialist}",
                        f"input_valid_{specialist}",
                        f"invalid_reason_{specialist}",
                        f"score_origin_{specialist}",
                    ]
                )
        else:
            candidates = [
                f"prob_{detector}",
                f"prob_{detector}_raw",
                f"pred_{detector}",
                f"input_valid_{detector}",
                f"invalid_reason_{detector}",
                f"score_origin_{detector}",
                f"threshold_{detector}",
            ]
        for col in candidates:
            if col in table.columns and col not in ordered:
                ordered.append(col)
    remainder = [c for c in table.columns if c not in ordered]
    return table.loc[:, ordered + remainder]


def write_predictions_parquet(path: Path | str, table: pd.DataFrame) -> Path:
    """Atomic write for a non-empty prediction parquet."""

    if table.empty:
        raise PublicationError(f"Refusing to publish empty parquet: {path}")
    return _atomic_parquet_write(path, table)


def write_optional_predictions_parquet(path: Path | str, table: pd.DataFrame) -> Path:
    """Atomic write that allows an empty specialist table (still creates file)."""

    return _atomic_parquet_write(path, table)


def _atomic_parquet_write(path: Path | str, table: pd.DataFrame) -> Path:
    out = Path(path).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    try:
        table.to_parquet(tmp, index=False)
        tmp.replace(out)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    return out


def load_predictions_parquet(path: Path | str) -> pd.DataFrame:
    file_path = Path(path).expanduser().resolve()
    if not file_path.is_file():
        raise PublicationError(f"predictions parquet missing: {file_path}")
    try:
        return pd.read_parquet(file_path)
    except Exception as exc:  # noqa: BLE001
        raise PublicationError(
            f"Could not read predictions parquet: {file_path}: {exc}"
        ) from exc


def build_model_manifest(
    result: BehavioralInferenceResult,
    *,
    config_settings: Mapping[str, Any],
    registry_root: str | None,
    feature_spec_digests: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Build ``model_manifest.json`` payload (no model binary copy)."""

    from behavioral_inference.contracts import MODEL_MANIFEST_SCHEMA_VERSION

    bundles: list[dict[str, Any]] = []
    for meta in result.detector_meta:
        key = meta.specialist_name or meta.detector_name
        digest = None
        if feature_spec_digests is not None:
            digest = feature_spec_digests.get(key)
        bundles.append(
            {
                "family": meta.family,
                "detector_name": meta.detector_name,
                "specialist_name": meta.specialist_name,
                "output_role": meta.output_role,
                "model_version": meta.model_version,
                "model_bundle_dir": meta.model_bundle_dir,
                "model_sha256": meta.model_sha256,
                "feature_spec_sha256": digest,
                "threshold": meta.threshold,
                "threshold_source": meta.threshold_source,
                "routing_mode": meta.routing_mode,
                "n_rows": meta.n_rows,
                "n_scored": meta.n_scored,
                "n_positive": meta.n_positive,
                "extras": dict(meta.extras),
            }
        )

    return {
        "schema_version": MODEL_MANIFEST_SCHEMA_VERSION,
        "registry_root": registry_root,
        "selected_detectors": {
            "dyadic": list(result.selected_detectors_dyadic),
            "focal": list(result.selected_detectors_focal),
        },
        "required_specialist_bundles": sorted(
            {
                m.specialist_name
                for m in result.detector_meta
                if m.specialist_name is not None
            }
        ),
        "thresholds": {k: dict(v) for k, v in result.thresholds.items()},
        "model_identities": {k: dict(v) for k, v in result.model_identities.items()},
        "library_versions": dict(result.metadata.get("library_versions", {})),
        "parity_mode": result.metadata.get("parity_mode"),
        "adapters": dict(result.metadata.get("adapters", {})),
        "dyadic_collapse_policy": (
            dict(result.metadata.get("adapters", {})).get("dyadic_collapse")
        ),
        "smoothing": {
            "nose_to_body": dict(result.routing_diagnostics.get("nose_to_body", {})),
        },
        "inference_settings_echo": dict(config_settings),
        "bundles": bundles,
        "routing_diagnostics": dict(result.routing_diagnostics),
    }


def feature_spec_sha256_for_bundle(bundle_dir: Path | str | None) -> str | None:
    """Hash ``feature_spec.json`` when present beside a model bundle."""

    if bundle_dir is None:
        return None
    path = Path(bundle_dir) / "feature_spec.json"
    if not path.is_file():
        return None
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def collect_feature_spec_digests(result: BehavioralInferenceResult) -> dict[str, str]:
    digests: dict[str, str] = {}
    for meta in result.detector_meta:
        key = meta.specialist_name or meta.detector_name
        sha = feature_spec_sha256_for_bundle(meta.model_bundle_dir)
        if sha is not None:
            digests[key] = sha
    return digests
