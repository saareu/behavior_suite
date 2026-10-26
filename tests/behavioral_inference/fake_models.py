"""Lightweight fake estimators / LoadedModel builders for Sprint 2 unit tests."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from behavioral_inference.contracts import (
    BUNDLE_FALLBACK,
    BUNDLE_SUPPORTED,
    SERIALIZATION_JOBLIB_XGBOOST,
)
from behavioral_inference.loader import LoadedModel
from behavioral_inference.registry import ModelBundleInfo


class FakeBinaryEstimator:
    """Deterministic binary classifier using a linear score on the first feature."""

    def __init__(
        self,
        *,
        weight: float = 1.0,
        bias: float = 0.0,
        classes: tuple[int, ...] = (0, 1),
    ) -> None:
        self.classes_ = np.asarray(classes)
        self.weight = float(weight)
        self.bias = float(bias)

    def predict_proba(self, X: Any) -> np.ndarray:
        arr = np.asarray(X, dtype=float)
        if arr.ndim != 2:
            raise ValueError("expected 2-D feature matrix")
        # Map first column through a logistic-ish squash into (0, 1).
        raw = self.bias + self.weight * arr[:, 0]
        # Clip extreme values; NaN stays NaN.
        pos = 1.0 / (1.0 + np.exp(-np.clip(raw, -20, 20)))
        # Preserve NaNs from input first feature.
        pos = np.where(np.isnan(arr[:, 0]), np.nan, pos)
        neg = 1.0 - pos
        if list(self.classes_) == [0, 1]:
            return np.column_stack([neg, pos])
        if list(self.classes_) == [1, 0]:
            return np.column_stack([pos, neg])
        raise ValueError(f"unsupported classes {self.classes_}")


def make_bundle_info(
    name: str,
    family: str,
    feature_cols: list[str],
    *,
    support: str = BUNDLE_SUPPORTED,
    tmp_path: Path | None = None,
) -> ModelBundleInfo:
    root = tmp_path if tmp_path is not None else Path(".")
    bundle_dir = root / family / "v1" / name
    return ModelBundleInfo(
        detector_name=name,
        family=family,
        model_version="v1",
        bundle_dir=bundle_dir,
        feature_spec_path=None,
        model_path=bundle_dir / "model.joblib",
        feature_cols=tuple(feature_cols),
        n_features=len(feature_cols),
        model_sha256=f"fake-{name}",
        expected_sha256=None,
        sha256_matches_expected=None,
        serialization_format=SERIALIZATION_JOBLIB_XGBOOST,
        support=support,
        model_family="xgboost",
        train_meta=None,
        notes=(),
    )


def make_loaded_model(
    name: str,
    family: str,
    feature_cols: list[str],
    *,
    weight: float = 1.0,
    bias: float = 0.0,
    support: str | None = None,
    tmp_path: Path | None = None,
) -> LoadedModel:
    if support is None:
        support = BUNDLE_FALLBACK if name == "nose_to_body" else BUNDLE_SUPPORTED
    bundle = make_bundle_info(
        name, family, feature_cols, support=support, tmp_path=tmp_path
    )
    estimator = FakeBinaryEstimator(weight=weight, bias=bias)
    return LoadedModel(
        bundle=bundle,
        estimator=estimator,
        classes=(0, 1),
        positive_class_index=1,
        positive_class_label=1,
        n_features_in=len(feature_cols),
        library_versions={"xgboost": "fake", "sklearn": "fake", "joblib": "fake"},
    )


def make_two_mouse_feature_frame(
    *,
    n_frames: int = 4,
    feat_cols: list[str] | None = None,
    occlusion: list[float] | None = None,
) -> pd.DataFrame:
    """Build a minimal directional S6-like table for K=2 mice."""

    cols = feat_cols or ["feat_a", "feat_b"]
    rows: list[dict[str, Any]] = []
    for frame in range(n_frames):
        occ = 0.0 if occlusion is None else float(occlusion[frame % len(occlusion)])
        for focal, partner in ((0, 1), (1, 0)):
            # Direction-dependent first feature so swap tests can detect reuse.
            feat_a = float(frame + 1) if focal == 0 else float(-(frame + 1))
            rec: dict[str, Any] = {
                "exp_name": "E",
                "sample_idx": frame,
                "dense_frame_idx": frame,
                "time_sec": frame / 120.0,
                "focal_id": focal,
                "partner_id": partner,
                "is_total_occlusion": occ,
            }
            for i, col in enumerate(cols):
                rec[col] = feat_a if i == 0 else float(abs(feat_a) + i)
            rows.append(rec)
    return pd.DataFrame(rows)
