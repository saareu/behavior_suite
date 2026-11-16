"""S7 cooperative cancellation, publication gate, and S8 eligibility tests."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from behavioral_inference import (
    FAILED_OR_INCOMPLETE,
    PREDICTIONS_FILENAME,
    QC_FILENAME,
    RUN_META_FILENAME,
    STATUS_CANCELED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    BehavioralInferenceCancelledError,
    BehavioralInferenceRequest,
    PublicationError,
    find_latest_eligible_s7_run,
    is_s7_run_eligible_for_downstream,
    load_predictions_parquet,
    run_behavioral_inference,
    run_behavioral_inference_publication,
    summarize_behavioral_inference_project,
    validate_s7_run_for_downstream,
)
from behavioral_inference.contracts import (
    MODEL_MANIFEST_FILENAME,
    SETTINGS_USED_FILENAME,
    InferenceError,
)
from behavioral_inference.loader import LoadedModel
from tests.behavioral_inference.fake_models import (
    FakeBinaryEstimator,
    make_loaded_model,
    make_two_mouse_feature_frame,
)
from tests.behavioral_inference.test_runner_discovery import (
    _config,
    _patch_models_to_fakes,
    _patch_s6_ok,
    _write_fake_registry,
    _write_s6_run,
)


class _CancelOnNthCheck:
    def __init__(self, n: int) -> None:
        self.n = n
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls >= self.n


def _run_meta(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / RUN_META_FILENAME).read_text(encoding="utf-8"))


def _request(
    session: Path,
    s6: Path,
    registry: Path,
    *,
    timestamp: str,
    dyadic: tuple[str, ...] = ("nose_to_nose",),
    focal: tuple[str, ...] = ("approach",),
) -> BehavioralInferenceRequest:
    return BehavioralInferenceRequest(
        session_root=session,
        s6_run_dir=s6,
        config=_config(
            registry,
            active_detectors_dyadic=dyadic,
            active_detectors_focal=focal,
        ),
        selected_detectors_dyadic=list(dyadic),
        selected_detectors_focal=list(focal),
        run_purpose="test",
        timestamp=timestamp,
    )


def _prepare_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    n_frames: int = 3,
    dyadic: tuple[str, ...] = ("nose_to_nose",),
    focal: tuple[str, ...] = ("approach",),
) -> dict[str, Any]:
    session = tmp_path / "session"
    session.mkdir(parents=True, exist_ok=True)
    registry = tmp_path / "models"
    names = {name: "dyadic" for name in dyadic}
    names.update({name: "focal" for name in focal})
    _write_fake_registry(registry, names=names)
    s6 = _write_s6_run(session, n_frames=n_frames)
    _patch_s6_ok(monkeypatch, s6, session)
    _patch_models_to_fakes(monkeypatch, registry)
    return {
        "session": session,
        "registry": registry,
        "s6": s6,
        "dyadic": dyadic,
        "focal": focal,
    }


def _prediction_fingerprint(path: Path) -> dict[str, Any]:
    table = load_predictions_parquet(path)
    numeric_cols = [
        name
        for name in table.columns
        if pd.api.types.is_numeric_dtype(table[name])
    ]
    return {
        "columns": list(table.columns),
        "n_rows": len(table),
        "detector_name": table["detector_name"].astype(str).tolist(),
        "dense_frame_idx": table["dense_frame_idx"].to_numpy(dtype=np.int64).copy(),
        "focal_id": (
            table["focal_id"].to_numpy(dtype=float).copy()
            if "focal_id" in table.columns
            else None
        ),
        "prediction": table["prediction"].to_numpy(dtype=np.int8).copy(),
        "prob_effective": table["prob_effective"].to_numpy(dtype=float).copy(),
        "input_valid": table["input_valid"].to_numpy(dtype=bool).copy(),
        "numeric": {
            name: table[name].to_numpy(dtype=float).copy() for name in numeric_cols
        },
    }


def test_no_callback_preserves_successful_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = _prepare_session(tmp_path, monkeypatch)
    result = run_behavioral_inference_publication(
        _request(info["session"], info["s6"], info["registry"], timestamp="20260101T100000")
    )
    assert result.success is True
    assert result.status in {STATUS_COMPLETED, "completed_with_warnings"}
    assert result.predictions_path is not None
    assert result.predictions_path.is_file()


def test_false_callback_preserves_successful_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = _prepare_session(tmp_path, monkeypatch)
    result = run_behavioral_inference_publication(
        _request(info["session"], info["s6"], info["registry"], timestamp="20260101T100001"),
        cancellation_requested=lambda: False,
    )
    assert result.success is True
    assert result.predictions_path is not None


def test_scientific_invariance_no_callback_vs_false_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info_a = _prepare_session(tmp_path / "a", monkeypatch)
    baseline = run_behavioral_inference_publication(
        _request(
            info_a["session"],
            info_a["s6"],
            info_a["registry"],
            timestamp="20260101T100010",
        )
    )
    info_b = _prepare_session(tmp_path / "b", monkeypatch)
    false_cb = run_behavioral_inference_publication(
        _request(
            info_b["session"],
            info_b["s6"],
            info_b["registry"],
            timestamp="20260101T100011",
        ),
        cancellation_requested=lambda: False,
    )
    assert baseline.predictions_path is not None
    assert false_cb.predictions_path is not None
    left = _prediction_fingerprint(baseline.predictions_path)
    right = _prediction_fingerprint(false_cb.predictions_path)
    assert left["columns"] == right["columns"]
    assert left["n_rows"] == right["n_rows"]
    assert left["detector_name"] == right["detector_name"]
    np.testing.assert_array_equal(left["dense_frame_idx"], right["dense_frame_idx"])
    np.testing.assert_array_equal(left["prediction"], right["prediction"])
    np.testing.assert_allclose(left["prob_effective"], right["prob_effective"], equal_nan=True)
    np.testing.assert_array_equal(left["input_valid"], right["input_valid"])
    if left["focal_id"] is not None and right["focal_id"] is not None:
        np.testing.assert_allclose(left["focal_id"], right["focal_id"], equal_nan=True)
    for name in left["numeric"]:
        np.testing.assert_allclose(
            left["numeric"][name],
            right["numeric"][name],
            equal_nan=True,
            rtol=0.0,
            atol=0.0,
        )
    assert baseline.qc is not None and false_cb.qc is not None
    assert baseline.qc["structural_ok"] == false_cb.qc["structural_ok"]
    assert baseline.model_manifest is not None and false_cb.model_manifest is not None
    assert [b.get("detector_name") or b.get("specialist_name") for b in baseline.model_manifest["bundles"]] == [
        b.get("detector_name") or b.get("specialist_name") for b in false_cb.model_manifest["bundles"]
    ]


def test_engine_scientific_invariance_with_false_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = _prepare_session(tmp_path, monkeypatch)
    config = _config(info["registry"])
    features = make_two_mouse_feature_frame(n_frames=4)
    models = {
        "nose_to_nose": make_loaded_model(
            "nose_to_nose", "dyadic", ["feat_a", "feat_b"], tmp_path=info["registry"]
        ),
        "approach": make_loaded_model(
            "approach", "focal", ["feat_a", "feat_b"], tmp_path=info["registry"]
        ),
    }
    baseline = run_behavioral_inference(
        features,
        config,
        models=models,
        selected_detectors_dyadic=["nose_to_nose"],
        selected_detectors_focal=["approach"],
    )
    false_cb = run_behavioral_inference(
        features,
        config,
        models=models,
        selected_detectors_dyadic=["nose_to_nose"],
        selected_detectors_focal=["approach"],
        cancellation_requested=lambda: False,
    )
    assert list(baseline.selected_detectors_dyadic) == list(false_cb.selected_detectors_dyadic)
    assert list(baseline.selected_detectors_focal) == list(false_cb.selected_detectors_focal)
    pd.testing.assert_frame_equal(baseline.predictions, false_cb.predictions)
    pd.testing.assert_frame_equal(baseline.diagnostics, false_cb.diagnostics)


def test_engine_scientific_invariance_multi_specialist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No-callback vs always-False must match for contact + nose_to_body specialists."""

    del monkeypatch
    config = _config(tmp_path)
    features = make_two_mouse_feature_frame(n_frames=4, occlusion=[0, 1, 0, 1])
    models = {
        "contact_intact": make_loaded_model(
            "contact_intact", "dyadic", ["feat_a", "feat_b"], tmp_path=tmp_path
        ),
        "contact_occluded": make_loaded_model(
            "contact_occluded", "dyadic", ["feat_a", "feat_b"], tmp_path=tmp_path
        ),
        "nose_body_intact": make_loaded_model(
            "nose_body_intact", "focal", ["feat_a", "feat_b"], tmp_path=tmp_path
        ),
        "nose_body_occ": make_loaded_model(
            "nose_body_occ", "focal", ["feat_a", "feat_b"], tmp_path=tmp_path
        ),
    }
    baseline = run_behavioral_inference(
        features,
        config,
        models=models,
        selected_detectors_dyadic=["contact"],
        selected_detectors_focal=["nose_to_body"],
    )
    false_cb = run_behavioral_inference(
        features,
        config,
        models=models,
        selected_detectors_dyadic=["contact"],
        selected_detectors_focal=["nose_to_body"],
        cancellation_requested=lambda: False,
    )
    assert list(baseline.selected_detectors_dyadic) == ["contact"]
    assert list(false_cb.selected_detectors_dyadic) == ["contact"]
    assert list(baseline.selected_detectors_focal) == ["nose_to_body"]
    assert list(false_cb.selected_detectors_focal) == ["nose_to_body"]
    pd.testing.assert_frame_equal(baseline.predictions, false_cb.predictions)
    pd.testing.assert_frame_equal(baseline.diagnostics, false_cb.diagnostics)
    assert [m.specialist_name for m in baseline.detector_meta] == [
        m.specialist_name for m in false_cb.detector_meta
    ]
    assert baseline.routing_diagnostics == false_cb.routing_diagnostics


def test_cancel_before_run_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = _prepare_session(tmp_path, monkeypatch)
    with pytest.raises(BehavioralInferenceCancelledError):
        run_behavioral_inference_publication(
            _request(info["session"], info["s6"], info["registry"], timestamp="20260101T100100"),
            cancellation_requested=lambda: True,
        )
    root = info["session"] / "behavioral_inference"
    assert not root.exists() or list(root.iterdir()) == []


@pytest.mark.parametrize(
    ("nth", "label"),
    [
        (3, "after_bootstrap"),
        (4, "before_or_during_model_load"),
        (8, "before_or_during_first_detector"),
        (10, "between_or_during_detectors"),
        (14, "after_detectors_before_tables"),
        (16, "after_qc_or_revalidation"),
        (18, "final_publication_gate"),
    ],
)
def test_cancel_at_safe_checkpoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    nth: int,
    label: str,
) -> None:
    del label
    info = _prepare_session(tmp_path, monkeypatch)
    cancel = _CancelOnNthCheck(nth)
    with pytest.raises(BehavioralInferenceCancelledError):
        run_behavioral_inference_publication(
            _request(
                info["session"],
                info["s6"],
                info["registry"],
                timestamp=f"20260101T11{nth:04d}",
            ),
            cancellation_requested=cancel,
        )
    runs = list((info["session"] / "behavioral_inference").iterdir())
    assert len(runs) == 1
    meta = _run_meta(runs[0])
    assert meta["status"] == STATUS_CANCELED
    assert not (runs[0] / PREDICTIONS_FILENAME).exists()
    assert not (runs[0] / QC_FILENAME).exists()
    assert not (runs[0] / MODEL_MANIFEST_FILENAME).exists()
    assert (runs[0] / RUN_META_FILENAME).is_file()
    assert (runs[0] / SETTINGS_USED_FILENAME).is_file()
    assert (runs[0] / "processing_log.txt").is_file()
    err = (meta.get("error") or "").lower()
    assert "cancel" in err


def _loaded_with_estimator(
    name: str,
    family: str,
    tmp_path: Path,
    estimator: FakeBinaryEstimator,
) -> LoadedModel:
    stub = make_loaded_model(name, family, ["feat_a", "feat_b"], tmp_path=tmp_path)
    return LoadedModel(
        bundle=stub.bundle,
        estimator=estimator,
        classes=stub.classes,
        positive_class_index=stub.positive_class_index,
        positive_class_label=stub.positive_class_label,
        n_features_in=stub.n_features_in,
        library_versions=stub.library_versions,
    )


def test_specialist_cancel_before_first_contact_specialist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del monkeypatch
    config = _config(tmp_path)
    features = make_two_mouse_feature_frame(n_frames=3, occlusion=[0, 1, 0])
    calls = {"n": 0}

    class _Counting(FakeBinaryEstimator):
        def predict_proba(self, X: Any) -> np.ndarray:  # type: ignore[override]
            calls["n"] += 1
            return super().predict_proba(X)

    models = {
        "contact_intact": _loaded_with_estimator(
            "contact_intact", "dyadic", tmp_path, _Counting()
        ),
        "contact_occluded": _loaded_with_estimator(
            "contact_occluded", "dyadic", tmp_path, _Counting()
        ),
    }
    with pytest.raises(BehavioralInferenceCancelledError):
        run_behavioral_inference(
            features,
            config,
            models=models,
            selected_detectors_dyadic=["contact"],
            selected_detectors_focal=[],
            cancellation_requested=lambda: True,
        )
    assert calls["n"] == 0


def test_specialist_cancel_between_contact_specialists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancel after first specialist returns; second specialist must not run."""

    del monkeypatch
    config = _config(tmp_path)
    features = make_two_mouse_feature_frame(n_frames=4, occlusion=[0, 1, 0, 1])
    calls: list[str] = []

    class _Named(FakeBinaryEstimator):
        def __init__(self, label: str) -> None:
            super().__init__()
            self.label = label

        def predict_proba(self, X: Any) -> np.ndarray:  # type: ignore[override]
            calls.append(self.label)
            return super().predict_proba(X)

    models = {
        "contact_intact": _loaded_with_estimator(
            "contact_intact", "dyadic", tmp_path, _Named("intact")
        ),
        "contact_occluded": _loaded_with_estimator(
            "contact_occluded", "dyadic", tmp_path, _Named("occluded")
        ),
    }
    with pytest.raises(BehavioralInferenceCancelledError):
        run_behavioral_inference(
            features,
            config,
            models=models,
            selected_detectors_dyadic=["contact"],
            selected_detectors_focal=[],
            cancellation_requested=lambda: len(calls) >= 1,
        )
    assert calls == ["intact"]


def test_specialist_cancel_between_when_second_mask_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Empty second mask must still honor cancel before fusion/publication path."""

    del monkeypatch
    config = _config(tmp_path)
    features = make_two_mouse_feature_frame(n_frames=4, occlusion=[0, 0, 0, 0])
    calls = {"n": 0}

    class _Counting(FakeBinaryEstimator):
        def predict_proba(self, X: Any) -> np.ndarray:  # type: ignore[override]
            calls["n"] += 1
            return super().predict_proba(X)

    models = {
        "contact_intact": _loaded_with_estimator(
            "contact_intact", "dyadic", tmp_path, _Counting()
        ),
        "contact_occluded": _loaded_with_estimator(
            "contact_occluded", "dyadic", tmp_path, _Counting()
        ),
    }
    with pytest.raises(BehavioralInferenceCancelledError):
        run_behavioral_inference(
            features,
            config,
            models=models,
            selected_detectors_dyadic=["contact"],
            selected_detectors_focal=[],
            cancellation_requested=lambda: calls["n"] >= 1,
        )
    assert calls["n"] == 1


def test_specialist_cancel_while_executing_waits_for_predict_proba(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del monkeypatch
    config = _config(tmp_path)
    features = make_two_mouse_feature_frame(n_frames=3, occlusion=[0, 1, 0])
    started = threading.Event()
    release = threading.Event()
    cancel_flag = {"value": False}
    calls: list[str] = []

    class _SlowNamed(FakeBinaryEstimator):
        def __init__(self, label: str) -> None:
            super().__init__()
            self.label = label

        def predict_proba(self, X: Any) -> np.ndarray:  # type: ignore[override]
            calls.append(self.label)
            if self.label == "intact":
                started.set()
                assert release.wait(timeout=5.0)
            return super().predict_proba(X)

    models = {
        "contact_intact": _loaded_with_estimator(
            "contact_intact", "dyadic", tmp_path, _SlowNamed("intact")
        ),
        "contact_occluded": _loaded_with_estimator(
            "contact_occluded", "dyadic", tmp_path, _SlowNamed("occluded")
        ),
    }
    errors: list[BaseException] = []

    def _worker() -> None:
        try:
            run_behavioral_inference(
                features,
                config,
                models=models,
                selected_detectors_dyadic=["contact"],
                selected_detectors_focal=[],
                cancellation_requested=lambda: cancel_flag["value"],
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=_worker)
    thread.start()
    assert started.wait(timeout=5.0)
    cancel_flag["value"] = True
    release.set()
    thread.join(timeout=30.0)
    assert not thread.is_alive()
    assert errors and isinstance(errors[0], BehavioralInferenceCancelledError)
    assert calls == ["intact"]


def test_specialist_ordinary_failure_not_rewritten_as_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """In-flight specialist exception remains InferenceError even if cancel flips."""

    del monkeypatch
    config = _config(tmp_path)
    features = make_two_mouse_feature_frame(n_frames=3, occlusion=[0, 1, 0])
    cancel_flag = {"value": False}

    class _Boom(FakeBinaryEstimator):
        def predict_proba(self, X: Any) -> np.ndarray:  # type: ignore[override]
            cancel_flag["value"] = True
            raise RuntimeError("injected specialist boom")

    models = {
        "contact_intact": _loaded_with_estimator(
            "contact_intact", "dyadic", tmp_path, _Boom()
        ),
        "contact_occluded": _loaded_with_estimator(
            "contact_occluded", "dyadic", tmp_path, FakeBinaryEstimator()
        ),
    }
    with pytest.raises(InferenceError, match="injected specialist boom"):
        run_behavioral_inference(
            features,
            config,
            models=models,
            selected_detectors_dyadic=["contact"],
            selected_detectors_focal=[],
            cancellation_requested=lambda: cancel_flag["value"],
        )
    assert cancel_flag["value"] is True


def test_nose_to_body_specialist_cancel_between_stages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del monkeypatch
    config = _config(tmp_path)
    features = make_two_mouse_feature_frame(n_frames=4, occlusion=[0, 1, 0, 1])
    calls: list[str] = []

    class _Named(FakeBinaryEstimator):
        def __init__(self, label: str) -> None:
            super().__init__()
            self.label = label

        def predict_proba(self, X: Any) -> np.ndarray:  # type: ignore[override]
            calls.append(self.label)
            return super().predict_proba(X)

    models = {
        "nose_body_intact": _loaded_with_estimator(
            "nose_body_intact", "focal", tmp_path, _Named("intact")
        ),
        "nose_body_occ": _loaded_with_estimator(
            "nose_body_occ", "focal", tmp_path, _Named("occ")
        ),
    }
    with pytest.raises(BehavioralInferenceCancelledError):
        run_behavioral_inference(
            features,
            config,
            models=models,
            selected_detectors_dyadic=[],
            selected_detectors_focal=["nose_to_body"],
            cancellation_requested=lambda: len(calls) >= 1,
        )
    assert calls == ["intact"]


def test_cancel_while_detector_running_waits_for_predict_proba(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An in-flight predict_proba finishes; cancel is honored at the next checkpoint."""

    info = _prepare_session(
        tmp_path,
        monkeypatch,
        dyadic=("nose_to_nose", "side_by_side"),
        focal=(),
    )
    started = threading.Event()
    release = threading.Event()
    cancel_flag = {"value": False}

    class _SlowEstimator(FakeBinaryEstimator):
        def predict_proba(self, X: Any) -> np.ndarray:  # type: ignore[override]
            started.set()
            assert release.wait(timeout=5.0)
            return super().predict_proba(X)

    def _fake_load(info_bundle: Any, *, allow_fallback: bool = False) -> Any:
        del allow_fallback
        from behavioral_inference.loader import LoadedModel

        stub = make_loaded_model(
            info_bundle.detector_name,
            info_bundle.family,
            list(info_bundle.feature_cols) or ["feat_a", "feat_b"],
            tmp_path=info["registry"],
            support=info_bundle.support,
        )
        estimator = _SlowEstimator(weight=1.0, bias=0.0)
        return LoadedModel(
            bundle=info_bundle,
            estimator=estimator,
            classes=stub.classes,
            positive_class_index=stub.positive_class_index,
            positive_class_label=stub.positive_class_label,
            n_features_in=stub.n_features_in,
            library_versions=stub.library_versions,
        )

    monkeypatch.setattr("behavioral_inference.engine.load_model_bundle", _fake_load)

    errors: list[BaseException] = []

    def _worker() -> None:
        try:
            run_behavioral_inference_publication(
                _request(
                    info["session"],
                    info["s6"],
                    info["registry"],
                    timestamp="20260101T120050",
                    dyadic=("nose_to_nose", "side_by_side"),
                    focal=(),
                ),
                cancellation_requested=lambda: cancel_flag["value"],
            )
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=_worker)
    thread.start()
    assert started.wait(timeout=5.0)
    cancel_flag["value"] = True
    release.set()
    thread.join(timeout=30.0)
    assert not thread.is_alive()
    assert errors and isinstance(errors[0], BehavioralInferenceCancelledError)
    run_dir = next((info["session"] / "behavioral_inference").iterdir())
    assert _run_meta(run_dir)["status"] == STATUS_CANCELED
    assert not (run_dir / PREDICTIONS_FILENAME).exists()


def test_cancel_after_publication_gate_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = _prepare_session(tmp_path, monkeypatch)
    state = {"in_publication": False}

    def cancel() -> bool:
        return state["in_publication"]

    def publication_started() -> None:
        state["in_publication"] = True

    result = run_behavioral_inference_publication(
        _request(info["session"], info["s6"], info["registry"], timestamp="20260101T120200"),
        cancellation_requested=cancel,
        publication_started=publication_started,
    )
    assert state["in_publication"] is True
    assert result.success is True
    assert result.status in {STATUS_COMPLETED, "completed_with_warnings"}
    assert result.predictions_path is not None
    assert result.predictions_path.is_file()
    meta = _run_meta(result.run_dir)
    assert meta["status"] == result.status


def test_cancel_during_publication_writes_still_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = _prepare_session(tmp_path, monkeypatch)
    state = {"cancel": False}
    import behavioral_inference.runner as runner_mod

    original_write = runner_mod.write_predictions_parquet

    def _write_and_cancel(path: Path, table: pd.DataFrame) -> None:
        state["cancel"] = True
        original_write(path, table)

    monkeypatch.setattr(runner_mod, "write_predictions_parquet", _write_and_cancel)
    result = run_behavioral_inference_publication(
        _request(info["session"], info["s6"], info["registry"], timestamp="20260101T120250"),
        cancellation_requested=lambda: state["cancel"],
    )
    assert state["cancel"] is True
    assert result.success is True
    assert _run_meta(result.run_dir)["status"] in {
        STATUS_COMPLETED,
        "completed_with_warnings",
    }
    assert is_s7_run_eligible_for_downstream(result.run_dir) is True


def test_cancel_after_success_is_harmless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = _prepare_session(tmp_path, monkeypatch)
    cancel = _CancelOnNthCheck(1000)
    result = run_behavioral_inference_publication(
        _request(info["session"], info["s6"], info["registry"], timestamp="20260101T120300"),
        cancellation_requested=cancel,
    )
    assert result.success is True


def test_repeated_cancellation_is_harmless(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = _prepare_session(tmp_path, monkeypatch)
    cancel = _CancelOnNthCheck(3)
    with pytest.raises(BehavioralInferenceCancelledError):
        run_behavioral_inference_publication(
            _request(info["session"], info["s6"], info["registry"], timestamp="20260101T120400"),
            cancellation_requested=cancel,
        )
    with pytest.raises(BehavioralInferenceCancelledError):
        run_behavioral_inference_publication(
            _request(info["session"], info["s6"], info["registry"], timestamp="20260101T120401"),
            cancellation_requested=lambda: True,
        )


def test_canceled_run_rejected_by_s8_eligibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = _prepare_session(tmp_path, monkeypatch)
    with pytest.raises(BehavioralInferenceCancelledError):
        run_behavioral_inference_publication(
            _request(info["session"], info["s6"], info["registry"], timestamp="20260101T130000"),
            cancellation_requested=_CancelOnNthCheck(3),
        )
    run_dir = next((info["session"] / "behavioral_inference").iterdir())
    assert is_s7_run_eligible_for_downstream(run_dir) is False
    with pytest.raises((PublicationError, Exception), match="not completed"):
        validate_s7_run_for_downstream(run_dir)
    summary = summarize_behavioral_inference_project(info["session"])
    assert summary.runs
    assert summary.runs[0].classification == FAILED_OR_INCOMPLETE
    assert summary.eligible_runs == ()
    assert find_latest_eligible_s7_run(info["session"]) is None


def test_canceled_run_does_not_affect_prior_completed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = _prepare_session(tmp_path, monkeypatch)
    completed = run_behavioral_inference_publication(
        _request(info["session"], info["s6"], info["registry"], timestamp="20260101T140000")
    )
    with pytest.raises(BehavioralInferenceCancelledError):
        run_behavioral_inference_publication(
            _request(info["session"], info["s6"], info["registry"], timestamp="20260101T140100"),
            cancellation_requested=_CancelOnNthCheck(3),
        )
    assert completed.predictions_path is not None
    assert completed.predictions_path.is_file()
    assert is_s7_run_eligible_for_downstream(completed.run_dir) is True
    summary = summarize_behavioral_inference_project(info["session"])
    assert summary.eligible_runs
    assert summary.eligible_runs[0].run_id == completed.run_id


def test_ordinary_failure_remains_failed_even_if_cancel_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = _prepare_session(tmp_path, monkeypatch)
    state = {"cancel": False}

    def boom(*_args: object, **_kwargs: object) -> None:
        state["cancel"] = True
        raise PublicationError("injected structural failure")

    monkeypatch.setattr(
        "behavioral_inference.runner.evaluate_behavioral_inference_qc",
        boom,
    )
    with pytest.raises(PublicationError, match="injected structural failure"):
        run_behavioral_inference_publication(
            _request(info["session"], info["s6"], info["registry"], timestamp="20260101T150000"),
            cancellation_requested=lambda: state["cancel"],
        )
    assert state["cancel"] is True
    run_dir = next((info["session"] / "behavioral_inference").iterdir())
    assert _run_meta(run_dir)["status"] == STATUS_FAILED


def test_rerun_after_cancellation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = _prepare_session(tmp_path, monkeypatch)
    with pytest.raises(BehavioralInferenceCancelledError):
        run_behavioral_inference_publication(
            _request(info["session"], info["s6"], info["registry"], timestamp="20260101T160000"),
            cancellation_requested=_CancelOnNthCheck(3),
        )
    result = run_behavioral_inference_publication(
        _request(info["session"], info["s6"], info["registry"], timestamp="20260101T160100")
    )
    assert result.success is True
    assert is_s7_run_eligible_for_downstream(result.run_dir) is True


def test_progress_stages_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = _prepare_session(tmp_path, monkeypatch)
    phases: list[str] = []

    def progress(event: object) -> None:
        phases.append(getattr(event, "phase", ""))

    result = run_behavioral_inference_publication(
        _request(info["session"], info["s6"], info["registry"], timestamp="20260101T170000"),
        progress_callback=progress,
        cancellation_requested=lambda: False,
    )
    assert result.success is True
    assert "validating_sources" in phases
    assert "loading_models" in phases
    assert "preparing_inference" in phases
    assert "running_detector_inference" in phases
    assert "constructing_prediction_tables" in phases
    assert "computing_qc" in phases
    assert "revalidating_sources" in phases
    assert "writing_artifacts" in phases
    assert "completed" in phases


def test_partial_predictions_never_published_on_cancel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = _prepare_session(
        tmp_path,
        monkeypatch,
        dyadic=("nose_to_nose", "side_by_side", "mutually_disengaged"),
        focal=("approach",),
    )
    # Cancel after first detector path is underway / between detectors.
    with pytest.raises(BehavioralInferenceCancelledError):
        run_behavioral_inference_publication(
            _request(
                info["session"],
                info["s6"],
                info["registry"],
                timestamp="20260101T180000",
                dyadic=("nose_to_nose", "side_by_side", "mutually_disengaged"),
                focal=("approach",),
            ),
            cancellation_requested=_CancelOnNthCheck(12),
        )
    run_dir = next((info["session"] / "behavioral_inference").iterdir())
    assert _run_meta(run_dir)["status"] == STATUS_CANCELED
    assert not (run_dir / PREDICTIONS_FILENAME).exists()
    assert not (run_dir / "specialist_predictions.parquet").exists()
    assert not (run_dir / "dyadic_predictions_wide.parquet").exists()
    assert not (run_dir / "focal_predictions_wide.parquet").exists()


def test_cancellation_poll_overhead_is_small(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    info = _prepare_session(tmp_path, monkeypatch, n_frames=6)
    request_base = dict(
        session_root=info["session"],
        s6_run_dir=info["s6"],
        config=_config(info["registry"]),
        selected_detectors_dyadic=["nose_to_nose"],
        selected_detectors_focal=["approach"],
        run_purpose="test",
    )
    t0 = time.perf_counter()
    run_behavioral_inference_publication(
        BehavioralInferenceRequest(**request_base, timestamp="20260101T190000")
    )
    baseline = time.perf_counter() - t0
    t1 = time.perf_counter()
    run_behavioral_inference_publication(
        BehavioralInferenceRequest(**request_base, timestamp="20260101T190001"),
        cancellation_requested=lambda: False,
    )
    with_cb = time.perf_counter() - t1
    # Coarse bound only: cancellation polling must not dominate runtime.
    assert with_cb < baseline * 3.0 + 1.0
