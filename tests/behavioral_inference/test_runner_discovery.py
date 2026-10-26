"""Sprint 3 runner, atomic publication, staleness, and discovery tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from behavioral_inference import (
    COMPLETED_WITH_WARNINGS,
    ELIGIBLE_DOWNSTREAM,
    FAILED_OR_INCOMPLETE,
    PREDICTIONS_FILENAME,
    STALE_MODELS,
    STALE_SOURCE,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    AdapterSettings,
    BehavioralInferenceConfig,
    BehavioralInferenceRequest,
    PublicationError,
    assess_model_source_staleness,
    find_latest_eligible_s7_run,
    is_s7_run_eligible_for_downstream,
    run_behavioral_inference_publication,
    summarize_behavioral_inference_project,
    validate_s7_run_for_downstream,
)
from behavioral_inference.config import DEFAULT_WORKING_THRESHOLDS
from combined_features.config import ModelRegistrySettings
from tests.behavioral_inference.fake_models import (
    make_loaded_model,
    make_two_mouse_feature_frame,
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_fake_registry(root: Path, *, names: dict[str, str]) -> dict[str, str]:
    """Write tiny joblib-like bytes + feature_spec; return name→sha256."""

    digests: dict[str, str] = {}
    for name, family in names.items():
        bundle = root / family / "v1" / name
        bundle.mkdir(parents=True, exist_ok=True)
        payload = f"fake-model::{family}::{name}".encode()
        (bundle / "model.joblib").write_bytes(payload)
        digests[name] = _sha256_bytes(payload)
        (bundle / "feature_spec.json").write_text(
            json.dumps(
                {
                    "detector_name": name,
                    "feature_cols": ["feat_a", "feat_b"],
                    "selected_feature_cols": ["feat_a", "feat_b"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
    return digests


def _write_s6_run(
    session: Path,
    *,
    run_id: str = "combined_features__20260101T000000",
    n_frames: int = 3,
    features: pd.DataFrame | None = None,
) -> Path:
    s6_root = session / "combined_features" / run_id
    s6_root.mkdir(parents=True, exist_ok=True)
    frame = features if features is not None else make_two_mouse_feature_frame(n_frames=n_frames)
    # Stamp minimal S6 identity columns expected by load_dyadic_features_table.
    table = frame.copy()
    if "orig_frame_idx" not in table.columns:
        table["orig_frame_idx"] = table["dense_frame_idx"]
    table["schema_version"] = "dyadic_features_full_v1"
    table["session_root"] = str(session.resolve())
    table["source_s4_run_id"] = "pose_finalization__x"
    table["source_s5_run_id"] = "blob_features__x"
    identity = [
        "schema_version",
        "session_root",
        "focal_id",
        "partner_id",
        "sample_idx",
        "dense_frame_idx",
        "time_sec",
        "orig_frame_idx",
        "source_s4_run_id",
        "source_s5_run_id",
    ]
    ordered = identity + [c for c in table.columns if c not in identity]
    table = table.loc[:, ordered]
    parquet = s6_root / "dyadic_features_full.parquet"
    table.to_parquet(parquet, index=False)
    feat_sha = hashlib.sha256(parquet.read_bytes()).hexdigest()
    meta = {
        "schema_version": "combined_features_run_meta_v1",
        "run_id": run_id,
        "status": "completed",
        "session_root": str(session.resolve()),
        "source_s4": {
            "run_dir": str(session / "pose_finalization" / "pose_finalization__x"),
            "run_id": "pose_finalization__x",
            "session_root": str(session.resolve()),
            "final_pose": str(session / "pose_finalization" / "pose_finalization__x" / "final_pose.parquet"),
            "final_pose_sha256": "a" * 64,
            "prepared_sync": str(session / "preprocess" / "prepared_sync.json"),
            "prepared_sync_sha256": "b" * 64,
        },
        "source_s5": {
            "run_dir": str(session / "blob_features" / "blob_features__x"),
            "run_id": "blob_features__x",
            "session_root": str(session.resolve()),
            "blob_features": str(
                session / "blob_features" / "blob_features__x" / "blob_features.parquet"
            ),
            "blob_features_sha256": "c" * 64,
            "prepared_sync_sha256": "b" * 64,
        },
        "outputs": {
            "dyadic_features_parquet": str(parquet),
        },
    }
    (s6_root / "run_meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    (s6_root / "qc.json").write_text(
        json.dumps({"schema_version": "combined_features_qc_v1", "structural_ok": True})
        + "\n",
        encoding="utf-8",
    )
    (s6_root / "model_compat.json").write_text("{}\n", encoding="utf-8")
    (s6_root / "settings_used.yaml").write_text("schema_version: x\n", encoding="utf-8")
    (s6_root / "processing_log.txt").write_text("ok\n", encoding="utf-8")
    # Store sha for tests.
    (s6_root / "_features_sha.txt").write_text(feat_sha, encoding="utf-8")
    return s6_root


def _patch_s6_ok(monkeypatch: pytest.MonkeyPatch, s6_run_dir: Path, session: Path) -> None:
    feat = s6_run_dir / "dyadic_features_full.parquet"
    feat_sha = hashlib.sha256(feat.read_bytes()).hexdigest()

    def _validate(_path: Any) -> dict[str, Any]:
        return {
            "s6_run_dir": Path(s6_run_dir).resolve(),
            "status": "completed",
            "session_root": session.resolve(),
            "s4_run_dir": session / "pose_finalization" / "pose_finalization__x",
            "s5_run_dir": session / "blob_features" / "blob_features__x",
            "prepared_sync_sha256": "b" * 64,
            "final_pose_sha256": "a" * 64,
            "blob_features_sha256": "c" * 64,
            "s6_run_id": s6_run_dir.name,
        }

    monkeypatch.setattr(
        "behavioral_inference.runner.validate_s6_run_for_downstream",
        _validate,
    )
    monkeypatch.setattr(
        "behavioral_inference.runner.is_s6_run_eligible_for_downstream",
        lambda _p: True,
    )
    monkeypatch.setattr(
        "behavioral_inference.run_discovery.is_s6_run_eligible_for_downstream",
        lambda _p: True,
    )
    monkeypatch.setattr(
        "combined_features.runner.validate_s6_run_for_downstream",
        _validate,
    )
    monkeypatch.setattr(
        "combined_features.runner.is_s6_run_eligible_for_downstream",
        lambda _p: True,
    )
    # Keep feat_sha referenced for clarity in failure messages.
    assert feat_sha


def _patch_models_to_fakes(monkeypatch: pytest.MonkeyPatch, registry: Path) -> None:
    """Bypass joblib unpickle; keep discovered digests/paths from the registry."""

    from behavioral_inference.loader import LoadedModel

    def _fake_load(info: Any, *, allow_fallback: bool = False) -> LoadedModel:
        # Use discovered ModelBundleInfo (real SHA-256 / paths) with a fake estimator.
        stub = make_loaded_model(
            info.detector_name,
            info.family,
            list(info.feature_cols) or ["feat_a", "feat_b"],
            tmp_path=registry,
            support=info.support,
        )
        return LoadedModel(
            bundle=info,
            estimator=stub.estimator,
            classes=stub.classes,
            positive_class_index=stub.positive_class_index,
            positive_class_label=stub.positive_class_label,
            n_features_in=stub.n_features_in,
            library_versions=stub.library_versions,
        )

    monkeypatch.setattr(
        "behavioral_inference.engine.load_model_bundle",
        _fake_load,
    )
    monkeypatch.setattr(
        "behavioral_inference.loader.load_model_bundle",
        _fake_load,
    )


def _config(registry: Path, **kwargs: Any) -> BehavioralInferenceConfig:
    defaults = dict(
        model_registry=ModelRegistrySettings(root=registry).validated(),
        active_detectors_dyadic=("nose_to_nose",),
        active_detectors_focal=("approach",),
        working_thresholds=dict(DEFAULT_WORKING_THRESHOLDS),
        adapters=AdapterSettings(emit_wide_views=True),
    )
    defaults.update(kwargs)
    return BehavioralInferenceConfig(**defaults).validated()


def test_atomic_publication_and_ordinary_warning_eligibility(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = tmp_path / "session"
    session.mkdir()
    registry = tmp_path / "models"
    digests = _write_fake_registry(
        registry,
        names={"nose_to_nose": "dyadic", "approach": "focal"},
    )
    s6 = _write_s6_run(session, n_frames=2)
    _patch_s6_ok(monkeypatch, s6, session)
    _patch_models_to_fakes(monkeypatch, registry)

    # Force a soft warning via raw_framewise parity (disables smoother; approach only).
    config = _config(registry, parity_mode="legacy_parity")
    result = run_behavioral_inference_publication(
        BehavioralInferenceRequest(
            session_root=session,
            s6_run_dir=s6,
            config=config,
            selected_detectors_dyadic=["nose_to_nose"],
            selected_detectors_focal=["approach"],
            timestamp="20260101T120000",
            run_purpose="test",
        )
    )
    assert result.success
    assert result.status in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS)
    assert result.predictions_path is not None
    assert result.predictions_path.is_file()
    assert (result.run_dir / "run_meta.json").is_file()
    assert (result.run_dir / "settings_used.yaml").is_file()
    assert (result.run_dir / "qc.json").is_file()
    assert (result.run_dir / "model_manifest.json").is_file()
    assert (result.run_dir / "processing_log.txt").is_file()
    assert (result.run_dir / "dyadic_predictions_wide.parquet").is_file()
    assert (result.run_dir / "focal_predictions_wide.parquet").is_file()

    assert is_s7_run_eligible_for_downstream(result.run_dir)
    summary = summarize_behavioral_inference_project(session)
    assert summary.eligible_runs
    assert summary.eligible_runs[0].run_id == result.run_id
    assert summary.eligible_runs[0].classification in (
        ELIGIBLE_DOWNSTREAM,
        COMPLETED_WITH_WARNINGS,
    )
    # Digests recorded.
    manifest = json.loads((result.run_dir / "model_manifest.json").read_text(encoding="utf-8"))
    by_name = {
        (b.get("specialist_name") or b.get("detector_name")): b
        for b in manifest["bundles"]
    }
    assert by_name["nose_to_nose"]["model_sha256"] == digests["nose_to_nose"]


def test_failure_without_eligible_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = tmp_path / "session"
    session.mkdir()
    registry = tmp_path / "models"
    _write_fake_registry(registry, names={"nose_to_nose": "dyadic"})
    s6 = _write_s6_run(session, n_frames=2)
    _patch_s6_ok(monkeypatch, s6, session)
    _patch_models_to_fakes(monkeypatch, registry)

    # Missing required model for selected focal detector → fail before publish.
    config = _config(
        registry,
        active_detectors_dyadic=("nose_to_nose",),
        active_detectors_focal=("approach",),
    )
    with pytest.raises(PublicationError):
        run_behavioral_inference_publication(
            BehavioralInferenceRequest(
                session_root=session,
                s6_run_dir=s6,
                config=config,
                selected_detectors_dyadic=["nose_to_nose"],
                selected_detectors_focal=["approach"],
                timestamp="20260101T130000",
            )
        )
    run_dir = session / "behavioral_inference" / "behavioral_inference__20260101T130000"
    assert run_dir.is_dir()
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] == STATUS_FAILED
    assert not (run_dir / PREDICTIONS_FILENAME).exists()
    assert not (run_dir / "qc.json").exists()
    project = summarize_behavioral_inference_project(session)
    failed = [r for r in project.runs if r.run_id.endswith("130000")]
    assert failed
    assert failed[0].classification == FAILED_OR_INCOMPLETE


def test_upstream_s6_staleness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = tmp_path / "session"
    session.mkdir()
    registry = tmp_path / "models"
    _write_fake_registry(
        registry, names={"nose_to_nose": "dyadic", "approach": "focal"}
    )
    s6 = _write_s6_run(session, n_frames=2)
    _patch_s6_ok(monkeypatch, s6, session)
    _patch_models_to_fakes(monkeypatch, registry)
    config = _config(registry)
    published = run_behavioral_inference_publication(
        BehavioralInferenceRequest(
            session_root=session,
            s6_run_dir=s6,
            config=config,
            selected_detectors_dyadic=["nose_to_nose"],
            selected_detectors_focal=["approach"],
            timestamp="20260101T140000",
        )
    )
    # Historical status remains completed; eligibility flips when S6 goes stale.
    monkeypatch.setattr(
        "behavioral_inference.runner.is_s6_run_eligible_for_downstream",
        lambda _p: False,
    )
    monkeypatch.setattr(
        "behavioral_inference.run_discovery.is_s6_run_eligible_for_downstream",
        lambda _p: False,
    )
    assert not is_s7_run_eligible_for_downstream(published.run_dir)
    meta = json.loads((published.run_dir / "run_meta.json").read_text(encoding="utf-8"))
    assert meta["status"] in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS)
    project = summarize_behavioral_inference_project(session)
    run = next(r for r in project.runs if r.run_id == published.run_id)
    assert run.classification == STALE_SOURCE
    assert find_latest_eligible_s7_run(session) is None


def test_missing_or_changed_model_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = tmp_path / "session"
    session.mkdir()
    registry = tmp_path / "models"
    _write_fake_registry(
        registry, names={"nose_to_nose": "dyadic", "approach": "focal"}
    )
    s6 = _write_s6_run(session, n_frames=2)
    _patch_s6_ok(monkeypatch, s6, session)
    _patch_models_to_fakes(monkeypatch, registry)
    published = run_behavioral_inference_publication(
        BehavioralInferenceRequest(
            session_root=session,
            s6_run_dir=s6,
            config=_config(registry),
            selected_detectors_dyadic=["nose_to_nose"],
            selected_detectors_focal=["approach"],
            timestamp="20260101T150000",
        )
    )
    # Change model bytes → digest mismatch → stale_models.
    model_path = registry / "dyadic" / "v1" / "nose_to_nose" / "model.joblib"
    model_path.write_bytes(b"tampered-model-bytes")
    info = assess_model_source_staleness(published.run_dir)
    assert info["stale"] is True
    assert any("SHA256 changed" in r for r in info["reasons"])
    assert not is_s7_run_eligible_for_downstream(published.run_dir)
    project = summarize_behavioral_inference_project(session)
    run = next(r for r in project.runs if r.run_id == published.run_id)
    assert run.classification == STALE_MODELS

    # Restoring identical digest clears model staleness (moved-registry policy).
    model_path.write_bytes(b"fake-model::dyadic::nose_to_nose")
    info_ok = assess_model_source_staleness(published.run_dir)
    assert info_ok["stale"] is False

    # Deleted recorded model.joblib → stale_models (relocated/removed registry).
    model_path.unlink()
    info_missing = assess_model_source_staleness(published.run_dir)
    assert info_missing["stale"] is True
    assert any("missing" in r for r in info_missing["reasons"])
    project_missing = summarize_behavioral_inference_project(session)
    run_missing = next(
        r for r in project_missing.runs if r.run_id == published.run_id
    )
    assert run_missing.classification == STALE_MODELS
    model_path.write_bytes(b"fake-model::dyadic::nose_to_nose")
    assert assess_model_source_staleness(published.run_dir)["stale"] is False


def test_deterministic_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = tmp_path / "session"
    session.mkdir()
    registry = tmp_path / "models"
    _write_fake_registry(
        registry, names={"nose_to_nose": "dyadic", "approach": "focal"}
    )
    s6 = _write_s6_run(session, n_frames=2)
    _patch_s6_ok(monkeypatch, s6, session)
    _patch_models_to_fakes(monkeypatch, registry)
    config = _config(registry)
    a = run_behavioral_inference_publication(
        BehavioralInferenceRequest(
            session_root=session,
            s6_run_dir=s6,
            config=config,
            selected_detectors_dyadic=["nose_to_nose"],
            selected_detectors_focal=["approach"],
            timestamp="20260101T160000",
            output_root=tmp_path / "out_a",
        )
    )
    b = run_behavioral_inference_publication(
        BehavioralInferenceRequest(
            session_root=session,
            s6_run_dir=s6,
            config=config,
            selected_detectors_dyadic=["nose_to_nose"],
            selected_detectors_focal=["approach"],
            timestamp="20260101T160001",
            output_root=tmp_path / "out_b",
        )
    )
    pa = pd.read_parquet(a.predictions_path)
    pb = pd.read_parquet(b.predictions_path)
    cols = [
        c
        for c in pa.columns
        if c
        not in (
            "source_s6_run_id",
            "session_root",
            "model_bundle_dir",
        )
    ]
    pd.testing.assert_frame_equal(
        pa.loc[:, cols].reset_index(drop=True),
        pb.loc[:, cols].reset_index(drop=True),
        check_dtype=False,
    )


def test_completed_with_warnings_still_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = tmp_path / "session"
    session.mkdir()
    registry = tmp_path / "models"
    _write_fake_registry(
        registry,
        names={
            "contact_intact": "dyadic",
            "contact_occluded": "dyadic",
            "approach": "focal",
        },
    )
    # Invalid occlusion → warning but structural OK.
    features = make_two_mouse_feature_frame(n_frames=2, occlusion=[0, 2])
    s6 = _write_s6_run(session, n_frames=2, features=features)
    _patch_s6_ok(monkeypatch, s6, session)
    _patch_models_to_fakes(monkeypatch, registry)
    config = _config(
        registry,
        active_detectors_dyadic=("contact",),
        active_detectors_focal=("approach",),
    )
    result = run_behavioral_inference_publication(
        BehavioralInferenceRequest(
            session_root=session,
            s6_run_dir=s6,
            config=config,
            selected_detectors_dyadic=["contact"],
            selected_detectors_focal=["approach"],
            timestamp="20260101T170000",
        )
    )
    assert result.status == STATUS_COMPLETED_WITH_WARNINGS
    assert result.warnings
    assert is_s7_run_eligible_for_downstream(result.run_dir)
    details = validate_s7_run_for_downstream(result.run_dir)
    assert details["status"] == STATUS_COMPLETED_WITH_WARNINGS
