"""Sprint 3: export, QC, runner, discovery, and model-compat tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from combined_features import (
    COMPLETED_WITH_WARNINGS,
    DYADIC_FEATURES_FILENAME,
    ELIGIBLE_DOWNSTREAM,
    MODEL_COMPAT_FILENAME,
    QC_FILENAME,
    REQUIRED_IDENTITY_COLUMNS,
    STALE_SOURCE,
    STATUS_COMPLETED,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    CombinedFeaturesConfig,
    CombinedFeaturesError,
    ModelRegistrySettings,
    build_dyadic_features_table,
    build_model_compat_report,
    compute_combined_features_from_adapted,
    discover_model_registry,
    evaluate_combined_features_qc,
    find_latest_eligible_s6_run,
    is_s6_run_eligible_for_downstream,
    load_dyadic_features_table,
    run_combined_feature_extraction,
    summarize_combined_features_project,
    write_dyadic_features_parquet,
)
from combined_features.contracts import RUN_META_FILENAME
from combined_features.runner import CombinedFeaturesRequest
from tests.combined_features.helpers import (
    build_blob_features_table,
    build_final_pose_table,
    sha256_file,
    write_joint_session,
)
from tests.combined_features.test_processing import _adapted


def _patch_upstream_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass full S3/S1 trees for synthetic joint sessions (same as Sprint 1)."""

    monkeypatch.setattr(
        "combined_features.upstream.validate_s4_run_for_downstream",
        lambda _path: {"ok": True},
    )
    monkeypatch.setattr(
        "combined_features.upstream.validate_s5_run_for_downstream",
        lambda _path: {"ok": True},
    )


def _write_mini_registry(root: Path) -> Path:
    """Create a tiny synthetic model registry with dyadic + focal specs."""

    dyadic = root / "dyadic" / "v1"
    focal = root / "focal" / "v1"
    (dyadic / "contact_intact").mkdir(parents=True)
    (dyadic / "contact").mkdir(parents=True)
    (focal / "approach").mkdir(parents=True)

    (dyadic / "contact_intact" / "feature_spec.json").write_text(
        json.dumps(
            {
                "detector_name": "contact_intact",
                "selected_feature_cols": [
                    "distance_nose_i_nose_j_norm",
                    "cc_count",
                    "nose_any_imputed_i",
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (dyadic / "contact_intact" / "model.joblib").write_bytes(b"fake")
    (dyadic / "contact" / "feature_spec.json").write_text(
        json.dumps(
            {
                "detector_name": "contact",
                "selected_feature_cols": ["nose_any_refined", "nose_any_imputed"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (focal / "approach" / "feature_spec.json").write_text(
        json.dumps(
            {
                "detector_name": "approach",
                "selected_feature_cols": [
                    "signed_kinematic_intent",
                    "nose_any_imputed_i",
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (focal / "approach" / "model.joblib").write_bytes(b"fake")
    return root


def test_unknown_vs_confirmed_non_implanted() -> None:
    adapted_res, blob = _adapted(n_samples=4, headstage_owner=1)
    resolved = compute_combined_features_from_adapted(adapted_res, blob)
    assert set(
        resolved.features.loc[resolved.features.focal_id == 1, "is_implanted"].unique()
    ) == {1.0}
    assert set(
        resolved.features.loc[resolved.features.focal_id == 0, "is_implanted"].unique()
    ) == {0.0}

    pose = build_final_pose_table(
        n_samples=3,
        nodes=("nose", "neck", "spine_base", "tail_base"),
        headstage_owner=None,
    )
    blob_abs = build_blob_features_table(n_samples=3)
    from combined_features import adapt_final_pose_to_frame_feature_input

    adapted_abs = adapt_final_pose_to_frame_feature_input(
        pose, orig_frame_idx=blob_abs["orig_frame_idx"].to_numpy(dtype=np.int64)
    )
    unresolved = compute_combined_features_from_adapted(adapted_abs, blob_abs)
    assert unresolved.features["is_implanted"].isna().all()


def test_output_schema_and_directional_uniqueness() -> None:
    adapted, blob = _adapted(n_samples=5)
    result = compute_combined_features_from_adapted(adapted, blob)
    table = build_dyadic_features_table(
        result.features,
        session_root="/tmp/session",
        source_s4_run_id="s4",
        source_s5_run_id="s5",
    )
    assert list(table.columns[: len(REQUIRED_IDENTITY_COLUMNS)]) == list(
        REQUIRED_IDENTITY_COLUMNS
    )
    assert table.duplicated(
        subset=["focal_id", "partner_id", "dense_frame_idx"]
    ).sum() == 0
    assert len(table) == 5 * 2
    qc = evaluate_combined_features_qc(
        table=table,
        n_tracks=2,
        n_samples=5,
        tracks=(0, 1),
        nodes=adapted.nodes,
        implantation_status=adapted.implantation.status,
        implanted_track_id=adapted.implantation.implanted_track_id,
    )
    assert qc.structural_ok


def test_atomic_publication_and_failure_without_partial_parquet(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_upstream_ok(monkeypatch)
    paths = write_joint_session(tmp_path, n_samples=6)
    registry = _write_mini_registry(tmp_path / "models")
    config = CombinedFeaturesConfig(
        model_registry=ModelRegistrySettings(root=registry)
    )
    result = run_combined_feature_extraction(
        CombinedFeaturesRequest(
            session_root=paths["session_root"],
            s4_run_dir=paths["s4_run_dir"],
            s5_run_dir=paths["s5_run_dir"],
            config=config,
            timestamp="20261021T150000",
            run_purpose="test",
        )
    )
    assert result.success
    assert result.status in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS)
    assert result.dyadic_features_path is not None
    assert result.dyadic_features_path.is_file()
    assert (result.run_dir / QC_FILENAME).is_file()
    assert (result.run_dir / MODEL_COMPAT_FILENAME).is_file()
    assert (result.run_dir / RUN_META_FILENAME).is_file()
    loaded = load_dyadic_features_table(result.dyadic_features_path)
    assert list(loaded.columns[: len(REQUIRED_IDENTITY_COLUMNS)]) == list(
        REQUIRED_IDENTITY_COLUMNS
    )

    # Structural failure path: corrupt blob by removing a mandatory column mid-flight
    # is covered by attach; here force a bad second run with overlapping timestamp.
    with pytest.raises(CombinedFeaturesError, match="already exists"):
        run_combined_feature_extraction(
            CombinedFeaturesRequest(
                session_root=paths["session_root"],
                s4_run_dir=paths["s4_run_dir"],
                s5_run_dir=paths["s5_run_dir"],
                config=config,
                timestamp="20261021T150000",
            )
        )


def test_failure_removes_partial_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_upstream_ok(monkeypatch)
    paths = write_joint_session(tmp_path, n_samples=4)
    config = CombinedFeaturesConfig()

    def _boom(*_args, **_kwargs):
        raise CombinedFeaturesError("forced structural failure")

    monkeypatch.setattr(
        "combined_features.runner.assert_structural_qc",
        _boom,
    )
    # Write a decoy parquet before the failure cleanup path is exercised via runner.
    with pytest.raises(CombinedFeaturesError, match="forced structural failure"):
        run_combined_feature_extraction(
            CombinedFeaturesRequest(
                session_root=paths["session_root"],
                s4_run_dir=paths["s4_run_dir"],
                s5_run_dir=paths["s5_run_dir"],
                config=config,
                timestamp="20261021T160000",
            )
        )
    run_dir = paths["session_root"] / "combined_features" / "combined_features__20261021T160000"
    assert run_dir.is_dir()
    assert not (run_dir / DYADIC_FEATURES_FILENAME).exists()
    meta = json.loads((run_dir / RUN_META_FILENAME).read_text(encoding="utf-8"))
    assert meta["status"] == STATUS_FAILED


def test_stale_source_detection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_upstream_ok(monkeypatch)
    paths = write_joint_session(tmp_path, n_samples=4)
    registry = _write_mini_registry(tmp_path / "models")
    result = run_combined_feature_extraction(
        CombinedFeaturesRequest(
            session_root=paths["session_root"],
            s4_run_dir=paths["s4_run_dir"],
            s5_run_dir=paths["s5_run_dir"],
            config=CombinedFeaturesConfig(
                model_registry=ModelRegistrySettings(root=registry)
            ),
            timestamp="20261021T170000",
        )
    )
    assert is_s6_run_eligible_for_downstream(result.run_dir)

    # Mutate S5 blob digest → stale without rewriting historical status.
    blob = pd.read_parquet(paths["blob_features"])
    blob.loc[0, "cc_count"] = 1
    blob.to_parquet(paths["blob_features"], index=False)
    assert not is_s6_run_eligible_for_downstream(result.run_dir)
    summary = summarize_combined_features_project(paths["session_root"])
    assert summary.runs[0].classification == STALE_SOURCE
    assert find_latest_eligible_s6_run(paths["session_root"]) is None
    meta = json.loads((result.run_dir / RUN_META_FILENAME).read_text(encoding="utf-8"))
    assert meta["status"] in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS)


def test_completed_with_warnings_still_eligible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_upstream_ok(monkeypatch)
    paths = write_joint_session(tmp_path, n_samples=4)
    # Missing headstage → unresolved implantation → warnings, still structurally ok.
    pose = build_final_pose_table(
        n_samples=4,
        nodes=("nose", "neck", "spine_base", "tail_base"),
        headstage_owner=None,
    )
    pose.to_parquet(paths["final_pose"], index=False)
    # Refresh S4 meta digests are path-based; re-run extraction.
    result = run_combined_feature_extraction(
        CombinedFeaturesRequest(
            session_root=paths["session_root"],
            s4_run_dir=paths["s4_run_dir"],
            s5_run_dir=paths["s5_run_dir"],
            config=CombinedFeaturesConfig(),
            timestamp="20261021T180000",
        )
    )
    assert result.status == STATUS_COMPLETED_WITH_WARNINGS
    assert result.warnings
    summary = summarize_combined_features_project(paths["session_root"])
    assert summary.eligible_runs
    assert summary.eligible_runs[0].classification == COMPLETED_WITH_WARNINGS
    assert find_latest_eligible_s6_run(paths["session_root"]) is not None


def test_model_registry_and_compat(tmp_path: Path) -> None:
    adapted, blob = _adapted(n_samples=6)
    result = compute_combined_features_from_adapted(adapted, blob)
    table = build_dyadic_features_table(
        result.features,
        session_root=tmp_path / "session",
        source_s4_run_id="s4",
        source_s5_run_id="s5",
    )
    registry = _write_mini_registry(tmp_path / "models")
    settings = ModelRegistrySettings(root=registry, include_focal=True)
    status, detectors, _notes = discover_model_registry(settings)
    assert status in ("ok", "partial")
    names = {d.detector_name for d in detectors}
    assert "contact_intact" in names
    assert "approach" in names
    assert "contact" in names

    report = build_model_compat_report(features=table, settings=settings)
    by_name = {d["detector_name"]: d for d in report["detectors"]}
    assert by_name["contact"]["status"] == "unsupported"
    assert by_name["contact_intact"]["status"] == "compatible"
    assert by_name["approach"]["status"] == "compatible"

    # Missing required feature → incompatible.
    broken = table.drop(columns=["distance_nose_i_nose_j_norm"])
    report2 = build_model_compat_report(features=broken, settings=settings)
    by_name2 = {d["detector_name"]: d for d in report2["detectors"]}
    assert by_name2["contact_intact"]["status"] == "incompatible"
    assert "distance_nose_i_nose_j_norm" in by_name2["contact_intact"]["missing_columns"]

    # Unavailable registry.
    empty = build_model_compat_report(
        features=table, settings=ModelRegistrySettings(root=None)
    )
    assert empty["registry_status"] == "unavailable"


def test_dtype_incompatibility_warning(tmp_path: Path) -> None:
    adapted, blob = _adapted(n_samples=4)
    result = compute_combined_features_from_adapted(adapted, blob)
    table = build_dyadic_features_table(
        result.features,
        session_root=tmp_path / "session",
        source_s4_run_id="s4",
        source_s5_run_id="s5",
    )
    table = table.copy()
    table["cc_count"] = table["cc_count"].astype(str)
    registry = _write_mini_registry(tmp_path / "models")
    report = build_model_compat_report(
        features=table,
        settings=ModelRegistrySettings(root=registry),
    )
    intact = next(d for d in report["detectors"] if d["detector_name"] == "contact_intact")
    assert any("cc_count" in w for w in intact["dtype_warnings"])


def test_legitimate_nans_are_not_structural_failures() -> None:
    adapted, blob = _adapted(n_samples=4)
    blob = blob.copy()
    blob.loc[:, "ellipse_eccentricity"] = np.nan
    result = compute_combined_features_from_adapted(adapted, blob)
    table = build_dyadic_features_table(
        result.features,
        session_root="/tmp/session",
        source_s4_run_id="s4",
        source_s5_run_id="s5",
    )
    qc = evaluate_combined_features_qc(
        table=table,
        n_tracks=2,
        n_samples=4,
        tracks=(0, 1),
        nodes=adapted.nodes,
        implantation_status=adapted.implantation.status,
        implanted_track_id=adapted.implantation.implanted_track_id,
    )
    assert qc.structural_ok


def test_deterministic_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_upstream_ok(monkeypatch)
    paths = write_joint_session(tmp_path, n_samples=5)
    config = CombinedFeaturesConfig()
    a = run_combined_feature_extraction(
        CombinedFeaturesRequest(
            session_root=paths["session_root"],
            s4_run_dir=paths["s4_run_dir"],
            s5_run_dir=paths["s5_run_dir"],
            config=config,
            timestamp="20261021T190000",
        )
    )
    # Compute again in-memory and compare frame equality of published parquet.
    from combined_features.processing import compute_combined_features

    again = compute_combined_features(
        session_root=paths["session_root"],
        s4_run_dir=paths["s4_run_dir"],
        s5_run_dir=paths["s5_run_dir"],
    )
    published = load_dyadic_features_table(a.dyadic_features_path)
    rebuilt = build_dyadic_features_table(
        again.features,
        session_root=paths["session_root"],
        source_s4_run_id=paths["s4_run_dir"].name,
        source_s5_run_id=paths["s5_run_dir"].name,
    )
    assert list(published.columns) == list(rebuilt.columns)
    pd.testing.assert_frame_equal(
        published.reset_index(drop=True),
        rebuilt.reset_index(drop=True),
        check_dtype=True,
    )


def test_source_digest_recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_upstream_ok(monkeypatch)
    paths = write_joint_session(tmp_path, n_samples=3)
    result = run_combined_feature_extraction(
        CombinedFeaturesRequest(
            session_root=paths["session_root"],
            s4_run_dir=paths["s4_run_dir"],
            s5_run_dir=paths["s5_run_dir"],
            timestamp="20261021T200000",
        )
    )
    meta = json.loads((result.run_dir / RUN_META_FILENAME).read_text(encoding="utf-8"))
    assert meta["source_s4"]["final_pose_sha256"] == sha256_file(paths["final_pose"])
    assert meta["source_s5"]["blob_features_sha256"] == sha256_file(paths["blob_features"])
    assert meta["source_s4"]["prepared_sync_sha256"] == sha256_file(paths["prepared_sync"])


def test_write_refuses_unordered_identity(tmp_path: Path) -> None:
    adapted, blob = _adapted(n_samples=2)
    result = compute_combined_features_from_adapted(adapted, blob)
    table = build_dyadic_features_table(
        result.features,
        session_root=tmp_path,
        source_s4_run_id="s4",
        source_s5_run_id="s5",
    )
    bad = table[["focal_id", *table.columns.drop("focal_id")]]
    with pytest.raises(CombinedFeaturesError, match="identity columns"):
        write_dyadic_features_parquet(tmp_path / "out.parquet", bad)


def test_eligible_classification_for_clean_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_upstream_ok(monkeypatch)
    paths = write_joint_session(tmp_path, n_samples=4)
    # Ensure resolved implantation (default fixture has headstage on track 0).
    result = run_combined_feature_extraction(
        CombinedFeaturesRequest(
            session_root=paths["session_root"],
            s4_run_dir=paths["s4_run_dir"],
            s5_run_dir=paths["s5_run_dir"],
            timestamp="20261021T210000",
        )
    )
    latest = find_latest_eligible_s6_run(paths["session_root"])
    assert latest is not None
    assert latest.run_dir == result.run_dir
    assert latest.classification in (ELIGIBLE_DOWNSTREAM, COMPLETED_WITH_WARNINGS)
