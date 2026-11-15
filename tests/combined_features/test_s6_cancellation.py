"""S6 cooperative cancellation, publication gate, and S7 eligibility tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from combined_features import (
    BLOB_FEATURE_COLS,
    DYADIC_FEATURES_FILENAME,
    FAILED_OR_INCOMPLETE,
    MODEL_COMPAT_FILENAME,
    QC_FILENAME,
    RUN_META_FILENAME,
    STATUS_CANCELED,
    STATUS_COMPLETED,
    STATUS_FAILED,
    CombinedFeaturesCancelledError,
    CombinedFeaturesError,
    CombinedFeaturesRequest,
    is_s6_run_eligible_for_downstream,
    load_dyadic_features_table,
    run_combined_feature_extraction,
    summarize_combined_features_project,
    validate_s6_run_for_downstream,
)
from combined_features.contracts import SETTINGS_USED_FILENAME
from combined_features.pose_adapter import adapt_final_pose_to_frame_feature_input
from combined_features.processing import (
    compute_combined_features,
    compute_combined_features_from_adapted,
)
from tests.combined_features.helpers import (
    build_blob_features_table,
    build_final_pose_table,
    write_joint_session,
)


def _patch_upstream_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bypass full S3/S1 trees for synthetic joint sessions (same as export tests)."""

    monkeypatch.setattr(
        "combined_features.upstream.validate_s4_run_for_downstream",
        lambda _path: {"ok": True},
    )
    monkeypatch.setattr(
        "combined_features.upstream.validate_s5_run_for_downstream",
        lambda _path: {"ok": True},
    )


@pytest.fixture(autouse=True)
def _upstream_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_upstream_ok(monkeypatch)


class _CancelOnNthCheck:
    def __init__(self, n: int) -> None:
        self.n = n
        self.calls = 0

    def __call__(self) -> bool:
        self.calls += 1
        return self.calls >= self.n


def _run_meta(run_dir: Path) -> dict[str, Any]:
    return json.loads((run_dir / RUN_META_FILENAME).read_text(encoding="utf-8"))


def _artifact_fingerprint(result_path: Path) -> dict[str, Any]:
    table = load_dyadic_features_table(result_path)
    numeric_cols = [
        name
        for name in table.columns
        if pd.api.types.is_numeric_dtype(table[name]) and name != "schema_version"
    ]
    return {
        "columns": list(table.columns),
        "n_rows": len(table),
        "sample_idx": table["sample_idx"].to_numpy(dtype=np.int64).copy(),
        "dense_frame_idx": table["dense_frame_idx"].to_numpy(dtype=np.int64).copy(),
        "time_sec": table["time_sec"].to_numpy(dtype=np.float64).copy(),
        "source_s4_run_id": table["source_s4_run_id"].astype(str).tolist(),
        "source_s5_run_id": table["source_s5_run_id"].astype(str).tolist(),
        "numeric": {
            name: table[name].to_numpy(dtype=np.float64).copy() for name in numeric_cols
        },
    }


def test_no_callback_preserves_successful_output(tmp_path: Path) -> None:
    paths = write_joint_session(tmp_path, n_samples=4)
    result = run_combined_feature_extraction(
        CombinedFeaturesRequest(
            session_root=paths["session_root"],
            s4_run_dir=paths["s4_run_dir"],
            s5_run_dir=paths["s5_run_dir"],
            run_purpose="test",
            timestamp="20260101T100000",
        )
    )
    assert result.success is True
    assert result.status == STATUS_COMPLETED
    assert result.dyadic_features_path is not None
    assert result.dyadic_features_path.is_file()


def test_false_callback_preserves_successful_output(tmp_path: Path) -> None:
    paths = write_joint_session(tmp_path, n_samples=4)
    result = run_combined_feature_extraction(
        CombinedFeaturesRequest(
            session_root=paths["session_root"],
            s4_run_dir=paths["s4_run_dir"],
            s5_run_dir=paths["s5_run_dir"],
            run_purpose="test",
            timestamp="20260101T100001",
        ),
        cancellation_requested=lambda: False,
    )
    assert result.success is True
    assert result.dyadic_features_path is not None


def test_scientific_invariance_no_callback_vs_false_callback(tmp_path: Path) -> None:
    paths_a = write_joint_session(tmp_path / "a", n_samples=6)
    paths_b = write_joint_session(tmp_path / "b", n_samples=6)
    baseline = run_combined_feature_extraction(
        CombinedFeaturesRequest(
            session_root=paths_a["session_root"],
            s4_run_dir=paths_a["s4_run_dir"],
            s5_run_dir=paths_a["s5_run_dir"],
            run_purpose="test",
            timestamp="20260101T100010",
        )
    )
    false_cb = run_combined_feature_extraction(
        CombinedFeaturesRequest(
            session_root=paths_b["session_root"],
            s4_run_dir=paths_b["s4_run_dir"],
            s5_run_dir=paths_b["s5_run_dir"],
            run_purpose="test",
            timestamp="20260101T100011",
        ),
        cancellation_requested=lambda: False,
    )
    assert baseline.dyadic_features_path is not None
    assert false_cb.dyadic_features_path is not None
    left = _artifact_fingerprint(baseline.dyadic_features_path)
    right = _artifact_fingerprint(false_cb.dyadic_features_path)
    assert left["columns"] == right["columns"]
    assert left["n_rows"] == right["n_rows"]
    np.testing.assert_array_equal(left["sample_idx"], right["sample_idx"])
    np.testing.assert_array_equal(left["dense_frame_idx"], right["dense_frame_idx"])
    np.testing.assert_allclose(left["time_sec"], right["time_sec"], equal_nan=True)
    assert left["source_s4_run_id"] == right["source_s4_run_id"]
    assert left["source_s5_run_id"] == right["source_s5_run_id"]
    assert left["numeric"].keys() == right["numeric"].keys()
    for name in left["numeric"]:
        np.testing.assert_allclose(
            left["numeric"][name], right["numeric"][name], equal_nan=True, rtol=0.0, atol=0.0
        )
    assert baseline.qc is not None and false_cb.qc is not None
    assert baseline.qc["structural_ok"] == false_cb.qc["structural_ok"]
    assert baseline.qc["warning_count"] == false_cb.qc["warning_count"]
    assert baseline.qc["metrics"]["n_samples"] == false_cb.qc["metrics"]["n_samples"]
    assert baseline.qc["metrics"]["row_count"] == false_cb.qc["metrics"]["row_count"]
    assert (
        baseline.qc["metrics"]["n_feature_columns"]
        == false_cb.qc["metrics"]["n_feature_columns"]
    )
    assert baseline.model_compat is not None and false_cb.model_compat is not None
    assert baseline.model_compat.get("registry_status") == false_cb.model_compat.get(
        "registry_status"
    )


def test_cancel_before_run_creation(tmp_path: Path) -> None:
    paths = write_joint_session(tmp_path, n_samples=4)
    with pytest.raises(CombinedFeaturesCancelledError):
        run_combined_feature_extraction(
            CombinedFeaturesRequest(
                session_root=paths["session_root"],
                s4_run_dir=paths["s4_run_dir"],
                s5_run_dir=paths["s5_run_dir"],
                run_purpose="test",
                timestamp="20260101T100100",
            ),
            cancellation_requested=lambda: True,
        )
    root = paths["session_root"] / "combined_features"
    assert not root.exists() or list(root.iterdir()) == []


@pytest.mark.parametrize(
    ("nth", "label"),
    [
        (2, "after_bootstrap"),
        (3, "before_adapt"),
        (4, "after_adapt"),
        (5, "before_frame_features"),
        (6, "after_frame_features"),
        (7, "before_pipeline50"),
        (8, "after_pipeline50"),
        (9, "after_feature_construction"),
        (10, "after_qc_compat"),
        (11, "final_publication_gate"),
    ],
)
def test_cancel_at_safe_checkpoints(tmp_path: Path, nth: int, label: str) -> None:
    del label
    paths = write_joint_session(tmp_path, n_samples=5)
    cancel = _CancelOnNthCheck(nth)
    with pytest.raises(CombinedFeaturesCancelledError):
        run_combined_feature_extraction(
            CombinedFeaturesRequest(
                session_root=paths["session_root"],
                s4_run_dir=paths["s4_run_dir"],
                s5_run_dir=paths["s5_run_dir"],
                run_purpose="test",
                timestamp=f"20260101T11{nth:04d}",
            ),
            cancellation_requested=cancel,
        )
    runs = list((paths["session_root"] / "combined_features").iterdir())
    assert len(runs) == 1
    meta = _run_meta(runs[0])
    assert meta["status"] == STATUS_CANCELED
    assert not (runs[0] / DYADIC_FEATURES_FILENAME).exists()
    assert not (runs[0] / QC_FILENAME).exists()
    assert not (runs[0] / MODEL_COMPAT_FILENAME).exists()
    assert (runs[0] / RUN_META_FILENAME).is_file()
    assert (runs[0] / SETTINGS_USED_FILENAME).is_file()
    assert (runs[0] / "processing_log.txt").is_file()
    err = (meta.get("error") or "").lower()
    assert "cancel" in err


def test_cancel_after_publication_gate_completes(tmp_path: Path) -> None:
    paths = write_joint_session(tmp_path, n_samples=4)
    state = {"in_publication": False}

    def cancel() -> bool:
        return state["in_publication"]

    def publication_started() -> None:
        state["in_publication"] = True

    result = run_combined_feature_extraction(
        CombinedFeaturesRequest(
            session_root=paths["session_root"],
            s4_run_dir=paths["s4_run_dir"],
            s5_run_dir=paths["s5_run_dir"],
            run_purpose="test",
            timestamp="20260101T120200",
        ),
        cancellation_requested=cancel,
        publication_started=publication_started,
    )
    assert state["in_publication"] is True
    assert result.success is True
    assert result.status == STATUS_COMPLETED
    assert result.dyadic_features_path is not None
    assert result.dyadic_features_path.is_file()
    meta = _run_meta(result.run_dir)
    assert meta["status"] == STATUS_COMPLETED
    assert (result.run_dir / DYADIC_FEATURES_FILENAME).is_file()
    assert (result.run_dir / QC_FILENAME).is_file()
    assert (result.run_dir / MODEL_COMPAT_FILENAME).is_file()


def test_cancel_during_publication_writes_still_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Cancel requested while artifact writes run must not rewrite a completed run."""

    paths = write_joint_session(tmp_path, n_samples=4)
    state = {"cancel": False}

    import combined_features.runner as runner_mod

    original_write = runner_mod.write_dyadic_features_parquet

    def _write_parquet_and_cancel(path: Path, table: pd.DataFrame) -> None:
        # Flip cancel only after the publication gate has been passed.
        state["cancel"] = True
        original_write(path, table)

    monkeypatch.setattr(
        runner_mod, "write_dyadic_features_parquet", _write_parquet_and_cancel
    )
    result = run_combined_feature_extraction(
        CombinedFeaturesRequest(
            session_root=paths["session_root"],
            s4_run_dir=paths["s4_run_dir"],
            s5_run_dir=paths["s5_run_dir"],
            run_purpose="test",
            timestamp="20260101T120250",
        ),
        cancellation_requested=lambda: state["cancel"],
    )
    assert state["cancel"] is True
    assert result.success is True
    assert result.status == STATUS_COMPLETED
    assert _run_meta(result.run_dir)["status"] == STATUS_COMPLETED
    assert is_s6_run_eligible_for_downstream(result.run_dir) is True


def test_cancel_after_success_is_harmless(tmp_path: Path) -> None:
    paths = write_joint_session(tmp_path, n_samples=3)
    cancel = _CancelOnNthCheck(1000)
    result = run_combined_feature_extraction(
        CombinedFeaturesRequest(
            session_root=paths["session_root"],
            s4_run_dir=paths["s4_run_dir"],
            s5_run_dir=paths["s5_run_dir"],
            run_purpose="test",
            timestamp="20260101T120300",
        ),
        cancellation_requested=cancel,
    )
    assert result.success is True
    assert result.status == STATUS_COMPLETED


def test_repeated_cancellation_is_harmless(tmp_path: Path) -> None:
    paths = write_joint_session(tmp_path, n_samples=3)
    cancel = _CancelOnNthCheck(2)
    with pytest.raises(CombinedFeaturesCancelledError):
        run_combined_feature_extraction(
            CombinedFeaturesRequest(
                session_root=paths["session_root"],
                s4_run_dir=paths["s4_run_dir"],
                s5_run_dir=paths["s5_run_dir"],
                run_purpose="test",
                timestamp="20260101T120400",
            ),
            cancellation_requested=cancel,
        )
    # Callback remains True; a new request still cancels cleanly.
    with pytest.raises(CombinedFeaturesCancelledError):
        run_combined_feature_extraction(
            CombinedFeaturesRequest(
                session_root=paths["session_root"],
                s4_run_dir=paths["s4_run_dir"],
                s5_run_dir=paths["s5_run_dir"],
                run_purpose="test",
                timestamp="20260101T120401",
            ),
            cancellation_requested=lambda: True,
        )


def test_canceled_run_rejected_downstream(tmp_path: Path) -> None:
    paths = write_joint_session(tmp_path, n_samples=4)
    with pytest.raises(CombinedFeaturesCancelledError):
        run_combined_feature_extraction(
            CombinedFeaturesRequest(
                session_root=paths["session_root"],
                s4_run_dir=paths["s4_run_dir"],
                s5_run_dir=paths["s5_run_dir"],
                run_purpose="test",
                timestamp="20260101T130000",
            ),
            cancellation_requested=_CancelOnNthCheck(2),
        )
    run_dir = next((paths["session_root"] / "combined_features").iterdir())
    assert is_s6_run_eligible_for_downstream(run_dir) is False
    with pytest.raises(CombinedFeaturesError, match="not completed"):
        validate_s6_run_for_downstream(run_dir)
    summary = summarize_combined_features_project(paths["session_root"])
    assert summary.runs
    assert summary.runs[0].classification == FAILED_OR_INCOMPLETE
    assert summary.eligible_runs == ()


def test_canceled_run_does_not_affect_prior_completed(tmp_path: Path) -> None:
    paths = write_joint_session(tmp_path, n_samples=4)
    completed = run_combined_feature_extraction(
        CombinedFeaturesRequest(
            session_root=paths["session_root"],
            s4_run_dir=paths["s4_run_dir"],
            s5_run_dir=paths["s5_run_dir"],
            run_purpose="test",
            timestamp="20260101T140000",
        )
    )
    with pytest.raises(CombinedFeaturesCancelledError):
        run_combined_feature_extraction(
            CombinedFeaturesRequest(
                session_root=paths["session_root"],
                s4_run_dir=paths["s4_run_dir"],
                s5_run_dir=paths["s5_run_dir"],
                run_purpose="test",
                timestamp="20260101T140100",
            ),
            cancellation_requested=_CancelOnNthCheck(2),
        )
    assert completed.dyadic_features_path is not None
    assert completed.dyadic_features_path.is_file()
    assert is_s6_run_eligible_for_downstream(completed.run_dir) is True
    summary = summarize_combined_features_project(paths["session_root"])
    assert len(summary.completed_runs) == 1
    assert summary.eligible_runs
    assert summary.eligible_runs[0].run_id == completed.run_id


def test_ordinary_failure_remains_failed_even_if_cancel_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = write_joint_session(tmp_path, n_samples=4)
    state = {"cancel": False}

    def boom(*_args: object, **_kwargs: object) -> None:
        state["cancel"] = True
        raise CombinedFeaturesError("injected structural failure")

    monkeypatch.setattr(
        "combined_features.runner.evaluate_combined_features_qc",
        boom,
    )
    with pytest.raises(CombinedFeaturesError, match="injected structural failure"):
        run_combined_feature_extraction(
            CombinedFeaturesRequest(
                session_root=paths["session_root"],
                s4_run_dir=paths["s4_run_dir"],
                s5_run_dir=paths["s5_run_dir"],
                run_purpose="test",
                timestamp="20260101T150000",
            ),
            cancellation_requested=lambda: state["cancel"],
        )
    assert state["cancel"] is True
    run_dir = next((paths["session_root"] / "combined_features").iterdir())
    assert _run_meta(run_dir)["status"] == STATUS_FAILED


def test_rerun_after_cancellation(tmp_path: Path) -> None:
    paths = write_joint_session(tmp_path, n_samples=4)
    with pytest.raises(CombinedFeaturesCancelledError):
        run_combined_feature_extraction(
            CombinedFeaturesRequest(
                session_root=paths["session_root"],
                s4_run_dir=paths["s4_run_dir"],
                s5_run_dir=paths["s5_run_dir"],
                run_purpose="test",
                timestamp="20260101T160000",
            ),
            cancellation_requested=_CancelOnNthCheck(3),
        )
    result = run_combined_feature_extraction(
        CombinedFeaturesRequest(
            session_root=paths["session_root"],
            s4_run_dir=paths["s4_run_dir"],
            s5_run_dir=paths["s5_run_dir"],
            run_purpose="test",
            timestamp="20260101T160100",
        )
    )
    assert result.success is True
    assert result.status == STATUS_COMPLETED
    assert is_s6_run_eligible_for_downstream(result.run_dir) is True


def test_progress_stages_reported(tmp_path: Path) -> None:
    paths = write_joint_session(tmp_path, n_samples=3)
    phases: list[str] = []

    def progress(event: object) -> None:
        phases.append(getattr(event, "phase", ""))

    result = run_combined_feature_extraction(
        CombinedFeaturesRequest(
            session_root=paths["session_root"],
            s4_run_dir=paths["s4_run_dir"],
            s5_run_dir=paths["s5_run_dir"],
            run_purpose="test",
            timestamp="20260101T170000",
        ),
        progress_callback=progress,
        cancellation_requested=lambda: False,
    )
    assert result.success is True
    assert "validating_sources" in phases
    assert "loading_inputs" in phases
    assert "adapting_sources" in phases
    assert "computing_features" in phases
    assert "postprocessing" in phases
    assert "computing_qc" in phases
    assert "revalidating_sources" in phases
    assert "writing_artifacts" in phases
    assert "completed" in phases


def test_s7_eligibility_excludes_canceled_s6(tmp_path: Path) -> None:
    """S7 intake uses the same completed-status gate; canceled must never qualify."""

    paths = write_joint_session(tmp_path, n_samples=4)
    completed = run_combined_feature_extraction(
        CombinedFeaturesRequest(
            session_root=paths["session_root"],
            s4_run_dir=paths["s4_run_dir"],
            s5_run_dir=paths["s5_run_dir"],
            run_purpose="test",
            timestamp="20260101T180000",
        )
    )
    with pytest.raises(CombinedFeaturesCancelledError):
        run_combined_feature_extraction(
            CombinedFeaturesRequest(
                session_root=paths["session_root"],
                s4_run_dir=paths["s4_run_dir"],
                s5_run_dir=paths["s5_run_dir"],
                run_purpose="test",
                timestamp="20260101T180100",
            ),
            cancellation_requested=_CancelOnNthCheck(2),
        )
    from behavioral_inference.run_discovery import find_latest_eligible_s6_for_s7

    latest = find_latest_eligible_s6_for_s7(paths["session_root"])
    assert latest is not None
    assert latest.run_id == completed.run_id
    assert latest.run_dir == completed.run_dir


def _assert_feature_tables_identical(left: pd.DataFrame, right: pd.DataFrame) -> None:
    assert list(left.columns) == list(right.columns)
    assert len(left) == len(right)
    for col in ("focal_id", "partner_id", "dense_frame_idx", "sample_idx", "orig_frame_idx"):
        if col in left.columns:
            np.testing.assert_array_equal(
                left[col].to_numpy(), right[col].to_numpy(), err_msg=col
            )
    if "time_sec" in left.columns:
        np.testing.assert_allclose(
            left["time_sec"].to_numpy(dtype=np.float64),
            right["time_sec"].to_numpy(dtype=np.float64),
            equal_nan=True,
            rtol=0.0,
            atol=0.0,
        )
    # Forward-fill / temporal echo columns and Pipeline50 blob join must match bit-for-bit.
    for col in BLOB_FEATURE_COLS:
        if col in left.columns:
            np.testing.assert_allclose(
                left[col].to_numpy(dtype=np.float64),
                right[col].to_numpy(dtype=np.float64),
                equal_nan=True,
                rtol=0.0,
                atol=0.0,
                err_msg=col,
            )
    for name in left.columns:
        if not pd.api.types.is_numeric_dtype(left[name]):
            assert left[name].astype(str).tolist() == right[name].astype(str).tolist()
            continue
        np.testing.assert_allclose(
            left[name].to_numpy(dtype=np.float64),
            right[name].to_numpy(dtype=np.float64),
            equal_nan=True,
            rtol=0.0,
            atol=0.0,
            err_msg=name,
        )


def test_processing_checkpoints_scientific_invariance(tmp_path: Path) -> None:
    """processing.py cancel polls must not alter science when callback is None/False."""

    paths = write_joint_session(tmp_path, n_samples=8)
    baseline = compute_combined_features(
        session_root=paths["session_root"],
        s4_run_dir=paths["s4_run_dir"],
        s5_run_dir=paths["s5_run_dir"],
        exp_name="invariance",
    )
    false_cb = compute_combined_features(
        session_root=paths["session_root"],
        s4_run_dir=paths["s4_run_dir"],
        s5_run_dir=paths["s5_run_dir"],
        exp_name="invariance",
        cancellation_requested=lambda: False,
    )
    _assert_feature_tables_identical(baseline.features, false_cb.features)
    _assert_feature_tables_identical(
        baseline.frame_feature_result.features_df,
        false_cb.frame_feature_result.features_df,
    )
    assert baseline.warnings == false_cb.warnings
    assert baseline.metadata["alignment"] == false_cb.metadata["alignment"]
    assert baseline.metadata["n_samples"] == false_cb.metadata["n_samples"]
    assert baseline.metadata["s4_run_id"] == false_cb.metadata["s4_run_id"]
    assert baseline.metadata["s5_run_id"] == false_cb.metadata["s5_run_id"]
    assert baseline.features.attrs.get("implantation_status") == false_cb.features.attrs.get(
        "implantation_status"
    )
    np.testing.assert_array_equal(baseline.adapted_pose.x, false_cb.adapted_pose.x)
    np.testing.assert_array_equal(baseline.adapted_pose.y, false_cb.adapted_pose.y)


def test_processing_cancel_before_frame_features(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = write_joint_session(tmp_path, n_samples=4)
    calls = {"ff": 0, "p50": 0}

    def boom_ff(*_args: object, **_kwargs: object) -> None:
        calls["ff"] += 1
        raise AssertionError("build_frame_features must not run after cancel")

    def boom_p50(*_args: object, **_kwargs: object) -> None:
        calls["p50"] += 1
        raise AssertionError("pipeline50 must not run after cancel")

    monkeypatch.setattr("combined_features.processing.build_frame_features", boom_ff)
    monkeypatch.setattr("combined_features.processing.apply_pipeline50_post", boom_p50)
    with pytest.raises(CombinedFeaturesCancelledError):
        compute_combined_features(
            session_root=paths["session_root"],
            s4_run_dir=paths["s4_run_dir"],
            s5_run_dir=paths["s5_run_dir"],
            cancellation_requested=_CancelOnNthCheck(2),  # after adapt, before FF
        )
    assert calls["ff"] == 0
    assert calls["p50"] == 0


def test_processing_exception_propagation_not_rewritten_as_cancel(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ordinary science failures must remain CombinedFeaturesError, not canceled."""

    pose = build_final_pose_table(n_samples=4)
    blob = build_blob_features_table(n_samples=4)
    adapted = adapt_final_pose_to_frame_feature_input(
        pose,
        orig_frame_idx=blob["orig_frame_idx"].to_numpy(dtype=np.int64),
    )
    state = {"cancel": False}

    def boom(*_args: object, **_kwargs: object) -> None:
        state["cancel"] = True
        raise CombinedFeaturesError("injected frame-feature failure")

    monkeypatch.setattr("combined_features.processing.build_frame_features", boom)
    with pytest.raises(CombinedFeaturesError, match="injected frame-feature failure") as excinfo:
        compute_combined_features_from_adapted(
            adapted,
            blob,
            cancellation_requested=lambda: state["cancel"],
        )
    assert not isinstance(excinfo.value, CombinedFeaturesCancelledError)
    assert state["cancel"] is True


def test_processing_pipeline50_exception_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pose = build_final_pose_table(n_samples=4)
    blob = build_blob_features_table(n_samples=4)
    adapted = adapt_final_pose_to_frame_feature_input(
        pose,
        orig_frame_idx=blob["orig_frame_idx"].to_numpy(dtype=np.int64),
    )

    def boom(*_args: object, **_kwargs: object) -> None:
        raise CombinedFeaturesError("injected pipeline50 failure")

    monkeypatch.setattr("combined_features.processing.apply_pipeline50_post", boom)
    with pytest.raises(CombinedFeaturesError, match="injected pipeline50 failure") as excinfo:
        compute_combined_features_from_adapted(
            adapted,
            blob,
            cancellation_requested=lambda: False,
        )
    assert not isinstance(excinfo.value, CombinedFeaturesCancelledError)
