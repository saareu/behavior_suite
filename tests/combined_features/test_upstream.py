"""Joint S4/S5 eligibility and timeline alignment tests."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from combined_features import (
    REASON_INDEX_INTEGRITY,
    REASON_S4_INELIGIBLE,
    REASON_S4_STRUCTURE,
    REASON_S5_INELIGIBLE,
    REASON_SAMPLE_COUNT_MISMATCH,
    REASON_SESSION_MISMATCH,
    REASON_SYNC_DIGEST_MISMATCH,
    REASON_SYNC_PATH_MISMATCH,
    REASON_TIME_GRID_MISMATCH,
    CombinedFeaturesError,
    evaluate_joint_alignment,
    extract_s4_sample_timeline,
    select_default_joint_pair,
    validate_joint_s4_s5_pair,
)
from pose_finalization.contracts import RUN_META_FILENAME as S4_RUN_META_FILENAME
from tests.combined_features.helpers import (
    build_blob_features_table,
    build_final_pose_table,
    sha256_file,
    write_joint_session,
    write_prepared_sync,
)


def _load_metas(paths: dict[str, Path]) -> tuple[dict, dict]:
    s4 = json.loads((paths["s4_run_dir"] / S4_RUN_META_FILENAME).read_text(encoding="utf-8"))
    s5 = json.loads(
        (paths["s5_run_dir"] / "run_meta.json").read_text(encoding="utf-8")
    )
    return s4, s5


def _patch_upstream_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "combined_features.upstream.validate_s4_run_for_downstream",
        lambda _path: {"ok": True},
    )
    monkeypatch.setattr(
        "combined_features.upstream.validate_s5_run_for_downstream",
        lambda _path: {"ok": True},
    )


def test_joint_alignment_passes_for_shared_sync_and_timeline(joint_session: dict[str, Path]) -> None:
    import pandas as pd

    s4_meta, s5_meta = _load_metas(joint_session)
    final_pose = pd.read_parquet(joint_session["final_pose"])
    blob = pd.read_parquet(joint_session["blob_features"])
    report = evaluate_joint_alignment(
        session_root=joint_session["session_root"],
        s4_run_dir=joint_session["s4_run_dir"],
        s5_run_dir=joint_session["s5_run_dir"],
        s4_meta=s4_meta,
        s5_meta=s5_meta,
        final_pose=final_pose,
        blob_features=blob,
    )
    assert report.eligible is True
    assert report.reason_code is None
    assert report.sample_count == 4
    assert report.prepared_sync_sha256 == sha256_file(joint_session["prepared_sync"])
    assert all(report.checks.values())


def test_timestamp_and_sample_index_alignment(tmp_path: Path) -> None:
    paths = write_joint_session(tmp_path, n_samples=5)
    import pandas as pd

    s4_meta, s5_meta = _load_metas(paths)
    final_pose = pd.read_parquet(paths["final_pose"])
    blob = pd.read_parquet(paths["blob_features"])
    times_s4, samples_s4 = extract_s4_sample_timeline(final_pose)
    report = evaluate_joint_alignment(
        session_root=paths["session_root"],
        s4_run_dir=paths["s4_run_dir"],
        s5_run_dir=paths["s5_run_dir"],
        s4_meta=s4_meta,
        s5_meta=s5_meta,
        final_pose=final_pose,
        blob_features=blob,
    )
    assert report.eligible
    assert np.array_equal(samples_s4, np.arange(5))
    assert np.allclose(times_s4, blob["time_sec"].to_numpy(), rtol=0.0, atol=1e-12)


def test_mismatched_s1_lineage_paths(tmp_path: Path) -> None:
    other_sync = write_prepared_sync(tmp_path / "other" / "prepared_sync.npz", frame_count=4)
    paths = write_joint_session(tmp_path, s5_sync_path=other_sync)
    import pandas as pd

    s4_meta, s5_meta = _load_metas(paths)
    report = evaluate_joint_alignment(
        session_root=paths["session_root"],
        s4_run_dir=paths["s4_run_dir"],
        s5_run_dir=paths["s5_run_dir"],
        s4_meta=s4_meta,
        s5_meta=s5_meta,
        final_pose=pd.read_parquet(paths["final_pose"]),
        blob_features=pd.read_parquet(paths["blob_features"]),
    )
    assert report.eligible is False
    assert report.reason_code == REASON_SYNC_PATH_MISMATCH


def test_session_mismatch(tmp_path: Path) -> None:
    paths = write_joint_session(
        tmp_path,
        s5_session_root=tmp_path / "other_session",
    )
    (tmp_path / "other_session").mkdir(parents=True, exist_ok=True)
    import pandas as pd

    s4_meta, s5_meta = _load_metas(paths)
    report = evaluate_joint_alignment(
        session_root=paths["session_root"],
        s4_run_dir=paths["s4_run_dir"],
        s5_run_dir=paths["s5_run_dir"],
        s4_meta=s4_meta,
        s5_meta=s5_meta,
        final_pose=pd.read_parquet(paths["final_pose"]),
        blob_features=pd.read_parquet(paths["blob_features"]),
    )
    assert report.eligible is False
    assert report.reason_code == REASON_SESSION_MISMATCH


def test_s1_synchronization_content_change(tmp_path: Path) -> None:
    paths = write_joint_session(tmp_path, n_samples=4, frame_count=4)
    # Re-write sync with different content but same path → digest drift.
    write_prepared_sync(paths["prepared_sync"], frame_count=4, fps=60.0)
    import pandas as pd

    s4_meta, s5_meta = _load_metas(paths)
    report = evaluate_joint_alignment(
        session_root=paths["session_root"],
        s4_run_dir=paths["s4_run_dir"],
        s5_run_dir=paths["s5_run_dir"],
        s4_meta=s4_meta,
        s5_meta=s5_meta,
        final_pose=pd.read_parquet(paths["final_pose"]),
        blob_features=pd.read_parquet(paths["blob_features"]),
    )
    assert report.eligible is False
    assert report.reason_code == REASON_SYNC_DIGEST_MISMATCH


def test_time_grid_mismatch(tmp_path: Path) -> None:
    blob = build_blob_features_table(n_samples=4, time_offset=1e-9)
    paths = write_joint_session(tmp_path, blob_features=blob)
    import pandas as pd

    s4_meta, s5_meta = _load_metas(paths)
    report = evaluate_joint_alignment(
        session_root=paths["session_root"],
        s4_run_dir=paths["s4_run_dir"],
        s5_run_dir=paths["s5_run_dir"],
        s4_meta=s4_meta,
        s5_meta=s5_meta,
        final_pose=pd.read_parquet(paths["final_pose"]),
        blob_features=pd.read_parquet(paths["blob_features"]),
    )
    assert report.eligible is False
    assert report.reason_code == REASON_TIME_GRID_MISMATCH


def test_sample_count_mismatch(tmp_path: Path) -> None:
    paths = write_joint_session(
        tmp_path,
        final_pose=build_final_pose_table(n_samples=4),
        blob_features=build_blob_features_table(n_samples=3),
        n_samples=4,
    )
    import pandas as pd

    s4_meta, s5_meta = _load_metas(paths)
    report = evaluate_joint_alignment(
        session_root=paths["session_root"],
        s4_run_dir=paths["s4_run_dir"],
        s5_run_dir=paths["s5_run_dir"],
        s4_meta=s4_meta,
        s5_meta=s5_meta,
        final_pose=pd.read_parquet(paths["final_pose"]),
        blob_features=pd.read_parquet(paths["blob_features"]),
    )
    assert report.eligible is False
    assert report.reason_code == REASON_SAMPLE_COUNT_MISMATCH


def test_duplicate_s4_sample_keys() -> None:
    import pandas as pd

    pose = build_final_pose_table(n_samples=2)
    pose = pd.concat([pose, pose.iloc[[0]]], ignore_index=True)
    with pytest.raises(CombinedFeaturesError) as exc:
        extract_s4_sample_timeline(pose)
    assert exc.value.reason_code == REASON_S4_STRUCTURE


def test_missing_sample_index_rejected() -> None:
    pose = build_final_pose_table(n_samples=3)
    pose = pose[pose["sample_idx"] != 1].reset_index(drop=True)
    with pytest.raises(CombinedFeaturesError) as exc:
        extract_s4_sample_timeline(pose)
    assert exc.value.reason_code == REASON_INDEX_INTEGRITY


def test_stale_upstream_s4_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = write_joint_session(tmp_path)
    monkeypatch.setattr(
        "combined_features.upstream.validate_s4_run_for_downstream",
        lambda _path: (_ for _ in ()).throw(
            __import__("pose_finalization.contracts", fromlist=["PoseFinalizationError"]).PoseFinalizationError(
                "stale source"
            )
        ),
    )
    monkeypatch.setattr(
        "combined_features.upstream.validate_s5_run_for_downstream",
        lambda _path: {"ok": True},
    )
    with pytest.raises(CombinedFeaturesError) as exc:
        validate_joint_s4_s5_pair(
            session_root=paths["session_root"],
            s4_run_dir=paths["s4_run_dir"],
            s5_run_dir=paths["s5_run_dir"],
        )
    assert exc.value.reason_code == REASON_S4_INELIGIBLE


def test_stale_upstream_s5_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = write_joint_session(tmp_path)
    monkeypatch.setattr(
        "combined_features.upstream.validate_s4_run_for_downstream",
        lambda _path: {"ok": True},
    )
    monkeypatch.setattr(
        "combined_features.upstream.validate_s5_run_for_downstream",
        lambda _path: (_ for _ in ()).throw(
            __import__("blob_features.contracts", fromlist=["BlobFeaturesError"]).BlobFeaturesError(
                "stale source"
            )
        ),
    )
    with pytest.raises(CombinedFeaturesError) as exc:
        validate_joint_s4_s5_pair(
            session_root=paths["session_root"],
            s4_run_dir=paths["s4_run_dir"],
            s5_run_dir=paths["s5_run_dir"],
        )
    assert exc.value.reason_code == REASON_S5_INELIGIBLE


def test_validate_joint_pair_success(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = write_joint_session(tmp_path)
    _patch_upstream_ok(monkeypatch)
    pair = validate_joint_s4_s5_pair(
        session_root=paths["session_root"],
        s4_run_dir=paths["s4_run_dir"],
        s5_run_dir=paths["s5_run_dir"],
    )
    assert pair.sample_count == 4
    assert pair.alignment.eligible is True
    assert len(pair.final_pose) == 4 * 2 * 5
    assert len(pair.blob_features) == 4


def test_select_default_joint_pair_uses_latest_eligible(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = write_joint_session(tmp_path)
    _patch_upstream_ok(monkeypatch)

    from blob_features.run_discovery import BlobFeaturesRunSummary
    from pose_finalization.run_discovery import PoseFinalizationRunSummary

    s4_summary = PoseFinalizationRunSummary(
        run_dir=paths["s4_run_dir"],
        run_id=paths["s4_run_dir"].name,
        status="completed",
        classification="eligible",
        source_s3_run_id="s3",
        source_s3_run_dir=None,
        final_pose_path=paths["final_pose"],
        created_at="2026-10-21T12:00:00",
        completed_at="2026-10-21T12:01:00",
        warning_count=0,
        error_message=None,
    )
    s5_summary = BlobFeaturesRunSummary(
        run_dir=paths["s5_run_dir"],
        run_id=paths["s5_run_dir"].name,
        status="completed",
        classification="eligible",
        source_s1_session_root=paths["session_root"],
        source_prepared_sync=paths["prepared_sync"],
        blob_features_path=paths["blob_features"],
        created_at="2026-10-21T12:01:00",
        completed_at="2026-10-21T12:02:00",
        warning_count=0,
        error_message=None,
    )
    monkeypatch.setattr(
        "combined_features.upstream.find_latest_eligible_s4_run",
        lambda _root: s4_summary,
    )
    monkeypatch.setattr(
        "combined_features.upstream.find_latest_eligible_s5_run",
        lambda _root: s5_summary,
    )
    pair = select_default_joint_pair(paths["session_root"])
    assert pair.s4_run_dir == paths["s4_run_dir"]
    assert pair.s5_run_dir == paths["s5_run_dir"]
