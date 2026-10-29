"""Sprint 3 immutable publication lifecycle tests."""

from __future__ import annotations

import json
from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from behavioral_arbitration.contracts import STATUS_COMPLETED
from project.file_digest import sha256_file
from results_packaging.config import ExecutionSettings, ResultsPackagingConfig
from results_packaging.contracts import (
    BEHAVIOR_BOUTS_FILENAME,
    ETHOGRAM_SUMMARY_FILENAME,
    FINGERPRINT_TYPE_SHA256,
    FINGERPRINT_TYPE_SIZE_MTIME,
    PACKAGE_MANIFEST_FILENAME,
    QC_FILENAME,
    RESULTS_PACKAGE_DIRNAME,
    REVIEW_EMIT_PROFILE_ID,
    RUN_META_FILENAME,
    STALE_SOURCE,
    STATUS_COMPLETED_WITH_WARNINGS,
    STATUS_FAILED,
    SYNC_INDEX_FILENAME,
    VERIFICATION_STRENGTH_CONTENT_HASH,
    VERIFICATION_STRENGTH_SIZE_MTIME,
    ProvenanceError,
    PublicationError,
)
from results_packaging.publication import (
    is_s9_run_eligible_for_downstream,
    reopen_results_package,
    run_results_packaging_publication,
)
from results_packaging.run_discovery import (
    find_latest_eligible_s9_run,
    list_completed_s9_runs,
    summarize_results_package_project,
)
from results_packaging.runner import ResultsPackagingRequest
from tests.results_packaging._fixtures import (
    build_packaging_session,
    make_multi_channel_bouts,
)
from tests.results_packaging._fixtures import (
    sha256_file as fixture_sha,
)


def _eligible_patches(info: dict[str, Any]):
    def fake_validate(path: Path | str) -> dict[str, Any]:
        return {
            "s8_run_dir": Path(path).resolve(),
            "status": STATUS_COMPLETED,
            "source_s7_run_dir": info["s7_run_dir"].resolve(),
            "source_s7_run_id": info["s7_run_id"],
            "source_s6_run_dir": info["s6_run_dir"].resolve(),
            "source_s6_run_id": info["s6_run_id"],
        }

    return (
        patch(
            "results_packaging.eligibility.validate_s8_run_for_downstream",
            side_effect=fake_validate,
        ),
        patch(
            "results_packaging.eligibility.is_s8_run_eligible_for_downstream",
            return_value=True,
        ),
        patch(
            "results_packaging.publication.is_s8_run_eligible_for_downstream",
            return_value=True,
            create=True,
        ),
        patch(
            "behavioral_arbitration.runner.is_s8_run_eligible_for_downstream",
            return_value=True,
        ),
        patch(
            "behavioral_arbitration.runner.validate_s8_run_for_downstream",
            side_effect=fake_validate,
        ),
    )


def _publish(fixture: dict[str, Any], **kwargs: Any):
    s8 = make_multi_channel_bouts()
    s8 = s8[s8["end_sample_idx"] < fixture["n_dense"]].reset_index(drop=True)
    s8.to_parquet(fixture["bouts_path"], index=False)
    cfg = kwargs.pop("config", ResultsPackagingConfig())
    request = ResultsPackagingRequest(
        session_root=fixture["session_root"],
        s8_run_dir=fixture["s8_run_dir"],
        config=cfg,
        timestamp=kwargs.pop("timestamp", "20260101T010000"),
        **kwargs,
    )
    with ExitStack() as stack:
        for p in _eligible_patches(fixture):
            stack.enter_context(p)
        return run_results_packaging_publication(request)


def test_successful_immutable_publication(tmp_path: Path) -> None:
    fixture = build_packaging_session(tmp_path, n_prepared=10, prepared_fps=30.0)
    s8 = make_multi_channel_bouts()
    s8 = s8[s8["end_sample_idx"] < fixture["n_dense"]].reset_index(drop=True)
    s8.to_parquet(fixture["bouts_path"], index=False)
    states_sha = fixture_sha(fixture["states_path"])
    bouts_sha = fixture_sha(fixture["bouts_path"])
    sync_sha = fixture_sha(fixture["prepared_sync"])
    video_bytes = fixture["prepared_video"].read_bytes()

    request = ResultsPackagingRequest(
        session_root=fixture["session_root"],
        s8_run_dir=fixture["s8_run_dir"],
        timestamp="20260101T010000",
    )
    with ExitStack() as stack:
        for p in _eligible_patches(fixture):
            stack.enter_context(p)
        result = run_results_packaging_publication(request)
        assert result.success
        assert result.status in (STATUS_COMPLETED, STATUS_COMPLETED_WITH_WARNINGS)
        assert result.run_dir.is_dir()
        assert result.run_id == "results_package__20260101T010000"
        for name in (
            SYNC_INDEX_FILENAME,
            BEHAVIOR_BOUTS_FILENAME,
            ETHOGRAM_SUMMARY_FILENAME,
            PACKAGE_MANIFEST_FILENAME,
            QC_FILENAME,
            RUN_META_FILENAME,
            "settings_used.yaml",
            "processing_log.txt",
        ):
            assert (result.run_dir / name).is_file()
        assert not (result.run_dir / "bout_index.parquet").exists()

        # No upstream mutation.
        assert fixture_sha(fixture["states_path"]) == states_sha
        assert fixture_sha(fixture["bouts_path"]) == bouts_sha
        assert fixture_sha(fixture["prepared_sync"]) == sync_sha
        assert fixture["prepared_video"].read_bytes() == video_bytes

        meta = json.loads((result.run_dir / RUN_META_FILENAME).read_text(encoding="utf-8"))
        assert meta["status"] == result.status
        assert meta["science_written"] is True
        assert is_s9_run_eligible_for_downstream(result.run_dir)


def test_unique_run_directories_and_no_overwrite(tmp_path: Path) -> None:
    fixture = build_packaging_session(tmp_path, n_prepared=10, prepared_fps=30.0)
    first = _publish(fixture, timestamp="20260101T020000")
    assert first.success
    with pytest.raises(PublicationError, match="already exists"):
        _publish(fixture, timestamp="20260101T020000")
    second = _publish(fixture, timestamp="20260101T020001")
    assert second.success
    assert second.run_dir != first.run_dir


def test_failure_cleans_science_preserves_diagnostics(tmp_path: Path) -> None:
    fixture = build_packaging_session(tmp_path, n_prepared=10, prepared_fps=30.0)
    s8 = make_multi_channel_bouts()
    s8 = s8[s8["end_sample_idx"] < fixture["n_dense"]].reset_index(drop=True)
    s8.to_parquet(fixture["bouts_path"], index=False)

    request = ResultsPackagingRequest(
        session_root=fixture["session_root"],
        s8_run_dir=fixture["s8_run_dir"],
        timestamp="20260101T030000",
    )
    with ExitStack() as stack:
        for p in _eligible_patches(fixture):
            stack.enter_context(p)
        with patch(
            "results_packaging.publication.write_sync_index_parquet",
            side_effect=RuntimeError("forced publish failure"),
        ):
            result = run_results_packaging_publication(request)

    assert result.success is False
    assert result.status == STATUS_FAILED
    assert result.run_dir.is_dir()
    assert (result.run_dir / RUN_META_FILENAME).is_file()
    assert (result.run_dir / "settings_used.yaml").is_file()
    assert (result.run_dir / "processing_log.txt").is_file()
    assert not (result.run_dir / SYNC_INDEX_FILENAME).exists()
    assert not (result.run_dir / BEHAVIOR_BOUTS_FILENAME).exists()
    assert not (result.run_dir / ETHOGRAM_SUMMARY_FILENAME).exists()
    assert not (result.run_dir / PACKAGE_MANIFEST_FILENAME).exists()
    meta = json.loads((result.run_dir / RUN_META_FILENAME).read_text(encoding="utf-8"))
    assert meta["status"] == STATUS_FAILED
    assert meta["science_written"] is False
    assert meta["error"]


def test_source_change_during_processing_rejects(tmp_path: Path) -> None:
    fixture = build_packaging_session(tmp_path, n_prepared=10, prepared_fps=30.0)
    s8 = make_multi_channel_bouts()
    s8 = s8[s8["end_sample_idx"] < fixture["n_dense"]].reset_index(drop=True)
    s8.to_parquet(fixture["bouts_path"], index=False)

    request = ResultsPackagingRequest(
        session_root=fixture["session_root"],
        s8_run_dir=fixture["s8_run_dir"],
        timestamp="20260101T040000",
    )

    original = None

    def tamper_then_revalidate(*args: Any, **kwargs: Any):
        nonlocal original
        # Mutate S8 bouts after packaging computed digests.
        fixture["bouts_path"].write_bytes(fixture["bouts_path"].read_bytes() + b"x")
        raise ProvenanceError("S8 behavior_bouts SHA256 changed during S9 processing.")

    with ExitStack() as stack:
        for p in _eligible_patches(fixture):
            stack.enter_context(p)
        with patch(
            "results_packaging.publication._revalidate_sources_before_publish",
            side_effect=tamper_then_revalidate,
        ):
            result = run_results_packaging_publication(request)

    assert result.success is False
    assert result.status == STATUS_FAILED
    assert not (result.run_dir / BEHAVIOR_BOUTS_FILENAME).exists()


def test_manifest_fingerprint_semantics(tmp_path: Path) -> None:
    fixture = build_packaging_session(tmp_path, n_prepared=10, prepared_fps=30.0)
    result = _publish(fixture, timestamp="20260101T050000")
    manifest = json.loads(
        (result.run_dir / PACKAGE_MANIFEST_FILENAME).read_text(encoding="utf-8")
    )
    assert manifest["package_mode"] == "reference_lightweight"
    assert manifest["portability"]["self_contained"] is False
    assert manifest["full_bout_preservation"] is True
    assert manifest["fingerprint_policy"]["size_mtime_is_not_sha256"] is True

    embedded = {a["role"]: a for a in manifest["package_artifacts"]}
    assert embedded["sync_index"]["fingerprint_type"] == FINGERPRINT_TYPE_SHA256
    assert (
        embedded["sync_index"]["verification_strength"]
        == VERIFICATION_STRENGTH_CONTENT_HASH
    )
    assert embedded["sync_index"]["sha256"] == sha256_file(
        result.run_dir / SYNC_INDEX_FILENAME
    )
    assert embedded["package_manifest"]["verification_strength"] == "lifecycle_path"
    assert embedded["package_manifest"]["sha256"] is None

    video_refs = [
        e for e in manifest["external_references"] if e.get("role") == "prepared_video"
    ]
    assert video_refs
    video = video_refs[0]
    # Default path: trusted recorded digest or size/mtime — never silent equate.
    assert video["fingerprint_type"] in (
        FINGERPRINT_TYPE_SHA256,
        FINGERPRINT_TYPE_SIZE_MTIME,
    )
    if video["sha256"] is None:
        assert video["verification_strength"] == VERIFICATION_STRENGTH_SIZE_MTIME
        assert "NOT equivalent to SHA256" in video["notes"]


def test_profile_artifact_isolation(tmp_path: Path) -> None:
    fixture = build_packaging_session(tmp_path, n_prepared=10, prepared_fps=30.0)
    cfg = ResultsPackagingConfig(
        execution=ExecutionSettings(
            selected_export_profiles=(REVIEW_EMIT_PROFILE_ID,),
        )
    )
    result = _publish(fixture, timestamp="20260101T060000", config=cfg)
    assert result.success
    profile_dir = result.run_dir / "export_profiles" / REVIEW_EMIT_PROFILE_ID
    assert profile_dir.is_dir()
    assert (profile_dir / "bout_index.parquet").is_file()
    assert (profile_dir / "profile_manifest.json").is_file()
    assert not (result.run_dir / "bout_index.parquet").exists()
    full = pd.read_parquet(result.run_dir / BEHAVIOR_BOUTS_FILENAME)
    index = pd.read_parquet(profile_dir / "bout_index.parquet")
    assert len(index) <= len(full)
    meta = json.loads((profile_dir / "profile_manifest.json").read_text(encoding="utf-8"))
    assert meta["n_included"] + meta["n_excluded"] == len(full)


def test_missing_video_blocks_when_required(tmp_path: Path) -> None:
    from results_packaging.contracts import IntakeError

    fixture = build_packaging_session(
        tmp_path, n_prepared=10, prepared_fps=30.0, include_video=False
    )
    s8 = make_multi_channel_bouts()
    s8 = s8[s8["end_sample_idx"] < fixture["n_dense"]].reset_index(drop=True)
    s8.to_parquet(fixture["bouts_path"], index=False)
    request = ResultsPackagingRequest(
        session_root=fixture["session_root"],
        s8_run_dir=fixture["s8_run_dir"],
        timestamp="20260101T070000",
        config=ResultsPackagingConfig(
            execution=ExecutionSettings(require_prepared_video=True)
        ),
    )
    with ExitStack() as stack:
        for p in _eligible_patches(fixture):
            stack.enter_context(p)
        with pytest.raises(IntakeError, match="Prepared video missing"):
            run_results_packaging_publication(request)
    # No completed science package left behind.
    run_dir = (
        fixture["session_root"]
        / RESULTS_PACKAGE_DIRNAME
        / "results_package__20260101T070000"
    )
    assert not run_dir.exists() or not (run_dir / BEHAVIOR_BOUTS_FILENAME).exists()


def test_historical_status_preserved_when_source_goes_stale(tmp_path: Path) -> None:
    fixture = build_packaging_session(tmp_path, n_prepared=10, prepared_fps=30.0)
    result = _publish(fixture, timestamp="20260101T080000")
    assert result.success
    historical = json.loads((result.run_dir / RUN_META_FILENAME).read_text(encoding="utf-8"))[
        "status"
    ]

    # Break prepared sync identity after successful publish.
    fixture["prepared_sync"].write_bytes(b"tampered-sync")

    with ExitStack() as stack:
        for p in _eligible_patches(fixture):
            stack.enter_context(p)
        assert is_s9_run_eligible_for_downstream(result.run_dir) is False
        summary = summarize_results_package_project(fixture["session_root"])
        run = summary.runs[0]
        assert run.status == historical  # historical status unchanged
        assert run.classification == STALE_SOURCE
        completed = list_completed_s9_runs(fixture["session_root"])
        assert any(r.run_id == result.run_id for r in completed)
        assert find_latest_eligible_s9_run(fixture["session_root"]) is None


def test_reopen_existing_package(tmp_path: Path) -> None:
    fixture = build_packaging_session(tmp_path, n_prepared=10, prepared_fps=30.0)
    result = _publish(fixture, timestamp="20260101T090000")
    with ExitStack() as stack:
        for p in _eligible_patches(fixture):
            stack.enter_context(p)
        opened = reopen_results_package(result.run_dir)
    assert opened["currently_eligible"] is True
    assert len(opened["sync_index"]) == len(pd.read_parquet(result.sync_index_path))
    assert len(opened["behavior_bouts"]) == len(
        pd.read_parquet(result.behavior_bouts_path)
    )


def test_deterministic_repeat_packaging(tmp_path: Path) -> None:
    fixture = build_packaging_session(tmp_path, n_prepared=10, prepared_fps=30.0)
    a = _publish(fixture, timestamp="20260101T100000")
    b = _publish(fixture, timestamp="20260101T100001")
    bouts_a = pd.read_parquet(a.behavior_bouts_path)
    bouts_b = pd.read_parquet(b.behavior_bouts_path)
    cols = [
        "source_s8_bout_id",
        "channel",
        "behavior_name",
        "start_sample_idx",
        "end_sample_idx",
        "bout_duration_frames",
    ]
    assert bouts_a[cols].reset_index(drop=True).equals(bouts_b[cols].reset_index(drop=True))
    sync_a = pd.read_parquet(a.sync_index_path)
    sync_b = pd.read_parquet(b.sync_index_path)
    assert sync_a["prepared_frame_idx"].tolist() == sync_b["prepared_frame_idx"].tolist()


def test_no_writes_outside_results_package(tmp_path: Path) -> None:
    fixture = build_packaging_session(tmp_path, n_prepared=10, prepared_fps=30.0)
    # Match _publish bout rewrite so mtime baseline excludes helper setup.
    s8 = make_multi_channel_bouts()
    s8 = s8[s8["end_sample_idx"] < fixture["n_dense"]].reset_index(drop=True)
    s8.to_parquet(fixture["bouts_path"], index=False)
    before = {
        p: (p.stat().st_mtime_ns, fixture_sha(p) if p.suffix in {".parquet", ".npz", ".mp4", ".json"} else p.stat().st_size)
        for p in fixture["session_root"].rglob("*")
        if p.is_file() and RESULTS_PACKAGE_DIRNAME not in p.parts
    }
    request = ResultsPackagingRequest(
        session_root=fixture["session_root"],
        s8_run_dir=fixture["s8_run_dir"],
        timestamp="20260101T110000",
    )
    with ExitStack() as stack:
        for p in _eligible_patches(fixture):
            stack.enter_context(p)
        result = run_results_packaging_publication(request)
    assert result.success
    after = {
        p: (p.stat().st_mtime_ns, fixture_sha(p) if p.suffix in {".parquet", ".npz", ".mp4", ".json"} else p.stat().st_size)
        for p in fixture["session_root"].rglob("*")
        if p.is_file() and RESULTS_PACKAGE_DIRNAME not in p.parts
    }
    # Content identity of upstream artifacts must be unchanged (mtime may differ
    # on some filesystems after reads; compare digests/sizes).
    assert {p: v[1] for p, v in before.items()} == {p: v[1] for p, v in after.items()}
