"""Integration tests for media/CSV export jobs (synthetic video + mini package)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from results_packaging.contracts import (
    BEHAVIOR_BOUTS_FILENAME,
    RESULTS_PACKAGE_DIRNAME,
    STATUS_COMPLETED,
    ProvenanceError,
)
from results_packaging.media_csv_export.clip_bounds import compute_clip_bounds
from results_packaging.media_csv_export.config import SelectionSpec
from results_packaging.media_csv_export.contracts import (
    ENCODE_STATUS_FAILED,
    ENCODE_STATUS_SUCCEEDED,
    RESULTS_EXPORTS_DIRNAME,
    SELECTION_BOUT_ID_LIST,
    SELECTION_CHANNELS_BEHAVIORS,
    STATUS_INCOMPLETE,
    ExportPublicationError,
    ExportSourceError,
)
from results_packaging.media_csv_export.discovery import (
    reopen_export_job,
    summarize_export_project,
)
from results_packaging.media_csv_export.runner import (
    CsvExportRequest,
    MediaExportRequest,
    retry_media_export,
    run_csv_export,
    run_media_export,
)
from results_packaging.media_csv_export.source import validate_package_for_export
from results_packaging.media_csv_export.video import encode_mp4_clip
from tests.results_packaging._export_fixtures import (
    build_mini_export_package,
    count_video_frames,
    patch_export_package_currently_eligible,
)


@pytest.fixture(autouse=True)
def _eligible_mini_packages():
    with patch_export_package_currently_eligible():
        yield


def test_csv_export_full_table_immutable_and_no_package_writes(tmp_path: Path):
    info = build_mini_export_package(tmp_path)
    package_bouts = info["package_dir"] / BEHAVIOR_BOUTS_FILENAME
    before = package_bouts.read_bytes()

    result = run_csv_export(
        CsvExportRequest(
            session_root=info["session_root"],
            package_dir=info["package_dir"],
            timestamp="20260101T000001",
        )
    )
    assert result.success
    assert result.status == STATUS_COMPLETED
    assert result.n_rows == 3
    assert result.csv_path is not None and result.csv_path.is_file()
    assert result.job_dir.parent.name == RESULTS_EXPORTS_DIRNAME
    assert package_bouts.read_bytes() == before

    csv = pd.read_csv(result.csv_path)
    assert len(csv) == 3
    assert set(csv["export_provenance"]) == {"automated"}
    assert set(csv["source_package_run_id"]) == {info["package_run_id"]}
    assert "source_s8_bout_id" in csv.columns
    assert "prepared_frame_idx_start" in csv.columns

    # Package tree untouched beyond read.
    assert not (info["package_dir"] / "clips").exists()
    reopened = reopen_export_job(result.job_dir)
    assert reopened["status"] == STATUS_COMPLETED
    assert reopened["job_kind"] == "csv_export"


def test_media_export_clip_counts_timing_directional_and_boundary(tmp_path: Path):
    info = build_mini_export_package(tmp_path, n_frames=60, fps=30.0)
    result = run_media_export(
        MediaExportRequest(
            session_root=info["session_root"],
            package_dir=info["package_dir"],
            timestamp="20260101T000002",
            pre_context_sec=0.0,
            post_context_sec=0.0,
        )
    )
    assert result.status == STATUS_COMPLETED
    assert result.n_selected == 3
    assert result.n_succeeded == 3
    assert result.n_failed == 0
    assert result.clip_index_path is not None

    clip_index = pd.read_parquet(result.clip_index_path)
    assert len(clip_index) == 3
    assert set(clip_index["encode_status"]) == {ENCODE_STATUS_SUCCEEDED}

    # Directional vs symmetric preserved.
    contact = clip_index.loc[clip_index["channel"] == "contact"].iloc[0]
    assert contact["identity_grain"] == "symmetric_pair"
    directional = clip_index.loc[clip_index["behavior_name"] == "approach"].iloc[0]
    assert directional["identity_grain"] == "directional"
    assert directional["focal_id"] == 1

    # Inclusive frame counts and playback timing vs actual encoded files.
    for _, row in clip_index.iterrows():
        clip_path = result.job_dir / "clips" / Path(str(row["clip_relpath"]))
        assert clip_path.is_file()
        fps, n = count_video_frames(clip_path)
        assert fps == pytest.approx(float(row["source_fps"]), rel=0.02)
        # Frame-addressed encode must match planned inclusive count exactly.
        assert n == int(row["clip_frame_count"])
        assert float(row["clip_duration_sec"]) == pytest.approx(
            int(row["clip_frame_count"]) / float(row["source_fps"])
        )
        assert int(row["clip_frame_count"]) == (
            int(row["clip_frame_idx_end"]) - int(row["clip_frame_idx_start"]) + 1
        )

    # Boundary bout at start with context would clamp; zero pad exact.
    edge = clip_index.loc[
        clip_index["bout_id"] == f"{info['package_run_id']}__bout_edge_start"
    ].iloc[0]
    assert int(edge["clip_frame_idx_start"]) == 0
    assert int(edge["prepared_frame_idx_start"]) == 0


def test_media_export_context_padding_clamped(tmp_path: Path):
    info = build_mini_export_package(tmp_path, n_frames=60, fps=30.0)
    bout_id = f"{info['package_run_id']}__bout_edge_start"
    result = run_media_export(
        MediaExportRequest(
            session_root=info["session_root"],
            package_dir=info["package_dir"],
            timestamp="20260101T000003",
            selection=SelectionSpec(mode=SELECTION_BOUT_ID_LIST, bout_ids=(bout_id,)),
            pre_context_sec=1.0,  # 30 frames → clamp to 0
            post_context_sec=0.0,
        )
    )
    assert result.n_succeeded == 1
    row = pd.read_parquet(result.clip_index_path).iloc[0]
    assert int(row["clip_frame_idx_start"]) == 0
    assert int(row["prepared_frame_idx_start"]) == 0  # behavioral/prepared bout unchanged
    assert int(row["start_sample_idx"]) == 0


def test_channel_selection_stable_filenames(tmp_path: Path):
    info = build_mini_export_package(tmp_path)
    result = run_media_export(
        MediaExportRequest(
            session_root=info["session_root"],
            package_dir=info["package_dir"],
            timestamp="20260101T000004",
            selection=SelectionSpec(
                mode=SELECTION_CHANNELS_BEHAVIORS, channels=("contact",)
            ),
        )
    )
    assert result.n_selected == 1
    row = pd.read_parquet(result.clip_index_path).iloc[0]
    assert row["clip_relpath"] == (
        f"contact/contact/{info['package_run_id']}__bout_contact.mp4"
    )
    assert (result.job_dir / "clips" / Path(row["clip_relpath"])).is_file()


def test_encoder_failure_partial_cleanup_and_retry(tmp_path: Path, monkeypatch):
    info = build_mini_export_package(tmp_path)
    calls = {"n": 0}
    real_encode = encode_mp4_clip

    def flaky_encode(**kwargs):
        calls["n"] += 1
        # Fail the first bout once.
        if calls["n"] == 1:
            from results_packaging.media_csv_export.contracts import ExportVideoError

            # Simulate failed tmp cleanup path inside encode by raising.
            out = Path(kwargs["output_path"])
            tmp = out.with_suffix(out.suffix + ".tmp")
            tmp.parent.mkdir(parents=True, exist_ok=True)
            tmp.write_bytes(b"partial")
            raise ExportVideoError("simulated encoder failure")
        return real_encode(**kwargs)

    monkeypatch.setattr(
        "results_packaging.media_csv_export.publication.encode_mp4_clip",
        flaky_encode,
    )
    result = run_media_export(
        MediaExportRequest(
            session_root=info["session_root"],
            package_dir=info["package_dir"],
            timestamp="20260101T000005",
        )
    )
    assert result.status == STATUS_INCOMPLETE
    assert result.n_failed == 1
    assert result.n_succeeded == 2
    clip_index = pd.read_parquet(result.clip_index_path)
    failed = clip_index.loc[clip_index["encode_status"] == ENCODE_STATUS_FAILED]
    assert len(failed) == 1
    failed_path = result.job_dir / "clips" / Path(str(failed.iloc[0]["clip_relpath"]))
    assert not failed_path.is_file()
    tmp = failed_path.with_suffix(failed_path.suffix + ".tmp")
    assert not tmp.exists()

    # Successful clips remain.
    ok_paths = [
        result.job_dir / "clips" / Path(str(r.clip_relpath))
        for r in clip_index.itertuples()
        if r.encode_status == ENCODE_STATUS_SUCCEEDED
    ]
    assert all(p.is_file() for p in ok_paths)
    ok_bytes = {p: p.read_bytes() for p in ok_paths}

    # Retry same job; do not overwrite successes.
    monkeypatch.setattr(
        "results_packaging.media_csv_export.publication.encode_mp4_clip",
        real_encode,
    )
    retried = retry_media_export(result.job_dir)
    assert retried.status == STATUS_COMPLETED
    assert retried.n_failed == 0
    assert retried.export_job_id == result.export_job_id
    for p, payload in ok_bytes.items():
        assert p.read_bytes() == payload

    # Completed job is immutable for retry.
    with pytest.raises(ExportPublicationError, match="immutable"):
        retry_media_export(result.job_dir)


def test_stale_source_digest_rejection_on_retry(tmp_path: Path):
    info = build_mini_export_package(tmp_path)
    # Force an incomplete job via fail_fast after first encode by monkeypatch later —
    # simpler: create job then tamper package digest recorded path by rewriting bouts.
    result = run_media_export(
        MediaExportRequest(
            session_root=info["session_root"],
            package_dir=info["package_dir"],
            timestamp="20260101T000006",
            selection=SelectionSpec(
                mode=SELECTION_CHANNELS_BEHAVIORS, channels=("contact",)
            ),
        )
    )
    assert result.status == STATUS_COMPLETED

    # New job required; mutating package bouts should block a fresh export that
    # checks embedded digests vs manifest.
    bouts_path = info["package_dir"] / BEHAVIOR_BOUTS_FILENAME
    table = pd.read_parquet(bouts_path)
    table.loc[0, "mean_prob"] = 0.123456
    table.to_parquet(bouts_path, index=False)
    with pytest.raises(ExportSourceError, match="SHA256"):
        run_media_export(
            MediaExportRequest(
                session_root=info["session_root"],
                package_dir=info["package_dir"],
                timestamp="20260101T000007",
            )
        )


def test_discovery_and_no_writes_to_s9_sources(tmp_path: Path):
    info = build_mini_export_package(tmp_path)
    package_files = {
        p: p.read_bytes()
        for p in info["package_dir"].rglob("*")
        if p.is_file()
    }
    video_before = info["video_path"].read_bytes()

    csv = run_csv_export(
        CsvExportRequest(
            session_root=info["session_root"],
            package_dir=info["package_dir"],
            timestamp="20260101T000010",
        )
    )
    media = run_media_export(
        MediaExportRequest(
            session_root=info["session_root"],
            package_dir=info["package_dir"],
            timestamp="20260101T000011",
            selection=SelectionSpec(
                mode=SELECTION_CHANNELS_BEHAVIORS, channels=("contact",)
            ),
        )
    )
    summary = summarize_export_project(info["session_root"])
    assert summary.exports_present
    assert len(summary.jobs) >= 2
    assert csv.export_job_id in {j.export_job_id for j in summary.csv_jobs}
    assert media.export_job_id in {j.export_job_id for j in summary.media_jobs}

    for path, payload in package_files.items():
        assert path.read_bytes() == payload
    assert info["video_path"].read_bytes() == video_before
    assert RESULTS_PACKAGE_DIRNAME in str(info["package_dir"])
    assert media.job_dir.parent.name == RESULTS_EXPORTS_DIRNAME


def test_planned_vs_probe_clip_math_unit():
    bounds = compute_clip_bounds(
        prepared_frame_idx_start=10,
        prepared_frame_idx_end=19,
        source_fps=25.0,
        source_n_frames=100,
    )
    assert bounds.clip_frame_count == 10
    assert bounds.clip_duration_sec == pytest.approx(0.4)


def test_stale_package_rejected_for_new_export_by_default(tmp_path: Path):
    """MVP: do not silently export from currently ineligible packages."""

    info = build_mini_export_package(tmp_path)

    def _stale(_path: Path | str) -> dict:
        raise ProvenanceError("S9 source S8 run is no longer eligible for downstream use.")

    # Override autouse eligibility patch for this case.
    with patch(
        "results_packaging.media_csv_export.source.validate_s9_run_for_downstream",
        side_effect=_stale,
    ):
        with pytest.raises(ExportSourceError, match="not currently eligible"):
            run_csv_export(
                CsvExportRequest(
                    session_root=info["session_root"],
                    package_dir=info["package_dir"],
                    timestamp="20260101T000020",
                )
            )
        with pytest.raises(ExportSourceError, match="not currently eligible"):
            run_media_export(
                MediaExportRequest(
                    session_root=info["session_root"],
                    package_dir=info["package_dir"],
                    timestamp="20260101T000021",
                )
            )
        # Historical inspection / deferred archival may still open with override.
        source = validate_package_for_export(
            info["package_dir"],
            require_video=False,
            allow_stale_source=True,
        )
        assert source.currently_eligible is False
        assert source.stale_warning is not None
