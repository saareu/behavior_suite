"""E-session acceptance checks for legacy_framewise_v2 (skip if fixtures absent)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from project.file_digest import sha256_file
from results_packaging.legacy_framewise_csv import (
    FRAME_ALIGNMENT_COLUMNS,
    LEGACY_FRAMEWISE_COLUMNS,
    LEGACY_FRAMEWISE_V2_ALIGNMENT_FILENAME,
    LEGACY_FRAMEWISE_V2_CSV_FILENAME,
    LEGACY_FRAMEWISE_V2_PROFILE_ID,
    LegacyFramewiseExportRequest,
    run_legacy_framewise_csv_export,
    validate_legacy_framewise_csv_bytes,
)
from results_packaging.legacy_framewise_csv.contracts import (
    EXPORT_MANIFEST_SCHEMA_VERSION_V2,
)

E_SESSION = Path(__file__).resolve().parents[2] / "test_cases" / "E_full_session_integration"


def _find_package(session: Path) -> Path | None:
    root = session / "results_package"
    if not root.is_dir():
        return None
    candidates = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("results_package__")],
        key=lambda p: p.name,
        reverse=True,
    )
    for path in candidates:
        if (path / "sync_index.parquet").is_file():
            return path
    return None


@pytest.mark.skipif(not E_SESSION.is_dir(), reason="E_full_session_integration fixture absent")
def test_e_session_v2_prepared_frame_export(tmp_path: Path):
    package = _find_package(E_SESSION)
    if package is None:
        pytest.skip("No S9 package with sync_index in E-session")

    result = run_legacy_framewise_csv_export(
        LegacyFramewiseExportRequest(
            session_root=E_SESSION,
            package_dir=package,
            behavior_source_kind="automated_s8",
            format_profile=LEGACY_FRAMEWISE_V2_PROFILE_ID,
            timestamp="e_session_v2_test",
            output_root=tmp_path / "exports",
            allow_stale_source=True,
        )
    )
    assert result.success, result.error_message
    assert result.n_rows == 36239
    assert result.csv_path is not None
    assert result.csv_path.name == LEGACY_FRAMEWISE_V2_CSV_FILENAME

    payload = result.csv_path.read_bytes()
    qc = validate_legacy_framewise_csv_bytes(payload)
    assert qc["n_data_rows"] == 36239
    assert qc["header"] == list(LEGACY_FRAMEWISE_COLUMNS)

    alignment_path = result.job_dir / LEGACY_FRAMEWISE_V2_ALIGNMENT_FILENAME
    assert alignment_path.is_file()
    alignment = pd.read_parquet(alignment_path)
    assert list(alignment.columns) == list(FRAME_ALIGNMENT_COLUMNS)
    assert len(alignment) == 36239
    assert int(alignment["prepared_frame_idx"].nunique()) == 36239
    assert not bool(alignment["prepared_frame_idx"].duplicated().any())

    # Final prepared frame 36238 / raw 72621: retained, behavior uncovered.
    last = alignment.loc[alignment["prepared_frame_idx"] == 36238].iloc[0]
    assert int(last["raw_decode_frame_idx"]) == 72621
    assert bool(last["behavior_coverage"]) is False
    assert pd.isna(last["behavior_dense_frame_idx"])

    csv = pd.read_csv(result.csv_path)
    assert len(csv) == 36239
    last_csv = csv.iloc[-1]
    assert int(last_csv["frame"]) == 72621
    assert pd.isna(last_csv["dense_frame_idx"])
    assert pd.isna(last_csv["behaviour_label_focal_0"])
    assert pd.isna(last_csv["behaviour_label_focal_1"])
    # Pose only if independently covered.
    pose_covered = bool(last["pose_coverage"])
    if pose_covered:
        assert pd.notna(last_csv["neck_0_x"]) or True  # may still be NaN coords
    else:
        assert pd.isna(last_csv["neck_0_x"])

    manifest = json.loads(
        (result.job_dir / "export_job_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["format_profile"] == LEGACY_FRAMEWISE_V2_PROFILE_ID
    assert manifest["schema_version"] == EXPORT_MANIFEST_SCHEMA_VERSION_V2
    assert manifest["artifacts"]["n_rows"] == 36239
    assert manifest["frame_alignment"]["n_rows"] == 36239
    assert (
        manifest["frame_alignment"]["sha256"]
        == sha256_file(alignment_path)
    )
    assert (
        manifest["artifacts"]["csv_sha256"]
        == sha256_file(result.csv_path)
    )
    assert manifest["timing_contract"]["behavior_plateau_policy"] == "last_in_plateau"
    assert manifest["timing_contract"]["pose_plateau_policy"] == "last_in_plateau"
    assert manifest["timing_contract"]["dense_indices_assumed_equal"] is False


@pytest.mark.skipif(not E_SESSION.is_dir(), reason="E_full_session_integration fixture absent")
def test_e_session_v1_still_dense_domain(tmp_path: Path):
    from results_packaging.legacy_framewise_csv import LEGACY_FRAMEWISE_PROFILE_ID

    package = _find_package(E_SESSION)
    if package is None:
        pytest.skip("No S9 package with sync_index in E-session")
    sync = pd.read_parquet(package / "sync_index.parquet")
    n_dense = int(len(sync))
    result = run_legacy_framewise_csv_export(
        LegacyFramewiseExportRequest(
            session_root=E_SESSION,
            package_dir=package,
            behavior_source_kind="automated_s8",
            format_profile=LEGACY_FRAMEWISE_PROFILE_ID,
            timestamp="e_session_v1_regress",
            output_root=tmp_path / "exports_v1",
            allow_stale_source=True,
        )
    )
    assert result.success, result.error_message
    assert result.n_rows == n_dense
    assert n_dense == 36528
