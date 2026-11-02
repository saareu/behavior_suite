"""Unit tests for legacy_framewise_v1 CSV format and joins."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from results_packaging.legacy_framewise_csv import (
    LEGACY_FRAMEWISE_COLUMNS,
    LEGACY_FRAMEWISE_PROFILE_ID,
    LegacyFramewiseEligibilityError,
    build_legacy_framewise_table,
    format_coordinate,
    validate_legacy_framewise_csv_bytes,
    write_legacy_framewise_csv,
)
from results_packaging.media_csv_export.config import CsvExportSettings

FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "legacy_framewise"
    / "header_and_sample_rows.csv"
)


def _sync(n: int = 4, *, start_raw: int = 100, many_to_one: bool = True) -> pd.DataFrame:
    dense = np.arange(n, dtype=np.int64)
    prepared = dense.copy()
    if many_to_one and n >= 4:
        prepared[2] = prepared[1]  # dense 1 and 2 share prepared/original
    raw = prepared + int(start_raw)
    return pd.DataFrame(
        {
            "schema_version": ["results_package_sync_v1"] * n,
            "sample_idx": dense,
            "dense_frame_idx": dense,
            "time_sec": dense / 120.0,
            "prepared_frame_idx": prepared,
            "prepared_time_sec": prepared / 120.0,
            "is_exact_source_sample": [True] * n,
            "timing_residual_sec": [0.0] * n,
            "raw_decode_frame_idx": raw,
        }
    )


def _states(n: int = 4) -> pd.DataFrame:
    rows = []
    for d in range(n):
        rows.append(
            {
                "dense_frame_idx": d,
                "focal_id": 0,
                "partner_id": 1,
                "behavior_label": "approach" if d % 2 == 0 else "engaged",
                "pred_side_by_side": 1 if d == 1 else 0,
            }
        )
        rows.append(
            {
                "dense_frame_idx": d,
                "focal_id": 1,
                "partner_id": 0,
                "behavior_label": "retract",
                "pred_side_by_side": 1 if d == 1 else 0,
            }
        )
    return pd.DataFrame(rows)


def _pose(n: int = 4) -> pd.DataFrame:
    nodes = ["headstage", "neck", "nose", "spine_base", "tail_base"]
    rows = []
    for sample in range(n):
        for track in (0, 1):
            for i, node in enumerate(nodes):
                if node == "headstage" and track == 1 or sample == 2 and node == "nose" and track == 0:
                    x = y = np.nan
                else:
                    x = 10.4 + i + track * 100 + sample
                    y = 20.6 + i + track * 50 + sample
                rows.append(
                    {
                        "sample_idx": sample,
                        "track": track,
                        "node": node,
                        "x": x,
                        "y": y,
                        "provenance": 2,
                    }
                )
    return pd.DataFrame(rows)


def test_reference_fixture_header_and_crlf():
    payload = FIXTURE.read_bytes()
    assert b"\r\n" in payload
    qc = validate_legacy_framewise_csv_bytes(payload)
    assert qc["header"] == list(LEGACY_FRAMEWISE_COLUMNS)
    assert qc["n_header_columns"] == 23
    assert qc["n_data_rows"] == 7


def test_coordinate_formatting_convention():
    assert format_coordinate(406.4) == "406.0"
    assert format_coordinate(406.5) == "406.0" or format_coordinate(406.5) == "407.0"
    assert format_coordinate(np.nan) == ""
    assert format_coordinate(None) == ""
    assert format_coordinate(104.0) == "104.0"


def test_builder_original_frame_mapping_and_many_to_one():
    built = build_legacy_framewise_table(
        sync_index=_sync(start_raw=37511),
        behavior_states=_states(),
        final_pose=_pose(),
    )
    assert built.n_rows == 4
    assert list(built.table.columns) == list(LEGACY_FRAMEWISE_COLUMNS)
    assert int(built.table.loc[0, "frame"]) == 37511
    assert int(built.table.loc[0, "dense_frame_idx"]) == 0
    # many-to-one: dense 1 and 2 share original frame
    assert int(built.table.loc[1, "frame"]) == int(built.table.loc[2, "frame"])
    assert built.n_duplicate_original_frame_rows >= 1
    assert built.table.loc[0, "behaviour_label_focal_0"] == "approach"
    assert built.table.loc[0, "behaviour_label_focal_1"] == "retract"
    assert int(built.table.loc[1, "pred_side_by_side"]) == 1
    # missing nose_0 on sample 2 → empty after write
    assert pd.isna(built.table.loc[2, "nose_0_x"])


def test_writer_crlf_and_empty_missing_coords(tmp_path: Path):
    built = build_legacy_framewise_table(
        sync_index=_sync(start_raw=10),
        behavior_states=_states(),
        final_pose=_pose(),
    )
    out = write_legacy_framewise_csv(tmp_path / "legacy_framewise_v1.csv", built.table)
    payload = out.read_bytes()
    qc = validate_legacy_framewise_csv_bytes(payload)
    assert qc["n_data_rows"] == 4
    text = payload.decode("utf-8")
    lines = text.split("\r\n")
    assert lines[0] == ",".join(LEGACY_FRAMEWISE_COLUMNS)
    # row dense=2 has empty nose_0_x,nose_0_y (two consecutive commas in pose region)
    assert ",," in lines[3]


def test_missing_raw_decode_is_eligibility_error():
    sync = _sync().drop(columns=["raw_decode_frame_idx"])
    with pytest.raises(LegacyFramewiseEligibilityError, match="raw_decode_frame_idx"):
        build_legacy_framewise_table(
            sync_index=sync,
            behavior_states=_states(),
            final_pose=_pose(),
        )


def test_default_csv_settings_profile_is_legacy():
    settings = CsvExportSettings().validated()
    assert settings.format_profile == LEGACY_FRAMEWISE_PROFILE_ID


def test_both_mouse_directions_and_sbs_conflict_rejected():
    states = _states()
    states.loc[states["focal_id"] == 1, "pred_side_by_side"] = 0
    states.loc[
        (states["focal_id"] == 0) & (states["dense_frame_idx"] == 1),
        "pred_side_by_side",
    ] = 1
    with pytest.raises(Exception, match="pred_side_by_side disagrees"):
        build_legacy_framewise_table(
            sync_index=_sync(),
            behavior_states=states,
            final_pose=_pose(),
        )


def test_dense_gap_is_blocked_without_renumbering():
    sync = _sync(n=4)
    sync.loc[2, "dense_frame_idx"] = 99
    with pytest.raises(Exception, match="not contiguous|cardinality|dense_frame_idx"):
        build_legacy_framewise_table(
            sync_index=sync,
            behavior_states=_states(),
            final_pose=_pose(),
        )


def test_dense_duplicate_is_blocked():
    sync = _sync(n=4)
    sync.loc[3, "dense_frame_idx"] = 1
    with pytest.raises(Exception, match="unique|duplicates"):
        build_legacy_framewise_table(
            sync_index=sync,
            behavior_states=_states(),
            final_pose=_pose(),
        )


def test_conflicting_focal_labels_rejected():
    states = _states()
    dup = states.loc[(states["focal_id"] == 0) & (states["dense_frame_idx"] == 0)].copy()
    dup["behavior_label"] = "nose_to_nose"
    states = pd.concat([states, dup], ignore_index=True)
    with pytest.raises(Exception, match="Conflicting behaviour labels"):
        build_legacy_framewise_table(
            sync_index=_sync(),
            behavior_states=states,
            final_pose=_pose(),
        )


def test_behavior_states_outside_sync_grid_rejected():
    states = _states()
    extra = states.loc[states["dense_frame_idx"] == 0].copy()
    extra["dense_frame_idx"] = 50
    states = pd.concat([states, extra], ignore_index=True)
    with pytest.raises(Exception, match="outside the authoritative sync grid"):
        build_legacy_framewise_table(
            sync_index=_sync(),
            behavior_states=states,
            final_pose=_pose(),
        )


def test_half_even_rounding_and_dot_zero_format():
    assert format_coordinate(406.5) == "406.0"
    assert format_coordinate(405.5) == "406.0"
    assert format_coordinate(1.5) == "2.0"
    assert format_coordinate(2.5) == "2.0"
    assert format_coordinate(104.49) == "104.0"
    assert format_coordinate(104.5) == "104.0" or format_coordinate(104.5) == "105.0"


def test_builder_dense_diagnostics_contiguous_from_zero():
    built = build_legacy_framewise_table(
        sync_index=_sync(n=4, start_raw=37511),
        behavior_states=_states(),
        final_pose=_pose(),
    )
    assert built.diagnostics["dense_min"] == 0
    assert built.diagnostics["dense_max"] == 3
    assert built.diagnostics["n_unique_dense"] == 4
    assert built.diagnostics["n_duplicate_dense"] == 0
    assert built.diagnostics["n_missing_dense"] == 0
    assert built.diagnostics["dense_contiguous_from_zero"] is True
    assert built.diagnostics["nonzero_trim_offset"] is True
    assert built.diagnostics["trim_offset_frames"] == 37511
    assert built.diagnostics["many_to_one_original_frames"] is True
    assert built.diagnostics["focal0_present_rows"] == 4
    assert built.diagnostics["focal1_present_rows"] == 4


def test_pose_alignment_rejects_sample_mismatch():
    from results_packaging.legacy_framewise_csv.source import _validate_pose_alignment

    pose = _pose(n=3)
    with pytest.raises(Exception, match="sample_idx set does not match"):
        _validate_pose_alignment(pose, sync_index=_sync(n=4))


def test_no_latest_s4_fallback_without_lineage_or_explicit(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    from results_packaging.legacy_framewise_csv import (
        LegacyFramewiseEligibilityError,
    )
    from results_packaging.legacy_framewise_csv.source import _resolve_final_pose

    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    (pkg_dir / "package_manifest.json").write_text("{}", encoding="utf-8")
    package = SimpleNamespace(package_dir=pkg_dir)

    def _boom(*_a, **_k):  # pragma: no cover - must never be called
        raise AssertionError("latest S4 fallback must not be invoked")

    monkeypatch.setattr(
        "pose_finalization.run_discovery.find_latest_eligible_s4_run",
        _boom,
        raising=False,
    )
    with pytest.raises(LegacyFramewiseEligibilityError, match="explicit pose_run_dir"):
        _resolve_final_pose(
            root=tmp_path,
            package=package,
            pose_run_dir=None,
            sync_index=_sync(),
        )


def _write_pose_run(
    session_root: Path,
    *,
    run_id: str,
    n: int = 4,
    pose_table: pd.DataFrame | None = None,
) -> Path:
    run_dir = session_root / "pose_finalization" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    table = pose_table if pose_table is not None else _pose(n=n)
    table.to_parquet(run_dir / "final_pose.parquet", index=False)
    (run_dir / "run_meta.json").write_text(
        __import__("json").dumps(
            {
                "run_id": run_id,
                "status": "completed",
                "session_root": str(session_root),
                "source_s3": {
                    "run_dir": str(session_root / "tracking_correction" / "s3"),
                    "run_id": "s3",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    return run_dir


def test_missing_lineage_requires_explicit_selection(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    from results_packaging.legacy_framewise_csv import LegacyFramewiseEligibilityError
    from results_packaging.legacy_framewise_csv.source import (
        _resolve_final_pose,
        inspect_package_pose_lineage,
    )

    pkg_dir = tmp_path / "pkg"
    pkg_dir.mkdir()
    (pkg_dir / "package_manifest.json").write_text("{}", encoding="utf-8")
    (pkg_dir / "run_meta.json").write_text("{}", encoding="utf-8")
    info = inspect_package_pose_lineage(pkg_dir)
    assert info.available is False
    assert "explicit" in info.reason.lower() or "unavailable" in info.reason.lower()

    package = SimpleNamespace(package_dir=pkg_dir, package_manifest={}, run_meta={})
    monkeypatch.setattr(
        "results_packaging.legacy_framewise_csv.source.validate_s4_run_for_downstream",
        lambda _d: {"status": "completed"},
    )
    with pytest.raises(LegacyFramewiseEligibilityError, match="explicit pose_run_dir"):
        _resolve_final_pose(
            root=tmp_path,
            package=package,
            pose_run_dir=None,
            sync_index=_sync(),
        )


def test_explicit_compatible_pose_selection_succeeds(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    from results_packaging.legacy_framewise_csv.source import _resolve_final_pose

    session = tmp_path / "session"
    session.mkdir()
    pose_dir = _write_pose_run(session, run_id="pose_finalization__explicit_ok")
    pkg_dir = session / "pkg"
    pkg_dir.mkdir()
    (pkg_dir / "package_manifest.json").write_text("{}", encoding="utf-8")
    package = SimpleNamespace(package_dir=pkg_dir, package_manifest={}, run_meta={})
    monkeypatch.setattr(
        "results_packaging.legacy_framewise_csv.source.validate_s4_run_for_downstream",
        lambda _d: {"status": "completed"},
    )
    resolved = _resolve_final_pose(
        root=session,
        package=package,
        pose_run_dir=pose_dir,
        sync_index=_sync(),
    )
    assert resolved.selection_mode == "explicit"
    assert resolved.pose_run_id == "pose_finalization__explicit_ok"
    assert resolved.pose_run_dir == pose_dir.resolve()
    assert resolved.final_pose_sha256
    assert resolved.alignment["timeline_match"] is True


def test_explicit_incompatible_pose_selection_rejected(tmp_path: Path, monkeypatch):
    from types import SimpleNamespace

    from results_packaging.legacy_framewise_csv import LegacyFramewiseSourceError
    from results_packaging.legacy_framewise_csv.source import _resolve_final_pose

    session = tmp_path / "session"
    session.mkdir()
    bad_pose = _pose(n=2)  # timeline length mismatch vs sync n=4
    pose_dir = _write_pose_run(
        session, run_id="pose_finalization__bad_timeline", pose_table=bad_pose
    )
    pkg_dir = session / "pkg"
    pkg_dir.mkdir()
    (pkg_dir / "package_manifest.json").write_text("{}", encoding="utf-8")
    package = SimpleNamespace(package_dir=pkg_dir, package_manifest={}, run_meta={})
    monkeypatch.setattr(
        "results_packaging.legacy_framewise_csv.source.validate_s4_run_for_downstream",
        lambda _d: {"status": "completed"},
    )
    with pytest.raises(LegacyFramewiseSourceError, match="sample_idx set does not match"):
        _resolve_final_pose(
            root=session,
            package=package,
            pose_run_dir=pose_dir,
            sync_index=_sync(n=4),
        )


def test_lineage_pinned_pose_and_conflict_rejection(tmp_path: Path, monkeypatch):
    import json
    from types import SimpleNamespace

    from results_packaging.legacy_framewise_csv import LegacyFramewiseSourceError
    from results_packaging.legacy_framewise_csv.source import (
        _resolve_final_pose,
        inspect_package_pose_lineage,
    )

    session = tmp_path / "session"
    session.mkdir()
    lineage_pose = _write_pose_run(session, run_id="pose_finalization__lineage")
    other_pose = _write_pose_run(session, run_id="pose_finalization__other")
    s6_dir = session / "combined_features" / "combined_features__1"
    s6_dir.mkdir(parents=True)
    (s6_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "source_s4": {
                    "run_id": "pose_finalization__lineage",
                    "run_dir": str(lineage_pose),
                    "final_pose": str(lineage_pose / "final_pose.parquet"),
                    "final_pose_sha256": None,
                    "session_root": str(session),
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )
    pkg_dir = session / "results_package" / "pkg"
    pkg_dir.mkdir(parents=True)
    manifest = {
        "source_references": {
            "source_s6": {"run_id": s6_dir.name, "run_dir": str(s6_dir)}
        }
    }
    (pkg_dir / "package_manifest.json").write_text(
        json.dumps(manifest) + "\n", encoding="utf-8"
    )
    (pkg_dir / "run_meta.json").write_text(
        json.dumps({"source_s6": {"run_id": s6_dir.name, "run_dir": str(s6_dir)}})
        + "\n",
        encoding="utf-8",
    )

    info = inspect_package_pose_lineage(pkg_dir)
    assert info.available is True
    assert info.run_id == "pose_finalization__lineage"

    package = SimpleNamespace(
        package_dir=pkg_dir,
        package_manifest=manifest,
        run_meta={"source_s6": {"run_dir": str(s6_dir)}},
    )
    monkeypatch.setattr(
        "results_packaging.legacy_framewise_csv.source.validate_s4_run_for_downstream",
        lambda _d: {"status": "completed"},
    )
    resolved = _resolve_final_pose(
        root=session,
        package=package,
        pose_run_dir=None,
        sync_index=_sync(),
    )
    assert resolved.selection_mode == "lineage_s6"
    assert resolved.pose_run_id == "pose_finalization__lineage"

    with pytest.raises(LegacyFramewiseSourceError, match="disagrees with S6 lineage"):
        _resolve_final_pose(
            root=session,
            package=package,
            pose_run_dir=other_pose,
            sync_index=_sync(),
        )


def test_previous_csv_profile_still_supported():
    settings = CsvExportSettings(format_profile="package_bouts_automated").validated()
    assert settings.format_profile == "package_bouts_automated"
    assert settings.pose_run_dir is None
    with pytest.raises(Exception, match="pose_run_dir is only valid"):
        CsvExportSettings(
            format_profile="package_bouts_automated",
            pose_run_dir="/tmp/pose",
        ).validated()


def test_list_pose_candidates_never_auto_selected(tmp_path: Path, monkeypatch):
    from results_packaging.legacy_framewise_csv import list_pose_run_candidates
    from ui.controllers.results_export_support import build_legacy_pose_ui_state

    session = tmp_path / "session"
    session.mkdir()
    older = _write_pose_run(session, run_id="pose_finalization__20260101T000001")
    newer = _write_pose_run(session, run_id="pose_finalization__20260102T000002")

    class _Run:
        def __init__(self, run_id: str, run_dir: Path):
            self.run_id = run_id
            self.run_dir = run_dir
            self.status = "completed"
            self.classification = "eligible"

    class _Summary:
        completed_runs = (
            _Run(newer.name, newer),
            _Run(older.name, older),
        )

    monkeypatch.setattr(
        "pose_finalization.run_discovery.summarize_pose_finalization_project",
        lambda _root: _Summary(),
    )
    candidates = list_pose_run_candidates(session)
    assert [c.run_id for c in candidates] == [newer.name, older.name]

    pkg_dir = session / "pkg"
    pkg_dir.mkdir()
    (pkg_dir / "package_manifest.json").write_text("{}", encoding="utf-8")
    state = build_legacy_pose_ui_state(
        session_root=session,
        package_dir=pkg_dir,
        format_profile=LEGACY_FRAMEWISE_PROFILE_ID,
        selected_pose_run_dir=None,
    )
    assert state.requires_explicit_pose is True
    assert state.selected_pose_run_dir is None
    assert len(state.candidates) == 2
