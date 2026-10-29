"""Dependency manifest + Sprint 1 runner integration tests."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from behavioral_arbitration.contracts import STATUS_COMPLETED
from project.validation_memo import artifact_fingerprint
from results_packaging.contracts import DependencyError
from results_packaging.dependencies import (
    build_dependency_manifest,
    dependency_fingerprint_token,
)
from results_packaging.eligibility import validate_s8_source_for_packaging
from results_packaging.runner import (
    ResultsPackagingRequest,
    run_results_packaging_sprint1,
)
from tests.results_packaging._fixtures import build_packaging_session, sha256_file


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
    )


def test_dependency_fingerprint_invalidation(tmp_path: Path) -> None:
    info = build_packaging_session(tmp_path, n_prepared=4, prepared_fps=30.0)
    with ExitStack() as stack:
        for p in _eligible_patches(info):
            stack.enter_context(p)
        source = validate_s8_source_for_packaging(
            info["session_root"],
            s8_run_dir=info["s8_run_dir"],
        )
    manifest = build_dependency_manifest(source)
    token_before = dependency_fingerprint_token(manifest)
    assert manifest.artifact_map()["prepared_sync"].sha256 == info["prepared_sync_sha256"]

    # Modify sync bytes → live digest check fails (no silent replace).
    assert info["prepared_sync"] is not None
    info["prepared_sync"].write_bytes(info["prepared_sync"].read_bytes() + b"x")
    with pytest.raises(DependencyError, match="digest mismatch"):
        manifest.assert_required_resolvable()

    # Fingerprint of the path also changes.
    fp_after = artifact_fingerprint(info["prepared_sync"])
    assert fp_after != manifest.artifact_map()["prepared_sync"].fingerprint
    assert token_before[0] == manifest.schema_version


def test_runner_sprint1_in_memory_no_publish(tmp_path: Path) -> None:
    info = build_packaging_session(
        tmp_path,
        n_prepared=5,
        prepared_fps=30.0,
        bout_intervals=[(0, 1), (2, 2)],
    )
    before_states = sha256_file(info["states_path"])
    before_bouts = sha256_file(info["bouts_path"])
    before_sync = sha256_file(info["prepared_sync"])

    with ExitStack() as stack:
        for p in _eligible_patches(info):
            stack.enter_context(p)
        result = run_results_packaging_sprint1(
            ResultsPackagingRequest(
                session_root=info["session_root"],
                s8_run_dir=info["s8_run_dir"],
            )
        )

    assert result.schema_version.startswith("results_package_sprint1_")
    assert result.sync.n_dense == info["n_dense"]
    assert result.sync.n_dense != result.sync.n_prepared_source
    assert len(result.bout_boundary_map) == 2
    assert list(result.bout_boundary_map["bout_duration_frames"]) == [2, 1]
    assert result.to_metadata()["published"] is False
    assert not (info["session_root"] / "results_package").exists()
    assert sha256_file(info["states_path"]) == before_states
    assert sha256_file(info["bouts_path"]) == before_bouts
    assert sha256_file(info["prepared_sync"]) == before_sync
    assert result.settings_used["notes"]["default_package_retains_all_s8_bouts"] is True
    assert result.settings_used["notes"]["review_emit_seconds_use_120hz"] is True


def test_review_emit_defaults_recorded_not_auto_applied(tmp_path: Path) -> None:
    from results_packaging.config import ExecutionSettings, load_results_packaging_config
    from results_packaging.contracts import (
        REVIEW_EMIT_PROFILE_ID,
        REVIEW_EMIT_V1_INCLUDE_CONTACT,
        REVIEW_EMIT_V1_INCLUDE_ENGAGED,
    )

    cfg = load_results_packaging_config(None)
    assert cfg.review_emit_v1_defaults.include_contact is REVIEW_EMIT_V1_INCLUDE_CONTACT
    assert cfg.review_emit_v1_defaults.include_engaged is REVIEW_EMIT_V1_INCLUDE_ENGAGED
    assert cfg.review_emit_v1_defaults.include_contact is False
    assert cfg.review_emit_v1_defaults.include_engaged is True
    # Default YAML leaves selected_export_profiles empty (not auto-applied).
    assert cfg.execution.selected_export_profiles == ()
    # Sprint 2 allows explicit selection of review_emit_v1.
    assert (
        ExecutionSettings(selected_export_profiles=(REVIEW_EMIT_PROFILE_ID,))
        .validated()
        .selected_export_profiles
        == (REVIEW_EMIT_PROFILE_ID,)
    )
    with pytest.raises(Exception, match="Unsupported export profile"):
        ExecutionSettings(selected_export_profiles=("not_a_real_profile",)).validated()
