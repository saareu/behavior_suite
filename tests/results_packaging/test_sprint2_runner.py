"""Sprint 2 headless runner tests (in-memory package; no S9 publish)."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

from behavioral_arbitration.contracts import STATUS_COMPLETED
from results_packaging.config import ExecutionSettings, ResultsPackagingConfig
from results_packaging.contracts import REVIEW_EMIT_PROFILE_ID
from results_packaging.runner import (
    ResultsPackagingRequest,
    run_results_packaging_sprint2,
)
from tests.results_packaging._fixtures import (
    build_packaging_session,
    make_multi_channel_bouts,
    sha256_file,
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
    )


def test_sprint2_default_package_unfiltered_and_no_upstream_writes(tmp_path) -> None:
    fixture = build_packaging_session(tmp_path, n_prepared=10, prepared_fps=30.0)
    s8 = make_multi_channel_bouts()
    s8 = s8[s8["end_sample_idx"] < fixture["n_dense"]].reset_index(drop=True)
    s8.to_parquet(fixture["bouts_path"], index=False)

    states_sha = sha256_file(fixture["states_path"])
    bouts_sha = sha256_file(fixture["bouts_path"])
    sync_sha = sha256_file(fixture["prepared_sync"])

    with ExitStack() as stack:
        for p in _eligible_patches(fixture):
            stack.enter_context(p)
        result = run_results_packaging_sprint2(
            ResultsPackagingRequest(
                session_root=fixture["session_root"],
                s8_run_dir=fixture["s8_run_dir"],
                package_run_id="results_package__unit",
            )
        )

    assert result.to_metadata()["published"] is False
    assert len(result.behavior_bouts) == len(s8)
    assert REVIEW_EMIT_PROFILE_ID not in result.export_profiles
    assert not (fixture["session_root"] / "results_package").exists()
    assert sha256_file(fixture["states_path"]) == states_sha
    assert sha256_file(fixture["bouts_path"]) == bouts_sha
    assert sha256_file(fixture["prepared_sync"]) == sync_sha
    assert not result.ethogram_summary.empty


def test_sprint2_optional_profile_explicit_only(tmp_path) -> None:
    fixture = build_packaging_session(tmp_path, n_prepared=10, prepared_fps=30.0)
    s8 = make_multi_channel_bouts()
    s8 = s8[s8["end_sample_idx"] < fixture["n_dense"]].reset_index(drop=True)
    s8.to_parquet(fixture["bouts_path"], index=False)

    cfg = ResultsPackagingConfig(
        execution=ExecutionSettings(
            selected_export_profiles=(REVIEW_EMIT_PROFILE_ID,),
        )
    )
    with ExitStack() as stack:
        for p in _eligible_patches(fixture):
            stack.enter_context(p)
        result = run_results_packaging_sprint2(
            ResultsPackagingRequest(
                session_root=fixture["session_root"],
                s8_run_dir=fixture["s8_run_dir"],
                config=cfg,
            )
        )
    assert REVIEW_EMIT_PROFILE_ID in result.export_profiles
    profile = result.export_profiles[REVIEW_EMIT_PROFILE_ID]
    assert profile.n_included + profile.n_excluded == len(result.behavior_bouts)
    assert len(result.behavior_bouts) == len(s8)
