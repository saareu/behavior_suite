"""S8 source eligibility / schema / sync-video validation for S9."""

from __future__ import annotations

from contextlib import ExitStack
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from behavioral_arbitration.contracts import STATUS_COMPLETED
from behavioral_inference.contracts import ProvenanceError
from results_packaging.contracts import IntakeError
from results_packaging.eligibility import (
    validate_s8_source_for_packaging,
)
from tests.results_packaging._fixtures import (
    build_packaging_session,
    make_states,
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


def test_eligible_source_validation(tmp_path: Path) -> None:
    info = build_packaging_session(tmp_path, n_prepared=5, prepared_fps=30.0)
    with ExitStack() as stack:
        for p in _eligible_patches(info):
            stack.enter_context(p)
        result = validate_s8_source_for_packaging(
            info["session_root"],
            s8_run_dir=info["s8_run_dir"],
        )
    assert result.s8_eligible
    assert result.source_s7_run_id == info["s7_run_id"]
    assert result.source_s6_run_id == info["s6_run_id"]
    assert result.n_dense_samples == info["n_dense"]
    assert result.sync_video.prepared_sync_path == info["prepared_sync"].resolve()
    assert result.sync_video.prepared_video_path == info["prepared_video"].resolve()
    assert result.states_schema_version.startswith("behavioral_arbitration_states_")
    assert result.bouts_schema_version.startswith("behavioral_arbitration_bouts_")


def test_stale_s8_source_rejected(tmp_path: Path) -> None:
    info = build_packaging_session(tmp_path)
    with (
        patch(
            "results_packaging.eligibility.validate_s8_run_for_downstream",
            side_effect=ProvenanceError("S6 digest drifted"),
        ),
        pytest.raises(IntakeError, match="not currently eligible|stale"),
    ):
        validate_s8_source_for_packaging(
            info["session_root"],
            s8_run_dir=info["s8_run_dir"],
        )


def test_linked_s7_s6_identity_mismatch(tmp_path: Path) -> None:
    info = build_packaging_session(tmp_path)

    def fake_validate(path: Path | str) -> dict[str, Any]:
        return {
            "s8_run_dir": Path(path).resolve(),
            "status": STATUS_COMPLETED,
            "source_s7_run_dir": info["s7_run_dir"].resolve(),
            "source_s7_run_id": "WRONG_S7",
            "source_s6_run_dir": info["s6_run_dir"].resolve(),
            "source_s6_run_id": info["s6_run_id"],
        }

    with (
        patch(
            "results_packaging.eligibility.validate_s8_run_for_downstream",
            side_effect=fake_validate,
        ),
        patch(
            "results_packaging.eligibility.is_s8_run_eligible_for_downstream",
            return_value=True,
        ),
        pytest.raises(IntakeError, match="Linked S7 identity mismatch"),
    ):
        validate_s8_source_for_packaging(
            info["session_root"],
            s8_run_dir=info["s8_run_dir"],
        )


def test_missing_prepared_video(tmp_path: Path) -> None:
    info = build_packaging_session(tmp_path, include_video=False)
    with ExitStack() as stack:
        for p in _eligible_patches(info):
            stack.enter_context(p)
        with pytest.raises(IntakeError, match="Prepared video missing"):
            validate_s8_source_for_packaging(
                info["session_root"],
                s8_run_dir=info["s8_run_dir"],
                require_prepared_video=True,
            )


def test_missing_and_modified_sync(tmp_path: Path) -> None:
    info = build_packaging_session(tmp_path, include_sync=True)
    assert info["prepared_sync"] is not None
    info["prepared_sync"].unlink()
    with ExitStack() as stack:
        for p in _eligible_patches(info):
            stack.enter_context(p)
        with pytest.raises(IntakeError, match="Prepared sync missing"):
            validate_s8_source_for_packaging(
                info["session_root"],
                s8_run_dir=info["s8_run_dir"],
            )

    info2 = build_packaging_session(tmp_path / "tamper", tamper_sync_digest=True)
    with ExitStack() as stack:
        for p in _eligible_patches(info2):
            stack.enter_context(p)
        with pytest.raises(IntakeError, match="Prepared sync SHA256"):
            validate_s8_source_for_packaging(
                info2["session_root"],
                s8_run_dir=info2["s8_run_dir"],
            )


def test_schema_and_directional_key_validation(tmp_path: Path) -> None:
    info = build_packaging_session(tmp_path)
    bad_states = make_states(n_dense=info["n_dense"], time_sec=info["time_sec"])
    bad_states.loc[0, "schema_version"] = "wrong_schema"
    with ExitStack() as stack:
        for p in _eligible_patches(info):
            stack.enter_context(p)
        with pytest.raises(IntakeError, match="schema_version"):
            validate_s8_source_for_packaging(
                info["session_root"],
                s8_run_dir=info["s8_run_dir"],
                states=bad_states,
                bouts=pd.read_parquet(info["bouts_path"]),
            )

    dup = make_states(n_dense=info["n_dense"], time_sec=info["time_sec"])
    dup = pd.concat([dup, dup.iloc[[0]]], ignore_index=True)
    with ExitStack() as stack:
        for p in _eligible_patches(info):
            stack.enter_context(p)
        with pytest.raises(IntakeError, match="duplicate directional"):
            validate_s8_source_for_packaging(
                info["session_root"],
                s8_run_dir=info["s8_run_dir"],
                states=dup,
                bouts=pd.read_parquet(info["bouts_path"]),
            )


def test_no_unintended_writes_to_upstream(tmp_path: Path) -> None:
    info = build_packaging_session(tmp_path)
    before = {
        str(p): sha256_file(p)
        for p in (
            info["states_path"],
            info["bouts_path"],
            info["prepared_sync"],
            info["prepared_video"],
        )
        if p is not None and p.is_file()
    }
    with ExitStack() as stack:
        for p in _eligible_patches(info):
            stack.enter_context(p)
        validate_s8_source_for_packaging(
            info["session_root"],
            s8_run_dir=info["s8_run_dir"],
        )
    after = {
        str(p): sha256_file(p)
        for p in (
            info["states_path"],
            info["bouts_path"],
            info["prepared_sync"],
            info["prepared_video"],
        )
        if p is not None and p.is_file()
    }
    assert before == after
