"""Joint S7/S6 lineage validation and S8 readiness checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from behavioral_arbitration.contracts import IntakeError
from behavioral_arbitration.intake import (
    resolve_linked_s6_run,
    validate_joint_s7_s6_for_s8,
)
from tests.behavioral_arbitration._fixtures import build_synthetic_session


def _patch_eligibility(session_info: dict[str, Any]):
    """Bypass full S4/S5/model digest checks; keep joint S8 logic under test."""

    s7_dir = session_info["s7_run_dir"]
    s6_dir = session_info["s6_run_dir"]

    def fake_s7_validate(path: Path | str) -> dict[str, Any]:
        return {
            "s7_run_dir": Path(path).resolve(),
            "status": "completed",
            "source_s6_run_dir": s6_dir.resolve(),
            "source_s6_run_id": session_info["s6_run_id"],
            "dyadic_features_sha256": session_info["feat_sha"],
        }

    def fake_s6_validate(path: Path | str) -> dict[str, Any]:
        return {"run_id": Path(path).name, "s6_run_dir": Path(path).resolve()}

    return (
        patch(
            "behavioral_arbitration.intake.is_s7_run_eligible_for_downstream",
            return_value=True,
        ),
        patch(
            "behavioral_arbitration.intake.validate_s7_run_for_downstream",
            side_effect=fake_s7_validate,
        ),
        patch(
            "behavioral_arbitration.intake.is_s6_run_eligible_for_downstream",
            side_effect=lambda p: Path(p).resolve() == s6_dir.resolve()
            and Path(p).is_dir(),
        ),
        patch(
            "behavioral_arbitration.intake.validate_s6_run_for_downstream",
            side_effect=fake_s6_validate,
        ),
        patch(
            "behavioral_arbitration.intake.select_eligible_s7_run",
            side_effect=lambda session_root, s7_run_dir=None: _FakeSummary(
                s7_dir, session_info["s7_run_id"]
            ),
        ),
    )


class _FakeSummary:
    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.classification = "eligible"


def test_joint_lineage_validation_succeeds(tmp_path: Path) -> None:
    info = build_synthetic_session(tmp_path)
    patches = _patch_eligibility(info)
    with patches[0], patches[1], patches[2], patches[3], patches[4]:
        bundle = validate_joint_s7_s6_for_s8(
            info["session_root"], s7_run_dir=info["s7_run_dir"]
        )
    assert bundle.s7_run_id == info["s7_run_id"]
    assert bundle.s6_run_id == info["s6_run_id"]
    assert bundle.mouse_ids == (1, 2)
    assert bundle.n_frames == 4
    assert len(bundle.detectors_dyadic) == 4
    assert len(bundle.detectors_focal) == 4


def test_stale_linked_s6_refuses_newest_substitution(tmp_path: Path) -> None:
    info = build_synthetic_session(tmp_path, stale_s6=True)
    # Linked path missing; newest S6 exists — must still fail.
    with patch(
        "behavioral_arbitration.intake.select_eligible_s7_run",
        side_effect=lambda session_root, s7_run_dir=None: _FakeSummary(
            info["s7_run_dir"], info["s7_run_id"]
        ),
    ), patch(
        "behavioral_arbitration.intake.validate_s7_run_for_downstream",
        return_value={
            "s7_run_dir": info["s7_run_dir"],
            "status": "completed",
            "source_s6_run_dir": Path(
                info["session_root"]
                / "combined_features"
                / "combined_features__MISSING"
            ),
            "source_s6_run_id": "combined_features__MISSING",
            "dyadic_features_sha256": info["feat_sha"],
        },
    ), patch(
        "behavioral_arbitration.intake.is_s6_run_eligible_for_downstream",
        return_value=False,
    ), pytest.raises(IntakeError, match="will not substitute"):
        validate_joint_s7_s6_for_s8(
            info["session_root"], s7_run_dir=info["s7_run_dir"]
        )


def test_resolve_linked_s6_unavailable(tmp_path: Path) -> None:
    info = build_synthetic_session(tmp_path, stale_s6=True)
    with pytest.raises(IntakeError, match="unavailable|will not substitute"):
        resolve_linked_s6_run(info["s7_run_dir"])


def test_missing_required_detectors(tmp_path: Path) -> None:
    info = build_synthetic_session(tmp_path, missing_detectors=("side_by_side",))
    patches = _patch_eligibility(info)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        pytest.raises(IntakeError, match="required detector coverage"),
    ):
        validate_joint_s7_s6_for_s8(
            info["session_root"], s7_run_dir=info["s7_run_dir"]
        )


def test_missing_required_s6_columns(tmp_path: Path) -> None:
    info = build_synthetic_session(
        tmp_path, drop_hard_gap=("distance_nose_i_nose_j_norm",)
    )
    patches = _patch_eligibility(info)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        pytest.raises(IntakeError, match="required hard-gap"),
    ):
        validate_joint_s7_s6_for_s8(
            info["session_root"], s7_run_dir=info["s7_run_dir"]
        )


def test_exactly_two_mice_required(tmp_path: Path) -> None:
    info = build_synthetic_session(tmp_path, three_mice=True)
    patches = _patch_eligibility(info)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        pytest.raises(IntakeError, match="exactly two tracked mice"),
    ):
        validate_joint_s7_s6_for_s8(
            info["session_root"], s7_run_dir=info["s7_run_dir"]
        )
