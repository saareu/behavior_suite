"""Canceled S4/S5 must not enter S6 source selection or joint-pair validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from blob_features.contracts import STATUS_CANCELED as S5_STATUS_CANCELED
from blob_features.contracts import BlobFeaturesError
from combined_features import (
    CombinedFeaturesError,
    list_eligible_joint_pairs,
    select_default_joint_pair,
    validate_joint_s4_s5_pair,
)
from pose_finalization.contracts import STATUS_CANCELED as S4_STATUS_CANCELED
from pose_finalization.contracts import PoseFinalizationError
from tests.combined_features.helpers import write_joint_session


def _status_from_run(path: Path) -> str | None:
    meta = json.loads((path / "run_meta.json").read_text(encoding="utf-8"))
    status = meta.get("status")
    return str(status) if status is not None else None


def _patch_status_aware_upstream(monkeypatch: pytest.MonkeyPatch) -> None:
    """Allow synthetic completed fixtures while preserving canceled rejection."""

    def validate_s4(path: Path | str) -> dict[str, object]:
        status = _status_from_run(Path(path))
        if status not in {"completed", "completed_with_warnings"}:
            raise PoseFinalizationError(
                f"S4 run is not completed (status={status or 'missing'!r})."
            )
        return {"ok": True}

    def validate_s5(path: Path | str) -> dict[str, object]:
        status = _status_from_run(Path(path))
        if status not in {"completed", "completed_with_warnings"}:
            raise BlobFeaturesError(
                f"S5 run is not completed (status={status or 'missing'!r})."
            )
        return {"ok": True}

    def eligible_s4(path: Path | str) -> bool:
        try:
            validate_s4(path)
        except PoseFinalizationError:
            return False
        return True

    def eligible_s5(path: Path | str) -> bool:
        try:
            validate_s5(path)
        except BlobFeaturesError:
            return False
        return True

    monkeypatch.setattr(
        "combined_features.upstream.validate_s4_run_for_downstream", validate_s4
    )
    monkeypatch.setattr(
        "combined_features.upstream.validate_s5_run_for_downstream", validate_s5
    )
    monkeypatch.setattr(
        "pose_finalization.run_discovery.is_s4_run_eligible_for_downstream",
        eligible_s4,
    )
    monkeypatch.setattr(
        "blob_features.run_discovery.is_s5_run_eligible_for_downstream",
        eligible_s5,
    )


def test_canceled_s4_excluded_from_s6_selection_and_joint_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_status_aware_upstream(monkeypatch)
    completed = write_joint_session(
        tmp_path,
        s4_run_id="pose_finalization__20261021T120000",
        s5_run_id="blob_features__20261021T120100",
    )
    canceled = write_joint_session(
        tmp_path,
        s4_run_id="pose_finalization__20261021T130000",
        s5_run_id="blob_features__20261021T120100",
        s4_status=S4_STATUS_CANCELED,
    )

    with pytest.raises(CombinedFeaturesError, match="not eligible|not completed"):
        validate_joint_s4_s5_pair(
            session_root=completed["session_root"],
            s4_run_dir=canceled["s4_run_dir"],
            s5_run_dir=completed["s5_run_dir"],
        )

    pairs = list_eligible_joint_pairs(completed["session_root"])
    assert pairs
    assert all(s4.run_id != canceled["s4_run_dir"].name for s4, _s5, _report in pairs)
    assert any(s4.run_id == completed["s4_run_dir"].name for s4, _s5, _report in pairs)

    selected = select_default_joint_pair(completed["session_root"])
    assert selected.s4_run_dir == completed["s4_run_dir"].resolve()
    assert selected.s5_run_dir == completed["s5_run_dir"].resolve()


def test_canceled_s5_excluded_from_s6_selection_and_joint_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_status_aware_upstream(monkeypatch)
    completed = write_joint_session(
        tmp_path,
        s4_run_id="pose_finalization__20261021T120000",
        s5_run_id="blob_features__20261021T120100",
    )
    canceled = write_joint_session(
        tmp_path,
        s4_run_id="pose_finalization__20261021T120000",
        s5_run_id="blob_features__20261021T130000",
        s5_status=S5_STATUS_CANCELED,
    )

    with pytest.raises(CombinedFeaturesError, match="not eligible|not completed"):
        validate_joint_s4_s5_pair(
            session_root=completed["session_root"],
            s4_run_dir=completed["s4_run_dir"],
            s5_run_dir=canceled["s5_run_dir"],
        )

    pairs = list_eligible_joint_pairs(completed["session_root"])
    assert pairs
    assert all(s5.run_id != canceled["s5_run_dir"].name for _s4, s5, _report in pairs)
    assert any(s5.run_id == completed["s5_run_dir"].name for _s4, s5, _report in pairs)

    selected = select_default_joint_pair(completed["session_root"])
    assert selected.s4_run_dir == completed["s4_run_dir"].resolve()
    assert selected.s5_run_dir == completed["s5_run_dir"].resolve()
