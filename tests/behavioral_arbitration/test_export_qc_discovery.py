"""Publication, QC uniqueness/cardinality, and discovery eligibility/staleness."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from behavioral_arbitration.bouts import extract_behavior_bouts
from behavioral_arbitration.config import load_behavioral_arbitration_config
from behavioral_arbitration.contracts import (
    BEHAVIOR_BOUTS_FILENAME,
    BEHAVIOR_STATES_FILENAME,
    ELIGIBLE_DOWNSTREAM,
    QC_FILENAME,
    RUN_META_FILENAME,
    STALE_SOURCE,
    PublicationError,
)
from behavioral_arbitration.export import build_behavior_states_table
from behavioral_arbitration.qc import evaluate_behavioral_arbitration_qc
from behavioral_arbitration.run_discovery import (
    find_latest_eligible_s8_run,
    summarize_behavioral_arbitration_project,
)
from behavioral_arbitration.runner import (
    BehavioralArbitrationRequest,
    run_behavioral_arbitration_publication,
)
from behavioral_inference.contracts import PREDICTIONS_FILENAME
from tests.behavioral_arbitration._fixtures import build_synthetic_session
from tests.behavioral_arbitration.test_intake import _patch_eligibility


def _patch_publication(session_info: dict[str, Any]):
    """Eligibility patches for intake + runner publish / revalidation paths."""

    intake = _patch_eligibility(session_info)
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

    return intake + (
        patch(
            "behavioral_arbitration.runner.is_s7_run_eligible_for_downstream",
            return_value=True,
        ),
        patch(
            "behavioral_arbitration.runner.validate_s7_run_for_downstream",
            side_effect=fake_s7_validate,
        ),
        patch(
            "behavioral_arbitration.runner.is_s6_run_eligible_for_downstream",
            return_value=True,
        ),
        patch(
            "behavioral_arbitration.runner.validate_s6_run_for_downstream",
            side_effect=fake_s6_validate,
        ),
        patch(
            "behavioral_arbitration.run_discovery.is_s7_run_eligible_for_downstream",
            return_value=True,
        ),
        patch(
            "behavioral_arbitration.run_discovery.find_latest_eligible_s7_run",
            return_value=type(
                "S7",
                (),
                {
                    "run_dir": session_info["s7_run_dir"],
                    "run_id": session_info["s7_run_id"],
                },
            )(),
        ),
    )


def _enter_all(patches):
    return _PatchStack(patches)


class _PatchStack:
    def __init__(self, patches: tuple) -> None:
        self._patches = patches
        self._started: list[Any] = []

    def __enter__(self):
        for p in self._patches:
            self._started.append(p.start())
        return self

    def __exit__(self, *exc):
        for p in reversed(self._patches):
            p.stop()
        return False


def test_atomic_publish_writes_required_artifacts(tmp_path: Path) -> None:
    info = build_synthetic_session(tmp_path, n_frames=8)
    config = load_behavioral_arbitration_config(None)
    with _enter_all(_patch_publication(info)):
        result = run_behavioral_arbitration_publication(
            BehavioralArbitrationRequest(
                session_root=info["session_root"],
                s7_run_dir=info["s7_run_dir"],
                config=config,
                run_purpose="test",
                timestamp="20260101T120000",
            )
        )
    assert result.success
    assert result.status in {"completed", "completed_with_warnings"}
    run_dir = result.run_dir
    assert (run_dir / BEHAVIOR_STATES_FILENAME).is_file()
    assert (run_dir / BEHAVIOR_BOUTS_FILENAME).is_file()
    assert (run_dir / QC_FILENAME).is_file()
    assert (run_dir / RUN_META_FILENAME).is_file()
    assert (run_dir / "settings_used.yaml").is_file()
    states = pd.read_parquet(run_dir / BEHAVIOR_STATES_FILENAME)
    bouts = pd.read_parquet(run_dir / BEHAVIOR_BOUTS_FILENAME)
    assert not states.empty
    assert "behavior_label" in states.columns
    assert "bout_id" in bouts.columns
    # Uniqueness / cardinality on published states.
    assert not states.duplicated(
        subset=["dense_frame_idx", "focal_id", "partner_id"]
    ).any()
    assert len(states) == info["dyadic"].shape[0] * 2  # K=2 directional


def test_duplicate_timestamp_rejected(tmp_path: Path) -> None:
    info = build_synthetic_session(tmp_path, n_frames=6)
    config = load_behavioral_arbitration_config(None)
    req = BehavioralArbitrationRequest(
        session_root=info["session_root"],
        s7_run_dir=info["s7_run_dir"],
        config=config,
        run_purpose="test",
        timestamp="20260101T130000",
    )
    with _enter_all(_patch_publication(info)):
        first = run_behavioral_arbitration_publication(req)
        assert first.success
        with pytest.raises(PublicationError, match="already exists"):
            run_behavioral_arbitration_publication(req)


def test_failure_unlinks_science_artifacts(tmp_path: Path) -> None:
    info = build_synthetic_session(tmp_path, n_frames=6)
    config = load_behavioral_arbitration_config(None)
    patches = _patch_publication(info) + (
        patch(
            "behavioral_arbitration.runner.write_behavior_bouts_parquet",
            side_effect=RuntimeError("forced publish failure"),
        ),
    )
    with _enter_all(patches):
        result = run_behavioral_arbitration_publication(
            BehavioralArbitrationRequest(
                session_root=info["session_root"],
                s7_run_dir=info["s7_run_dir"],
                config=config,
                run_purpose="test",
                timestamp="20260101T140000",
            )
        )
    assert not result.success
    assert result.status == "failed"
    assert result.behavior_states_path is None
    assert result.behavior_bouts_path is None
    run_dir = result.run_dir
    assert not (run_dir / BEHAVIOR_STATES_FILENAME).exists()
    assert not (run_dir / BEHAVIOR_BOUTS_FILENAME).exists()
    assert not (run_dir / QC_FILENAME).exists()
    assert (run_dir / RUN_META_FILENAME).is_file()


def test_qc_uniqueness_and_cardinality_metrics() -> None:
    from tests.behavioral_arbitration._fixtures import make_directional_timeline

    ab = make_directional_timeline(["engaged"] * 4, focal=1, partner=2)
    ba = make_directional_timeline(["engaged"] * 4, focal=2, partner=1)
    states = pd.concat([ab, ba], ignore_index=True)
    for det in (
        "nose_to_nose",
        "nose_to_body",
        "nose_to_tail",
        "approach",
        "retract",
        "mutually_disengaged",
        "contact",
        "side_by_side",
    ):
        if f"pred_{det}" not in states.columns:
            states[f"pred_{det}"] = 0
    bouts = extract_behavior_bouts(
        states,
        fps=120.0,
        source_s7_run_id="s7",
        source_s6_run_id="s6",
        run_id="s8",
    )
    provenance = {
        "source_s7": {"run_id": "s7", "eligible_at_publish": True},
        "source_s6": {"run_id": "s6", "eligible_at_publish": True},
        "settings": {"fps": 120.0},
    }
    summary = evaluate_behavioral_arbitration_qc(
        states=states,
        bouts=bouts,
        diagnostics=(),
        n_frames=4,
        source_provenance=provenance,
        linked_s7_eligible=True,
        linked_s6_eligible=True,
    )
    assert summary.structural_ok
    assert summary.metrics["directional"]["n_duplicate_key_rows"] == 0
    assert summary.metrics["directional"]["expected_rows_k2"] == 8


def test_discovery_eligible_then_stale_when_s7_digest_changes(
    tmp_path: Path,
) -> None:
    info = build_synthetic_session(tmp_path, n_frames=6)
    config = load_behavioral_arbitration_config(None)
    with _enter_all(_patch_publication(info)):
        published = run_behavioral_arbitration_publication(
            BehavioralArbitrationRequest(
                session_root=info["session_root"],
                s7_run_dir=info["s7_run_dir"],
                config=config,
                run_purpose="test",
                timestamp="20260101T150000",
            )
        )
        assert published.success
        latest = find_latest_eligible_s8_run(info["session_root"])
        assert latest is not None
        assert latest.classification in {
            ELIGIBLE_DOWNSTREAM,
            "completed_with_warnings",
        }

        # Mutate S7 predictions digest → recorded S8 provenance no longer matches.
        pred_path = info["s7_run_dir"] / PREDICTIONS_FILENAME
        pred_path.write_bytes(b"mutated-predictions-content")

        summary = summarize_behavioral_arbitration_project(info["session_root"])
        assert len(summary.runs) >= 1
        run = next(r for r in summary.runs if r.run_id == published.run_id)
        assert run.classification == STALE_SOURCE
        assert find_latest_eligible_s8_run(info["session_root"]) is None


def test_in_memory_states_tables_equal_for_same_inputs(tmp_path: Path) -> None:
    info = build_synthetic_session(tmp_path, n_frames=6)
    config = load_behavioral_arbitration_config(None)
    from behavioral_arbitration.runner import run_behavioral_arbitration_sprint3

    with _enter_all(_patch_publication(info)):
        req = BehavioralArbitrationRequest(
            session_root=info["session_root"],
            s7_run_dir=info["s7_run_dir"],
            config=config,
        )
        a = run_behavioral_arbitration_sprint3(req)
        b = run_behavioral_arbitration_sprint3(req)
    states_a = build_behavior_states_table(
        a.frames,
        session_root=info["session_root"],
        source_s7_run_id=a.source_s7_run_id,
        source_s6_run_id=a.source_s6_run_id,
        run_id="fixed_run",
    )
    states_b = build_behavior_states_table(
        b.frames,
        session_root=info["session_root"],
        source_s7_run_id=b.source_s7_run_id,
        source_s6_run_id=b.source_s6_run_id,
        run_id="fixed_run",
    )
    pd.testing.assert_frame_equal(
        states_a.reset_index(drop=True),
        states_b.reset_index(drop=True),
        check_dtype=False,
    )
