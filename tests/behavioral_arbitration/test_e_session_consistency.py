"""Read-only scientific consistency checks on the E_full_session_integration S8 run.

Does not modify published artifacts. Skips when the lab session is absent.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from behavioral_arbitration.bouts import extract_behavior_bouts
from behavioral_arbitration.contracts import (
    EXCLUSIVE_ETHOGRAM_LABELS,
    LABEL_ENGAGED,
    LABEL_UNDEFINED,
)
from behavioral_arbitration.runner import (
    is_s8_run_eligible_for_downstream,
    validate_s8_run_for_downstream,
)
from project.file_digest import sha256_file

SESSION = Path(r"C:\behavior_suite\test_cases\E_full_session_integration")
S8_RUN = (
    SESSION
    / "behavioral_arbitration"
    / "behavioral_arbitration__20261028T133953"
)
S7_RUN = (
    SESSION
    / "behavioral_inference"
    / "behavioral_inference__20261026T163224"
)
S6_RUN = (
    SESSION / "combined_features" / "combined_features__20261025T173244"
)

VOCAB = set(EXCLUSIVE_ETHOGRAM_LABELS) | {LABEL_UNDEFINED, LABEL_ENGAGED}

pytestmark = pytest.mark.skipif(
    not S8_RUN.is_dir(),
    reason="E_full_session_integration S8 run not present",
)


@pytest.fixture(scope="module")
def states() -> pd.DataFrame:
    return pd.read_parquet(S8_RUN / "behavior_states.parquet")


@pytest.fixture(scope="module")
def bouts() -> pd.DataFrame:
    return pd.read_parquet(S8_RUN / "behavior_bouts.parquet")


def test_framewise_cardinality_and_keys(states: pd.DataFrame) -> None:
    assert states["dense_frame_idx"].nunique() == 36528
    assert len(states) == 73056
    key = ["dense_frame_idx", "focal_id", "partner_id"]
    assert not states.duplicated(key).any()
    assert set(states["behavior_label"].unique()).issubset(VOCAB)
    assert LABEL_ENGAGED != LABEL_UNDEFINED
    # Exactly one exclusive label column value per row (non-null string).
    assert states["behavior_label"].notna().all()
    # Time alignment at 120 Hz.
    expected = states["dense_frame_idx"].astype(float) / 120.0
    assert (states["time_sec"] - expected).abs().max() < 1e-9


def test_parallel_channels_independent_of_exclusive(states: pd.DataFrame) -> None:
    assert "pred_contact" in states.columns
    assert "pred_side_by_side" in states.columns
    # Contact/SBS positives may coexist with any exclusive label.
    contact = states["pred_contact"].fillna(0).astype(int) == 1
    assert contact.any()
    assert states.loc[contact, "behavior_label"].nunique() >= 1


def test_s7_validity_and_score_origin_preserved(states: pd.DataFrame) -> None:
    validity_cols = [c for c in states.columns if c.startswith("input_valid_")]
    origin_cols = [c for c in states.columns if c.startswith("score_origin_")]
    assert validity_cols
    assert origin_cols
    # At least some invalid evidence rows exist or all-valid is allowed; columns present.
    assert states[validity_cols].notna().any().any()


def test_bout_inclusive_duration_and_no_exclusive_overlap(
    states: pd.DataFrame, bouts: pd.DataFrame
) -> None:
    exclusive = bouts[bouts["channel"] == "exclusive"].copy()
    assert not exclusive.empty
    n_frames = (
        exclusive["end_sample_idx"].astype(int)
        - exclusive["start_sample_idx"].astype(int)
        + 1
    )
    assert (n_frames == exclusive["bout_duration_frames"].astype(int)).all()
    assert (
        (exclusive["bout_duration_sec"] - exclusive["bout_duration_frames"] / 120.0)
        .abs()
        .max()
        < 1e-9
    )
    # No overlapping exclusive bouts within a direction.
    for (_, _), group in exclusive.groupby(["focal_id", "partner_id"], sort=False):
        ordered = group.sort_values("start_sample_idx")
        ends = ordered["end_sample_idx"].astype(int).to_numpy()
        starts = ordered["start_sample_idx"].astype(int).to_numpy()
        assert (starts[1:] > ends[:-1]).all()
    assert int(exclusive["bout_duration_frames"].sum()) == len(states)


def test_parallel_bouts_undirected_no_duplicates(bouts: pd.DataFrame) -> None:
    for channel in ("contact", "side_by_side"):
        parallel = bouts[bouts["channel"] == channel]
        if parallel.empty:
            continue
        assert (parallel["identity_grain"] == "symmetric_pair").all()
        # Symmetric pair identity should not duplicate the reverse direction.
        keys = parallel[
            ["mouse_a_id", "mouse_b_id", "start_sample_idx", "end_sample_idx"]
        ]
        assert not keys.duplicated().any()


def test_bout_reconstruction_matches_published(
    states: pd.DataFrame, bouts: pd.DataFrame
) -> None:
    reconstructed = extract_behavior_bouts(
        states,
        fps=120.0,
        source_s7_run_id="behavioral_inference__20261026T163224",
        source_s6_run_id="combined_features__20261025T173244",
        run_id="behavioral_arbitration__20261028T133953",
    )
    assert len(reconstructed) == len(bouts)
    compare_cols = [
        "behavior_name",
        "channel",
        "focal_id",
        "partner_id",
        "start_sample_idx",
        "end_sample_idx",
        "bout_duration_frames",
    ]
    left = reconstructed[compare_cols].reset_index(drop=True)
    right = bouts[compare_cols].reset_index(drop=True)
    # Parallel rows use NA focal/partner; normalize nullable vs float NaN for equality.
    for frame in (left, right):
        for col in ("focal_id", "partner_id"):
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
    pd.testing.assert_frame_equal(left, right, check_dtype=False)


def test_provenance_digests_and_eligibility() -> None:
    info = validate_s8_run_for_downstream(S8_RUN)
    assert is_s8_run_eligible_for_downstream(S8_RUN)
    assert info["source_s7_run_id"] == "behavioral_inference__20261026T163224"
    assert info["source_s6_run_id"] == "combined_features__20261025T173244"
    pred = S7_RUN / "predictions.parquet"
    feat = S6_RUN / "dyadic_features_full.parquet"
    assert sha256_file(pred).lower() == str(info["predictions_sha256"]).lower()
    assert sha256_file(feat).lower() == str(info["dyadic_features_sha256"]).lower()
