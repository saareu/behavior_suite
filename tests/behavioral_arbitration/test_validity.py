"""Valid negative vs routing sentinel and evidence normalization."""

from __future__ import annotations

import numpy as np

from behavioral_arbitration.contracts import (
    EVIDENCE_UNOBSERVED_INVALID_ROUTE,
    EVIDENCE_VALID_NEGATIVE,
    EVIDENCE_VALID_POSITIVE,
)
from behavioral_arbitration.expand import expand_dyadic_to_directional
from behavioral_arbitration.validity import (
    classify_evidence_state,
    normalize_arbitration_evidence,
)
from behavioral_inference.contracts import (
    SCORE_ORIGIN_MODEL_OBSERVED,
    SCORE_ORIGIN_ROUTING_SENTINEL,
)
from tests.behavioral_arbitration._fixtures import make_dyadic_wide, make_focal_wide


def test_routing_sentinel_zero_is_not_valid_negative() -> None:
    state = classify_evidence_state(
        input_valid=False,
        score_origin=SCORE_ORIGIN_ROUTING_SENTINEL,
        pred=0,
        column_present=True,
    )
    assert state == EVIDENCE_UNOBSERVED_INVALID_ROUTE

    valid_neg = classify_evidence_state(
        input_valid=True,
        score_origin=SCORE_ORIGIN_MODEL_OBSERVED,
        pred=0,
        column_present=True,
    )
    assert valid_neg == EVIDENCE_VALID_NEGATIVE


def test_normalize_masks_sentinel_but_keeps_valid_negative() -> None:
    dyadic = make_dyadic_wide(invalid_contact_frames={0})
    focal = make_focal_wide()
    frames, _ = expand_dyadic_to_directional(dyadic, focal)
    out, diag = normalize_arbitration_evidence(frames)

    sentinel_rows = out[out["dense_frame_idx"] == 0]
    assert (sentinel_rows["evidence_state_contact"] == EVIDENCE_UNOBSERVED_INVALID_ROUTE).all()
    assert sentinel_rows["arbitration_input_valid_contact"].astype(bool).eq(False).all()
    assert sentinel_rows["prob_contact"].isna().all()
    # Provenance preserved.
    assert (sentinel_rows["s7_prob_contact"] == 0.0).all()
    assert (sentinel_rows["score_origin_contact"] == SCORE_ORIGIN_ROUTING_SENTINEL).all()

    # Frame 1 contact is valid negative (odd frame → pred 0) in fixture.
    valid_rows = out[out["dense_frame_idx"] == 1]
    assert (valid_rows["evidence_state_contact"] == EVIDENCE_VALID_NEGATIVE).all()
    assert valid_rows["arbitration_input_valid_contact"].astype(bool).all()
    assert not valid_rows["prob_contact"].isna().any()
    assert diag.n_rows_changed >= 2  # both directions on frame 0


def test_valid_positive_preserved() -> None:
    dyadic = make_dyadic_wide()
    focal = make_focal_wide()
    frames, _ = expand_dyadic_to_directional(dyadic, focal)
    out, _ = normalize_arbitration_evidence(frames)
    # Even frames have nose_to_nose positive in fixture.
    pos = out[out["dense_frame_idx"] == 0]
    assert (pos["evidence_state_nose_to_nose"] == EVIDENCE_VALID_POSITIVE).all()
    assert np.allclose(pos["prob_nose_to_nose"], 0.9)


def test_parallel_contact_and_sbs_channels_both_normalized() -> None:
    dyadic = make_dyadic_wide()
    focal = make_focal_wide()
    frames, _ = expand_dyadic_to_directional(dyadic, focal)
    out, _ = normalize_arbitration_evidence(frames)
    assert "evidence_state_contact" in out.columns
    assert "evidence_state_side_by_side" in out.columns
    assert "prob_contact" in out.columns
    assert "prob_side_by_side" in out.columns
