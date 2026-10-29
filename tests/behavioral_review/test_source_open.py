"""S9 package intake / eligibility tests for S10 Sprint 1."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from behavioral_review.contracts import MODE_INSPECTION, SourceError
from behavioral_review.session import open_inspection_session
from behavioral_review.source import (
    assert_no_upstream_writes,
    open_review_source,
    snapshot_upstream_fingerprints,
)
from results_packaging.contracts import ProvenanceError, PublicationError
from tests.behavioral_review._fixtures import (
    build_mini_review_package,
    patch_s9_currently_eligible,
)


def test_open_eligible_package_preserves_exact_identity(tmp_path: Path) -> None:
    built = build_mini_review_package(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        source = open_review_source(built["package_dir"])

    assert source.package_run_id == built["package_run_id"]
    assert source.currently_eligible is True
    assert source.n_dense_samples == built["n_dense"]
    assert source.n_bouts == 4
    assert source.source_s8.run_id == built["s8_run_id"]
    assert source.source_s8.run_dir == built["s8_run_dir"]
    assert source.source_s7_run_id == "behavioral_inference__synth"
    assert source.source_video.path == built["video_path"]
    assert source.source_video.available is True
    assert source.video_available is True


def test_stale_package_rejected(tmp_path: Path) -> None:
    built = build_mini_review_package(tmp_path)
    with (
        patch(
            "behavioral_review.source.validate_s9_run_for_downstream",
            side_effect=ProvenanceError("prepared sync digest drifted"),
        ),
        pytest.raises(SourceError, match="not currently eligible"),
    ):
        open_review_source(built["package_dir"])


def test_missing_package_rejected(tmp_path: Path) -> None:
    with pytest.raises(SourceError, match="does not exist"):
        open_review_source(tmp_path / "missing_package")


def test_historical_completed_but_ineligible_still_fails(tmp_path: Path) -> None:
    built = build_mini_review_package(tmp_path)
    with (
        patch(
            "behavioral_review.source.validate_s9_run_for_downstream",
            side_effect=PublicationError("required artifact missing"),
        ),
        pytest.raises(SourceError, match="not currently eligible"),
    ):
        open_review_source(built["package_dir"])


def test_broken_video_reference_fails_closed(tmp_path: Path) -> None:
    built = build_mini_review_package(tmp_path)
    assert built["video_path"] is not None
    built["video_path"].unlink()
    with (
        patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]),
        pytest.raises(SourceError, match="Prepared video missing"),
    ):
        open_review_source(built["package_dir"], require_video=True)


def test_video_digest_mismatch_no_substitute(tmp_path: Path) -> None:
    built = build_mini_review_package(tmp_path, video_sha_mismatch=True)
    with (
        patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]),
        pytest.raises(SourceError, match="SHA256"),
    ):
        open_review_source(built["package_dir"], require_video=True)


def test_invalid_sync_index_schema_rejected(tmp_path: Path) -> None:
    built = build_mini_review_package(tmp_path, corrupt_sync=True)
    with (
        patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]),
        pytest.raises(SourceError, match="Invalid sync_index"),
    ):
        open_review_source(built["package_dir"])


def test_opening_is_read_only_no_upstream_writes(tmp_path: Path) -> None:
    built = build_mini_review_package(tmp_path)
    with patch_s9_currently_eligible(source_s8_run_id=built["s8_run_id"]):
        source = open_review_source(built["package_dir"])
        snaps = snapshot_upstream_fingerprints(source)
        session = open_inspection_session(built["package_dir"], decode_initial_frame=False)
        session.seek_sample(3)
        session.seek_bout_start(f"{built['package_run_id']}__bout_contact")
        session.select_direction(focal_id=0, partner_id=1)

    assert session.mode == MODE_INSPECTION
    assert session.snapshot()["mode"] == MODE_INSPECTION
    # No draft review directory created.
    review_root = built["session_root"] / "behavioral_review"
    assert not review_root.exists()
    assert_no_upstream_writes(
        package_dir=source.package_dir,
        s8_run_dir=source.source_s8.run_dir,
        video_path=source.source_video.path,
        snapshots=snaps,
    )


def test_no_silent_package_substitution(tmp_path: Path) -> None:
    built_a = build_mini_review_package(
        tmp_path / "a", package_run_id="results_package__a"
    )
    built_b = build_mini_review_package(
        tmp_path / "b", package_run_id="results_package__b"
    )
    with patch_s9_currently_eligible(source_s8_run_id=built_a["s8_run_id"]):
        source = open_review_source(built_a["package_dir"])
    assert source.package_run_id == "results_package__a"
    assert source.package_dir == built_a["package_dir"]
    assert source.package_dir != built_b["package_dir"]
