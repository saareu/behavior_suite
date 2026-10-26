"""Registry discovery tests for Subsystem 07 Sprint 1."""

from __future__ import annotations

from pathlib import Path

from behavioral_inference import (
    BUNDLE_INCOMPLETE,
    BUNDLE_OBSOLETE,
    BUNDLE_SUPPORTED,
    BUNDLE_UNSUPPORTED,
    REGISTRY_UNAVAILABLE,
    discover_model_registry,
)
from combined_features import ModelRegistrySettings


def test_missing_registry_is_unavailable() -> None:
    snapshot = discover_model_registry(ModelRegistrySettings(root=None))
    assert snapshot.status == REGISTRY_UNAVAILABLE
    assert snapshot.bundles == ()
    assert any("not configured" in n for n in snapshot.notes)


def test_missing_registry_directory_is_unavailable(tmp_path: Path) -> None:
    snapshot = discover_model_registry(
        ModelRegistrySettings(root=tmp_path / "does_not_exist")
    )
    assert snapshot.status == REGISTRY_UNAVAILABLE
    assert snapshot.bundles == ()


def test_discover_dyadic_and_focal(mini_registry: Path) -> None:
    snapshot = discover_model_registry(ModelRegistrySettings(root=mini_registry))
    assert snapshot.status in ("ok", "partial")
    by_name = snapshot.by_name()
    assert by_name["contact_intact"].family == "dyadic"
    assert by_name["contact_intact"].support == BUNDLE_SUPPORTED
    assert by_name["contact_intact"].feature_cols == ("feat_a", "feat_b")
    assert by_name["contact_intact"].model_sha256 is not None
    assert by_name["contact_intact"].serialization_format == "joblib_xgboost"
    assert by_name["approach"].family == "focal"
    assert by_name["approach"].support == BUNDLE_SUPPORTED


def test_obsolete_contact_marked_unsupported(mini_registry: Path) -> None:
    snapshot = discover_model_registry(ModelRegistrySettings(root=mini_registry))
    contact = snapshot.by_name()["contact"]
    assert contact.support == BUNDLE_OBSOLETE
    assert contact.is_loadable is False


def test_unsupported_json_expert_focal(mini_registry: Path) -> None:
    snapshot = discover_model_registry(ModelRegistrySettings(root=mini_registry))
    nose_body = snapshot.by_name()["nose_body"]
    assert nose_body.support == BUNDLE_UNSUPPORTED
    assert nose_body.model_path is None


def test_incomplete_bundle_without_joblib(mini_registry: Path) -> None:
    snapshot = discover_model_registry(ModelRegistrySettings(root=mini_registry))
    side = snapshot.by_name()["side_by_side"]
    assert side.support == BUNDLE_INCOMPLETE
    assert side.is_loadable is False


def test_artifact_digest_recorded(mini_registry: Path) -> None:
    snapshot = discover_model_registry(ModelRegistrySettings(root=mini_registry))
    intact = snapshot.by_name()["contact_intact"]
    assert intact.model_sha256 is not None
    assert len(intact.model_sha256) == 64
    # Synthetic bytes will not match the lab inventory digest.
    assert intact.expected_sha256 is not None
    assert intact.sha256_matches_expected is False


def test_does_not_imply_legacy_path() -> None:
    snapshot = discover_model_registry()
    assert snapshot.status == REGISTRY_UNAVAILABLE
    assert snapshot.settings.root is None
