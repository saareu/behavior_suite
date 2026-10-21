"""Blob features configuration defaults and validation."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from blob_features import (
    CABLE_REMOVAL_ASSUMED_LEGACY_DEFAULT,
    CABLE_REMOVAL_EXCLUDED,
    CC_COUNT_GAP_FILL_MAX_GAP_DEFAULT,
    BlobExecutionSettings,
    BlobFeaturesConfig,
    BlobFeaturesError,
    BlobSegmentationSettings,
    BlobTemporalSettings,
    default_suite_config_path,
    legacy_lab_config_path,
    load_blob_features_config,
)


def test_dataclass_defaults_match_suite_yaml() -> None:
    cfg = BlobFeaturesConfig().validated()
    assert cfg.segmentation.blob_threshold == -2
    assert cfg.segmentation.blob_blur_kernel_size == 9
    assert cfg.segmentation.blob_kernel_size == 27
    assert cfg.segmentation.blob_tail_open_kernel_size == 7
    assert cfg.segmentation.blob_min_area == 2900
    assert cfg.segmentation.blob_max_area == 0
    assert cfg.segmentation.blob_prune_min_radius == 10
    assert cfg.segmentation.blob_prune_regrow_iters == 7
    assert cfg.segmentation.blob_prune_max_regrow_dist == 8
    assert cfg.segmentation.blob_post_prune_min_area == 2900
    assert cfg.segmentation.blob_post_prune_max_area == 0
    assert cfg.segmentation.blob_final_smooth_kernel_size == 3
    assert cfg.segmentation.abs_dark_threshold == 10
    assert cfg.execution.n_workers == 16
    assert cfg.execution.max_in_flight == 128
    assert cfg.temporal.cc_count_gap_fill_max_gap == CC_COUNT_GAP_FILL_MAX_GAP_DEFAULT
    assert cfg.target_fps == 120.0
    assert cfg.cable_removal_excluded is CABLE_REMOVAL_EXCLUDED
    assert cfg.cable_removal_assumed_legacy_default is CABLE_REMOVAL_ASSUMED_LEGACY_DEFAULT


def test_suite_yaml_loads_and_matches_defaults() -> None:
    path = default_suite_config_path()
    assert path.is_file()
    loaded = load_blob_features_config(path)
    default = BlobFeaturesConfig().validated()
    assert loaded.segmentation == default.segmentation
    assert loaded.segmentation.blob_min_area == 2900
    assert loaded.segmentation.blob_post_prune_min_area == 2900
    assert loaded.temporal == default.temporal
    assert loaded.execution == default.execution


def test_legacy_yaml_loads_dropping_cable_keys() -> None:
    path = legacy_lab_config_path()
    assert path.is_file()
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert raw["blob_features"]["remove_cable_artifact"] is False
    loaded = load_blob_features_config(path)
    assert loaded.segmentation.blob_threshold == -2
    # Legacy lab YAML still records the historical 3100 defaults.
    assert loaded.segmentation.blob_min_area == 3100
    assert loaded.segmentation.blob_post_prune_min_area == 3100
    assert loaded.execution.n_workers == 16
    assert loaded.temporal.cc_count_gap_fill_max_gap == 5
    # Cable keys must not appear in active settings serialization.
    settings = loaded.to_settings_dict()
    assert "remove_cable_artifact" not in settings["segmentation"]
    assert settings["temporal"]["cc_count_gap_fill_max_gap"] == 5
    assert settings["cable_removal"]["excluded_from_s5_mvp"] is True
    assert settings["cable_removal"]["assumed_legacy_default"] is False


def test_reject_enabled_cable_removal(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "blob_features": {
                    "blob_threshold": -1,
                    "remove_cable_artifact": True,
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(BlobFeaturesError, match="Cable-artifact removal is excluded"):
        load_blob_features_config(path)


def test_processing_separate_from_execution() -> None:
    seg = BlobSegmentationSettings(blob_threshold=-1).validated()
    temporal = BlobTemporalSettings(cc_count_gap_fill_max_gap=3).validated()
    exe = BlobExecutionSettings(n_workers=4, max_in_flight=8).validated()
    cfg = BlobFeaturesConfig(
        segmentation=seg, temporal=temporal, execution=exe
    ).validated()
    assert cfg.segmentation.blob_threshold == -1
    assert cfg.temporal.cc_count_gap_fill_max_gap == 3
    assert cfg.execution.n_workers == 4
    assert "n_workers" not in cfg.to_settings_dict()["segmentation"]
    assert "blob_threshold" not in cfg.to_settings_dict()["execution"]
    assert "cc_count_gap_fill_max_gap" not in cfg.to_settings_dict()["segmentation"]


def test_reject_non_120_target_fps(tmp_path: Path) -> None:
    path = tmp_path / "bad_fps.yaml"
    path.write_text(
        yaml.safe_dump({"blob_features": {"blob_threshold": -1, "target_fps": 60}}),
        encoding="utf-8",
    )
    with pytest.raises(BlobFeaturesError, match="target_fps is fixed"):
        load_blob_features_config(path)


def test_invalid_blur_kernel_even() -> None:
    with pytest.raises(BlobFeaturesError, match="blob_blur_kernel_size must be odd"):
        BlobSegmentationSettings(blob_blur_kernel_size=8).validated()


def test_invalid_threshold_below_multiotsu() -> None:
    with pytest.raises(BlobFeaturesError, match="blob_threshold must be >= -2"):
        BlobSegmentationSettings(blob_threshold=-3).validated()


def test_default_loader_without_path() -> None:
    cfg = load_blob_features_config(None)
    assert cfg.segmentation.blob_kernel_size == 27
