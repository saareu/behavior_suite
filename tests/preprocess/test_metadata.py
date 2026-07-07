import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import yaml

from preprocess import metadata as metadata_module
from preprocess.background import BackgroundEstimateResult
from preprocess.config import (
    CanonicalResolutionConfig,
    PrepareConfig,
    PreprocessConfig,
)
from preprocess.crop_plan import CropMode
from preprocess.exceptions import MetadataArtifactError
from preprocess.manual_crop import make_full_frame_crop_plan, make_manual_crop_plan
from preprocess.metadata import (
    PREPARE_METADATA_SCHEMA_VERSION,
    PREPARE_METADATA_TOP_LEVEL_SECTIONS,
    build_prepare_metadata,
    validate_prepare_metadata,
    write_prepare_meta_json,
    write_settings_used_yaml,
)
from preprocess.models import (
    PreprocessOutputs,
    SoftwareEnvironmentInfo,
    VideoProbeResult,
)
from preprocess.validation import PreparedVideoValidationResult
from project.models import Project


def _section(metadata: dict[str, object], name: str) -> dict[str, Any]:
    return cast(dict[str, Any], metadata[name])


def _build_metadata_fixture(
    tmp_path: Path,
    *,
    mode: CropMode = CropMode.MANUAL,
) -> tuple[dict[str, object], PreprocessConfig]:
    project = Project(root_dir=tmp_path / "ExampleProject", name="ExampleProject")
    outputs = PreprocessOutputs.from_preprocess_dir(project.root_dir / "preprocess")
    config = PreprocessConfig(
        prepare=PrepareConfig(
            canonical_resolution=CanonicalResolutionConfig(
                enabled=False,
                width=64,
                height=48,
            )
        )
    )
    if mode is CropMode.FULL_FRAME:
        crop_plan = make_full_frame_crop_plan(
            raw_frame_shape=(48, 64),
            canonical_resolution=config.prepare.canonical_resolution,
        ).model_copy(update={"accepted_by_user": True})
    else:
        crop_plan = make_manual_crop_plan(
            raw_frame_shape=(48, 64),
            points_tl_tr_br_bl=np.array(
                [[0.0, 0.0], [63.0, 0.0], [63.0, 47.0], [0.0, 47.0]]
            ),
            pre_crop_roi=None,
            canonical_resolution=config.prepare.canonical_resolution,
        ).model_copy(
            update={
                "mode": mode,
                "fit_score": 0.95 if mode is CropMode.AUTOMATIC else None,
                "rim_density": 0.12 if mode is CropMode.AUTOMATIC else None,
                "accepted_by_user": True,
            }
        )
    raw_probe = VideoProbeResult(
        source_path=tmp_path / "raw.avi",
        width=64,
        height=48,
        codec="mjpeg",
        pixel_format="yuvj420p",
        duration_sec=1.0,
        avg_frame_rate="10/1",
        r_frame_rate="10/1",
        time_base="1/10",
        frame_count_ffprobe=10,
        frame_count_opencv_reported=10,
        frame_count_opencv_readable=10,
        opencv_fps=10.0,
        raw_fps_effective=10.0,
        raw_fps_effective_method="ffprobe_avg_frame_rate",
        pts_status="not_extracted",
        ffprobe_succeeded=True,
    )
    prepared_validation = PreparedVideoValidationResult(
        is_valid=True,
        errors=(),
        opencv_reported_frame_count=4,
        opencv_readable_frame_count=4,
        opencv_reported_size_wh=(64, 48),
        opencv_reported_fps=10.0,
        expected_frame_count=4,
        expected_size_wh=(64, 48),
    )
    background_array = np.full((48, 64, 3), 50, dtype=np.uint8)
    background_array.setflags(write=False)
    background_result = BackgroundEstimateResult(
        background=background_array,
        sampled_frame_indices=(0, 2),
        sampled_frame_count=2,
        source_video_path=outputs.prepared_video_path,
        source_size_wh=(64, 48),
        method="median",
        output_dtype="uint8",
    )
    software_environment = SoftwareEnvironmentInfo(
        python_version="3.12.11",
        platform="Windows-11",
        opencv_version="4.11.0",
        numpy_version="2.2.0",
        ffmpeg_version="8.0",
        ffprobe_version="8.0",
        application_version="0.1.0",
    )
    metadata = build_prepare_metadata(
        project=project,
        raw_probe=raw_probe,
        prepared_validation=prepared_validation,
        crop_plan=crop_plan,
        preprocess_config=config,
        external_time_selection=None,
        sync_path=outputs.prepared_sync_path,
        background_result=background_result,
        outputs=outputs,
        start_frame=2,
        end_frame_exclusive=6,
        software_environment=software_environment,
        detector_diagnostics=(
            {"sampled_frame_count": 3, "detector": "test"}
            if mode is CropMode.AUTOMATIC
            else None
        ),
    )
    return metadata, config


def test_valid_metadata_contains_every_required_top_level_section(
    tmp_path: Path,
) -> None:
    metadata, _config = _build_metadata_fixture(tmp_path)

    assert set(metadata) == set(PREPARE_METADATA_TOP_LEVEL_SECTIONS)
    assert metadata["schema_version"] == PREPARE_METADATA_SCHEMA_VERSION
    validate_prepare_metadata(metadata)


def test_valid_metadata_is_json_serializable(tmp_path: Path) -> None:
    metadata, _config = _build_metadata_fixture(tmp_path)

    serialized = json.dumps(metadata, allow_nan=False)

    assert isinstance(serialized, str)


def test_crop_plan_arrays_are_json_safe_lists(tmp_path: Path) -> None:
    metadata, _config = _build_metadata_fixture(tmp_path)
    geometry = _section(metadata, "geometry")

    assert isinstance(geometry["quad_raw_tl_tr_br_bl"], list)
    assert isinstance(geometry["H_raw_to_prepared_3x3"], list)
    assert isinstance(geometry["H_raw_to_prepared_3x3"][0], list)
    assert isinstance(geometry["H_prepared_to_raw_3x3"], list)


def test_automatic_crop_populates_cage_detection_only(tmp_path: Path) -> None:
    metadata, _config = _build_metadata_fixture(
        tmp_path,
        mode=CropMode.AUTOMATIC,
    )

    cage = _section(metadata, "cage_detection")
    manual = _section(metadata, "manual_crop")
    assert cage["used"] is True
    assert cage["fit_score"] == pytest.approx(0.95)
    assert cage["rim_density"] == pytest.approx(0.12)
    assert cage["detector_diagnostics"]["detector"] == "test"
    assert manual == {"used": False}


def test_manual_crop_populates_manual_crop_only(tmp_path: Path) -> None:
    metadata, _config = _build_metadata_fixture(tmp_path)

    assert _section(metadata, "manual_crop") == {"used": True}
    assert _section(metadata, "cage_detection")["used"] is False
    assert _section(metadata, "geometry")["mode"] == "manual"


def test_full_frame_crop_populates_neither_detector_nor_manual_crop(
    tmp_path: Path,
) -> None:
    metadata, _config = _build_metadata_fixture(
        tmp_path,
        mode=CropMode.FULL_FRAME,
    )

    assert _section(metadata, "manual_crop") == {"used": False}
    assert _section(metadata, "cage_detection")["used"] is False
    assert _section(metadata, "geometry")["mode"] == "full_frame"
    assert _section(metadata, "geometry")["rotated_90"] is False


def test_no_ttl_metadata_is_explicitly_not_provided(tmp_path: Path) -> None:
    metadata, _config = _build_metadata_fixture(tmp_path)
    external = _section(metadata, "external_time")

    assert external == {
        "provided": False,
        "source_path": None,
        "selected_variable": None,
        "selected_units": None,
        "raw_vector_length": 0,
        "raw_video_frame_count_opencv_readable": 10,
        "median_dt": None,
        "estimated_fps": None,
        "validation_status": "not_provided",
    }


def test_metadata_validation_rejects_missing_top_level_section(
    tmp_path: Path,
) -> None:
    metadata, _config = _build_metadata_fixture(tmp_path)
    metadata.pop("sync")

    with pytest.raises(MetadataArtifactError, match="top-level"):
        validate_prepare_metadata(metadata)


def test_metadata_validation_rejects_mismatched_prepared_counts(
    tmp_path: Path,
) -> None:
    metadata, _config = _build_metadata_fixture(tmp_path)
    _section(metadata, "prepared_video")["frame_count_used_for_sleap"] = 3

    with pytest.raises(MetadataArtifactError, match="frame_count_used_for_sleap"):
        validate_prepare_metadata(metadata)


def test_metadata_validation_rejects_geometry_video_size_mismatch(
    tmp_path: Path,
) -> None:
    metadata, _config = _build_metadata_fixture(tmp_path)
    _section(metadata, "geometry")["prepared_size_wh"] = [66, 48]

    with pytest.raises(MetadataArtifactError, match="prepared_size_wh"):
        validate_prepare_metadata(metadata)


def test_metadata_validation_rejects_unaccepted_crop_plan(
    tmp_path: Path,
) -> None:
    metadata, _config = _build_metadata_fixture(tmp_path)
    _section(metadata, "geometry")["accepted_by_user"] = False

    with pytest.raises(MetadataArtifactError, match="accepted"):
        validate_prepare_metadata(metadata)


def test_metadata_validation_rejects_background_video_size_mismatch(
    tmp_path: Path,
) -> None:
    metadata, _config = _build_metadata_fixture(tmp_path)
    _section(metadata, "background")["source_size_wh"] = [64, 46]

    with pytest.raises(MetadataArtifactError, match="Background source size"):
        validate_prepare_metadata(metadata)


def test_metadata_validation_rejects_missing_output_path(tmp_path: Path) -> None:
    metadata, _config = _build_metadata_fixture(tmp_path)
    _section(metadata, "outputs").pop("prepared_sync_path")

    with pytest.raises(MetadataArtifactError, match="missing required fields"):
        validate_prepare_metadata(metadata)


def test_metadata_validation_rejects_missing_nested_encoding_field(
    tmp_path: Path,
) -> None:
    metadata, _config = _build_metadata_fixture(tmp_path)
    encoding = _section(metadata, "encoding")
    cast(dict[str, Any], encoding["ffmpeg"]).pop("codec")

    with pytest.raises(MetadataArtifactError, match="encoding.ffmpeg"):
        validate_prepare_metadata(metadata)


def test_metadata_validation_rejects_non_json_values(tmp_path: Path) -> None:
    metadata, _config = _build_metadata_fixture(tmp_path)
    _section(metadata, "project")["unsupported"] = np.array([1, 2])

    with pytest.raises(MetadataArtifactError, match="non-JSON"):
        validate_prepare_metadata(metadata)


def test_atomic_json_write_preserves_existing_destination_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata, _config = _build_metadata_fixture(tmp_path)
    destination = tmp_path / "prepare_meta.json"
    existing = b"existing validated metadata"
    destination.write_bytes(existing)

    def fail_dump(*_args: object, **_kwargs: object) -> None:
        raise OSError("forced JSON write failure")

    monkeypatch.setattr(metadata_module.json, "dump", fail_dump)

    with pytest.raises(MetadataArtifactError, match="forced JSON"):
        write_prepare_meta_json(destination, metadata)

    assert destination.read_bytes() == existing
    assert list(tmp_path.glob(".prepare_meta.metadata-*.json")) == []


def test_atomic_yaml_write_preserves_existing_destination_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _metadata, config = _build_metadata_fixture(tmp_path)
    destination = tmp_path / "settings_used.yaml"
    existing = b"existing validated settings"
    destination.write_bytes(existing)

    def fail_dump(*_args: object, **_kwargs: object) -> None:
        raise yaml.YAMLError("forced YAML write failure")

    monkeypatch.setattr(metadata_module.yaml, "safe_dump", fail_dump)

    with pytest.raises(MetadataArtifactError, match="forced YAML"):
        write_settings_used_yaml(destination, config)

    assert destination.read_bytes() == existing
    assert list(tmp_path.glob(".settings_used.settings-*.yaml")) == []


def test_json_and_yaml_round_trip_successfully(tmp_path: Path) -> None:
    metadata, config = _build_metadata_fixture(tmp_path)
    json_path = tmp_path / "prepare_meta.json"
    yaml_path = tmp_path / "settings_used.yaml"

    write_prepare_meta_json(json_path, metadata)
    write_settings_used_yaml(yaml_path, config)

    with json_path.open("r", encoding="utf-8") as stream:
        loaded_metadata = json.load(stream)
    with yaml_path.open("r", encoding="utf-8") as stream:
        loaded_settings = yaml.safe_load(stream)
    assert loaded_metadata == metadata
    assert loaded_settings == config.model_dump(mode="json")
    assert "geometry" not in loaded_settings
    assert "sampled_frame_indices" not in loaded_settings
