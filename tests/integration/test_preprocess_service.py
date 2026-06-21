import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from preprocess.background import validate_background
from preprocess.config import (
    BackgroundConfig,
    CanonicalResolutionConfig,
    PrepareConfig,
    PreprocessConfig,
)
from preprocess.exceptions import VideoPreparationError
from preprocess.manual_crop import make_manual_crop_plan
from preprocess.metadata import validate_prepare_metadata
from preprocess.models import PreprocessRequest
from preprocess.service import PreprocessService
from preprocess.sync_writer import load_prepared_sync_npz
from preprocess.validation import validate_prepared_video
from preprocess.video_prepare import resolve_ffmpeg_binary, resolve_ffprobe_binary
from project.service import ProjectService

_RAW_FRAME_COUNT = 8
_RAW_SIZE_WH = (64, 48)
_RAW_FPS = 10.0


def _require_ffmpeg() -> tuple[Path, Path]:
    try:
        return resolve_ffmpeg_binary(), resolve_ffprobe_binary()
    except VideoPreparationError as exc:
        pytest.skip(str(exc))


def _write_raw_video(path: Path) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), _RAW_FPS, _RAW_SIZE_WH)
    if not writer.isOpened():
        pytest.skip("This OpenCV build cannot create an MJPG AVI fixture.")
    try:
        for index in range(_RAW_FRAME_COUNT):
            frame = np.full((_RAW_SIZE_WH[1], _RAW_SIZE_WH[0], 3), 25 + index * 15, dtype=np.uint8)
            cv2.circle(frame, (8 + index * 5, 20), 4, (230, 40, 40), thickness=-1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def _config(ffmpeg: Path, ffprobe: Path) -> PreprocessConfig:
    return PreprocessConfig(
        prepare=PrepareConfig(
            canonical_resolution=CanonicalResolutionConfig(enabled=False, width=64, height=48)
        ),
        encoding={"ffmpeg": {"ffmpeg_path": ffmpeg, "ffprobe_path": ffprobe}},
        background=BackgroundConfig(method="median", sample_every_n=1, max_samples=8),
    )


def _crop_plan(config: PreprocessConfig, *, accepted: bool):
    plan = make_manual_crop_plan(
        raw_frame_shape=(_RAW_SIZE_WH[1], _RAW_SIZE_WH[0]),
        points_tl_tr_br_bl=np.array([[0.0, 0.0], [63.0, 0.0], [63.0, 47.0], [0.0, 47.0]]),
        pre_crop_roi=None,
        canonical_resolution=config.prepare.canonical_resolution,
    )
    return plan.model_copy(update={"accepted_by_user": accepted})


@pytest.mark.integration
def test_complete_preprocess_service_run_produces_valid_official_artifacts(tmp_path: Path) -> None:
    ffmpeg, ffprobe = _require_ffmpeg()
    project = ProjectService().create_project(tmp_path, "ServiceProject")
    raw_path = _write_raw_video(tmp_path / "raw.avi")
    config = _config(ffmpeg, ffprobe)
    request = PreprocessRequest(
        project_dir=project.root_dir,
        raw_video_path=raw_path,
        config=config,
        start_frame=2,
        end_frame_exclusive=6,
        crop_plan=_crop_plan(config, accepted=True),
        crop_accepted_by_user=True,
    )

    result = PreprocessService().run(request)

    assert result.success is True, result.message
    assert result.outputs is not None
    outputs = result.outputs
    official_paths = (
        outputs.prepared_video_path,
        outputs.prepare_meta_path,
        outputs.prepared_sync_path,
        outputs.cropped_background_path,
        outputs.settings_used_path,
        outputs.processing_log_path,
    )
    assert all(path.is_file() for path in official_paths)

    validation = validate_prepared_video(
        outputs.prepared_video_path, expected_frame_count=4, expected_size_wh=(64, 48)
    )
    assert validation.is_valid is True
    assert validation.opencv_readable_frame_count == 4

    sync = load_prepared_sync_npz(outputs.prepared_sync_path)
    assert sync.raw_decode_frame_idx.tolist() == [2, 3, 4, 5]
    assert sync.frame_count_used_for_sleap == 4
    assert sync.raw_pts_status == "not_extracted"
    assert np.all(np.isnan(sync.raw_pts_time_sec))

    background = cv2.imread(str(outputs.cropped_background_path), cv2.IMREAD_UNCHANGED)
    assert background is not None
    validate_background(background, validation.opencv_reported_size_wh)

    with outputs.prepare_meta_path.open("r", encoding="utf-8") as stream:
        metadata = json.load(stream)
    validate_prepare_metadata(metadata)
    assert metadata["validation"]["status"] == "passed"
    assert metadata["trim"]["expected_trimmed_frame_count"] == 4

    with outputs.settings_used_path.open("r", encoding="utf-8") as stream:
        assert yaml.safe_load(stream)["schema_version"] == "preprocess_config_v1"
    log_text = outputs.processing_log_path.read_text(encoding="utf-8")
    assert "stage=ffmpeg_stage_a" in log_text
    assert "run completed status=success" in log_text
    assert not list((outputs.preprocess_dir / ".internal").glob("*"))


@pytest.mark.integration
def test_unaccepted_crop_fails_before_stage_a_and_writes_no_success_metadata(
    tmp_path: Path,
) -> None:
    ffmpeg, ffprobe = _require_ffmpeg()
    project = ProjectService().create_project(tmp_path, "RejectedCropProject")
    raw_path = _write_raw_video(tmp_path / "raw.avi")
    config = _config(ffmpeg, ffprobe)
    request = PreprocessRequest(
        project_dir=project.root_dir,
        raw_video_path=raw_path,
        config=config,
        crop_plan=_crop_plan(config, accepted=False),
        crop_accepted_by_user=True,
    )

    result = PreprocessService().run(request)

    preprocess_dir = project.root_dir / "preprocess"
    assert result.success is False
    assert "accepted_by_user" in result.message
    assert not (preprocess_dir / "prepare_meta.json").exists()
    assert not list((preprocess_dir / ".internal").glob("*.mp4"))
    log_text = (preprocess_dir / "processing_log.txt").read_text(encoding="utf-8")
    assert "crop acceptance validation" in log_text
    assert "stage=ffmpeg_stage_a" not in log_text
