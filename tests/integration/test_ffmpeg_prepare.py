import subprocess
from pathlib import Path

import cv2
import numpy as np
import pytest

from preprocess import cage_detection
from preprocess.cage_detection import detect_cage_crop_plan
from preprocess.config import (
    CanonicalResolutionConfig,
    PrepareConfig,
    PreprocessConfig,
)
from preprocess.exceptions import VideoPreparationError
from preprocess.ffmpeg_runtime import ensure_supported_ffmpeg_runtime
from preprocess.manual_crop import make_manual_crop_plan
from preprocess.pre_crop import resolve_pre_crop
from preprocess.video_prepare import (
    reencode_intermediate_with_opencv,
    resolve_ffmpeg_binary,
    resolve_ffprobe_binary,
    run_ffmpeg_prepare,
)
from ui.controllers.crop_review_controller import build_crop_preview

# This admits interpolation and H.264 centroid noise while rejecting a
# larger-than-two-pixel spatial disagreement in the Stage A transform.
MAX_MARKER_CENTER_ERROR_PX = 2.0
PREVIEW_STAGE_MAX_MARKER_ERROR_PX = 2.5
VISIBLE_BOUNDS_MAX_ERROR_PX = 2
_MARKER_POINTS_RAW_XY = np.array(
    [[62.0, 32.0], [90.0, 35.0], [97.0, 84.0], [53.0, 81.0]],
    dtype=np.float64,
)
_FRACTIONAL_PERSPECTIVE_POINTS = np.array(
    [
        [25.05680077762755, 20.132692876841816],
        [276.7875596244743, 34.75081456266568],
        [268.4005883911093, 201.01696844513665],
        [32.9860241236059, 198.31592465770703],
    ],
    dtype=np.float64,
)
_AUTOMATIC_PARITY_QUAD = np.array(
    [[110.0, 20.0], [205.0, 30.0], [215.0, 220.0], [95.0, 210.0]],
    dtype=np.float64,
)
_MANUAL_PARITY_MARKERS = np.array(
    [[60.0, 55.0], [240.0, 65.0], [235.0, 170.0], [65.0, 165.0]],
    dtype=np.float64,
)
_AUTOMATIC_PARITY_MARKERS = np.array(
    [[130.0, 55.0], [180.0, 60.0], [185.0, 180.0], [125.0, 175.0]],
    dtype=np.float64,
)
_MARKER_COLORS_BGR = (
    (20, 20, 240),
    (20, 240, 20),
    (240, 20, 20),
    (20, 240, 240),
)


def _require_ffmpeg() -> tuple[Path, Path]:
    try:
        ffmpeg, ffprobe = resolve_ffmpeg_binary(), resolve_ffprobe_binary()
        ensure_supported_ffmpeg_runtime(ffmpeg_path=ffmpeg, ffprobe_path=ffprobe)
        return ffmpeg, ffprobe
    except VideoPreparationError as exc:
        pytest.skip(str(exc))


def _write_test_video(path: Path, frame_count: int = 8) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    if not writer.isOpened():
        pytest.skip("This OpenCV build cannot create an MJPG AVI fixture.")
    try:
        for index in range(frame_count):
            frame = np.full((48, 64, 3), 30 + index * 10, dtype=np.uint8)
            cv2.rectangle(frame, (8, 4), (55, 43), (40, 180, 80), thickness=-1)
            writer.write(frame)
    finally:
        writer.release()
    return path


def _write_spatial_marker_image(path: Path) -> Path:
    frame = np.full((120, 160, 3), 48, dtype=np.uint8)
    for point, color in zip(_MARKER_POINTS_RAW_XY, _MARKER_COLORS_BGR, strict=True):
        cv2.rectangle(
            frame,
            (int(point[0]) - 2, int(point[1]) - 2),
            (int(point[0]) + 2, int(point[1]) + 2),
            color=color,
            thickness=-1,
        )
    assert cv2.imwrite(str(path), frame)
    return path


def _write_fractional_perspective_image(path: Path) -> Path:
    frame = np.full((240, 320, 3), 48, dtype=np.uint8)
    polygon = np.rint(_FRACTIONAL_PERSPECTIVE_POINTS).astype(np.int32)
    cv2.fillConvexPoly(frame, polygon, color=(40, 180, 80))
    assert cv2.imwrite(str(path), frame)
    return path


def _write_parity_image(
    path: Path,
    quad: np.ndarray,
    marker_points: np.ndarray,
) -> tuple[Path, np.ndarray]:
    frame = np.full((240, 320, 3), (85, 25, 85), dtype=np.uint8)
    cv2.fillConvexPoly(
        frame,
        np.rint(quad).astype(np.int32),
        color=(55, 55, 55),
    )
    for point, color in zip(marker_points, _MARKER_COLORS_BGR, strict=True):
        cv2.rectangle(
            frame,
            (int(point[0]) - 3, int(point[1]) - 3),
            (int(point[0]) + 3, int(point[1]) + 3),
            color=color,
            thickness=-1,
        )
    assert cv2.imwrite(str(path), frame)
    return path, frame


def _transform_points(homography: np.ndarray, points_xy: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack(
        [points_xy, np.ones(points_xy.shape[0], dtype=np.float64)]
    )
    transformed = (homography @ homogeneous.T).T
    return transformed[:, :2] / transformed[:, 2, np.newaxis]


def _detect_spatial_marker_centers(frame: np.ndarray) -> np.ndarray:
    blue, green, red = [channel.astype(np.int16) for channel in cv2.split(frame)]
    masks = (
        (red > 120) & (red - green > 70) & (red - blue > 70),
        (green > 120) & (green - red > 70) & (green - blue > 70),
        (blue > 120) & (blue - red > 70) & (blue - green > 70),
        (red > 120) & (green > 120) & (blue < 100),
    )
    centers: list[np.ndarray] = []
    for mask in masks:
        component_count, _labels, statistics, centroids = cv2.connectedComponentsWithStats(
            mask.astype(np.uint8),
            connectivity=8,
        )
        assert component_count > 1, "Expected spatial marker was not detected."
        marker_index = 1 + int(np.argmax(statistics[1:, cv2.CC_STAT_AREA]))
        assert statistics[marker_index, cv2.CC_STAT_AREA] >= 5
        centers.append(centroids[marker_index])
    return np.asarray(centers, dtype=np.float64)


def _visible_content_bounds(frame: np.ndarray) -> tuple[int, int, int, int]:
    visible = np.max(frame, axis=2) > 20
    rows, columns = np.nonzero(visible)
    assert rows.size > 0
    return (
        int(columns.min()),
        int(rows.min()),
        int(columns.max()),
        int(rows.max()),
    )


def _assert_canonical_padding_is_black(frame: np.ndarray, crop_plan) -> None:
    canonical = crop_plan.canonical_geometry
    assert canonical is not None
    padding_regions = []
    if canonical.padding_top > 2:
        padding_regions.append(frame[: canonical.padding_top - 2])
    if canonical.padding_bottom > 2:
        padding_regions.append(frame[-canonical.padding_bottom + 2 :])
    if canonical.padding_left > 2:
        padding_regions.append(frame[:, : canonical.padding_left - 2])
    if canonical.padding_right > 2:
        padding_regions.append(frame[:, -canonical.padding_right + 2 :])
    assert padding_regions
    assert all(float(region.mean()) < 5.0 for region in padding_regions)


def _accepted_automatic_parity_plan(
    config: PreprocessConfig,
    raw_frame: np.ndarray,
    monkeypatch: pytest.MonkeyPatch,
):
    detected = cage_detection._DetectedRectangle(
        rectangle_in_search=((155.0, 120.0), (110.0, 190.0), 5.0),
        quad_in_pre_crop=_AUTOMATIC_PARITY_QUAD,
        fit_score=0.95,
        rim_density=0.1,
        contour_area=20_000.0,
        coarse_search_roi=(0, 0, 320, 240),
    )
    background = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
    monkeypatch.setattr(
        cage_detection,
        "_build_median_background_with_count",
        lambda _path, _step: cage_detection._MedianBackground(
            image=background,
            sampled_frame_count=1,
        ),
    )
    monkeypatch.setattr(
        cage_detection,
        "_detect_rotated_rectangle",
        lambda _background, _config, _margin: detected,
    )
    pre_crop = resolve_pre_crop(config.pre_crop, raw_size_wh=(320, 240))
    result = detect_cage_crop_plan(Path("unused.avi"), config, pre_crop)
    return result.crop_plan.model_copy(update={"accepted_by_user": True})


def _config(ffmpeg_binary: Path, ffprobe_binary: Path) -> PreprocessConfig:
    return PreprocessConfig(
        prepare=PrepareConfig(
            canonical_resolution=CanonicalResolutionConfig(
                enabled=True,
                width=96,
                height=96,
            )
        ),
        encoding={
            "ffmpeg": {
                "ffmpeg_path": ffmpeg_binary,
                "ffprobe_path": ffprobe_binary,
            }
        },
    )


def _accepted_crop_plan(config: PreprocessConfig):
    plan = make_manual_crop_plan(
        raw_frame_shape=(48, 64),
        points_tl_tr_br_bl=np.array(
            [[8.0, 4.0], [55.0, 4.0], [55.0, 43.0], [8.0, 43.0]]
        ),
        pre_crop_roi=None,
        canonical_resolution=config.prepare.canonical_resolution,
    )
    return plan.model_copy(update={"accepted_by_user": True})


def _count_readable_frames(path: Path) -> tuple[int, tuple[int, int], np.ndarray]:
    capture = cv2.VideoCapture(str(path))
    assert capture.isOpened()
    frames: list[np.ndarray] = []
    try:
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(frame)
    finally:
        capture.release()
    assert frames
    height, width = frames[0].shape[:2]
    return len(frames), (width, height), frames[0]


@pytest.mark.integration
def test_ffmpeg_prepare_produces_trimmed_canonical_intermediate(
    tmp_path: Path,
) -> None:
    ffmpeg_binary, ffprobe_binary = _require_ffmpeg()
    raw_path = _write_test_video(tmp_path / "raw.avi")
    output_path = tmp_path / "internal" / "rectified.mp4"
    config = _config(ffmpeg_binary, ffprobe_binary)
    crop_plan = _accepted_crop_plan(config)

    result = run_ffmpeg_prepare(
        raw_video_path=raw_path,
        intermediate_path=output_path,
        crop_plan=crop_plan,
        config=config,
        start_frame=2,
        end_frame_exclusive=6,
    )

    frame_count, size_wh, first_frame = _count_readable_frames(output_path)
    assert result.intermediate_path == output_path.resolve()
    assert result.expected_output_size_wh == (96, 96)
    assert result.return_code == 0
    assert output_path.is_file()
    assert size_wh == (96, 96)
    assert frame_count == 4
    assert float(first_frame[:6].mean()) < 5.0
    assert float(first_frame[20:76].mean()) > 20.0

    audio_probe = subprocess.run(
        [
            str(ffprobe_binary),
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(output_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert audio_probe.returncode == 0
    assert audio_probe.stdout.strip() == ""


@pytest.mark.integration
def test_ffmpeg_prepare_matches_crop_plan_spatial_geometry(
    tmp_path: Path,
) -> None:
    ffmpeg_binary, ffprobe_binary = _require_ffmpeg()
    raw_path = _write_spatial_marker_image(tmp_path / "spatial_markers.png")
    output_path = tmp_path / "internal" / "spatial_parity.mp4"
    config = PreprocessConfig(
        prepare=PrepareConfig(
            canonical_resolution=CanonicalResolutionConfig(
                enabled=True,
                width=192,
                height=192,
            )
        ),
        encoding={
            "ffmpeg": {
                "ffmpeg_path": ffmpeg_binary,
                "ffprobe_path": ffprobe_binary,
            }
        },
    )
    crop_plan = make_manual_crop_plan(
        raw_frame_shape=(120, 160),
        points_tl_tr_br_bl=np.array(
            [[55.0, 10.0], [100.0, 18.0], [112.0, 110.0], [38.0, 102.0]]
        ),
        pre_crop_roi=None,
        canonical_resolution=config.prepare.canonical_resolution,
    ).model_copy(update={"accepted_by_user": True})

    result = run_ffmpeg_prepare(
        raw_video_path=raw_path,
        intermediate_path=output_path,
        crop_plan=crop_plan,
        config=config,
        start_frame=0,
        end_frame_exclusive=1,
    )

    frame_count, size_wh, output_frame = _count_readable_frames(output_path)
    output_width, output_height = crop_plan.prepared_size_wh
    assert frame_count == 1
    assert size_wh == crop_plan.prepared_size_wh
    assert result.expected_output_size_wh == crop_plan.prepared_size_wh

    predicted_centers = _transform_points(
        crop_plan.H_raw_to_prepared_3x3,
        _MARKER_POINTS_RAW_XY,
    )
    assert np.all(predicted_centers[:, 0] >= 0.0)
    assert np.all(predicted_centers[:, 0] < output_width)
    assert np.all(predicted_centers[:, 1] >= 0.0)
    assert np.all(predicted_centers[:, 1] < output_height)

    assert crop_plan.rotated_90 is True
    assert "clockwise_rotation" in result.filtergraph_metadata.stages
    prepared_quad = _transform_points(
        crop_plan.H_raw_to_prepared_3x3,
        crop_plan.quad_raw_tl_tr_br_bl,
    )
    assert prepared_quad[0, 0] > prepared_quad[3, 0]
    assert prepared_quad[1, 0] > prepared_quad[2, 0]
    assert prepared_quad[0, 1] < prepared_quad[1, 1]
    assert prepared_quad[3, 1] < prepared_quad[2, 1]

    canonical = crop_plan.canonical_geometry
    assert canonical is not None
    assert canonical.padding_left == canonical.padding_right == 0
    assert canonical.padding_top > 4
    assert canonical.padding_bottom > 4
    top_padding = output_frame[: canonical.padding_top - 2]
    bottom_padding = output_frame[
        output_height - canonical.padding_bottom + 2 :
    ]
    assert float(top_padding.mean()) < 5.0
    assert float(bottom_padding.mean()) < 5.0
    assert float(
        output_frame[
            canonical.padding_top + 2 : canonical.padding_top + 6
        ].mean()
    ) > 20.0
    assert float(
        output_frame[
            output_height
            - canonical.padding_bottom
            - 6 : output_height
            - canonical.padding_bottom
            - 2
        ].mean()
    ) > 20.0

    observed_centers = _detect_spatial_marker_centers(output_frame)
    marker_errors = np.linalg.norm(observed_centers - predicted_centers, axis=1)
    maximum_error = float(marker_errors.max())
    print(f"maximum marker-center error: {maximum_error:.6f} px")
    assert maximum_error <= MAX_MARKER_CENTER_ERROR_PX


@pytest.mark.integration
def test_ffmpeg_prepare_accepts_fractional_manual_perspective_crop(
    tmp_path: Path,
) -> None:
    ffmpeg_binary, ffprobe_binary = _require_ffmpeg()
    raw_path = _write_fractional_perspective_image(
        tmp_path / "fractional_perspective.png"
    )
    output_path = tmp_path / "internal" / "fractional_perspective.mp4"
    config = PreprocessConfig(
        prepare=PrepareConfig(
            canonical_resolution=CanonicalResolutionConfig(
                enabled=True,
                width=192,
                height=192,
            )
        ),
        encoding={
            "ffmpeg": {
                "ffmpeg_path": ffmpeg_binary,
                "ffprobe_path": ffprobe_binary,
            }
        },
    )
    crop_plan = make_manual_crop_plan(
        raw_frame_shape=(240, 320),
        points_tl_tr_br_bl=_FRACTIONAL_PERSPECTIVE_POINTS,
        pre_crop_roi=(10, 10, 300, 220),
        canonical_resolution=config.prepare.canonical_resolution,
    ).model_copy(update={"accepted_by_user": True})

    result = run_ffmpeg_prepare(
        raw_video_path=raw_path,
        intermediate_path=output_path,
        crop_plan=crop_plan,
        config=config,
        start_frame=0,
        end_frame_exclusive=1,
    )

    frame_count, size_wh, _first_frame = _count_readable_frames(output_path)
    assert frame_count == 1
    assert size_wh == crop_plan.prepared_size_wh == (192, 192)
    assert result.expected_output_size_wh == crop_plan.prepared_size_wh
    assert result.filtergraph_metadata.rectified_content_size_wh == (254, 180)
    assert "perspective_rectification" in result.filtergraph_metadata.stages


@pytest.mark.integration
@pytest.mark.parametrize(
    ("crop_origin", "expected_rotation"),
    [("automatic", True), ("manual", False)],
)
def test_gui_preview_matches_stage_a_and_stage_b_geometry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    crop_origin: str,
    expected_rotation: bool,
) -> None:
    ffmpeg_binary, ffprobe_binary = _require_ffmpeg()
    config = PreprocessConfig(
        prepare=PrepareConfig(
            perspective_interpolation="linear",
            canonical_resolution=CanonicalResolutionConfig(
                enabled=True,
                width=240,
                height=200,
            ),
        ),
        encoding={
            "ffmpeg": {
                "ffmpeg_path": ffmpeg_binary,
                "ffprobe_path": ffprobe_binary,
            }
        },
    )
    if crop_origin == "automatic":
        quad = _AUTOMATIC_PARITY_QUAD
        marker_points = _AUTOMATIC_PARITY_MARKERS
    else:
        quad = _FRACTIONAL_PERSPECTIVE_POINTS
        marker_points = _MANUAL_PARITY_MARKERS
    raw_path, raw_frame = _write_parity_image(
        tmp_path / f"{crop_origin}_preview_source.png",
        quad,
        marker_points,
    )
    if crop_origin == "automatic":
        crop_plan = _accepted_automatic_parity_plan(config, raw_frame, monkeypatch)
    else:
        crop_plan = make_manual_crop_plan(
            raw_frame_shape=(240, 320),
            points_tl_tr_br_bl=quad,
            pre_crop_roi=None,
            canonical_resolution=config.prepare.canonical_resolution,
        ).model_copy(update={"accepted_by_user": True})

    preview = build_crop_preview(
        raw_frame,
        crop_plan,
        config.prepare.perspective_interpolation,
    )
    intermediate_path = tmp_path / f"{crop_origin}_stage_a.mp4"
    run_ffmpeg_prepare(
        raw_video_path=raw_path,
        intermediate_path=intermediate_path,
        crop_plan=crop_plan,
        config=config,
        start_frame=0,
        end_frame_exclusive=1,
    )
    stage_count, stage_size_wh, stage_a_frame = _count_readable_frames(
        intermediate_path
    )
    final_path = tmp_path / f"{crop_origin}_stage_b.mp4"
    reencode_intermediate_with_opencv(
        intermediate_video_path=intermediate_path,
        prepared_video_path=final_path,
        output_fps=10.0,
        output_size_wh=crop_plan.prepared_size_wh,
        config=config,
    )
    final_count, final_size_wh, stage_b_frame = _count_readable_frames(final_path)

    expected_height = crop_plan.prepared_size_wh[1]
    expected_width = crop_plan.prepared_size_wh[0]
    assert preview.shape[:2] == (expected_height, expected_width)
    assert stage_size_wh == crop_plan.prepared_size_wh
    assert final_size_wh == crop_plan.prepared_size_wh
    assert stage_count == final_count == 1
    assert crop_plan.rotated_90 is expected_rotation

    predicted_centers = _transform_points(
        crop_plan.H_raw_to_prepared_3x3,
        marker_points,
    )
    preview_centers = _detect_spatial_marker_centers(preview)
    stage_a_centers = _detect_spatial_marker_centers(stage_a_frame)
    stage_b_centers = _detect_spatial_marker_centers(stage_b_frame)
    preview_stage_errors = np.linalg.norm(
        preview_centers - stage_a_centers,
        axis=1,
    )
    preview_final_errors = np.linalg.norm(
        preview_centers - stage_b_centers,
        axis=1,
    )
    predicted_stage_errors = np.linalg.norm(
        predicted_centers - stage_a_centers,
        axis=1,
    )
    maximum_marker_error = float(
        max(
            preview_stage_errors.max(),
            preview_final_errors.max(),
            predicted_stage_errors.max(),
        )
    )
    print(
        f"{crop_origin} preview/encoded maximum marker error: "
        f"{maximum_marker_error:.6f} px"
    )
    assert maximum_marker_error <= PREVIEW_STAGE_MAX_MARKER_ERROR_PX

    preview_bounds = np.asarray(_visible_content_bounds(preview))
    stage_a_bounds = np.asarray(_visible_content_bounds(stage_a_frame))
    stage_b_bounds = np.asarray(_visible_content_bounds(stage_b_frame))
    assert np.max(np.abs(preview_bounds - stage_a_bounds)) <= VISIBLE_BOUNDS_MAX_ERROR_PX
    assert np.max(np.abs(preview_bounds - stage_b_bounds)) <= VISIBLE_BOUNDS_MAX_ERROR_PX
    _assert_canonical_padding_is_black(preview, crop_plan)
    _assert_canonical_padding_is_black(stage_a_frame, crop_plan)
    _assert_canonical_padding_is_black(stage_b_frame, crop_plan)


@pytest.mark.integration
def test_ffmpeg_prepare_failure_raises_domain_error(tmp_path: Path) -> None:
    ffmpeg_binary, ffprobe_binary = _require_ffmpeg()
    raw_path = tmp_path / "broken.avi"
    raw_path.write_text("not a video", encoding="utf-8")
    config = _config(ffmpeg_binary, ffprobe_binary)
    crop_plan = _accepted_crop_plan(config)

    with pytest.raises(VideoPreparationError, match="ffmpeg preparation failed"):
        run_ffmpeg_prepare(
            raw_video_path=raw_path,
            intermediate_path=tmp_path / "internal" / "failed.mp4",
            crop_plan=crop_plan,
            config=config,
            start_frame=0,
            end_frame_exclusive=2,
        )
