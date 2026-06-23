from pathlib import Path

import numpy as np
import pytest

from preprocess import video_prepare
from preprocess.config import (
    CanonicalResolutionConfig,
    FFmpegEncodingConfig,
    PrepareConfig,
    PreprocessConfig,
)
from preprocess.crop_plan import CropPlan
from preprocess.exceptions import VideoPreparationError
from preprocess.manual_crop import make_manual_crop_plan
from preprocess.video_prepare import (
    build_ffmpeg_prepare_command,
    build_prepare_filtergraph,
    resolve_ffmpeg_binary,
    resolve_ffprobe_binary,
)


def _config(
    *,
    canonical_enabled: bool = True,
    canonical_size_wh: tuple[int, int] = (400, 400),
    ffmpeg: FFmpegEncodingConfig | None = None,
) -> PreprocessConfig:
    return PreprocessConfig(
        prepare=PrepareConfig(
            canonical_resolution=CanonicalResolutionConfig(
                enabled=canonical_enabled,
                width=canonical_size_wh[0],
                height=canonical_size_wh[1],
            )
        ),
        encoding={"ffmpeg": ffmpeg or FFmpegEncodingConfig()},
    )


def _landscape_plan(config: PreprocessConfig):
    return make_manual_crop_plan(
        raw_frame_shape=(240, 320),
        points_tl_tr_br_bl=np.array(
            [[0.0, 0.0], [319.0, 0.0], [319.0, 239.0], [0.0, 239.0]]
        ),
        pre_crop_roi=None,
        canonical_resolution=config.prepare.canonical_resolution,
    )


def _portrait_plan(config: PreprocessConfig):
    return make_manual_crop_plan(
        raw_frame_shape=(320, 240),
        points_tl_tr_br_bl=np.array(
            [[0.0, 0.0], [239.0, 0.0], [239.0, 319.0], [0.0, 319.0]]
        ),
        pre_crop_roi=None,
        canonical_resolution=config.prepare.canonical_resolution,
    )


def _fractional_perspective_plan(config: PreprocessConfig) -> CropPlan:
    return make_manual_crop_plan(
        raw_frame_shape=(240, 320),
        points_tl_tr_br_bl=np.array(
            [
                [25.05680077762755, 20.132692876841816],
                [276.7875596244743, 34.75081456266568],
                [268.4005883911093, 201.01696844513665],
                [32.9860241236059, 198.31592465770703],
            ]
        ),
        pre_crop_roi=(10, 10, 300, 220),
        canonical_resolution=config.prepare.canonical_resolution,
    )


def test_resolve_ffmpeg_binary_prefers_configured_path(tmp_path: Path) -> None:
    configured = tmp_path / "managed-ffmpeg.exe"
    configured.write_bytes(b"binary")

    assert resolve_ffmpeg_binary(configured) == configured.resolve()


def test_resolve_ffprobe_binary_prefers_configured_path(tmp_path: Path) -> None:
    configured = tmp_path / "managed-ffprobe.exe"
    configured.write_bytes(b"binary")

    assert resolve_ffprobe_binary(configured) == configured.resolve()


def test_resolve_ffmpeg_binary_uses_path_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    discovered = tmp_path / "ffmpeg.exe"
    discovered.write_bytes(b"binary")
    monkeypatch.setattr(video_prepare, "_bundled_binary_candidates", lambda _name: ())
    monkeypatch.setattr(video_prepare.shutil, "which", lambda _name: str(discovered))

    assert resolve_ffmpeg_binary() == discovered.resolve()


def test_resolve_ffmpeg_binary_fails_clearly_when_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(video_prepare, "_bundled_binary_candidates", lambda _name: ())
    monkeypatch.setattr(video_prepare.shutil, "which", lambda _name: None)

    with pytest.raises(VideoPreparationError, match="configured, bundled, or PATH"):
        resolve_ffmpeg_binary()


def test_filtergraph_uses_decode_index_trim_without_temporal_resampling() -> None:
    config = _config()
    filtergraph, metadata = build_prepare_filtergraph(
        _landscape_plan(config),
        config,
        start_frame=3,
        end_frame_exclusive=8,
    )

    assert "select=between(n\\,3\\,7)" in filtergraph
    assert "setpts=PTS-STARTPTS" in filtergraph
    assert "fps=" not in filtergraph
    assert "minterpolate" not in filtergraph
    assert metadata.start_frame == 3
    assert metadata.end_frame_exclusive == 8


def test_filtergraph_contains_perspective_and_required_clockwise_rotation() -> None:
    config = _config(canonical_enabled=False, canonical_size_wh=(400, 400))
    landscape_filtergraph, landscape_metadata = build_prepare_filtergraph(
        _landscape_plan(config),
        config,
    )
    portrait_filtergraph, portrait_metadata = build_prepare_filtergraph(
        _portrait_plan(config),
        config,
    )

    assert "perspective=" in landscape_filtergraph
    assert "sense=source" in landscape_filtergraph
    assert (
        "perspective=x0=0:y0=0:x1=320:y1=0:x2=0:y2=240:x3=320:y3=240"
        in landscape_filtergraph
    )
    assert "transpose=" not in landscape_filtergraph
    assert landscape_metadata.rotated_90 is False
    assert "transpose=clock" in portrait_filtergraph
    assert portrait_metadata.rotated_90 is True


def test_filtergraph_applies_pre_crop_and_preserves_local_quad_metadata() -> None:
    config = _config(canonical_enabled=False)
    plan = make_manual_crop_plan(
        raw_frame_shape=(240, 320),
        points_tl_tr_br_bl=np.array(
            [[50.0, 50.0], [250.0, 50.0], [250.0, 150.0], [50.0, 150.0]]
        ),
        pre_crop_roi=(40, 30, 240, 180),
        canonical_resolution=config.prepare.canonical_resolution,
    )

    filtergraph, metadata = build_prepare_filtergraph(plan, config)

    assert "crop=240:180:40:30" in filtergraph
    assert "perspective=" in filtergraph
    assert metadata.pre_crop_roi_xywh == (40, 30, 240, 180)
    assert metadata.quad_pre_crop_tl_tr_br_bl[0] == (10.0, 20.0)


def test_filtergraph_accepts_fractional_non_axis_aligned_manual_crop() -> None:
    config = _config(canonical_size_wh=(192, 192))
    plan = _fractional_perspective_plan(config)

    filtergraph, metadata = build_prepare_filtergraph(plan, config)

    assert "crop=300:220:10:10" in filtergraph
    assert "perspective=" in filtergraph
    assert metadata.rectified_content_size_wh == (254, 180)
    assert metadata.rectification_padding_lrtb == (0, 0, 0, 0)
    assert metadata.expected_output_size_wh == plan.prepared_size_wh == (192, 192)
    assert metadata.quad_pre_crop_tl_tr_br_bl[0][1] != pytest.approx(
        metadata.quad_pre_crop_tl_tr_br_bl[1][1]
    )
    assert metadata.quad_pre_crop_tl_tr_br_bl[1][0] != pytest.approx(
        metadata.quad_pre_crop_tl_tr_br_bl[2][0]
    )


def test_filtergraph_rejects_crop_plan_homography_inconsistent_with_quad() -> None:
    config = _config(canonical_size_wh=(192, 192))
    plan = _fractional_perspective_plan(config)
    malformed_forward = np.array(plan.H_raw_to_prepared_3x3, copy=True)
    malformed_forward[0, 1] += 0.05
    metadata = plan.to_metadata_dict()
    metadata["H_raw_to_prepared_3x3"] = malformed_forward.tolist()
    metadata["H_prepared_to_raw_3x3"] = np.linalg.inv(malformed_forward).tolist()
    malformed_plan = CropPlan.from_metadata_dict(metadata)

    with pytest.raises(VideoPreparationError, match="cannot be reproduced"):
        build_prepare_filtergraph(malformed_plan, config)


def test_canonical_enabled_uses_crop_plan_scale_and_centered_padding() -> None:
    config = _config(canonical_size_wh=(400, 400))
    plan = _landscape_plan(config)

    filtergraph, metadata = build_prepare_filtergraph(plan, config)

    assert "scale=400:300:flags=lanczos" in filtergraph
    assert "pad=400:400:0:50:color=black" in filtergraph
    assert metadata.canonical_geometry.uniform_scale == pytest.approx(1.25)
    assert metadata.canonical_geometry.scaled_size_wh == (400, 300)
    assert metadata.canonical_geometry.padding_top == 50
    assert metadata.canonical_geometry.padding_bottom == 50
    assert metadata.expected_output_size_wh == plan.prepared_size_wh == (400, 400)


def test_canonical_padding_keeps_odd_remainder_on_bottom() -> None:
    config = _config(canonical_size_wh=(202, 120))
    plan = make_manual_crop_plan(
        raw_frame_shape=(52, 102),
        points_tl_tr_br_bl=np.array(
            [[0.0, 0.0], [101.0, 0.0], [101.0, 51.0], [0.0, 51.0]]
        ),
        pre_crop_roi=None,
        canonical_resolution=config.prepare.canonical_resolution,
    )

    filtergraph, metadata = build_prepare_filtergraph(plan, config)

    assert "scale=202:103:flags=lanczos" in filtergraph
    assert "pad=202:120:0:8:color=black" in filtergraph
    assert metadata.canonical_geometry.padding_top == 8
    assert metadata.canonical_geometry.padding_bottom == 9


def test_canonical_disabled_adds_no_scale_or_padding() -> None:
    config = _config(canonical_enabled=False, canonical_size_wh=(400, 400))
    plan = _landscape_plan(config)

    filtergraph, metadata = build_prepare_filtergraph(plan, config)

    assert "scale=" not in filtergraph
    assert "pad=" not in filtergraph
    assert "canonical_uniform_scale" not in metadata.stages
    assert "canonical_padding" not in metadata.stages
    assert metadata.expected_output_size_wh == plan.native_size_wh == (320, 240)


def test_command_is_argument_list_and_uses_configured_encoding() -> None:
    encoding = FFmpegEncodingConfig(
        codec="libx264",
        pixel_format="yuv420p",
        preset="slow",
        crf=23,
        bframes=0,
        faststart=True,
    )
    config = _config(ffmpeg=encoding)
    plan = _landscape_plan(config)

    command, filtergraph, metadata = build_ffmpeg_prepare_command(
        Path("raw.avi"),
        Path("intermediate.mp4"),
        plan,
        config,
        start_frame=0,
        end_frame_exclusive=None,
        ffmpeg_binary=Path("managed-ffmpeg"),
    )

    assert isinstance(command, list)
    assert command[0] == "managed-ffmpeg"
    assert command[command.index("-vf") + 1] == filtergraph
    assert "-an" in command
    assert command[command.index("-c:v") + 1] == "libx264"
    assert command[command.index("-pix_fmt") + 1] == "yuv420p"
    assert command[command.index("-preset") + 1] == "slow"
    assert command[command.index("-crf") + 1] == "23"
    assert command[command.index("-bf") + 1] == "0"
    assert command[command.index("-x264-params") + 1] == "bframes=0"
    assert command[command.index("-fps_mode") + 1] == "passthrough"
    assert "-n" in command
    assert "+faststart" in command
    assert metadata.expected_output_size_wh == plan.prepared_size_wh
