# pipelines/00_prepare.py
from __future__ import annotations

import fractions
import json
import os
import random
import re
import subprocess
import time
from pathlib import Path
from typing import Any
from fractions import Fraction
import cv2
import numpy as np

# Ensure project root on path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.io.path_utils import ensure_directory_exists, get_artifact_path, load_config_with_global
from src.preprocess.cage_detector import make_plan
from src.utils.platform_utils import configure_platform

LOG_PREFIX = "[prepare]"


# ----------------------------
# ffmpeg/ffprobe helpers
# ----------------------------

def _log(msg: str) -> None:
    print(f"{LOG_PREFIX} {msg}")


def _rat_to_float(r: str | None) -> float | None:
    if not r or r == "0/0":
        return None
    try:
        num, den = r.split("/")
        return float(num) / float(den)
    except Exception:
        return None


def _rat_to_fraction(r: str | None) -> fractions.Fraction | None:
    if not r or r == "0/0":
        return None
    try:
        return fractions.Fraction(r)
    except Exception:
        return None


def _ffprobe_cmd(path: str) -> list[str]:
    return [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,pix_fmt,codec_tag_string,avg_frame_rate,r_frame_rate,time_base,nb_frames",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        path,
    ]


def probe_video_metadata(path: str) -> dict[str, Any]:
    out = subprocess.check_output(_ffprobe_cmd(path), text=True)
    return json.loads(out)


def _fps_effective_from_ffprobe(stream: dict[str, Any], format_: dict[str, Any]) -> tuple[float, str]:
    afr = _rat_to_float(stream.get("avg_frame_rate"))
    rfr = _rat_to_float(stream.get("r_frame_rate"))
    nb_frames_str = stream.get("nb_frames")
    nb_frames = int(nb_frames_str) if nb_frames_str else None
    duration_str = format_.get("duration")
    duration = float(duration_str) if duration_str else None
    fps_fallback = nb_frames / duration if nb_frames and duration and duration > 0 else None

    if afr is not None:
        return afr, "ffprobe_avg"
    if rfr is not None:
        return rfr, "ffprobe_r"
    if fps_fallback is not None:
        return fps_fallback, "ffprobe_nbframes_duration"

    raise ValueError("ffprobe did not return a valid frame rate.")


def _check_executables() -> None:
    import shutil

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found in PATH. Install ffmpeg and retry.")
    if shutil.which("ffprobe") is None:
        raise RuntimeError("ffprobe not found in PATH. Install ffmpeg and retry.")


def _format_cmd(cmd: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(cmd)
    import shlex

    return shlex.join(cmd)


_SHOWINFO_RE = re.compile(r"n:\s*(\d+).*pts_time:\s*([0-9\.]+)")


def extract_raw_pts_time_for_trim(
    raw_video_path: str,
    start_frame: int | None,
    end_frame: int | None,
) -> np.ndarray:
    vf_parts = []
    if start_frame is not None or end_frame is not None:
        sf = int(start_frame or 0)
        if end_frame is None:
            vf_parts.append(f"select=gte(n\\,{sf})")
        else:
            ef = int(end_frame)
            vf_parts.append(f"select=between(n\\,{sf}\\,{ef-1})")
    vf_parts.append("showinfo")
    vf = ",".join(vf_parts)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-nostats",
        "-loglevel",
        "info",
        "-i",
        raw_video_path,
        "-an",
        "-vf",
        vf,
        "-f",
        "null",
        "-",
    ]

    pts_times: list[float] = []
    p = subprocess.Popen(cmd, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True)
    assert p.stderr is not None
    for line in p.stderr:
        m = _SHOWINFO_RE.search(line)
        if m:
            pts_times.append(float(m.group(2)))

    rc = p.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg showinfo failed with code {rc}")

    if len(pts_times) == 0:
        raise RuntimeError("No pts_time extracted from showinfo (unexpected).")

    return np.asarray(pts_times, dtype=np.float64)


# ----------------------------
# geometry helpers
# ----------------------------

def _order_tl_tr_br_bl(pts: np.ndarray) -> np.ndarray:
    # pts: (4,2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1)[:, 0]
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]
    return np.stack([tl, tr, br, bl], axis=0).astype(np.float32)


def _clamp_roi(x0: int, y0: int, x1: int, y1: int, w: int, h: int) -> tuple[int, int, int, int]:
    x0 = max(0, int(x0))
    y0 = max(0, int(y0))
    x1 = min(int(w) - 1, int(x1))
    y1 = min(int(h) - 1, int(y1))
    return x0, y0, x1, y1


def _quad_abs_from_plan(plan) -> np.ndarray:
    if hasattr(plan, "pre_crop_roi") and hasattr(plan, "rim_rect_trim"):
        x0_pc, y0_pc, _, _ = plan.pre_crop_roi
        rect = plan.rim_rect_trim
        pts = cv2.boxPoints(rect).astype(np.float32)
        pts[:, 0] += float(x0_pc)
        pts[:, 1] += float(y0_pc)
        return pts
    if hasattr(plan, "src_quad"):
        return np.array(plan.src_quad, dtype=np.float32)
    raise ValueError("Plan lacks pre_crop_roi/rim_rect_trim or src_quad.")


def save_debug_overlay(
    video_path: str,
    frame_idx: int,
    quad_abs: np.ndarray,
    roi: tuple[int, int, int, int],
    out_png: str,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open debug video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idx = int(frame_idx)
    if frame_count > 0:
        idx = min(max(0, idx), frame_count - 1)
        if idx != int(frame_idx):
            _log(
                f"debug overlay frame index {int(frame_idx)} out of range for {video_path} "
                f"(n={frame_count}); clamped to {idx}."
            )

    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Failed to read debug frame.")

    q = np.array(quad_abs).astype(np.int32)
    cv2.polylines(frame, [q], True, (0, 0, 255), 3)

    x0, y0, x1, y1 = roi
    cv2.rectangle(frame, (int(x0), int(y0)), (int(x1), int(y1)), (0, 255, 0), 2)

    outp = Path(out_png)
    outp.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(outp), frame)


def _even_size(w: int, h: int) -> tuple[int, int]:
    return w - (w % 2), h - (h % 2)


# ----------------------------
# fps sampling
# ----------------------------

def _sample_pts_stats_pyav(video_path: str, max_frames: int = 300) -> dict[str, float] | None:
    try:
        import av  # type: ignore
    except Exception:
        return None

    container = av.open(video_path)
    stream = container.streams.video[0]
    pts_sec: list[float] = []
    for frame in container.decode(video=0):
        if frame.pts is None:
            continue
        pts_sec.append(float(frame.pts * stream.time_base))
        if len(pts_sec) >= max_frames:
            break
    container.close()

    if len(pts_sec) < 2:
        return None
    dts = np.diff(np.array(pts_sec, dtype=np.float64))
    return {
        "n": int(len(dts)),
        "median_dt": float(np.median(dts)),
        "std_dt": float(np.std(dts)),
        "min_dt": float(np.min(dts)),
        "max_dt": float(np.max(dts)),
    }


# ----------------------------
# prepared background + mask
# ----------------------------

def estimate_background_prepared(
    video_path: str,
    sample_every_n: int,
    max_samples: int,
    method: str = "median",
) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open prepared video for background estimation.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    samples: list[np.ndarray] = []
    idx = 0
    while len(samples) < max_samples:
        if frame_count and idx >= frame_count:
            break
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        samples.append(gray)
        idx += sample_every_n
    cap.release()

    if not samples:
        raise RuntimeError("No frames sampled for prepared background.")

    stack = np.stack(samples, axis=0)
    bg = np.median(stack, axis=0).astype(np.uint8)
    return bg


def reencode_prepared_opencv(
    src_path: str,
    dst_path: str,
    fps: int,
) -> int:
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open rectified video for OpenCV re-encode.")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        cap.release()
        raise RuntimeError("Invalid rectified video dimensions for OpenCV re-encode.")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(dst_path, fourcc, float(fps), (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Failed to open VideoWriter for prepared video.")

    n_written = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        writer.write(frame)
        n_written += 1

    cap.release()
    writer.release()
    if n_written <= 0:
        raise RuntimeError("No frames written during OpenCV re-encode.")
    return n_written


def _mask_overlay(background_gray: np.ndarray, mask: np.ndarray) -> np.ndarray:
    bgr = cv2.cvtColor(background_gray, cv2.COLOR_GRAY2BGR)
    if mask.dtype != np.uint8:
        mask = mask.astype(np.uint8)
    outside = mask == 0
    bgr[outside] = (0, 0, 255)
    return bgr


def _enforce_strictly_increasing(times: np.ndarray, min_step: float) -> np.ndarray:
    if times.size == 0:
        return times
    out = times.astype(np.float64, copy=True)
    for i in range(1, out.size):
        if out[i] <= out[i - 1]:
            out[i] = out[i - 1] + min_step
    return out


def _resolve_initial_left_crop(
    prepare_cfg: dict[str, Any],
    src_w: int,
) -> tuple[bool, int | None]:
    crop_cfg = prepare_cfg.get("initial_left_crop", {})
    enabled = bool(crop_cfg.get("enabled", False))
    if not enabled:
        return False, None

    keep_left_of_x = crop_cfg.get("keep_left_of_x")
    if keep_left_of_x is None:
        raise ValueError("prepare.initial_left_crop requires keep_left_of_x when enabled.")

    x_cutoff = int(keep_left_of_x)
    if not (0 <= x_cutoff < src_w):
        raise ValueError(
            "prepare.initial_left_crop expects 0 <= keep_left_of_x < raw video width. "
            f"Received keep_left_of_x={x_cutoff}, raw_w={src_w}."
        )
    return True, x_cutoff


def run_initial_left_crop_ffmpeg(
    src_video_path: str,
    out_video_path: str,
    src_h: int,
    x_cutoff: int,
    preset: str,
    fps_num: int | None = None,
    fps_den: int | None = None,
) -> tuple[int, int]:
    out_w = int(x_cutoff + 1)
    if out_w <= 0:
        raise ValueError("Initial left crop produced non-positive width.")

    vf_parts = [f"crop={out_w}:{int(src_h)}:0:0"]
    if fps_num and fps_den:
        # Keep the intermediate crop at the exact source cadence.
        vf_parts.append(f"settb=1/{fps_num}")
        vf_parts.append(f"setpts=N*{fps_den}")
    vf = ",".join(vf_parts)

    track_timescale = "1000"
    if fps_num:
        track_timescale = str(int(fps_num))
    cmd = [
        "ffmpeg",
        "-y",
        "-fflags",
        "+genpts",
        "-i",
        src_video_path,
        "-vf",
        vf,
        "-an",
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        preset,
        "-movflags",
        "+faststart",
        "-enc_time_base",
        "demux",
        "-fps_mode",
        "passthrough",
        "-video_track_timescale",
        track_timescale,
        "-crf",
        "18",
        "-x264-params",
        "bframes=0",
        out_video_path,
    ]
    _log("initial left crop ffmpeg cmd:")
    _log(_format_cmd(cmd))
    subprocess.run(cmd, check=True)

    return int(out_w), int(src_h)


# ----------------------------
# ffmpeg prepare
# ----------------------------

def _build_filtergraph(
    start_frame: int | None,
    end_frame: int | None,
    fps_out: float,
    fps_num: int | None,
    fps_den: int | None,
    roi: tuple[int, int, int, int],
    quad_abs: np.ndarray,
    perspective_interpolation: str,
    rotated_90: bool,
    flipped_lr: bool,
    canonical_enabled: bool,
    canonical_w: int,
    canonical_h: int,
    canonical_flags: str,
) -> tuple[str, dict[str, Any]]:
    x0, y0, x1, y1 = roi
    roi_w = int(x1 - x0 + 1)
    roi_h = int(y1 - y0 + 1)

    quad_abs = _order_tl_tr_br_bl(np.array(quad_abs, dtype=np.float32))
    tl, tr, br, bl = quad_abs

    # quad in ROI-local coords
    tl2 = tl - [x0, y0]
    tr2 = tr - [x0, y0]
    br2 = br - [x0, y0]
    bl2 = bl - [x0, y0]

    persp = (
        "perspective="
        f"x0={float(tl2[0])}:y0={float(tl2[1])}:"
        f"x1={float(tr2[0])}:y1={float(tr2[1])}:"
        f"x2={float(bl2[0])}:y2={float(bl2[1])}:"
        f"x3={float(br2[0])}:y3={float(br2[1])}:"
        f"sense=source:interpolation={perspective_interpolation}"
    )

    vf_parts: list[str] = []

    if start_frame is not None or end_frame is not None:
        sf = int(start_frame or 0)
        if end_frame is None:
            vf_parts.append(f"select=gte(n\\,{sf})")
        else:
            ef = int(end_frame)
            vf_parts.append(f"select=between(n\\,{sf}\\,{ef-1})")

    fps_out_str = f"{fps_out:.9f}"
    # Use capped denominator for settb/setpts to avoid MPEG-4 and non-monotonic DTS errors
    if fps_num and fps_den:
        vf_parts.append(f"settb=1/{fps_num}")
        vf_parts.append(f"setpts=N*{fps_den}")
    else:
        vf_parts.append(f"settb=1/{fps_out_str}")
        vf_parts.append("setpts=N")



    vf_parts.append(f"crop={roi_w}:{roi_h}:{x0}:{y0}")
    vf_parts.append(persp)

    if rotated_90:
        vf_parts.append("transpose=1")
    if flipped_lr:
        vf_parts.append("hflip")

    vf_parts.append("crop=trunc(iw/2)*2:trunc(ih/2)*2")

    pre_scale_w = roi_w
    pre_scale_h = roi_h
    if rotated_90:
        pre_scale_w, pre_scale_h = roi_h, roi_w
    out_w_pre_scale, out_h_pre_scale = _even_size(pre_scale_w, pre_scale_h)
    out_w_final = out_w_pre_scale
    out_h_final = out_h_pre_scale
    uniform_scale = 1.0
    scaled_w = out_w_pre_scale
    scaled_h = out_h_pre_scale
    pad_left = 0
    pad_top = 0
    if canonical_enabled:
        out_w_final = int(canonical_w)
        out_h_final = int(canonical_h)
        if out_w_final * out_h_pre_scale <= out_h_final * out_w_pre_scale:
            scaled_w = out_w_final
            scaled_h = (out_h_pre_scale * out_w_final) // out_w_pre_scale
        else:
            scaled_h = out_h_final
            scaled_w = (out_w_pre_scale * out_h_final) // out_h_pre_scale
        if scaled_w <= 0 or scaled_h <= 0:
            raise ValueError("Canonical scaling produced non-positive size.")
        if scaled_w > out_w_final:
            scaled_w = out_w_final
        if scaled_h > out_h_final:
            scaled_h = out_h_final
        pad_left = int((out_w_final - scaled_w) // 2)
        pad_top = int((out_h_final - scaled_h) // 2)
        uniform_scale = float(scaled_w) / float(out_w_pre_scale)
        vf_parts.append(f"scale={scaled_w}:{scaled_h}:flags={canonical_flags}")
        vf_parts.append(f"pad={out_w_final}:{out_h_final}:{pad_left}:{pad_top}:color=black")
        vf_parts.append("setsar=1")

    return ",".join(vf_parts), {
        "roi_w": roi_w,
        "roi_h": roi_h,
        "quad_roi_local": np.stack([tl2, tr2, br2, bl2], axis=0),
        "out_w_pre_scale": int(out_w_pre_scale),
        "out_h_pre_scale": int(out_h_pre_scale),
        "out_w_final": int(out_w_final),
        "out_h_final": int(out_h_final),
        "scale_sx": float(uniform_scale),
        "scale_sy": float(uniform_scale),
        "uniform_scale": float(uniform_scale),
        "scaled_w": int(scaled_w),
        "scaled_h": int(scaled_h),
        "pad_left": int(pad_left),
        "pad_top": int(pad_top),
    }

def _make_roi_even(roi: tuple[int,int,int,int]) -> tuple[int,int,int,int]:
    x0, y0, x1, y1 = roi
    w = x1 - x0 + 1
    h = y1 - y0 + 1
    if w % 2 == 1 and x1 > x0:
        x1 -= 1
    if h % 2 == 1 and y1 > y0:
        y1 -= 1
    return (x0, y0, x1, y1)

def run_prepare_ffmpeg(
    raw_video_path: str,
    out_video_path: str,
    plan,
    src_w: int,
    src_h: int,
    start_frame: int | None,
    end_frame: int | None,
    roi_margin_px: int,
    quad_abs: np.ndarray | None,
    fps_out: float,
    fps_num: int | None,
    fps_den: int | None,
    perspective_interpolation: str,
    canonical_enabled: bool,
    canonical_w: int,
    canonical_h: int,
    canonical_flags: str,
    preset: str,
) -> tuple[tuple[int, int, int, int], np.ndarray, dict[str, Any]]:
    if quad_abs is None:
        quad_abs = _quad_abs_from_plan(plan)
    quad_abs = _order_tl_tr_br_bl(np.array(quad_abs, dtype=np.float32))

    xs = quad_abs[:, 0]
    ys = quad_abs[:, 1]
    x0 = int(np.floor(xs.min())) - roi_margin_px
    y0 = int(np.floor(ys.min())) - roi_margin_px
    x1 = int(np.ceil(xs.max())) + roi_margin_px
    y1 = int(np.ceil(ys.max())) + roi_margin_px
    x0, y0, x1, y1 = _clamp_roi(x0, y0, x1, y1, src_w, src_h)
    roi = (x0, y0, x1, y1)
    roi = _make_roi_even(roi)

    vf, fg_meta = _build_filtergraph(
        start_frame=start_frame,
        end_frame=end_frame,
        fps_out=fps_out,
        fps_num=fps_num,
        fps_den=fps_den,
        roi=roi,
        quad_abs=quad_abs,
        perspective_interpolation=perspective_interpolation,
        rotated_90=bool(getattr(plan, "rotated_90", False)),
        flipped_lr=bool(getattr(plan, "flipped_lr", False)),
        canonical_enabled=canonical_enabled,
        canonical_w=canonical_w,
        canonical_h=canonical_h,
        canonical_flags=canonical_flags,
    )

    cmd = [
        "ffmpeg",
        "-y",
        "-fflags",
        "+genpts",
        "-i",
        raw_video_path,
        "-vf",
        vf,
        "-an",
        "-map",
        "0:v:0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        preset,
        "-movflags",
        "+faststart",
        "-enc_time_base",
        "demux",
        "-fps_mode",
        "passthrough",
        "-video_track_timescale",
        "1000",
        "-crf",
        "18",
        "-x264-params",
        "bframes=0",
    ]

    cmd.append(out_video_path)

    _log("ffmpeg cmd:")
    _log(_format_cmd(cmd))
    subprocess.run(cmd, check=True)

    return roi, quad_abs, fg_meta


# ----------------------------
# transform helpers
# ----------------------------

def _compute_raw_to_prepared_homography(
    roi: tuple[int, int, int, int],
    roi_local_quad: np.ndarray,
    out_w_preflip: int,
    out_h_preflip: int,
    rotated_90: bool,
    flipped_lr: bool,
    out_w_final: int,
    out_h_final: int,
) -> np.ndarray:
    x0, y0, _, _ = roi
    src = np.array([
        roi_local_quad[0],
        roi_local_quad[1],
        roi_local_quad[3],
        roi_local_quad[2],
    ], dtype=np.float32)
    dst = np.array(
        [[0, 0], [out_w_preflip - 1, 0], [0, out_h_preflip - 1], [out_w_preflip - 1, out_h_preflip - 1]],
        dtype=np.float32,
    )

    hp = cv2.getPerspectiveTransform(src, dst).astype(np.float64)
    t_raw2roi = np.array([[1, 0, -x0], [0, 1, -y0], [0, 0, 1]], dtype=np.float64)
    h_raw2preflip = hp @ t_raw2roi

    # apply transpose/hflip (left-multiply)
    a = np.eye(3, dtype=np.float64)
    w, h = out_w_preflip, out_h_preflip
    if rotated_90:
        a_transpose = np.array([[0, -1, h - 1], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
        a = a_transpose @ a
        w, h = h, w
    if flipped_lr:
        a_hflip = np.array([[-1, 0, w - 1], [0, 1, 0], [0, 0, 1]], dtype=np.float64)
        a = a_hflip @ a

    h_raw2prep = a @ h_raw2preflip

    # even-crop drops right/bottom edges; mapping remains valid for remaining pixels
    if w != out_w_final or h != out_h_final:
        pass

    return h_raw2prep


# ----------------------------
# QC checks
# ----------------------------

def _qc_prepared_video(
    video_path: str,
    expected_w: int,
    expected_h: int,
) -> None:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to open prepared video for QC.")

    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if cap_w != expected_w or cap_h != expected_h:
        cap.release()
        raise RuntimeError(
            f"Prepared dimensions mismatch: capture={cap_w}x{cap_h}, meta={expected_w}x{expected_h}."
        )

    ok, _ = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Failed to read first prepared frame.")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    rng = random.Random(0)
    indices = [0]
    if frame_count > 1:
        for _ in range(5):
            indices.append(rng.randint(0, max(0, frame_count - 1)))
    else:
        indices.append(0)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, _ = cap.read()
        if not ok:
            cap.release()
            raise RuntimeError(f"Failed to read prepared frame at index {idx}.")

    mid = 0 if frame_count == 0 else frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(mid))
    ok, _ = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError("Prepared video is not seekable (mid-frame read failed).")

    if expected_w % 2 != 0 or expected_h % 2 != 0:
        raise RuntimeError("Prepared dimensions are not even.")


# ----------------------------
# main pipeline
# ----------------------------

def main(cfg_path: str = "configs/preprocess.yaml") -> None:
    # ...existing code...
    configure_platform()
    _check_executables()

    cfg = load_config_with_global(cfg_path)
    exp_name = cfg["exp_name"]
    iteration_name = cfg.get("iteration_name")

    raw_video_path = get_artifact_path(exp_name=exp_name, artifact_type="raw_video", iteration_name=iteration_name)
    prepared_video_path = Path(
        get_artifact_path(exp_name=exp_name, artifact_type="prepared_video", iteration_name=iteration_name)
    )
    ensure_directory_exists(str(prepared_video_path.parent))
    rectified_video_path = prepared_video_path.with_name("rectified_raw.mp4")
    initial_left_crop_video_path = prepared_video_path.with_name("initial_left_crop_raw.mp4")

    # raw metadata + fps
    raw_meta = probe_video_metadata(raw_video_path)
    raw_stream = raw_meta.get("streams", [])[0]
    raw_format = raw_meta.get("format", {})
    raw_w = int(raw_stream.get("width"))
    raw_h = int(raw_stream.get("height"))
    source_fps_frac = _rat_to_fraction(raw_stream.get("avg_frame_rate"))
    if source_fps_frac is None:
        source_fps_frac = _rat_to_fraction(raw_stream.get("r_frame_rate"))
    source_fps_num = int(source_fps_frac.numerator) if source_fps_frac is not None else None
    source_fps_den = int(source_fps_frac.denominator) if source_fps_frac is not None else None

    prepare_cfg = cfg.get("prepare", {})
    sanitize_cfg = cfg.get("sanitize", {})
    preset = str(sanitize_cfg.get("preset", "veryfast"))

    initial_left_crop_enabled, keep_left_of_x = _resolve_initial_left_crop(prepare_cfg, raw_w)
    source_video_path = raw_video_path
    source_w = raw_w
    source_h = raw_h
    if initial_left_crop_enabled:
        if initial_left_crop_video_path.exists():
            initial_left_crop_video_path.unlink()
        source_w, source_h = run_initial_left_crop_ffmpeg(
            src_video_path=raw_video_path,
            out_video_path=str(initial_left_crop_video_path),
            src_h=raw_h,
            x_cutoff=int(keep_left_of_x),
            preset=preset,
            fps_num=source_fps_num,
            fps_den=source_fps_den,
        )
        source_video_path = str(initial_left_crop_video_path)

    fps_effective, fps_method = _fps_effective_from_ffprobe(raw_stream, raw_format)
    fps_effective_float = float(fps_effective)
    
    # 1. OpenCV will round the float to 3 decimal places internally. 
    # We must ensure that the resulting fraction has a numerator <= 65535.
    fps_round = round(fps_effective_float, 3)
    safe_fps = fps_round
    
    # Test offsets to find a safe fraction for the MP4V codec
    for offset in [0.0, -0.001, 0.001, -0.002, 0.002]:
        test_fps = round(fps_round + offset, 3)
        # Replicate OpenCV's internal fraction math
        frac = fractions.Fraction(int(round(test_fps * 1000)), 1000)
        if frac.numerator <= 65535:
            safe_fps = float(test_fps)
            fps_num = frac.numerator
            fps_den = frac.denominator
            break
            
    # 2. Use this perfectly safe float and fraction everywhere
    fps_sleap = safe_fps
    fps_out = safe_fps

    print(f"Original FPS: {fps_effective_float:.6f}")
    print(f"Safe OpenCV FPS: {fps_sleap:.3f} (Fraction: {fps_num}/{fps_den})")

    pts_stats = _sample_pts_stats_pyav(source_video_path, max_frames=300)
    vfr_detected = False
    vfr_method = None
    rfr = _rat_to_float(raw_stream.get("r_frame_rate"))
    afr = _rat_to_float(raw_stream.get("avg_frame_rate"))
    if afr is not None and rfr is not None:
        if abs(afr - rfr) / max(afr, 1e-9) > 0.001:
            vfr_detected = True
            vfr_method = "ffprobe_rate_mismatch"
    if pts_stats is not None:
        if pts_stats["median_dt"] > 0 and pts_stats["std_dt"] / pts_stats["median_dt"] > 0.01:
            vfr_detected = True
            vfr_method = "pyav_pts" if vfr_method is None else f"{vfr_method}+pyav_pts"


    # trim config
    trim_cfg = cfg.get("trim", {})
    start_frame = trim_cfg.get("start_frame")
    end_frame = trim_cfg.get("end_frame")

    dev_cfg = cfg.get("dev_clip", {})
    if dev_cfg.get("enable"):
        if start_frame is None:
            start_frame = 0
        if end_frame is None:
            duration_sec = dev_cfg.get("duration_sec")
            if duration_sec is not None:
                end_frame = int(round(int(start_frame) + float(duration_sec) * fps_effective))

    # cage detection
    plan_cfg = cfg.get("cage_detect", {})
    background, plan = make_plan(
        Path(source_video_path),
        sample_step=int(plan_cfg.get("sample_step", 500)),
        pad_px=int(plan_cfg.get("pad_px", 2)),
        save_debug=Path("debug_detector")
    )

    quad_abs = _order_tl_tr_br_bl(_quad_abs_from_plan(plan))

    # roi
    roi_margin_px = int(cfg.get("prepare", {}).get("roi_margin_px", 40))
    xs = quad_abs[:, 0]
    ys = quad_abs[:, 1]
    x0 = int(np.floor(xs.min())) - roi_margin_px
    y0 = int(np.floor(ys.min())) - roi_margin_px
    x1 = int(np.ceil(xs.max())) + roi_margin_px
    y1 = int(np.ceil(ys.max())) + roi_margin_px
    roi = _clamp_roi(x0, y0, x1, y1, source_w, source_h)

    # raw debug overlay
    background_path = Path(get_artifact_path(exp_name, "report_prep_background", iteration_name))
    report_dir = background_path.parent
    # save background image
    
    cv2.imwrite(str(background_path), background)
    raw_overlay_path = report_dir / "qc_raw_overlay.png"
    save_debug_overlay(source_video_path, int(start_frame or 0), quad_abs, roi, str(raw_overlay_path))

    # ffmpeg
    perspective_interpolation = str(prepare_cfg.get("perspective_interpolation", "cubic"))
    if perspective_interpolation not in {"linear", "cubic"}:
        raise ValueError("prepare.perspective_interpolation must be 'linear' or 'cubic'.")

    canonical_cfg = prepare_cfg.get("canonical_resolution", {})
    canonical_enabled = bool(canonical_cfg.get("enabled", False))
    canonical_w = int(canonical_cfg.get("width", 928))
    canonical_h = int(canonical_cfg.get("height", 528))
    canonical_flags = str(canonical_cfg.get("flags", "lanczos"))
    if canonical_enabled:
        if canonical_w <= 0 or canonical_h <= 0:
            raise ValueError("prepare.canonical_resolution width/height must be positive.")
        if canonical_w % 2 != 0 or canonical_h % 2 != 0:
            raise ValueError("prepare.canonical_resolution width/height must be even.")

    roi_used, quad_used, fg_meta = run_prepare_ffmpeg(
        raw_video_path=source_video_path,
        out_video_path=str(rectified_video_path),
        plan=plan,
        src_w=source_w,
        src_h=source_h,
        start_frame=start_frame,
        end_frame=end_frame,
        roi_margin_px=roi_margin_px,
        quad_abs=quad_abs,
        fps_out=fps_out,
        fps_num=fps_num,
        fps_den=fps_den,
        perspective_interpolation=perspective_interpolation,
        canonical_enabled=canonical_enabled,
        canonical_w=canonical_w,
        canonical_h=canonical_h,
        canonical_flags=canonical_flags,
        preset=preset,
    )

    # Re-encode rectified video to final prepared video at float FPS for SLEAP
    if prepared_video_path.exists():
        prepared_video_path.unlink()
    n_prepared_written = reencode_prepared_opencv(
        str(rectified_video_path),
        str(prepared_video_path),
        fps=fps_sleap,
    )
    if rectified_video_path.exists():
        rectified_video_path.unlink()
    if initial_left_crop_video_path.exists():
        initial_left_crop_video_path.unlink()

    # prepared probe
    prep_meta = probe_video_metadata(str(prepared_video_path))
    prep_stream = prep_meta.get("streams", [])[0]
    prep_format = prep_meta.get("format", {})

    prep_w = int(prep_stream.get("width"))
    prep_h = int(prep_stream.get("height"))
    prep_fps, _ = _fps_effective_from_ffprobe(prep_stream, prep_format)

    # ---- Sidecar sync: prepared frame j -> raw decode index and raw pts_time ----
    # This is the scientific timing source (raw PTS), while prepared MP4 is CFR for SLEAP.
    cap = cv2.VideoCapture(str(prepared_video_path))
    n_prepared_cv = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    n_prepared = int(n_prepared_written)
    if n_prepared <= 0:
        raise RuntimeError("Could not determine prepared frame count for sync generation.")

    # Option A: SLEAP backend cannot reliably read the last frame.
    n_sleap = max(0, n_prepared - 1)

    sf = int(start_frame or 0)
    ef = int(end_frame) if end_frame is not None else (sf + n_prepared)

    if end_frame is not None:
        expected = int(end_frame) - sf
        if expected != n_prepared:
            raise RuntimeError(
                f"Prepared frame count ({n_prepared}) != expected trim count ({expected}). "
                "This indicates resampling or selection mismatch; remove -r/-fps_mode or any fps filters."
            )

    raw_pts_time = extract_raw_pts_time_for_trim(raw_video_path, start_frame=sf, end_frame=ef)
    if raw_pts_time.shape[0] < n_sleap:
        raise RuntimeError(
            f"raw_pts_time count ({raw_pts_time.shape[0]}) < usable prepared frames ({n_sleap}). "
            "This indicates duplication/dropping; check prepare timing."
        )

    prepared_frame_idx = np.arange(n_sleap, dtype=np.int64)
    raw_frame_idx = sf + prepared_frame_idx
    prepared_pts_time = prepared_frame_idx.astype(np.float64) / float(fps_sleap)
    prepared_time_source = "prepared_pts_time = j / fps_sleap (OpenCV CFR)"
    cfr_notes = (
        "Prepared MP4 is CFR at fps_sleap via OpenCV re-encode; "
        "rectified intermediate uses in-filtergraph timestamp rewrite (settb+setpts); "
        "container uses enc_time_base=demux and fps_mode=passthrough (no resampling)."
    )
    frame_mapping_notes = (
        "Prepared MP4 is CFR at integer fps_sleap via OpenCV re-encode; "
        "rectified intermediate uses in-filtergraph CFR-like timestamp rewrite (settb+setpts); "
        "container stores demux time base, fps_mode=passthrough; "
        "SLEAP backend is expected to read frames [0..nb_frames_sleap_usable-1], "
        "ignoring the last container frame."
    )

    meta_path = Path(get_artifact_path(exp_name, "prepared_video_meta", iteration_name))
    sync_path = Path(get_artifact_path(exp_name, "prepared_sync_npz", iteration_name))
    np.savez_compressed(
        sync_path,
        prepared_frame_idx=prepared_frame_idx,
        prepared_pts_time=prepared_pts_time,
        raw_frame_idx=raw_frame_idx,
        raw_pts_time=raw_pts_time[:n_sleap],
        fps_out=float(fps_sleap),
        start_frame=int(sf),
        end_frame=int(ef) if end_frame is not None else None,
    )

    # background from prepared
    bg_cfg = cfg.get("background", {})
    bg = estimate_background_prepared(
        str(prepared_video_path),
        sample_every_n=int(bg_cfg.get("sample_every_n", 500)),
        max_samples=int(bg_cfg.get("max_samples", 80)),
    )
    bg_path = Path(get_artifact_path(exp_name, "report_prep_background", iteration_name))
    bg_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(bg_path), bg)








    # transforms
    quad_roi_local = fg_meta["quad_roi_local"]
    roi_w = fg_meta["roi_w"]
    roi_h = fg_meta["roi_h"]
    out_w_pre_scale = int(fg_meta["out_w_pre_scale"])
    out_h_pre_scale = int(fg_meta["out_h_pre_scale"])
    out_w_final = int(fg_meta["out_w_final"])
    out_h_final = int(fg_meta["out_h_final"])
    uniform_scale = float(fg_meta.get("uniform_scale", fg_meta.get("scale_sx", 1.0)))
    scaled_w = int(fg_meta.get("scaled_w", out_w_pre_scale))
    scaled_h = int(fg_meta.get("scaled_h", out_h_pre_scale))
    pad_left = int(fg_meta.get("pad_left", 0))
    pad_top = int(fg_meta.get("pad_top", 0))
    if canonical_enabled:
        pad_frac_w = 0.0 if out_w_final == 0 else (out_w_final - scaled_w) / float(out_w_final)
        pad_frac_h = 0.0 if out_h_final == 0 else (out_h_final - scaled_h) / float(out_h_final)
        _log(
            "canonical: pre_scale_wh={}x{}, canonical_wh={}x{}, uniform_scale={:.6f}, "
            "scaled_wh={}x{}, pad_left_top={}, {}, pad_frac_wh={:.3f},{:.3f}".format(
                out_w_pre_scale,
                out_h_pre_scale,
                out_w_final,
                out_h_final,
                uniform_scale,
                scaled_w,
                scaled_h,
                pad_left,
                pad_top,
                pad_frac_w,
                pad_frac_h,
            )
        )
        if pad_frac_w > 0.10 or pad_frac_h > 0.10:
            _log("WARNING: canonical padding fraction exceeds 0.10 in width or height.")

    h_raw_to_prepared = _compute_raw_to_prepared_homography(
        roi=roi_used,
        roi_local_quad=quad_roi_local,
        out_w_preflip=roi_w,
        out_h_preflip=roi_h,
        rotated_90=bool(getattr(plan, "rotated_90", False)),
        flipped_lr=bool(getattr(plan, "flipped_lr", False)),
        out_w_final=out_w_pre_scale,
        out_h_final=out_h_pre_scale,
    )
    if canonical_enabled:
        s = np.array([[uniform_scale, 0.0, 0.0], [0.0, uniform_scale, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
        p = np.array([[1.0, 0.0, float(pad_left)], [0.0, 1.0, float(pad_top)], [0.0, 0.0, 1.0]], dtype=np.float64)
        h_raw_to_prepared = p @ s @ h_raw_to_prepared
    h_prepared_to_raw = np.linalg.inv(h_raw_to_prepared)

    # QC
    _qc_prepared_video(
        str(prepared_video_path),
        expected_w=out_w_final,
        expected_h=out_h_final,
    )

    # metadata
    raw_path_obj = Path(raw_video_path)
    raw_stat = raw_path_obj.stat()

    meta = {
        "schema_version": "prepare_v2",
        "experiment": {
            "name": exp_name,
            "iteration": iteration_name,
            "pipeline": "00_prepare",
            "created_at_unix": float(time.time()),
        },
        "raw_video": {
            "path": str(raw_video_path),
            "stat": {"size_bytes": int(raw_stat.st_size), "mtime_unix": float(raw_stat.st_mtime)},
            "ffprobe": {
                "format": {"duration": raw_format.get("duration")},
                "streams": [
                    {
                        "width": raw_stream.get("width"),
                        "height": raw_stream.get("height"),
                        "pix_fmt": raw_stream.get("pix_fmt"),
                        "codec_tag_string": raw_stream.get("codec_tag_string"),
                        "avg_frame_rate": raw_stream.get("avg_frame_rate"),
                        "r_frame_rate": raw_stream.get("r_frame_rate"),
                        "time_base": raw_stream.get("time_base"),
                        "nb_frames": raw_stream.get("nb_frames"),
                    }
                ],
            },
            "width": raw_w,
            "height": raw_h,
            "pix_fmt": raw_stream.get("pix_fmt"),
            "codec_tag_string": raw_stream.get("codec_tag_string"),
            "duration_sec": float(raw_format.get("duration")) if raw_format.get("duration") else None,
            "avg_frame_rate": raw_stream.get("avg_frame_rate"),
            "r_frame_rate": raw_stream.get("r_frame_rate"),
            "time_base": raw_stream.get("time_base"),
            "nb_frames": int(raw_stream.get("nb_frames")) if raw_stream.get("nb_frames") else None,
            "fps_effective": float(fps_effective),
            "fps_effective_method": fps_method,
            "is_vfr_detected": bool(vfr_detected),
            "vfr_detect_method": vfr_method,
            "pts_sample_stats": pts_stats,
        },
        "prepare": {
            "trim": {"start_frame": int(start_frame or 0), "end_frame": int(end_frame) if end_frame else None},
            "cfr": {
                "enabled": True,
                "fps_out": float(fps_sleap),
                "notes": cfr_notes,
            },
            "initial_left_crop": {
                "enabled": bool(initial_left_crop_enabled),
                "keep_left_of_x": int(keep_left_of_x) if keep_left_of_x is not None else None,
                "source_size_wh": [int(source_w), int(source_h)],
            },
            "filters": {
                "roi": [int(v) for v in roi_used],
                "roi_margin_px": int(roi_margin_px),
                "perspective_interpolation": perspective_interpolation,
                "transpose": bool(getattr(plan, "rotated_90", False)),
                "hflip": bool(getattr(plan, "flipped_lr", False)),
                "even_crop": True,
            },
            "canonical_resolution": {
                "enabled": bool(canonical_enabled),
                "width": int(canonical_w),
                "height": int(canonical_h),
                "flags": canonical_flags,
                "uniform_scale": float(uniform_scale),
                "scaled_wh": [int(scaled_w), int(scaled_h)],
                "pad_left": int(pad_left),
                "pad_top": int(pad_top),
            },
            "geometry": {
                "quad_raw_tl_tr_br_bl": quad_used.astype(float).tolist(),
                "quad_roi_local_tl_tr_br_bl": quad_roi_local.astype(float).tolist(),
                "out_size_wh": [int(out_w_final), int(out_h_final)],
                "H_raw_to_prepared_3x3": h_raw_to_prepared.tolist(),
                "H_prepared_to_raw_3x3": h_prepared_to_raw.tolist(),
            },
            "frame_mapping": {
                "mode": "by_index_plus_raw_pts_sidecar",
                "sync_npz": str(sync_path),
                "sleap_usable_frames": int(n_sleap),
                "raw_index_formula": "raw_frame_idx = start_frame + prepared_frame_idx",
                "timing_source": "raw_pts_time from sync_npz (original timestamps)",
                "prepared_time_source": prepared_time_source,
                "notes": frame_mapping_notes,
            },
        },
        "prepared_video": {
            "path": str(prepared_video_path),
            "ffprobe": {
                "format": {"duration": prep_format.get("duration")},
                "streams": [
                    {
                        "width": prep_stream.get("width"),
                        "height": prep_stream.get("height"),
                        "pix_fmt": prep_stream.get("pix_fmt"),
                        "codec_tag_string": prep_stream.get("codec_tag_string"),
                        "avg_frame_rate": prep_stream.get("avg_frame_rate"),
                        "time_base": prep_stream.get("time_base"),
                        "nb_frames": prep_stream.get("nb_frames"),
                    }
                ],
            },
            "width": prep_w,
            "height": prep_h,
            "pix_fmt": prep_stream.get("pix_fmt"),
            "codec_tag_string": prep_stream.get("codec_tag_string"),
            "duration_sec": float(prep_format.get("duration")) if prep_format.get("duration") else None,
            "nb_frames": int(prep_stream.get("nb_frames")) if prep_stream.get("nb_frames") else None,
            "nb_frames_cv": int(n_prepared_cv),
            "nb_frames_sleap_usable": int(n_sleap),
            "avg_frame_rate": prep_stream.get("avg_frame_rate"),
            "time_base": prep_stream.get("time_base"),
            "fps_effective": float(prep_fps),
            "encoding": {
                "codec": "opencv_mp4v",
                "fourcc": "mp4v",
                "pix_fmt": "yuv420p",
                "preset": preset,
                "crf": 18,
                "x264_params": "bframes=0",
                "enc_time_base": "demux",
                "fps_mode": "passthrough",
                "fps_sleap": float(fps_sleap),
                "method": "opencv_reencode",
                "faststart": True,
            },
        },
        "outputs": {
            "background_png": str(bg_path),
            "mask_npz": None,
            "qc_raw_overlay_png": str(raw_overlay_path),
            "prepared_sync_npz": str(sync_path),
        },
    }

    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta["fps_header"] = float(fps_sleap)
    meta["prepared_video"]["fps_header"] = float(fps_sleap)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    _log(
        "summary: raw_fps={:.6f}, vfr_detected={}, fps_sleap={:.6f}, prepared={}x{}, trim=[{}, {}]".format(
            fps_effective,
            vfr_detected,
            fps_sleap,
            prep_w,
            prep_h,
            int(start_frame or 0),
            "None" if end_frame is None else int(end_frame),
        )
    )
    _log(f"wrote prepared video: {prepared_video_path}")
    _log(f"wrote metadata: {meta_path}")


if __name__ == "__main__":
    main()
