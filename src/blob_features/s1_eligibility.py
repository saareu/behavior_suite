"""Validate eligible Subsystem 01 artifacts for Subsystem 05.

S5 requires prepared video, cropped background, prepared sync, and prepare
metadata. It does **not** require any S2–S4 outputs and must succeed when those
subsystems have not been run.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from blob_features.contracts import REQUIRED_S1_FILES, BlobFeaturesError
from pose_finalization.timing import SourceTiming, load_source_timing
from preprocess.exceptions import SyncValidationError
from preprocess.sync_writer import load_prepared_sync_npz


@dataclass(frozen=True, slots=True)
class EligibleS1Input:
    """Resolved read-only Subsystem 01 inputs for blob feature extraction."""

    session_root: Path
    preprocess_dir: Path
    prepared_video: Path
    cropped_background: Path
    prepared_sync: Path
    prepare_meta: Path
    prepare_meta_payload: Mapping[str, Any]
    source_timing: SourceTiming
    frame_count: int
    fps_header: float
    frame_width: int
    frame_height: int


def validate_s1_for_blob_features(session_root_or_preprocess: Path | str) -> EligibleS1Input:
    """Validate required S1 artifacts and return resolved paths + timing.

    Accepts either a session root (containing ``preprocess/``) or the
    preprocess directory itself. Upstream S2–S4 directories are never consulted.
    """

    session_root, preprocess_dir = _resolve_preprocess_dir(session_root_or_preprocess)
    if not preprocess_dir.is_dir():
        raise BlobFeaturesError(f"Preprocess directory does not exist: {preprocess_dir}")

    missing = [name for name in REQUIRED_S1_FILES if not (preprocess_dir / name).is_file()]
    if missing:
        raise BlobFeaturesError(
            "Missing required Subsystem 01 artifacts for blob features: "
            + ", ".join(missing)
        )

    prepared_video = preprocess_dir / "prepared_video.mp4"
    cropped_background = preprocess_dir / "cropped_background.png"
    prepared_sync = preprocess_dir / "prepared_sync.npz"
    prepare_meta = preprocess_dir / "prepare_meta.json"
    prepare_meta_payload = _load_json_object(prepare_meta)

    try:
        source_timing = load_source_timing(prepared_sync)
    except Exception as exc:
        # load_source_timing raises PoseFinalizationError; re-wrap for S5 callers.
        raise BlobFeaturesError(
            f"Could not load authoritative S1 prepared_sync.npz: {exc}"
        ) from exc

    try:
        sync = load_prepared_sync_npz(prepared_sync)
    except (OSError, SyncValidationError) as exc:
        raise BlobFeaturesError(
            f"Subsystem 01 prepared_sync.npz is unreadable or invalid: {exc}"
        ) from exc

    expected_count = int(sync.frame_count_used_for_sleap)
    if expected_count != source_timing.frame_count:
        raise BlobFeaturesError(
            "prepared_sync frame_count_used_for_sleap does not match loaded "
            f"source timing frame count: sync={expected_count}, "
            f"timing={source_timing.frame_count}."
        )

    frame_width, frame_height, reported_count = _probe_prepared_video(prepared_video)
    if reported_count is not None and reported_count != expected_count:
        raise BlobFeaturesError(
            "Prepared-video header frame count does not match the authoritative S1 "
            f"sync contract: reported={reported_count}, expected={expected_count}."
        )

    prepared_meta = prepare_meta_payload.get("prepared_video")
    if isinstance(prepared_meta, Mapping):
        for key in (
            "opencv_reported_frame_count",
            "opencv_readable_frame_count",
            "frame_count_used_for_sleap",
        ):
            value = prepared_meta.get(key)
            if value is not None and int(value) != expected_count:
                raise BlobFeaturesError(
                    f"prepare_meta.json {key} does not match the authoritative "
                    f"S1 sync frame count {expected_count}."
                )
        meta_w = prepared_meta.get("width")
        meta_h = prepared_meta.get("height")
        if (
            meta_w is not None
            and meta_h is not None
            and (int(meta_w) != frame_width or int(meta_h) != frame_height)
        ):
            raise BlobFeaturesError(
                "prepare_meta.json prepared_video size does not match the "
                f"decoded video frame: meta=({int(meta_w)}, {int(meta_h)}), "
                f"video=({frame_width}, {frame_height})."
            )

    bg_height, bg_width = _probe_background(cropped_background)
    if (bg_height, bg_width) != (frame_height, frame_width):
        raise BlobFeaturesError(
            "cropped_background.png size does not match prepared-video frame size: "
            f"background=({bg_width}, {bg_height}), video=({frame_width}, {frame_height})."
        )

    return EligibleS1Input(
        session_root=session_root,
        preprocess_dir=preprocess_dir,
        prepared_video=prepared_video,
        cropped_background=cropped_background,
        prepared_sync=prepared_sync,
        prepare_meta=prepare_meta,
        prepare_meta_payload=prepare_meta_payload,
        source_timing=source_timing,
        frame_count=expected_count,
        fps_header=float(source_timing.fps_header),
        frame_width=frame_width,
        frame_height=frame_height,
    )


def _resolve_preprocess_dir(session_root_or_preprocess: Path | str) -> tuple[Path, Path]:
    candidate = Path(session_root_or_preprocess).expanduser().resolve()
    if not candidate.exists():
        raise BlobFeaturesError(f"Input path does not exist: {candidate}")
    if not candidate.is_dir():
        raise BlobFeaturesError(f"Input path must be a directory: {candidate}")

    contains_required = any((candidate / name).exists() for name in REQUIRED_S1_FILES)
    if candidate.name.lower() == "preprocess" or contains_required:
        preprocess_dir = candidate
        session_root = candidate.parent
    else:
        session_root = candidate
        preprocess_dir = candidate / "preprocess"
    return session_root, preprocess_dir


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BlobFeaturesError(f"prepare_meta.json is unreadable or invalid: {exc}") from exc
    if not isinstance(payload, dict):
        raise BlobFeaturesError("prepare_meta.json must contain a JSON object.")
    return payload


def _probe_prepared_video(video_path: Path) -> tuple[int, int, int | None]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise BlobFeaturesError(f"prepared_video.mp4 is unreadable: {video_path}")

    reported_count_value = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    reported_count = (
        int(reported_count_value)
        if math.isfinite(reported_count_value) and reported_count_value > 0
        else None
    )
    try:
        success, frame = capture.read()
        if not success or frame is None:
            raise BlobFeaturesError(
                "prepared_video.mp4 did not yield a readable first frame."
            )
    except cv2.error as exc:
        raise BlobFeaturesError(
            f"prepared_video.mp4 could not be decoded: {exc}"
        ) from exc
    finally:
        capture.release()

    height, width = int(frame.shape[0]), int(frame.shape[1])
    if height <= 0 or width <= 0:
        raise BlobFeaturesError(
            f"prepared_video.mp4 has invalid frame size: ({width}, {height})."
        )
    return width, height, reported_count


def _probe_background(background_path: Path) -> tuple[int, int]:
    image = cv2.imread(str(background_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise BlobFeaturesError(
            f"cropped_background.png is unreadable: {background_path}"
        )
    array = np.asarray(image)
    if array.ndim < 2:
        raise BlobFeaturesError(
            f"cropped_background.png has unsupported shape: {array.shape}"
        )
    height, width = int(array.shape[0]), int(array.shape[1])
    if height <= 0 or width <= 0:
        raise BlobFeaturesError(
            f"cropped_background.png has invalid size: ({width}, {height})."
        )
    return height, width
