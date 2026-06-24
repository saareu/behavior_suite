"""Project-local, non-authoritative cache for expensive raw-video probes."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import cv2
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from preprocess.models import (
    RawReadableCountProvenance,
    RawVideoFingerprint,
    VideoProbeResult,
)

RAW_PROBE_CACHE_SCHEMA_VERSION = "raw_probe_cache_v1"
_CACHE_DIRNAME = ".internal"
_CACHE_FILENAME = "raw_probe_cache.json"
_SIGNATURE_METHOD = "sha256:first-middle-last:65536"
_SIGNATURE_WINDOW_BYTES = 65_536


class RawProbeCacheRecord(BaseModel):
    """Versioned representation of one reusable raw-video probe."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    cache_schema_version: Literal["raw_probe_cache_v1"] = (
        RAW_PROBE_CACHE_SCHEMA_VERSION
    )
    raw_video_fingerprint: RawVideoFingerprint
    probe_timestamp: datetime
    frame_count_opencv_reported: int = Field(ge=0)
    frame_count_opencv_readable: int = Field(gt=0)
    fps_effective: float | None = Field(default=None, gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    codec: str | None = None
    pixel_format: str | None = None
    count_provenance: RawReadableCountProvenance
    probe_result: VideoProbeResult

    @model_validator(mode="after")
    def validate_consistency(self) -> RawProbeCacheRecord:
        """Reject internally inconsistent or non-reusable cache entries."""

        fingerprint = self.raw_video_fingerprint
        probe = self.probe_result
        if (self.width, self.height) != (fingerprint.width, fingerprint.height):
            raise ValueError("Cache dimensions differ from its fingerprint.")
        if (self.width, self.height) != (probe.width, probe.height):
            raise ValueError("Cache dimensions differ from its probe result.")
        if self.frame_count_opencv_reported != probe.frame_count_opencv_reported:
            raise ValueError("Cache reported counts are inconsistent.")
        if self.frame_count_opencv_readable != probe.frame_count_opencv_readable:
            raise ValueError("Cache readable counts are inconsistent.")
        if probe.source_path.expanduser().resolve(strict=False) != (
            fingerprint.resolved_path.expanduser().resolve(strict=False)
        ):
            raise ValueError("Cache probe source differs from its fingerprint path.")
        return self


def get_raw_probe_cache_path(preprocess_dir: Path) -> Path:
    """Return the clearly internal project-local raw-probe cache path."""

    return Path(preprocess_dir) / _CACHE_DIRNAME / _CACHE_FILENAME


def _read_video_dimensions(path: Path) -> tuple[int, int]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"OpenCV could not open raw video for fingerprinting: {path}")
        success, frame = capture.read()
    finally:
        capture.release()
    if not success or frame is None:
        raise ValueError(f"OpenCV could not decode raw video for fingerprinting: {path}")
    height, width = frame.shape[:2]
    if width <= 0 or height <= 0:
        raise ValueError(f"Raw video has invalid fingerprint dimensions: {path}")
    return int(width), int(height)


def _lightweight_content_signature(path: Path, file_size: int) -> str:
    """Hash bounded samples so same-stat replacements are also rejected."""

    digest = hashlib.sha256()
    final_offset = max(0, file_size - _SIGNATURE_WINDOW_BYTES)
    middle_offset = max(0, (file_size // 2) - (_SIGNATURE_WINDOW_BYTES // 2))
    offsets = sorted({0, middle_offset, final_offset})
    with path.open("rb") as stream:
        for offset in offsets:
            stream.seek(offset)
            chunk = stream.read(_SIGNATURE_WINDOW_BYTES)
            digest.update(offset.to_bytes(8, byteorder="big", signed=False))
            digest.update(len(chunk).to_bytes(8, byteorder="big", signed=False))
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_raw_video(path: Path) -> RawVideoFingerprint:
    """Build the exact cache identity from current filesystem and video state."""

    resolved = Path(path).expanduser().resolve(strict=True)
    before = resolved.stat()
    width, height = _read_video_dimensions(resolved)
    signature = _lightweight_content_signature(resolved, before.st_size)
    after = resolved.stat()
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
    ):
        raise ValueError(f"Raw video changed while it was fingerprinted: {resolved}")
    return RawVideoFingerprint(
        resolved_path=resolved,
        file_size_bytes=after.st_size,
        modified_time_ns=after.st_mtime_ns,
        width=width,
        height=height,
        content_signature=signature,
        content_signature_method=_SIGNATURE_METHOD,
    )


def read_raw_probe_cache(cache_path: Path) -> RawProbeCacheRecord | None:
    """Read a well-formed cache record, returning a miss for every read failure."""

    try:
        payload = Path(cache_path).read_text(encoding="utf-8")
        return RawProbeCacheRecord.model_validate_json(payload)
    except (OSError, UnicodeError, ValidationError, ValueError):
        return None


def load_matching_raw_probe_cache(
    cache_path: Path,
    fingerprint: RawVideoFingerprint,
) -> RawProbeCacheRecord | None:
    """Return a cache record only for one exact current fingerprint."""

    record = read_raw_probe_cache(cache_path)
    if record is None or record.raw_video_fingerprint != fingerprint:
        return None
    if record.frame_count_opencv_readable <= 0:
        return None
    return record


def validate_raw_probe_cache(cache_path: Path) -> RawProbeCacheRecord | None:
    """Re-fingerprint the cached source and return only an exact live match."""

    record = read_raw_probe_cache(cache_path)
    if record is None:
        return None
    try:
        current = fingerprint_raw_video(record.raw_video_fingerprint.resolved_path)
    except (OSError, ValueError):
        return None
    return record if current == record.raw_video_fingerprint else None


def build_raw_probe_cache_record(
    fingerprint: RawVideoFingerprint,
    probe: VideoProbeResult,
) -> RawProbeCacheRecord:
    """Build a current-session cache record from a completed full probe."""

    readable_count = probe.frame_count_opencv_readable
    if readable_count is None or readable_count <= 0:
        raise ValueError("A positive sequential readable count is required for caching.")
    return RawProbeCacheRecord(
        raw_video_fingerprint=fingerprint,
        probe_timestamp=datetime.now(UTC),
        frame_count_opencv_reported=probe.frame_count_opencv_reported,
        frame_count_opencv_readable=readable_count,
        fps_effective=probe.raw_fps_effective,
        width=probe.width,
        height=probe.height,
        codec=probe.codec,
        pixel_format=probe.pixel_format,
        count_provenance=RawReadableCountProvenance.VERIFIED_CURRENT_SESSION,
        probe_result=probe,
    )


def write_raw_probe_cache_atomic(
    cache_path: Path,
    record: RawProbeCacheRecord,
) -> None:
    """Atomically replace one cache after validating the complete temp payload."""

    destination = Path(cache_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(record.model_dump(mode="json"), stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        RawProbeCacheRecord.model_validate_json(
            temporary_path.read_text(encoding="utf-8")
        )
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)
