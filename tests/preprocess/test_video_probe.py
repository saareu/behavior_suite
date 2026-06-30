import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest

from preprocess import video_probe
from preprocess.exceptions import FFmpegRuntimeError, VideoProbeCancelledError, VideoProbeError
from preprocess.models import RawReadableCountProvenance
from preprocess.raw_probe_cache import (
    RAW_PROBE_CACHE_SCHEMA_VERSION,
    get_raw_probe_cache_path,
    read_raw_probe_cache,
    write_raw_probe_cache_atomic,
)


def _write_tiny_video(path: Path, frame_count: int = 5) -> Path:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (32, 24),
    )
    if not writer.isOpened():
        pytest.skip("This OpenCV build cannot create an MJPG AVI test fixture.")
    try:
        for index in range(frame_count):
            frame = np.full((24, 32, 3), index * 20, dtype=np.uint8)
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_opencv_reported_and_readable_frame_counts_are_distinct_and_correct(
    tmp_path: Path,
) -> None:
    video_path = _write_tiny_video(tmp_path / "tiny.avi", frame_count=5)

    assert video_probe.get_opencv_reported_frame_count(video_path) == 5
    assert video_probe.count_opencv_readable_frames(video_path) == 5


def test_probe_video_with_sequential_count(tmp_path: Path) -> None:
    video_path = _write_tiny_video(tmp_path / "tiny.avi", frame_count=4)

    result = video_probe.probe_video(video_path, require_sequential_count=True)

    assert result.width == 32
    assert result.height == 24
    assert result.frame_count_opencv_reported == 4
    assert result.frame_count_opencv_readable == 4
    assert result.opencv_fps == pytest.approx(10.0)


def test_read_raw_frame_at_index_returns_requested_decode_index(
    tmp_path: Path,
) -> None:
    video_path = _write_tiny_video(tmp_path / "indexed.avi", frame_count=5)

    frame = video_probe.read_raw_frame_at_index(video_path, 3)

    assert frame.shape == (24, 32, 3)
    assert float(frame.mean()) == pytest.approx(60.0, abs=8.0)


def test_read_raw_frame_at_index_uses_verified_direct_seek(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "fake.avi"
    video_path.write_bytes(b"placeholder")

    class FakeCapture:
        def __init__(self) -> None:
            self.position = 0
            self.released = False

        def set(self, _property: int, value: int) -> bool:
            self.position = int(value)
            return True

        def get(self, _property: int) -> float:
            return float(self.position)

        def read(self) -> tuple[bool, np.ndarray]:
            frame = np.full((2, 2, 3), self.position, dtype=np.uint8)
            self.position += 1
            return True, frame

        def release(self) -> None:
            self.released = True

    capture = FakeCapture()
    monkeypatch.setattr(video_probe, "_open_video", lambda _path: capture)

    frame = video_probe.read_raw_frame_at_index(video_path, 4)

    assert int(frame[0, 0, 0]) == 4
    assert capture.released is True


def test_unverified_direct_seek_falls_back_to_bounded_sequential_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "fake.avi"
    video_path.write_bytes(b"placeholder")

    class DirectUnverifiedCapture:
        def __init__(self) -> None:
            self.released = False

        def set(self, _property: int, _value: int) -> bool:
            return True

        def get(self, _property: int) -> float:
            return 0.0

        def read(self) -> tuple[bool, np.ndarray]:
            return True, np.full((2, 2, 3), 99, dtype=np.uint8)

        def release(self) -> None:
            self.released = True

    class SequentialCapture:
        def __init__(self) -> None:
            self.position = 0
            self.released = False

        def read(self) -> tuple[bool, np.ndarray]:
            frame = np.full((2, 2, 3), self.position, dtype=np.uint8)
            self.position += 1
            return True, frame

        def release(self) -> None:
            self.released = True

    direct = DirectUnverifiedCapture()
    fallback = SequentialCapture()
    captures = [direct, fallback]
    monkeypatch.setattr(video_probe, "_open_video", lambda _path: captures.pop(0))

    frame = video_probe.read_raw_frame_at_index(video_path, 3)

    assert int(frame[0, 0, 0]) == 3
    assert direct.released is True
    assert fallback.released is True


def test_read_raw_frame_at_index_rejects_unreadable_out_of_range_request(
    tmp_path: Path,
) -> None:
    video_path = _write_tiny_video(tmp_path / "short.avi", frame_count=3)

    with pytest.raises(VideoProbeError, match="Raw decode frame 5 is unreadable"):
        video_probe.read_raw_frame_at_index(video_path, 5)


def test_read_raw_frame_at_index_reports_bounded_fallback_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = tmp_path / "fake.avi"
    video_path.write_bytes(b"placeholder")

    class UnseekableCapture:
        def set(self, _property: int, _value: int) -> bool:
            return False

        def release(self) -> None:
            pass

    monkeypatch.setattr(video_probe, "_open_video", lambda _path: UnseekableCapture())

    with pytest.raises(VideoProbeError, match="bounded fallback decoding"):
        video_probe.read_raw_frame_at_index(
            video_path,
            2,
            max_fallback_decode_frames=1,
        )


def test_successful_sequential_probe_writes_versioned_project_cache(
    tmp_path: Path,
) -> None:
    video_path = _write_tiny_video(tmp_path / "tiny.avi", frame_count=4)
    cache_path = get_raw_probe_cache_path(tmp_path / "preprocess")

    result = video_probe.probe_video(
        video_path,
        require_sequential_count=True,
        cache_path=cache_path,
    )
    record = read_raw_probe_cache(cache_path)

    assert result.raw_readable_count_provenance is (
        RawReadableCountProvenance.VERIFIED_CURRENT_SESSION
    )
    assert record is not None
    assert record.cache_schema_version == RAW_PROBE_CACHE_SCHEMA_VERSION
    assert record.frame_count_opencv_readable == 4
    assert record.raw_video_fingerprint.resolved_path == video_path.resolve()
    assert record.raw_video_fingerprint.file_size_bytes == video_path.stat().st_size
    assert record.raw_video_fingerprint.modified_time_ns == video_path.stat().st_mtime_ns
    assert (record.raw_video_fingerprint.width, record.raw_video_fingerprint.height) == (
        32,
        24,
    )
    assert record.raw_video_fingerprint.content_signature


def test_matching_fingerprint_restores_cached_count_without_sequential_decode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = _write_tiny_video(tmp_path / "tiny.avi", frame_count=4)
    cache_path = get_raw_probe_cache_path(tmp_path / "preprocess")
    video_probe.probe_video(
        video_path,
        require_sequential_count=True,
        cache_path=cache_path,
    )

    def fail_if_called(*_args: object, **_kwargs: object) -> int:
        raise AssertionError("matching cache unexpectedly triggered sequential decode")

    monkeypatch.setattr(video_probe, "count_opencv_readable_frames", fail_if_called)

    restored = video_probe.probe_video(
        video_path,
        require_sequential_count=True,
        cache_path=cache_path,
    )

    assert restored.frame_count_opencv_readable == 4
    assert restored.raw_readable_count_provenance is (
        RawReadableCountProvenance.VALIDATED_REUSABLE_CACHE
    )


def test_path_only_cache_match_is_rejected_after_file_size_change(
    tmp_path: Path,
) -> None:
    video_path = _write_tiny_video(tmp_path / "tiny.avi", frame_count=4)
    cache_path = get_raw_probe_cache_path(tmp_path / "preprocess")
    video_probe.probe_video(
        video_path,
        require_sequential_count=True,
        cache_path=cache_path,
    )
    with video_path.open("ab") as stream:
        stream.write(b"cache-size-mismatch")

    restored = video_probe.probe_video(
        video_path,
        require_sequential_count=False,
        cache_path=cache_path,
    )

    assert restored.frame_count_opencv_readable is None
    assert restored.raw_readable_count_provenance is None


def test_path_only_cache_match_is_rejected_after_mtime_change(tmp_path: Path) -> None:
    video_path = _write_tiny_video(tmp_path / "tiny.avi", frame_count=4)
    cache_path = get_raw_probe_cache_path(tmp_path / "preprocess")
    video_probe.probe_video(
        video_path,
        require_sequential_count=True,
        cache_path=cache_path,
    )
    stat = video_path.stat()
    os.utime(
        video_path,
        ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000),
    )

    restored = video_probe.probe_video(
        video_path,
        require_sequential_count=False,
        cache_path=cache_path,
    )

    assert restored.frame_count_opencv_readable is None


def test_cache_is_rejected_when_fingerprinted_dimensions_differ(tmp_path: Path) -> None:
    video_path = _write_tiny_video(tmp_path / "tiny.avi", frame_count=4)
    cache_path = get_raw_probe_cache_path(tmp_path / "preprocess")
    video_probe.probe_video(
        video_path,
        require_sequential_count=True,
        cache_path=cache_path,
    )
    payload = json.loads(cache_path.read_text(encoding="utf-8"))
    payload["raw_video_fingerprint"]["width"] = 64
    payload["width"] = 64
    payload["probe_result"]["width"] = 64
    cache_path.write_text(json.dumps(payload), encoding="utf-8")

    restored = video_probe.probe_video(
        video_path,
        require_sequential_count=False,
        cache_path=cache_path,
    )

    assert restored.frame_count_opencv_readable is None


@pytest.mark.parametrize("cache_contents", [None, "{not-json"])
def test_missing_or_malformed_cache_is_a_nonfatal_miss(
    tmp_path: Path,
    cache_contents: str | None,
) -> None:
    video_path = _write_tiny_video(tmp_path / "tiny.avi", frame_count=4)
    cache_path = get_raw_probe_cache_path(tmp_path / "preprocess")
    if cache_contents is not None:
        cache_path.parent.mkdir(parents=True)
        cache_path.write_text(cache_contents, encoding="utf-8")

    result = video_probe.probe_video(
        video_path,
        require_sequential_count=False,
        cache_path=cache_path,
    )

    assert result.frame_count_opencv_readable is None


def test_cache_write_uses_atomic_replace_and_preserves_old_file_on_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = _write_tiny_video(tmp_path / "tiny.avi", frame_count=4)
    cache_path = get_raw_probe_cache_path(tmp_path / "preprocess")
    video_probe.probe_video(
        video_path,
        require_sequential_count=True,
        cache_path=cache_path,
    )
    record = read_raw_probe_cache(cache_path)
    assert record is not None
    cache_path.write_text("old-cache", encoding="utf-8")
    replace_calls: list[tuple[Path, Path]] = []

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        replace_calls.append((Path(source), Path(destination)))
        raise OSError("simulated replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(OSError, match="replace failure"):
        write_raw_probe_cache_atomic(cache_path, record)

    assert replace_calls and replace_calls[0][1] == cache_path
    assert cache_path.read_text(encoding="utf-8") == "old-cache"
    assert list(cache_path.parent.glob(f".{cache_path.name}.*.tmp")) == []


def test_force_recount_bypasses_and_replaces_cached_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = _write_tiny_video(tmp_path / "tiny.avi", frame_count=4)
    cache_path = get_raw_probe_cache_path(tmp_path / "preprocess")
    video_probe.probe_video(
        video_path,
        require_sequential_count=True,
        cache_path=cache_path,
    )
    monkeypatch.setattr(
        video_probe,
        "count_opencv_readable_frames",
        lambda *_args, **_kwargs: 7,
    )

    recounted = video_probe.probe_video(
        video_path,
        require_sequential_count=True,
        cache_path=cache_path,
        force_recount=True,
    )
    replaced = read_raw_probe_cache(cache_path)

    assert recounted.frame_count_opencv_readable == 7
    assert recounted.raw_readable_count_provenance is (
        RawReadableCountProvenance.VERIFIED_CURRENT_SESSION
    )
    assert replaced is not None
    assert replaced.frame_count_opencv_readable == 7


def test_probe_video_skips_full_count_when_disabled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = _write_tiny_video(tmp_path / "tiny.avi")

    def fail_if_called(_path: Path) -> int:
        raise AssertionError("sequential counter should not be called")

    monkeypatch.setattr(video_probe, "count_opencv_readable_frames", fail_if_called)

    result = video_probe.probe_video(video_path, require_sequential_count=False)

    assert result.frame_count_opencv_readable is None
    assert result.frame_count_opencv_reported == 5


def test_readable_video_remains_probeable_without_ffprobe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video_path = _write_tiny_video(tmp_path / "tiny.avi")
    monkeypatch.setattr(
        video_probe,
        "resolve_ffprobe_binary",
        lambda _path=None: (_ for _ in ()).throw(
            FFmpegRuntimeError("ffprobe executable was not found")
        ),
    )

    result = video_probe.probe_video(video_path, require_sequential_count=False)

    assert result.ffprobe_succeeded is False
    assert result.ffprobe_error is not None
    assert "not found" in result.ffprobe_error
    assert result.raw_fps_effective_method == "opencv_fps"


def test_unreadable_video_raises_domain_error(tmp_path: Path) -> None:
    unreadable_path = tmp_path / "broken.avi"
    unreadable_path.write_text("not a video", encoding="utf-8")

    with pytest.raises(VideoProbeError, match="OpenCV"):
        video_probe.probe_video(unreadable_path, require_sequential_count=False)


def test_sequential_count_cancellation_never_returns_partial_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeCapture:
        def __init__(self) -> None:
            self.read_count = 0
            self.released = False

        def read(self) -> tuple[bool, np.ndarray]:
            self.read_count += 1
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def release(self) -> None:
            self.released = True

    capture = FakeCapture()
    cancellation_checks = 0

    def cancellation_requested() -> bool:
        nonlocal cancellation_checks
        cancellation_checks += 1
        return cancellation_checks >= 4

    monkeypatch.setattr(video_probe, "_open_video", lambda _path: capture)

    progress = []
    with pytest.raises(VideoProbeCancelledError, match="cancelled"):
        video_probe.count_opencv_readable_frames(
            tmp_path / "raw.avi",
            cancellation_requested=cancellation_requested,
            progress_callback=progress.append,
        )

    assert capture.read_count == 3
    assert capture.released is True
    assert progress[-1].phase == "cancelling"
    assert all(event.phase != "complete" for event in progress)


@pytest.mark.parametrize(
    ("reported_total", "expected_indeterminate"),
    [(600, False), (0, True)],
)
def test_sequential_count_progress_is_monotonic_and_honest_about_denominator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    reported_total: int,
    expected_indeterminate: bool,
) -> None:
    frame_count = 600

    class FakeCapture:
        def __init__(self) -> None:
            self.read_count = 0

        def get(self, _property: int) -> float:
            return float(reported_total)

        def read(self) -> tuple[bool, np.ndarray | None]:
            if self.read_count >= frame_count:
                return False, None
            self.read_count += 1
            return True, np.zeros((2, 2, 3), dtype=np.uint8)

        def release(self) -> None:
            pass

    monkeypatch.setattr(video_probe, "_open_video", lambda _path: FakeCapture())
    progress = []

    count = video_probe.count_opencv_readable_frames(
        tmp_path / "raw.avi",
        progress_callback=progress.append,
    )

    completed = [event.completed_units for event in progress]
    assert count == frame_count
    assert completed == sorted(completed)
    assert completed[0] == 0
    assert completed[-1] == frame_count
    assert all(event.is_indeterminate is expected_indeterminate for event in progress)
    expected_total = None if expected_indeterminate else reported_total
    assert all(event.total_units == expected_total for event in progress)
    assert all(
        event.total_is_estimate is (not expected_indeterminate)
        for event in progress
    )
