import logging
import re
from pathlib import Path

from preprocess.logging_utils import create_processing_logger


def _flush(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        handler.flush()


def _close(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()


def test_logger_creates_output_file(tmp_path: Path) -> None:
    log_path = tmp_path / "processing_log.txt"
    logger = create_processing_logger(log_path)
    try:
        assert log_path.is_file()
        assert logger.propagate is False
    finally:
        _close(logger)


def test_logger_writes_timestamp_severity_and_message(tmp_path: Path) -> None:
    log_path = tmp_path / "processing_log.txt"
    logger = create_processing_logger(log_path)
    try:
        logger.info("prepared video validation passed")
        _flush(logger)
        content = log_path.read_text(encoding="utf-8")

        assert re.search(
            r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2} INFO "
            r"prepared video validation passed$",
            content.strip(),
        )
    finally:
        _close(logger)


def test_calling_helper_twice_does_not_duplicate_output_lines(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "processing_log.txt"
    first = create_processing_logger(log_path)
    second = create_processing_logger(log_path)
    try:
        assert first is second
        assert len(first.handlers) == 1
        second.warning("single event")
        _flush(second)

        lines = log_path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 1
        assert lines[0].endswith("WARNING single event")
    finally:
        _close(first)


def test_separate_log_paths_remain_isolated(tmp_path: Path) -> None:
    first_path = tmp_path / "first" / "processing_log.txt"
    second_path = tmp_path / "second" / "processing_log.txt"
    first = create_processing_logger(first_path)
    second = create_processing_logger(second_path)
    try:
        first.info("first event")
        second.error("second event")
        _flush(first)
        _flush(second)

        first_content = first_path.read_text(encoding="utf-8")
        second_content = second_path.read_text(encoding="utf-8")
        assert "first event" in first_content
        assert "second event" not in first_content
        assert "second event" in second_content
        assert "first event" not in second_content
    finally:
        _close(first)
        _close(second)
