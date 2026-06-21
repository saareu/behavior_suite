"""Isolated UTF-8 text logging for future preprocessing orchestration."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

from preprocess.exceptions import ProcessingLogError

_LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"
_LOG_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"


def _logger_name(log_path: Path) -> str:
    digest = hashlib.sha256(str(log_path).encode("utf-8")).hexdigest()
    return f"behavior_suite.preprocess.file.{digest}"


def create_processing_logger(log_path: Path) -> logging.Logger:
    """Create or reuse one isolated UTF-8 file logger for an exact path.

    Repeated calls for the same resolved path return the same logger without
    adding a second file handler. The logger does not propagate to root or
    application-global handlers.

    Raises:
        ProcessingLogError: If the output directory or UTF-8 file handler
            cannot be created.
    """

    path = Path(log_path).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path = path.resolve()
    except OSError as exc:
        raise ProcessingLogError(
            f"Could not create processing-log directory: {exc}"
        ) from exc

    logger = logging.getLogger(_logger_name(path))
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.disabled = False
    resolved_text = str(path)
    for handler in logger.handlers:
        if (
            isinstance(handler, logging.FileHandler)
            and str(Path(handler.baseFilename).resolve()) == resolved_text
        ):
            return logger

    try:
        handler = logging.FileHandler(
            path,
            mode="a",
            encoding="utf-8",
            delay=False,
        )
    except OSError as exc:
        raise ProcessingLogError(
            f"Could not create processing log {path}: {exc}"
        ) from exc
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter(
            fmt=_LOG_FORMAT,
            datefmt=_LOG_DATE_FORMAT,
        )
    )
    logger.addHandler(handler)
    return logger
