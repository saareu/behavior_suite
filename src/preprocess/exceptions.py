"""Domain-specific exceptions for video preprocessing."""


class PreprocessError(Exception):
    """Base exception for all preprocessing domain failures."""


class PreprocessConfigError(PreprocessError, ValueError):
    """Raised when preprocessing configuration cannot be loaded."""


class VideoProbeError(PreprocessError):
    """Raised when a video cannot be opened or probed as requested."""


class ExternalTimingError(PreprocessError, ValueError):
    """Raised when a MATLAB workspace or selected timing vector is invalid."""
