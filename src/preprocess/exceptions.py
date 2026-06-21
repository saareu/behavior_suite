"""Domain-specific exceptions for video preprocessing."""


class PreprocessError(Exception):
    """Base exception for all preprocessing domain failures."""


class PreprocessConfigError(PreprocessError, ValueError):
    """Raised when preprocessing configuration cannot be loaded."""


class VideoProbeError(PreprocessError):
    """Raised when a video cannot be opened or probed as requested."""


class ExternalTimingError(PreprocessError, ValueError):
    """Raised when a MATLAB workspace or selected timing vector is invalid."""


class CropPlanError(PreprocessError):
    """Raised when crop or canonical geometry is invalid."""


class PreCropError(CropPlanError):
    """Raised when a detection pre-crop cannot be resolved safely."""


class CageDetectionError(PreprocessError):
    """Raised when automatic cage detection cannot produce a valid CropPlan."""


class VideoPreparationError(PreprocessError):
    """Raised when the ffmpeg preparation stage cannot complete safely."""
