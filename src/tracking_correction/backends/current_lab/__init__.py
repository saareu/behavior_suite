"""Current-lab minimal-intervention corrector (validated Pipeline 20 backend).

Modules here are mechanical copies of the validated legacy corrector with only
import-path and packaging adjustments. Correction rules are not redesigned.
"""

from tracking_correction.backends.current_lab.correction_params import CorrectionParams
from tracking_correction.backends.current_lab.id_corrector_minimal import process_video_minimal
from tracking_correction.backends.current_lab.tracking_data import TrackingData

__all__ = [
    "CorrectionParams",
    "TrackingData",
    "process_video_minimal",
]
