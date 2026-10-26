"""Family-specific inference adapters (dyadic collapse / focal directional)."""

from behavioral_inference.adapters.dyadic import (
    collapse_to_unordered_pairs,
    run_dyadic_inference,
)
from behavioral_inference.adapters.focal import run_focal_inference

__all__ = [
    "collapse_to_unordered_pairs",
    "run_dyadic_inference",
    "run_focal_inference",
]
