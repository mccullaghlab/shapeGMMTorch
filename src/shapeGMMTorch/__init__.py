"""
ShapeGMMTorch is a Gaussian Mixture Model fitting code specifically for particle positions (in size-and-shape space, hence shapeGMM)

Provides:
- ShapeGMM: A PyTorch-accelerated EM algorithm for fitting GMMs to particle positions
- sgmm_fit_with_attempts: Utility for robust model fitting via multiple random initializations.
- sgmm_cross_validate_component_scan: Grid search over different numbers of mixture components.
"""

from .core import ShapeGMM
from .utils import sgmm_fit_with_attempts, sgmm_cross_validate_component_scan

__all__ = [
    "ShapeGMM",
    "sgmm_fit_with_attempts",
    "sgmm_cross_validate_component_scan"
]
