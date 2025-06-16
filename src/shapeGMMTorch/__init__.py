"""
shapeGMMTorch: A modular PyTorch library for GMM-based shape clustering.
"""

from .core import ShapeGMM
from .align import (
    remove_center_of_geometry,
    maximum_likelihood_kronecker_alignment,
    maximum_likelihood_uniform_alignment,
    maximum_likelihood_kronecker_alignment_frame_weighted,
    maximum_likelihood_uniform_alignment_frame_weighted,    
)
from .generation import gen_mv, generate, component_ids_from_rand

__all__ = [
    "ShapeGMM",
    "remove_center_of_geometry",
    "align_kronecker",
    "align_uniform",
    "maximum_likelihood_kronecker_alignment",
    "maximum_likelihood_uniform_alignment",
    "maximum_likelihood_kronecker_alignment_frame_weighted",
    "maximum_likelihood_uniform_alignment_weighted",
]

