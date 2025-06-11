"""
shapeGMMTorch: A modular PyTorch library for GMM-based shape clustering.
"""

from .core import ShapeGMM
from .align import (
    torch_remove_center_of_geometry,
    torch_iterative_align_kronecker,
    torch_iterative_align_uniform
)
from .em import kronecker, uniform
from .utils import io, plotting, similarity, generation

__all__ = [
    "ShapeGMM",
    "torch_remove_center_of_geometry",
    "torch_iterative_align_kronecker",
    "torch_iterative_align_uniform",
    "kronecker",
    "uniform",
    "io",
    "plotting",
    "similarity",
    "generation"
]

