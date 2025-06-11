"""
shapeGMMTorch: A modular PyTorch library for GMM-based shape clustering.
"""

from .core import ShapeGMM
from .align import (
    torch_remove_center_of_geometry,
    torch_iterative_align_kronecker,
    torch_iterative_align_uniform,
    torch_iterative_align_kronecker_weighted,
    torch_iterative_align_uniform_weighted,    
)
from .generation import gen_mv, generate, cluster_ids_from_rand

__all__ = [
    "ShapeGMM",
    "torch_remove_center_of_geometry",
    "torch_iterative_align_kronecker",
    "torch_iterative_align_uniform",
    "gen_mv", 
    "generate", 
    "cluster_ids_from_rand"
]

