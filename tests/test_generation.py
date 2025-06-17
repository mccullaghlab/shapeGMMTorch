import pytest
import numpy as np
from shapeGMMTorch import generation

n_frames = 50
n_atoms = 4
n_components = 2

def test_rand_component_ids():
    random_nums = np.random.rand(n_frames)
    weights = np.random.rand(n_components)
    weights /= np.sum(weights)
    component_ids = generation.component_ids_from_rand(random_nums, weights)
    assert component_ids.shape == (n_frames,)
    assert isinstance(component_ids, np.ndarray)
    assert np.issubdtype(component_ids.dtype, np.integer)

def test_gen_mv():
    mean = np.zeros((n_atoms, 3))
    precision = np.eye(n_atoms,dtype=np.float64)
    traj = generation.gen_mv(mean, precision, n_frames)
    assert traj.shape == (n_frames, n_atoms, 3)
    assert isinstance(traj, np.ndarray)
