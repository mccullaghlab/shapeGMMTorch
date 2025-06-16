import pytest
import numpy as np
from shapeGMMTorch import generation

def test_rand_component_ids():
    n_frames = 5
    n_components = 2
    random_nums = np.random.rand(n_frames)
    weights = np.random.rand(n_components)
    weights /= np.sum(weights)
    component_ids = generation.component_ids_from_rand(random_nums, weights)
    assert component_ids.shape == (n_frames,)
    assert isinstance(component_ids, np.ndarray)
    assert np.issubdtype(component_ids.dtype, np.integer)

def test_gen_mv():
    mean = np.zeros((10, 3))
    precision = np.eye(10,dtype=np.float64)
    n_frames = 5
    traj = generation.gen_mv(mean, precision, n_frames)
    assert traj.shape == (n_frames, 10, 3)
    assert isinstance(traj, np.ndarray)
