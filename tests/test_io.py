import numpy as np
import pytest
import torch
from shapeGMMTorch.utils import io

dtype = torch.float64
device = torch.device('cpu')
n_frames = 100
n_atoms = 4
n_components = 2

def test_cross_validate_component_scan_shapes():
    traj = np.random.randn(n_frames, n_atoms, 3)
    components = np.array([1, 2])
    train_ll, cv_ll = io.cross_validate_component_scan(traj, components, n_training_sets=1, n_attempts=1, verbose=False,dtype=dtype,device=device)
    assert train_ll.shape == (2, 1)
    assert cv_ll.shape == (2, 1)

def test_sgmm_fit_with_attempts_model():
    traj = np.random.randn(n_frames, n_atoms, 3)
    model = io.sgmm_fit_with_attempts(traj, n_components=n_components, n_attempts=1, verbose=False,dtype=dtype,device=device)
    assert model.is_fitted_
