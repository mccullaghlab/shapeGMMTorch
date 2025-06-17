import pytest
import torch
from shapeGMMTorch.em import uniform

n_frames, n_atoms, n_components = 100, 4, 2

def test_sgmm_expectation_uniform_shape():
    traj = torch.randn(n_frames, n_atoms, 3)
    means = torch.randn(n_components, n_atoms, 3)
    vars_ = torch.ones(n_components)
    result = uniform.sgmm_expectation_uniform(traj, means, vars_)
    assert result.shape == (n_frames, n_components)

def test_sgmm_uniform_em_runs():
    traj = torch.randn(n_frames, n_atoms, 3)
    weights = torch.ones(n_frames) / n_frames
    means = torch.randn(n_components, n_atoms, 3)
    vars_ = torch.ones(n_components)
    ln_weights = torch.zeros(n_components)
    ll = uniform.sgmm_uniform_em(traj, weights, means, vars_, ln_weights)
    assert torch.is_tensor(ll)
