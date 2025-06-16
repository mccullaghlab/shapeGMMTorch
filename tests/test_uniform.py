import pytest
import torch
from shapeGMMTorch.em import uniform

def test_sgmm_expectation_uniform_shape():
    traj = torch.randn(10, 5, 3)
    means = torch.randn(3, 5, 3)
    vars_ = torch.ones(3)
    result = uniform.sgmm_expectation_uniform(traj, means, vars_)
    assert result.shape == (10, 3)

def test_sgmm_uniform_em_runs():
    traj = torch.randn(10, 5, 3)
    weights = torch.ones(10) / 10
    means = torch.randn(3, 5, 3)
    vars_ = torch.ones(3)
    ln_weights = torch.zeros(3)
    ll = uniform.sgmm_uniform_em(traj, weights, means, vars_, ln_weights)
    assert torch.is_tensor(ll)
