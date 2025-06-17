
import pytest
import torch
from shapeGMMTorch.em import kronecker as kr

dtype = torch.float64
device = torch.device('cpu')
n_frames, n_atoms, n_components = 100, 4, 2

def test_sgmm_kronecker_em_runs():
    torch.manual_seed(0)
    traj = torch.randn((n_frames, n_atoms, 3), dtype=torch.float32)
    weights = torch.ones(n_frames, dtype=torch.float32) / n_frames
    means = torch.randn((n_components, n_atoms, 3), dtype=torch.float32)
    precisions = torch.stack([torch.eye(n_atoms,dtype=torch.float64) for _ in range(n_components)])
    lpdets = torch.zeros(n_components, dtype=torch.float64)
    ln_weights = torch.zeros(n_components, dtype=torch.float64)

    log_lik = kr.sgmm_kronecker_em(
        traj, weights, means, precisions, lpdets, ln_weights,
        thresh=1e-3, max_iter=5
    )

    assert log_lik.shape == ()
    assert isinstance(log_lik.item(), float)

def test_sgmm_expectation_kronecker_shapes():
    torch.manual_seed(0)
    traj = torch.randn((n_frames, n_atoms, 3), dtype=torch.float32)
    means = torch.randn((n_components, n_atoms, 3), dtype=torch.float32)
    precisions = torch.stack([torch.eye(n_atoms,dtype=torch.float64) for _ in range(n_components)])
    lpdets = torch.zeros(n_components, dtype=torch.float32)

    ln_liks = kr.sgmm_expectation_kronecker(traj, means, precisions, lpdets)

    assert ln_liks.shape == (n_frames, n_components)
