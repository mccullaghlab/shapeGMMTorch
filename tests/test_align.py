import torch
import pytest
from shapeGMMTorch.align import (
    align_rot_mats,
    remove_center_of_geometry,
    trajectory_sd,
    align_uniform,
    align_kronecker,
    maximum_likelihood_uniform_alignment,
    maximum_likelihood_uniform_alignment_frame_weighted,
    maximum_likelihood_kronecker_alignment,
    maximum_likelihood_kronecker_alignment_frame_weighted
)

n_frames = 100
n_atoms = 4

@pytest.fixture
def sample_trajectory():
    torch.manual_seed(0)
    return torch.rand(n_frames, n_atoms, 3, dtype=torch.float32)

def test_remove_center_of_geometry(sample_trajectory):
    centered = remove_center_of_geometry(sample_trajectory.clone())
    cog = centered.mean(dim=1)
    assert torch.allclose(cog, torch.zeros_like(cog), atol=1e-5)

def test_align_rot_mats(sample_trajectory):
    ref = sample_trajectory[0]
    rot_mats = align_rot_mats(sample_trajectory, ref)
    assert rot_mats.shape == (n_frames, 3, 3)
    identity = torch.bmm(rot_mats, rot_mats.transpose(1, 2))
    assert torch.allclose(identity, torch.eye(3).expand_as(identity), atol=1e-5)

def test_align_uniform(sample_trajectory):
    aligned = align_uniform(sample_trajectory.clone(), sample_trajectory[0])
    assert aligned.shape == sample_trajectory.shape

def test_align_kronecker(sample_trajectory):
    ref = sample_trajectory[0]
    precision = torch.eye(n_atoms).to(dtype=torch.float64)
    aligned = align_kronecker(sample_trajectory.clone(), ref, precision)
    assert aligned.shape == sample_trajectory.shape

def test_trajectory_sd(sample_trajectory):
    ref = sample_trajectory[0]
    sd = trajectory_sd(sample_trajectory.clone(), ref)
    assert sd.shape == (n_frames,)
    assert torch.all(sd >= 0)

def test_maximum_likelihood_uniform_alignment(sample_trajectory):
    traj, avg, var = maximum_likelihood_uniform_alignment(sample_trajectory.clone())
    assert traj.shape == sample_trajectory.shape
    assert avg.shape == (n_atoms, 3)
    assert isinstance(var, torch.Tensor)

def test_maximum_likelihood_uniform_alignment_frame_weighted(sample_trajectory):
    weights = torch.ones(n_frames)
    traj, avg, var = maximum_likelihood_uniform_alignment_frame_weighted(sample_trajectory.clone(), weights)
    assert traj.shape == sample_trajectory.shape
    assert avg.shape == (n_atoms, 3)
    assert isinstance(var, torch.Tensor)

def test_maximum_likelihood_kronecker_alignment(sample_trajectory):
    traj, avg, precision, lpdet = maximum_likelihood_kronecker_alignment(sample_trajectory.clone())
    assert traj.shape == sample_trajectory.shape
    assert avg.shape == (n_atoms, 3)
    assert precision.shape == (n_atoms, n_atoms)
    assert isinstance(lpdet, torch.Tensor)

def test_maximum_likelihood_kronecker_alignment_frame_weighted(sample_trajectory):
    weights = torch.ones(n_frames)
    traj, avg, precision, lpdet = maximum_likelihood_kronecker_alignment_frame_weighted(sample_trajectory.clone(), weights)
    assert traj.shape == sample_trajectory.shape
    assert avg.shape == (n_atoms, 3)
    assert precision.shape == (n_atoms, n_atoms)
    assert isinstance(lpdet, torch.Tensor)
