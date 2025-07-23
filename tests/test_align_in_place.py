import torch
import pytest
from shapeGMMTorch.align_in_place import (
    align_rot_mats,
    remove_center_of_geometry_in_place,
    trajectory_sd,
    align_uniform_in_place,
    align_kronecker_in_place,
    maximum_likelihood_uniform_alignment_in_place,
    maximum_likelihood_uniform_alignment_frame_weighted_in_place,
    maximum_likelihood_kronecker_alignment_in_place,
    maximum_likelihood_kronecker_alignment_frame_weighted_in_place
)

n_frames = 100
n_atoms = 4

@pytest.fixture
def sample_trajectory():
    torch.manual_seed(0)
    return torch.rand(n_frames, n_atoms, 3, dtype=torch.float32)

def test_remove_center_of_geometry(sample_trajectory):
    traj = sample_trajectory.clone()
    remove_center_of_geometry_in_place(traj)
    cog = traj.mean(dim=1)
    assert torch.allclose(cog, torch.zeros_like(cog), atol=1e-5)

def test_align_rot_mats(sample_trajectory):
    ref = sample_trajectory[0]
    rot_mats = align_rot_mats(sample_trajectory, ref)
    assert rot_mats.shape == (n_frames, 3, 3)
    identity = torch.bmm(rot_mats, rot_mats.transpose(1, 2))
    assert torch.allclose(identity, torch.eye(3).expand_as(identity), atol=1e-5)

def test_align_uniform_in_place(sample_trajectory):
    traj = sample_trajectory.clone()
    align_uniform_in_place(traj, sample_trajectory[0])
    assert traj.shape == sample_trajectory.shape

def test_align_kronecker_in_place(sample_trajectory):
    ref = sample_trajectory[0]
    precision = torch.eye(n_atoms).to(dtype=torch.float64)
    traj = sample_trajectory.clone()
    align_kronecker_in_place(traj, ref, precision)
    assert traj.shape == sample_trajectory.shape

def test_trajectory_sd(sample_trajectory):
    ref = sample_trajectory[0]
    sd = trajectory_sd(sample_trajectory.clone(), ref)
    assert sd.shape == (n_frames,)
    assert torch.all(sd >= 0)

def test_maximum_likelihood_uniform_alignment_in_place(sample_trajectory):
    traj = sample_trajectory.clone()
    avg, var = maximum_likelihood_uniform_alignment_in_place(traj)
    assert traj.shape == sample_trajectory.shape
    assert avg.shape == (n_atoms, 3)
    assert isinstance(var, torch.Tensor)

def test_maximum_likelihood_uniform_alignment_frame_weighted_in_place(sample_trajectory):
    traj = sample_trajectory.clone()
    weights = torch.ones(n_frames)
    avg, var = maximum_likelihood_uniform_alignment_frame_weighted_in_place(traj, weights)
    assert traj.shape == sample_trajectory.shape
    assert avg.shape == (n_atoms, 3)
    assert isinstance(var, torch.Tensor)

def test_maximum_likelihood_kronecker_alignment_in_place(sample_trajectory):
    traj = sample_trajectory.clone()
    avg, precision, lpdet = maximum_likelihood_kronecker_alignment_in_place(traj)
    assert traj.shape == sample_trajectory.shape
    assert avg.shape == (n_atoms, 3)
    assert precision.shape == (n_atoms, n_atoms)
    assert isinstance(lpdet, torch.Tensor)

def test_maximum_likelihood_kronecker_alignment_frame_weighted_in_place(sample_trajectory):
    traj = sample_trajectory.clone()
    weights = torch.ones(n_frames)
    avg, precision, lpdet = maximum_likelihood_kronecker_alignment_frame_weighted_in_place(traj, weights)
    assert traj.shape == sample_trajectory.shape
    assert avg.shape == (n_atoms, 3)
    assert precision.shape == (n_atoms, n_atoms)
    assert isinstance(lpdet, torch.Tensor)
