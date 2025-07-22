import numpy as np
import torch
from shapeGMMTorch import ShapeGMM
from shapeGMMTorch.utils.similarity import maha_dist2, kl_divergence, js_divergence, configurational_entropy, bhattacharyya_distance

device = torch.device("cpu")
dtype = torch.float64
n_frames = 100
n_atoms = 4
n_components = 2

def test_maha_dist2():
    x1 = np.random.rand(10, 3)
    x2 = np.copy(x1)
    weights = np.eye(10)
    dist = maha_dist2(x1, x2, weights)
    assert np.isclose(dist, 0.0), f"Expected 0.0, got {dist}"

def test_kl_divergence_kronecker():
    covar_type = "kronecker"
    traj = np.random.randn(n_frames, n_atoms, 3).astype(np.float32)
    model1 = ShapeGMM(n_components=n_components, covar_type=covar_type, dtype=dtype, device=device, log_thresh=1e-1)
    model1.fit(traj)
    n_points = 100
    kl, kl_error = kl_divergence(model1, model1, n_points)
    assert isinstance(kl, float)
    assert isinstance(kl_error, float)

def test_js_divergence_kronecker():
    covar_type = "kronecker"
    traj = np.random.randn(n_frames, n_atoms, 3).astype(np.float32)
    model1 = ShapeGMM(n_components=n_components, covar_type=covar_type, dtype=dtype, device=device, log_thresh=1e-1)
    model1.fit(traj)
    n_points = 100
    js, js_error = js_divergence(model1, model1, n_points)
    assert isinstance(js, float)
    assert isinstance(js_error, float)

def test_configurational_entropy_kronecker():
    covar_type = "kronecker"
    traj = np.random.randn(n_frames, n_atoms, 3).astype(np.float32)
    model1 = ShapeGMM(n_components=n_components, covar_type=covar_type, dtype=dtype, device=device, log_thresh=1e-1)
    model1.fit(traj)
    n_points = 100
    S, S_error = configurational_entropy(model1, n_points)
    assert isinstance(S, float)
    assert isinstance(S_error, float)

def test_bhattacharyya_distance_kronecker():
    covar_type = "kronecker"
    traj = np.random.randn(n_frames, n_atoms, 3).astype(np.float32)
    model1 = ShapeGMM(n_components=n_components, covar_type=covar_type, dtype=dtype, device=device, log_thresh=1e-1)
    model1.fit(traj)
    dist = bhattacharyya_distance(model1, 0, model1, 1)
    assert isinstance(dist, float)

def test_kl_divergence_uniform():
    covar_type = "uniform"
    traj = np.random.randn(n_frames, n_atoms, 3).astype(np.float32)
    model1 = ShapeGMM(n_components=n_components, covar_type=covar_type, dtype=dtype, device=device, log_thresh=1e-1)
    model1.fit(traj)
    n_points = 100
    kl, kl_error = kl_divergence(model1, model1, n_points)
    assert isinstance(kl, float)
    assert isinstance(kl_error, float)

def test_js_divergence_uniform():
    covar_type = "uniform"
    traj = np.random.randn(n_frames, n_atoms, 3).astype(np.float32)
    model1 = ShapeGMM(n_components=n_components, covar_type=covar_type, dtype=dtype, device=device, log_thresh=1e-1)
    model1.fit(traj)
    n_points = 100
    js, js_error = js_divergence(model1, model1, n_points)
    assert isinstance(js, float)
    assert isinstance(js_error, float)

def test_configurational_entropy_uniform():
    covar_type = "uniform"
    traj = np.random.randn(n_frames, n_atoms, 3).astype(np.float32)
    model1 = ShapeGMM(n_components=n_components, covar_type=covar_type, dtype=dtype, device=device, log_thresh=1e-1)
    model1.fit(traj)
    n_points = 100
    S, S_error = configurational_entropy(model1, n_points)
    assert isinstance(S, float)
    assert isinstance(S_error, float)

def test_bhattacharyya_distance_uniform():
    covar_type = "uniform"
    traj = np.random.randn(n_frames, n_atoms, 3).astype(np.float32)
    model1 = ShapeGMM(n_components=n_components, covar_type=covar_type, dtype=dtype, device=device, log_thresh=1e-1)
    model1.fit(traj)
    dist = bhattacharyya_distance(model1, 0, model1, 1)
    assert isinstance(dist, float)
