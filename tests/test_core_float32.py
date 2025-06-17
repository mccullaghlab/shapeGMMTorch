### Suggested tests/test_core.py
import numpy as np
import torch
from shapeGMMTorch import ShapeGMM

device = torch.device('cpu')
dtype = torch.float32
n_frames = 100
n_atoms = 4
n_components = 3

def test_uniform_fit():
    covar_type = "uniform"
    traj = np.random.randn(n_frames, n_atoms, 3).astype(np.float32)
    model = ShapeGMM(n_components=n_components, covar_type=covar_type, dtype=dtype, device=device, log_thresh=1e-1)
    model.fit(traj)
    assert model.is_fitted_ == True
    assert model.n_components == n_components
    assert model.weights_.shape == (n_components,)
    assert model.means_.shape == (n_components, n_atoms, 3)
    assert model.vars_.shape == (n_components,)

def test_uniform_predict():
    covar_type = "uniform"
    dtype = torch.float64
    traj = np.random.randn(n_frames, n_atoms, 3).astype(np.float32)
    model = ShapeGMM(n_components=n_components, covar_type=covar_type, dtype=dtype, device=device, log_thresh=1e-1)
    model.fit(traj)
    pred_ids = model.predict(traj)
    assert pred_ids.shape[0] == n_frames

def test_uniform_score():
    covar_type = "uniform"
    traj = np.random.randn(n_frames, n_atoms, 3).astype(np.float32)
    model = ShapeGMM(n_components=n_components, covar_type=covar_type, dtype=dtype, device=device, log_thresh=1e-1)
    model.fit(traj)
    model_score = model.score(traj)
    assert isinstance(model_score, float)

def test_uniform_generate():
    covar_type = "uniform"
    traj = np.random.randn(n_frames, n_atoms, 3).astype(np.float32)
    model = ShapeGMM(n_components=n_components, covar_type=covar_type, dtype=dtype, device=device, log_thresh=1e-1)
    model.fit(traj)
    n_points = 10
    gen_traj = model.generate(n_points)
    assert gen_traj.shape == (n_points, n_atoms, 3)

def test_kronecker_fit():
    covar_type = "kronecker"
    traj = np.random.randn(n_frames, n_atoms, 3).astype(np.float32)
    model = ShapeGMM(n_components=n_components, covar_type=covar_type, dtype=dtype, device=device, log_thresh=1e-1)
    model.fit(traj)
    assert model.is_fitted_ == True
    assert model.n_components == n_components
    assert model.means_.shape == (n_components, n_atoms, 3)
    assert model.precisions_.shape == (n_components, n_atoms, n_atoms)
    assert model.lpdets_.shape == (n_components,)

def test_kronecker_predict():
    covar_type = "kronecker"
    traj = np.random.randn(n_frames, n_atoms, 3).astype(np.float32)
    model = ShapeGMM(n_components=n_components, covar_type=covar_type, dtype=dtype, device=device, log_thresh=1e-1)
    model.fit(traj)
    pred_ids = model.predict(traj)
    assert pred_ids.shape[0] == n_frames

def test_kronecker_score():
    covar_type = "kronecker"
    traj = np.random.randn(n_frames, n_atoms, 3).astype(np.float32)
    model = ShapeGMM(n_components=n_components, covar_type=covar_type, dtype=dtype, device=device, log_thresh=1e-1)
    model.fit(traj)
    model_score = model.score(traj)
    assert isinstance(model_score, float)

def test_kronecker_generate():
    covar_type = "kronecker"
    traj = np.random.randn(n_frames, n_atoms, 3).astype(np.float32)
    model = ShapeGMM(n_components=n_components, covar_type=covar_type, dtype=dtype, device=device, log_thresh=1e-1)
    model.fit(traj)
    n_points = 10
    gen_traj = model.generate(n_points)
    assert gen_traj.shape == (n_points, n_atoms, 3)

