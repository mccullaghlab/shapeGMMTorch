### Suggested tests/test_core.py
import numpy as np
import torch
from shapeGMMTorch import ShapeGMM

device = torch.device('cpu')
dtype = torch.float64

def test_uniform_fit():
    covar_type = "uniform"
    traj = np.random.randn(50, 5, 3).astype(np.float32)
    model = ShapeGMM(n_components=2, covar_type=covar_type, dtype=dtype, device=device, log_thresh=1e-1)
    model.fit(traj)
    assert model.is_fitted_ == True
    assert model.n_components == 2
    assert model.weights_.shape == (2,)
    assert model.means_.shape == (2, 5, 3)
    assert model.vars_.shape == (2,)

def test_uniform_predict():
    covar_type = "uniform"
    dtype = torch.float64
    traj = np.random.randn(50, 4, 3).astype(np.float32)
    model = ShapeGMM(n_components=3, covar_type=covar_type, dtype=dtype, device=device, log_thresh=1e-1)
    model.fit(traj)
    pred_ids = model.predict(traj)
    assert pred_ids.shape[0] == 50

def test_uniform_score():
    covar_type = "uniform"
    traj = np.random.randn(50, 4, 3).astype(np.float32)
    model = ShapeGMM(n_components=3, covar_type=covar_type, dtype=dtype, device=device, log_thresh=1e-1)
    model.fit(traj)
    model_score = model.score(traj)
    assert isinstance(model_score, float)

def test_uniform_generate():
    covar_type = "uniform"
    traj = np.random.randn(50, 4, 3).astype(np.float32)
    model = ShapeGMM(n_components=3, covar_type=covar_type, dtype=dtype, device=device, log_thresh=1e-1)
    model.fit(traj)
    n_frames = 10
    gen_traj = model.generate(n_frames)
    assert gen_traj.shape == (n_frames, 4, 3)

def test_kronecker_fit():
    traj = np.random.randn(50, 5, 3).astype(np.float32)
    model = ShapeGMM(n_components=2, covar_type='kronecker', dtype=dtype, device=device, log_thresh=1e-1)
    model.fit(traj)
    assert model.is_fitted_ == True
    assert model.n_components == 2
    assert model.means_.shape == (2, 5, 3)
    assert model.precisions_.shape == (2, 5, 5)
    assert model.lpdets_.shape == (2,)

def test_kronecker_predict():
    covar_type = "kronecker"
    traj = np.random.randn(50, 4, 3).astype(np.float32)
    model = ShapeGMM(n_components=3, covar_type=covar_type, dtype=dtype, device=device, log_thresh=1e-1)
    model.fit(traj)
    pred_ids = model.predict(traj)
    assert pred_ids.shape[0] == 50

def test_kronecker_score():
    covar_type = "kronecker"
    traj = np.random.randn(50, 4, 3).astype(np.float32)
    model = ShapeGMM(n_components=3, covar_type=covar_type, dtype=dtype, device=device, log_thresh=1e-1)
    model.fit(traj)
    model_score = model.score(traj)
    assert isinstance(model_score, float)

def test_kronecker_generate():
    covar_type = "kronecker"
    traj = np.random.randn(50, 4, 3).astype(np.float32)
    model = ShapeGMM(n_components=3, covar_type=covar_type, dtype=dtype, device=device, log_thresh=1e-1)
    model.fit(traj)
    n_frames = 10
    gen_traj = model.generate(n_frames)
    assert gen_traj.shape == (n_frames, 4, 3)

