### Suggested tests/test_core.py
import numpy as np
import torch
from shapeGMMTorch import ShapeGMM

dtype = torch.float64
device = torch.device('cpu')

def test_uniform_fit():
    traj = np.random.randn(50, 5, 3).astype(np.float32)
    model = ShapeGMM(n_components=2, covar_type='uniform', dtype=dtype, device=device, log_thresh=1e-1)
    model.fit(traj)
    assert model.is_fitted_ == True
    assert model.n_components == 2
    assert model.weights_.shape == (2,)
    assert model.means_.shape == (2, 5, 3)
    assert model.vars_.shape == (2,)

def test_kronecker_fit():
    traj = np.random.randn(50, 5, 3).astype(np.float32)
    model = ShapeGMM(n_components=2, covar_type='kronecker', dtype=dtype, device=device, log_thresh=1e-1)
    model.fit(traj)
    assert model.is_fitted_ == True
    assert model.n_components == 2
    assert model.means_.shape == (2, 5, 3)
    assert model.precisions_.shape == (2, 5, 5)
    assert model.lpdets_.shape == (2,)

def test_predict():
    traj = np.random.randn(50, 4, 3).astype(np.float32)
    model = ShapeGMM(n_components=3, covar_type='kronecker', dtype=dtype, device=device, log_thresh=1e-1)
    model.fit(traj)
    pred_ids = model.predict(traj)
    assert pred_ids.shape[0] == 50

def test_score():
    traj = np.random.randn(50, 4, 3).astype(np.float32)
    model = ShapeGMM(n_components=3, covar_type='kronecker', dtype=dtype, device=device, log_thresh=1e-1)
    model.fit(traj)
    model_score = model.score(traj)
    assert isinstance(log_lik, float)


