### Suggested tests/test_core.py
import numpy as np
import torch
from shapeGMMTorch import ShapeGMM

device = torch.device('cpu')
dtype = torch.float64
n_frames = 100
n_atoms = 4
n_components = 2

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
    traj = np.random.randn(n_frames, n_atoms, 3).astype(np.float32)
    model = ShapeGMM(n_components=n_components, covar_type='kronecker', dtype=dtype, device=device, log_thresh=1e-1)
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

def test_save_and_load(tmp_path):
    # Create a dummy model
    model = ShapeGMM(n_components=2, covar_type='kronecker', random_seed=42, dtype=dtype, device=device)
    model.is_fitted_ = True
    model.n_atoms = 5
    model.weights_ = np.array([0.5, 0.5])
    model.means_ = np.random.rand(2, 5, 3)
    model.precisions_ = np.random.rand(2, 5, 5)
    model.lpdets_ = np.array([1.0, 1.0])

    # Save and load
    filename = tmp_path / "test_model.pt"
    model.save(filename)
    loaded_model = ShapeGMM.load(filename)

    # Check equivalence
    assert loaded_model.n_components == model.n_components
    assert loaded_model.is_fitted_ is True
    assert np.allclose(loaded_model.means_, model.means_)
    assert np.allclose(loaded_model.precisions_, model.precisions_)
    assert np.allclose(loaded_model.weights_, model.weights_)

