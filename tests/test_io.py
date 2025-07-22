import numpy as np
import os
import pytest
import torch
from shapeGMMTorch.utils import io
from shapeGMMTorch import ShapeGMM
import tempfile
from pathlib import Path

dtype = torch.float64
device = torch.device('cpu')
n_frames = 100
n_atoms = 4
n_components = 2

def test_cross_validate_component_scan_shapes():
    traj = np.random.randn(n_frames, n_atoms, 3)
    components = np.array([1, 2])
    train_ll, cv_ll = io.cross_validate_component_scan(traj, components, n_training_sets=1, n_attempts=1, verbose=False,dtype=dtype,device=device)
    assert train_ll.shape == (2, 1)
    assert cv_ll.shape == (2, 1)

def test_sgmm_fit_with_attempts_model():
    traj = np.random.randn(n_frames, n_atoms, 3)
    model = io.sgmm_fit_with_attempts(traj, n_components=n_components, n_attempts=1, verbose=False,dtype=dtype,device=device)
    assert model.is_fitted_

def test_generate_component_uniform():
    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = os.getcwd()
        os.chdir(tmpdir)
        # Create a dummy model
        model = ShapeGMM(n_components=n_components, covar_type='uniform', random_seed=42, dtype=dtype, device=device)
        model.is_fitted_ = True
        model.n_atoms = n_atoms
        model.weights_ = np.ones(n_components)
        model.weights_ /= model.weights_.sum()
        model.means_ = np.random.rand(n_components, n_atoms, 3)
        model.vars_ = np.random.rand(n_components)

        n_frames_per_component=10
        io.generate_component_trajectories(model, n_frames_per_component=n_frames_per_component)
        for component_id in range(n_components):
            pdb_file_name = "component" + str(component_id+1) + "_mean.pdb"
            assert os.path.exists(pdb_file_name), f"{pdb_file_name} does not exist."
            dcd_file_name = "component" + str(component_id+1) + "_" + str(n_frames_per_component) + "frames.dcd"
            assert os.path.exists(dcd_file_name), f"{dcd_file_name} does not exist."
    os.chdir(cwd)  # Restore original working directory

def test_generate_component_kronecker():
    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = os.getcwd()
        os.chdir(tmpdir)

        # Create a dummy model
        model = ShapeGMM(n_components=n_components, covar_type='kronecker', random_seed=42, dtype=dtype, device=device)
        model.is_fitted_ = True
        model.n_atoms = n_atoms
        model.weights_ = np.ones(n_components)
        model.weights_ /= model.weights_.sum()
        model.means_ = np.random.rand(n_components, n_atoms, 3)
        model.precisions_ = np.empty((n_components, n_atoms, n_atoms))
        for component_id in range(n_components):
            model.precisions_[component_id] = np.random.rand()*np.eye(n_atoms,dtype=np.float64)

        n_frames_per_component=10
        io.generate_component_trajectories(model, n_frames_per_component=n_frames_per_component)
        for component_id in range(n_components):
            pdb_file_name = "component" + str(component_id+1) + "_mean.pdb"
            assert os.path.exists(pdb_file_name), f"{pdb_file_name} does not exist."
            dcd_file_name = "component" + str(component_id+1) + "_" + str(n_frames_per_component) + "frames.dcd"
            assert os.path.exists(dcd_file_name), f"{dcd_file_name} does not exist."

        os.chdir(cwd)  # restore original working directory

def test_write_aligned_component_trajectories_kronecker():
    n_components = 2
    traj_data = np.random.rand(n_frames, n_atoms, 3)
    n_frames_per_component = n_frames//2
    component_ids = np.concatenate([np.zeros(n_frames_per_component, dtype=int), np.ones(n_frames_per_component,dtype=int)])

    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = os.getcwd()
        os.chdir(tmpdir)
        io.write_aligned_component_trajectories(
            traj_data,
            component_ids,
            covar_type="kronecker",
            dtype=torch.float64,
            device=torch.device("cpu")
        )

        # Expected output files
        expected_files = []
        for i in range(1, n_components + 1):
            expected_files.append(f"component{i}_frame1.pdb")
            expected_files.append(f"component{i}_{n_frames_per_component}frames.dcd")
        # check they exist
        for filename in expected_files:
            assert os.path.exists(filename), f"Missing file: {filename}"
        os.chdir(cwd)  # restore original working directory


def test_write_aligned_component_trajectories_uniform():
    n_components = 2
    traj_data = np.random.rand(n_frames, n_atoms, 3)
    n_frames_per_component = n_frames//2
    component_ids = np.concatenate([np.zeros(n_frames_per_component, dtype=int), np.ones(n_frames_per_component,dtype=int)])

    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = os.getcwd()
        os.chdir(tmpdir)
        io.write_aligned_component_trajectories(
            traj_data,
            component_ids,
            covar_type="uniform",
            dtype=torch.float64,
            device=torch.device("cpu")
        )

        # Expected output files
        expected_files = []
        for i in range(1, n_components + 1):
            expected_files.append(f"component{i}_frame1.pdb")
            expected_files.append(f"component{i}_{n_frames_per_component}frames.dcd")
        # check they exist
        for filename in expected_files:
            assert os.path.exists(filename), f"Missing file: {filename}"
        os.chdir(cwd)  # restore original working directory

def test_write_representative_frame_kronecker():
    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = os.getcwd()
        os.chdir(tmpdir)

        # Create a dummy data
        traj_data = np.random.randn(n_frames, n_atoms, 3)
        # fit model
        model = ShapeGMM(n_components=n_components, covar_type='kronecker', random_seed=42, dtype=dtype, device=device)
        model.fit(traj_data)
        component_ids = model.predict(traj_data)

        io.write_representative_frames(model, traj_data, component_ids)
        for component_id in range(n_components):
            pdb_file_name = "component" + str(component_id+1) + "_reperesentative_frame_*.pdb"
            pdb_files = list(Path(".").glob(pdb_file_name))
            assert pdb_files, f"{pdb_file_name} does not exist."

        os.chdir(cwd)  # restore original working directory


def test_write_representative_frame_uniform():
    with tempfile.TemporaryDirectory() as tmpdir:
        cwd = os.getcwd()
        os.chdir(tmpdir)

        # Create a dummy data
        traj_data = np.random.randn(n_frames, n_atoms, 3)
        # fit model
        model = ShapeGMM(n_components=n_components, covar_type='uniform', random_seed=42, dtype=dtype, device=device)
        model.fit(traj_data)
        component_ids = model.predict(traj_data)

        io.write_representative_frames(model, traj_data, component_ids)
        for component_id in range(n_components):
            pdb_file_name = "component" + str(component_id+1) + "_reperesentative_frame_*.pdb"
            pdb_files = list(Path(".").glob(pdb_file_name))
            assert pdb_files, f"{pdb_file_name} does not exist."

        os.chdir(cwd)  # restore original working directory
