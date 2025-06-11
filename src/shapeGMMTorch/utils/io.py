import numpy as np
import time
import sys
import torch
from typing import Literal
import MDAnalysis as md
from scipy import stats
from ..core import ShapeGMM
from .. import align
from .. import generation


def cross_validate_component_scan(traj_data, component_array, train_fraction=0.9, frame_weights=None, thresh=1e-3, kabsch_thresh=1e-1, covar_type="kronecker", n_training_sets=3, n_attempts=10, dtype=torch.float32, device=torch.device("cuda:0"), verbose=True):
    """
    Perform cross-validation for ShapeGMM over a range of number of components.

    Parameters
    ----------
    traj_data : np.ndarray
        Trajectory data of shape (n_frames, n_atoms, 3).
    component_array : np.ndarray type int
        Array of components numbers to scan. e.g. [1, 2, 3, 4, 5]
    train_fraction : float
        Fraction of frames to use for training (0 < train_fraction < 1).
    frame_weights : list or np.ndarray, optional
        Frame weights of shape (n_frames,).
    thresh : float
        Log-likelihood convergence threshold.
    kabsch_thresh : float
        Convergence threshold for alignment.
    covar_type : str
        Covariance type ('uniform' or 'kronecker').
    n_training_sets : int
        Number of training/CV splits.
    n_attempts : int
        Number of ShapeGMM fitting attempts per component number.
    dtype : torch.dtype
        Data type for torch tensors.
    device : torch.device
        Device on which computation will run.
    verbose : bool
        Whether to print progress and timing information.

    Returns
    -------
    train_log_liks : np.ndarray
        Log-likelihoods on training sets.
    cv_log_liks : np.ndarray
        Log-likelihoods on validation sets.
    """
    n_frames = traj_data.shape[0]
    n_atoms = traj_data.shape[1]
    assert 0.0 < train_fraction < 1.0, "train_fraction must be between 0 and 1 (exclusive)."
    n_train_frames = int(np.floor(train_fraction * n_frames))
    train_log_liks = np.zeros((n_training_sets, len(component_array)))
    cv_log_liks = np.zeros((n_training_sets, len(component_array)))

    if verbose:
        print("Number of atoms:", n_atoms)
        print("Covariance type:", covar_type)
        print("Number of frames to train each model:", n_train_frames)
        print("Number of frames to predict each model:", n_frames - n_train_frames)
        print("Number of training sets:", n_training_sets)
        print("Number of attempts per set/component:", n_attempts)
        print("Component array:", component_array)
        print("%15s %15s %15s %19s %15s" % ("Training Set", "N Components", "Attempt", "Log Like per Frame", "Wallclock Time (s)"))
        print("%84s" % ("-" * 90))
        sys.stdout.flush()

    for i in range(n_training_sets):
        train_indices = np.random.choice(n_frames, size=n_train_frames, replace=False)
        cv_indices = np.setdiff1d(np.arange(n_frames), train_indices)

        traj_train = traj_data[train_indices]
        traj_cv = traj_data[cv_indices]
        if frame_weights is not None:
            weights_train = frame_weights[train_indices]
            weights_cv = frame_weights[cv_indices]
        else:
            weights_train = None
            weights_cv = None

        for j, k in enumerate(component_array):
            best_model = None
            best_log_lik = -np.inf
            current_attempts = 1 if k == 1 else n_attempts

            for attempt in range(current_attempts):
                start_time = time.time()
                model = ShapeGMM(
                    n_components=k,
                    log_thresh=thresh,
                    covar_type=covar_type,
                    dtype=dtype,
                    device=device,
                    kabsch_thresh=kabsch_thresh,
                    random_seed=1234 + attempt*11,
                    verbose=False
                )
                model.fit(traj_train, frame_weights=weights_train)
                log_likelihood = model.score(traj_train, weights_train)
                elapsed_time = time.time() - start_time

                if verbose:
                    print("%15d %15d %15d %19.4f %15.3f" % (i + 1, k, attempt + 1, log_likelihood, elapsed_time))
                    sys.stdout.flush()

                if log_likelihood > best_log_lik:
                    best_log_lik = log_likelihood
                    best_model = model

            train_log_liks[i, j] = best_log_lik
            cv_log_lik = best_model.score(traj_cv, weights_cv)
            cv_log_liks[i, j] = cv_log_lik

    return train_log_liks, cv_log_liks

def sgmm_fit_with_attempts(traj_data, n_components, n_attempts=10, covar_type="kronecker", frame_weights=None, thresh=1e-3, kabsch_thresh=1e-1, dtype=torch.float32, device=torch.device("cuda:0"), verbose=True):
    """
    Fit ShapeGMM model using multiple attempts and return the best model.

    Parameters
    ----------
    traj_data : np.ndarray
        Trajectory data of shape (n_frames, n_atoms, 3).
    n_components : int
        Number of components to fit.
    n_attempts : int
        Number of fitting attempts.
    covar_type : str
        Covariance type ('uniform' or 'kronecker').
        frame_weights : list or np.ndarray
        Optional frame weights.
    thresh : float
        Log-likelihood convergence threshold.
    kabsch_thresh : float
        Convergence threshold for alignment.
    dtype : torch.dtype
        Tensor data type.
    device : torch.device
        Device to use for computation.
    verbose : bool
        If True, prints per-attempt summary.

    Returns
    -------
    ShapeGMM
        Best fitted ShapeGMM model.
    """
    n_frames = traj_data.shape[0]
    n_atoms = traj_data.shape[1]

    if verbose:
        print(f"Number of components    : {n_components}")
        print(f"Number of attempts      : {n_attempts}")
        print(f"Covariance type         : {covar_type}")
        print(f"Data type (dtype)       : {dtype}")
        print(f"Device                  : {device}")
        print(f"Number of train frames  : {n_frames}")
        print(f"Number of atoms         : {n_atoms}")
        print("%8s %19s %15s" % ("Attempt", "Log Like per Frame", "Wallclock Time (s)"))
        print("%50s" % ("-" * 60))
        sys.stdout.flush()

    best_model = None
    best_log_lik = -np.inf

    for attempt in range(n_attempts):
        start_time = time.time()
        model = ShapeGMM(
            n_components=n_components,
            log_thresh=thresh,
            covar_type=covar_type,
            dtype=dtype,
            device=device,
            kabsch_thresh=kabsch_thresh,
            random_seed= 45612 + 133*attempt,
            verbose=False
        )
        model.fit(traj_data, frame_weights=frame_weights)
        log_likelihood = model.score(traj_data, frame_weights)
        elapsed_time = time.time() - start_time

        if verbose:
            print("%8d %19.4f %15.3f" % (attempt + 1, log_likelihood , elapsed_time))

        if log_likelihood > best_log_lik:
            best_log_lik = log_likelihood
            best_model = model

    return best_model

# write cluster trajectories
def generate_cluster_trajectories(sgmm, n_frames_per_cluster=100):
    """
    Write generated trajectories for each cluster in shapeGMM object sgmm
    """
    # create MDAnalysis universe
    u = md.Universe.empty(sgmm.n_atoms, 1, atom_resindex=np.zeros(sgmm.n_atoms), trajectory=True)
    u.trajectory.n_frames = n_frames_per_cluster
    sel_all = u.select_atoms("all")
    # loop through clusters
    for cluster_id in range(sgmm.n_clusters):

        trj = generation.gen_mv(sgmm.centers[cluster_id],sgmm.precisions[cluster_id],n_frames_per_cluster)
        pdb_file_name = "cluster" + str(cluster_id+1) + "_mean.pdb"
        dcd_file_name = "cluster" + str(cluster_id+1) + "_" + str(n_frames_per_cluster) + "frames.dcd"
        # write pdb of mean structure
        sel_all.positions = sgmm.centers[cluster_id]
        sel_all.write(pdb_file_name)
        # write dcd of generated trajectory
        with md.Writer(dcd_file_name, sel_all.n_atoms) as W:
            for ts in range(n_frames_per_cluster):
                sel_all.positions = trj[ts]
                W.write(sel_all)
        W.close()

# write aligned cluster trajectories
def write_aligned_cluster_trajectories(
    traj_data: np.ndarray,
    cluster_ids: np.ndarray,
    covar_type: Literal["kronecker", "uniform"] = "kronecker",
    dtype: torch.dtype = torch.float64,
    device: torch.device = torch.device("cpu")
) -> None:
    """
    Write aligned trajectories for each cluster in a trajectory dataset.
    
    Parameters
    ----------
    traj_data : np.ndarray
        Trajectory data of shape (n_frames, n_atoms, 3).
    cluster_ids : np.ndarray
        Cluster ID for each frame in the trajectory.
    covar_type : {"kronecker", "uniform"}, optional
        Type of covariance model to use for alignment. Default is "kronecker".
    dtype : torch.dtype, optional
        Data type for internal tensor operations. Default is torch.float64.
    device : torch.device, optional
        Device to perform computations on. Default is CPU.
    """
    n_frames, n_atoms, _ = traj_data.shape
    unique_ids, cluster_sizes = np.unique(cluster_ids, return_counts=True)
    n_clusters = len(unique_ids)

    align_fn = {
        "kronecker": align.torch_iterative_align_kronecker,
        "uniform": align.torch_iterative_align_uniform
    }.get(covar_type.lower())

    if align_fn is None:
        raise ValueError(f"Unsupported covariance model: '{covar_type}'. Choose 'kronecker' or 'uniform'.")

    for cluster_id, size in zip(unique_ids, cluster_sizes):
        # Extract and center frames in cluster
        trj_tensor = torch.tensor(traj_data[cluster_ids == cluster_id], dtype=dtype, device=device)
        trj_tensor = torch_remove_center_of_geometry(trj_tensor)

        # Align frames
        aligned_traj_tensor, *_ = align_fn(trj_tensor)
        aligned_traj = aligned_traj_tensor.cpu().numpy()

        # Create MDAnalysis universe
        u = md.Universe.empty(n_atoms, 1, atom_resindex=np.zeros(n_atoms), trajectory=True)
        u.trajectory.n_frames = size
        sel_all = u.select_atoms("all")

        # Generate file names
        cluster_str = f"cluster{cluster_id + 1}"
        pdb_file = f"{cluster_str}_frame1.pdb"
        dcd_file = f"{cluster_str}_{size}frames.dcd"

        # Write PDB (first frame)
        sel_all.positions = aligned_traj[0]
        sel_all.write(pdb_file)

        # Write DCD trajectory
        with md.Writer(dcd_file, sel_all.n_atoms) as writer:
            for ts in range(size):
                sel_all.positions = aligned_traj[ts]
                writer.write(sel_all)


# write representative frames
def write_representative_frames(sgmm, traj_data, cluster_ids):
    """
    Write representative frames for each cluster
    This is defined as the frame with the largest LL to each cluster
    """
    # get meta data from inputs
    n_frames = traj_data.shape[0]
    n_atoms = sgmm.n_atoms
    n_clusters = sgmm.n_clusters
    # loop through clusters
    for cluster_id in range(n_clusters):
        # create a shapeGMM object with just this cluster
        sgmmM = ShapeGMM(1,covar_type=sgmm.covar_type,device=torch.device("cpu"),dtype=sgmm.dtype)
        sgmmM.weights = np.array([1.0])
        sgmmM.centers = sgmm.centers[cluster_id].reshape(1,n_atoms,3)
        sgmmM.precisions = sgmm.precisions[cluster_id].reshape(1,n_atoms,n_atoms)
        sgmmM.lpdets = np.array([sgmm.lpdets[cluster_id]])
        sgmmM.n_atoms = sgmm.n_atoms
        sgmmM.is_fitted_ = True
        # compute LL using the predict function
        indeces = np.argwhere(cluster_ids==cluster_id).flatten()
        representive_frame_id = indeces[np.argmax(sgmmM.predict_proba(traj_data[indeces]))]
        # create MDAnalysis universe to print frame
        u = md.Universe.empty(n_atoms, 1, atom_resindex=np.zeros(n_atoms), trajectory=True)
        sel_all = u.select_atoms("all")
        # print pdb
        pdb_file_name = "cluster" + str(cluster_id+1) + "_reperesentative_frame_" + str(representive_frame_id+1) + ".pdb"
        sel_all.positions = traj_data[representive_frame_id]
        sel_all.write(pdb_file_name)
