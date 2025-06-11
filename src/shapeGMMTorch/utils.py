import numpy as np
import time
import sys
import torch
import MDAnalysis as md
from scipy import stats
from core import ShapeGMM
import align
import generate_points

def cross_validate_component_scan(traj_data, train_fraction=0.9, frame_weights=[], thresh=1e-3, kabsch_thresh=1e-1, covar_type="kronecker", component_array=np.arange(2, 9, 1).astype(int), n_training_sets=10, n_attempts=5, dtype=torch.float64, device=torch.device("cpu"), verbose=True):
    """
    Perform cross-validation for ShapeGMM over a range of number of components.

    Parameters
    ----------
    traj_data : np.ndarray
        Trajectory data of shape (n_frames, n_atoms, 3).
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
    component_array : np.ndarray
        Array of components numbers to scan.
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
        print("%15s %15s %15s %19s %15s" % ("Training Set", "N Components", "Attempt", "Log Like per Frame", "CPU Time (s)"))
        print("%84s" % ("-" * 84))
        sys.stdout.flush()

    for i in range(n_training_sets):
        train_indices = np.random.choice(n_frames, size=n_train_frames, replace=False)
        cv_indices = np.setdiff1d(np.arange(n_frames), train_indices)

        traj_train = traj_data[train_indices]
        traj_cv = traj_data[cv_indices]
        if len(frame_weights) > 0:
            weights_train = frame_weights[train_indices]
            weights_cv = frame_weights[cv_indices]
        else:
            weights_train = []
            weights_cv = []

        for j, k in enumerate(component_array):
            best_model = None
            best_log_lik = -np.inf
            current_attempts = 1 if k == 1 else n_attempts

            for attempt in range(current_attempts):
                start_time = time.process_time()
                model = ShapeGMM(
                    n_components=k,
                    log_thresh=thresh,
                    covar_type=covar_type,
                    dtype=dtype,
                    device=device,
                    kabsch_thresh=kabsch_thresh,
                    random_seed=np.random.randint(0, 10000),
                    verbose=False
                )
                model.fit(traj_train, frame_weights=weights_train)
                _, log_likelihood = model.predict(traj_train, weights_train)
                elapsed_time = time.process_time() - start_time

                if verbose:
                    print("%15d %15d %15d %19.4f %15.3f" % (i + 1, k, attempt + 1, log_likelihood / n_train_frames, elapsed_time))
                    sys.stdout.flush()

                if log_likelihood > best_log_lik:
                    best_log_lik = log_likelihood
                    best_model = model

            train_log_liks[i, j] = best_log_lik
            _, cv_log_lik = best_model.predict(traj_cv, weights_cv)
            cv_log_liks[i, j] = cv_log_lik

    return train_log_liks, cv_log_liks

def sgmm_fit_with_attempts(traj_data, n_components=4, thresh=1e-3, kabsch_thresh=1e-1, covar_type="kronecker", frame_weights=[], n_attempts=5, dtype=torch.float32, device=torch.device("cuda:0"), verbose=True):
    """
    Fit ShapeGMM model using multiple attempts and return the best model.

    Parameters
    ----------
    traj_data : np.ndarray
        Trajectory data of shape (n_frames, n_atoms, 3).
    n_components : int
        Number of components to fit.
    thresh : float
        Log-likelihood convergence threshold.
    kabsch_thresh : float
        Convergence threshold for alignment.
    covar_type : str
        Covariance type ('uniform' or 'kronecker').
    frame_weights : list or np.ndarray
        Optional frame weights.
    n_attempts : int
        Number of fitting attempts.
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

    if verbose:
        print("Number of training frames:", n_frames)
        print("Number of components:", n_components)
        print("Number of attempts:", n_attempts)
        print("%8s %19s %15s" % ("Attempt", "Log Like per Frame", "CPU Time (s)"))
        print("%50s" % ("-" * 50))
        sys.stdout.flush()

    best_model = None
    best_log_lik = -np.inf

    for attempt in range(n_attempts):
        start_time = time.process_time()
        model = ShapeGMM(
            n_components=n_components,
            log_thresh=thresh,
            covar_type=covar_type,
            dtype=dtype,
            device=device,
            kabsch_thresh=kabsch_thresh,
            random_seed=np.random.randint(0, 10000),
            verbose=False
        )
        model.fit(traj_data, frame_weights=frame_weights)
        _, log_likelihood = model.predict(traj_data, frame_weights)
        elapsed_time = time.process_time() - start_time

        if verbose:
            print("%8d %19.4f %15.3f" % (attempt + 1, log_likelihood / n_frames, elapsed_time))

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

        trj = generate_points.gen_mv(sgmm.centers[cluster_id],sgmm.precisions[cluster_id],n_frames_per_cluster)
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
def write_aligned_cluster_trajectories(traj_data, cluster_ids, covar_type='kronecker',dtype=torch.float64,device=torch.device("cpu")):
    """
    Write trajectories for each cluster from trajectory data and cluster ids
    """
    # get meta data from inputs
    n_frames = traj_data.shape[0]
    n_atoms = traj_data.shape[1]
    n_cluster_frames = np.unique(cluster_ids,return_counts=True)[1]
    n_clusters = n_cluster_frames.size
    # loop through clusters
    for cluster_id in range(n_clusters):
        # create MDAnalysis universe
        u = md.Universe.empty(n_atoms, 1, atom_resindex=np.zeros(n_atoms), trajectory=True)
        u.trajectory.n_frames = n_cluster_frames[cluster_id]
        sel_all = u.select_atoms("all")
        # align cluster trajectory
        trj_tensor = torch.tensor(traj_data[cluster_ids==cluster_id],dtype=dtype,device=device)
        trj_tensor = align.torch_remove_center_of_geometry(trj_tensor)
        aligned_traj_tensor = align.torch_iterative_align_kronecker(trj_tensor)[0]
        trj = aligned_traj_tensor.cpu().numpy()
        # create file names
        pdb_file_name = "cluster" + str(cluster_id+1) + "_frame1.pdb"
        dcd_file_name = "cluster" + str(cluster_id+1) + "_" + str(n_cluster_frames[cluster_id]) + "frames.dcd"
        # write pdb of mean structure
        sel_all.positions = trj[0]
        sel_all.write(pdb_file_name)
        # write dcd of generated trajectory
        with md.Writer(dcd_file_name, sel_all.n_atoms) as W:
            for ts in range(n_cluster_frames[cluster_id]):
                sel_all.positions = trj[ts]
                W.write(sel_all)
        W.close()

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
        sgmmM._gmm_fit_flag = True
        # compute LL using the predict function
        indeces = np.argwhere(cluster_ids==cluster_id).flatten()
        sgmmM.predict(traj_data[indeces])
        representive_frame_id = indeces[np.argmax(sgmmM.predict_frame_log_likelihood)]
        # create MDAnalysis universe to print frame
        u = md.Universe.empty(n_atoms, 1, atom_resindex=np.zeros(n_atoms), trajectory=True)
        sel_all = u.select_atoms("all")
        # print pdb
        pdb_file_name = "cluster" + str(cluster_id+1) + "_reperesentative_frame_" + str(representive_frame_id+1) + ".pdb"
        sel_all.positions = traj_data[representive_frame_id]
        sel_all.write(pdb_file_name)
