import numpy as np
import time
import sys
import torch
import MDAnalysis as md
from scipy import stats
from . import torch_sgmm
from . import torch_align
from . import generate_points

def cross_validate_cluster_scan(traj_data, n_train_frames, frame_weights = [], covar_type="kronecker", cluster_array = np.arange(2,9,1).astype(int), n_training_sets=10, n_attempts = 5, dtype=torch.float32, device=torch.device("cuda:0")):
    """
    perform cross validation weighted shape-GMM for range of cluster sizes. Return train and CV log likelihoods as a function of number of clusters for each training set.
    Inputs:
        traj_data                    (required)  : float64 array with dimensions (n_frames, n_atoms,3) of molecular configurations
        n_train_frames               (required)  : int     scalar dictating number of frames to use as training (rest is used for CV)
        frame_weights               (default []) : float   array defining frame weights.  Default is empty/uniform
        covar_type         (default "kronecker") : string defining the covariance type.  Options are 'kronecker' and 'uniform'.  Defualt is 'uniform'.
        cluster_array         (default: [2..8])  : int     array of cluster sizes - can be of any number but must be ints. Default is [2, 3, 4, 5, 6, 7, 8]
        n_training_sets           (default: 10)  : int     scalar dictating how many training sets to choose. Default is 10
        n_attempts                 (default: 5)  : int     scalar dictating how many attempts to perform shape-GMM on same set.  Default is 5
        dtype          (default: torch.float32)  : torch data type for trajectory and centers
        device (default: torch.device("cuda:0")  : torch data type for trajectory and centers
    Returns:
        weighted_train_log_lik                 : float64 array with dimensions (n_clusters, n_training_sets) containing log likelihoods for each training set
        weighted_predict_log_lik               : float64 array with dimensions (n_clusters, n_training_sets) containing log likelihoods on each CV set
    """
    # meta data from input array
    n_frames = traj_data.shape[0]
    # set parameters
    n_predict_frames = n_frames - n_train_frames
    print("Number of frames to train each model:", n_train_frames)
    print("Number of frames to predict each model:", n_predict_frames)
    print("Number of training sets:", n_training_sets)
    print("Number of clusters:", cluster_array.size)
    print("Number of attempts per set/cluster:", n_attempts)
    sys.stdout.flush()
    # open data files
    weighted_train_log_lik = np.empty((cluster_array.size,n_training_sets),dtype=np.float64)
    weighted_predict_log_lik = np.empty((cluster_array.size,n_training_sets),dtype=np.float64)
    # print log info
    print("%15s %15s %15s %19s %15s" % ("Training Set", "N Clusters", "Attempt", "Log Like per Frame","CPU Time (s)"))
    print("%84s" % ("------------------------------------------------------------------------------------"))
    # loop over training sets
    index_array = np.arange(n_frames).astype(int)
    for training_set in range(n_training_sets):
        # shuffle trajectory data
        np.random.shuffle(index_array)
        # create training and predict data
        train_data = traj_data[index_array[:n_train_frames]]
        predict_data = traj_data[index_array[n_train_frames:]]
        if len(frame_weights) == 0:
            train_frame_weights = []
            predict_frame_weights = []
        else:
            train_frame_weights = frame_weights[index_array[:n_train_frames]]
            predict_frame_weights = frame_weights[index_array[n_train_frames:]]
        # loop over all number of clusters
        for cluster_index, cluster_size in enumerate(cluster_array):
            w_log_lik = []
            w_objs = []
            # for each n_clusters and training set, perform shape-GMM n_attempts times and take object with largest log likelihood
            # if cluster size is 1 there is no need to do multiple attempts
            if cluster_size == 1:
                current_attempts = 1
            else:
                current_attempts = n_attempts
            for attempt in range(current_attempts):
                start_time = time.process_time()
                wsgmm = torch_sgmm.ShapeGMMTorch(cluster_size, covar_type=covar_type, dtype=dtype, device=device)
                wsgmm.fit(train_data, frame_weights=train_frame_weights)
                w_log_lik.append(wsgmm.log_likelihood)
                w_objs.append(wsgmm)
                elapsed_time = time.process_time()-start_time
                print("%15d %15d %15d %19.3f %15.3f" % (training_set+1, cluster_size, attempt+1, np.round(wsgmm.log_likelihood,3), np.round(elapsed_time,3)))
                sys.stdout.flush()
            # determine maximum
            w_arg = np.nanargmax(w_log_lik)
            # save training log likes
            weighted_train_log_lik[cluster_index,training_set] = w_log_lik[w_arg]
            # save prediction log likes
            weighted_predict_log_lik[cluster_index,training_set] = w_objs[w_arg].predict(predict_data,predict_frame_weights)[2]

    #return
    return weighted_train_log_lik, weighted_predict_log_lik

def sgmm_fit_with_attempts(train_data, n_clusters, n_attempts, frame_weights = [], covar_type='kronecker', dtype=torch.float32, device=torch.device("cuda:0")):
    """
    Initialize and fit shapeGMM object with training data but do so a number of times and return the max log likelihood object

    Arguments:
        train_data (required)   - (n_train_frames,n_atoms,3) float array containing training data
        n_clusters (required)   - integer number of clusters must be input
        n_attempts (required)   - integer dictating the number of random intializations to attempt
        frame_weights (optional)- (n_train_frames) float array containing relative frame weights.  Defaults to empty (uniform weights)
        covar_type              - string defining the covariance type.  Options are 'kronecker' and 'uniform'.  Defualt is 'kronecker'.
        dtype                   - Data type to be used.  Default is torch.float32.
        device                  - device to be used.  Default is torch.device('cuda:0') device.

    Returns:
        shapeGMM object with max log likelhood from attempts
    """
    # meta data from input array
    n_frames = train_data.shape[0]
    # set parameters
    print("Number of training frames:", n_frames)
    print("Number of clusters:", n_clusters)
    print("Number of attempts:", n_attempts)
    sys.stdout.flush()
    objs = []
    log_likes = []
    # print log info
    print("%8s %19s %15s" % ("Attempt", "Log Like per Frame","CPU Time (s)"))
    print("%50s" % ("--------------------------------------------------"))
    #
    for i in range(n_attempts):
        start_time = time.process_time()
        wsgmm = torch_sgmm.ShapeGMMTorch(n_clusters,covar_type=covar_type, dtype=dtype, device=device)
        wsgmm.fit(train_data, frame_weights = frame_weights)
        elapsed_time = time.process_time()-start_time
        print("%8d %19.3f %15.3f" % (i+1, np.round(wsgmm.log_likelihood,3), np.round(elapsed_time,3)))
        objs.append(wsgmm)
        log_likes.append(wsgmm.log_likelihood)
    # return obj with max log likelihood per frame
    return objs[np.nanargmax(log_likes)]

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
        torch_align.torch_remove_center_of_geometry(trj_tensor,dtype=dtype,device=device)
        aligned_traj_tensor = torch_align.torch_iterative_align_kronecker(trj_tensor,dtype=dtype,device=device)[0]
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
        sgmmM = torch_sgmm.ShapeGMMTorch(1,covar_type=sgmm.covar_type,device=torch.device("cpu"),dtype=sgmm.dtype)
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
