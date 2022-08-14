import numpy as np
import time
import sys
import torch
from . import torch_sgmm

def cross_validate_cluster_scan(traj_data, n_train_frames, covar_type="kronecker", cluster_array = np.arange(2,9,1).astype(int), n_training_sets=10, n_attempts = 5, dtype=torch.float32, device=torch.device("cuda:0")):
    """
    perform cross validation weighted shape-GMM for range of cluster sizes
    Inputs:
        traj_data                    (required)  : float64 array with dimensions (n_frames, n_atoms,3) of molecular configurations
        n_train_frames               (required)  : int     scalar dictating number of frames to use as training (rest is used for CV)
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
    for training_set in range(n_training_sets):
        # shuffle trajectory data
        np.random.shuffle(traj_data)
        # create training and predict data
        train_data = traj_data[:n_train_frames]
        predict_data = traj_data[n_train_frames:]
        # loop over all number of clusters
        for cluster_index, cluster_size in enumerate(cluster_array):
            w_log_lik = []
            w_objs = []
            # for each n_clusters and training set, perform shape-GMM n_attempts times and take object with largest log likelihood
            for attempt in range(n_attempts):
                start_time = time.process_time()
                wsgmm = torch_sgmm.ShapeGMMTorch(cluster_size, covar_type=covar_type, dtype=dtype, device=device)
                wsgmm.fit(train_data)
                w_log_lik.append(wsgmm.log_likelihood)
                w_objs.append(wsgmm)
                elapsed_time = time.process_time()-start_time
                print("%15d %15d %15d %19.3f %15.3f" % (training_set+1, cluster_size, attempt+1, np.round(wsgmm.log_likelihood,3), np.round(elapsed_time,3)))
            # determine maximum
            w_arg = np.argmax(w_log_lik)
            # save training log likes
            weighted_train_log_lik[cluster_index,training_set] = w_log_lik[w_arg]
            # save prediction log likes
            weighted_predict_log_lik[cluster_index,training_set] = w_objs[w_arg].predict(predict_data)[2]

    #return
    return weighted_train_log_lik, weighted_predict_log_lik


