import numpy as np
import time
import sys
import torch
#import torch_sgmm
from . import torch_sgmm

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
            w_arg = np.argmax(w_log_lik)
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
    return objs[np.argmax(log_likes)]

