import numpy as np
import warnings
warnings.filterwarnings('ignore')
import random
from . import torch_align
#import torch_align
import torch

GAMMA_THRESH = 1e-15

##

def uniform_sgmm_log_likelihood(traj_tensor, clusters, thresh=1e-3, dtype=torch.float32, device=torch.device("cuda:0")):
    # meta data from inputs
    n_frames = traj_tensor.shape[0]
    n_clusters = np.amax(clusters) + 1
    n_atoms = traj_tensor.shape[1]
    n_dim = traj_tensor.shape[2]
    n_features = n_atoms*n_dim
    # declare arrays 
    cluster_frame_ln_likelihoods = torch.empty((n_frames,n_clusters),dtype=torch.float64, device=device)
    # compute likelihood of each frame at each Gaussian
    for k in range(n_clusters):
        indeces = np.argwhere(clusters == k).flatten()
        # initialize weights as populations of clusters
        ln_weight = torch.tensor(np.log(indeces.size/n_frames),dtype=torch.float64,device=device)
        # determine center and variance of cluster using iterative alignment
        center, var = torch_align.torch_iterative_align_uniform(traj_tensor[indeces], thresh=thresh, dtype=dtype, device=device)[1:]
        # compute log likelihood
        cluster_frame_ln_likelihoods[:,k] = torch_align.torch_sd(traj_tensor,center)
        # divide be variance and normalize
        cluster_frame_ln_likelihoods[:,k] *= -0.5/var
        cluster_frame_ln_likelihoods[:,k] -= 1.5*(n_atoms-1)*torch.log(var)
        cluster_frame_ln_likelihoods[:,k] += ln_weight
    # compute log likelihood
    log_likelihood = torch.logsumexp(cluster_frame_ln_likelihoods,1)
    log_likelihood = torch.mean(log_likelihood)
    del cluster_frame_ln_likelihoods
    del ln_weight
    del center
    del var
    torch.cuda.empty_cache()
    return log_likelihood

def init_random(traj_tensor, n_clusters, dtype=torch.float32, device=torch.device("cuda:0")):
    # meta data from inputs
    n_frames = traj_tensor.shape[0]
    n_atoms = traj_tensor.shape[1]
    # declare arrayes
    dists = torch.empty((n_frames,n_clusters), dtype=dtype, device=device)
    clustersPass = False
    while clustersPass == False:
        clustersPass = True
        randFrames = random.sample(range(n_frames),n_clusters)
        centers = torch.clone(traj_tensor[randFrames])
        # make initial clustering based on SD distance from centers
        # measure distance to every center
        for k in range(n_clusters):
            dists[:,k] = torch_align.torch_sd(traj_tensor, centers[k])
        # assign frame to nearest center
        clusters = torch.argmin(dists, dim = 1).cpu().numpy()
        # make sure there are at least n_atoms frames in each cluster for (co)variance determination
        for k in range(n_clusters):
            indeces = np.argwhere(clusters == k).flatten()
            if indeces.size < n_atoms:
                clustersPass = False
                break
    return clusters


## Expectation Maximization for GMM with uniform covariance model
def torch_sgmm_uniform_em(traj_tensor, centers_tensor, vars_tensor, ln_weights_tensor, dtype=torch.float32, device=torch.device("cuda:0"), thresh=1e-3):
    
    # get metadata from trajectory data
    n_frames = traj_tensor.shape[0]
    n_atoms = traj_tensor.shape[1]
    n_dim = traj_tensor.shape[2]
    n_clusters = ln_weights_tensor.shape[0]
    gamma_thresh_tensor = torch.tensor(GAMMA_THRESH,dtype=torch.float64,device=device)

    # Expectation step:
    cluster_frame_ln_likelihoods_tensor = torch_sgmm_expectation_uniform(traj_tensor, centers_tensor, vars_tensor, device=device)
    
    # compute log likelihood and gamma normalization
    for k in range(n_clusters):
        cluster_frame_ln_likelihoods_tensor[:,k] += ln_weights_tensor[k]
    log_norm = torch.logsumexp(cluster_frame_ln_likelihoods_tensor,1)
    log_likelihood = torch.mean(log_norm)
    # determine gamma values
    # use the current values for the parameters to evaluate the posterior
    # probabilities of the data to have been generanted by each gaussian
    gamma_tensor = torch.exp(cluster_frame_ln_likelihoods_tensor - log_norm.view(-1,1))
    # update the weights
    ln_weights_tensor = torch.log(torch.mean(gamma_tensor,0))
    # update averages and variances of each cluster
    for k in range(n_clusters):
        gamma_indeces = torch.argwhere(gamma_tensor[:,k] > gamma_thresh_tensor).flatten()
        # update mean and variance
        centers_tensor[k], vars_tensor[k] = torch_align.torch_iterative_align_uniform_weighted(traj_tensor[gamma_indeces], gamma_tensor[gamma_indeces,k].to(dtype), ref_tensor=centers_tensor[k], dtype=dtype, thresh=thresh, device=device)[1:]
    return centers_tensor, vars_tensor, ln_weights_tensor, log_likelihood    
    del gamma_tensor
    del gamma_indeces
    torch.cuda.empty_cache()


# Expectation step
def torch_sgmm_expectation_uniform(traj_tensor, centers_tensor, vars_tensor, device=torch.device("cuda:0")):
    # meta data
    n_clusters = centers_tensor.shape[0]
    n_frames = traj_tensor.shape[0] 
    n_atoms = traj_tensor.shape[1]
    # declare torch tensor
    cluster_frame_ln_likelihoods = torch.empty((n_frames,n_clusters),dtype=torch.float64,device=device)
    # compute likelihood of each frame at each Gaussian
    for k in range(n_clusters):
        # Determine square deviation of each frame aligned to each mean
        cluster_frame_ln_likelihoods[:,k] = torch_align.torch_sd(traj_tensor,centers_tensor[k])
        # divide be variance and normalize
        cluster_frame_ln_likelihoods[:,k] *= -0.5/vars_tensor[k]
        cluster_frame_ln_likelihoods[:,k] -= 1.5*(n_atoms-1)*torch.log(vars_tensor[k])
    return cluster_frame_ln_likelihoods
