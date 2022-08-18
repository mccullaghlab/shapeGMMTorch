import numpy as np
import warnings
warnings.filterwarnings('ignore')
import random
from . import torch_align
import torch

GAMMA_THRESH = 1e-15

##

def kronecker_sgmm_log_likelihood(traj_tensor, clusters, thresh=1e-3, dtype=torch.float32, device=torch.device("cuda:0")):
    # meta data from inputs
    n_frames = traj_tensor.shape[0]
    n_clusters = np.amax(clusters) + 1
    n_atoms = traj_tensor.shape[1]
    n_dim = traj_tensor.shape[2]
    n_features = n_atoms*n_dim
    # declare arrays 
    cluster_frame_ln_likelihoods = torch.empty((n_frames, n_clusters),dtype=torch.float64, device=device)
    # compute likelihood of each frame at each Gaussian
    for k in range(n_clusters):
        indeces = np.argwhere(clusters == k).flatten()
        # initialize weights as populations of clusters
        ln_weight = torch.tensor(np.log(indeces.size/n_frames),dtype=torch.float64,device=device)
        # determine center and precision of cluster using iterative alignment
        center_tensor, precision_tensor, lpdet_tensor = torch_align.torch_iterative_align_kronecker(traj_tensor[indeces], thresh=thresh, dtype=dtype, device=device)[1:]
        # align the entire trajectory to each cluster mean
        traj_tensor = torch_align.torch_align_kronecker(traj_tensor,center_tensor, precision_tensor)
        # compute log likelihood per frame
        disp = (traj_tensor - center_tensor).to(torch.float64)
        # Determine square deviation of each frame aligned to each mean
        cluster_frame_ln_likelihoods[:,k] = torch.matmul(disp[:,:,0].view(n_frames,1,n_atoms),torch.matmul(precisions_tensor,disp[:,:,0].view(n_frames,n_atoms,1)))[:,0,0]
        cluster_frame_ln_likelihoods[:,k] += torch.matmul(disp[:,:,1].view(n_frames,1,n_atoms),torch.matmul(precisions_tensor,disp[:,:,1].view(n_frames,n_atoms,1)))[:,0,0]
        cluster_frame_ln_likelihoods[:,k] += torch.matmul(disp[:,:,2].view(n_frames,1,n_atoms),torch.matmul(precisions_tensor,disp[:,:,2].view(n_frames,n_atoms,1)))[:,0,0]
        # divide be variance and normalize
        cluster_frame_ln_likelihoods[:,k] *= -0.5
        cluster_frame_ln_likelihoods[:,k] -= 1.5*lpdet_tensor
    # compute log likelihood
    log_likelihood = torch.logsumexp(cluster_frame_ln_likelihoods,1)
    log_likelihood = torch.mean(log_likelihood)
    del cluster_frame_ln_likelihoods
    del ln_weight
    del center_tensor
    del precision_tensor
    del lpdet_tensor
    torch.cuda.empty_cache()
    return log_likelihood

## Expectation Maximization for GMM with Kronecker covariance model
def torch_sgmm_kronecker_em(traj_tensor, centers_tensor, precisions_tensor, lpdets_tensor, ln_weights_tensor, dtype=torch.float32, device=torch.device("cuda:0"), thresh=1e-1):
    
    # get metadata from trajectory data
    n_frames = traj_tensor.shape[0]
    n_atoms = traj_tensor.shape[1]
    n_dim = traj_tensor.shape[2]
    n_clusters = ln_weights_tensor.shape[0]

    gamma_thresh_tensor = torch.tensor(GAMMA_THRESH,dtype=torch.float64,device=device)

    # Expectation step:
    cluster_frame_ln_likelihoods_tensor = torch_sgmm_expectation_kronecker(traj_tensor, centers_tensor, precisions_tensor, lpdets_tensor, dtype=dtype, device=device)
    
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
        if gamma_indeces.shape[0] > n_atoms:
            # update mean and variance
            centers_tensor[k], precisions_tensor[k], lpdets_tensor[k] = torch_align.torch_iterative_align_kronecker_weighted(traj_tensor[gamma_indeces], gamma_tensor[gamma_indeces,k].to(dtype), ref_tensor=centers_tensor[k], ref_precision_tensor=precisions_tensor[k], dtype=dtype, thresh=thresh, device=device)[1:]
    return centers_tensor, precisions_tensor, lpdets_tensor, ln_weights_tensor, log_likelihood    
    del cluster_frame_ln_likelihoods
    del log_lorm
    del gamma_tensor
    del gamma_indeces
    torch.cuda.empty_cache()


# Expectation step
def torch_sgmm_expectation_kronecker(traj_tensor, centers_tensor, precisions_tensor, lpdets_tensor, dtype=torch.float32, device=torch.device("cuda:0")):
    # meta data
    n_clusters = centers_tensor.shape[0]
    n_frames = traj_tensor.shape[0] 
    n_atoms = traj_tensor.shape[1]
    # declare torch tensor
    cluster_frame_ln_likelihoods = torch.empty((n_frames,n_clusters),dtype=torch.float64,device=device)
    # compute likelihood of each frame at each Gaussian
    for k in range(n_clusters):
        # align trajectory to center with given preicions
        traj_tensor = torch_align.torch_align_kronecker(traj_tensor, centers_tensor[k], precisions_tensor[k],dtype=dtype, device=device)
        disp = (traj_tensor - centers_tensor[k]).to(torch.float64)
        # Determine square deviation of each frame aligned to each mean
        cluster_frame_ln_likelihoods[:,k] = torch.matmul(disp[:,:,0].view(n_frames,1,n_atoms),torch.matmul(precisions_tensor[k],disp[:,:,0].view(n_frames,n_atoms,1)))[:,0,0]
        cluster_frame_ln_likelihoods[:,k] += torch.matmul(disp[:,:,1].view(n_frames,1,n_atoms),torch.matmul(precisions_tensor[k],disp[:,:,1].view(n_frames,n_atoms,1)))[:,0,0]
        cluster_frame_ln_likelihoods[:,k] += torch.matmul(disp[:,:,2].view(n_frames,1,n_atoms),torch.matmul(precisions_tensor[k],disp[:,:,2].view(n_frames,n_atoms,1)))[:,0,0]
        # divide be variance and normalize
        cluster_frame_ln_likelihoods[:,k] *= -0.5
        cluster_frame_ln_likelihoods[:,k] -= 1.5*lpdets_tensor[k]
    return cluster_frame_ln_likelihoods



