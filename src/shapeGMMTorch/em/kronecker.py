import numpy as np
import warnings
warnings.filterwarnings('ignore')
import random
from .. import align_in_place
from .. import align
import torch

@torch.no_grad()
def sgmm_kronecker_em(
    traj_tensor: torch.Tensor,
    frame_weights_tensor: torch.Tensor,
    means_tensor: torch.Tensor,
    precisions_tensor: torch.Tensor,
    lpdets_tensor: torch.Tensor,
    ln_weights_tensor: torch.Tensor,
    thresh: float = 1e-1,
    max_iter: int = 200,
    gamma_thresh: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Expectation-Maximization update for Gaussian Mixture Model with Kronecker covariance.

    Parameters
    ----------
    traj_tensor : torch.Tensor
        Tensor of shape (n_frames, n_atoms, 3).
    frame_weights_tensor : torch.Tensor
        Tensor of shape (n_frames,).
    means_tensor : torch.Tensor (modified in-place)
        Tensor of shape (n_clusters, n_atoms, 3). Updated during M-step.
    precisions_tensor : torch.Tensor (modified in-place)
        Tensor of shape (n_clusters, n_atoms, n_atoms). Updated in-place.
    lpdets_tensor : torch.Tensor (modified in-place)
        Tensor of shape (n_clusters,). Updated in-place.
    ln_weights_tensor : torch.Tensor (modified in-place)
        Log cluster weights. Updated in-place.
    thresh : float
        Convergence threshold for alignment steps. Default is 1e-1.
    max_iter : int
        Maximum iterations for alignment steps. Default is 200.
    gamma_thresh : float
        Threshold for gamma filtering during M-step. Default is 1e-10.

    Returns
    -------
    log_likelihood : torch.Tensor
        Log-likelihood of the model after the EM step.
    """

    device = traj_tensor.device
    dtype = traj_tensor.dtype

    n_frames, n_atoms, _ = traj_tensor.shape
    n_clusters = ln_weights_tensor.shape[0]

    # Expectation step: Compute log-likelihoods
    cluster_frame_ln_likelihoods_tensor = sgmm_expectation_kronecker(
        traj_tensor, means_tensor, precisions_tensor, lpdets_tensor
    )

    # Add log cluster weights
    cluster_frame_ln_likelihoods_tensor += ln_weights_tensor.view(1, -1)

    # Log-sum-exp normalization (log p(x))
    log_norm = torch.logsumexp(cluster_frame_ln_likelihoods_tensor, dim=1)

    # Weighted log-likelihood (using frame weights)
    log_likelihood = torch.sum(frame_weights_tensor * log_norm)

    # Compute gamma (posterior probabilities)
    gamma_tensor = torch.exp(cluster_frame_ln_likelihoods_tensor - log_norm.unsqueeze(1))

    # Weight gamma by frame weights
    gamma_tensor *= frame_weights_tensor.unsqueeze(1)

    # Update log cluster weights
    ln_weights_tensor.copy_(torch.log(torch.sum(gamma_tensor, dim=0)))

    # Precompute mask of frames above gamma threshold for all clusters
    gamma_mask = gamma_tensor > gamma_thresh

    # Precompute list of valid frame indices per cluster
    gamma_indices_list = [torch.nonzero(gamma_mask[:, k], as_tuple=False).squeeze() for k in range(n_clusters)]

    # Update cluster means and precisions
    for k in range(n_clusters):
        gamma_indices = gamma_indices_list[k]

        if gamma_indices.numel() > n_atoms:
            # Extract once
            selected_traj = traj_tensor.index_select(0, gamma_indices)
            selected_gamma = gamma_tensor.index_select(0, gamma_indices)[:, k]
            # Perform weighted alignment update
            new_mean, new_precision, new_lpdet = align_in_place.maximum_likelihood_kronecker_alignment_frame_weighted_in_place(
                selected_traj, 
                selected_gamma,
                ref_tensor=means_tensor[k],
                ref_precision_tensor=precisions_tensor[k],
                thresh=thresh,
                max_iter=max_iter
                )
            means_tensor[k].copy_(new_mean)
            precisions_tensor[k].copy_(new_precision)
            lpdets_tensor[k].copy_(new_lpdet)

    return log_likelihood  


@torch.no_grad()
def sgmm_expectation_kronecker(
    traj_tensor: torch.Tensor, 
    means_tensor: torch.Tensor, 
    precisions_tensor: torch.Tensor, 
    lpdets_tensor: torch.Tensor, 
) -> torch.Tensor:
    """
    Compute the log-likelihoods of each frame under each Gaussian cluster
    using Kronecker structure for the covariance matrices.

    Parameters
    ----------
    traj_tensor : torch.Tensor
        Tensor of shape (n_frames, n_atoms, 3).
    means_tensor : torch.Tensor
        Tensor of shape (n_clusters, n_atoms, 3).
    precisions_tensor : torch.Tensor
        Tensor of shape (n_clusters, n_atoms, n_atoms).
    lpdets_tensor : torch.Tensor
        Tensor of shape (n_clusters,).

    Returns
    -------
    cluster_frame_ln_likelihoods : torch.Tensor
        Tensor of shape (n_frames, n_clusters) containing the log-likelihoods.
    """

    device = traj_tensor.device
    dtype = traj_tensor.dtype
    n_clusters = means_tensor.shape[0]
    n_frames, n_atoms = traj_tensor.shape[0], traj_tensor.shape[1]

    # Output tensor
    cluster_frame_ln_likelihoods = torch.empty((n_frames, n_clusters), dtype=torch.float64, device=device)


    for k in range(n_clusters):
        # Align to the mean of each cluster using in place operation
        aligned_traj = align.align_kronecker(traj_tensor, means_tensor[k], precisions_tensor[k])

        # Displacement tensor
        disp = aligned_traj - means_tensor[k]  # (n_frames, n_atoms, 3) 
        # compute Kronecker quadratic form as sum over x, y, z quadratic forms
        quad = 0.0
        for d in range(3):  # x, y, z
            v = disp[:,:,d].to(torch.float64)  # promote just 2D slice
            vT = v.view(n_frames, 1, n_atoms)
            v = v.view(n_frames, n_atoms, 1)
            quad += torch.matmul(vT, torch.matmul(precisions_tensor[k], v))[:,0,0]

        # Compute quadratic form using batch matrix multiplication
        # transpose (n_frames, n_atoms, 3) -> (n_frames, 3, n_atoms)
        #disp_t = disp.transpose(1, 2)  # (n_frames, 3, n_atoms)

        # Multiply: (n_frames, 3, n_atoms) @ (n_atoms, n_atoms) @ (n_frames, n_atoms, 3)
        # Efficient batched bilinear form: sum over atoms
        #middle = torch.matmul(disp_t, precisions_tensor[k])  # (n_frames, 3, n_atoms)
        #quad = (middle * disp_t).sum(dim=(1, 2))  # sum over atom and dimension axes casting for mixed precision support

        # Compute log-likelihood
        cluster_frame_ln_likelihoods[:, k] = -0.5 * quad - 1.5 * lpdets_tensor[k]

    return cluster_frame_ln_likelihoods




