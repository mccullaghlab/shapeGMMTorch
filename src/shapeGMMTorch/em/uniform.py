import numpy as np
import warnings
warnings.filterwarnings('ignore')
import random
from .. import align
from .. import align_in_place
import torch

@torch.no_grad()
def sgmm_uniform_em(
    traj_tensor: torch.Tensor,
    frame_weights_tensor: torch.Tensor,
    means_tensor: torch.Tensor,
    vars_tensor: torch.Tensor,
    ln_weights_tensor: torch.Tensor,
    thresh: float = 1e-3,
    max_iter: int = 200,
    gamma_thresh: float = 1e-10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Expectation-Maximization update for Gaussian Mixture Model with uniform covariance.

    Parameters
    ----------
    traj_tensor : torch.Tensor
        Tensor of shape (n_frames, n_atoms, 3).
    frame_weights_tensor : torch.Tensor
        Tensor of shape (n_frames,).
    means_tensor : torch.Tensor (modified in-place)
        Tensor of shape (n_clusters, n_atoms, 3). Updated in-place.
    vars_tensor : torch.Tensor (modified in-place)
        Tensor of shape (n_clusters,). Updated in-place.
    ln_weights_tensor : torch.Tensor (modified in-place)
        Tensor of shape (n_clusters,). Updated in-place.
    thresh : float
        Convergence threshold for alignment steps. Default is 1e-3.
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

    n_frames, n_atoms, _ = traj_tensor.shape
    n_clusters = ln_weights_tensor.shape[0]

    gamma_thresh_tensor = torch.tensor(gamma_thresh, dtype=torch.float64, device=device)

    # Expectation step
    cluster_frame_ln_likelihoods_tensor = sgmm_expectation_uniform(
        traj_tensor, means_tensor, vars_tensor
    )

    # Add log cluster weights
    cluster_frame_ln_likelihoods_tensor += ln_weights_tensor.view(1, -1)

    # Log-sum-exp normalization (log p(x))
    log_norm = torch.logsumexp(cluster_frame_ln_likelihoods_tensor, dim=1)

    # Weighted log-likelihood
    log_likelihood = torch.sum(frame_weights_tensor * log_norm)

    # Posterior probabilities (gamma)
    gamma_tensor = torch.exp(cluster_frame_ln_likelihoods_tensor - log_norm.unsqueeze(1))
    gamma_tensor *= frame_weights_tensor.unsqueeze(1)

    # Update log cluster weights
    ln_weights_tensor.copy_(torch.log(torch.sum(gamma_tensor, dim=0)))

    # Precompute mask of frames above gamma threshold for all clusters
    gamma_mask = gamma_tensor > gamma_thresh_tensor

    # Precompute list of valid frame indices per cluster
    gamma_indices_list = [torch.nonzero(gamma_mask[:, k], as_tuple=False).squeeze() for k in range(n_clusters)]

    # Update cluster means and variances
    for k in range(n_clusters):
        gamma_indices = gamma_indices_list[k]

        if gamma_indices.numel() > n_atoms:
            selected_traj = traj_tensor.index_select(0, gamma_indices)
            selected_gamma = gamma_tensor.index_select(0, gamma_indices)[:, k].to(traj_tensor.dtype)

            new_mean, new_var = align_in_place.maximum_likelihood_uniform_alignment_frame_weighted_in_place(
                selected_traj,
                selected_gamma,
                ref_tensor=means_tensor[k],
                thresh=thresh,
                max_iter=max_iter
                )#[1:]
            means_tensor[k].copy_(new_mean)
            vars_tensor[k].copy_(new_var)

    return log_likelihood



@torch.no_grad()
def sgmm_expectation_uniform(
    traj_tensor: torch.Tensor,
    means_tensor: torch.Tensor,
    vars_tensor: torch.Tensor,
) -> torch.Tensor:
    """
    Compute the log-likelihoods of each frame under each Gaussian cluster
    assuming a uniform covariance model (scalar variance).

    Parameters
    ----------
    traj_tensor : torch.Tensor
        Tensor of shape (n_frames, n_atoms, 3).
    means_tensor : torch.Tensor
        Tensor of shape (n_clusters, n_atoms, 3).
    vars_tensor : torch.Tensor
        Tensor of shape (n_clusters,).

    Returns
    -------
    cluster_frame_ln_likelihoods : torch.Tensor
        Tensor of shape (n_frames, n_clusters) containing the log-likelihoods.
    """
    device = traj_tensor.device
    dtype = traj_tensor.dtype

    n_frames, n_atoms, _ = traj_tensor.shape
    n_clusters = means_tensor.shape[0]

    # Allocate output tensor
    cluster_frame_ln_likelihoods = torch.empty((n_frames, n_clusters), dtype=torch.float64, device=device)

    # Precompute constant part for each cluster (no need to recompute per frame)
    log_prefactors = 1.5 * (n_atoms - 1) * torch.log(vars_tensor.to(torch.float64))  # (n_clusters,)
    inv_vars = -0.5 / vars_tensor.to(torch.float64)  # (n_clusters,)

    for k in range(n_clusters):
        # Compute squared displacements after alignment
        sd = align.trajectory_sd(traj_tensor, means_tensor[k])  # (n_frames,)

        # Compute log-likelihood: -0.5 * (sq / var) - const
        cluster_frame_ln_likelihoods[:, k] = sd * inv_vars[k] - log_prefactors[k]

    return cluster_frame_ln_likelihoods

