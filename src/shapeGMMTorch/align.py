# library of trajectory alignment protocols using PyTorch

# import libraries
import torch

# =========================
# Low-level utilities
# =========================

@torch.no_grad()
def align_rot_mats(traj_tensor: torch.Tensor, ref_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute optimal rotation matrices to align traj_tensor frames to ref_tensor.
    """
    # compute correlation matrices using batched matmul
    c_mats = torch.matmul(ref_tensor.T,traj_tensor)
    # perfrom SVD of c_mats using batched SVD
    u, s, v = torch.linalg.svd(c_mats)
    # ensure true rotation by correcting sign of determinant
    prod_dets = torch.linalg.det(u)*torch.linalg.det(v)
    u[:,:,-1] *= prod_dets.unsqueeze(-1)
    # compute rotation matrices
    rot_mat = torch.matmul(u,v).transpose(1,2)
    return rot_mat

@torch.no_grad()
def remove_center_of_geometry(traj_tensor: torch.Tensor) -> torch.Tensor:
    """
    Remove the center of geometry from each frame of traj_tensor.
    """
    dtype = traj_tensor.dtype
    # compute geometric center of each frame
    cog = torch.mean(traj_tensor.to(torch.float64),dim=1,keepdim=True).to(dtype)
    # substract from each frame
    return traj_tensor - cog

@torch.no_grad()
def trajectory_sd(traj_tensor: torch.Tensor, ref_tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute squared displacement after uniform alignment to reference structure.
    """
    dtype = traj_tensor.dtype
    # meta data
    n_atoms = traj_tensor.shape[1]
    # get rotation matrices
    rot_mat = align_rot_mats(traj_tensor, ref_tensor)
    # do rotation
    traj_tensor = torch.matmul(traj_tensor,rot_mat)
    disp = (traj_tensor - ref_tensor).to(torch.float64)
    sd = torch.matmul(disp.view(-1,1,n_atoms*3),disp.view(-1,n_atoms*3,1))[:,0,0].to(dtype)
    # return values
    return sd

@torch.no_grad()
def _kronecker_covar(disp: torch.Tensor, covar_norm: float) -> torch.Tensor:
    n_frames = disp.shape[0]
    disp = torch.transpose(disp,0,1).reshape(-1,n_frames*3)
    covar = disp @ disp.T
    covar *= covar_norm
    return covar

    
# determine the ln(det) of a singular matrix ignoring eigenvalues below threshold
@torch.no_grad()
def _pseudo_lndet(sigma: torch.Tensor, EigenValueThresh: float = 1e-10) -> torch.Tensor:
    e = torch.linalg.eigvalsh(sigma) 
    e = torch.where(e > EigenValueThresh, e, 1.0)
    lpdet = torch.sum(torch.log(e))
    return lpdet

# determine the pseudo inverse and ln(det) of a singular matrix ignoring first eigenvalue
@torch.no_grad()
def _pseudo_inv(sigma: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    # diagonalize sigma
    e, v = torch.linalg.eigh(sigma) 
    # compute the log of the pseudo determinant of sigma
    lpdet = torch.sum(torch.log(e[1:]))
    # now compute multiplicative reciprocal of eigenvalues except first
    e[1:] = torch.reciprocal(e[1:])
    # set first eigenvalue/recirpical to zero
    e[0] = 0.0
    # construct the inverse
    inv = v @ torch.diag(e) @ v.T
    return inv, lpdet

    
# ===============================================
# Non-iterative Alignment Routines
# ===============================================

# perform uniform Kabsch alignment between trajectory frames and reference
@torch.no_grad()
def align_uniform(traj_tensor: torch.Tensor, ref_tensor: torch.Tensor) -> torch.Tensor:
    """
    Uniform Kabsch alignment between trajectory frames and reference
    """

    assert ref_tensor.device == traj_tensor.device, "ref_tensor must be on the same device as traj_tensor"
    # get rotation matrices
    rot_mat = align_rot_mats(traj_tensor, ref_tensor)
    # do rotation
    traj_tensor = torch.matmul(traj_tensor,rot_mat)
    return traj_tensor


@torch.no_grad()
def align_kronecker(traj_tensor, ref_tensor, precision_tensor):
    """
    Precision weighted Kabsch alignment between trajectory frames and reference
    """
    
    # Ensure all tensors are on the same device and dtype
    device = traj_tensor.device
    dtype = traj_tensor.dtype

    assert ref_tensor.device == device, "ref_tensor must be on the same device as traj_tensor"
    assert precision_tensor.device == device, "precision_tensor must be on the same device as traj_tensor"

    # make weighted ref
    weighted_ref = torch.matmul(precision_tensor,ref_tensor.to(torch.float64)).to(dtype)
    # get rotation matrices
    rot_mat = align_rot_mats(traj_tensor, weighted_ref)
    # do rotation
    traj_tensor = torch.matmul(traj_tensor,rot_mat)
    # return aligned trajectory
    return traj_tensor


# ===============================================
# Iterative Maximum Likelihood Alignment Routines
# ===============================================

@torch.no_grad()
def maximum_likelihood_uniform_alignment(
    traj_tensor: torch.Tensor,
    thresh: float = 1e-3,
    max_iter: int = 200,
    verbose: bool = False
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Perform iterative maximum likelihood Kabsch alignment of a trajectory to its mean structure
    using uniform frame and per-atom weighting.

    Parameters
    ----------
    traj_tensor : torch.Tensor
        Tensor of shape (n_frames, n_atoms, 3) containing the trajectory to be aligned.
    thresh : float, optional
        Convergence threshold for change in log-likelihood between iterations. Default is 1e-3.
    max_iter : int, optional
        Maximum number of iterations allowed. Default is 200.
    verbose : bool, optional
        If True, prints log-likelihood values during iteration.

    Returns
    -------
    traj_tensor : torch.Tensor
        The aligned trajectory tensor of shape (n_frames, n_atoms, 3).
    avg : torch.Tensor
        The final average structure of shape (n_atoms, 3).
    var : float
        The final estimate of the variance.
    """

    # Set device and dtype from input trajectory
    device = traj_tensor.device
    dtype = traj_tensor.dtype

    # Meta data
    n_frames, n_atoms = traj_tensor.shape[0], traj_tensor.shape[1]

    # Initialize with first frame
    avg = traj_tensor[0]

    # Precompute constants
    var_norm = 1.0 / (n_frames * 3 * (n_atoms - 1))
    log_lik_prefactor = -1.5 * (n_atoms - 1)

    delta_log_lik = thresh + 10
    old_log_lik = 0.0
    kabsch_iter = 0

    while delta_log_lik > thresh and kabsch_iter < max_iter:
        # Get rotation matrices
        rot_mat = align_rot_mats(traj_tensor, avg)

        # Apply rotation
        traj_tensor = torch.matmul(traj_tensor, rot_mat)

        # Compute new average
        avg = torch.mean(traj_tensor, dim=0, keepdim=False)

        # Compute displacements
        disp = (traj_tensor - avg).to(torch.float64)

        # Compute variance
        var = torch.sum(disp*disp) * var_norm

        # Compute log-likelihood
        log_lik = log_lik_prefactor * (torch.log(var) + 1.0)

        delta_log_lik = torch.abs(log_lik - old_log_lik)
        if verbose:
            print(f"Iteration {kabsch_iter}: log-likelihood = {log_lik.item()}")

        old_log_lik = log_lik
        kabsch_iter += 1

    if kabsch_iter == max_iter:
        print("Warning: ML alignment not completely converged")

    # Return results
    return traj_tensor, avg, var


@torch.no_grad()
def maximum_likelihood_uniform_alignment_frame_weighted(
    traj_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    ref_tensor: torch.Tensor = None,
    thresh: float = 1e-3,
    max_iter: int = 200,
    verbose: bool = False
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    Perform iterative maximum likelihood alignment of a trajectory to a reference structure using frame weights
    and uniform per-atom weighting (i.e., no mass weighting).

    Parameters
    ----------
    traj_tensor : torch.Tensor
        Tensor of shape (n_frames, n_atoms, 3) containing the trajectory to be aligned.
    weight_tensor : torch.Tensor
        Tensor of shape (n_frames,) containing weights for each frame. Will be normalized internally.
    ref_tensor : torch.Tensor, optional
        Tensor of shape (n_atoms, 3) representing the reference structure.
        If None, the first frame of the trajectory is used as the initial reference.
    thresh : float, optional
        Convergence threshold for change in log-likelihood between iterations. Default is 1e-3.
    max_iter : int, optional
        Maximum number of iterations allowed. Default is 200.
    verbose : bool, optional
        If True, prints log-likelihood values during iteration.

    Returns
    -------
    traj_tensor : torch.Tensor
        The aligned trajectory tensor of shape (n_frames, n_atoms, 3).
    avg : torch.Tensor
        The final weighted average structure of shape (n_atoms, 3).
    var : float
        The final estimate of the variance.
    """

    # Set device and dtype from input trajectory
    device = traj_tensor.device
    dtype = traj_tensor.dtype

    # Assert input tensor devices
    assert weight_tensor.device == device, "weight_tensor must be on the same device as traj_tensor"
    if ref_tensor is not None:
        assert ref_tensor.device == device, "ref_tensor must be on the same device as traj_tensor"

    # Meta data
    n_frames, n_atoms = traj_tensor.shape[0], traj_tensor.shape[1]

    # Initialize average structure
    if ref_tensor is None:
        avg = traj_tensor[0]
    else:
        avg = ref_tensor

    # Normalize weights - but do so in f64 for mixed precision support
    weight_tensor = weight_tensor.to(dtype)
    weight_tensor = weight_tensor / weight_tensor.sum()

    # Precompute constants
    var_norm = 1.0 / (3 * (n_atoms - 1))
    log_lik_prefactor = -1.5 * (n_atoms - 1)

    delta_log_lik = thresh + 10
    old_log_lik = 0.0
    kabsch_iter = 0

    while delta_log_lik > thresh and kabsch_iter < max_iter:
        # Get rotation matrices
        rot_mat = align_rot_mats(traj_tensor, avg)

        # Apply rotation
        traj_tensor = torch.matmul(traj_tensor, rot_mat)

        # Compute weighted average
        avg = torch.einsum('ijk,i->jk', traj_tensor, weight_tensor)  # weighted average 

        # compute displacement then cast to f64
        disp = (traj_tensor - avg).to(torch.float64)
        
        # Compute variance
        disp_squared = torch.sum(disp ** 2, dim=(1, 2))  # (n_frames,)
        var = torch.dot(disp_squared.to(dtype), weight_tensor) * var_norm

        # Compute log-likelihood
        log_lik = log_lik_prefactor * (torch.log(var) + 1.0)

        delta_log_lik = torch.abs(log_lik - old_log_lik)
        if verbose:
            print(f"Iteration {kabsch_iter}: log-likelihood = {log_lik.item()}")

        old_log_lik = log_lik
        kabsch_iter += 1

    if kabsch_iter == max_iter:
        print("Warning: ML alignment not completely converged")

    # Return results
    return traj_tensor, avg, var


@torch.no_grad()
def maximum_likelihood_kronecker_alignment(
    traj_tensor: torch.Tensor,
    thresh: float = 1e-3,
    max_iter: int = 200,
    verbose: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform iterative maximum likelihood alignment of a trajectory to its mean structure using Kronecker-based covariance.
    Assumes uniform weighting of all frames.

    Parameters
    ----------
    traj_tensor : torch.Tensor
        Tensor of shape (n_frames, n_atoms, 3) containing the trajectory to be aligned.
    thresh : float, optional
        Convergence threshold for change in log-likelihood between iterations. Default is 1e-3.
    max_iter : int, optional
        Maximum number of iterations allowed. Default is 200.
    verbose : bool, optional
        If True, prints log-likelihood values during iteration.

    Returns
    -------
    traj_tensor : torch.Tensor
        The aligned trajectory tensor of shape (n_frames, n_atoms, 3).
    avg : torch.Tensor
        The final average structure of shape (n_atoms, 3).
    precision : torch.Tensor
        The precision matrix (pseudo-inverse of covariance matrix).
    lpdet : float
        Log pseudo-determinant of the covariance matrix.
    """

    # Set device and dtype from input trajectory
    device = traj_tensor.device
    dtype = traj_tensor.dtype

    # Meta data
    n_frames, n_atoms = traj_tensor.shape[0], traj_tensor.shape[1]

    # Normalization factor for covariance
    covar_norm = 1.0 / (3 * n_frames - 1)  # scalar

    # Initialize weighted average
    weighted_avg = traj_tensor[0]

    delta_log_lik = thresh + 10
    old_log_lik = 0.0
    kabsch_iter = 0

    while delta_log_lik > thresh and kabsch_iter < max_iter:
        # Get rotation matrices
        rot_mat = align_rot_mats(traj_tensor, weighted_avg)

        # Apply rotation
        traj_tensor = torch.matmul(traj_tensor, rot_mat)

        # Compute new average
        avg = torch.mean(traj_tensor, dim=0, keepdim=False)

        # Compute displacements
        disp = (traj_tensor - avg).to(torch.float64)

        # Compute covariance
        covar = _kronecker_covar(disp, covar_norm)

        # Compute precision matrix and log pseudo-determinant
        precision, lpdet = _pseudo_inv(covar)

        # Compute log-likelihood
        log_lik = -1.5 * lpdet

        delta_log_lik = torch.abs(log_lik - old_log_lik)
        if verbose:
            print(f"Iteration {kabsch_iter}: log-likelihood = {log_lik.item()}")

        old_log_lik = log_lik

        # Update weighted average
        weighted_avg = torch.matmul(precision.to(dtype), avg)

        kabsch_iter += 1

    if kabsch_iter == max_iter:
        print("Warning: ML alignment not completely converged")

    # Return results
    return traj_tensor, avg, precision, lpdet


@torch.no_grad()
def maximum_likelihood_kronecker_alignment_frame_weighted(
    traj_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    ref_tensor: torch.Tensor = None,
    ref_precision_tensor: torch.Tensor = None,
    thresh: float = 1e-3,
    max_iter: int = 200,
    verbose: bool = False
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Perform iterative maximum likelihood alignment of a trajectory to a reference using weighted Kronecker-based covariance.

    Parameters
    ----------
    traj_tensor : torch.Tensor
        Tensor of shape (n_frames, n_atoms, 3) containing the trajectory to be aligned.
    weight_tensor : torch.Tensor
        Tensor of shape (n_frames,) containing weights for each frame. Will be normalized internally.
    ref_tensor : torch.Tensor, optional
        Tensor of shape (n_atoms, 3) representing the reference structure.
        If None, the first frame of the trajectory is used as the initial reference.
    ref_precision_tensor : torch.Tensor, optional
        Tensor representing the reference precision matrix (covariance inverse) used for weighting.
        Required if `ref_tensor` is provided.
    thresh : float, optional
        Convergence threshold for change in log-likelihood between iterations. Default is 1e-3.
    max_iter : int, optional
        Maximum number of iterations allowed. Default is 200.
    verbose : bool, optional
        If True, prints log-likelihood values during iteration.

    Returns
    -------
    traj_tensor : torch.Tensor
        The aligned trajectory tensor of shape (n_frames, n_atoms, 3).
    avg : torch.Tensor
        The final weighted average structure of shape (n_atoms, 3).
    precision : torch.Tensor
        The precision matrix (pseudo-inverse of covariance matrix).
    lpdet : float
        Log pseudo-determinant of the covariance matrix.
    """

    # Ensure all tensors are on the same device and dtype
    device = traj_tensor.device
    dtype = traj_tensor.dtype

    assert weight_tensor.device == device, "weight_tensor must be on the same device as traj_tensor"
    if ref_tensor is not None:
        assert ref_tensor.device == device, "ref_tensor must be on the same device as traj_tensor"
        assert ref_precision_tensor is not None, "ref_precision_tensor must be provided if ref_tensor is given"
        assert ref_precision_tensor.device == device, "ref_precision_tensor must be on the same device as traj_tensor"

    # Normalize weights and cache sqrt of reshaped version 
    weight_tensor = weight_tensor.to(dtype)
    weight_tensor = weight_tensor / weight_tensor.sum()
    sqrt_w = torch.sqrt(weight_tensor).view(-1, 1, 1).to(torch.float64)

    # Handle reference
    if ref_tensor is None:
        weighted_avg = traj_tensor[0]
    else:
        weighted_avg = torch.matmul(ref_precision_tensor, ref_tensor.to(torch.float64)).to(dtype)

    # Pre-define convergence control
    delta_log_lik = thresh + 10
    old_log_lik = 0.0
    kabsch_iter = 0

    covar_norm = 1.0 / 3.0  # scalar normalization for covariance

    while delta_log_lik > thresh and kabsch_iter < max_iter:
        # Get rotation matrices
        rot_mat = align_rot_mats(traj_tensor, weighted_avg)

        # Apply rotation
        traj_tensor = torch.matmul(traj_tensor, rot_mat)

        # cast traj tensor to f64 for mixed precision support
        #traj_tensor_f64 = traj_tensor.to(torch.float64)
        
        # Compute weighted average
        #avg = torch.einsum('ijk,i->jk', traj_tensor64, weight_tensor)
        avg = torch.einsum('ijk,i->jk', traj_tensor, weight_tensor)

        # Compute displacements
        #disp = traj_tensor_f64 - avg
        disp = (traj_tensor - avg).to(torch.float64)

        # Compute covariance
        covar = _kronecker_covar(disp * sqrt_w, covar_norm)

        # Compute precision matrix and log pseudo-determinant
        precision, lpdet = _pseudo_inv(covar)

        # Compute log-likelihood
        log_lik = -1.5 * lpdet

        delta_log_lik = torch.abs(log_lik - old_log_lik)
        if verbose:
            print(f"Iteration {kabsch_iter}: log-likelihood = {log_lik.item()}")

        old_log_lik = log_lik

        # Update weighted average
        weighted_avg = torch.matmul(precision.to(dtype), avg)

        kabsch_iter += 1

    if kabsch_iter == max_iter:
        print("Warning: ML alignment not completely converged")

    # clear up memory
    del disp, covar, weighted_avg
    torch.cuda.empty_cache()

    # Return results
    return traj_tensor, avg, precision, lpdet


# =========================
# Optional torch.compile
# =========================

if hasattr(torch, 'compile'):
    #align_rot_mats = torch.compile(align_rot_mats)
    maximum_likelihood_uniform_alignment = torch.compile(maximum_likelihood_uniform_alignment)
    maximum_likelihood_uniform_alignment_frame_weighted = torch.compile(maximum_likelihood_uniform_alignment_frame_weighted)
    maximum_likelihood_kronecker_alignment = torch.compile(maximum_likelihood_kronecker_alignment)
    maximum_likelihood_kronecker_alignment_frame_weighted = torch.compile(maximum_likelihood_kronecker_alignment_frame_weighted)

