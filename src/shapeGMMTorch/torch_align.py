# library of trajectory alignment protocols using PyTorch

# import libraries
import numpy as np
import torch
import importlib
# check to see if fast SVD library is available
svd_loader = importlib.util.find_spec('torch_bath_svd')
if svd_loader is not None:
    from torch_batch_svd import svd
    def torch_align_rot_mat(traj_tensor, ref_tensor):
        # compute correlation matrices using batched matmul
        c_mats = torch.matmul(ref_tensor.T,traj_tensor)
        # perfrom SVD of c_mats using batched SVD
        u, s, v = svd(c_mats)
        # ensure true rotation by correcting sign of determinant
        prod_dets = torch.linalg.det(u)*torch.linalg.det(v)
        u[:,:,-1] *= prod_dets.view(-1,1)
        # free up local variables
        del c_mats
        del s
        del prod_dets
        torch.cuda.empty_cache()
        # return rotation matrices
        return torch.matmul(v,torch.transpose(u,1,2))
# otherwise use PyTorch native SVD
else:
    def torch_align_rot_mat(traj_tensor, ref_tensor):
        # compute correlation matrices using batched matmul
        c_mats = torch.matmul(ref_tensor.T,traj_tensor)
        # perfrom SVD of c_mats using batched SVD
        u, s, v = torch.linalg.svd(c_mats)
        # ensure true rotation by correcting sign of determinant
        prod_dets = torch.linalg.det(u)*torch.linalg.det(v)
        u[:,:,-1] *= prod_dets.view(-1,1)
        # free up local variables
        del c_mats
        del s
        del prod_dets
        torch.cuda.empty_cache()
        # return rotation matrices
        return torch.transpose(torch.matmul(u,v),1,2)

# remove center-of-geometry from trajectory
def torch_remove_center_of_geometry(traj_tensor, dtype=torch.float32, device=torch.device("cuda:0")):
    # meta data
    n_frames = traj_tensor.shape[0]
    # compute geometric center of each frame
    cog = torch.mean(traj_tensor.to(torch.float64),1,True)
    # substract from each frame
    traj_tensor -= cog
    # free up local variables 
    del cog
    torch.cuda.empty_cache()

# compute the squared displacement (sq) between trajectory frames and reference after uniform alignment
def torch_sd(traj_tensor, ref_tensor, dtype=torch.float64):
    # meta data
    n_atoms = traj_tensor.shape[1]
    # get rotation matrices
    rot_mat = torch_align_rot_mat(traj_tensor, ref_tensor)
    # do rotation
    traj_tensor = torch.matmul(traj_tensor,rot_mat)
    disp = (traj_tensor - ref_tensor).to(torch.float64)
    sd = torch.matmul(disp.view(-1,1,n_atoms*3),disp.view(-1,n_atoms*3,1))[:,0,0].to(dtype)
    # free up local variables 
    del rot_mat
    del disp
    torch.cuda.empty_cache()    
    # return values
    return sd
    
# perform uniform Kabsch alignment between trajectory frames and reference
def torch_align_uniform(traj_tensor, ref_tensor):

    # get rotation matrices
    rot_mat = torch_align_rot_mat(traj_tensor, ref_tensor)
    # do rotation
    traj_tensor = torch.matmul(traj_tensor,rot_mat)
    # delete local variables
    del rot_mat
    torch.cuda.empty_cache()
    return traj_tensor

# compute the log likelihood of uniform Kabsch alignment 
def _torch_uniform_log_likelihood(disp, n_frames, n_atoms, var_norm, log_lik_prefactor):
    # reshape displacement 
    disp = disp.view(n_frames*n_atoms*3,1)
    # compute variance
    var = torch.sum(disp*disp)
    var *= var_norm
    # compute log likelihood (per frame)
    log_lik = log_lik_prefactor*(torch.log(var) + 1)
    return log_lik, var

# Perform iterative Kabsch alignment to compute aligned trajectory, average and variance
def torch_iterative_align_uniform(traj_tensor, dtype=torch.float32, device=torch.device("cuda:0"), thresh=1e-3, verbose=False):
    # meta data
    n_frames = traj_tensor.shape[0]
    n_atoms = traj_tensor.shape[1]
    # initialize with average as the first frame (arbitrary choice)
    avg = traj_tensor[0]
    # setup some stuff
    var_norm = torch.tensor(1/(n_frames*3*(n_atoms-1)),dtype=torch.float64,device=device)
    log_lik_prefactor = torch.tensor(-1.5*(n_atoms-1),dtype=torch.float64,device=device)
    delta_log_lik = thresh+10
    old_log_lik = 0
    # loop
    while (delta_log_lik > thresh):
        # get rotation matrices
        rot_mat = torch_align_rot_mat(traj_tensor, avg)
        # do rotation
        traj_tensor = torch.matmul(traj_tensor,rot_mat)
        # compute new average
        avg = torch.mean(traj_tensor,0,False)
        disp = (traj_tensor - avg).to(torch.float64)
        # compute log likelihood and variance
        log_lik, var = _torch_uniform_log_likelihood(disp, n_frames, n_atoms, var_norm, log_lik_prefactor)
        delta_log_lik = abs(log_lik - old_log_lik)
        if verbose==True:
            print(log_lik.cpu().numpy())
        old_log_lik = log_lik
    
    # free up local variables 
    del rot_mat
    del disp
    torch.cuda.empty_cache()
    # return values
    return traj_tensor, avg, var

# Perform iterative uniform Kabsch alignment of trajectory using frame weights
def torch_iterative_align_uniform_weighted(traj_tensor, weight_tensor, ref_tensor=[], dtype=torch.float32, device=torch.device("cuda:0"), thresh=1e-3, verbose=False):
    # meta data
    n_frames = traj_tensor.shape[0]
    n_atoms = traj_tensor.shape[1]

    if ref_tensor == []:
        # initialize with average as the first frame (arbitrary choice)
        avg = traj_tensor[0]
    else:
        avg = ref_tensor
        
    # ensure weights are normalized
    weight_tensor /= torch.sum(weight_tensor)
    var_norm = torch.tensor(1/(3*(n_atoms-1)),device=device,dtype=torch.float64)
    log_lik_prefactor = torch.tensor(-1.5*(n_atoms-1),dtype=torch.float64,device=device)

    delta_log_lik = thresh+10
    old_log_lik = 0
    while (delta_log_lik > thresh):
        # get rotation matrices
        rot_mat = torch_align_rot_mat(traj_tensor, avg)
        # do rotation
        traj_tensor = torch.matmul(traj_tensor,rot_mat)
        # compute new average
        avg = torch.sum((traj_tensor * weight_tensor.view(-1,1,1)),0,False)
        # compute displacement
        disp = (traj_tensor - avg).to(torch.float64) 
        # square and weight displacement 
        disp = torch.matmul(disp.view(-1,1,n_atoms*3),disp.view(-1,n_atoms*3,1))[:,0,0] * weight_tensor
        # compute variance
        var = torch.sum(disp)
        var *= var_norm
        # compute log likelihood (per frame)
        log_lik = log_lik_prefactor * (torch.log(var) + 1)
        delta_log_lik = abs(log_lik - old_log_lik)
        if verbose==True:
            print(log_lik)
        old_log_lik = log_lik

    # free up local variables 
    del rot_mat
    del disp
    torch.cuda.empty_cache()
    # return values
    return traj_tensor, avg, var

def torch_align_kronecker(traj_tensor, ref_tensor, precision_tensor, dtype=torch.float32, device=torch.device("cuda:0")):
    
    # make weighted ref
    weighted_ref = torch.matmul(precision_tensor,ref_tensor.to(torch.float64)).to(dtype)
    # get rotation matrices
    rot_mat = torch_align_rot_mat(traj_tensor, weighted_ref)
    # do rotation
    traj_tensor = torch.matmul(traj_tensor,rot_mat)
    # free up local variables 
    del weighted_ref
    del rot_mat
    torch.cuda.empty_cache()    
    # return aligned trajectory
    return traj_tensor

# determine the ln(det) of a singular matrix ignoring eigenvalues below threshold
def _torch_pseudo_lndet(sigma, EigenValueThresh=1e-10):
    e = torch.linalg.eigvalsh(sigma) 
    e = torch.where(e > EigenValueThresh, e, 1.0)
    lpdet = torch.sum(torch.log(e))
    return lpdet

def _torch_kronecker_log_lik(disp, precision, lpdet):
    # meta data
    n_frames = disp.shape[0]
    n_atoms = disp.shape[1]
    # compute log Likelihood for all points
    log_lik = torch.matmul(disp[:,:,0].view(n_frames,1,n_atoms),torch.matmul(precision,disp[:,:,0].view(n_frames,n_atoms,1)))[:,0,0]
    log_lik += torch.matmul(disp[:,:,1].view(n_frames,1,n_atoms),torch.matmul(precision,disp[:,:,1].view(n_frames,n_atoms,1)))[:,0,0]
    log_lik += torch.matmul(disp[:,:,2].view(n_frames,1,n_atoms),torch.matmul(precision,disp[:,:,2].view(n_frames,n_atoms,1)))[:,0,0]
    log_lik = torch.sum(log_lik)
    log_lik /= -2*n_frames
    log_lik -= 1.5 * lpdet
    return log_lik

def torch_iterative_align_kronecker(traj_tensor, stride=1000, dtype=torch.float32, device=torch.device("cuda:0"), thresh=1e-3, verbose=False):
    # meta data
    n_frames = traj_tensor.shape[0]
    n_atoms = traj_tensor.shape[1]
    
    # pass trajectory to device
    covar_norm = torch.tensor(1/(3*(n_frames-1)),dtype=torch.float64,device=device)
    
    # initialize with average as the first frame (arbitrary choice)
    weighted_avg = traj_tensor[0]
    
    delta_log_lik = thresh+10
    old_log_lik = 0
    while (delta_log_lik > thresh):
        # get rotation matrices
        rot_mat = torch_align_rot_mat(traj_tensor, weighted_avg)
        # do rotation
        traj_tensor = torch.matmul(traj_tensor,rot_mat)
        # compute new average
        avg = torch.mean(traj_tensor,0,False)
        disp = (traj_tensor - avg).to(torch.float64)
        # compute covar using strided data
        covar = torch.zeros((n_atoms,n_atoms),dtype=torch.float64,device=device)
        for frame in range(0,n_frames,stride):
            covar += torch.sum(torch.matmul(disp[frame:frame+stride],torch.transpose(disp[frame:frame+stride],1,2)),0)
        covar *= covar_norm
        # log likelihood
        precision = torch.linalg.pinv(covar, atol=1e-10, hermitian= True)
        lpdet = _torch_pseudo_lndet(covar)
        log_lik = _torch_kronecker_log_lik(disp, precision, lpdet)
        delta_log_lik = abs(log_lik - old_log_lik)
        if verbose == True:
            print(log_lik)
        old_log_lik = log_lik
        weighted_avg = torch.matmul(precision,avg.to(torch.float64)).to(dtype)
    # free up local variables 
    del rot_mat
    del covar
    del disp
    del weighted_avg
    torch.cuda.empty_cache()   
    # return values
    return traj_tensor, avg, precision, lpdet
    

def _torch_kronecker_weighted_log_lik(disp, weight_tensor, precision, lpdet):
    # meta data
    n_frames = disp.shape[0]
    n_atoms = disp.shape[1]
    # compute log Likelihood for all points
    log_lik = torch.matmul(disp[:,:,0].view(n_frames,1,n_atoms),torch.matmul(precision,disp[:,:,0].view(n_frames,n_atoms,1)))[:,0,0]
    log_lik += torch.matmul(disp[:,:,1].view(n_frames,1,n_atoms),torch.matmul(precision,disp[:,:,1].view(n_frames,n_atoms,1)))[:,0,0]
    log_lik += torch.matmul(disp[:,:,2].view(n_frames,1,n_atoms),torch.matmul(precision,disp[:,:,2].view(n_frames,n_atoms,1)))[:,0,0]
    log_lik = torch.sum(log_lik*weight_tensor)
    #log_lik = torch.sum(torch.matmul(torch.transpose(disp[:,:,0].view(n_frames,n_atoms,1),1,2),torch.matmul(precision,disp[:,:,0].view(n_frames,n_atoms,1)))*weight_tensor.view(-1,1,1),0)
    #log_lik += torch.sum(torch.matmul(torch.transpose(disp[:,:,1].view(n_frames,n_atoms,1),1,2),torch.matmul(precision,disp[:,:,1].view(n_frames,n_atoms,1)))*weight_tensor.view(-1,1,1),0)
    #log_lik += torch.sum(torch.matmul(torch.transpose(disp[:,:,2].view(n_frames,n_atoms,1),1,2),torch.matmul(precision,disp[:,:,2].view(n_frames,n_atoms,1)))*weight_tensor.view(-1,1,1),0)
    log_lik *= -0.5
    log_lik -= 1.5 * lpdet
    return log_lik

def torch_iterative_align_kronecker_weighted(traj_tensor, weight_tensor, ref_tensor=[], ref_precision_tensor=[], stride=1000, dtype=torch.float32, device=torch.device("cuda:0"), thresh=1e-3, verbose=False):
    # meta data
    n_frames = traj_tensor.shape[0]
    n_atoms = traj_tensor.shape[1]
    # ensure weights are normalized
    weight_tensor /= torch.sum(weight_tensor)
    # set ref
    if ref_tensor == []:
        # initialize with average as the first frame (arbitrary choice)
        weighted_avg = traj_tensor[0]
    else:
        weighted_avg = torch.matmul(ref_precision_tensor, ref_tensor.to(torch.float64)).to(dtype)
    # pass normalization value to device
    covar_norm = torch.tensor(1/3,dtype=torch.float64,device=device)
    
    delta_log_lik = thresh+10
    old_log_lik = 0
    while (delta_log_lik > thresh):
        # get rotation matrices
        rot_mat = torch_align_rot_mat(traj_tensor, weighted_avg)
        # do rotation
        traj_tensor = torch.matmul(traj_tensor,rot_mat)
        # compute new average
        avg = torch.sum((traj_tensor*weight_tensor.view(-1,1,1)),0,False)
        disp = (traj_tensor - avg).to(torch.float64)
        # compute covar using strided data
        covar = torch.zeros((n_atoms,n_atoms),dtype=torch.float64,device=device)
        for frame in range(0,n_frames,stride):
            covar += torch.sum(torch.matmul(disp[frame:frame+stride],torch.transpose(disp[frame:frame+stride],1,2))*weight_tensor[frame:frame+stride].view(-1,1,1),0)
        covar *= covar_norm
        # log likelihood
        precision = torch.linalg.pinv(covar, atol=1e-10, hermitian= True)
        lpdet = _torch_pseudo_lndet(covar)
        log_lik = _torch_kronecker_weighted_log_lik(disp, weight_tensor, precision, lpdet)
        delta_log_lik = abs(log_lik - old_log_lik)
        if verbose == True:
            print(log_lik)
        old_log_lik = log_lik
        weighted_avg = torch.matmul(precision, avg.to(torch.float64)).to(dtype)     
    # free up local variables 
    del rot_mat
    del covar
    del disp
    del weighted_avg
    torch.cuda.empty_cache()   
    # return values
    return traj_tensor, avg, precision, lpdet
    
