import numpy as np
import torch
from torch_batch_svd import svd

def torch_remove_center_of_geometry(traj_tensor, dtype=torch.float32, device=torch.device("cuda:0")):
    # meta data
    n_frames = traj_tensor.shape[0]
    # compute geometric center of each frame
    cog = torch.mean(traj_tensor,1,False)
    # substract from each frame
    for i in range(n_frames):
        traj_tensor[i] -= cog[i]
    # free up local variables 
    del cog
    torch.cuda.empty_cache()

def torch_sd(traj_tensor, ref_tensor, dtype=torch.float32):
    # meta data
    n_atoms = traj_tensor.shape[1]
    # compute correlation matrices using batched matmul
    c_mats = torch.matmul(ref_tensor.T,traj_tensor)
    # perfrom SVD of c_mats using batched SVD
    u, s, v = svd(c_mats)
    #u, s, v = torch.linalg.svd(c_mats)
    # ensure true rotation by correcting sign of determinant
    prod_dets = torch.linalg.det(u)*torch.linalg.det(v)
    u[:,:,-1] *= prod_dets.view(-1,1)
    rot_mat = torch.matmul(v,torch.transpose(u,1,2))
    #rot_mat = torch.transpose(torch.matmul(u,v),1,2)
    # do rotation
    traj_tensor = torch.matmul(traj_tensor,rot_mat)
    disp = (traj_tensor - ref_tensor).to(torch.float64)
    sd = torch.matmul(disp.view(-1,1,n_atoms*3),disp.view(-1,n_atoms*3,1))[:,0,0].to(dtype)
    return sd
    # free up local variables 
    del c_mats
    del u
    del s
    del v
    del prod_dets
    del rot_mat
    del disp
    torch.cuda.empty_cache()    
    

def torch_align_uniform(traj_tensor, ref_tensor):

    # compute correlation matrices using batched matmul
    c_mats = torch.matmul(ref_tensor.T,traj_tensor)
    # perfrom SVD of c_mats using batched SVD
    u, s, v = svd(c_mats)
    #u, s, v = torch.linalg.svd(c_mats)
    # ensure true rotation by correcting sign of determinant
    prod_dets = torch.linalg.det(u)*torch.linalg.det(v)
    u[:,:,-1] *= prod_dets.view(-1,1)
    rot_mat = torch.matmul(v,torch.transpose(u,1,2))
    #rot_mat = torch.transpose(torch.matmul(u,v),1,2)
    # do rotation
    traj_tensor = torch.matmul(traj_tensor,rot_mat)
    return traj_tensor
    # free up local variables 
    del c_mats
    del u
    del s
    del v
    del prod_dets
    del rot_mat
    torch.cuda.empty_cache()


def _torch_uniform_log_likelihood(disp, n_frames, n_atoms, var_norm, log_lik_prefactor):
    # reshape displacement 
    disp = disp.view(n_frames*n_atoms*3,1)
    # compute variance
    var = torch.sum(disp*disp)
    var *= var_norm
    # compute log likelihood (per frame)
    log_lik = log_lik_prefactor*(torch.log(var) + 1)
    return log_lik, var

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
        # compute correlation matrices using batched matmul
        c_mats = torch.matmul(avg.T,traj_tensor)
        # perfrom SVD of c_mats using batched SVD
        #u, s, v = torch.linalg.svd(c_mats)
        u, s, v = svd(c_mats)
        # ensure true rotation by correcting sign of determinant
        prod_dets = torch.linalg.det(u)*torch.linalg.det(v)
        u[:,:,-1] *= prod_dets.view(-1,1)
        #rot_mat = torch.transpose(torch.matmul(u,v),1,2)
        rot_mat = torch.matmul(v,torch.transpose(u,1,2))
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
    
    return traj_tensor, avg, var
    # free up local variables 
    del c_mats
    del u
    del s
    del v
    del prod_dets
    del rot_mat
    del disp
    torch.cuda.empty_cache()


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
        # compute correlation matrices using batched matmul
        c_mats = torch.matmul(avg.T,traj_tensor)
        # perfrom SVD of c_mats using batched SVD
        #u, s, v = torch.linalg.svd(c_mats)
        u, s, v = svd(c_mats)
        # ensure true rotation by correcting sign of determinant
        prod_dets = torch.linalg.det(u)*torch.linalg.det(v)
        u[:,:,-1] *= prod_dets.view(-1,1)
        #rot_mat = torch.transpose(torch.matmul(u,v),1,2)
        rot_mat = torch.matmul(v,torch.transpose(u,1,2))
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

    return traj_tensor, avg, var
    # free up local variables 
    del c_mats
    del u
    del s
    del v
    del prod_dets
    del rot_mat
    del disp
    torch.cuda.empty_cache()

def torch_align_kronecker(traj_tensor, ref_tensor, precision_tensor, dtype=torch.float32, device=torch.device("cuda:0")):
    
    # make weighted ref
    weighted_avg = torch.matmul(ref_tensor.T,precision_tensor).to(dtype)
    
    # compute correlation matrices using batched matmul
    c_mats = torch.matmul(weighted_avg,traj_tensor)
    # perfrom SVD of c_mats using batched SVD
    u, s, v = svd(c_mats)
    # ensure true rotation by correcting sign of determinant 
    prod_dets = torch.linalg.det(u)*torch.linalg.det(v)
    u[:,:,-1] *= prod_dets.view(-1,1)
    rot_mat = torch.matmul(v,torch.transpose(u,1,2))
    # do rotation
    traj_tensor = torch.matmul(traj_tensor,rot_mat)
    return traj_tensor
    # free up local variables 
    del c_mats
    del u
    del s
    del v
    del prod_dets
    del rot_mat
    torch.cuda.empty_cache()    

def _torch_pseudo_inv(sigma, dtype=torch.float64, device=torch.device("cuda:0"),EigenValueThresh=1e-10):
    N = sigma.shape[0]
    e, v = torch.linalg.eigh(sigma)
    pinv = torch.zeros(sigma.shape,dtype=dtype,device=device)
    for i in range(N):
        if (e[i] > EigenValueThresh):
            pinv += 1.0/e[i]*torch.outer(v[:,i],v[:,i])
    return pinv
    
    
def _torch_pseudo_inv_lndet(sigma, dtype=torch.float64, device=torch.device("cuda:0"),EigenValueThresh=1e-10):
    N = sigma.shape[0]
    e, v = torch.linalg.eigh(sigma)
    pinv = torch.zeros(sigma.shape,dtype=dtype,device=device)
    lpdet = torch.tensor(0.0,dtype=dtype,device=device)
    for i in range(N):
        if (e[i] > EigenValueThresh):
            lpdet += torch.log(e[i])
            pinv += 1.0/e[i]*torch.outer(v[:,i],v[:,i])
    return pinv, lpdet

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
    log_lik = torch.sum(torch.matmul(torch.transpose(disp[:,:,0].view(n_frames,n_atoms,1),1,2),torch.matmul(precision,disp[:,:,0].view(n_frames,n_atoms,1))),0)
    log_lik += torch.sum(torch.matmul(torch.transpose(disp[:,:,1].view(n_frames,n_atoms,1),1,2),torch.matmul(precision,disp[:,:,1].view(n_frames,n_atoms,1))),0)
    log_lik += torch.sum(torch.matmul(torch.transpose(disp[:,:,2].view(n_frames,n_atoms,1),1,2),torch.matmul(precision,disp[:,:,2].view(n_frames,n_atoms,1))),0)
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
    weighted_avg = traj_tensor[0].T
    
    delta_log_lik = thresh+10
    old_log_lik = 0
    while (delta_log_lik > thresh):
        # compute correlation matrices using batched matmul
        c_mats = torch.matmul(weighted_avg,traj_tensor)
        # perfrom SVD of c_mats using batched SVD
        #u, s, v = torch.linalg.svd(c_mats)
        u, s, v = svd(c_mats)
        # ensure true rotation by correcting sign of determinant 
        prod_dets = torch.linalg.det(u)*torch.linalg.det(v)
        u[:,:,-1] *= prod_dets.view(-1,1)
        #rot_mat = torch.transpose(torch.matmul(u,v),1,2)
        rot_mat = torch.matmul(v,torch.transpose(u,1,2))
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
        log_lik = _torch_kronecker_log_lik(disp, precision, lpdet)[0][0]
        delta_log_lik = abs(log_lik - old_log_lik)
        if verbose == True:
            print(log_lik)
        old_log_lik = log_lik
        weighted_avg = torch.matmul(avg.T.to(torch.float64),precision).to(dtype)
    return traj_tensor, avg, precision, lpdet
    # free up local variables 
    del c_mats
    del u
    del s
    del v
    del prod_dets
    del rot_mat
    del covar
    del disp
    del weighted_avg
    torch.cuda.empty_cache()   
    

def _torch_kronecker_weighted_log_lik(disp, weight_tensor, precision, lpdet):
    # meta data
    n_frames = disp.shape[0]
    n_atoms = disp.shape[1]
    # compute log Likelihood for all points
    log_lik = torch.sum(torch.matmul(torch.transpose(disp[:,:,0].view(n_frames,n_atoms,1),1,2),torch.matmul(precision,disp[:,:,0].view(n_frames,n_atoms,1)))*weight_tensor.view(-1,1,1),0)
    log_lik += torch.sum(torch.matmul(torch.transpose(disp[:,:,1].view(n_frames,n_atoms,1),1,2),torch.matmul(precision,disp[:,:,1].view(n_frames,n_atoms,1)))*weight_tensor.view(-1,1,1),0)
    log_lik += torch.sum(torch.matmul(torch.transpose(disp[:,:,2].view(n_frames,n_atoms,1),1,2),torch.matmul(precision,disp[:,:,2].view(n_frames,n_atoms,1)))*weight_tensor.view(-1,1,1),0)
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
        weighted_avg = traj_tensor[0].T
    else:
        weighted_avg = torch.matmul(ref_tensor.T,ref_precision_tensor).to(dtype)
    # pass normalization value to device
    covar_norm = torch.tensor(1/3,dtype=torch.float64,device=device)
    
    delta_log_lik = thresh+10
    old_log_lik = 0
    while (delta_log_lik > thresh):
        # compute correlation matrices using batched matmul
        c_mats = torch.matmul(weighted_avg,traj_tensor)
        # perfrom SVD of c_mats using batched SVD
        #u, s, v = torch.linalg.svd(c_mats)
        u, s, v = svd(c_mats)
        # ensure true rotation by correcting sign of determinant 
        prod_dets = torch.linalg.det(u)*torch.linalg.det(v)
        u[:,:,-1] *= prod_dets.view(-1,1)
        #rot_mat = torch.transpose(torch.matmul(u,v),1,2)
        rot_mat = torch.matmul(v,torch.transpose(u,1,2))
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
        log_lik = _torch_kronecker_weighted_log_lik(disp, weight_tensor, precision, lpdet)[0][0]
        delta_log_lik = abs(log_lik - old_log_lik)
        if verbose == True:
            print(log_lik)
        old_log_lik = log_lik
        weighted_avg = torch.matmul(avg.T.to(torch.float64),precision).to(dtype)     
    return traj_tensor, avg, precision, lpdet
    # free up local variables 
    del c_mats
    del u
    del s
    del v
    del prod_dets
    del rot_mat
    del covar
    del disp
    del weighted_avg
    torch.cuda.empty_cache()   
    
