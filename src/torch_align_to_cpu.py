import numpy as np
import torch
from torch_batch_svd import svd

def torch_remove_center_of_geometry(traj, dtype=torch.float32, device=torch.device("cuda:0")):
    # meta data
    n_frames = traj.shape[0]
    n_atoms = traj.shape[1]

    # pass trajectory to device
    traj_tensor = torch.tensor(traj,dtype=dtype,device=device)

    cog = torch.mean(traj_tensor.to(torch.float64),1,False)
    for i in range(n_frames):
        traj_tensor[i] -= cog[i]

    return traj_tensor.cpu().numpy()

def torch_align_uniform(traj, ref, dtype=torch.float32, device=torch.device("cuda:0")):
    # meta data
    n_frames = traj.shape[0]
    n_atoms = traj.shape[1]

    # pass trajectory to device
    traj_tensor = torch.tensor(traj,dtype=dtype,device=device)
    ref_tensor = torch.tensor(ref,dtype=dtype,device=device)

    # compute correlation matrices using batched matmul
    c_mats = torch.matmul(avg.T,traj_tensor)
    # perfrom SVD of c_mats using batched SVD
    u, s, v = svd(c_mats)
    # ensure true rotation by correcting sign of determinant
    prod_dets = torch.linalg.det(u)*torch.linalg.det(v)
    u[:,0,-1] *= prod_dets
    u[:,1,-1] *= prod_dets
    u[:,2,-1] *= prod_dets
    rot_mat = torch.transpose(torch.matmul(u,torch.transpose(v,1,2)),1,2)
    # do rotation
    traj_tensor = torch.matmul(traj_tensor,rot_mat)

    return traj_tensor.cpu().numpy()


def _torch_uniform_log_likelihood(disp):
    # meta data
    n_frames = disp.shape[0]
    n_atoms = disp.shape[1]
    # reshape displacement 
    disp = torch.reshape(disp,(n_frames*n_atoms*3,1))
    # compute variance
    var = torch.sum(disp*disp).cpu().numpy()
    var /= n_frames*3*(n_atoms-1)
    # compute log likelihood (per frame)
    log_lik = -1.5*(n_atoms-1)*(np.log(var) + 1)
    return log_lik, var

def torch_iterative_align_uniform(traj, dtype=torch.float32, device=torch.device("cuda:0"), thresh=1e-3, verbose=False):
    # meta data
    n_frames = traj.shape[0]
    n_atoms = traj.shape[1]

    # pass trajectory to device
    traj_tensor = torch.tensor(traj,dtype=dtype,device=device)

    # initialize with average as the first frame (arbitrary choice)
    avg = traj_tensor[0]

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
        u[:,0,-1] *= prod_dets
        u[:,1,-1] *= prod_dets
        u[:,2,-1] *= prod_dets
        #rot_mat = torch.transpose(torch.matmul(u,v),1,2)
        rot_mat = torch.transpose(torch.matmul(u,torch.transpose(v,1,2)),1,2)
        # do rotation
        traj_tensor = torch.matmul(traj_tensor,rot_mat)
        # compute new average
        avg = torch.mean(traj_tensor,0,False)
        disp = (traj_tensor - avg).to(torch.float64)
        # compute log likelihood and variance
        log_lik, var = _torch_uniform_log_likelihood(disp)
        delta_log_lik = abs(log_lik - old_log_lik)
        if verbose==True:
            print(log_lik)
        old_log_lik = log_lik

    return avg.cpu().numpy(), traj_tensor.cpu().numpy(), var


def torch_iterative_align_uniform_weighted(traj, weights, dtype=torch.float32, device=torch.device("cuda:0"), thresh=1e-3, verbose=False):
    # meta data
    n_frames = traj.shape[0]
    n_atoms = traj.shape[1]

    # pass trajectory to device
    traj_tensor = torch.tensor(traj,dtype=dtype,device=device)
    weight_tensor = torch.tensor(weights,dtype=dtype,device=device)

    # initialize with average as the first frame (arbitrary choice)
    avg = traj_tensor[0]

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
        u[:,0,-1] *= prod_dets
        u[:,1,-1] *= prod_dets
        u[:,2,-1] *= prod_dets
        #rot_mat = torch.transpose(torch.matmul(u,v),1,2)
        rot_mat = torch.transpose(torch.matmul(u,torch.transpose(v,1,2)),1,2)
        # do rotation
        traj_tensor = torch.matmul(traj_tensor,rot_mat)
        # compute new average
        avg = torch.sum((traj_tensor * weight_tensor.view(-1,1,1)),0,False)
        # compute displacement
        disp = (traj_tensor - avg).to(torch.float64) 
        # reshape displacement 
        disp = torch.reshape(disp* disp * weight_tensor.view(-1,1,1),(n_frames*n_atoms*3,1))
        # compute variance
        var = torch.sum(disp).cpu().numpy()
        var /= 3*(n_atoms-1)
        # compute log likelihood (per frame)
        log_lik = -1.5*(n_atoms-1)*(np.log(var) + 1)
        delta_log_lik = abs(log_lik - old_log_lik)
        if verbose==True:
            print(log_lik)
        old_log_lik = log_lik

    return avg.cpu().numpy(), traj_tensor.cpu().numpy(), var

def torch_align_kronecker(traj, ref, precision, dtype=torch.float32, device=torch.device("cuda:0")):
    # meta data
    n_frames = traj.shape[0]
    n_atoms = traj.shape[1]
    
    # pass trajectory to device
    traj_tensor = torch.tensor(traj, dtype=dtype, device=device)
    ref_tensor = torch.tensor(ref, dtype=dtype, device=device)
    precision_tensor = torch.tensor(precision, device=device)
    
    # make weighted ref
    weighted_avg = torch.matmul(ref_tensor.T,precision_tensor).to(dtype)
    
    # compute correlation matrices using batched matmul
    c_mats = torch.matmul(weighted_avg,traj_tensor)
    # perfrom SVD of c_mats using batched SVD
    u, s, v = svd(c_mats)
    # ensure true rotation by correcting sign of determinant 
    prod_dets = torch.linalg.det(u)*torch.linalg.det(v)
    u[:,0,-1] *= prod_dets
    u[:,1,-1] *= prod_dets
    u[:,2,-1] *= prod_dets
    rot_mat = torch.transpose(torch.matmul(u,torch.transpose(v,1,2)),1,2)
    # do rotation
    traj_tensor = torch.matmul(traj_tensor,rot_mat)
        

    return traj_tensor.cpu().numpy()



def _torch_pseudo_inv(sigma, dtype=torch.float64, device=torch.device("cuda:0"),EigenValueThresh=1e-10):
    N = sigma.shape[0]
    e, v = torch.linalg.eigh(sigma)
    pinv = torch.tensor(np.zeros(sigma.shape),dtype=dtype,device=device)
    lpdet = torch.tensor(0.0,dtype=dtype,device=device)
    for i in range(N):
        if (e[i] > EigenValueThresh):
            lpdet += torch.log(e[i])
            pinv += 1.0/e[i]*torch.outer(v[:,i],v[:,i])
    return pinv, lpdet

def _torch_kronecker_log_lik(disp, precision, lpdet):
    # meta data
    n_frames = disp.shape[0]
    n_atoms = disp.shape[1]
    # compute log Likelihood for all points
    #log_lik = torch.trace(torch.sum(torch.matmul(torch.transpose(disp,1,2),torch.matmul(precision,disp)),0))
    log_lik = torch.sum(torch.matmul(torch.transpose(torch.reshape(disp[:,:,0],(n_frames,n_atoms,1)),1,2),torch.matmul(precision,torch.reshape(disp[:,:,0],(n_frames,n_atoms,1)))),0)
    log_lik += torch.sum(torch.matmul(torch.transpose(torch.reshape(disp[:,:,1],(n_frames,n_atoms,1)),1,2),torch.matmul(precision,torch.reshape(disp[:,:,1],(n_frames,n_atoms,1)))),0)
    log_lik += torch.sum(torch.matmul(torch.transpose(torch.reshape(disp[:,:,2],(n_frames,n_atoms,1)),1,2),torch.matmul(precision,torch.reshape(disp[:,:,2],(n_frames,n_atoms,1)))),0)
    log_lik /= -2*n_frames
    log_lik -= 1.5 * lpdet
    return log_lik

def torch_iterative_align_kronecker(traj, stride=1000, dtype=torch.float32, device=torch.device("cuda:0"), thresh=1e-3, verbose=False):
    # meta data
    n_frames = traj.shape[0]
    n_atoms = traj.shape[1]
    
    # pass trajectory to device
    traj_tensor = torch.tensor(traj, dtype=dtype, device=device)
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
        u[:,0,-1] *= prod_dets
        u[:,1,-1] *= prod_dets
        u[:,2,-1] *= prod_dets
        #rot_mat = torch.transpose(torch.matmul(u,v),1,2)
        rot_mat = torch.transpose(torch.matmul(u,torch.transpose(v,1,2)),1,2)
        # do rotation
        traj_tensor = torch.matmul(traj_tensor,rot_mat)
        # compute new average
        avg = torch.mean(traj_tensor.to(torch.float64),0,False)
        disp = traj_tensor.to(torch.float64) - avg
        # compute covar using strided data
        covar = torch.tensor(np.zeros((n_atoms,n_atoms)),dtype=torch.float64,device=device)
        for frame in range(0,n_frames,stride):
            covar += torch.sum(torch.matmul(disp[frame:frame+stride],torch.transpose(disp[frame:frame+stride],1,2)),0)
        covar *= covar_norm
        # log likelihood
        precision, lpdet = _torch_pseudo_inv(covar,dtype=torch.float64,device=device)
        log_lik = _torch_kronecker_log_lik(disp, precision, lpdet).cpu().numpy()[0][0]
        delta_log_lik = abs(log_lik - old_log_lik)
        if verbose == True:
            print(log_lik)
        old_log_lik = log_lik
        weighted_avg = torch.matmul(avg.T,precision).to(dtype)

    return avg.cpu().numpy(), traj_tensor.cpu().numpy(), precision.cpu().numpy(), lpdet.cpu().numpy()

def _torch_kronecker_weighted_log_lik(disp, weight_tensor, precision, lpdet):
    # meta data
    n_frames = disp.shape[0]
    n_atoms = disp.shape[1]
    # compute log Likelihood for all points
    #log_lik = torch.trace(torch.sum(torch.matmul(torch.transpose(disp,1,2),torch.matmul(precision,disp)),0))
    log_lik = torch.sum(torch.matmul(torch.transpose(torch.reshape(disp[:,:,0],(n_frames,n_atoms,1)),1,2),torch.matmul(precision,torch.reshape(disp[:,:,0],(n_frames,n_atoms,1))))*weight_tensor.view(-1,1,1),0)
    log_lik += torch.sum(torch.matmul(torch.transpose(torch.reshape(disp[:,:,1],(n_frames,n_atoms,1)),1,2),torch.matmul(precision,torch.reshape(disp[:,:,1],(n_frames,n_atoms,1))))*weight_tensor.view(-1,1,1),0)
    log_lik += torch.sum(torch.matmul(torch.transpose(torch.reshape(disp[:,:,2],(n_frames,n_atoms,1)),1,2),torch.matmul(precision,torch.reshape(disp[:,:,2],(n_frames,n_atoms,1))))*weight_tensor.view(-1,1,1),0)
    log_lik *= -0.5
    log_lik -= 1.5 * lpdet
    return log_lik

def torch_iterative_align_kronecker_weighted(traj, weights, stride=1000, dtype=torch.float32, device=torch.device("cuda:0"), thresh=1e-3, verbose=False):
    # meta data
    n_frames = traj.shape[0]
    n_atoms = traj.shape[1]
    
    # pass trajectory to device
    traj_tensor = torch.tensor(traj, dtype=dtype, device=device)
    weight_tensor = torch.tensor(weights,dtype=dtype,device=device)
    covar_norm = torch.tensor(1/3,dtype=torch.float64,device=device)
    
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
        u[:,0,-1] *= prod_dets
        u[:,1,-1] *= prod_dets
        u[:,2,-1] *= prod_dets
        #rot_mat = torch.transpose(torch.matmul(u,v),1,2)
        rot_mat = torch.transpose(torch.matmul(u,torch.transpose(v,1,2)),1,2)
        # do rotation
        traj_tensor = torch.matmul(traj_tensor,rot_mat)
        # compute new average
        avg = torch.sum((traj_tensor*weight_tensor.view(-1,1,1)).to(torch.float64),0,False)
        disp = traj_tensor.to(torch.float64) - avg
        # compute covar using strided data
        covar = torch.tensor(np.zeros((n_atoms,n_atoms)),dtype=torch.float64,device=device)
        for frame in range(0,n_frames,stride):
            covar += torch.sum(torch.matmul(disp[frame:frame+stride],torch.transpose(disp[frame:frame+stride],1,2))*weight_tensor[frame:frame+stride].view(-1,1,1),0)
        covar *= covar_norm
        # log likelihood
        precision, lpdet = _torch_pseudo_inv(covar,dtype=torch.float64,device=device)
        log_lik = _torch_kronecker_weighted_log_lik(disp, weight_tensor, precision, lpdet).cpu().numpy()[0][0]
        delta_log_lik = abs(log_lik - old_log_lik)
        if verbose == True:
            print(log_lik)
        old_log_lik = log_lik
        weighted_avg = torch.matmul(avg.T,precision).to(dtype)

    return avg.cpu().numpy(), traj_tensor.cpu().numpy(), precision.cpu().numpy(), lpdet.cpu().numpy()

