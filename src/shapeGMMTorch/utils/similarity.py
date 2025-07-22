import numpy as np
import sys
import torch
from scipy import stats
from ..core import ShapeGMM

def maha_dist2(x1, x2, weights):
    """
    Compute the squared Mahalabonis distance between positions x1 and x2 
    x1                      (required)  : float64 array with dimensions (n_atoms,3) of one molecular configuration
    x2                      (required)  : float64 array with dimensions (n_atoms,3) of another molecular configuration
    weights                 (required)  : float64 matrix with dimensions (n_atoms, n_atoms) of inverse (n_atoms, n_atoms) covariance
    """
    # zero distance
    dist = 0.0
    # compute squared distance as sum over indepdent (because covar is n_atoms x n_atoms) dimensions
    for i in range(3):
        disp = x1[:,i] - x2[:,i]
        dist += np.dot(disp,np.dot(weights,disp))
    # return squared distance
    return dist

def kl_divergence(sgmmP, sgmmQ, n_points):
    """
    Compute the Kullback-Leibler divergence, Dkl(P||Q), from sgmmQ (Q) to sgmmP (P) by sampling from sgmmP
    with n_points

    sgmmP             : reference shapeGMM object
    sgmmQ             : target shapeGMM object
    n_points          : (integer) number of frames to generate to estimate KL divergence

    returns:
    lnP - lnQ         : (float) KL divergence
    sterr(lnP - lnQ)  : (float) standard error of sampled KL divergence
    """
    trj = sgmmP.generate(n_points)
    lnP = sgmmP.score(trj)  # LL per frame 
    lnQ = sgmmQ.score(trj)  # LL per frame
    return lnP - lnQ, stats.sem(sgmmP.lnpdf(trj)-sgmmQ.lnpdf(trj))

def js_divergence(sgmmP, sgmmQ, n_points):
    """
    Compute the Jensen-Shannon divergence, JSD(P||Q), from sgmmQ (Q) to sgmmP (P) sampling using
    with n_points

    sgmmP      : reference shapeGMM object
    sgmmQ      : target shapeGMM object
    n_points   : (integer) number of frames to generate to estimate KL divergences

    returns:
    js         : (float) JS divergence normalized to be between 0 and 1
    sterr(js)  : (float) propagated error of sampled JS divergence
    """
    # create new M object that is 0.5(Q+P)
    assert sgmmP.covar_type == sgmmQ.covar_type, "covar_types do not match ... cannot compute JS divergence of mismatched covariance types"
    sgmmM = ShapeGMM(sgmmP.n_components+sgmmQ.n_components,covar_type=sgmmP.covar_type,device=torch.device("cpu"),dtype=torch.float64)
    sgmmM.weights_ = 0.5*np.append(sgmmP.weights_,sgmmQ.weights_)
    sgmmM.means_ = np.concatenate((sgmmP.means_,sgmmQ.means_))
    if sgmmM.covar_type == "kronecker":
        sgmmM.precisions_ = np.concatenate((sgmmP.precisions_,sgmmQ.precisions_))
        sgmmM.lpdets_ = np.concatenate((sgmmP.lpdets_,sgmmQ.lpdets_))
    else:
        sgmmM.vars_ = np.concatenate((sgmmP.vars_,sgmmQ.vars_))
    sgmmM.n_atoms = sgmmP.n_atoms
    sgmmM.is_fitted_ = True
    # now measure Kullback Leibler from M to P (or D(P||M))
    kl_P_M, kl_P_M_e  = kl_divergence(sgmmP,sgmmM,n_points)
    # now measure Kullback Leibler from M to Q (or D(Q||M))
    kl_Q_M, kl_Q_M_e = kl_divergence(sgmmQ,sgmmM,n_points)
    return 0.5/np.log(2)*(kl_P_M + kl_Q_M), 0.5/np.log(2)*np.sqrt(kl_P_M_e**2 + kl_Q_M_e**2)

def configurational_entropy(sgmmP, n_points):
    """
    Compute the configurational entropy of shapeGMM object sgmmP using sampling
    with n_points

    sgmmP      : shapeGMM object
    n_points   : (integer) number of frames to generate to sample

    returns:
    -lnP       : (float) configurational entropy
    sterr(lnP) : (float) standard error of sampled configurational entropy
    """
    # sample the object and compute probabilities of each sampled point
    trj = sgmmP.generate(n_points)
    lnP = sgmmP.score(trj)
    return -lnP, stats.sem(sgmmP.lnpdf(trj))

def _pinv(sigma):
    e, v = np.linalg.eigh(sigma)
    e[0] = 0.0
    e[1:] = 1/e[1:]
    return np.dot(v,np.dot(np.diag(e),v.T))

def _pinv_lnpdet(sigma):
    e, v = np.linalg.eigh(sigma)
    e[0] = 0.0
    lnpdet = np.sum(np.log(e[1:]))
    e[1:] = 1/e[1:]
    return np.dot(v,np.dot(np.diag(e),v.T)), lnpdet

def _weight_kabsch_log_lik(x, mu, precision, lpdet):
    # meta data
    n_frames = x.shape[0]
    # compute log Likelihood for all points
    log_lik = 0.0
    for i in range(n_frames):
        for j in range(3):
            disp = x[i,:,j] - mu[:,j]
            log_lik += np.dot(disp,np.dot(precision,disp))
    log_lik += 3 * n_frames * lpdet
    log_lik *= -0.5
    return log_lik

def _kabsch_rotate(mobile, target):
    correlation_matrix = np.dot(np.transpose(mobile), target)
    V, S, W_tr = np.linalg.svd(correlation_matrix)
    if np.linalg.det(V) * np.linalg.det(W_tr) < 0.0:
        V[:, -1] = -V[:, -1]
    rotation = np.dot(V, W_tr)
    mobile_prime = np.dot(mobile,rotation)
    return mobile_prime

def _iterative_average_precision_weighted_kabsch(traj_data,precision,lpdet,thresh=1E-3,max_steps=300):
    # trajectory metadata
    n_frames = traj_data.shape[0]
    n_atoms = traj_data.shape[1]
    nDim = traj_data.shape[2]
    # Initialize with first frame
    avg = traj_data[0]
    aligned_pos = np.copy(traj_data)
    # compute log likelihood
    log_lik = _weight_kabsch_log_lik(aligned_pos, avg, precision, lpdet)
    # perform iterative alignment and average to converge average
    log_lik_diff = 10+thresh
    step = 0
    while log_lik_diff > thresh and step < max_steps:
        # rezero new average
        new_avg = np.zeros((n_atoms,nDim),dtype=np.float64)
        # align trajectory to average and accumulate new average
        weights_target = np.dot(precision,avg)
        for ts in range(n_frames):
            aligned_pos[ts] = _kabsch_rotate(aligned_pos[ts], weights_target)
            new_avg += aligned_pos[ts]
        # finish average
        new_avg /= n_frames
        # compute log likelihood
        new_log_lik = _weight_kabsch_log_lik(aligned_pos, new_avg, precision, lpdet)
        log_lik_diff = np.abs(new_log_lik-log_lik)
        log_lik = new_log_lik
        avg = np.copy(new_avg)
        step += 1
    return aligned_pos

def bhattacharyya_distance(sgmm1,component_id1,sgmm2,component_id2):
    """
    Compute the Bhattacharya distance between two multivariate Gaussians

    sgmm1       : first shapeGMM object
    component_id1 : component id of MV Gaussian in first shapeGMM object
    sgmm2       : second shapeGMM object
    component_id2 : component id of MV Gaussian in second shapeGMM object

    returns:
    D           : (float) Bhattacharya distance
    """
    assert sgmm1.covar_type == sgmm2.covar_type, "covar_types do not match ... cannot compute Bhattacharya distance between objects with mismatched covariance types"
    if sgmm1.covar_type == "kronecker":
        sigma = 0.5*(_pinv(sgmm1.precisions_[component_id1]) + _pinv(sgmm2.precisions_[component_id2]))
        lnpdet1 = sgmm1.lpdets_[component_id1]
        lnpdet2 = sgmm2.lpdets_[component_id2]
    else: # uniform
        sigma = 0.5*(sgmm1.vars_[component_id1] + sgmm2.vars_[component_id2])*np.eye(sgmm1.n_atoms)
        lnpdet1 = sgmm1.n_atoms * np.log(sgmm1.vars_[component_id1])
        lnpdet2 = sgmm2.n_atoms * np.log(sgmm2.vars_[component_id2])
    prec, lnpdet = _pinv_lnpdet(sigma)
    traj = np.empty((2,sgmm1.n_atoms,3))
    traj[0] = sgmm1.means_[component_id1]
    traj[1] = sgmm2.means_[component_id2]
    traj = _iterative_average_precision_weighted_kabsch(traj,prec,lnpdet)
    D = maha_dist2(traj[0],traj[1],prec)
    D /= 8
    D += 0.5*lnpdet
    D -= 0.25 * (lnpdet1 + lnpdet2)
    return D

