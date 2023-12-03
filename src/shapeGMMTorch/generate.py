import numpy as np
from scipy.stats import multivariate_normal

def cluster_ids_from_rand(random_nums,weights):
    running_sum = 0.0
    cluster_ids = np.empty(random_nums.size,dtype=np.int32)
    for cluster_count, weight in enumerate(weights):
        cluster_ids[np.argwhere((random_nums > running_sum) &
                                (random_nums < running_sum + weight)).flatten()] = cluster_count
        running_sum += weight
    return cluster_ids

def gen_mv(cluster_mean, cluster_covariance, n_samples=10000):
    rv = multivariate_normal(mean=cluster_mean.flatten(), cov=cluster_covariance, allow_singular=True)
    trj = rv.rvs(size=n_samples).reshape(n_samples, -1, 3)
    return trj

def cov_from_prec(prec):
    e, v = np.linalg.eigh(prec)
    e[1:] = 1.0/e[1:]
    e[0] = 0
    covN = np.dot(v.T,np.dot(np.diag(e),v))
    return np.kron(covN,np.identity(3))

def generate(sgmm,n_frames):
    cluster_ids = cluster_ids_from_rand(np.random.rand(n_frames),sgmm.weights)
    trj = np.empty((n_frames,sgmm.n_atoms,3))
    for cluster_id in range(sgmm.n_clusters):
        indeces = np.argwhere(cluster_ids == cluster_id).flatten()
        trj[indeces] = gen_mv(sgmm.centers[cluster_id],cov_from_prec(sgmm.precisions[cluster_id]),indeces.size)
    return trj
