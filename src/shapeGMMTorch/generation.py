import numpy as np

def component_ids_from_rand(random_nums,weights):
    running_sum = 0.0
    component_ids = np.empty(random_nums.size,dtype=np.int32)
    for component_count, weight in enumerate(weights):
        component_ids[np.argwhere((random_nums > running_sum) &
                                (random_nums < running_sum + weight)).flatten()] = component_count
        running_sum += weight
    return component_ids

def gen_mv(mean, prec, n_samples=10000):
    # meta data
    n_atoms = mean.shape[0]
    e, v = np.linalg.eigh(prec)
    # compute stdev of each mode forcing first mode to have a zero
    stdev = np.zeros_like(e)
    stdev[1:] = np.sqrt(1/e[1:])
    # generate normally distributed random variables (mean 0, stdev 1)
    norms = np.random.normal(size=(n_atoms,n_samples*3))
    # multiply by normal mode stdev
    norms *= stdev.reshape((-1,1))
    # rotate back into original basis
    trj = np.dot(v,norms)
    # reshape trajectory to be (n_samples, n_atoms, 3)
    trj = trj.reshape(n_atoms,n_samples,3)
    trj = np.transpose(trj,(1,0,2))
    # add mean
    trj += mean
    return trj

def cov_from_prec(prec):
    e, v = np.linalg.eigh(prec)
    e[1:] = 1.0/e[1:]
    e[0] = 0
    covN = np.dot(v.T,np.dot(np.diag(e),v))
    return np.kron(covN,np.identity(3))

