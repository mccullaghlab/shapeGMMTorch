import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')
import random
# the following are local libraries
from . import torch_align
from . import torch_uniform_sgmm_lib
from . import torch_kronecker_sgmm_lib
#import torch_align
#import torch_uniform_sgmm_lib
#import torch_kronecker_sgmm_lib

# class
class ShapeGMMTorch:
    """
    ShapeGMMTorch is a class that can be used to perform Gaussian Mixture Model clustering in size-and-shape space.
    The class is designed to mimic similar clustering methods implemented in sklearn.  The model is first initialized
    and then fit with supplied data.  Fit parameters for the model include average structures and (co)variances.
    Once fit, the model can be used to predict clustering on an alternative (but same feature space size) data set.
    Author: Martin McCullagh
    Date: 8/9/2022
    """
    def __init__(self, n_clusters, log_thresh=1E-3, max_steps=200, covar_type="uniform", init_cluster_method="random", init_iter=5, kabsch_thresh=1E-1, kabsch_max_steps=500, dtype=torch.float32, device=torch.device("cuda:0"), verbose=False):
        """
        Initialize size-and-shape GMM.
        n_clusters (required)   - integer number of clusters must be input
        log_thresh              - float threshold in log likelihood difference to determine convergence. Default value is 1e-3.
        max_steps               - integer maximum number of steps that the GMM procedure will do.  Default is 200.
        covar_type              - string defining the covariance type.  Options are 'kronecker' and 'uniform'.  Defualt is 'uniform'.
        init_cluster_method     - string dictating how to initialize clusters.  Understood values are 'chunk', 'read' and 'random'.  Default is 'random'.
        init_iter               - integer dictating number of iterations done to initialize for 'random'.  Default is 5.
        kabsch_thresh           - float dictating convergence criteria for each alignment step.  Default value is 1e-1.
        kabsch_max_steps        - integer dictating maximum number of allowed iterations in each alignment step. Default is 500.
        dtype                   - Data type to be used.  Default is torch.float32.
        device                  - device to be used.  Default is torch.device('cuda:0') device.
        verbose                 - boolean dictating whether to print various things at every step. Defulat is False.
        """
        
        self.n_clusters = n_clusters                            # integer
        self.log_thresh = log_thresh                            # float
        self.max_steps = max_steps                              # integer
        self.covar_type = covar_type                            # string
        self.init_cluster_method = init_cluster_method          # string
        self.init_iter = init_iter                              # integer
        self.kabsch_thresh = kabsch_thresh                      # float
        self.kabsch_max_steps = kabsch_max_steps                # integer
        self.dtype = dtype                                      # torch dtype
        self.device = device                                    # torch device
        self.verbose = verbose                                  # boolean
        self.init_clusters_flag = False                         # boolean tracking if clusters have been initialized or not.
        self.gmm_fit_flag = False                               # boolean tracking if GMM has been fit.

    # initialize clusters
    def init_clusters(self, traj_tensor, clusters=[]):
        
        # get metadata
        self.n_frames = int(traj_tensor.shape[0])
        self.n_atoms = traj_tensor.shape[1]
        self.n_dim = traj_tensor.shape[2]
        self.n_features = self.n_dim*self.n_atoms
        if (self.verbose == True):
            # print metadata to stdout
            print("Number of frames being analyzed:", self.n_frames)
            print("Number of particles being analyzed:", self.n_atoms)
            print("Number of dimensions (must be 3):", self.n_dim)
            print("Initializing clustering using method:", self.init_cluster_method)
        # declare clusters
        self.clusters = np.zeros(self.n_frames,dtype=np.int)

        # Remove the center-of-geometry from entire trajectory
        torch_align.torch_remove_center_of_geometry(traj_tensor)

        # make initial clustering based on input user choice (default is random)
        if (self.init_cluster_method == "chunk"):
            for i in range(self.n_frames):
                self.clusters[i] = i*self.n_clusters // self.n_frames
        elif (self.init_cluster_method == "read"):
            # should affirm that there are n_frames clusters
            self.clusters = clusters
        else: # default is random
            for i in range(self.init_iter):
                init_clusters = torch_uniform_sgmm_lib.init_random(traj_tensor,self.n_clusters,dtype=self.dtype,device=self.device)
                log_lik = torch_uniform_sgmm_lib.uniform_sgmm_log_likelihood(traj_tensor,init_clusters,dtype=self.dtype, device=self.device).cpu().numpy()
                if (i==0 or log_lik > max_log_lik):
                    max_log_lik = log_lik
                    self.clusters = init_clusters
        # clusters have been initialized
        self.init_clusters_flag = True
        # 

        
    # fit
    def fit(self, traj_data, clusters = []):
        
        # pass trajectory data to device
        traj_tensor = torch.tensor(traj_data,dtype=self.dtype,device=self.device)
        # Initialize clusters if they have not been already
        if (self.init_clusters_flag == False):
            self.init_clusters(traj_tensor, clusters)
            
        # declare some important arrays for the model
        centers_tensor = torch.empty((self.n_clusters,self.n_atoms,self.n_dim),dtype=self.dtype,device=self.device)
        self.weights = np.empty(self.n_clusters,dtype=np.float64)
        # uniform/weighted specific variables
        if self.covar_type == 'uniform': 
            vars_tensor = torch.empty(self.n_clusters,dtype=torch.float64,device=self.device)
        else: # assume Kronecker
            # declare precision matrices (inverse covariances)
            precisions_tensor = torch.empty((self.n_clusters,self.n_atoms,self.n_atoms),dtype=torch.float64,device=self.device)
            # declare array for log determinants for each clusters
            lpdets_tensor = torch.empty(self.n_clusters, dtype=torch.float64,device=self.device)

        # compute average and covariance of initial clustering
        for k in range(self.n_clusters):
            indeces = np.argwhere(self.clusters == k).flatten()
            # initialize weights as populations of clusters
            self.weights[k] = indeces.size/self.n_frames
            if self.covar_type == 'uniform':
                centers_tensor[k], vars_tensor[k] = torch_align.torch_iterative_align_uniform(traj_tensor[indeces],thresh=self.kabsch_thresh,device=self.device,dtype=self.dtype)[1:]
            else:
                centers_tensor[k], precisions_tensor[k], lpdets_tensor[k] = torch_align.torch_iterative_align_kronecker(traj_tensor[indeces],thresh=self.kabsch_thresh,device=self.device,dtype=self.dtype)[1:]        
        if (self.verbose == True):
            print("Weights from initial clusters in fit:", self.weights)
    
        # pass remaining data to device
        ln_weights_tensor = torch.tensor(np.log(self.weights),dtype=torch.float64,device=self.device)
        
        # Determine initial log likelihood
        #if (self.verbose == True):
        #    log_likelihood = torch_uniform_sgmm_lib.uniform_sgmm_log_likelihood(traj_tensor,self.clusters, thresh=self.kabsch_thresh, dtype=self.dtype, device=self.device)
        #    print("Initial Uniform log likelihood:", log_likelihood.cpu().numpy())
        
        # perform Expectation Maximization
        delta_log_lik = 100.0 + self.log_thresh
        step = 0
        while step < self.max_steps and delta_log_lik > self.log_thresh:
            # Expectation maximization
            if self.covar_type == 'uniform':
                centers_tensor, vars_tensor, ln_weights_tensor, log_likelihood = torch_uniform_sgmm_lib.torch_sgmm_uniform_em(traj_tensor, centers_tensor, vars_tensor, ln_weights_tensor, thresh=self.kabsch_thresh, dtype=self.dtype, device=self.device)
            else:
                centers_tensor, precisions_tensor, lpdets_tensor, ln_weights_tensor, log_likelihood = torch_kronecker_sgmm_lib.torch_sgmm_kronecker_em(traj_tensor, centers_tensor, precisions_tensor, lpdets_tensor, ln_weights_tensor, thresh=self.kabsch_thresh, dtype=self.dtype, device=self.device)
            if (self.verbose == True):
                print(step+1, np.round(torch.exp(ln_weights_tensor).cpu().numpy(),3), np.round(log_likelihood.cpu().numpy(),3))
            # compute convergence criteria
            if step>0:
                delta_log_lik = torch.abs(old_log_likelihood - log_likelihood)
            old_log_likelihood = log_likelihood
            step += 1

        # pass data back to cpu and delete from gpu
        self.weights = torch.exp(ln_weights_tensor).cpu().numpy()
        self.centers = centers_tensor.cpu().numpy()
        self.log_likelihood = log_likelihood.cpu().numpy()
        # uniform/weighted specific variables
        if self.covar_type == 'uniform': 
            self.cluster_frame_ln_likelihoods =  torch_uniform_sgmm_lib.torch_sgmm_expectation_uniform(traj_tensor, centers_tensor, vars_tensor, device=self.device).cpu().numpy()
            self.vars = vars_tensor.cpu().numpy()
            del vars_tensor
        else: # assume weighted
            self.cluster_frame_ln_likelihoods =  torch_kronecker_sgmm_lib.torch_sgmm_expectation_kronecker(traj_tensor, centers_tensor, precisions_tensor, lpdets_tensor, device=self.device).cpu().numpy()
            self.precisions = precisions_tensor.cpu().numpy()
            self.lpdets = lpdets_tensor.cpu().numpy()
            del precisions_tensor
            del lpdets_tensor
        del traj_tensor
        del ln_weights_tensor
        del centers_tensor
        torch.cuda.empty_cache()
        
        # assign clusters based on largest likelihood 
        self.clusters = np.argmax(self.cluster_frame_ln_likelihoods, axis = 1)
        # uniform has been performed
        self.gmm_fit_flag = True
        # return aligned trajectory
        #return traj_data

    # predict clustering of provided data based on prefit parameters from fit_weighted
    def predict(self,traj_data):
        if self.gmm_fit_flag == True:
            # send data to device
            traj_tensor = torch.tensor(traj_data,dtype=self.dtype,device=self.device)
            centers_tensor = torch.tensor(self.centers,dtype=self.dtype,device=self.device)
            ln_weights_tensor = torch.tensor(np.log(self.weights),dtype=torch.float64,device=self.device)
            # uniform/weighted specific variables
            if self.covar_type == 'uniform': 
                vars_tensor = torch.tensor(self.vars,dtype=torch.float64,device=self.device)
            else: # assume weighted
                # declare precision matrices (inverse covariances)
                precisions_tensor = torch.tensor(self.precisions,dtype=torch.float64,device=self.device)
                # declare array for log determinants for each clusters
                lpdets_tensor = torch.tensor(self.lpdets, dtype=torch.float64,device=self.device)
            
            # make sure trajectory is centered
            torch_align.torch_remove_center_of_geometry(traj_tensor,dtype=self.dtype,device=self.device)
            # Expectation step
            if self.covar_type == 'uniform': 
                cluster_frame_ln_likelihoods_tensor = torch_uniform_sgmm_lib.torch_sgmm_expectation_uniform(traj_tensor, centers_tensor, vars_tensor, device=self.device)
            else: # assume weighted
                cluster_frame_ln_likelihoods_tensor = torch_kronecker_sgmm_lib.torch_sgmm_expectation_kronecker(traj_tensor, centers_tensor, precisions_tensor, lpdets_tensor, device=self.device)
            for k in range(self.n_clusters):
                cluster_frame_ln_likelihoods_tensor[:,k] += ln_weights_tensor[k]
            log_norm = torch.logsumexp(cluster_frame_ln_likelihoods_tensor,1)
            log_likelihood = torch.mean(log_norm).cpu().numpy()
            # assign clusters based on largest likelihood (probability density)
            clusters = torch.argmax(cluster_frame_ln_likelihoods_tensor, dim = 0).cpu().numpy()
            # center trajectory around averages
            for k in range(self.n_clusters):
                indeces = np.argwhere(clusters == k).flatten()
                #traj_data[indeces] = #traj_tools.traj_align_weighted_kabsch(traj_data[indeces],self.centers[k],self.precisions[k])
            return clusters, traj_data, log_likelihood
            # delete data from gpu
            del traj_tensor
            del ln_weights_tensor
            del centers_tensor
            if self.covar_type == 'uniform': 
                del vars_tensor 
            else: # assume weighted
                del precisions_tensor 
                del lpdets_tensor 
            torch.cuda.empty_cache()
        else:
            print("shapeGMM must be fit before it can predict.")
#
#    def predict_uniform(self,traj_data):
#        if self.gmm_uniform_flag == True:
#            # get metadata from trajectory data
#            n_frames = traj_data.shape[0]
#            # declare likelihood array
#            cluster_frame_ln_likelihoods = np.empty((self.n_clusters,n_frames),dtype=np.float64)
#            # make sure trajectory is centered
#            traj_data = traj_tools.traj_remove_cog_translation(traj_data)
#            # Expectation step
#            for k in range(self.n_clusters):
#                # align the entire trajectory to each cluster mean
#                traj_data = traj_tools.traj_align(traj_data,self.centers[k])
#                cluster_frame_ln_likelihoods[k,:] = gmm_shapes_uniform_library.ln_spherical_gaussian_pdf(traj_data.reshape(n_frames,self.n_features), self.centers[k].reshape(self.n_features), self.var[k])
            # compute log likelihood
#            log_likelihood = 0.0
#            for i in range(n_frames):
#                log_likelihood += gmm_shapes_uniform_library.logsumexp(cluster_frame_ln_likelihoods[:,i] + self.ln_weights)
            # assign clusters based on largest likelihood (probability density)
#            clusters = np.argmax(cluster_frame_ln_likelihoods, axis = 0)
            # center trajectory around averages
#            for k in range(self.n_clusters):
#                indeces = np.argwhere(clusters == k).flatten()
#                traj_data[indeces] = traj_tools.traj_align(traj_data[indeces],self.centers[k])
#            return clusters, traj_data, log_likelihood
#        else:
#            print("Uniform shape-GMM must be fitted before you can predict.")
