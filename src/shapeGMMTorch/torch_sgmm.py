import numpy as np
import torch
import warnings
warnings.filterwarnings('ignore')
import random
# the following are local libraries
from . import torch_align
from . import torch_uniform_sgmm_lib                            
from . import torch_kronecker_sgmm_lib  
from . import generate_points

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
    def __init__(self, n_clusters, log_thresh=1E-3, max_steps=200, covar_type="uniform", init_cluster_method="random", sort = True, kabsch_thresh=1E-1, kabsch_max_steps=500, dtype=torch.float32, device=torch.device("cuda:0"), verbose=False):
        """
        Initialize size-and-shape GMM.
        n_clusters (required)   - integer number of clusters must be input
        covar_type              - string defining the covariance type.  Options are 'kronecker' and 'uniform'.  Defualt is 'uniform'.
        log_thresh              - float threshold in log likelihood difference to determine convergence. Default value is 1e-3.
        max_steps               - integer maximum number of steps that the GMM procedure will do.  Default is 200.
        init_cluster_method     - string dictating how to initialize clusters.  Understood values are 'chunk', 'read' and 'random'.  Default is 'random'.
        sort                    - boolean dictating whether or not the object by cluster population after fitting.  Default is True.
        kabsch_thresh           - float dictating convergence criteria for each alignment step.  Default value is 1e-1.
        dtype                   - Data type to be used.  Default is torch.float32.
        device                  - device to be used.  Default is torch.device('cuda:0') device.
        verbose                 - boolean dictating whether to print various things at every step. Defulat is False.
        """
        
        self.n_clusters = n_clusters                            # integer
        self.log_thresh = log_thresh                            # float
        self.max_steps = max_steps                              # integer
        self.covar_type = covar_type                            # string
        self.init_cluster_method = init_cluster_method          # string
        self.sort = sort                                        # boolean
        self.kabsch_thresh = kabsch_thresh                      # float
        self.kabsch_max_steps = kabsch_max_steps                # integer
        self.dtype = dtype                                      # torch dtype
        self.device = device                                    # torch device
        self.verbose = verbose                                  # boolean
        self._init_clusters_flag = False                        # boolean tracking if clusters have been initialized or not.
        self._gmm_fit_flag = False                              # boolean tracking if GMM has been fit.

    # fit the model
    def fit(self, traj_data, cluster_ids = [], frame_weights = []):
        """
        Fit size-and-shape GMM using traj_data as the training data.
        traj_data     (required) - (n_frames, n_atoms, 3) float32 or float64 numpy array of particle positions. 
        cluster_ids   (optional) - (n_frames) integer numpy array of initial cluster assignments.  
        frame_weights (optional) - (n_frames) float numpy array of frame weights

        Returns an aligned trajectory, if requested.  Each cluster aligned to respective average.
        """

        # pass trajectory data to device
        traj_tensor = torch.tensor(traj_data,dtype=self.dtype,device=self.device)
        # Remove the center-of-geometry from entire trajectory
        torch_align.torch_remove_center_of_geometry(traj_tensor)
        # Initialize clusters if they have not been already
        if (self._init_clusters_flag == False):
            self._init_clusters(traj_tensor, cluster_ids)
        # frame weights
        if len(frame_weights) == 0:
            if (self.verbose==True):
                print("Setting uniform frame weights")
            self.train_frame_weights = np.ones(self.n_train_frames,dtype=np.float64)/self.n_train_frames
        else:
            if (self.verbose==True):
                print("Using user provided frame weights")
            # use provided frame weights but make sure they are normalized
            self.train_frame_weights = frame_weights/np.sum(frame_weights)   
        # pass the frame weights to the device (enforce double precision for accuracy)
        frame_weights_tensor = torch.tensor(self.train_frame_weights,dtype=torch.float64,device=self.device)
            
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
            indeces = np.argwhere(self.cluster_ids == k).flatten()
            # initialize weights as populations of clusters
            self.weights[k] = indeces.size/self.n_train_frames
            if self.covar_type == 'uniform':
                centers_tensor[k], vars_tensor[k] = torch_align.torch_iterative_align_uniform_weighted(traj_tensor[indeces],frame_weights_tensor[indeces],thresh=self.kabsch_thresh,device=self.device,dtype=self.dtype)[1:]
            else:
                centers_tensor[k], precisions_tensor[k], lpdets_tensor[k] = torch_align.torch_iterative_align_kronecker_weighted(traj_tensor[indeces],frame_weights_tensor[indeces],thresh=self.kabsch_thresh,max_iter=self.kabsch_max_steps, device=self.device,dtype=self.dtype)[1:]        
        if (self.verbose == True):
            print("Weights from initial clusters in fit:", self.weights)
    
        # pass remaining data to device
        ln_weights_tensor = torch.tensor(np.log(self.weights),dtype=torch.float64,device=self.device)
        
        # perform Expectation Maximization
        delta_log_lik = 100.0 + self.log_thresh
        step = 0
        while step < self.max_steps and delta_log_lik > self.log_thresh:
            # Expectation maximization
            if self.covar_type == 'uniform':
                centers_tensor, vars_tensor, ln_weights_tensor, log_likelihood = torch_uniform_sgmm_lib.torch_sgmm_uniform_em(traj_tensor, frame_weights_tensor, centers_tensor, vars_tensor, ln_weights_tensor, thresh=self.kabsch_thresh, dtype=self.dtype, device=self.device)
            else:
                centers_tensor, precisions_tensor, lpdets_tensor, ln_weights_tensor, log_likelihood = torch_kronecker_sgmm_lib.torch_sgmm_kronecker_em(traj_tensor, frame_weights_tensor, centers_tensor, precisions_tensor, lpdets_tensor, ln_weights_tensor, thresh=self.kabsch_thresh, max_iter=self.kabsch_max_steps, dtype=self.dtype, device=self.device)
            if (self.verbose == True):
                print(step+1, np.round(torch.exp(ln_weights_tensor).cpu().numpy(),3), np.round(log_likelihood.cpu().numpy(),3))
            # compute convergence criteria
            if step>0:
                delta_log_lik = torch.abs(old_log_likelihood - log_likelihood)
            old_log_likelihood = log_likelihood
            step += 1

        # SGMM fit has been performed
        self._gmm_fit_flag = True
        # pass data back to cpu and delete from gpu
        self.weights = torch.exp(ln_weights_tensor).cpu().numpy()
        # uniform/weighted specific variables
        if self.covar_type == 'uniform': 
            cluster_frame_ln_likelihoods_tensor =  torch_uniform_sgmm_lib.torch_sgmm_expectation_uniform(traj_tensor, centers_tensor, vars_tensor, device=self.device)
            # assign clusters based on largest likelihood 
            self.cluster_ids = torch.argmax(cluster_frame_ln_likelihoods_tensor, dim = 1).cpu().numpy()
            traj_data, self.centers = self._align_clusters_uniform(traj_tensor,centers_tensor)
            self.vars = vars_tensor.cpu().numpy()
            del vars_tensor
        else: # assume Kronecker product covariance
            cluster_frame_ln_likelihoods_tensor =  torch_kronecker_sgmm_lib.torch_sgmm_expectation_kronecker(traj_tensor, centers_tensor, precisions_tensor, lpdets_tensor, dtype=self.dtype, device=self.device)
            # assign clusters based on largest likelihood 
            self.cluster_ids = torch.argmax(cluster_frame_ln_likelihoods_tensor, dim = 1).cpu().numpy()
            traj_data, self.centers = self._align_clusters_kronecker(traj_tensor,centers_tensor, precisions_tensor)
            self.precisions = precisions_tensor.cpu().numpy()
            self.lpdets = lpdets_tensor.cpu().numpy()
            del precisions_tensor
            del lpdets_tensor
        # determine log likelihood per frame and save it
        for k in range(self.n_clusters):
            cluster_frame_ln_likelihoods_tensor[:,k] += ln_weights_tensor[k]
        self.train_frame_log_likelihood = (torch.logsumexp(cluster_frame_ln_likelihoods_tensor,1)).cpu().numpy()
        self.log_likelihood = np.sum(self.train_frame_weights*self.train_frame_log_likelihood)
        # delete data
        del traj_tensor
        del ln_weights_tensor
        del cluster_frame_ln_likelihoods_tensor
        del centers_tensor
        torch.cuda.empty_cache()
        
        # sort object
        if self.sort == True:
            self._sort_object()
        # return aligned trajectory
        #return traj_data

    # predict clustering of provided data based on prefit parameters from fit_weighted
    def predict(self,traj_data, frame_weights = []):
        """
        Predict size-and-shape GMM using traj_data as prediction set and already fit object parameters.
        traj_data (required)     - (n_frames, n_atoms, 3) float32 or float64 numpy array of particle positions. 
        frame_weights (optional) - (n_frames) float numpy array of frame weights

        Returns:
        cluster ids             - (n_frames) int array
        aligned trajectory      - (n_frames, n_atoms, 3) float32 or float64 numpy array of particle positions. 
        log likelihood          - float64 scalar of the log likelihood of the data given the fit model
        """

        if self._gmm_fit_flag == True:
            # metadata
            n_predict_frames = traj_data.shape[0]
            # send data to device
            traj_tensor = torch.tensor(traj_data,dtype=self.dtype,device=self.device)
            centers_tensor = torch.tensor(self.centers,dtype=self.dtype,device=self.device)
            ln_weights_tensor = torch.tensor(np.log(self.weights),dtype=torch.float64,device=self.device)
            # uniform/kronecker covariance specific variables
            if self.covar_type == 'uniform': 
                vars_tensor = torch.tensor(self.vars,dtype=torch.float64,device=self.device)
            else: # assume weighted
                # declare precision matrices (inverse covariances)
                precisions_tensor = torch.tensor(self.precisions,dtype=torch.float64,device=self.device)
                # declare array for log determinants for each clusters
                lpdets_tensor = torch.tensor(self.lpdets, dtype=torch.float64,device=self.device)
            # frame weights
            if len(frame_weights) == 0:
                if (self.verbose==True):
                    print("Assuming uniform frame weights")
                self.predict_frame_weights = np.ones(n_predict_frames,dtype=np.float64)/n_predict_frames
            else:
                if (self.verbose==True):
                    print("Using user provided frame weights")
                # use provided frame weights but make sure they are normalized
                self.predict_frame_weights = frame_weights.astype(np.float64)
                self.predict_frame_weights /= np.sum(self.predict_frame_weights)   
            # pass the frame weights to the device (enforce double precision for accuracy)
            frame_weights_tensor = torch.tensor(self.predict_frame_weights,dtype=torch.float64,device=self.device)
            
            # make sure trajectory is centered
            torch_align.torch_remove_center_of_geometry(traj_tensor,dtype=self.dtype,device=self.device)
            # Expectation step
            if self.covar_type == 'uniform': 
                cluster_frame_ln_likelihoods_tensor =  torch_uniform_sgmm_lib.torch_sgmm_expectation_uniform(traj_tensor, centers_tensor, vars_tensor, device=self.device)
            else: # assume weighted
                cluster_frame_ln_likelihoods_tensor = torch_kronecker_sgmm_lib.torch_sgmm_expectation_kronecker(traj_tensor, centers_tensor, precisions_tensor, lpdets_tensor, dtype=self.dtype, device=self.device)
            for k in range(self.n_clusters):
                cluster_frame_ln_likelihoods_tensor[:,k] += ln_weights_tensor[k]
            self.predict_frame_log_likelihood = (torch.logsumexp(cluster_frame_ln_likelihoods_tensor,1)).cpu().numpy()
            log_likelihood = np.sum(self.predict_frame_weights*self.predict_frame_log_likelihood)
            # assign clusters based on largest likelihood (probability density)
            clusters = torch.argmax(cluster_frame_ln_likelihoods_tensor, dim = 1).cpu().numpy()
            # align each cluster to its average
            for k in range(self.n_clusters):
                indeces = np.argwhere(clusters == k).flatten()
                if self.covar_type == 'uniform':
                    traj_tensor[indeces] = torch_align.torch_align_uniform(traj_tensor[indeces], centers_tensor[k])
                else:
                    traj_tensor[indeces] = torch_align.torch_align_kronecker(traj_tensor[indeces], centers_tensor[k], precisions_tensor[k], dtype=self.dtype, device=self.device)
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
            # return values
            return clusters, log_likelihood
        else:
            print("shapeGMM must be fit before it can predict.")

    # generate a trajectory from shapeGMM object - no time correlation!
    def generate(self,n_frames):
        """
        Generate a trajectory from a fit shapeGMM object using multivariate Gaussian generator (from scipy)
        n_frames (required)     - int of number of frames to generater

        Returns:
        trajectory      - (n_frames, n_atoms, 3) float32 or float64 numpy array of particle positions generated from shapeGMM object. 
        """

        if self._gmm_fit_flag == True:
            # generate random cluster ids based on frame weights - not could adapt this to account for transition matrix
            cluster_ids = generate_points.cluster_ids_from_rand(np.random.rand(n_frames),self.weights)
            trj = np.empty((n_frames,self.n_atoms,3))
            for cluster_id in range(self.n_clusters):
                if self.covar_type == "kronecker":
                    precision = self.precisions[cluster_id]
                else:
                    precision = 1/self.vars[cluster_id] * np.identity(self.n_atoms)
                    # now enforece that constant vector is in null space of precision
                    wsum = -1/self.vars[cluster_id]/(self.n_atoms-1)
                    for i in range(self.n_atoms):
                        for j in range(self.n_atoms):
                            if i != j:
                                precision[i,j] = wsum
                indeces = np.argwhere(cluster_ids == cluster_id).flatten()
                trj[indeces] = generate_points.gen_mv(self.centers[cluster_id],precision,indeces.size)
            return trj
        else:
            print("shapeGMM must be fit before it can generate.")
        
    # initialize clusters
    def _init_clusters(self, traj_tensor, cluster_ids=[]):
        
        # get metadata
        self.n_train_frames = int(traj_tensor.shape[0])
        self.n_atoms = traj_tensor.shape[1]
        self.n_dim = traj_tensor.shape[2]
        self.n_features = self.n_dim*self.n_atoms
        if (self.verbose == True):
            # print metadata to stdout
            print("Number of frames being analyzed:", self.n_train_frames)
            print("Number of particles being analyzed:", self.n_atoms)
            print("Number of dimensions (must be 3):", self.n_dim)
            print("Initializing clustering using method:", self.init_cluster_method)
        # declare clusters
        self.cluster_ids = np.zeros(self.n_train_frames,dtype=np.int32)

        # make initial clustering based on input user choice (default is random)
        if (self.init_cluster_method == "chunk"):
            for i in range(self.n_train_frames):
                self.cluster_ids[i] = i*self.n_clusters // self.n_train_frames
        elif (self.init_cluster_method == "read"):
            # should affirm that there are n_train_frames clusters
            self.cluster_ids = cluster_ids
        else: # default is random
            self.cluster_ids = torch_uniform_sgmm_lib.init_random(traj_tensor,self.n_clusters,dtype=self.dtype, device=self.device)
            #log_lik = torch_uniform_sgmm_lib.uniform_sgmm_log_likelihood(traj_tensor,self.cluster_ids,device=self.device).cpu().numpy()
        # clusters have been initialized
        self._init_clusters_flag = True

    # align the trajectory and averages
    def _align_clusters_uniform(self, traj_tensor, centers_tensor):
        if self._gmm_fit_flag == True:
            # determine a global average 
            global_center_tensor, global_var_tensor = torch_align.torch_iterative_align_uniform(traj_tensor, dtype=self.dtype, device=self.device, thresh=self.kabsch_thresh)[1:]
            # align centers to global average
            centers_tensor = torch_align.torch_align_uniform(centers_tensor, global_center_tensor)
            # align each cluster to its average
            for k in range(self.n_clusters):
                indeces = np.argwhere(self.cluster_ids == k).flatten()
                traj_tensor[indeces] = torch_align.torch_align_uniform(traj_tensor[indeces], centers_tensor[k])
            return traj_tensor.cpu().numpy(), centers_tensor.cpu().numpy()
        else:
            print("shapeGMM must be fit before it can be aligned.")

    def _align_clusters_kronecker(self, traj_tensor, centers_tensor, precisions_tensor):
        if self._gmm_fit_flag == True:
            # determine a global average 
            global_center_tensor, global_precision_tensor = torch_align.torch_iterative_align_kronecker(traj_tensor, dtype=self.dtype, device=self.device, thresh=self.kabsch_thresh, max_iter=self.kabsch_max_steps)[1:3]
            # align centers to global average (NxN covars/precisions are rotationally invariant so don't need to rotate them)
            centers_tensor = torch_align.torch_align_kronecker(centers_tensor, global_center_tensor, global_precision_tensor, dtype=self.dtype, device=self.device)
            # align each cluster to its average
            for k in range(self.n_clusters):
                indeces = np.argwhere(self.cluster_ids == k).flatten()
                traj_tensor[indeces] = torch_align.torch_align_kronecker(traj_tensor[indeces], centers_tensor[k], precisions_tensor[k], dtype=self.dtype, device=self.device)
            return traj_tensor.cpu().numpy(), centers_tensor.cpu().numpy()
        else:
            print("shapeGMM must be fit before it can be aligned.")

    # sort the object based on cluster weights
    def _sort_object(self):
        if self._gmm_fit_flag == True:
            # determine sort key
            sort_key = np.argsort(self.weights)[::-1]
            cluster_ids = np.arange(self.n_clusters).astype(int)
            sorted_cluster_ids = cluster_ids[sort_key]
            new_clusters = np.empty(self.n_train_frames,dtype=int)
            for frame in range(self.n_train_frames):
                new_clusters[frame] = np.argwhere(sorted_cluster_ids == self.cluster_ids[frame])
            # repopulate object
            self.centers    = self.centers[sort_key]
            self.weights    = self.weights[sort_key]
            self.cluster_ids   = new_clusters
            if self.covar_type == "uniform":
                self.vars = self.vars[sort_key]
            else:
                self.precisions = self.precisions[sort_key]
                self.lpdets     = self.lpdets[sort_key]
        else:
            print("shapeGMM must be fit before it can be sorted.")
