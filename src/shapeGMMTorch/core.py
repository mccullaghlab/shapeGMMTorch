import numpy as np
import torch
from . import align_in_place
from . import generation
from . import version
from .em import kronecker
from .em import uniform

class ShapeGMM:
    """
    ShapeGMM performs Gaussian Mixture Model (GMM) componenting in the 
    size-and-shape space of molecular trajectories. It supports both 
    uniform and Kronecker-structured covariance models and allows 
    flexible initialization and convergence control.

    Parameters
    ----------
    n_components : int
        Number of components (Gaussian components) to fit.
    log_thresh : float, optional
        Convergence threshold on the change in log-likelihood between EM iterations.
        The algorithm stops when the change falls below this value. Default is 1e-3.
    max_steps : int, optional
        Maximum number of EM iterations. Default is 200.
    covar_type : str, optional
        Covariance structure to use in the model. Must be either:
        - 'uniform': all components share a spherical variance.
        - 'kronecker': each component has a Kronecker-structured covariance matrix.
        Default is 'uniform'.
    init_component_method : str, optional
        Method for initializing component assignments. Options include:
        - 'kmeans++': Use kmeans++ style seeding.  Has an initial random component.
        - 'random': randomly assign frames to components.
        - 'chunk': assign frames in sequential blocks.
        - 'read': assume component_ids are provided externally.
        Default is 'kmeans++'.
    sort : bool, optional
        If True, components will be sorted by population after fitting. Default is True.
    kabsch_thresh : float, optional
        Convergence threshold for the Kabsch-based alignment procedure used during M-steps.
        Default is 1e-1.
    kabsch_max_steps : int, optional
        Maximum number of iterations allowed for Kabsch alignment during each M-step.
        Default is 500.
    dtype : torch.dtype, optional
        Floating point precision for tensors. Default is torch.float32.
    device : torch.device, optional
        Torch device on which to perform computations (e.g., 'cuda:0' or 'cpu'). Default is GPU.
    verbose : bool, optional
        If True, prints progress and detailed logs during EM fitting. Default is False.
    random_seed : int, optional
        Seed for random number generation (NumPy and PyTorch). If provided, ensures reproducibility.

    Fitted Attributes
    -----------------
    weights_ : np.ndarray
        Learned mixture weights for each component.
    means_ : np.ndarray
        Learned component mean structures in shape space.
    vars_ : np.ndarray (if covar_type='uniform')
        Learned uniform variances for each component.
    precisions_ : np.ndarray (if covar_type='kronecker')
        Learned Kronecker precision matrices for each component.
    lpdets_ : np.ndarray (if covar_type='kronecker')
        Log pseudo-determinants for each Kronecker precision matrix.
    is_fitted_ : bool
        Indicates whether the model has been fitted to data.

    Functional Methods
    ------------------
    fit(traj_data, component_ids=None, frame_weights=None)
        Fit the model to input trajectory data, optionally using provided initial component IDs
        and frame weights.

    predict(traj_data)
        Assign component labels to new trajectory data based on maximum posterior probability.

    score(traj_data, frame_weights=None)
        Compute the total (or weighted) log-likelihood of the data under the fitted model.

    generate(n_frames)
        Generate new synthetic trajectory frames by sampling from the trained ShapeGMM model.    
    """

    def __init__(
        self,
        n_components: int,
        log_thresh: float = 1E-3,
        max_steps: int = 200,
        covar_type: str = "kronecker",
        init_component_method: str = "kmeans++",
        sort: bool = True,
        kabsch_thresh: float = 1E-1,
        kabsch_max_steps: int = 500,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cuda:0"),
        verbose: bool = False,
        random_seed: int = None
    ):
        self.n_components = n_components
        self.log_thresh = log_thresh
        self.max_steps = max_steps
        self.covar_type = covar_type
        self.init_component_method = init_component_method
        self.sort = sort
        self.kabsch_thresh = kabsch_thresh
        self.kabsch_max_steps = kabsch_max_steps
        self.dtype = dtype
        self.device = device
        self.verbose = verbose
        self._init_components_flag = False
        self._is_fitted = False
        self.random_seed = random_seed

        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

    def _verbose_print(self, *args):
        if self.verbose:
            print(*args)

    def _kmeans_pp_seeding(self, traj_tensor):
        """
        KMeans++-style initialization for molecular trajectories.

        Parameters
        ----------
        traj_tensor : torch.tensor
            Array of shape (n_frames, n_atoms, 3)

        Returns
        -------
        component_ids : np.ndarray
            Initial cluster assignment of shape (n_frames,)
        """

        rng = np.random.default_rng(self.random_seed if hasattr(self, "random_seed") else None)
        n_frames = traj_tensor.shape[0]

        # choose first frame
        dists2 = torch.empty((n_frames,self.n_components), dtype=self.dtype, device=self.device)
        dists2[:,0] = align_in_place.trajectory_sd(traj_tensor, traj_tensor[rng.choice(n_frames)])

        for c in range(1, self.n_components):
            # determine min distances
            min_dists2 = torch.amin(dists2[:,:c],dim=1)

            # probs
            probs = min_dists2 / min_dists2.sum()
            next_idx = torch.multinomial(probs, 1).item()
            # compute new distances
            dists2[:,c] = align_in_place.trajectory_sd(traj_tensor, traj_tensor[next_idx])

        # assign frames to centers
        return torch.argmin(dists2, dim=1).cpu().numpy()

    def _init_components(self, traj_tensor: torch.Tensor, component_ids: np.ndarray = None):
        self.n_train_frames = traj_tensor.shape[0]
        self.n_atoms = traj_tensor.shape[1]
        self.n_dim = traj_tensor.shape[2]
        self.n_features = self.n_atoms * self.n_dim

        self._verbose_print("========== ShapeGMM Initialization ==========")
        self._verbose_print(f"Covariance type         : {self.covar_type}")
        self._verbose_print(f"Data type (dtype)       : {self.dtype}")
        self._verbose_print(f"Device                  : {self.device}")
        self._verbose_print(f"Component init method   : {self.init_component_method}")
        self._verbose_print(f"Random seed             : {self.random_seed}")
        self._verbose_print(f"Number of frames        : {self.n_train_frames}")
        self._verbose_print(f"Number of atoms         : {self.n_atoms}")
        self._verbose_print(f"Number of dimensions    : {self.n_dim}")
        self._verbose_print(f"LL convergence threshold: {self.log_thresh}")
        self._verbose_print("=============================================")

        if self.init_component_method == "chunk":
            component_ids = np.arange(self.n_train_frames) * self.n_components // self.n_train_frames
        elif self.init_component_method == "read" and component_ids is not None:
            component_ids = component_ids
        elif self.init_component_method == "kmeans++":
            component_ids = self._kmeans_pp_seeding(traj_tensor)
        else:
            if self.n_components > 1:
                component_ids = np.random.choice(self.n_components, size=self.n_train_frames)
            else:
                component_ids = np.zeros(self.n_train_frames,dtype=int)

        self._init_components_flag = True
        return component_ids

    @torch.no_grad()
    def fit(self, traj_data: np.ndarray, component_ids: np.ndarray = None, frame_weights: np.ndarray = None):
        """
        Fit the ShapeGMM model to data using Expectation Maximization (EM) with Maximum Likelihood alignments of each
        component in every EM step.

        Parameters
        ----------
        traj_data : np.ndarray
            Trajectory data (n_frames, n_atoms, 3).
        component_ids : np.ndarray, optional
            Initial component assignments.
        frame_weights : np.ndarray, optional
            Weights for each frame.
        """
        traj_tensor = torch.tensor(traj_data, dtype=self.dtype, device=self.device)
        align_in_place.remove_center_of_geometry_in_place(traj_tensor)

        if not self._init_components_flag:
            component_ids = self._init_components(traj_tensor, component_ids)

        if frame_weights is None:
            self._verbose_print("Setting uniform frame weights.")
            self.train_frame_weights = np.ones(self.n_train_frames) / self.n_train_frames
        else:
            self._verbose_print("Using provided frame weights.")
            self.train_frame_weights = frame_weights / np.sum(frame_weights)

        frame_weights_tensor = torch.tensor(self.train_frame_weights, dtype=torch.float64, device=self.device)

        means_tensor = torch.empty((self.n_components, self.n_atoms, self.n_dim), dtype=self.dtype, device=self.device)
        self.weights_ = np.bincount(component_ids, minlength=self.n_components) / self.n_train_frames
        ln_weights_tensor = torch.log(torch.tensor(self.weights_, dtype=torch.float64, device=self.device))

        if self.covar_type == "uniform":
            vars_tensor = torch.empty(self.n_components, dtype=torch.float64, device=self.device)
            em_step = uniform.sgmm_uniform_em
            em_args = (traj_tensor, frame_weights_tensor, means_tensor, vars_tensor, ln_weights_tensor)
        else:
            precisions_tensor = torch.empty((self.n_components, self.n_atoms, self.n_atoms), dtype=torch.float64, device=self.device)
            lpdets_tensor = torch.empty(self.n_components, dtype=torch.float64, device=self.device)
            em_step = kronecker.sgmm_kronecker_em
            em_args = (traj_tensor, frame_weights_tensor, means_tensor, precisions_tensor, lpdets_tensor, ln_weights_tensor)

        for k in range(self.n_components):
            indices = np.argwhere(component_ids == k).flatten()
            if indices.ndim == 0 or indices.shape == ():  # Handle n_components=1
                indices = indices.reshape(1)
            if self.covar_type == "uniform":
                mean, var = align_in_place.maximum_likelihood_uniform_alignment_frame_weighted_in_place(
                    traj_tensor[indices], frame_weights_tensor[indices], thresh=self.kabsch_thresh
                )
                means_tensor[k].copy_(mean)
                vars_tensor[k].copy_(var)
            else:
                mean, precision, lpdet = align_in_place.maximum_likelihood_kronecker_alignment_frame_weighted_in_place(
                    traj_tensor[indices], frame_weights_tensor[indices], thresh=self.kabsch_thresh, max_iter=self.kabsch_max_steps
                )
                means_tensor[k].copy_(mean)
                precisions_tensor[k].copy_(precision)
                lpdets_tensor[k].copy_(lpdet)

        self._verbose_print("Initial component weights:", self.weights_)


        delta_log_lik = self.log_thresh + 100.0
        old_log_likelihood = torch.tensor(float("inf"), dtype=torch.float64, device=self.device)
        step = 0
        while step < self.max_steps and delta_log_lik > self.log_thresh:

            log_likelihood = em_step(*em_args, thresh=self.kabsch_thresh, max_iter=self.kabsch_max_steps)

            if self.verbose:
                if step == 0:
                    header = (
                        f"{'Step':>5} | "
                        + " ".join([f" w{k+1:02d}  " for k in range(self.n_components)])
                        + " | LogLikelihood"
                    )
                    print(header)
                    print("-" * len(header))

                weights_now = torch.exp(ln_weights_tensor).cpu().numpy()
                weights_fmt = " ".join(f"{w:.4f}" for w in weights_now)  # 4 decimal places
                log_lik_fmt = f"{log_likelihood.item():12.4f}"  # 12 width, 4 decimal places
                print(f"{step+1:5d} | {weights_fmt} | {log_lik_fmt}")

            delta_log_lik = torch.abs(old_log_likelihood - log_likelihood)
            old_log_likelihood = log_likelihood
            step += 1

        self.weights_ = torch.exp(ln_weights_tensor).cpu().numpy()
        self.means_ = means_tensor.cpu().numpy()

        if self.covar_type == "uniform":
            self.vars_ = vars_tensor.cpu().numpy()
        else:
            self.precisions_ = precisions_tensor.cpu().numpy()
            self.lpdets_ = lpdets_tensor.cpu().numpy()

        self.is_fitted_ = True  # model has been fitted!
        
        # sort object
        if self.sort == True:
            self._sort_object()

    @torch.no_grad()
    def refine(self, traj_data: np.ndarray, frame_weights: np.ndarray = None):
        """
        Refine the ShapeGMM model with new data using Expectation Maximization (EM) with Maximum Likelihood alignments of each
        component in every EM step.

        Parameters
        ----------
        traj_data : np.ndarray
            Trajectory data (n_frames, n_atoms, 3).
        frame_weights : np.ndarray, optional
            Weights for each frame.
        """
        if not self.is_fitted_:
            raise RuntimeError("ShapeGMM must be fit before it can be refined.")

        traj_tensor = torch.tensor(traj_data, dtype=self.dtype, device=self.device)
        align_in_place.remove_center_of_geometry_in_place(traj_tensor)

        if frame_weights is None:
            self._verbose_print("Setting uniform frame weights.")
            self.refine_frame_weights = np.ones(traj_tensor.shape[0]) / traj_tensor.shape[0]
        else:
            self._verbose_print("Using provided frame weights.")
            self.refine_frame_weights = frame_weights / np.sum(frame_weights)

        frame_weights_tensor = torch.tensor(self.refine_frame_weights, dtype=torch.float64, device=self.device)

        means_tensor = torch.tensor(self.means_, dtype=self.dtype, device=self.device)
        ln_weights_tensor = torch.log(torch.tensor(self.weights_, dtype=torch.float64, device=self.device))

        if self.covar_type == "uniform":
            vars_tensor = torch.tensor(self.vars_, dtype=self.dtype, device=self.device)
            em_step = uniform.sgmm_uniform_em
            em_args = (traj_tensor, frame_weights_tensor, means_tensor, vars_tensor, ln_weights_tensor)
        else:
            precisions_tensor = torch.tensor(self.precisions_, dtype=torch.float64, device=self.device)
            lpdets_tensor = torch.tensor(self.lpdets_, dtype=torch.float64, device=self.device)
            em_step = kronecker.sgmm_kronecker_em
            em_args = (traj_tensor, frame_weights_tensor, means_tensor, precisions_tensor, lpdets_tensor, ln_weights_tensor)

        self._verbose_print("Initial component weights during refinement:", self.weights_)


        delta_log_lik = self.log_thresh + 100.0
        old_log_likelihood = torch.tensor(float("inf"), dtype=torch.float64, device=self.device)
        step = 0
        while step < self.max_steps and delta_log_lik > self.log_thresh:

            log_likelihood = em_step(*em_args, thresh=self.kabsch_thresh, max_iter=self.kabsch_max_steps)

            if self.verbose:
                if step == 0:
                    header = (
                        f"{'Step':>5} | "
                        + " ".join([f" w{k+1:02d}  " for k in range(self.n_components)])
                        + " | LogLikelihood"
                    )
                    print(header)
                    print("-" * len(header))

                weights_now = torch.exp(ln_weights_tensor).cpu().numpy()
                weights_fmt = " ".join(f"{w:.4f}" for w in weights_now)  # 4 decimal places
                log_lik_fmt = f"{log_likelihood.item():12.4f}"  # 12 width, 4 decimal places
                print(f"{step+1:5d} | {weights_fmt} | {log_lik_fmt}")

            delta_log_lik = torch.abs(old_log_likelihood - log_likelihood)
            old_log_likelihood = log_likelihood
            step += 1

        self.weights_ = torch.exp(ln_weights_tensor).cpu().numpy()
        self.means_ = means_tensor.cpu().numpy()

        if self.covar_type == "uniform":
            self.vars_ = vars_tensor.cpu().numpy()
        else:
            self.precisions_ = precisions_tensor.cpu().numpy()
            self.lpdets_ = lpdets_tensor.cpu().numpy()

        self.is_refined_ = True  # model has been refined!
        
        # sort object
        if self.sort == True:
            self._sort_object()

    @torch.no_grad()
    def _compute_log_likelihoods(self, traj_data: np.ndarray) -> torch.Tensor:
        """
        Internal helper to compute component-frame log-likelihoods.

        Parameters
        ----------
        traj_data : np.ndarray
            (n_frames, n_atoms, 3) trajectory array

        Returns
        -------
        component_frame_ln_likelihoods_tensor : torch.Tensor
            (n_frames, n_components) log-likelihood tensor
        """
        traj_tensor = torch.tensor(traj_data, dtype=self.dtype, device=self.device)
        align_in_place.remove_center_of_geometry_in_place(traj_tensor)
        means_tensor = torch.tensor(self.means_, dtype=self.dtype, device=self.device)
        ln_weights_tensor = torch.tensor(np.log(self.weights_), dtype=torch.float64, device=self.device)

        if self.covar_type == 'uniform':
            vars_tensor = torch.tensor(self.vars_, dtype=torch.float64, device=self.device)
            component_frame_ln_likelihoods_tensor = uniform.sgmm_expectation_uniform(
                traj_tensor, means_tensor, vars_tensor
            )
        else:
            precisions_tensor = torch.tensor(self.precisions_, dtype=torch.float64, device=self.device)
            lpdets_tensor = torch.tensor(self.lpdets_, dtype=torch.float64, device=self.device)
            component_frame_ln_likelihoods_tensor = kronecker.sgmm_expectation_kronecker(
                traj_tensor, means_tensor, precisions_tensor, lpdets_tensor
            )

        # Add log weights
        component_frame_ln_likelihoods_tensor += ln_weights_tensor.view(1, -1)
        return component_frame_ln_likelihoods_tensor

    def predict(self, traj_data: np.ndarray) -> np.ndarray:
        """
        Assign each frame in the input data to a component based on maximum posterior probability.
    
        Parameters
        ----------
        traj_data : np.ndarray
            (n_frames, n_atoms, 3) input trajectory data.

        Returns
        -------
        components : np.ndarray
            (n_frames,) integer array of component assignments.
        """
        if not self.is_fitted_:
            raise RuntimeError("ShapeGMM must be fit before calling predict().")

        component_ln_lik = self._compute_log_likelihoods(traj_data)
        return torch.argmax(component_ln_lik, dim=1).cpu().numpy()

    def score(self, traj_data: np.ndarray, frame_weights: np.ndarray = None) -> float:
        """
        Compute the total log-likelihood of the data under the current model.

        Parameters
        ----------
        traj_data : np.ndarray
            (n_frames, n_atoms, 3) input trajectory data.
        frame_weights : np.ndarray, optional
            (n_frames,) array of frame weights. If None, uniform weights are used.

        Returns
        -------
        log_likelihood : float
            Total (weighted) log-likelihood of the dataset under the model.
        """
        if not self.is_fitted_:
            raise RuntimeError("ShapeGMM must be fit before calling score().")

        n_frames = traj_data.shape[0]
        frame_weights = (
            np.ones(n_frames, dtype=np.float64) / n_frames
            if frame_weights is None
            else np.asarray(frame_weights, dtype=np.float64)
        )
        frame_weights /= frame_weights.sum()

        component_ln_lik = self._compute_log_likelihoods(traj_data)
        frame_log_lik = torch.logsumexp(component_ln_lik, dim=1).cpu().numpy()
        return np.sum(frame_weights * frame_log_lik)

    def lnpdf(self, traj_data: np.ndarray) -> np.ndarray:
        """
        Compute the ln of the Probability for each frame in the fitted model

        Parameters
        ----------
        traj_data : np.ndarray
            (n_frames, n_atoms, 3) input trajectory data.

        Returns
        -------
        lnpdf : np.ndarry
            (n_frames,) array of ln probability of that frame in the current model
        """
        if not self.is_fitted_:
            raise RuntimeError("ShapeGMM must be fit before calling score().")

        component_ln_lik = self._compute_log_likelihoods(traj_data)
        return torch.logsumexp(component_ln_lik, dim=1).cpu().numpy()

    def predict_proba(self, traj_data: np.ndarray) -> np.ndarray:
        """
        Compute the posterior probabilities (responsibilities) for each frame and component.

        Parameters
        ----------
        traj_data : np.ndarray
            (n_frames, n_atoms, 3) input trajectory data.

        Returns
        -------
        responsibilities : np.ndarray
            (n_frames, n_components) array where each element is the probability
            that the corresponding frame belongs to a component.
        """
        if not self.is_fitted_:
            raise RuntimeError("ShapeGMM must be fit before calling predict_proba().")

        component_ln_lik = self._compute_log_likelihoods(traj_data)
        log_norm = torch.logsumexp(component_ln_lik, dim=1, keepdim=True)
        log_resp = component_ln_lik - log_norm  # log-softmax

        return torch.exp(log_resp).cpu().numpy()

    

    # generate a trajectory from shapeGMM object - no time correlation!
    def generate(self,n_frames):
        """
        Generate a trajectory from a fit shapeGMM object using multivariate Gaussian generator (from scipy)
        n_frames (required)     - int of number of frames to generater

        Returns:
        trajectory      - (n_frames, n_atoms, 3) float32 or float64 numpy array of particle positions generated from shapeGMM object. 
        """

        if not self.is_fitted_:
            raise RuntimeError("ShapeGMM must be fit before calling generate().")
            
        # generate random component ids based on frame weights - not could adapt this to account for transition matrix
        component_ids = generation.component_ids_from_rand(np.random.rand(n_frames),self.weights_)
        trj = np.empty((n_frames,self.n_atoms,3))
        for component_id in range(self.n_components):
            if self.covar_type == "kronecker":
                precision = self.precisions_[component_id]
            else:
                precision = 1/self.vars_[component_id] * np.identity(self.n_atoms)
                # now enforece that constant vector is in null space of precision
                wsum = -1/self.vars_[component_id]/(self.n_atoms-1)
                for i in range(self.n_atoms):
                    for j in range(self.n_atoms):
                        if i != j:
                            precision[i,j] = wsum
            indeces = np.argwhere(component_ids == component_id).flatten()
            trj[indeces] = generation.gen_mv(self.means_[component_id],precision,indeces.size)
        return trj

    # sort the object based on component weights
    def _sort_object(self):
        if self.is_fitted_ == True:
            # determine sort key
            sort_key = np.argsort(self.weights_)[::-1]
            # repopulate object
            self.means_   = self.means_[sort_key]
            self.weights_    = self.weights_[sort_key]
            if self.covar_type == "uniform":
                self.vars_ = self.vars_[sort_key]
            else:
                self.precisions_ = self.precisions_[sort_key]
                self.lpdets_     = self.lpdets_[sort_key]
        else:
            print("shapeGMM must be fit before it can be sorted.")


    def save(self, filename):
        """
        Save the ShapeGMM model to a file using torch.save().
        """
        if self.covar_type == 'kronecker':
            torch.save({
                'n_components': self.n_components,         # int
                'log_thresh': self.log_thresh,             # float
                'max_steps': self.max_steps,               # int
                'covar_type': self.covar_type,             # str
                'init_component_method': self.init_component_method,  # str
                'sort': self.sort,                         # bool
                'kabsch_thresh': self.kabsch_thresh,       # float
                'kabsch_max_steps': self.kabsch_max_steps, # int
                'random_seed': self.random_seed,           # int
                'verbose': self.verbose,                   # bool
                'dtype': str(self.dtype),                  # stored as str
                'device': str(self.device),                # stored as str
                'n_atoms': self.n_atoms,                   # int
                'weights': self.weights_,                  # np.ndarray 
                'means': self.means_,                      # np.ndarray 
                'precisions': self.precisions_,            # np.ndarray 
                'lnpdet': self.lpdets_,                    # np.ndarray 
                'is_fitted': self.is_fitted_,              # bool
                'version': version.__version__             # str
            }, filename)
        else:
            # Uniform or other covariance models
            torch.save({
                'n_components': self.n_components,
                'log_thresh': self.log_thresh,
                'max_steps': self.max_steps,
                'covar_type': self.covar_type,
                'init_component_method': self.init_component_method,
                'sort': self.sort,
                'kabsch_thresh': self.kabsch_thresh,
                'kabsch_max_steps': self.kabsch_max_steps,
                'random_seed': self.random_seed,
                'verbose': self.verbose,
                'dtype': str(self.dtype),
                'device': str(self.device),
                'n_atoms': self.n_atoms,
                'weights': self.weights_,
                'means': self.means_,
                'variances': self.vars_,      
                'is_fitted': self.is_fitted_,
                'version': version.__version__
            }, filename)



    @staticmethod
    def load(filename):
        """
        Load a saved ShapeGMM model from file.

        Parameters
        ----------
        filename : str
            Path to the saved model file.

        Returns
        -------
        ShapeGMM
            Reconstructed ShapeGMM instance.
        """

        state = torch.load(filename, map_location="cpu", weights_only=False)
        
        model = ShapeGMM(
            n_components=state['n_components'],
            log_thresh=state['log_thresh'],
            max_steps=state['max_steps'],
            covar_type=state['covar_type'],
            init_component_method=state['init_component_method'],
            sort=state['sort'],
            kabsch_thresh=state['kabsch_thresh'],
            kabsch_max_steps=state['kabsch_max_steps'],
            random_seed=state['random_seed'],
            verbose=state['verbose'],
            dtype=getattr(torch, state['dtype'].split('.')[-1]),  # from string back to torch dtype
            device=torch.device(state['device'])
        )

        # Restore learned parameters
        model.n_atoms = state['n_atoms']
        model.weights_ = state['weights']
        model.means_ = state['means']
        model.is_fitted_ = state['is_fitted']

        if model.covar_type == "kronecker":
            model.precisions_ = state['precisions']
            model.lpdets_ = state['lnpdet']
        else:
            model.variances_ = state['variances']

        return model

