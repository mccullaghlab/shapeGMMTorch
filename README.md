# shapeGMMTorch

[![build](https://github.com/mccullaghlab/shapeGMMTorch/actions/workflows/python-package.yml/badge.svg)](https://github.com/mccullaghlab/shapeGMMTorch/actions/workflows/python-package.yml)
[![PyPI](https://img.shields.io/pypi/v/shapeGMMTorch.svg)](https://pypi.org/project/shapeGMMTorch/)
[![Examples](https://img.shields.io/badge/examples-notebooks-blue.svg)](https://github.com/mccullaghlab/shapeGMMTorch/tree/main/examples)


## Overview

This is a package to perform Gaussian Mixture Model (GMM) clustering on particle positions (in <img src="https://render.githubusercontent.com/render/math?math=\mathbb{R}^3">). Like other GMM schemes, the user must specify the number of clusters and a cluster initialization scheme (defaults to random).  This is specified in the object initialization line, analogous to how it is done for the sklearn GaussianMixture package.  There are two choices for the form of the covariance  specified by the `covar_type` keyword in the object initialization.  See paper (Klem et al JCTC 2022, https://pubs.acs.org/doi/abs/10.1021/acs.jctc.1c01290) for additional details.

## Dependencies

This package is dependent on the following packages:

1. Python>=3.6 
2. numpy
3. scipy
4. torch>=2.6  
5. MDAnalysis
6. Matplotlib

## Installation

After the dependencies have been installed, the package can be installed from pip

`pip install shapeGMMTorch`

or by downloading from github and then running

`python setup.py install`

## Usage 

This package is designed to mimic the usage of the sklearn package.  You first initiliaze the object and then fit.  Predict and score can be done once the model is fit.  Fit and predict functions take particle position trajectories as input in the form of a `(n_frames, n_atoms, 3)` numpy array.

### Initialize:

`from shapeGMMTorch import ShapeGMM`

Uniform covariance (spherical, uncorrelated, homogeneous):

`uni_sgmm = ShapeGMM(n_components, covar_type = 'uniform', verbose=True)`

Kronecker product covariance (formerly called weighted covariance; spherical, correlated, heterogeneous):

`kron_sgmm = ShapeGMM(n_components, covar_type = 'kronecker', verbose=True)`

During initialization, the following options are available:

	- n_components (required) - integer number of components must be input
	- covar_type              - string defining the covariance type.  Options are 'kronecker' and 'uniform'.  Default is 'kronecker'.
	- log_thresh              - float threshold in log likelihood difference to determine convergence. Default value is 1e-3.
	- max_steps               - integer maximum number of steps that the GMM procedure will do.  Default is 200.
	- init_component_method   - string dictating how to initialize components.  Understood values are 'kmeans++', 'chunk', 'read' and 'random'.  Default is 'kmeans++'.
	- random_seed             - integer setting the seed for numpy and torch random number generators.  If None, both are seeded by the clock.  Default is None.
	- sort                    - boolean dictating whether to sort the object by component population after fitting.  Default is True.
	- kabsch_thresh           - float dictating convergence criteria for each iterative alignment (Maximization step).  Default value is 1e-1.
	- dtype                   - Torch data type to be used.  Default is torch.float32.
	- device                  - Torch device to be used.  Default is torch.device('cuda:0') device.
	- verbose                 - boolean dictating whether to print various things at every step. Default is False.

### Fit:

A standard fit can be performed in the following manner:

`uni_sgmm.fit(train_positions)`

`kron_sgmm.fit(train_positions)`

where `train_positions` is an array of dimensions `(n_train_frames, n_atoms, 3)`. Notice there is no difference in syntax when fitting the two covariance types. The fit method also accepts two optional arguments, which may be required depending on the initialization method and/or the type of simulation from which the data stems: 

	- component_ids   (optional) - (n_train_frames) integer array of initial component/cluster ids.  This option is only used if init_component_method = 'read', in which case it is necessary.
	- frame_weights (optional) - (n_train_frames) float array of relative frame weights.  If none are provided the code assumes equal weights for all frames.

If these options are used the fit call looks like

`uni_sgmm.fit(train_positions, component_ids = initial_component_ids, frame_weights = train_frame_weights)`

`kron_sgmm.fit(train_positions, component_ids = initial_component_ids, frame_weights = train_frame_weights)`

### Label and Score:

Once fit, a ShapeGMM object can be used to label and score data. The data need not be the training data but must have the same number of atoms/particles.  To label and score (log likelihood per frame) trajectory data, `cv_positions`, of dimensions `(n_cv_frames, n_atoms, 3)`:

for uniform model:

`uni_train_component_ids  = uni_sgmm.predict(cv_positions)`

`uni_train_log_likelihood = uni_sgmm.score(cv_positions)`

for kronecker model:

`kron_train_component_ids = kron_sgmm.predict(cv_positions)`

`kron_train_log_likelihood = kron_sgmm.score(cv_positions)`

If the trajectory has non-uniform frame weights this must be taken into account in the scoring function with float array `cv_frame_weights` of dimension `(n_cv_frames)`:

uniform model:

`uni_train_log_likelihood = uni_sgmm.score(cv_positions, frame_weights = cv_frame_weights)`

kronecker model:

`kron_train_log_likelihood = kron_sgmm.score(cv_positions, frame_weights = cv_frame_weights)`

The result of the `predict` function will be an integer array of component assignment of dimension `(n_cv_frames)`.  The result of the `score` function will be a scalar float (log-likelihood per frame).

## Attributes

After being properly fit, a ShapeGMM object will have the following attributes:

	- weights_                   - (n_components) float array of component weights
	- means_                     - (n_components, n_atoms, 3) float array of component centers/averages

Uniform covariance specific attributes

	- vars_		       	         - (n_components) float array of component variances

Kronecker covariance specific attributes

	- precisions_	   	         - (n_components, n_atoms, n_atoms) float array of component precisions (inverse covariances)
	- lpdets_	    	         - (n_components) float array of ln(det(covar))

## Fitting with multiple attempts

Because the EM algorithm is prone to getting caught in local maxima, we suggest performing multiple fit attempts, with different random initializations, to help determine the optimal ShapeGMM parameters.  We provide a script to achieve this.


```python
from shapeGMMTorch.utils import sgmm_fit_with_attempts

best_model = sgmm_fit_with_attempts(
    traj_data=my_trajectory,         # shape: (n_frames, n_atoms, 3)
    n_components=3,                  # number of clusters/components
    n_attempts=10,                   # number of random restarts
    covar_type="kronecker",         # or "uniform"
    frame_weights=weights,          # optional: weighting frames by importance
    init_component_method="kmeans++",  # initialization strategy
    dtype=torch.float32,
    device=torch.device("cuda:0"),  # use GPU if available
    random_seed=42,
    verbose=True
)
```

### Parameters

| Name                   | Description                                                                 |
|------------------------|-----------------------------------------------------------------------------|
| `traj_data`            | NumPy array of shape `(n_frames, n_atoms, 3)` containing trajectory data    |
| `n_components`         | Number of Gaussian components (clusters)                                   |
| `n_attempts`           | Number of fitting attempts with different initializations                   |
| `covar_type`           | Covariance model: `"uniform"` or `"kronecker"`                              |
| `frame_weights`        | Optional array of weights (per frame) for likelihood computation            |
| `thresh`               | Log-likelihood convergence threshold (default `1e-3`)                       |
| `kabsch_thresh`        | Threshold for maximum-likelihood alignment convergence (default `1e-1`)     |
| `dtype`                | PyTorch tensor data type (e.g., `torch.float32` or `torch.float64`)         |
| `device`               | PyTorch device (`"cpu"` or `"cuda:0"`)                                     |
| `random_seed`          | Optional integer seed for reproducibility                                   |
| `init_component_method`| Initialization method for component IDs:<br>• `'kmeans++'` (default)<br>• `'random'`<br>• `'chunk'`<br>• `'read'` |
| `verbose`              | If `True`, print log-likelihood summary for each attempt                    |

### Returns

- The `ShapeGMM` instance corresponding to the **best fit** across all attempts (i.e., the highest final log-likelihood).


## Cross-Validation for Model Selection

`shapeGMMTorch` provides a utility function to perform cross-validation across a range of GMM component numbers to guide model selection.

```python
from shapeGMMTorch.utils import cross_validate_component_scan
```

This routine performs k-fold-style cross-validation to evaluate the model log-likelihood on both training and validation sets for multiple values of `n_components`.

#### Parameters

| Parameter              | Type                 | Description |
|------------------------|----------------------|-------------|
| `traj_data`            | `np.ndarray`         | Trajectory data of shape `(n_frames, n_atoms, 3)` |
| `component_array`      | `np.ndarray`         | Array of integers specifying the number of components to scan, e.g. `[1, 2, 3, 4, 5]` |
| `train_fraction`       | `float`              | Fraction of frames used for training (default: `0.9`) |
| `frame_weights`        | `np.ndarray` or `list`, optional | Optional frame weights for likelihood weighting |
| `thresh`               | `float`              | Log-likelihood convergence threshold (default: `1e-3`) |
| `kabsch_thresh`        | `float`              | Convergence threshold for Kabsch alignment (default: `1e-1`) |
| `covar_type`           | `str`                | Type of covariance (`'uniform'` or `'kronecker'`) |
| `n_training_sets`      | `int`                | Number of random train/test splits (default: `3`) |
| `n_attempts`           | `int`                | Number of ShapeGMM fitting attempts per split (default: `10`) |
| `dtype`                | `torch.dtype`        | Tensor precision for PyTorch (default: `torch.float32`) |
| `device`               | `torch.device`       | PyTorch device for computation (default: `"cuda:0"`) |
| `init_component_method`| `str`                | Initialization method: `'kmeans++'`, `'random'`, `'chunk'`, or `'read'` |
| `random_seed`          | `int`, optional      | Seed for reproducibility |
| `verbose`              | `bool`               | Print timing/log-likelihood diagnostics (default: `True`) |

#### Returns

- `train_log_liks` – Log-likelihoods on training sets. Shape: `(len(component_array), n_training_sets)`
- `cv_log_liks` – Log-likelihoods on validation sets. Shape: `(len(component_array), n_training_sets)`

#### Example

```python
component_range = np.arange(1, 6)
train_lls, cv_lls = cross_validate_component_scan(
    traj_data=trajectory_array,
    component_array=component_range,
    covar_type="kronecker",
    n_training_sets=5,
    n_attempts=10,
    init_component_method="kmeans++",
    device=torch.device("cuda:0"),
    dtype=torch.float64,
)
```

You can then use the `train_lls` and `cv_lls` values to help guide the selection of the number component based on the elbow heuristic and deviation between training and cross-validation values.  
