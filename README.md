# shapeGMMTorch

[![CI](https://github.com/mccullaghlab/shapeGMMTorch/actions/workflows/ci.yml/badge.svg)](https://github.com/mccullaghlab/shapeGMMTorch/actions/workflows/ci.yml)
[![PyPI version](https://img.shields.io/pypi/v/shapeGMMTorch.svg)](https://pypi.org/project/shapeGMMTorch/)
[![Examples](https://img.shields.io/badge/examples-notebooks-blue.svg)](https://github.com/mccullaghlab/shapeGMMTorch/tree/main/examples)


## Overview

This is a package to perform Gaussian Mixture Model (GMM) clustering on particle positions (in <img src="https://render.githubusercontent.com/render/math?math=\mathbb{R}^3">). Like other GMM schemes, the user must specify the number of clusters and a cluster initialization scheme (defaults to random).  This is specified in the object initialization line, analagous to how it is done for the sklearn GaussianMixture package.  There are two choices for the form of the covariance  specified by the `covar_type` keyword in the object initialization.  See paper (Klem et al JCTC 2022, https://pubs.acs.org/doi/abs/10.1021/acs.jctc.1c01290) for additional details.

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

Kronecker product covariance (formerly call weighted covariance; spherical, correlated, heterogeneous):

`kron_sgmm = ShapeGMM(n_components, covar_type = 'kronecker', verbose=True)`

During initialization, the following options are availble:

	- n_components (required) - integer number of copmonents must be input
	- covar_type              - string defining the covariance type.  Options are 'kronecker' and 'uniform'.  Defualt is 'uniform'.
	- log_thresh              - float threshold in log likelihood difference to determine convergence. Default value is 1e-3.
	- max_steps               - integer maximum number of steps that the GMM procedure will do.  Default is 200.
	- init_component_method   - string dictating how to initialize components.  Understood values are 'chunk', 'read' and 'random'.  Default is 'random'.
	- sort                    - boolean dictating whether to sort the object by component population after fitting.  Default is True.
	- kabsch_thresh           - float dictating convergence criteria for each iterative alignment (Maximization step).  Default value is 1e-1.
	- dtype                   - Torch data type to be used.  Default is torch.float32.
	- device                  - Torch device to be used.  Default is torch.device('cuda:0') device.
	- verbose                 - boolean dictating whether to print various things at every step. Defualt is False.

### Fit:

A standard fit can be performed in the following manner:

`uni_sgmm.fit(train_positions)`

`kron_sgmm.fit(train_positions)`

where `train_positions` is an array of dimensions `(n_train_frames, n_atoms, 3)`. Notice there is no difference in syntax when fitting the two covariance types.  Two additional options are available during the fit routine which may be necessary under certain situations:

	- component_ids   (optional) - (n_train_frames) integer array of initial component/cluster ids.  This option is necessary if init_component_method = 'read'
	- frame_weights (optional) - (n_train_frames) float array of relative frame weights.  If none are provided the code assumes equal weights for all frames.

If these options are used the fit call looks like

`uni_sgmm.fit(train_positions, component_ids = initial_component_ids, frame_weights = train_frame_weights)`

`kron_sgmm.fit(train_positions, component_ids = initial_component_ids, frame_weights = train_frame_weights)`

### Label and Score:

To label and score (log likelihood per frame) the training set:

for uniform model:

`uni_train_component_ids  = uni_sgmm.predict(train_positions)`

`uni_train_log_likelihood = uni_sgmm.score(train_positions)`

for kronecker model:

`kron_train_component_ids = kron_sgmm.predict(train_positions)`

`kron_train_log_likelihood = kron_sgmm.score(train_positions)`

If the training set has non-uniform frame weights this must be taken into account in the scoring function:

uniform model:

`uni_train_log_likelihood = uni_sgmm.score(train_positions, frame_weights = train_frame_weights)`

kronecker model:

`kron_train_log_likelihood = kron_sgmm.score(train_positions, frame_weights = train_frame_weights)`

### Predict:

Once the shapeGMM object has been fit, it can be used to predict component IDs and log likelihood per frame for a new, or cross validation, trajectory.  The number of atoms must remain the same.  The simple syntax is as follows:

uniform model:

`uni_cv_component_ids  = uni_sgmm.predict(predict_positions)`

`uni_cv_log_likelihood = uni_sgmm.score(predict_positions)`

kronecker model:

`kron_cv_component_ids = kron_sgmm.predict(predict_positions)`

`kron_cv_log_likelihood = kron_sgmm.score(predict_positions)`

where `predict_positions` is an array of dimensions `(n_predict_frames, n_atoms, 3)`. Notice there is no difference in syntax when precicting the two covariance types.  If the predict frames have a non-unifrom frame weight, this can be accounted for in the score function with an additional option 

	- frame_weights (optional) - (n_predict_frames) float array of relative frame weights.  If none are provided the code assumes equal weights for all frames.

If this option is used the predict call will look like

uniform model:

`uni_cv_log_likelihood = uni_sgmm.score(predict_positions, frame_weights = predict_frame_weights)`

kronecker model:

`kron_cv_log_likelihood = kron_sgmm.score(predict_positions, frame_weights = predict_frame_weights)`

## Attributes

After being properly fit, a shapeGMM object will have the following attributes:

	- n_components	             - integer of how many clusters were used in training
	- n_atoms                    - integer of how many atoms were in the training data
	- n_train_frames             - integer of how many frames were in the training data
	- weights_                   - (n_components) float array of cluster weights
	- means_                     - (n_components, n_atoms, 3) float array of cluster centers/averages

Uniform covariance specific attributes

	- vars_		       	         - (n_components) float array of cluster variances

Kronecker covariance specific attributes

	- precisions_	   	         - (n_components, n_atoms, n_atoms) float array of component precisions (inverse covariances)
	- lpdets_	    	         - (n_components) float array of ln(det(covar))


