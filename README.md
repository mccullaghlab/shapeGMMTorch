# shapeGMMTorch

## Overview

This is a package to perform Gaussian Mixture Model (GMM) clustering on particle positions (in <img src="https://render.githubusercontent.com/render/math?math=\mathbb{R}^3">). Like other GMM schemes, the user must specify the number of clusters and a cluster initialization scheme (defaults to random).  This is specified in the object initialization line, analagous to how it is done for the sklearn GaussianMixture package.  There are two choices for the form of the covariance  specified by the `covar_type` keyword in the object initialization.  See paper (Klem et al JCTC 2022, https://pubs.acs.org/doi/abs/10.1021/acs.jctc.1c01290) for additional details.

## Dependencies

This package is dependent on the following packages:

1. Python>=3.6 
2. numpy
3. torch>=1.11 (==1.11 if option 4 is used)
4. Optional: torch_batch_svd available from https://github.com/KinglittleQ/torch-batch-svd

The last package is for the SVD part of the alignment and is much faster than the native batch torch library.  It is, however, not compatible with the current version of torch (1.12) thus the requirement of torch 1.11.

The examples are also dependent on:

1. MDAnalysis
2. matplotlib
3. pyemma
4. shapeGMM

## Installation

After the dependencies have been installed, the package can be installed from pip

`pip install shapeGMMTorch`

or by downloading from github and then running

`python setup.py install`

### Installation on a MAC

I have found that PyTorch does not work on a mac using standard `pip install torch` or conda.  Instead, you have to create a special `nomkl` conda environment and install all necessary packages in there.  There are various blogs and/or stack overflow threads on this.

After the `nomkl` conda environment has been created and all dependencies have been added to that environment, shapeGMMTorch can be installed.  Only the `torch.device("cpu")` device will work on a mac, however.

## Usage 

This package is designed to mimic the usage of the sklearn package.  You first initiliaze the object and then fit.  Predict can be done once the model is fit.  Fit and ppredict functions take particle position trajectories as input in the form of a `(n_frames, n_atoms, 3)` numpy array.

### Initialize:

`from shapeGMMTorch import torch_sgmm`

Uniform covariance (spherical, uncorrelated, homogeneous):

`usgmm = torch_sgmm.ShapeGMMTorch(n_clusters, covar_type = 'uniform', verbose=True)`

Kronecker product covariance (formerly call weighted covariance; spherical, correlated, heterogeneous):

`wsgmm = torch_sgmm.ShapeGMMTorch(n_clusters, covar_type = 'kronecker', verbose=True)`

During initialization, the following options are availble:

	- n_clusters (required)   - integer number of clusters must be input
	- covar_type              - string defining the covariance type.  Options are 'kronecker' and 'uniform'.  Defualt is 'uniform'.
	- log_thresh              - float threshold in log likelihood difference to determine convergence. Default value is 1e-3.
	- max_steps               - integer maximum number of steps that the GMM procedure will do.  Default is 200.
	- init_cluster_method     - string dictating how to initialize clusters.  Understood values are 'chunk', 'read' and 'random'.  Default is 'random'.
	- sort                    - boolean dictating whether to sort the object by cluster population after fitting.  Default is True.
	- kabsch_thresh           - float dictating convergence criteria for each iterative alignment (Maximization step).  Default value is 1e-1.
	- dtype                   - Torch data type to be used.  Default is torch.float32.
	- device                  - Torch device to be used.  Default is torch.device('cuda:0') device.
	- verbose                 - boolean dictating whether to print various things at every step. Defualt is False.

### Fit:

A standard fit can be performed in the following manner:

`uniform_aligned_trajectory = usgmm.fit(train_positions)`

`kronecker_aligned_trajectory = wsgmm.fit(train_positions)`

where `train_positions` is an array of dimensions `(n_train_frames, n_atoms, 3)`. Notice there is no difference in syntax when fitting the two covariance types.  Two additional options are available during the fit routine which may be necessary under certain situations:

	- cluster_ids   (optional) - (n_train_frames) integer array of initial cluster ids.  This option is necessary if init_cluster_method = 'read'
	- frame_weights (optional) - (n_train_frames) float array of relative frame weights.  If none are provided the code assumes equal weights for all frames.

If these options are used the fit call looks like

`uniform_aligned_trajectory = usgmm.fit(train_positions, cluster_ids = initial_cluster_ids, frame_weights = train_frame_weights)`

`kronecker_aligned_trajectory = wsgmm.fit(train_positions, cluster_ids = initial_cluster_ids, frame_weights = train_frame_weights)`

### Predict:

Once the shapeGMM object has been fit, it can be used to precict cluster IDs, aligned trajectory, and log likelihood per frame for a new, or cross validation, trajectory.  The number of atoms must remain the same.  The simple syntax is as follows:

`cluster_ids, aligned_traj, log_likelihood = usgmm.predict(predict_positions)`

`clusters_ids, aligned_traj, log_likelihood = wsgmm.predict(predict_positions)`

where `predict_positions` is an array of dimensions `(n_predict_frames, n_atoms, 3)`. Notice there is no difference in syntax when precicting the two covariance types.  If the predict frames have a non-unifrom frame weight, this can be accounted for  

	- frame_weights (optional) - (n_predict_frames) float array of relative frame weights.  If none are provided the code assumes equal weights for all frames.

If this option is used the predict call will look like

`cluster_ids, aligned_traj, log_likelihood = usgmm.predict(predict_positions, frame_weights = predict_frame_weights)`

`clusters_ids, aligned_traj, log_likelihood = wsgmm.predict(predict_positions, frame_weights = predict_frame_weights)`

## Attributes

After being properly fit, a shapeGMM object will have the following attributes:

	- n_clusters	    - integer of how many clusters were used in training
	- n_atoms           - integer of how many atoms were in the training data
	- n_train_frames    - integer of how many frames were in the training data
	- clusters_ids      - integer array of cluster ids for training data
	- log_likelihood    - float log likelihood of training set
	- weights           - (n_clusters) float array of cluster weights
	- centers	        - (n_clusters, n_atoms, 3) float array of cluster centers/averages

Uniform covariance specific attributes

	- vars		       	- (n_clusters) float array of cluster variances

Kronecker covariance specific attributes

	- precisions	   	- (n_clusters, n_atoms, n_atoms) float array of cluster precisions (inverse covariances)
	- lpdets	    	- (n_clusters) float array of ln(det(covar))


