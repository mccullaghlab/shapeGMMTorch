# shapeGMMTorch

## Overview

This is a package to perform Gaussian Mixture Model (GMM) clustering on particle positions (in <img src="https://render.githubusercontent.com/render/math?math=\mathbb{R}^3">). Like other GMM schemes, the user must specify the number of clusters and a cluster initialization scheme (defaults to random).  This is specified in the object initialization line, analagous to how it is done for the sklearn GaussianMixture package.  There are two choices for the form of the covariance  specified by the `covar_type` keyword in the object initialization.  See paper (Klem et al JCTC 2022, https://pubs.acs.org/doi/abs/10.1021/acs.jctc.1c01290) for additional details.

## Dependencies

This package is dependent on the following packages:

1. numpy
2. torch==1.11
3. torch_batch_svd  available from https://github.com/KinglittleQ/torch-batch-svd

The last package is for the SVD part of the alignment and is much faster than the native batch torch library.  It is, however, not compatible with the current version of torch (1.12) thus the requirement of torch 1.11.

## Installation

After the dependencies have been installed, the package can be installed by downloading and then running

`python setup.py install`

## Usage 

This package is designed to mimic the usage of the sklearn package.  You first initiliaze the object and then fit.  Predict can be done once the model is fit.  Fit and ppredict functions take particle position trajectories as input in the form of a `(n_frames, n_atoms, 3)` numpy array.

### Initialize:

`from shapeGMMTorch import torch_sgmm`

Uniform (spherical, uncorrelated) covariance:

`usgmm = torch_sgmm.ShapeGMMTorch(n_clusters, covar_type = 'uniform', verbose=True)`

Weighted (Kronecker product) covariance:

`wsgmm = torch_sgmm.ShapeGMMTorch(n_clusters, covar_type = 'kronecker', verbose=True)`

### Fit:

`usgmm.fit(training_set_positions)`

`wsgmm.fit(training_set_positions)`

### Predict:


`clusters, aligned_traj, log_likelihood = usgmm.predict(full_trajectory_positions)`

`clusters, aligned_traj, log_likelihood = wsgmm.predict(full_trajectory_positions)`

## Options

## Description of Contents

## Test Cases

