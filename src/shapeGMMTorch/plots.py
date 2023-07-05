
import matplotlib.pyplot as plt
import numpy as np

#make a plot of log likelihood as a function of number of clusters along with the second derivative of that curve
def plot_log_likelihood_with_dd(axis,cluster_array,train_log_lik,valid_log_lik,fontsize=16,xlabel=True,ylabel1=True,ylabel2=True):
    """
    Plot log likelihood as a function of number of clusters for both training and cross validation.  These will be computed as the average over provided training sets.  Also compute and plot the second derivative of the training set data.
    Inputs:
        axis                    (required)  : matplotlib axis object
        cluster_array           (required)  : (n_clusters) int numpy array of number of clusters (x-values for this plot)
        train_log_lik           (required)  : (n_clusters, n_training_sets) float numpy array of log likelihood for training sets
        valid_log_lik           (required)  : (n_clusters, n_training_sets) float numpy array of log likelihood for CV sets
        fontsize                (optional)  : int defining fontsize to be used in the ploat, default is 16
        xlabel                  (optional)  : boolean defining whether or not to put an xlabel, default is True
        ylabel1                 (optional)  : boolean defining whether or not to put a left ylabel, default is True
        ylabel2                 (optional)  : boolean defining whether or not to put a right ylabel, default is True
    """
    colors = ["tab:blue", "tab:red"]
    # Training Data
    train_mean = np.mean(train_log_lik,axis=1)
    train_std = np.std(train_log_lik,axis=1)
    axis.errorbar(cluster_array,train_mean,train_std,fmt='-o',lw=3,capsize=3,c=colors[0],label="Training")
    # Validation
    valid_mean = np.mean(valid_log_lik,axis=1)
    valid_std = np.std(valid_log_lik,axis=1)
    axis.errorbar(cluster_array,valid_mean,valid_std,fmt='--x',lw=3,capsize=3,c=colors[0],label="CV")
    # Second derivative
    ax2 = axis.twinx()
    n_samples = train_log_lik.shape[1]
    n_clusters = train_log_lik.shape[0]
    dd = np.empty(train_log_lik.shape)
    for sample in range(n_samples):
        dd[:,sample] = np.gradient(np.gradient(train_log_lik[:,sample]))
    dd_mean = np.mean(dd,axis=1)
    dd_std = np.std(dd,axis=1)
    ax2.errorbar(cluster_array,dd_mean,dd_std,fmt='-o',lw=2,capsize=3,alpha=0.75,c=colors[1])
    #
    axis.grid(which='major', axis='both', color='#808080', linestyle='--')
    if ylabel1==True:
        axis.set_ylabel("Log Likelihood per Frame",fontsize=fontsize)
    if ylabel2==True:
        ax2.set_ylabel("Second Derivative",fontsize=fontsize,color=colors[1])
    if xlabel==True:
        axis.set_xlabel("Number of Clusters",fontsize=fontsize)
    ax2.tick_params(axis='both',labelsize=fontsize,labelcolor=colors[1])
    axis.tick_params(axis='both',labelsize=fontsize)
    axis.legend(fontsize=fontsize)

