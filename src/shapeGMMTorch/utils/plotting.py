import matplotlib.pyplot as plt
import numpy as np

def plot_log_likelihood_with_dd(
    axis,
    component_array,
    train_log_lik,
    valid_log_lik,
    fontsize: int = 16,
    xlabel: bool = True,
    ylabel1: bool = True,
    ylabel2: bool = True,
    legend: bool = True
):
    """
    Plot average log-likelihood per frame (with standard deviation) as a function of the number
    of components for training and cross-validation data, along with the second derivative of the training log-likelihood.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        The axis on which to plot.
    component_array : np.ndarray of shape (n_components,)
        Array of component counts (x-axis values).
    train_log_lik : np.ndarray of shape (n_components, n_splits)
        Log-likelihood values for training data across multiple runs/splits.
    valid_log_lik : np.ndarray of shape (n_components, n_splits)
        Log-likelihood values for validation data across multiple runs/splits.
    fontsize : int, optional
        Font size for labels and ticks. Default is 16.
    xlabel : bool, optional
        Whether to label the x-axis. Default is True.
    ylabel1 : bool, optional
        Whether to label the left y-axis (log-likelihood). Default is True.
    ylabel2 : bool, optional
        Whether to label the right y-axis (second derivative). Default is True.
    legend : bool, optional
        Whether to display a legend. Default is True.
    """

    # --- Validate inputs ---
    assert isinstance(component_array, np.ndarray), "component_array must be a NumPy array"
    assert component_array.ndim == 1, "component_array must be 1D"
    assert train_log_lik.shape == valid_log_lik.shape, "train and valid log-likelihood arrays must match in shape"
    assert train_log_lik.shape[0] == len(component_array), "component_array length must match first axis of log-likelihood arrays"

    # --- Colors ---
    colors = {
        "train": "tab:blue",
        "valid": "tab:green",
        "second_derivative": "tab:red"
    }

    # --- Mean and Std ---
    train_mean = np.mean(train_log_lik, axis=1)
    train_std = np.std(train_log_lik, axis=1)
    valid_mean = np.mean(valid_log_lik, axis=1)
    valid_std = np.std(valid_log_lik, axis=1)

    # --- Primary plot (log-likelihood) ---
    axis.errorbar(
        component_array, train_mean, train_std,
        fmt='-o', lw=2, capsize=3, color=colors["train"], label="Training"
    )
    axis.errorbar(
        component_array, valid_mean, valid_std,
        fmt='--x', lw=2, capsize=3, color=colors["valid"], label="Cross-Validation"
    )

    # --- Second derivative on secondary axis ---
    ax2 = axis.twinx()
    dd = np.gradient(np.gradient(train_log_lik, axis=0), axis=0)
    dd_mean = np.mean(dd, axis=1)
    dd_std = np.std(dd, axis=1)

    ax2.errorbar(
        component_array, dd_mean, dd_std,
        fmt='-s', lw=1.5, capsize=3, alpha=0.75, color=colors["second_derivative"]
    )

    # --- Labels and formatting ---
    axis.grid(True, linestyle='--', color='gray', alpha=0.4)

    if ylabel1:
        axis.set_ylabel("Log Likelihood per Frame", fontsize=fontsize)
    if ylabel2:
        ax2.set_ylabel("Second Derivative", fontsize=fontsize, color=colors["second_derivative"])
        ax2.tick_params(axis='y', labelcolor=colors["second_derivative"])

    if xlabel:
        axis.set_xlabel("Number of Components", fontsize=fontsize)

    axis.tick_params(labelsize=fontsize)
    ax2.tick_params(labelsize=fontsize)

    if legend:
        axis.legend(fontsize=fontsize, loc="best")

