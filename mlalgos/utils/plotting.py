import numpy as np
import matplotlib.pyplot as plt
from typing import Union


ORANGE, BLUE, PURPLE = '#E69F00', '#56B4E9', '#A020F0'
GRAY1, GRAY4 = '#231F20', '#646369'


def plot_cv_results(
    cv_means: list[float],
    cv_stds: list[float],
    x_axis: Union[list, np.array],
    x_label: str,
) -> plt.figure:
    """
    Produces the plot for cross-validation results with the best model marked.

    Args:
        cv_means              (list[float]): List of mean errors for the models.
        cv_stds               (list[float]): List of std of errors for the models. Must be of same length as cv_means.
        x_axis      (Union[list, np.array]): Values to place on x axis. Must be of same length as cv_means and cv_stds.
        x_label                       (str): Label to display on x axis.
    
    Returns:
        plt.figure: The generated figure.
    """
    # Compute the chosen model as the one that is the smallest that is within
    # one standard deviation of the best model's mean error
    best = np.argmin(cv_means)
    chosen = 0
    while (cv_means[chosen] > cv_means[best] + cv_stds[best]):
        chosen += 1

    # Create plot
    fig, ax = plt.subplots(figsize=(4,3))
    ax.plot(x_axis, cv_means, c=ORANGE, linewidth=0.8)
    ax.errorbar(x_axis, cv_means, color=ORANGE, linestyle=None, marker='o',
                elinewidth=0.8, markersize=3, yerr=cv_stds, ecolor=BLUE, capsize=3)
    ax.axhline(cv_means[best] + cv_stds[best], c=PURPLE, linewidth=0.8, linestyle='--')
    ax.axvline(x_axis[chosen], c=PURPLE, linewidth=0.8, linestyle='--')
    ax.set_xlabel(x_label)
    ax.set_ylabel('CV Error')
    fig.tight_layout()
    return fig
