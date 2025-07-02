import numpy as np
import matplotlib.pyplot as plt
from typing import Dict
from ..loss_functions import BaseFunction

def plot_convergence(
    loss_function: BaseFunction,
    histories: Dict[str, np.ndarray],
    title: str,
    log_scale: bool = True,
    save_path: str = None
):
    """
    Plots the loss value over iterations for different optimizers.

    Args:
        loss_function (BaseFunction): The loss function used in the optimization.
        histories (Dict[str, np.ndarray]): A dictionary where keys are optimizer names
                                           and values are the optimization paths (a series of parameters).
        title (str): The title for the plot.
        log_scale (bool): Whether to use a logarithmic scale for the y-axis (loss).
                          Defaults to True, which is useful for seeing large changes in loss.
        save_path (str, optional): Path to save the figure. If None, shows the plot.
    """
    plt.figure(figsize=(12, 8))

    for name, history in histories.items():
        # Calculate the loss at each step in the history
        loss_values = [loss_function.evaluate(params) for params in history]
        plt.plot(loss_values, label=name)

    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)

    if log_scale:
        plt.yscale('log')
        plt.ylabel('Loss Value (log scale)', fontsize=12)
        plt.grid(True, which="both", ls="--", alpha=0.6)
    else:
        plt.grid(True, ls="--", alpha=0.6)

    plt.title(title, fontsize=16)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Convergence plot saved to {save_path}")
    else:
        plt.show()