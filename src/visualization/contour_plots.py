import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
from ..loss_functions import BaseFunction

def plot_contour(
    loss_function: BaseFunction,
    histories: Dict[str, np.ndarray],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    title: str,
    save_path: str = None
):
    """
    Creates a contour plot of a loss function and overlays optimization paths.

    Args:
        loss_function (BaseFunction): The loss function to plot.
        histories (Dict[str, np.ndarray]): A dictionary where keys are optimizer names
                                           and values are the optimization paths.
        x_range (Tuple[float, float]): The range for the x-axis.
        y_range (Tuple[float, float]): The range for the y-axis.
        title (str): The title of the plot.
        save_path (str, optional): Path to save the figure. If None, shows the plot.
    """
    x = np.linspace(x_range[0], x_range[1], 400)
    y = np.linspace(y_range[0], y_range[1], 400)
    X, Y = np.meshgrid(x, y)

    # Vectorize the evaluation for performance
    Z = np.vectorize(lambda x, y: loss_function.evaluate(np.array([x, y])))(X, Y)

    plt.figure(figsize=(12, 8))
    # Use log scale for contours to see details in steep functions like Rosenbrock
    plt.contourf(X, Y, Z, levels=np.logspace(0, 5, 35), cmap='viridis_r', alpha=0.8)
    plt.colorbar(label='Loss Value (log scale)')

    for name, history in histories.items():
        plt.plot(history[:, 0], history[:, 1], 'o-', label=name, markersize=3, linewidth=1.5)

    # Plot start and end points
    start_point = list(histories.values())[0][0]
    plt.plot(start_point[0], start_point[1], 'go', markersize=10, label='Start')

    # If the function has a known minimum, plot it
    if hasattr(loss_function, 'global_minimum'):
        min_point = loss_function.global_minimum
        plt.plot(min_point[0], min_point[1], 'r*', markersize=15, label='Global Minimum')

    plt.title(title, fontsize=16)
    plt.xlabel('x-parameter', fontsize=12)
    plt.ylabel('y-parameter', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Plot saved to {save_path}")
    else:
        plt.show()