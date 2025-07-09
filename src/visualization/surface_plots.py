import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Dict
from ..loss_functions import BaseFunction

def plot_surface(
    loss_function: BaseFunction,
    histories: Dict[str, np.ndarray],
    x_range: tuple,
    y_range: tuple,
    title: str,
    save_path: str = None
):
    """
    Creates a 3D surface plot of a loss function and overlays optimization paths.

    Args:
        loss_function (BaseFunction): The loss function to plot.
        histories (Dict[str, np.ndarray]): Dictionary of optimizer names to their paths.
        x_range (tuple): The range for the x-axis.
        y_range (tuple): The range for the y-axis.
        title (str): The title of the plot.
        save_path (str, optional): Path to save the figure. If None, shows the plot.
    """
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(projection='3d')

    # Create the grid for the surface plot
    x = np.linspace(x_range[0], x_range[1], 100)
    y = np.linspace(y_range[0], y_range[1], 100)
    X, Y = np.meshgrid(x, y)

    # Calculate Z values (loss) for each point on the grid
    Z = np.vectorize(lambda x, y: loss_function.evaluate(np.array([x, y])))(X, Y)

    # Plot the surface with some transparency
    ax.plot_surface(X, Y, Z, cmap='viridis_r', alpha=0.75, rstride=1, cstride=1, edgecolor='none')

    # Plot the optimization paths on top of the surface
    for name, history in histories.items():

        z_path = np.array([loss_function.evaluate(p) for p in history])

        ax.plot(history[:, 0], history[:, 1], z_path + 0.1, 'o-', label=name, markersize=3, linewidth=1.5)

    # Set labels and title
    ax.set_xlabel('x-parameter', fontsize=12, labelpad=10)
    ax.set_ylabel('y-parameter', fontsize=12, labelpad=10)
    ax.set_zlabel('Loss Value', fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=18)
    ax.legend()


    ax.view_init(elev=25, azim=125)

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Surface plot saved to {save_path}")
    else:
        plt.show()