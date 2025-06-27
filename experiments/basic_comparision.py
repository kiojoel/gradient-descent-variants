import numpy as np
import os
from src.loss_functions import Rosenbrock, Quadratic
from src.optimizers import SGD, Adam, RMSprop
from src.utils.experiment_runner import run_optimization
from src.visualization.contour_plots import plot_contour

def main():
    """
    Main function to run the basic comparison experiment.
    """
    # --- Experiment Configuration ---
    # Choose loss function: Rosenbrock() or Quadratic()
    loss_function = Quadratic()

    # Common settings
    start_point = np.array([-1.5, -1.0])
    num_iterations = 500

    # Define optimizer configurations, not instances
    optimizer_configs = {
        "SGD (lr=0.001)":    {"class": SGD, "params": {"lr": 0.001}},
        "RMSprop (lr=0.01)": {"class": RMSprop, "params": {"lr": 0.01}},
        "Adam (lr=0.02)":    {"class": Adam, "params": {"lr": 0.02}},
    }

    # --- Run Experiments ---
    histories = {}
    for name, config in optimizer_configs.items():
        # Create a new optimizer instance for each run to ensure no state is shared
        optimizer = config["class"](**config["params"])

        history = run_optimization(
            loss_function=loss_function,
            optimizer=optimizer,
            start_point=start_point,
            num_iterations=num_iterations
        )
        histories[name] = history

    # --- Visualize Results ---
    plot_title = f"Optimizer Comparison on {loss_function.name}"

    # Define plot ranges based on the function
    if isinstance(loss_function, Rosenbrock):
        x_range, y_range = (-2, 2), (-1, 3)
    else: # Quadratic
        x_range, y_range = (-6, 6), (-6, 6)

    # Create results directory if it doesn't exist
    os.makedirs("results/plots", exist_ok=True)
    save_path = f"results/plots/{loss_function.__class__.__name__}_comparison.png"

    plot_contour(
        loss_function=loss_function,
        histories=histories,
        x_range=x_range,
        y_range=y_range,
        title=plot_title,
        save_path=save_path
    )

if __name__ == "__main__":
    # This allows the script to be run with `python -m experiments.basic_comparison`
    main()