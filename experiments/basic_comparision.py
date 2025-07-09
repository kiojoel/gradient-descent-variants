import numpy as np
import os
from src.loss_functions import Rosenbrock, Quadratic
from src.optimizers import SGD, Adam, RMSprop
from src.utils.experiment_runner import run_optimization
from src.visualization import plot_contour, plot_surface, plot_convergence

def main():
    """
    Main function to run the basic comparison experiment.
    """
    # Experiment Configuration
    # Choose loss function: Rosenbrock() or Quadratic()
    #loss_function = Quadratic()
    loss_function = Rosenbrock()

    # Common settings
    start_point = np.array([-1.5, -1.0])
    num_iterations = 500

    # Define optimizer configurations, not instances
    optimizer_configs = {
        "SGD (lr=0.001)":    {"class": SGD, "params": {"lr": 0.001}},
        "RMSprop (lr=0.01)": {"class": RMSprop, "params": {"lr": 0.01}},
        "Adam (lr=0.02)":    {"class": Adam, "params": {"lr": 0.02}},
    }

    # Run Experiments
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

    #  Visualize Results
    plot_dir = "results/plots"
    os.makedirs(plot_dir, exist_ok=True)

    # Generate and save contour plot
    contour_title = f"Optimizer Comparison on {loss_function.name}"
    contour_save_path = os.path.join(plot_dir, f"{loss_function.__class__.__name__}_contour.png")


    if isinstance(loss_function, Rosenbrock):
        x_range, y_range = (-2, 2), (-1, 3)
    else: # Quadratic
        x_range, y_range = (-6, 6), (-6, 6)

    plot_contour(
        loss_function=loss_function,
        histories=histories,
        x_range=x_range,
        y_range=y_range,
        title=contour_title,
        save_path=contour_save_path
    )

    convergence_title = f"Optimizer Comparison on {loss_function.name}"
    convergence_save_path = os.path.join(plot_dir, f"{loss_function.__class__.__name__}_convergence.png")

    plot_convergence(
        loss_function=loss_function,
        histories=histories,
        title=convergence_title,
        save_path=convergence_save_path
    )

    if isinstance(loss_function, Rosenbrock):
        plot_surface(loss_function, histories, x_range, y_range,
                     f"3D View of Optimization on {loss_function.name}",
                     os.path.join(plot_dir, f"{loss_function.__class__.__name__}_surface.png"))

if __name__ == "__main__":
    main()