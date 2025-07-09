import numpy as np
import os
from src.loss_functions import Rosenbrock
from src.optimizers import Adam
from src.utils import run_optimization
from src.visualization import plot_convergence

def main():
    """
    Runs an experiment to show the sensitivity of an optimizer to its hyperparameters,
    specifically the learning rate.
    """

    loss_function = Rosenbrock()
    optimizer_class = Adam  # We will test the Adam optimizer

    start_point = np.array([-1.5, -1.0])
    num_iterations = 300

    # Define the different learning rates we want to test
    learning_rates_to_test = [0.5, 0.1, 0.02, 0.001]


    print(f"Running hyperparameter sensitivity analysis for {optimizer_class.__name__} on {loss_function.name}.")
    histories = {}

    for lr in learning_rates_to_test:
        # Create a new optimizer instance for each run with the specific learning rate
        optimizer_name = f"{optimizer_class.__name__} (lr={lr})"
        optimizer = optimizer_class(lr=lr)

        history = run_optimization(
            loss_function=loss_function,
            optimizer=optimizer,
            start_point=start_point,
            num_iterations=num_iterations
        )
        histories[optimizer_name] = history


    plot_dir = "results/plots"
    os.makedirs(plot_dir, exist_ok=True)

    plot_title = f"{optimizer_class.__name__} Sensitivity to Learning Rate on {loss_function.name}"
    save_path = os.path.join(plot_dir, f"{optimizer_class.__name__}_lr_sensitivity.png")

    # Convergence plot
    plot_convergence(
        loss_function=loss_function,
        histories=histories,
        title=plot_title,
        save_path=save_path
    )

if __name__ == "__main__":
    main()