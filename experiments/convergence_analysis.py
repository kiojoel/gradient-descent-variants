import numpy as np
import os
import pandas as pd
from src.loss_functions import Rosenbrock, Quadratic
from src.optimizers import SGD, Adam, RMSprop
from src.utils import run_optimization

def main():
    """
    Runs a quantitative analysis of optimizer convergence.
    It collects data on final loss and distance to the minimum and saves it to a CSV file.
    """
    # Experiment Configuration
    # We use Rosenbrock because it's a good differentiator
    loss_function = Rosenbrock()

    start_point = np.array([-1.5, -1.0])
    num_iterations = 2000

    optimizer_configs = {
        "SGD":     {"class": SGD, "params": {"lr": 0.001}},
        "RMSprop": {"class": RMSprop, "params": {"lr": 0.01}},
        "Adam":    {"class": Adam, "params": {"lr": 0.02}},
    }

    # Run Experiments and Collect Data
    print(f"Running convergence analysis on {loss_function.name} for {num_iterations} iterations.")

    results_data = []

    for name, config in optimizer_configs.items():
        print(f"Running optimizer: {name}...")

        optimizer = config["class"](**config["params"])
        history = run_optimization(
            loss_function=loss_function,
            optimizer=optimizer,
            start_point=start_point,
            num_iterations=num_iterations
        )

        # Get the final parameters from the history
        final_params = history[-1]

        # Calculate the final loss
        final_loss = loss_function.evaluate(final_params)

        # Calculate the final distance to the true minimum
        if hasattr(loss_function, 'global_minimum'):
            distance_to_minimum = np.linalg.norm(final_params - loss_function.global_minimum)
        else:
            distance_to_minimum = np.nan # Use Not-a-Number if minimum is unknown

        # Store the results
        results_data.append({
            "optimizer": name,
            "learning_rate": config["params"]["lr"],
            "iterations": num_iterations,
            "final_loss": final_loss,
            "distance_to_minimum": distance_to_minimum,
            "final_params_x": final_params[0],
            "final_params_y": final_params[1],
        })

    # Save Results to a CSV File

    data_dir = "results/data"
    os.makedirs(data_dir, exist_ok=True)

    # Convert the list of dictionaries to a pandas DataFrame
    results_df = pd.DataFrame(results_data)

    # Save the DataFrame to a CSV file
    save_path = os.path.join(data_dir, "convergence_analysis_results.csv")
    results_df.to_csv(save_path, index=False)

    print("\nAnalysis Results ")
    print(results_df)
    print(f"\nResults saved to {save_path}")

if __name__ == "__main__":
    main()