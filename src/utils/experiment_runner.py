import numpy as np
from tqdm import tqdm
from ..loss_functions import BaseFunction
from ..optimizers import BaseOptimizer

def run_optimization(
    loss_function: BaseFunction,
    optimizer: BaseOptimizer,
    start_point: np.ndarray,
    num_iterations: int
) -> np.ndarray:
    """
    Runs an optimization process for a given number of iterations.

    Args:
        loss_function (BaseFunction): The function to optimize.
        optimizer (BaseOptimizer): The optimizer to use.
        start_point (np.ndarray): The initial parameters.
        num_iterations (int): The number of steps to run.

    Returns:
        np.ndarray: A history of the parameter values at each step.
    """
    params = np.copy(start_point)
    history = [params]

    for _ in tqdm(range(num_iterations), desc=f"Optimizing with {optimizer.name}"):
        gradients = loss_function.gradient(params)
        params = optimizer.step(params, gradients)
        history.append(params)

    return np.array(history)