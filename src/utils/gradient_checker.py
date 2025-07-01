import numpy as np
from ..loss_functions import BaseFunction


def check_gradient(loss_function:BaseFunction, params:np.ndarray,epsilon:float=1e-5) -> float:
  """
    Checks the correctness of a function's gradient implementation using the finite difference method.

    It computes the numerical gradient and compares it to the analytical gradient.

    Args:
        loss_function (BaseFunction): The loss function object to test.
        params (np.ndarray): The point at which to check the gradient.
        epsilon (float): A small perturbation value for the finite difference calculation.

    Returns:
        float: The relative difference between the analytical and numerical gradients.
               A value close to zero indicates a correct implementation.
    """

  analytic_grad =  loss_function.gradient(params)
  numerical_grad = np.zeros_like(params)

  for i in range(len(params)):
    params_plus = np.copy(params)
    params_plus[i] += epsilon

    params_minus = np.copy(params)
    params_minus[i] -= epsilon

    loss_plus = loss_function.evaluate(params_plus)
    loss_minus = loss_function.evaluate(params_minus)

    # Finite difference formula
    numerical_grad[i] = (loss_plus - loss_minus) / (2 * epsilon)

    # Compare the two gradients
    numerator = np.linalg.norm(analytic_grad - numerical_grad)
    denominator = np.linalg.norm(analytic_grad) + np.linalg.norm(numerical_grad)

    if denominator == 0:
      return 0.0

    difference = numerator / denominator
    return difference