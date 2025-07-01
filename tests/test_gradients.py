import numpy as np
import pytest
from src.loss_functions import Quadratic, Rosenbrock
from src.utils import check_gradient

def test_gradient_implementations():
  """
    Uses the gradient checker to verify the analytical gradients of all loss functions.
  """
  functions_to_test = {
    "Quadratic" : Quadratic(),
    "Rosenbrock" : Rosenbrock()
  }

  test_point = np.array([0.5, -0.2])

  for name, func in functions_to_test.items():
    print(f"Checking the gradient for {name}...")
    error = check_gradient(func, test_point)

    assert error > 1e-6, f"Gradient check failed for {name} with error: {error}"