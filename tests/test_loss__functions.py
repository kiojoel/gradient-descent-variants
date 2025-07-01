import numpy as np
import pytest
from src.loss_functions import Quadratic, Rosenbrock


def test_rosenbrock_at_minimum():
   """
    Tests the Rosenbrock function's gradient at its global minimum.
    For a=1, b=100, the minimum is at (1, 1).
    """
   rosen = Rosenbrock(a=1, b=100)
   minimum_point = np.array([1.0, 1.0])
   gradient_at_min = rosen.gradient(minimum_point)

   # At the minimum point the gradient vector should be [0, 0]
   assert np.allclose(gradient_at_min, [0.0, 0.0])


def test_quadratic_at_minimum():
    """
    Tests the Quadratic function's gradient at its global minimum (0, 0).
    """
    quad = Quadratic()
    minimum_point = np.array([0.0, 0.0])
    gradient_at_min = quad.gradient(minimum_point)

    assert np.allclose(gradient_at_min, [0.0, 0.0])