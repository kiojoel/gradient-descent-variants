"""
Quadratic loss functions for testing optimization algorithms.
"""

import numpy as np
from .base_function import BaseFunction


class Quadratic(BaseFunction):
  """
  A simple quadratic function:
  f(x,y) = a*x^2 + b*y^2:
  """

  def __init__(self, a: float = 1.0, b: float = 10.0):
    super().__init__()
    self.a = a
    self.b = b
    self.name = f"Quadratic (a={a}, b={b})"

  def evaluate(self, params:np.ndarray) -> float:
    """
    f(x,y) = a*x^2 + b*y^2
    """
    x, y = params
    return self.a * x**2 + self.b * y**2

  def gradient(self, params:np.ndarray) -> np.ndarray:
    """ Gradient is [2*a*x, 2*b*y]"""
    x,y = params
    return np.array([2 * self.a * x, 2 * self.b * y])