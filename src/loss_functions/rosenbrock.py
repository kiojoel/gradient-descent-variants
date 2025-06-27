import numpy as np
from .base_function import BaseFunction

class Rosenbrock(BaseFunction):
  """
    The Rosenbrock function, a non-convex function used for performance testing.
    f(x, y) = (a - x)^2 + b * (y - x^2)^2
    The global minimum is at (a, a^2). For a=1, b=100, the minimum is at (1, 1).
    """
  def __init__(self, a: float = 1.0, b: float = 100.0):
    super().__init__()
    self.a = a
    self.b = b
    self.name = f"Rosenbrock (a={a}, b={b})"
    self.global_minimum = np.array([self.a, self.a**2])

  def evaluate(self, params: np.ndarray) -> float:
    x, y = params
    return (self.a - x)**2 + self.b * (y - x**2)**2

  def gradient(self, params: np.ndarray) -> np.ndarray:
    x, y = params
    grad_x = -2 * (self.a - x) - 4 * self.b * (y - x**2) * x
    grad_y = 2 * self.b * (y - x**2)
    return np.array([grad_x, grad_y])