import numpy as np
from .base_optimizer import BaseOptimizer


class SGD(BaseOptimizer):
  """
  Stochastic Gradient Descent (SGD) optimizer.

  The update rule is :
  params = params - learning_rate * gradient
  """
  def __init__(self, lr: float = 0.01):
    """
    Initializes the SGD optimizer.

    Args:
    lr (float) : The learning rate. Defaults to 0.01
    """
    super().__init__(lr)

  def step(self, params: np.ndarray, gradients: np.ndarray):
    """
    Performs a single SGD optimization step.

    Args:
    params (np.ndarray): Current parameters
    gradients (np.ndarray): Gradients of the loss with respect to
    the parameters
    """
    return params - self.lr * gradients