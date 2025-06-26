from abc import ABC, abstractmethod
import numpy as np


class BaseOptimizer(ABC):
  def __init__(self, lr: float):
    """
    Initializes the optimizer.

    Args:
    lr(float): The learning rate.
    """
    if lr <= 0:
      raise ValueError("Learning rate must be positive.")
    self.lr = lr
    self.name = self.__class__.__name__

  @abstractmethod
  def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
      """
      Performs a single optimization step.

      Args:
      params (np.ndarray): Current parameters.
      gradients (np.ndarray) : Gradients of the loss with respect to
      the parameters.

      Returns:
      np.ndarray : Updated parameters.
      """
      pass
