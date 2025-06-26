from abc import ABC, abstractmethod
import numpy as np


class BaseFunction(ABC):
  def __init__(self):
    self.name = self.__class__.__name__

  @abstractmethod
  def evaluate(self, params: np.ndarray) -> float:
    """
    Evaluate the function at a given point.

    Args:
    params (np.ndarray): The point (e.g., [x,y]) at which to
    evaluate the function.

    Returns:
    float: The value of the function.
    """
    pass

  @abstractmethod
  def gradient(self, params: np.ndarray) -> np.ndarray:
    """
    Calculate the gradient of the function
    at a given point.

    Args:
    params (np.ndarray): The point (e.g., [x,y]) at which to
    evaluate the gradient.

    Returns:
    np.ndarray: The gradient of the vector.
    """
    pass

  def __call__(self, params: np.ndarray) -> float:
    """
    Allows the object to be called like a function
    """
    return self.evaluate(params)