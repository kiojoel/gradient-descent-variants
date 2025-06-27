import numpy as np
from .base_optimizer import BaseOptimizer

class RMSprop(BaseOptimizer):
    """
    RMSprop optimizer.
    Maintains a moving average of the square of gradients.
    """
    def __init__(self, lr: float = 0.01, beta: float = 0.9, epsilon: float = 1e-8):
        """
        Initializes the RMSprop optimizer.

        Args:
            lr (float): The learning rate.
            beta (float): The decay rate for the moving average of squared gradients.
            epsilon (float): A small constant for numerical stability.
        """
        super().__init__(lr)
        if not (0.0 <= beta < 1.0):
            raise ValueError("Beta must be in [0, 1).")
        self.beta = beta
        self.epsilon = epsilon
        self.s = None

    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Performs a single RMSprop optimization step.
        """
        if self.s is None:

            self.s = np.zeros_like(params)


        # s_t = beta * s_{t-1} + (1 - beta) * g_t^2
        self.s = self.beta * self.s + (1 - self.beta) * (gradients ** 2)

        # Update parameters
        # theta_{t+1} = theta_t - (lr / (sqrt(s_t) + epsilon)) * g_t
        update = (self.lr / (np.sqrt(self.s) + self.epsilon)) * gradients
        return params - update