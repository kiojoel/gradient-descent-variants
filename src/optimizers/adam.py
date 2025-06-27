import numpy as np
from .base_optimizer import BaseOptimizer

class Adam(BaseOptimizer):
    """
    Adam (Adaptive Moment Estimation) optimizer.
    Combines ideas from RMSprop and Momentum.
    """
    def __init__(self, lr: float = 0.001, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        """
        Initializes the Adam optimizer.

        Args:
            lr (float): The learning rate.
            beta1 (float): Decay rate for the first moment estimate (mean of gradients).
            beta2 (float): Decay rate for the second moment estimate (variance of gradients).
            epsilon (float): A small constant for numerical stability.
        """
        super().__init__(lr)
        if not (0.0 <= beta1 < 1.0):
            raise ValueError("Beta1 must be in [0, 1).")
        if not (0.0 <= beta2 < 1.0):
            raise ValueError("Beta2 must be in [0, 1).")

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.m = None
        self.v = None
        self.t = 0

    def step(self, params: np.ndarray, gradients: np.ndarray) -> np.ndarray:
        """
        Performs a single Adam optimization step.
        """
        if self.m is None:
            # Initialize moments and timestep on the first step
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)

        self.t += 1

        # Update biased first moment estimate
        # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
        self.m = self.beta1 * self.m + (1 - self.beta1) * gradients

        # Update biased second raw moment estimate
        # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
        self.v = self.beta2 * self.v + (1 - self.beta2) * (gradients ** 2)

        # Compute bias-corrected first moment estimate
        # m_hat_t = m_t / (1 - beta1^t)
        m_hat = self.m / (1 - self.beta1**self.t)

        # Compute bias-corrected second raw moment estimate
        # v_hat_t = v_t / (1 - beta2^t)
        v_hat = self.v / (1 - self.beta2**self.t)

        # Update parameters
        # theta_{t+1} = theta_t - lr * m_hat_t / (sqrt(v_hat_t) + epsilon)
        update = self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)
        return params - update