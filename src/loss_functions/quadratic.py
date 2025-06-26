"""
Quadratic loss functions for testing optimization algorithms.
"""

import numpy as np
from typing import Optional, Tuple
from .base_function import BaseLossFunction


class QuadraticFunction(BaseLossFunction):
    """
    General quadratic function: f(x) = 0.5 * x^T * A * x + b^T * x + c

    This is a convex function when A is positive definite.
    The global minimum is at x* = -A^(-1) * b with value c - 0.5 * b^T * A^(-1) * b
    """

    def __init__(self,
                 A: Optional[np.ndarray] = None,
                 b: Optional[np.ndarray] = None,
                 c: float = 0.0,
                 dim: int = 2):
        """
        Initialize quadratic function.

        Args:
            A: Quadratic term matrix (dim x dim). If None, uses identity matrix.
            b: Linear term vector (dim,). If None, uses zero vector.
            c: Constant term (scalar).
            dim: Dimensionality if A and b are not provided.
        """
        super().__init__(name="Quadratic", dim=dim)

        # Set default values if not provided
        if A is None:
            A = np.eye(dim)
        if b is None:
            b = np.zeros(dim)

        self.A = A
        self.b = b
        self.c = c
        self.dim = A.shape[0]

        # Compute global minimum if A is invertible
        try:
            A_inv = np.linalg.inv(A)
            self.global_minimum = -A_inv @ b
            self.global_min_value = c - 0.5 * b.T @ A_inv @ b
        except np.linalg.LinAlgError:
            self.global_minimum = None
            self.global_min_value = None

        # Set reasonable bounds for visualization
        self.bounds = [(-10.0, 10.0)] * self.dim

    def forward(self, params: np.ndarray) -> float:
        """
        Compute quadratic function value.

        Args:
            params: Parameter vector

        Returns:
            Function value
        """
        return 0.5 * params.T @ self.A @ params + self.b.T @ params + self.c

    def gradient(self, params: np.ndarray) -> np.ndarray:
        """
        Compute gradient of quadratic function.

        Args:
            params: Parameter vector

        Returns:
            Gradient vector
        """
        return self.A @ params + self.b

    def hessian(self, params: np.ndarray) -> np.ndarray:
        """
        Compute Hessian matrix (constant for quadratic functions).

        Args:
            params: Parameter vector (not used, included for interface consistency)

        Returns:
            Hessian matrix
        """
        return self.A


class SimpleQuadratic(QuadraticFunction):
    """
    Simple quadratic bowl: f(x) = 0.5 * ||x - center||^2

    This is the simplest convex quadratic function, useful for basic testing.
    """

    def __init__(self, center: Optional[np.ndarray] = None, dim: int = 2):
        """
        Initialize simple quadratic bowl.

        Args:
            center: Center of the bowl (global minimum)
            dim: Dimensionality if center is not provided
        """
        if center is None:
            center = np.zeros(dim)

        # Create identity matrix and linear term to shift minimum to center
        A = np.eye(len(center))
        b = -center  # This makes the minimum at x = center

        super().__init__(A=A, b=b, c=0.0, dim=len(center))
        self.center = center
        self.name = "Simple_Quadratic"


class IllConditionedQuadratic(QuadraticFunction):
    """
    Ill-conditioned quadratic function with high condition number.

    This creates a narrow valley that is challenging for optimization algorithms.
    """

    def __init__(self, condition_number: float = 100.0, dim: int = 2):
        """
        Initialize ill-conditioned quadratic.

        Args:
            condition_number: Ratio of largest to smallest eigenvalue
            dim: Dimensionality
        """
        # Create eigenvalues with specified condition number
        eigenvalues = np.logspace(0, np.log10(condition_number), dim)

        # Create random orthogonal matrix for rotation
        Q = np.linalg.qr(np.random.randn(dim, dim))[0]

        # Construct A = Q * diag(eigenvalues) * Q^T
        A = Q @ np.diag(eigenvalues) @ Q.T

        # Random linear term
        b = np.random.randn(dim)

        super().__init__(A=A, b=b, c=0.0, dim=dim)
        self.condition_number = condition_number
        self.name = f"IllConditioned_Quadratic_k{condition_number}"


class RotatedQuadratic(QuadraticFunction):
    """
    Rotated quadratic function to test optimizer behavior with non-axis-aligned landscapes.
    """

    def __init__(self,
                 eigenvalues: Optional[np.ndarray] = None,
                 rotation_angle: Optional[float] = None,
                 dim: int = 2):
        """
        Initialize rotated quadratic.

        Args:
            eigenvalues: Eigenvalues of the quadratic form
            rotation_angle: Rotation angle (for 2D only)
            dim: Dimensionality
        """
        if eigenvalues is None:
            eigenvalues = np.array([1.0, 10.0]) if dim == 2 else np.logspace(0, 1, dim)

        if dim == 2 and rotation_angle is not None:
            # Create 2D rotation matrix
            cos_theta = np.cos(rotation_angle)
            sin_theta = np.sin(rotation_angle)
            Q = np.array([[cos_theta, -sin_theta],
                         [sin_theta, cos_theta]])
        else:
            # Create random orthogonal matrix
            Q = np.linalg.qr(np.random.randn(dim, dim))[0]

        # Construct A = Q * diag(eigenvalues) * Q^T
        A = Q @ np.diag(eigenvalues) @ Q.T

        super().__init__(A=A, b=np.zeros(dim), c=0.0, dim=dim)
        self.eigenvalues = eigenvalues
        self.rotation_matrix = Q
        self.name = f"Rotated_Quadratic"