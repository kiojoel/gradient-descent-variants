import numpy as np
import pytest
from src.optimizers import SGD, RMSprop, Adam
from src.loss_functions import Quadratic


@pytest.mark.parametrize("optimizer_class", [SGD, RMSprop, Adam])

def test_optimizer_reduces_loss(optimizer_class):
  """
    A check to ensure that each optimizer decreases the loss
    on a simple quadratic function after a few steps.
    """
  loss_function = Quadratic()
  optimizer = optimizer_class(lr=0.01)

  # Start at a point away from the minimum
  params = np.array([5.0, 8.0])

  # Calculate initial loss
  initial_loss = loss_function.evaluate(params)

  # Run the optimizer for a few iterations
  num_steps = 20
  for _ in range(num_steps):
      gradients = loss_function.gradient(params)
      params = optimizer.step(params, gradients)

  # Calculate final loss
  final_loss = loss_function.evaluate(params)

  assert final_loss < initial_loss, f"{optimizer.name} did not reduce the loss."
