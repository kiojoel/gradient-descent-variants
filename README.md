# Gradient Descent Variants

A Python project that implements and compares optimization algorithms: **SGD**, **Adam**, and **RMSprop**. The project explores how each optimizer behaves on different types of loss landscapes using visualizations.

## Features

- Custom implementation of:
  - Stochastic Gradient Descent (SGD)
  - RMSprop
  - Adam
- Optimization over test functions:
  - Quadratic (Convex)
  - Rosenbrock (Non-convex)
- Visualization including 2D contour, 3D surface, and convergence plots.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/gradient-descent-variants.git
    cd gradient-descent-variants
    ```

2.  **Install the project and its dependencies:**
    _(This command uses the `setup.py` file to install everything needed.)_
    ```bash
    pip install -e .
    ```

## Run an Experiment

To run a pre-configured comparison and generate all plots:

```bash
python -m experiments.basic_comparison
```
