from setuptools import setup, find_packages

setup(
    name="gradient-descent-variants",
    version="0.1.0",
    description="Implementation and comparison of gradient descent optimization algorithms",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "scipy>=1.7.0",
        "plotly>=5.0.0",
        "jupyter>=1.0.0",
        "pytest>=6.0.0",
        "tqdm>=4.62.0",
        "pandas>=1.3.0",
    ],
    python_requires=">=3.8",
)