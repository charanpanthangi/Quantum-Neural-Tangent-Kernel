"""Dataset utilities for small regression task used in NTK and Q-NTK demos."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split


def generate_regression_data(n_points: int = 20, test_size: float = 0.2, seed: int | None = 0):
    """Generate a tiny 1D regression dataset.

    The inputs are evenly spaced in ``[-1, 1]`` and targets follow ``y = sin(pi * x)``.
    This tiny dataset keeps kernel matrices small so plots render quickly.

    Args:
        n_points: Total number of samples to create.
        test_size: Fraction of data reserved for testing.
        seed: Optional random seed for reproducibility.

    Returns:
        Tuple of ``(X_train, X_test, y_train, y_test)`` as NumPy arrays with shapes
        ``(n_train, 1)`` and ``(n_test, 1)`` for inputs.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(-1.0, 1.0, n_points)
    y = np.sin(np.pi * x)

    x += rng.normal(scale=0.05, size=x.shape)  # small noise to avoid perfect symmetry
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)

    return train_test_split(x, y, test_size=test_size, random_state=seed)
