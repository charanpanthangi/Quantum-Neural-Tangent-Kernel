"""Tiny classical neural network used alongside the QNN for NTK comparisons."""
from __future__ import annotations

import numpy as np


def init_classical_params(hidden_dim: int = 10, seed: int | None = 0) -> dict:
    """Initialize parameters for a one-hidden-layer MLP."""
    rng = np.random.default_rng(seed)
    params = {
        "W1": rng.normal(scale=0.5, size=(hidden_dim, 1)),
        "b1": np.zeros((hidden_dim, 1)),
        "W2": rng.normal(scale=0.5, size=(1, hidden_dim)),
        "b2": np.zeros((1, 1)),
    }
    return params


def classical_forward(x_batch: np.ndarray, params: dict) -> np.ndarray:
    """Forward pass of the MLP for a batch of inputs."""
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    h = np.tanh(W1 @ x_batch.T + b1)  # shape (hidden_dim, batch)
    out = W2 @ h + b2  # shape (1, batch)
    return out.T
