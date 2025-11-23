"""Classical Neural Tangent Kernel utilities for the simple MLP."""
from __future__ import annotations

import numpy as np

from .classical_model import classical_forward


def _flatten_params(params: dict) -> np.ndarray:
    """Flatten parameter dictionary into a 1D vector."""
    return np.concatenate([params["W1"].ravel(), params["b1"].ravel(), params["W2"].ravel(), params["b2"].ravel()])


def _unflatten_params(vec: np.ndarray, shapes: dict) -> dict:
    """Rebuild parameter dictionary from flattened vector using stored shapes."""
    params = {}
    idx = 0
    for key, shape in shapes.items():
        size = np.prod(shape)
        params[key] = vec[idx : idx + size].reshape(shape)
        idx += size
    return params


def compute_classical_ntk_matrix(x_inputs: np.ndarray, params_init: dict, epsilon: float = 1e-4) -> np.ndarray:
    """Compute NTK matrix via finite differences for clarity and simplicity."""
    shapes = {k: v.shape for k, v in params_init.items()}
    base_vector = _flatten_params(params_init)

    grads = []
    for x in x_inputs.flatten():
        grad = np.zeros_like(base_vector)
        for i in range(len(base_vector)):
            shift = np.zeros_like(base_vector)
            shift[i] = epsilon
            plus_params = _unflatten_params(base_vector + shift, shapes)
            minus_params = _unflatten_params(base_vector - shift, shapes)
            f_plus = classical_forward(np.array([[x]]), plus_params)[0, 0]
            f_minus = classical_forward(np.array([[x]]), minus_params)[0, 0]
            grad[i] = (f_plus - f_minus) / (2 * epsilon)
        grads.append(grad)
    grads = np.stack(grads)
    return grads @ grads.T


def eigendecomposition(kernel: np.ndarray):
    """Return eigenvalues and eigenvectors of a kernel matrix."""
    vals, vecs = np.linalg.eigh(kernel)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]
