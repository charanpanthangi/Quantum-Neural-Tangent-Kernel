"""Quantum Neural Tangent Kernel (Q-NTK) utilities for small QNNs."""
from __future__ import annotations

import numpy as np
import pennylane as qml

from .qnn_model import _layer


def _qnode(n_layers: int):
    """Create a QNode that outputs a scalar expectation value."""
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(x, params):
        qml.RY(x * np.pi, wires=0)
        for layer_params in params:
            _layer(layer_params)
        return qml.expval(qml.PauliZ(0))

    return circuit


def compute_qntk_matrix(x_inputs: np.ndarray, params_init: np.ndarray) -> np.ndarray:
    """Compute the Q-NTK matrix for a batch of inputs at given parameters."""
    circuit = _qnode(len(params_init))
    jac_fn = qml.jacobian(circuit, argnum=1)

    grads = []
    for x in x_inputs.flatten():
        grad = jac_fn(x, params_init)
        grads.append(qml.math.stack([g for g in qml.math.stack(grad).flatten()]).astype(float))
    grads = np.stack(grads)

    return grads @ grads.T


def eigendecomposition(kernel: np.ndarray):
    """Return eigenvalues and eigenvectors of a kernel matrix."""
    vals, vecs = np.linalg.eigh(kernel)
    order = np.argsort(vals)[::-1]
    return vals[order], vecs[:, order]
