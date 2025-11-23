"""Quantum neural network definition using PennyLane for Q-NTK experiments."""
from __future__ import annotations

import pennylane as qml
import numpy as np


def init_qnn_params(n_layers: int = 2, seed: int | None = 0) -> np.ndarray:
    """Initialize QNN parameters with small random angles.

    Args:
        n_layers: Number of repeated rotation layers.
        seed: Optional random seed.

    Returns:
        NumPy array of shape ``(n_layers, 2)`` representing rotation angles per layer.
    """
    rng = np.random.default_rng(seed)
    return rng.normal(scale=0.1, size=(n_layers, 2))


def _layer(params: np.ndarray):
    """Apply one variational layer given a row from the parameter matrix."""
    qml.RY(params[0], wires=0)
    qml.RZ(params[1], wires=0)


def qnn_forward(x: float, params: np.ndarray) -> float:
    """Evaluate the QNN for a single scalar input.

    The circuit uses one qubit with simple angle encoding followed by repeated
    rotation layers. The output is the expectation value of PauliZ, producing
    a scalar suitable for regression.
    """
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(val):
        qml.RY(val * np.pi, wires=0)  # encode input as rotation angle
        for layer_params in params:
            _layer(layer_params)
        return qml.expval(qml.PauliZ(0))

    return float(circuit(x))


def qnn_model(x_batch: np.ndarray, params: np.ndarray) -> np.ndarray:
    """Vectorized evaluation of the QNN over a batch of inputs."""
    return np.array([qnn_forward(float(x), params) for x in x_batch.flatten()]).reshape(-1, 1)
