"""Training dynamics experiments comparing QNN and classical MLP."""
from __future__ import annotations

import numpy as np
import pennylane as qml

from .qnn_model import _layer
from .classical_model import classical_forward


def _qnode_with_params(n_layers: int):
    """Return a QNode that outputs a scalar expectation for training."""
    dev = qml.device("default.qubit", wires=1)

    @qml.qnode(dev)
    def circuit(x, params):
        qml.RY(x * np.pi, wires=0)
        for layer_params in params:
            _layer(layer_params)
        return qml.expval(qml.PauliZ(0))

    return circuit


def train_qnn_gd(x_train: np.ndarray, y_train: np.ndarray, params_init: np.ndarray, lr: float = 0.1, steps: int = 50):
    """Simple gradient descent training for the QNN."""
    params = np.array(params_init, dtype=float)
    circuit = _qnode_with_params(len(params))
    loss_history = []

    def loss_fn(params):
        preds = np.array([circuit(float(x), params) for x in x_train.flatten()])
        return np.mean((preds.reshape(-1, 1) - y_train) ** 2)

    grad_fn = qml.grad(loss_fn)
    for _ in range(steps):
        loss = loss_fn(params)
        loss_history.append(float(loss))
        grads = grad_fn(params)
        params = params - lr * grads
    return params, np.array(loss_history)


def train_classical_gd(x_train: np.ndarray, y_train: np.ndarray, params_init: dict, lr: float = 0.1, steps: int = 50):
    """Gradient descent for the NumPy MLP using finite differences for gradients."""
    params = {k: v.copy() for k, v in params_init.items()}
    loss_history = []
    epsilon = 1e-4

    def loss_fn(p):
        preds = classical_forward(x_train, p)
        return float(np.mean((preds - y_train) ** 2))

    for _ in range(steps):
        loss = loss_fn(params)
        loss_history.append(loss)
        grads = {}
        for key, value in params.items():
            grads[key] = np.zeros_like(value)
            it = np.nditer(value, flags=["multi_index"], op_flags=["readwrite"])
            while not it.finished:
                idx = it.multi_index
                value[idx] += epsilon
                loss_plus = loss_fn(params)
                value[idx] -= 2 * epsilon
                loss_minus = loss_fn(params)
                value[idx] += epsilon
                grads[key][idx] = (loss_plus - loss_minus) / (2 * epsilon)
                it.iternext()
        for key in params:
            params[key] = params[key] - lr * grads[key]
    return params, np.array(loss_history)


def kernel_regression(kernel_train: np.ndarray, y_train: np.ndarray, kernel_test: np.ndarray, reg: float = 1e-3) -> np.ndarray:
    """Closed-form kernel regression using the NTK or Q-NTK."""
    identity = np.eye(kernel_train.shape[0])
    alpha = np.linalg.solve(kernel_train + reg * identity, y_train)
    return kernel_test @ alpha


def training_kernel_predictions(kernel_train: np.ndarray, y_train: np.ndarray, x_train: np.ndarray) -> np.ndarray:
    """Helper to compute leave-in kernel regression predictions on the training set."""
    return kernel_regression(kernel_train, y_train, kernel_train)
