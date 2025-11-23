import numpy as np
from app.qnn_model import init_qnn_params
from app.classical_model import init_classical_params
from app.dynamics import train_qnn_gd, train_classical_gd, kernel_regression
from app.qntk import compute_qntk_matrix
from app.classical_ntk import compute_classical_ntk_matrix


def test_training_loss_decreases():
    x = np.array([[-0.5], [0.0], [0.5]])
    y = np.sin(np.pi * x)
    q_params = init_qnn_params(n_layers=1, seed=0)
    c_params = init_classical_params(hidden_dim=3, seed=0)

    _, q_loss = train_qnn_gd(x, y, q_params, steps=5, lr=0.2)
    _, c_loss = train_classical_gd(x, y, c_params, steps=3, lr=0.1)

    assert np.any(q_loss[1:] < q_loss[:-1])
    assert np.any(c_loss[1:] < c_loss[:-1])


def test_kernel_regression_shapes():
    x = np.array([[-0.2], [0.3]])
    y = np.sin(np.pi * x)
    q_params = init_qnn_params(n_layers=1, seed=1)
    c_params = init_classical_params(hidden_dim=2, seed=1)
    Kq = compute_qntk_matrix(x, q_params)
    Kc = compute_classical_ntk_matrix(x, c_params, epsilon=1e-3)
    preds_q = kernel_regression(Kq, y, Kq)
    preds_c = kernel_regression(Kc, y, Kc)
    assert preds_q.shape == y.shape
    assert preds_c.shape == y.shape
