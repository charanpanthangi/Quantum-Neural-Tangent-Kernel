import numpy as np
from app.qnn_model import init_qnn_params, qnn_model


def test_qnn_forward_runs():
    params = init_qnn_params(n_layers=1, seed=0)
    x = np.array([[0.1], [0.2]])
    outputs = qnn_model(x, params)
    assert outputs.shape == (2, 1)
    assert np.all(np.isfinite(outputs))
