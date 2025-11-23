import numpy as np
from app.classical_model import init_classical_params, classical_forward


def test_classical_forward_runs():
    params = init_classical_params(hidden_dim=5, seed=0)
    x = np.array([[0.0], [0.1]])
    outputs = classical_forward(x, params)
    assert outputs.shape == (2, 1)
    assert np.all(np.isfinite(outputs))
