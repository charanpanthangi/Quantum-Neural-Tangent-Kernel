import numpy as np
from app.classical_model import init_classical_params
from app.classical_ntk import compute_classical_ntk_matrix


def test_classical_ntk_matrix_symmetry():
    params = init_classical_params(hidden_dim=3, seed=1)
    x = np.array([[-0.2], [0.3]])
    K = compute_classical_ntk_matrix(x, params, epsilon=1e-3)
    assert K.shape == (2, 2)
    assert np.allclose(K, K.T)
