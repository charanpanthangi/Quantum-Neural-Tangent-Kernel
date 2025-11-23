import numpy as np
from app.qnn_model import init_qnn_params
from app.qntk import compute_qntk_matrix


def test_qntk_matrix_symmetry():
    params = init_qnn_params(n_layers=1, seed=0)
    x = np.array([[-0.1], [0.2], [0.5]])
    K = compute_qntk_matrix(x, params)
    assert K.shape == (3, 3)
    assert np.allclose(K, K.T)
