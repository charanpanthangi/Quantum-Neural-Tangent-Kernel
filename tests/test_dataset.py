import numpy as np
from app.dataset import generate_regression_data


def test_dataset_shapes():
    X_train, X_test, y_train, y_test = generate_regression_data(n_points=10, test_size=0.2, seed=1)
    assert X_train.shape[1] == 1
    assert y_train.shape[1] == 1
    assert X_test.shape[1] == 1
    assert np.all(X_train <= 1.5) and np.all(X_train >= -1.5)
