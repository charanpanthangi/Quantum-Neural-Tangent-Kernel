"""Command line entrypoint for running Q-NTK vs NTK experiments."""
from __future__ import annotations

import argparse
import numpy as np

from .dataset import generate_regression_data
from .qnn_model import init_qnn_params
from .classical_model import init_classical_params
from .qntk import compute_qntk_matrix, eigendecomposition as q_eig
from .classical_ntk import compute_classical_ntk_matrix, eigendecomposition as c_eig
from .dynamics import train_qnn_gd, train_classical_gd
from .plots import plot_kernel_heatmap, plot_eigenvalues, plot_training_dynamics


EXAMPLES_DIR = "examples"


def run_experiment(n_points: int, hidden_dim: int, steps: int, seed: int):
    """Run the full pipeline: data, kernels, training, and plots."""
    X_train, X_test, y_train, y_test = generate_regression_data(n_points=n_points, seed=seed)

    qnn_params = init_qnn_params(seed=seed)
    classical_params = init_classical_params(hidden_dim=hidden_dim, seed=seed)

    q_kernel = compute_qntk_matrix(X_train, qnn_params)
    c_kernel = compute_classical_ntk_matrix(X_train, classical_params)

    q_vals, _ = q_eig(q_kernel)
    c_vals, _ = c_eig(c_kernel)

    _, q_loss = train_qnn_gd(X_train, y_train, qnn_params, steps=steps)
    _, c_loss = train_classical_gd(X_train, y_train, classical_params, steps=steps)

    plot_kernel_heatmap(q_kernel, "Q-NTK heatmap", f"{EXAMPLES_DIR}/qntk_matrix_heatmap.svg")
    plot_kernel_heatmap(c_kernel, "Classical NTK heatmap", f"{EXAMPLES_DIR}/classical_ntk_matrix_heatmap.svg")
    plot_eigenvalues(q_vals, c_vals, f"{EXAMPLES_DIR}/qntk_vs_classical_eigenvalues.svg")
    plot_training_dynamics(q_loss, c_loss, f"{EXAMPLES_DIR}/training_dynamics_comparison.svg")

    print("Q-NTK eigenvalues (top 3):", q_vals[:3])
    print("Classical NTK eigenvalues (top 3):", c_vals[:3])
    print("Final QNN loss:", q_loss[-1])
    print("Final classical NN loss:", c_loss[-1])
    print("SVG plots saved in examples/ directory.")


def build_argparser():
    parser = argparse.ArgumentParser(description="Quantum Neural Tangent Kernel demo")
    parser.add_argument("--n-points", type=int, default=20, help="Number of data points")
    parser.add_argument("--hidden-dim", type=int, default=10, help="Hidden units in the classical MLP")
    parser.add_argument("--steps", type=int, default=50, help="Gradient descent steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    return parser


def main():
    args = build_argparser().parse_args()
    run_experiment(args.n_points, args.hidden_dim, args.steps, args.seed)


if __name__ == "__main__":
    main()
