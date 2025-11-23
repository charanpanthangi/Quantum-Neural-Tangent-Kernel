"""Plotting utilities that save SVG figures for kernels and training curves."""
from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


plt.rcParams["figure.dpi"] = 120


def plot_kernel_heatmap(kernel: np.ndarray, title: str, output_path: str):
    """Plot a kernel matrix as a heatmap and save to an SVG file."""
    plt.figure(figsize=(4, 3))
    sns.heatmap(kernel, cmap="viridis")
    plt.title(title)
    plt.xlabel("Sample index")
    plt.ylabel("Sample index")
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_eigenvalues(ev_q: np.ndarray, ev_c: np.ndarray, output_path: str):
    """Plot eigenvalue spectra of Q-NTK and classical NTK on a log scale."""
    plt.figure(figsize=(5, 3))
    plt.semilogy(ev_q, label="Q-NTK eigenvalues")
    plt.semilogy(ev_c, label="Classical NTK eigenvalues")
    plt.xlabel("Eigenvalue index (sorted)")
    plt.ylabel("Eigenvalue magnitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()


def plot_training_dynamics(loss_q: np.ndarray, loss_c: np.ndarray, output_path: str):
    """Plot loss curves for QNN and classical network gradient descent."""
    plt.figure(figsize=(5, 3))
    plt.plot(loss_q, label="QNN loss")
    plt.plot(loss_c, label="Classical NN loss")
    plt.xlabel("Training step")
    plt.ylabel("Mean squared error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()
