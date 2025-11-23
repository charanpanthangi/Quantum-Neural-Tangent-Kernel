# Quantum Neural Tangent Kernel (Q-NTK)

## What This Project Does
- Builds a tiny Quantum Neural Network (QNN) and a small classical neural network.
- Computes the Neural Tangent Kernel (NTK) for both models.
- Compares how these kernels influence training behavior.
- Shows loss curves and kernel visualizations saved as SVG files.

## Why Q-NTK Is Interesting
- NTK helps us understand how neural networks learn near initialization.
- Q-NTK extends this idea to quantum neural networks.
- Comparing Q-NTK with classical NTK shows when QNNs might converge faster or behave differently.

## Why SVG Instead of PNG
GitHub’s CODEX interface cannot preview PNG/JPG and often shows “Binary files are not supported” in pull request views.
To avoid this, all visualizations in this repository are saved as lightweight SVG (vector) images.
SVGs are text-based, easy to diff, and render cleanly inside GitHub and CODEX.

## How It Works (Plain English)
- A small QNN encodes the input as a rotation on one qubit and measures a PauliZ expectation value.
- We compute derivatives of the QNN output with respect to its parameters and assemble the Q-NTK.
- A simple classical tanh network is treated the same way to build the classical NTK.
- We compare kernel heatmaps, eigenvalues, and gradient descent loss curves to see how kernels predict learning behavior.

## Repository Structure
- `app/` – all source code (models, kernels, plots, CLI)
- `notebooks/` – Q-NTK demo notebook
- `examples/` – SVG heatmaps and plots
- `tests/` – unit tests

## How to Run
```bash
pip install -r requirements.txt
python app/main.py
```

Notebook:
```bash
jupyter notebook notebooks/qntk_demo.ipynb
```

## What You Should See
- Q-NTK and classical NTK heatmaps.
- Eigenvalue comparison plot.
- Training loss curves for QNN vs classical NN.

## Future Ideas
- Larger QNNs / deeper circuits.
- Different data and tasks.
- Study Q-NTK on real hardware.
