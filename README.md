# AI4LS Summer School: Unsupervised Representation Learning

Materials for the unsupervised representation learning tutorials.

## Local setup

From this repository directory:

```bash
uv sync
uv run python -m ipykernel install --user \
  --name ai4ls-workspace \
  --display-name "Python 3.12 (AI4LS Workspace)"
```

In VS Code or Jupyter, open `AI4LS_URL_tutorial1.ipynb` and select the
**Python 3.12 (AI4LS Workspace)** kernel.

Tutorial 1 uses the sample at:

```text
data/class0/aadd_00000.nii.gz
```

Its lightweight 2D VAE implementation is in `train_toy_vae.py`. Trained
outputs and figures are written under `outputs/toy_vae/`.
