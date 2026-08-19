# Unsupervised Representation Learning for Life Science

This repository contains three practical tutorials from the AI for Life Science
Summer School. They introduce ways of turning medical images into smaller,
useful numerical summaries called **representations**.

The material is written for researchers who may be new to deep learning. An
epidemiologist can think of a representation as a set of automatically derived
image features. These features may later be explored as predictors, used to
compare images, or included in clustering and other downstream analyses.

> These tutorials are educational. The example data and model outputs are not
> validated for diagnosis, prognosis, causal inference, or clinical decisions.

## What the tutorials cover

### Tutorial 1: Autoencoders and latent representations

Notebook: `AI4LS_URL_tutorial1.ipynb`

A small two-dimensional variational autoencoder (VAE) is trained on axial
slices from one three-dimensional brain scan. The tutorial examines:

- reconstruction of an image after compression;
- reconstruction error;
- repeated samples from the same input;
- random samples from the model;
- interpolation and movement through the latent space; and
- a UMAP display of the learned slice representations.

The model is intentionally small so it can be trained on a CPU in about a
minute. Its implementation is in `train_toy_vae.py`.

### Tutorial 2: Self-supervised learning

Notebook: `AI4LS_URL_tutorial2.ipynb`

This tutorial uses a public DINOv2 model to obtain features from medical image
slices. It includes:

- global image features from the CLS token;
- an attention-map illustration;
- patch-level features displayed with PCA colors;
- comparison with a natural image;
- UMAP visualization of slice features; and
- a small rotation-prediction task in which training labels are created from
  known image rotations rather than manual annotation.

The original exercise suggests DINOv3. DINOv3 requires separate acceptance of
Meta's access conditions on Hugging Face, so the runnable example uses the
public `facebook/dinov2-small` model. The implementation is in
`tutorial2_self_supervision.py`.

### Tutorial 3: Implicit and explicit image representations

Notebook: `AI4LS_URL_tutorial3.ipynb`

A representative two-dimensional slice is stored in two different ways:

- an **implicit representation**, where a small neural network predicts image
  intensity from an `(x, y)` coordinate; and
- an **explicit representation**, where the image is approximated using 128
  learned Gaussian shapes.

The tutorial compares reconstruction quality, fitting time, number of stored
parameters, and rendering at a different resolution. The implementation is in
`tutorial3_representations.py`.

## Repository structure

```text
AI4LS_URL_tutorial1.ipynb       Tutorial 1 notebook
AI4LS_URL_tutorial2.ipynb       Tutorial 2 notebook
AI4LS_URL_tutorial3.ipynb       Tutorial 3 notebook
train_toy_vae.py                Small VAE used in tutorial 1
tutorial2_self_supervision.py   DINO and rotation-task analysis
tutorial3_representations.py    Coordinate-network and Gaussian analysis
data/                            Local input data; large images are not committed
outputs/                         Models, metrics, and figures created by the tutorials
lectureNotes/                    Associated lecture material
pyproject.toml                  Python package requirements
uv.lock                         Exact package versions for reproducibility
```

## Requirements

- A computer running Linux, macOS, or Windows
- Python 3.12
- `uv`, used to create and manage the Python environment
- Approximately 6 GB of free disk space
- Internet access the first time packages and DINOv2 weights are downloaded

A GPU is not required. The examples have been tested on a CPU-only machine.

## Set up the environment

Open a terminal in this repository and run:

```bash
uv sync
uv run python -m ipykernel install --user \
  --name ai4ls-workspace \
  --display-name "Python 3.12 (AI4LS Workspace)"
```

Then open a notebook in VS Code or Jupyter and select:

```text
Python 3.12 (AI4LS Workspace)
```

If a notebook reports that a package is missing, first check the selected
kernel. In a notebook code cell, run:

```python
import sys
print(sys.prefix)
```

The result should end with:

```text
AI4LS_SummerSchool_UnsupervisedRepresentationLearning/.venv
```

## Add the example data

The tutorials expect this file:

```text
data/class0/aadd_00000.nii.gz
```

It is a NIfTI medical-image volume from the gated AADD dataset on Hugging Face.
You must sign in, accept the dataset access conditions, and use your own access
token to download it. Do not commit the image or your access token to Git.

After downloading, confirm that the file is present:

```bash
ls -lh data/class0/aadd_00000.nii.gz
```

## Run the tutorials

Run the notebooks in numerical order. Within each notebook, restart the kernel
and choose **Run All** so that variables are created in the expected order.

The scripts can also be run directly:

```bash
uv run python train_toy_vae.py
uv run python tutorial2_self_supervision.py
uv run python tutorial3_representations.py
```

Results are saved under:

```text
outputs/toy_vae/
outputs/tutorial2/
outputs/tutorial3/
```

Each folder contains a `metrics.json` file together with figures and, where
relevant, a saved model checkpoint.

## Important interpretation cautions

Only one image volume from `class0` is used in the current examples. Several
plots divide slices into inferior, middle, and superior thirds. These are
**slice-position proxy groups**, not disease classes, exposure groups, patient
outcomes, or independent observations.

Slices from one volume are strongly correlated. They must not be treated as
independent study participants. The reported accuracy and silhouette values are
therefore demonstrations of the methods, not estimates of clinical validity or
generalizability.

For an epidemiological study, a stronger evaluation would require:

- multiple participants or independent image volumes;
- meaningful exposure, outcome, or phenotype definitions;
- train, validation, and test splits made at participant level;
- assessment of confounding, selection, and measurement bias;
- external validation where possible; and
- uncertainty estimates and sensitivity analyses.

## Common problems

### `ModuleNotFoundError`

The notebook is probably using the wrong Python kernel. Select **Python 3.12
(AI4LS Workspace)**, restart the kernel, and run all cells again.

### `No pyproject.toml found`

Run `uv` commands from the repository directory containing `pyproject.toml`.

### A Hugging Face download returns `401` or `403`

- `401` usually means authentication is missing or invalid.
- `403` usually means the account has not been granted access to the gated
  dataset or model.

### DINOv3 cannot be downloaded

DINOv3 has separate access conditions. Tutorial 2 works without DINOv3 because
it uses public DINOv2 weights by default.

### Git cannot push

Check GitHub authentication with:

```bash
gh auth status
```

Then push from this repository with:

```bash
git push origin main
```

## Reproducibility

Random seeds are fixed in the example scripts. Exact Python dependencies are
recorded in `uv.lock`. Small numerical differences can still occur across
hardware and library builds.

## License and data use

Consult the repository license and the original licenses or access conditions
for every external dataset and pretrained model. Do not place identifiable or
restricted health data in a public repository.
