"""Build guided tutorial 2 and 3 notebooks around the verified class0 workflows."""

import json
from pathlib import Path


def md(text):
    return {"cell_type": "markdown", "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]}


def code(text):
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]}


def intro(path):
    nb = json.loads(path.read_text())
    return nb, nb["cells"][:6]


root = Path(".")

path2 = root / "AI4LS_URL_tutorial2.ipynb"
nb2, prefix2 = intro(path2)
nb2["cells"] = prefix2 + [
    md("""## Reproducible setup and the current dataset

Select **Python 3.12 (AI4LS Workspace)**. This notebook uses the single downloaded brain volume in `data/class0`. Because there is only one volume and no true labels, inferior/middle/superior slice thirds are used only as proxy groups for visualization—not as diagnostic classes."""),
    code("""%matplotlib inline
import json
from pathlib import Path
from IPython.display import display, Image

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

file_path = Path('data/class0/aadd_00000.nii.gz')
assert file_path.exists(), f'Missing: {file_path.resolve()}'
img = nib.as_closest_canonical(nib.load(file_path))
volume = img.get_fdata(dtype=np.float32)
volume = (volume - volume.min()) / max(float(volume.max() - volume.min()), 1e-8)
middle = volume.shape[2] // 2
plt.figure(figsize=(5, 5))
plt.imshow(volume[:, :, middle], cmap='gray')
plt.title(f'class0 axial slice {middle}')
plt.axis('off')
plt.show()
print('Volume:', volume.shape, 'range:', (float(volume.min()), float(volume.max())))"""),
    md("""# Model: DINO dense and global representations

The tutorial suggests DINOv3. Its official weights are gated and require accepting Meta's access terms. This executable version uses the public `facebook/dinov2-small` fallback, which exposes the same representations needed here: a global CLS token, self-attention matrices, and patch tokens.

After DINOv3 access is granted, set the environment variable `DINO_MODEL_ID=facebook/dinov3-vits16-pretrain-lvd1689m` before running the script. The code selects the final spatial patch tokens, so it also handles DINOv3 register tokens."""),
    code("""# Processes 30 representative slices and trains the tailored pretext model.
%run tutorial2_self_supervision.py"""),
    code("""tutorial2_output = Path('outputs/tutorial2')
tutorial2_metrics = json.loads((tutorial2_output / 'metrics.json').read_text())
for key, value in tutorial2_metrics.items():
    print(f'{key}: {value}')"""),
    md("""## 1. CLS-token attention

The overlay averages the final-layer attention from the CLS query to all spatial patches. Check whether high-attention areas correspond to anatomy rather than background or fixed image positions. Attention is descriptive, not a validated segmentation or causal explanation."""),
    code("""display(Image(filename=str(tutorial2_output / 'medical_attention_pca.png')))"""),
    md("""## 2. Contextualized patch embeddings reduced with PCA

The right panel maps the first three principal components of the patch embeddings to RGB. Similar colors mean similar DINO features within this one image. Look for consistency within tissue regions and transitions at boundaries; the 14-pixel patches limit granularity."""),
    md("""## 3. Natural-image comparison

DINO was pretrained on natural/web imagery. Comparing the medical slice with a natural image reveals whether attention and PCA colors are more object-centered in-domain and more influenced by image position or gross intensity out-of-domain."""),
    code("""display(Image(filename=str(tutorial2_output / 'natural_attention_pca.png')))"""),
    md("""# Analysis: global slice representations

CLS tokens from 30 slices are reduced with UMAP and colored by slice-position proxy group. This is an exploratory continuity test, not class evaluation."""),
    code("""display(Image(filename=str(tutorial2_output / 'dino_cls_umap.png')))
print(f\"DINO proxy silhouette: {tutorial2_metrics['dino_proxy_silhouette']:.3f}\")
print(f\"Raw-image proxy silhouette: {tutorial2_metrics['raw_image_proxy_silhouette']:.3f}\")"""),
    md("""### Interpretation

For this run, DINO's proxy silhouette is **0.073**, below the raw-image value of **0.166**. Thus the frozen natural-image DINO representation does not provide an advantage for separating coarse anatomical slice positions in this single scan. This may reflect domain shift, grayscale replication, limited sample size, or the fact that slice position is strongly encoded by raw anatomy. True class conclusions require independent volumes from multiple classes."""),
    md("""# Tailored self-supervised task: rotation prediction

We create labels automatically by rotating each slice by 0°, 90°, 180°, or 270°. A small CNN must retain global orientation and asymmetric anatomical layout to solve the task. This is self-supervised because the transformation supplies the target without manual labels.

An initial model with global average pooling stayed at chance because pooling erased spatial arrangement. Preserving the final `8×8` feature grid corrected that failure—an example of matching architecture to the pretext task."""),
    code("""display(Image(filename=str(tutorial2_output / 'rotation_training.png')))
print(f\"Rotation validation accuracy: {tutorial2_metrics['rotation_validation_accuracy']:.1%}\")
print('Rotation-network parameters:', tutorial2_metrics['rotation_parameters'])"""),
    md("""### Tailored-task result and limitations

The 137,284-parameter network reaches approximately **94.6%** validation accuracy, so the pretext task successfully forces orientation-sensitive features. However, rotated versions of the same source slices were randomly split across train and validation; this evaluates the transformation task, not subject-level generalization. With multiple scans, split by volume before augmentation."""),
]
path2.write_text(json.dumps(nb2, indent=1, ensure_ascii=False) + "\n")


path3 = root / "AI4LS_URL_tutorial3.ipynb"
nb3, prefix3 = intro(path3)
nb3["cells"] = prefix3 + [
    md("""## Reproducible setup and target image

Select **Python 3.12 (AI4LS Workspace)**. We fit two continuous representations to the middle nonblank axial slice from the existing `class0` volume:

1. **Implicit:** a coordinate MLP maps `(x, y) → intensity` using Fourier features.
2. **Explicit:** 128 learned axis-aligned Gaussians store centers, scales, and amplitudes and are composited to render intensity."""),
    code("""%matplotlib inline
import json
from pathlib import Path
from IPython.display import display, Image

file_path = Path('data/class0/aadd_00000.nii.gz')
assert file_path.exists(), f'Missing: {file_path.resolve()}'"""),
    md("""# Fit both representations

The implicit model samples coordinate/intensity pairs during optimization. The explicit model is fitted at `64×64` for speed and rendered at `128×128`. Both can be queried on a `256×256` grid without changing their stored representation."""),
    code("""%run tutorial3_representations.py"""),
    code("""tutorial3_output = Path('outputs/tutorial3')
tutorial3_metrics = json.loads((tutorial3_output / 'metrics.json').read_text())
for key, value in tutorial3_metrics.items():
    print(f'{key}: {value}')"""),
    md("""## Reconstruction and error comparison

The coordinate MLP has greater capacity and represents fine structures more accurately. The 128-Gaussian model is much smaller and directly interpretable, but smooth Gaussian primitives blur sharp boundaries and fine texture."""),
    code("""display(Image(filename=str(tutorial3_output / 'representation_comparison.png')))"""),
    md("""## Continuous rendering

Both representations can be evaluated at arbitrary coordinates. Rendering at `256×256` demonstrates continuity, but it does not create new measured anatomical detail: it interpolates what the fitted representation learned from the `128×128` target."""),
    code("""display(Image(filename=str(tutorial3_output / 'arbitrary_resolution.png')))
display(Image(filename=str(tutorial3_output / 'fitting_curves.png')))"""),
    md("""# Quantitative comparison"""),
    code("""print(
    f\"Implicit: {tutorial3_metrics['implicit_parameters']:,} parameters, \"
    f\"{tutorial3_metrics['implicit_seconds']:.1f}s, \"
    f\"MAE={tutorial3_metrics['implicit_mae']:.4f}, \"
    f\"PSNR={tutorial3_metrics['implicit_psnr_db']:.2f} dB\"
)
print(
    f\"Explicit: {tutorial3_metrics['gaussian_parameters']:,} parameters, \"
    f\"{tutorial3_metrics['gaussian_seconds']:.1f}s, \"
    f\"MAE={tutorial3_metrics['gaussian_mae']:.4f}, \"
    f\"PSNR={tutorial3_metrics['gaussian_psnr_db']:.2f} dB\"
)"""),
    md("""## Which representation is useful downstream?

**Implicit coordinate MLP**

- Best here for compact, high-fidelity continuous rendering (31.48 dB PSNR).
- Differentiable with respect to coordinates; useful for interpolation, registration research, continuous resampling, or coordinate-conditioned analysis.
- Weights are distributed and harder to interpret or edit locally; fitting is image-specific.

**Explicit Gaussians**

- Only 640 learned scalars in this demonstration and fast to render or edit by moving/removing primitives.
- Useful when spatial primitives, sparse structure, local editing, or interpretable geometry matter.
- Lower fidelity here (25.02 dB PSNR); fixed 128 smooth Gaussians miss sharp and fine-scale anatomy.

For segmentation or classification across subjects, neither per-image representation is automatically ideal: fitting each image independently does not align features across subjects. Shared encoders or canonical coordinate systems would be needed. For compression/rendering of individual images, the implicit model gives the strongest fidelity-size tradeoff in this experiment."""),
]
path3.write_text(json.dumps(nb3, indent=1, ensure_ascii=False) + "\n")

print(f"Updated {path2} ({len(nb2['cells'])} cells)")
print(f"Updated {path3} ({len(nb3['cells'])} cells)")
