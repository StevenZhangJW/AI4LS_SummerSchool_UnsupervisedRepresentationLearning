"""Replace tutorial 1's MedVAE section with the verified TinyVAE workflow."""

import json
from pathlib import Path


NOTEBOOK = Path("AI4LS_URL_tutorial1.ipynb")


def markdown(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


def code(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in text.strip().split("\n")],
    }


nb = json.loads(NOTEBOOK.read_text())
original = nb["cells"]

# Preserve the original introduction through the model prompt (cells 0-12).
prefix = original[:13]

replacement = [
    markdown("""## Small 2D variational autoencoder

The pretrained 3D MedVAE is too expensive for repeated CPU experiments on this machine. Instead, this section trains a compact 2D VAE on the nonblank axial slices of the downloaded volume.

- Input: one-channel `128 × 128` axial slices
- Encoder: three strided convolutions
- Latent representation: 16 values (`mu` and `logvar`)
- Decoder: one fully connected layer and three transposed convolutions
- Parameters: 417,721

A VAE is used rather than a deterministic autoencoder because the tutorial asks for repeated samples, prior sampling, and latent-space modifications. The full implementation is in `train_toy_vae.py`."""),
    code("""# Training takes about 30 seconds on this two-core CPU.
# Re-running this cell overwrites outputs/toy_vae with a reproducible run.
%run train_toy_vae.py --epochs 15 --latent-dim 16 --seed 42"""),
    markdown("""## Load the trained model and results

The following cell reloads the saved checkpoint, demonstrating that the result is reusable without retraining."""),
    code("""import json
from IPython.display import display, Image
from train_toy_vae import TinyVAE, load_slices

toy_output = Path('outputs/toy_vae')
checkpoint = torch.load(toy_output / 'tiny_vae.pt', map_location='cpu', weights_only=True)
toy_model = TinyVAE(checkpoint['latent_dim'])
toy_model.load_state_dict(checkpoint['model_state_dict'])
toy_model.eval()

toy_images, toy_slice_indices, _ = load_slices(file_path, checkpoint['image_size'])
metrics = json.loads((toy_output / 'metrics.json').read_text())

print('[TRACE] Trained checkpoint loaded')
for name, value in metrics.items():
    print(f'{name}: {value}')"""),
    markdown("""### 1–2. Reconstruction quality

The figure shows original slices, deterministic reconstructions using the posterior mean, and absolute errors. MAE, MSE, and PSNR quantify fidelity. Expect the VAE to preserve large brain structures while smoothing fine texture and sharp boundaries because the latent has only 16 values."""),
    code("""display(Image(filename=str(toy_output / 'training_history.png')))
display(Image(filename=str(toy_output / 'reconstructions.png')))
print(f\"MAE: {metrics['mae']:.4f}\")
print(f\"MSE: {metrics['mse']:.4f}\")
print(f\"PSNR: {metrics['psnr_db']:.2f} dB\")"""),
    markdown("""### 3. Multiple samples for the same input

The encoder predicts a Gaussian distribution. Sampling it repeatedly gives related but non-identical reconstructions. Differences indicate uncertainty/variation allowed by the learned posterior."""),
    code("""display(Image(filename=str(toy_output / 'repeated_reconstructions.png')))"""),
    markdown("""### 4. Random samples from the prior

These images decode random vectors drawn from the assumed standard-normal prior. With only one volume and 15 training epochs, they may look blurrier or less anatomically plausible than reconstructions."""),
    code("""display(Image(filename=str(toy_output / 'prior_samples.png')))"""),
    markdown("""### 5. Modify and interpolate the latent space

The traversal moves one encoded slice along a normalized random direction. The interpolation moves between latent means from two distant axial slices. Smooth image changes suggest a locally organized representation."""),
    code("""display(Image(filename=str(toy_output / 'latent_traversal.png')))
display(Image(filename=str(toy_output / 'latent_interpolation.png')))"""),
    markdown("""### 6. Relevance and limitations

Latent variations may be useful for compression, anomaly detection, similarity search, or initialization for downstream learning. They must not be interpreted as clinically meaningful variations here: the model was trained on slices from only one synthetic volume, without labels or external validation."""),
    markdown("""# Analysis

The original tutorial asks for latent codes from different classes and a t-SNE/UMAP plot. Only one scan/class is available, so this notebook uses inferior/middle/superior slice-position thirds as **proxy groups**. These are not diagnostic or biological classes. Download samples from additional AADD classes before drawing conclusions about class separation."""),
    code("""display(Image(filename=str(toy_output / 'latent_umap.png')))
print('Latent-space silhouette (proxy groups):', metrics['latent_silhouette_proxy'])
print('Image-space silhouette (proxy groups):', metrics['image_silhouette_proxy'])"""),
    markdown("""## Interpretation

For this run, the latent-space proxy-group silhouette score is approximately **0.336**, compared with **0.249** for downsampled image space. This suggests that the encoder organizes slices by broad anatomical position somewhat more clearly than raw pixels.

This does **not** demonstrate class discrimination: adjacent slices from the same volume are highly correlated, and the groups were derived from slice position. A valid class experiment requires independent volumes from multiple classes, a train/test split at the volume level, and evaluation on held-out subjects."""),
    code("""# Latent arrays are saved for further experiments.
latent_results = np.load(toy_output / 'latent_codes.npz')
print('Latent matrix:', latent_results['latent'].shape)
print('Slice indices:', latent_results['slice_index'].shape)
print('Proxy groups:', np.unique(latent_results['group'], return_counts=True))"""),
]

nb["cells"] = prefix + replacement
NOTEBOOK.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
print(f"Updated {NOTEBOOK} with {len(nb['cells'])} cells")
