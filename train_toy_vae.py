"""Train and analyse a small 2D VAE on axial slices from one AADD volume."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, TensorDataset, random_split
import umap


class TinyVAE(nn.Module):
    def __init__(self, latent_dim: int = 16):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
        )
        self.fc_mu = nn.Linear(32 * 16 * 16, latent_dim)
        self.fc_logvar = nn.Linear(32 * 16 * 16, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 32 * 16 * 16)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(16, 8, 4, 2, 1), nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def encode(self, x):
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def decode(self, z):
        return self.decoder(self.fc_decode(z).view(-1, 32, 16, 16))

    def forward(self, x, sample=True):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar) if sample else mu
        return self.decode(z), mu, logvar


def load_slices(path: Path, size: int = 128):
    volume = nib.as_closest_canonical(nib.load(path)).get_fdata(dtype=np.float32)
    lo, hi = float(volume.min()), float(volume.max())
    volume = (volume - lo) / max(hi - lo, 1e-8)
    # Axial slices become independent 2D training samples.
    slices = torch.from_numpy(np.moveaxis(volume, 2, 0)).unsqueeze(1)
    foreground = slices.flatten(1).amax(1) > 0.05
    indices = torch.arange(len(slices))[foreground]
    slices = slices[foreground]
    slices = F.interpolate(slices, (size, size), mode="bilinear", align_corners=False)
    return slices.float(), indices.numpy(), volume.shape


def vae_loss(recon, x, mu, logvar, beta):
    reconstruction = F.mse_loss(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.square() - logvar.exp())
    return reconstruction + beta * kl, reconstruction, kl


def plot_grid(images, titles, path, ncols=4, cmap="gray"):
    images = np.asarray(images)
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()
    for ax, image, title in zip(axes, images, titles):
        ax.imshow(image, cmap=cmap, vmin=0, vmax=1)
        ax.set_title(title)
        ax.axis("off")
    for ax in axes[len(images):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/class0/aadd_00000.nii.gz")
    parser.add_argument("--output", default="outputs/toy_vae")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print("[TRACE 1/8] Loading and preparing slices", flush=True)
    images, source_indices, volume_shape = load_slices(Path(args.input))
    dataset = TensorDataset(images)
    n_val = max(1, round(0.2 * len(dataset)))
    n_train = len(dataset) - n_val
    generator = torch.Generator().manual_seed(args.seed)
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=generator)
    train_loader = DataLoader(train_set, batch_size=16, shuffle=True, generator=generator)
    val_loader = DataLoader(val_set, batch_size=32)
    print(f"Volume={volume_shape}; usable slices={len(images)}; train={n_train}; val={n_val}", flush=True)

    print("[TRACE 2/8] Building TinyVAE", flush=True)
    model = TinyVAE(args.latent_dim).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    beta = 1e-4
    print(f"Device={device}; parameters={parameter_count:,}; latent_dim={args.latent_dim}", flush=True)

    print("[TRACE 3/8] Training", flush=True)
    history = {"train": [], "val": []}
    start = time.perf_counter()
    for epoch in range(args.epochs):
        model.train()
        train_total = 0.0
        for (batch,) in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            recon, mu, logvar = model(batch, sample=True)
            loss, _, _ = vae_loss(recon, batch, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            train_total += loss.item() * len(batch)
        model.eval()
        val_total = 0.0
        with torch.inference_mode():
            for (batch,) in val_loader:
                batch = batch.to(device)
                recon, mu, logvar = model(batch, sample=False)
                loss, _, _ = vae_loss(recon, batch, mu, logvar, beta)
                val_total += loss.item() * len(batch)
        history["train"].append(train_total / n_train)
        history["val"].append(val_total / n_val)
        print(f"epoch={epoch + 1:02d} train={history['train'][-1]:.6f} val={history['val'][-1]:.6f}", flush=True)
    train_seconds = time.perf_counter() - start

    print("[TRACE 4/8] Computing reconstructions and metrics", flush=True)
    model.eval()
    all_images = images.to(device)
    with torch.inference_mode():
        deterministic, mu, logvar = model(all_images, sample=False)
    mae = F.l1_loss(deterministic, all_images).item()
    mse = F.mse_loss(deterministic, all_images).item()
    psnr = float(10 * np.log10(1.0 / max(mse, 1e-12)))

    chosen = [len(images) // 4, len(images) // 2, 3 * len(images) // 4]
    comparison = []
    titles = []
    for idx in chosen:
        comparison.extend([images[idx, 0].numpy(), deterministic[idx, 0].cpu().numpy(),
                           np.abs(images[idx, 0].numpy() - deterministic[idx, 0].cpu().numpy())])
        titles.extend([f"Original slice {source_indices[idx]}", "Reconstruction", "Absolute error"])
    plot_grid(comparison, titles, out / "reconstructions.png", ncols=3)

    print("[TRACE 5/8] Sampling repeated reconstructions and the prior", flush=True)
    target = images[len(images) // 2:len(images) // 2 + 1].to(device)
    with torch.inference_mode():
        target_mu, target_logvar = model.encode(target)
        repeated = torch.cat([model.decode(model.reparameterize(target_mu, target_logvar)) for _ in range(8)])
        prior = model.decode(torch.randn(8, args.latent_dim, device=device))
    repeated_images = [target[0, 0].cpu().numpy()] + [x[0].cpu().numpy() for x in repeated]
    plot_grid(repeated_images, ["Input"] + [f"Sample {i + 1}" for i in range(8)],
              out / "repeated_reconstructions.png", ncols=3)
    plot_grid([x[0].cpu().numpy() for x in prior], [f"Prior {i + 1}" for i in range(8)],
              out / "prior_samples.png", ncols=4)

    print("[TRACE 6/8] Traversing and interpolating latent space", flush=True)
    with torch.inference_mode():
        direction = torch.randn_like(target_mu)
        direction = direction / direction.norm(dim=1, keepdim=True)
        scales = torch.linspace(-3, 3, 7, device=device)
        traversal = model.decode(target_mu + scales[:, None] * direction)
        a, b = mu[len(images) // 4], mu[3 * len(images) // 4]
        alpha = torch.linspace(0, 1, 7, device=device)[:, None]
        interpolation = model.decode((1 - alpha) * a + alpha * b)
    plot_grid([x[0].cpu().numpy() for x in traversal], [f"offset {s:.1f}" for s in scales.tolist()],
              out / "latent_traversal.png", ncols=7)
    plot_grid([x[0].cpu().numpy() for x in interpolation], [f"alpha {a:.2f}" for a in alpha[:, 0].tolist()],
              out / "latent_interpolation.png", ncols=7)

    print("[TRACE 7/8] Running UMAP and representation comparison", flush=True)
    latent_vectors = mu.cpu().numpy()
    # Proxy groups are anatomical slice thirds, not diagnostic classes.
    q1, q2 = np.quantile(source_indices, [1 / 3, 2 / 3])
    groups = np.digitize(source_indices, [q1, q2])
    group_names = np.array(["inferior", "middle", "superior"])
    embedding = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=args.seed).fit_transform(latent_vectors)
    raw_vectors = F.interpolate(images, (32, 32), mode="bilinear", align_corners=False).flatten(1).numpy()
    raw_pca = PCA(n_components=min(16, len(raw_vectors) - 1), random_state=args.seed).fit_transform(raw_vectors)
    latent_silhouette = float(silhouette_score(latent_vectors, groups))
    image_silhouette = float(silhouette_score(raw_pca, groups))
    fig, ax = plt.subplots(figsize=(7, 6))
    for group in range(3):
        keep = groups == group
        ax.scatter(embedding[keep, 0], embedding[keep, 1], s=24, alpha=0.8, label=group_names[group])
    ax.set(title="UMAP of TinyVAE slice latents", xlabel="UMAP 1", ylabel="UMAP 2")
    ax.legend(title="Slice-position proxy")
    fig.tight_layout()
    fig.savefig(out / "latent_umap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(range(1, args.epochs + 1), history["train"], label="train")
    ax.plot(range(1, args.epochs + 1), history["val"], label="validation")
    ax.set(xlabel="Epoch", ylabel="VAE loss", title="Training history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "training_history.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "latent_dim": args.latent_dim,
        "image_size": 128,
        "seed": args.seed,
    }
    torch.save(checkpoint, out / "tiny_vae.pt")
    np.savez(out / "latent_codes.npz", latent=latent_vectors, slice_index=source_indices, group=groups)
    metrics = {
        "parameters": parameter_count,
        "epochs": args.epochs,
        "train_seconds": train_seconds,
        "usable_slices": len(images),
        "final_train_loss": history["train"][-1],
        "final_validation_loss": history["val"][-1],
        "mae": mae,
        "mse": mse,
        "psnr_db": psnr,
        "latent_silhouette_proxy": latent_silhouette,
        "image_silhouette_proxy": image_silhouette,
        "proxy_group_warning": "Slice-position thirds are not true biological or diagnostic classes.",
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("[TRACE 8/8] Complete", flush=True)
    print(json.dumps(metrics, indent=2), flush=True)
    print(f"Artifacts: {out.resolve()}", flush=True)


if __name__ == "__main__":
    main()
