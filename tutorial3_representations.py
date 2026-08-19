"""Tutorial 3: implicit coordinate MLP and explicit 2D Gaussian representation."""

from __future__ import annotations

import json
import math
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


SEED = 42


def load_middle_slice(path: Path, size=128):
    volume = nib.as_closest_canonical(nib.load(path)).get_fdata(dtype=np.float32)
    volume = (volume - volume.min()) / max(float(volume.max() - volume.min()), 1e-8)
    slices = torch.from_numpy(np.moveaxis(volume, 2, 0)).unsqueeze(1)
    valid = torch.where(slices.flatten(1).amax(1) > 0.05)[0]
    index = int(valid[len(valid) // 2])
    image = F.interpolate(slices[index:index + 1], (size, size), mode="bilinear", align_corners=False)
    return image[0, 0].float(), index


def coordinate_grid(size, device):
    axis = torch.linspace(-1, 1, size, device=device)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    return torch.stack([xx, yy], dim=-1).reshape(-1, 2)


class FourierFeatures(nn.Module):
    def __init__(self, levels=6):
        super().__init__()
        self.register_buffer("frequencies", 2.0 ** torch.arange(levels) * math.pi)

    def forward(self, coordinates):
        phases = coordinates[..., None] * self.frequencies
        return torch.cat([coordinates, phases.sin().flatten(-2), phases.cos().flatten(-2)], dim=-1)


class CoordinateMLP(nn.Module):
    def __init__(self, levels=6, width=64):
        super().__init__()
        self.encoding = FourierFeatures(levels)
        input_dim = 2 + 4 * levels
        self.network = nn.Sequential(
            nn.Linear(input_dim, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, 1), nn.Sigmoid(),
        )

    def forward(self, coordinates):
        return self.network(self.encoding(coordinates)).squeeze(-1)


class GaussianImage(nn.Module):
    """An explicit image represented by a finite set of axis-aligned Gaussians."""
    def __init__(self, target, count=128):
        super().__init__()
        size = target.shape[0]
        grid = coordinate_grid(size, target.device)
        weights = target.flatten().clamp_min(1e-4)
        selected = torch.multinomial(weights / weights.sum(), count, replacement=True)
        self.centers = nn.Parameter(grid[selected] + 0.01 * torch.randn(count, 2, device=target.device))
        self.log_scales = nn.Parameter(torch.full((count, 2), math.log(0.08), device=target.device))
        initial = target.flatten()[selected].clamp(0.02, 0.5)
        self.logits = nn.Parameter(torch.logit(initial.clamp(1e-4, 1 - 1e-4)))

    def forward(self, coordinates, chunk=8192):
        outputs = []
        scales = self.log_scales.exp().clamp(0.01, 0.5)
        amplitudes = self.logits.sigmoid()
        for part in coordinates.split(chunk):
            delta = (part[:, None, :] - self.centers[None, :, :]) / scales[None, :, :]
            basis = torch.exp(-0.5 * delta.square().sum(-1))
            outputs.append((basis * amplitudes).sum(-1).clamp(0, 1))
        return torch.cat(outputs)


def psnr(prediction, target):
    mse = F.mse_loss(prediction, target).item()
    return float(10 * np.log10(1.0 / max(mse, 1e-12)))


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path("outputs/tutorial3")
    out.mkdir(parents=True, exist_ok=True)

    print("[TRACE 1/6] Loading representative class0 slice", flush=True)
    target, slice_index = load_middle_slice(Path("data/class0/aadd_00000.nii.gz"))
    target = target.to(device)
    coordinates = coordinate_grid(len(target), device)
    values = target.flatten()

    print("[TRACE 2/6] Fitting implicit coordinate MLP", flush=True)
    implicit = CoordinateMLP().to(device)
    optimizer = torch.optim.Adam(implicit.parameters(), lr=2e-3)
    implicit_losses = []
    start = time.perf_counter()
    for step in range(1200):
        chosen = torch.randint(len(coordinates), (4096,), device=device)
        prediction = implicit(coordinates[chosen])
        loss = F.mse_loss(prediction, values[chosen])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 100 == 0:
            implicit_losses.append(float(loss))
    implicit_seconds = time.perf_counter() - start
    with torch.inference_mode():
        implicit_image = implicit(coordinates).reshape_as(target)
    print("Implicit seconds:", round(implicit_seconds, 2), "PSNR:", round(psnr(implicit_image, target), 2), flush=True)

    print("[TRACE 3/6] Fitting explicit Gaussian representation", flush=True)
    # Fit at 64x64 for a lightweight demonstration, then render at 128x128.
    target64 = F.interpolate(target[None, None], (64, 64), mode="bilinear", align_corners=False)[0, 0]
    coords64 = coordinate_grid(64, device)
    gaussian = GaussianImage(target64, count=128).to(device)
    optimizer = torch.optim.Adam(gaussian.parameters(), lr=2e-2)
    gaussian_losses = []
    start = time.perf_counter()
    for step in range(500):
        prediction = gaussian(coords64)
        loss = F.mse_loss(prediction, target64.flatten())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            gaussian_losses.append(float(loss))
    gaussian_seconds = time.perf_counter() - start
    with torch.inference_mode():
        gaussian_image = gaussian(coordinates).reshape_as(target)
    print("Gaussian seconds:", round(gaussian_seconds, 2), "PSNR:", round(psnr(gaussian_image, target), 2), flush=True)

    print("[TRACE 4/6] Rendering arbitrary 256x256 resolution", flush=True)
    coords256 = coordinate_grid(256, device)
    with torch.inference_mode():
        implicit_256 = implicit(coords256).reshape(256, 256)
        gaussian_256 = gaussian(coords256).reshape(256, 256)

    print("[TRACE 5/6] Saving comparisons and representations", flush=True)
    target_np = target.cpu().numpy()
    implicit_np = implicit_image.cpu().numpy()
    gaussian_np = gaussian_image.cpu().numpy()
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    axes[0, 0].imshow(target_np, cmap="gray", vmin=0, vmax=1); axes[0, 0].set_title("Target")
    axes[0, 1].imshow(implicit_np, cmap="gray", vmin=0, vmax=1); axes[0, 1].set_title("Implicit coordinate MLP")
    axes[0, 2].imshow(gaussian_np, cmap="gray", vmin=0, vmax=1); axes[0, 2].set_title("Explicit 128 Gaussians")
    axes[1, 0].axis("off")
    axes[1, 1].imshow(np.abs(target_np - implicit_np), cmap="magma"); axes[1, 1].set_title("Implicit absolute error")
    axes[1, 2].imshow(np.abs(target_np - gaussian_np), cmap="magma"); axes[1, 2].set_title("Gaussian absolute error")
    for ax in axes.ravel(): ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "representation_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(implicit_256.cpu(), cmap="gray", vmin=0, vmax=1); axes[0].set_title("Implicit render 256x256")
    axes[1].imshow(gaussian_256.cpu(), cmap="gray", vmin=0, vmax=1); axes[1].set_title("Gaussian render 256x256")
    for ax in axes: ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "arbitrary_resolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(np.arange(len(implicit_losses)) * 100, implicit_losses)
    axes[0].set(title="Implicit fitting", xlabel="Step", ylabel="MSE")
    axes[1].plot(np.arange(len(gaussian_losses)) * 50, gaussian_losses)
    axes[1].set(title="Gaussian fitting", xlabel="Step", ylabel="MSE")
    fig.tight_layout()
    fig.savefig(out / "fitting_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    torch.save(implicit.cpu().state_dict(), out / "implicit_mlp.pt")
    torch.save({"centers": gaussian.centers.detach().cpu(),
                "log_scales": gaussian.log_scales.detach().cpu(),
                "logits": gaussian.logits.detach().cpu()}, out / "explicit_gaussians.pt")
    implicit_parameters = sum(p.numel() for p in implicit.parameters())
    gaussian_parameters = sum(p.numel() for p in gaussian.parameters())
    metrics = {
        "slice_index": slice_index,
        "target_shape": list(target.shape),
        "implicit_parameters": implicit_parameters,
        "implicit_seconds": implicit_seconds,
        "implicit_mae": F.l1_loss(implicit_image, target).item(),
        "implicit_psnr_db": psnr(implicit_image, target),
        "gaussian_count": 128,
        "gaussian_parameters": gaussian_parameters,
        "gaussian_seconds": gaussian_seconds,
        "gaussian_mae": F.l1_loss(gaussian_image, target).item(),
        "gaussian_psnr_db": psnr(gaussian_image, target),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("[TRACE 6/6] Complete")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
