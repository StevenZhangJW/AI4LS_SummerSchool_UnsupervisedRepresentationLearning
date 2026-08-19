"""Tutorial 2: DINO features and a tailored rotation-prediction pretext task."""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.cbook as cbook
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, TensorDataset, random_split
from transformers import AutoImageProcessor, AutoModel
import umap


SEED = 42
MODEL_ID = os.environ.get("DINO_MODEL_ID", "facebook/dinov2-small")  # Public fallback for gated DINOv3.


def load_axial_slices(path: Path):
    volume = nib.as_closest_canonical(nib.load(path)).get_fdata(dtype=np.float32)
    volume = (volume - volume.min()) / max(float(volume.max() - volume.min()), 1e-8)
    slices = torch.from_numpy(np.moveaxis(volume, 2, 0)).unsqueeze(1)
    keep = slices.flatten(1).amax(1) > 0.05
    return slices[keep].float(), torch.arange(len(slices))[keep].numpy()


def to_pil(batch):
    return [Image.fromarray((x[0].numpy() * 255).astype(np.uint8)).convert("RGB") for x in batch]


def save_feature_figure(image, attention, pca_rgb, path, title):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    axes[0].imshow(image)
    axes[0].set_title(title)
    axes[1].imshow(image)
    axes[1].imshow(attention, cmap="inferno", alpha=0.55)
    axes[1].set_title("CLS attention")
    axes[2].imshow(pca_rgb)
    axes[2].set_title("Patch features: PCA → RGB")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


class RotationNet(nn.Module):
    def __init__(self, embedding_dim=64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 8, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(8, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
        )
        # Retain the 8x8 spatial grid: global pooling would erase orientation.
        self.project = nn.Linear(32 * 8 * 8, embedding_dim)
        self.classifier = nn.Linear(embedding_dim, 4)

    def forward(self, x, return_embedding=False):
        z = self.project(self.features(x).flatten(1))
        return z if return_embedding else self.classifier(F.relu(z))


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = Path("data/class0/aadd_00000.nii.gz")
    out = Path("outputs/tutorial2")
    out.mkdir(parents=True, exist_ok=True)

    print("[TRACE 1/7] Loading class0 slices", flush=True)
    slices, slice_indices = load_axial_slices(source)
    selected = torch.linspace(0, len(slices) - 1, 30).round().long().unique()
    dino_slices = slices[selected]
    dino_indices = slice_indices[selected.numpy()]
    proxy = np.digitize(dino_indices, np.quantile(dino_indices, [1 / 3, 2 / 3]))

    print("[TRACE 2/7] Loading frozen DINO", MODEL_ID, flush=True)
    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    dino = AutoModel.from_pretrained(MODEL_ID, attn_implementation="eager").to(device).eval()
    dino.requires_grad_(False)
    cls_vectors = []
    mean_patch_vectors = []
    start = time.perf_counter()
    for start_idx in range(0, len(dino_slices), 5):
        batch_pil = to_pil(dino_slices[start_idx:start_idx + 5])
        inputs = processor(images=batch_pil, return_tensors="pt").to(device)
        with torch.inference_mode():
            result = dino(**inputs)
        tokens = result.last_hidden_state
        cls_vectors.append(tokens[:, 0].cpu())
        mean_patch_vectors.append(tokens[:, 1:].mean(1).cpu())
    cls_vectors = torch.cat(cls_vectors).numpy()
    mean_patch_vectors = torch.cat(mean_patch_vectors).numpy()
    dino_seconds = time.perf_counter() - start
    print("DINO CLS matrix:", cls_vectors.shape, "seconds:", round(dino_seconds, 2), flush=True)

    print("[TRACE 3/7] Attention and PCA patch maps", flush=True)
    middle_pil = to_pil(dino_slices[len(dino_slices) // 2:len(dino_slices) // 2 + 1])[0]
    inputs = processor(images=middle_pil, return_tensors="pt").to(device)
    with torch.inference_mode():
        result = dino(**inputs, output_attentions=True)
    tokens = result.last_hidden_state
    patch_size = dino.config.patch_size
    h = inputs.pixel_values.shape[-2] // patch_size
    w = inputs.pixel_values.shape[-1] // patch_size
    patch = tokens[:, 1:, :]
    patch_count = h * w
    patch = patch[:, -patch_count:, :]
    attention = result.attentions[-1][0, :, 0, -patch_count:].mean(0).reshape(h, w)
    attention = F.interpolate(attention[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    patch_np = patch[0].cpu().numpy()
    rgb = PCA(n_components=3, random_state=SEED).fit_transform(patch_np).reshape(h, w, 3)
    rgb = (rgb - rgb.min((0, 1), keepdims=True)) / (np.ptp(rgb, axis=(0, 1), keepdims=True) + 1e-8)
    image_224 = np.asarray(middle_pil.resize((224, 224)))
    save_feature_figure(image_224, attention.cpu().numpy(), rgb, out / "medical_attention_pca.png", "AADD middle slice")

    natural = Image.open(cbook.get_sample_data("grace_hopper.jpg")).convert("RGB")
    inputs_nat = processor(images=natural, return_tensors="pt").to(device)
    with torch.inference_mode():
        result_nat = dino(**inputs_nat, output_attentions=True)
    nat_tokens = result_nat.last_hidden_state
    nh = inputs_nat.pixel_values.shape[-2] // patch_size
    nw = inputs_nat.pixel_values.shape[-1] // patch_size
    ncount = nh * nw
    nat_attention = result_nat.attentions[-1][0, :, 0, -ncount:].mean(0).reshape(nh, nw)
    nat_attention = F.interpolate(nat_attention[None, None], size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    nat_attention = (nat_attention - nat_attention.min()) / (nat_attention.max() - nat_attention.min() + 1e-8)
    nat_patch = nat_tokens[0, -ncount:].cpu().numpy()
    nat_rgb = PCA(n_components=3, random_state=SEED).fit_transform(nat_patch).reshape(nh, nw, 3)
    nat_rgb = (nat_rgb - nat_rgb.min((0, 1), keepdims=True)) / (np.ptp(nat_rgb, axis=(0, 1), keepdims=True) + 1e-8)
    save_feature_figure(np.asarray(natural.resize((224, 224))), nat_attention.cpu().numpy(), nat_rgb,
                        out / "natural_attention_pca.png", "Natural-image comparison")

    print("[TRACE 4/7] UMAP and image-space comparison", flush=True)
    embedding = umap.UMAP(n_neighbors=8, min_dist=0.15, random_state=SEED).fit_transform(cls_vectors)
    raw = F.interpolate(dino_slices, (32, 32), mode="bilinear", align_corners=False).flatten(1).numpy()
    raw_pca = PCA(n_components=min(16, len(raw) - 1), random_state=SEED).fit_transform(raw)
    dino_silhouette = float(silhouette_score(cls_vectors, proxy))
    raw_silhouette = float(silhouette_score(raw_pca, proxy))
    fig, ax = plt.subplots(figsize=(7, 6))
    names = ["inferior", "middle", "superior"]
    for group in range(3):
        mask = proxy == group
        ax.scatter(embedding[mask, 0], embedding[mask, 1], label=names[group], s=45)
    ax.set(title="DINO CLS-token UMAP", xlabel="UMAP 1", ylabel="UMAP 2")
    ax.legend(title="Slice-position proxy")
    fig.tight_layout()
    fig.savefig(out / "dino_cls_umap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("[TRACE 5/7] Building rotation-prediction pretext dataset", flush=True)
    small = F.interpolate(slices, (64, 64), mode="bilinear", align_corners=False)
    rotated, labels = [], []
    for k in range(4):
        rotated.append(torch.rot90(small, k, dims=(-2, -1)))
        labels.append(torch.full((len(small),), k, dtype=torch.long))
    rotation_images = torch.cat(rotated)
    rotation_labels = torch.cat(labels)
    permutation = torch.randperm(len(rotation_images), generator=torch.Generator().manual_seed(SEED))
    rotation_images, rotation_labels = rotation_images[permutation], rotation_labels[permutation]
    dataset = TensorDataset(rotation_images, rotation_labels)
    n_val = round(0.2 * len(dataset))
    train_set, val_set = random_split(dataset, [len(dataset) - n_val, n_val],
                                      generator=torch.Generator().manual_seed(SEED))
    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=64)

    print("[TRACE 6/7] Training tailored rotation task", flush=True)
    rotation_model = RotationNet().to(device)
    optimizer = torch.optim.Adam(rotation_model.parameters(), lr=2e-3)
    history = []
    for epoch in range(8):
        rotation_model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.cross_entropy(rotation_model(x), y)
            loss.backward()
            optimizer.step()
        rotation_model.eval()
        correct = total = 0
        with torch.inference_mode():
            for x, y in val_loader:
                pred = rotation_model(x.to(device)).argmax(1).cpu()
                correct += int((pred == y).sum())
                total += len(y)
        accuracy = correct / total
        history.append(accuracy)
        print(f"epoch={epoch + 1} validation_accuracy={accuracy:.3f}", flush=True)
    torch.save(rotation_model.cpu().state_dict(), out / "rotation_net.pt")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(1, len(history) + 1), history, marker="o")
    ax.axhline(0.25, color="gray", linestyle="--", label="chance")
    ax.set(xlabel="Epoch", ylabel="Validation accuracy", ylim=(0, 1), title="Rotation pretext task")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "rotation_training.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    metrics = {
        "dino_model": MODEL_ID,
        "dino_parameters": sum(p.numel() for p in dino.parameters()),
        "processed_slices": len(dino_slices),
        "dino_seconds": dino_seconds,
        "cls_shape": list(cls_vectors.shape),
        "dino_proxy_silhouette": dino_silhouette,
        "raw_image_proxy_silhouette": raw_silhouette,
        "rotation_parameters": sum(p.numel() for p in rotation_model.parameters()),
        "rotation_validation_accuracy": history[-1],
        "warning": "Slice-position thirds are proxy groups, not diagnostic classes.",
    }
    np.savez(out / "dino_features.npz", cls=cls_vectors, mean_patch=mean_patch_vectors,
             slice_index=dino_indices, proxy_group=proxy)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print("[TRACE 7/7] Complete")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
