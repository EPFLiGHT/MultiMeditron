from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import VisionTextDualEncoderModel

from load_from_clip import encode_img


DEFAULT_CACHE_ROOT = Path(__file__).resolve().parents[1] / "embeddings"


class BenchmarkDataset(Dataset):
    """Simple dataset wrapper around precomputed embeddings and labels."""

    def __init__(self, data: torch.Tensor, labels: torch.Tensor) -> None:
        self.data = data
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]


def resolve_image_path(raw_path: str, dataset_root: Path) -> Path:
    if raw_path.startswith("/mloscratch/"):
        raw_path = raw_path.replace("/mloscratch/", "/lightscratch/", 1)

    path = Path(raw_path)
    if path.is_absolute():
        return path
    return dataset_root / path


def read_jsonl(jsonl_path: Path) -> list[dict]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def load_embeddings_from_examples(
    examples: list[dict],
    labels: list[int],
    model: VisionTextDualEncoderModel,
    dataset_root: Path,
    desc: str,
    embed_example: Callable[[dict, int, VisionTextDualEncoderModel, Path], torch.Tensor | None] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute image embeddings for a list of examples using the vision encoder.

    If embed_example is provided, it is called per example (useful for benchmarks
    with custom image loading logic). Otherwise falls back to the default path:
    resolve the image path from example['modalities'][0]['value'] and call encode_img.
    Examples whose image is missing or whose embed_example returns None are skipped.
    """
    embeddings = []
    kept_labels = []
    missing_images = 0

    if len(examples) != len(labels):
        raise ValueError(f"Examples/labels length mismatch for {desc}: {len(examples)} != {len(labels)}")

    for example, label in tqdm(zip(examples, labels, strict=True), total=len(examples), desc=desc):
        if embed_example is not None:
            embedding = embed_example(example, label, model, dataset_root)
            if embedding is None:
                missing_images += 1
                continue
        else:
            image_value = example["modalities"][0]["value"]
            image_path = resolve_image_path(image_value, dataset_root)

            if not image_path.exists():
                missing_images += 1
                continue

            embedding = encode_img(model, str(image_path))
        embeddings.append(embedding.cpu())
        kept_labels.append(label)

    if not embeddings:
        raise ValueError(f"No embeddings generated for {desc} (all images missing?)")

    if missing_images:
        print(f"[{desc}] skipped {missing_images} missing image(s)")

    return torch.stack(embeddings), torch.tensor(kept_labels, dtype=torch.long)


def load_or_build_dataset(
    *,
    cache_prefix: str,
    examples: list[dict],
    labels: list[int],
    model: VisionTextDualEncoderModel,
    dataset_root: Path,
    cache_root: Path | None = None,
    use_cache: bool = True,
    desc: str,
    prepare_images: Callable[[list[dict], Path, str], None] | None = None,
    embed_example: Callable[[dict, int, VisionTextDualEncoderModel, Path], torch.Tensor | None] | None = None,
) -> BenchmarkDataset:
    """Load a cached embedding dataset or build it from scratch.

    Cache files are stored as two .pt files: {cache_root}/{cache_prefix}_embeddings.pt
    and {cache_prefix}_labels.pt. If both exist and use_cache=True, they are loaded
    directly without re-encoding. Otherwise embeddings are computed and saved for
    future runs. The cache key is cache_prefix — it must encode the model identity
    and split name to avoid stale cache hits across different models or train/test splits.
    """
    cache_root = Path(cache_root or DEFAULT_CACHE_ROOT)
    cache_root.mkdir(parents=True, exist_ok=True)

    data_cache = cache_root / f"{cache_prefix}_embeddings.pt"
    labels_cache = cache_root / f"{cache_prefix}_labels.pt"

    if use_cache and data_cache.exists() and labels_cache.exists():
        data = torch.load(data_cache, map_location="cpu")
        labels_tensor = torch.load(labels_cache, map_location="cpu")
        return BenchmarkDataset(data=data, labels=labels_tensor)

    if prepare_images is not None:
        prepare_images(examples, dataset_root, desc)

    data, labels_tensor = load_embeddings_from_examples(
        examples=examples,
        labels=labels,
        model=model,
        dataset_root=dataset_root,
        desc=desc,
        embed_example=embed_example,
    )
    torch.save(data, data_cache)
    torch.save(labels_tensor, labels_cache)
    return BenchmarkDataset(data=data, labels=labels_tensor)


def build_class_weights(labels: torch.Tensor, num_classes: int | None = None) -> torch.Tensor:
    labels_np = labels.cpu().numpy().astype(int)
    classes = np.arange(num_classes) if num_classes is not None else np.unique(labels_np)
    # For classes absent from the subset, assign weight 1.0
    present = np.unique(labels_np)
    weights = np.ones(len(classes), dtype=np.float64)
    if len(present) > 1:
        present_weights = compute_class_weight(class_weight="balanced", classes=present, y=labels_np)
        for i, c in enumerate(classes):
            if c in present:
                weights[i] = present_weights[np.where(present == c)[0][0]]
    return torch.tensor(weights, dtype=torch.float32)
