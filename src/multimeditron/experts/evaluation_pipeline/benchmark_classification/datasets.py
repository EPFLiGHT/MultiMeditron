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

from ..load_from_clip import encode_img


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


def build_class_weights(labels: torch.Tensor) -> torch.Tensor:
    labels_np = labels.cpu().numpy().astype(int)
    classes = np.unique(labels_np)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels_np)
    return torch.tensor(weights, dtype=torch.float32)
