from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import VisionTextDualEncoderModel

from Benchmark import Benchmark
from load_from_clip import encode_img
from mlp_eval import MLP_eval


BREAST = 0
OTHER = 1
ABDOMEN = 2
THYROID = 3

NUM_CLASSES = 4

DATASET_ROOT = Path("/lightscratch/users/deschryv/clipFineTune/ultrasound_evaluation")
PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = PROJECT_ROOT / "src" / "multimeditron" / "experts" / "embeddings"





def _resolve_dataset_file(dataset_root: Path, filename: str) -> Path:
    path = dataset_root / filename
    if path.exists():
        return path

    fallback_map = {
        "classifier-lungs-radiopedia-final_test.jsonl": "classifier-lungs-radiopedia-2_test.jsonl",
    }
    fallback = fallback_map.get(filename)
    if fallback is not None:
        fallback_path = dataset_root / fallback
        if fallback_path.exists():
            return fallback_path

    return path

DATASET_FILES = {
    "train": [
        ("classifier-breast-radiopedia-final_train.jsonl", BREAST),
        ("classifier-heart-radiopedia-final_train.jsonl", OTHER),
        ("classifier-lungs-radiopedia-final_train.jsonl", OTHER),
        ("classifier-abdomen-radiopedia-final_train.jsonl", ABDOMEN),
        ("classifier-thyroid-radiopedia-final_train.jsonl", THYROID),
    ],
    "test": [
        ("classifier-breast-radiopedia-final_test.jsonl", BREAST),
        ("classifier-heart-radiopedia-final_test.jsonl", OTHER),
        ("classifier-lungs-radiopedia-final_test.jsonl", OTHER),
        ("classifier-abdomen-radiopedia-final_test.jsonl", ABDOMEN),
        ("classifier-thyroid-radiopedia-final_test.jsonl", THYROID),
    ],
}


def _resolve_image_path(raw_path: str, dataset_root: Path) -> Path:
    if raw_path.startswith("/mloscratch/"):
        raw_path = raw_path.replace("/mloscratch/", "/lightscratch/", 1)

    path = Path(raw_path)
    if path.is_absolute():
        return path

    return dataset_root / path


def _extract_image_path(example: dict, dataset_root: Path) -> Path:
    modalities = example.get("modalities")
    if not modalities:
        raise ValueError("Missing or empty 'modalities' field")

    image_value = modalities[0].get("value")
    if not isinstance(image_value, str) or not image_value:
        raise ValueError("Missing image path in modalities[0]['value']")

    return _resolve_image_path(image_value, dataset_root)


def load_jsonl_embeddings(
    jsonl_path: Path,
    label: int,
    model: VisionTextDualEncoderModel,
    dataset_root: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
    embeddings: list[torch.Tensor] = []
    labels: list[int] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc=jsonl_path.name):
            example = json.loads(line)
            image_path = _extract_image_path(example, dataset_root)

            embedding = encode_img(model, str(image_path))
            embeddings.append(embedding.cpu())
            labels.append(label)

    if not embeddings:
        raise ValueError(f"No embeddings generated from {jsonl_path}")

    return torch.stack(embeddings), torch.tensor(labels, dtype=torch.long)


class BodyPartsDataset(Dataset):
    def __init__(
        self,
        model: VisionTextDualEncoderModel,
        model_name: str,
        split: str,
        dataset_root: Path = DATASET_ROOT,
        cache_root: Path = CACHE_ROOT,
        use_cache: bool = True,
    ) -> None:
        if split not in DATASET_FILES:
            raise ValueError(f"Unknown split: {split}")

        self.dataset_root = Path(dataset_root)
        self.cache_root = Path(cache_root) if cache_root is not None else CACHE_ROOT
        self.cache_root.mkdir(parents=True, exist_ok=True)

        data_cache = self.cache_root / f"{model_name}_{split}_embeddings.pt"
        labels_cache = self.cache_root / f"{model_name}_{split}_labels.pt"

        if use_cache and data_cache.exists() and labels_cache.exists():
            self.data = torch.load(data_cache, map_location="cpu")
            self.labels = torch.load(labels_cache, map_location="cpu")
            return

        tensors = []
        label_tensors = []

        for filename, label in DATASET_FILES[split]:
            data, labels = load_jsonl_embeddings(
                jsonl_path=_resolve_dataset_file(self.dataset_root, filename),
                label=label,
                model=model,
                dataset_root=self.dataset_root,
            )
            tensors.append(data)
            label_tensors.append(labels)

        self.data = torch.cat(tensors, dim=0)
        self.labels = torch.cat(label_tensors, dim=0)

        torch.save(self.data, data_cache)
        torch.save(self.labels, labels_cache)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.data[idx], self.labels[idx]


def build_class_weights(labels: torch.Tensor) -> torch.Tensor:
    labels_np = labels.cpu().numpy().astype(int)
    classes = np.unique(labels_np)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=labels_np,
    )
    return torch.tensor(weights, dtype=torch.float32)


def evaluate_pipeline(model: VisionTextDualEncoderModel, model_name: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print("Starting anatomical ultrasound evaluation")

    train_dataset = BodyPartsDataset(
        model=model,
        model_name=model_name,
        split="train",
        use_cache=True,
    )
    print(f"Training dataset loaded: {len(train_dataset)} samples")

    test_dataset = BodyPartsDataset(
        model=model,
        model_name=model_name,
        split="test",
        use_cache=True,
    )
    print(f"Test dataset loaded: {len(test_dataset)} samples")

    class_weights = build_class_weights(train_dataset.labels)
    loss = nn.CrossEntropyLoss(weight=class_weights)

    benchmark = MLP_eval(
        output_dim=NUM_CLASSES,
        training_set=train_dataset,
        test_set=test_dataset,
        loss=loss,
    )
    return benchmark.evaluate()


class AnatomicalBenchmark(Benchmark):
    def evaluate(self, model_path: str):
        model = VisionTextDualEncoderModel.from_pretrained(model_path)
        model_name = Path(model_path).name
        return evaluate_pipeline(model=model, model_name=model_name)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python ultrasound_new_benchmark.py <model_path>")

    benchmark = AnatomicalBenchmark()
    result = benchmark.evaluate(sys.argv[1])
    print(result)
