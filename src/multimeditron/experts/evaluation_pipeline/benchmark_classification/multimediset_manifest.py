from __future__ import annotations

import json
import random
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import torch
from datasets import DatasetDict, load_from_disk
from tqdm import tqdm

from load_from_clip import encode_img, encode_img_bytes

from .datasets import BenchmarkDataset, DEFAULT_CACHE_ROOT


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_MANIFEST_ROOT = REPO_ROOT / "benchmark_splits" / "multimediset"


def read_manifest(manifest_path: str | Path) -> list[dict[str, Any]]:
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_records(records: list[dict[str, Any]], max_examples: int | None, seed: int) -> list[dict[str, Any]]:
    if max_examples is None or len(records) <= max_examples:
        return records
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)
    return shuffled[:max_examples]


@lru_cache(maxsize=16)
def _load_source_dataset(source_root: str):
    root = Path(source_root)
    if (root / "dataset_dict.json").exists() or (root / "state.json").exists():
        loaded = load_from_disk(str(root))
        if isinstance(loaded, DatasetDict):
            return {split: loaded[split] for split in loaded.keys()}
        return {"train": loaded}

    train_jsonl = root / "MRI-glob-train.jsonl"
    test_jsonl = root / "MRI-glob-test.jsonl"
    if train_jsonl.exists() and test_jsonl.exists():
        return {
            "train": _read_jsonl(train_jsonl),
            "test": _read_jsonl(test_jsonl),
        }

    jsonl = root / "MRI-glob.jsonl"
    if jsonl.exists():
        return {"all": _read_jsonl(jsonl)}

    raise FileNotFoundError(f"Could not load source dataset from {root}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _get_source_row(record: dict[str, Any]) -> dict[str, Any]:
    source_dataset = _load_source_dataset(record["source_root"])
    split = record["source_split"]
    if split not in source_dataset:
        raise KeyError(f"Split {split!r} not found in {record['source_root']}")
    return dict(source_dataset[split][int(record["source_index"])])


def _first_image_bytes(row: dict[str, Any]) -> bytes | None:
    image = row.get("image")
    if isinstance(image, dict) and image.get("bytes") is not None:
        return image["bytes"]

    modalities_images = row.get("modalities_images")
    if isinstance(modalities_images, list) and modalities_images:
        first = modalities_images[0]
        if isinstance(first, dict) and first.get("bytes") is not None:
            return first["bytes"]

    return None


def _first_image_path(row: dict[str, Any], source_root: Path) -> Path | None:
    modalities = row.get("modalities") or []
    if not modalities:
        return None
    first = modalities[0]
    if not isinstance(first, dict):
        return None
    value = first.get("value")
    if not isinstance(value, str) or not value:
        return None

    path = Path(value)
    if path.is_absolute():
        return path if path.exists() else None

    candidates = [
        source_root / path,
        source_root / path.name,
        source_root / "images" / path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def encode_manifest_record(model, record: dict[str, Any]) -> torch.Tensor:
    row = _get_source_row(record)
    image_bytes = _first_image_bytes(row)
    if image_bytes is not None:
        return encode_img_bytes(model, image_bytes)

    image_path = _first_image_path(row, Path(record["source_root"]))
    if image_path is None:
        raise FileNotFoundError(
            "Could not resolve image for manifest record "
            f"{record['dataset']}:{record['source_split']}[{record['source_index']}]"
        )
    return encode_img(model, str(image_path))


def load_or_build_manifest_dataset(
    *,
    manifest_path: str | Path,
    cache_prefix: str,
    model,
    cache_root: str | Path | None = None,
    use_cache: bool = True,
    desc: str,
    max_examples: int | None = None,
    seed: int = 42,
    label_builder: Callable[[dict[str, Any], dict[str, Any]], torch.Tensor | int] | None = None,
) -> BenchmarkDataset:
    cache_dir = Path(cache_root or DEFAULT_CACHE_ROOT)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_cache = cache_dir / f"{cache_prefix}_embeddings.pt"
    labels_cache = cache_dir / f"{cache_prefix}_labels.pt"

    if use_cache and data_cache.exists() and labels_cache.exists():
        return BenchmarkDataset(
            data=torch.load(data_cache, map_location="cpu"),
            labels=torch.load(labels_cache, map_location="cpu"),
        )

    records = sample_records(read_manifest(manifest_path), max_examples=max_examples, seed=seed)
    embeddings: list[torch.Tensor] = []
    labels: list[torch.Tensor | int] = []
    skipped = 0

    for record in tqdm(records, desc=desc):
        try:
            embedding = encode_manifest_record(model, record)
        except (FileNotFoundError, OSError, ValueError) as exc:
            skipped += 1
            print(f"[{desc}] skipped {record.get('dataset')}:{record.get('source_index')} ({exc})")
            continue
        embeddings.append(embedding.cpu())
        if label_builder is None:
            labels.append(int(record["label_id"]))
        else:
            row = _get_source_row(record)
            labels.append(label_builder(row, record))

    if not embeddings:
        raise ValueError(f"No embeddings generated for {desc}")
    if skipped:
        print(f"[{desc}] skipped {skipped} record(s)")

    data = torch.stack(embeddings)
    if labels and isinstance(labels[0], torch.Tensor):
        labels_tensor = torch.stack([label.float() for label in labels])
    else:
        labels_tensor = torch.tensor(labels, dtype=torch.long)
    torch.save(data, data_cache)
    torch.save(labels_tensor, labels_cache)
    return BenchmarkDataset(data=data, labels=labels_tensor)
