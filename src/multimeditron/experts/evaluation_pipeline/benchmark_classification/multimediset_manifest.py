import json
import random
from functools import lru_cache
from pathlib import Path

import torch
from datasets import DatasetDict, load_from_disk
from tqdm import tqdm

from load_from_clip import encode_img, encode_img_bytes

from .datasets import BenchmarkDataset, DEFAULT_CACHE_ROOT


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_MANIFEST_ROOT = REPO_ROOT / "benchmark_splits" / "multimediset"


def read_manifest(manifest_path):
    path = Path(manifest_path)
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_records(records, max_examples, seed):
    if max_examples is None or len(records) <= max_examples:
        return records
    rng = random.Random(seed)
    shuffled = records.copy()
    rng.shuffle(shuffled)
    return shuffled[:max_examples]


def sample_records_stratified(records, max_examples, seed):
    """Sample up to max_examples with equal representation per label_id.

    Classes smaller than the per-class budget are taken in full; their unused
    budget is redistributed to larger classes so the total stays at max_examples.
    """
    if max_examples is None or len(records) <= max_examples:
        return records

    rng = random.Random(seed)
    by_label = {}
    for record in records:
        key = str(record.get("label_id", record.get("label", "unknown")))
        by_label.setdefault(key, []).append(record)

    for label_records in by_label.values():
        rng.shuffle(label_records)

    # Sort ascending by class size so small classes are fully included first.
    sorted_classes = sorted(by_label.items(), key=lambda x: len(x[1]))
    selected = []
    remaining_budget = max_examples
    remaining_classes = len(sorted_classes)

    for _label_id, label_records in sorted_classes:
        per_class = remaining_budget // remaining_classes
        take = min(len(label_records), per_class)
        selected.extend(label_records[:take])
        remaining_budget -= take
        remaining_classes -= 1

    rng.shuffle(selected)
    return selected


@lru_cache(maxsize=16)
def _load_source_dataset(source_root):
    root = Path(source_root)
    if (root / "dataset_dict.json").exists() or (root / "state.json").exists():
        loaded = load_from_disk(str(root))
        if isinstance(loaded, DatasetDict):
            return {split: loaded[split] for split in loaded.keys()}
        return {"train": loaded}

    raise FileNotFoundError(f"Could not load source dataset from {root}")


def _read_jsonl(path):
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _get_source_row(record):
    source_dataset = _load_source_dataset(record["source_root"])
    split = record["source_split"]
    if split not in source_dataset:
        raise KeyError(f"Split {split!r} not found in {record['source_root']}")
    return dict(source_dataset[split][int(record["source_index"])])


def _first_image_bytes(row):
    image = row.get("image")
    if isinstance(image, dict) and image.get("bytes") is not None:
        return image["bytes"]

    modalities_images = row.get("modalities_images")
    if isinstance(modalities_images, list) and modalities_images:
        first = modalities_images[0]
        if isinstance(first, dict) and first.get("bytes") is not None:
            return first["bytes"]

    return None


def _first_image_path(row, source_root):
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


def encode_manifest_record(model, record, row=None):
    if row is None:
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
    manifest_path,
    cache_prefix,
    model,
    cache_root=None,
    use_cache=True,
    desc,
    max_examples=None,
    seed=42,
    label_builder=None,
    allowed_subdatasets=None,
    stratify_by_label=False,
):
    cache_dir = Path(cache_root or DEFAULT_CACHE_ROOT)
    cache_dir.mkdir(parents=True, exist_ok=True)
    data_cache = cache_dir / f"{cache_prefix}_embeddings.pt"
    labels_cache = cache_dir / f"{cache_prefix}_labels.pt"

    if use_cache and data_cache.exists() and labels_cache.exists():
        return BenchmarkDataset(
            data=torch.load(data_cache, map_location="cpu", weights_only=True),
            labels=torch.load(labels_cache, map_location="cpu", weights_only=True),
        )

    records = read_manifest(manifest_path)
    if allowed_subdatasets is not None:
        records = [r for r in records if r.get("subdataset") in allowed_subdatasets]
    if stratify_by_label:
        records = sample_records_stratified(records, max_examples=max_examples, seed=seed)
    else:
        records = sample_records(records, max_examples=max_examples, seed=seed)
    embeddings = []
    labels = []
    skipped = 0

    for record in tqdm(records, desc=desc):
        try:
            row = _get_source_row(record)
            embedding = encode_manifest_record(model, record, row=row)
        except (FileNotFoundError, OSError, ValueError) as exc:
            skipped += 1
            print(
                f"[{desc}] skipped {record.get('dataset')}:{record.get('source_index')} ({exc})"
            )
            continue
        embeddings.append(embedding.cpu())
        if label_builder is None:
            labels.append(int(record["label_id"]))
        else:
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
