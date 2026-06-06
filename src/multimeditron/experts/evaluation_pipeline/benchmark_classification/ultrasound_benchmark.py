import json
import sys
from pathlib import Path

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from .base import ClassificationBenchmark
from .multimediset_manifest import DEFAULT_MANIFEST_ROOT, load_or_build_manifest_dataset
from load_from_clip import encode_img


BREAST = 0
OTHER = 1
ABDOMEN = 2
THYROID = 3

NUM_CLASSES = 4
MANIFEST_NUM_CLASSES = 13

DATASET_ROOT = Path("/lightscratch/users/deschryv/clipFineTune/ultrasound_evaluation")
PROJECT_ROOT = Path(__file__).resolve().parents[4]
CACHE_ROOT = PROJECT_ROOT / "src" / "multimeditron" / "experts" / "embeddings"

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

# Some dataset files were renamed — fall back to alternate names when missing.
_FILENAME_FALLBACKS = {
    "classifier-lungs-radiopedia-final_test.jsonl": "classifier-lungs-radiopedia-2_test.jsonl",
}


def _resolve_dataset_file(dataset_root, filename):
    path = dataset_root / filename
    if path.exists():
        return path
    fallback = _FILENAME_FALLBACKS.get(filename)
    if fallback is not None:
        fallback_path = dataset_root / fallback
        if fallback_path.exists():
            return fallback_path
    return path


def _resolve_image_path(raw_path, dataset_root):
    if raw_path.startswith("/mloscratch/"):
        raw_path = raw_path.replace("/mloscratch/", "/lightscratch/", 1)
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return dataset_root / path


def _extract_image_path(example, dataset_root):
    modalities = example.get("modalities")
    if not modalities:
        raise ValueError("Missing or empty 'modalities' field")
    image_value = modalities[0].get("value")
    if not isinstance(image_value, str) or not image_value:
        raise ValueError("Missing image path in modalities[0]['value']")
    return _resolve_image_path(image_value, dataset_root)


def _load_jsonl_embeddings(jsonl_path, label, model, dataset_root):
    embeddings = []
    labels = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in tqdm(f, desc=jsonl_path.name):
            example = json.loads(line)
            image_path = _extract_image_path(example, dataset_root)
            embeddings.append(encode_img(model, str(image_path)).cpu())
            labels.append(label)
    if not embeddings:
        raise ValueError(f"No embeddings generated from {jsonl_path}")
    return torch.stack(embeddings), torch.tensor(labels, dtype=torch.long)


class BodyPartsDataset(Dataset):
    def __init__(
        self,
        model,
        model_name,
        split,
        dataset_root=DATASET_ROOT,
        cache_root=CACHE_ROOT,
        use_cache=True,
    ):
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
            data, labels = _load_jsonl_embeddings(
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

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class UltrasoundBenchmark(ClassificationBenchmark):
    num_classes = NUM_CLASSES
    default_manifest_root = DEFAULT_MANIFEST_ROOT / "ultrasound"

    def __init__(
        self,
        max_train_examples=None,
        max_test_examples=None,
        cache_root=None,
        dataset_root=None,
        manifest_root=None,
        use_manifest=True,
    ):
        super().__init__(
            cache_root=cache_root,
            max_train_examples=max_train_examples,
            max_test_examples=max_test_examples,
        )
        self._dataset_root = Path(dataset_root) if dataset_root else DATASET_ROOT
        self.manifest_root = Path(manifest_root) if manifest_root is not None else self.default_manifest_root
        self.use_manifest = use_manifest
        if self._manifest_available():
            self.num_classes = MANIFEST_NUM_CLASSES

    def _manifest_available(self):
        return (
            self.use_manifest
            and (self.manifest_root / "mlp_train.jsonl").exists()
            and (self.manifest_root / "benchmark_eval.jsonl").exists()
        )

    def build_train_dataset(self, model, model_name, use_cache=True):
        manifest_path = self.manifest_root / "mlp_train.jsonl"
        if self.use_manifest and manifest_path.exists():
            return load_or_build_manifest_dataset(
                manifest_path=manifest_path,
                cache_prefix=f"{model_name}_ultrasound_multimediset_mlp_train",
                model=model,
                cache_root=self.cache_root or CACHE_ROOT,
                use_cache=use_cache,
                desc="ultrasound-manifest-mlp-train",
                max_examples=self.max_train_examples,
                seed=42,
            )

        ds = BodyPartsDataset(
            model=model,
            model_name=model_name,
            split="train",
            dataset_root=self._dataset_root,
            cache_root=self.cache_root or CACHE_ROOT,
            use_cache=use_cache,
        )
        if self.max_train_examples is not None and len(ds) > self.max_train_examples:
            ds.data = ds.data[:self.max_train_examples]
            ds.labels = ds.labels[:self.max_train_examples]
        return ds

    def build_test_dataset(self, model, model_name, use_cache=True):
        manifest_path = self.manifest_root / "benchmark_eval.jsonl"
        if self.use_manifest and manifest_path.exists():
            return load_or_build_manifest_dataset(
                manifest_path=manifest_path,
                cache_prefix=f"{model_name}_ultrasound_multimediset_benchmark_eval",
                model=model,
                cache_root=self.cache_root or CACHE_ROOT,
                use_cache=use_cache,
                desc="ultrasound-manifest-benchmark-eval",
                max_examples=self.max_test_examples,
                seed=43,
            )

        ds = BodyPartsDataset(
            model=model,
            model_name=model_name,
            split="test",
            dataset_root=self._dataset_root,
            cache_root=self.cache_root or CACHE_ROOT,
            use_cache=use_cache,
        )
        if self.max_test_examples is not None and len(ds) > self.max_test_examples:
            ds.data = ds.data[:self.max_test_examples]
            ds.labels = ds.labels[:self.max_test_examples]
        return ds


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python ultrasound_benchmark.py <model_path>")
    benchmark = UltrasoundBenchmark()
    print(benchmark.evaluate(sys.argv[1]))
