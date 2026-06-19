import json
from pathlib import Path

from .base import ClassificationBenchmark
from .datasets import load_or_build_dataset
from load_from_clip import encode_img


class MRIBenchmark(ClassificationBenchmark):
    """Brain tumor MRI classification benchmark (4 classes).

    Source: Brain Tumor MRI Dataset (Masoud Nickparvar)
    https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

    Uses the pre-defined train/test split from the original dataset
    (5712 train / 1311 test). No random split is applied.

    Preprocessing: run
        python scripts/dataset_processing/mri_expert/process_brain_tumor.py
            --output_dir /lightscratch/users/cljordan/datasets/brain_tumor_mri
    """

    name = "mri"
    num_classes = 4

    labels = ["glioma", "meningioma", "no_tumor", "pituitary"]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    default_train_jsonl = Path(
        "/lightscratch/users/cljordan/datasets/brain_tumor_mri/brain_tumor_train.jsonl"
    )
    default_test_jsonl = Path(
        "/lightscratch/users/cljordan/datasets/brain_tumor_mri/brain_tumor_test.jsonl"
    )

    def __init__(
        self,
        train_jsonl=None,
        test_jsonl=None,
        cache_root=None,
        max_train_examples=None,
        max_test_examples=None,
    ):
        super().__init__(
            cache_root=cache_root,
            max_train_examples=max_train_examples,
            max_test_examples=max_test_examples,
        )
        self.train_jsonl = (
            Path(train_jsonl) if train_jsonl is not None else self.default_train_jsonl
        )
        self.test_jsonl = (
            Path(test_jsonl) if test_jsonl is not None else self.default_test_jsonl
        )

    def _read_jsonl(self, path, split_name):
        examples = []
        dropped_label = 0
        dropped_image = 0

        with path.open("r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                label = record.get("label") or record.get("text", "").strip()
                if label not in self.label_to_idx:
                    dropped_label += 1
                    continue
                image_path = Path(record["modalities"][0]["value"])
                if not image_path.exists():
                    dropped_image += 1
                    continue
                record["label"] = label
                examples.append(record)

        print(
            f"[{split_name}] kept {len(examples)}, "
            f"dropped {dropped_label} unknown-label, {dropped_image} missing-image"
        )
        return examples

    def examples_to_labels(self, examples):
        return [self.label_to_idx[ex["label"]] for ex in examples]

    def embed_example(self, example, _label, model, _dataset_root):
        return encode_img(model, str(example["modalities"][0]["value"]))

    def build_train_dataset(self, model, model_name, use_cache=True):
        examples = self._read_jsonl(self.train_jsonl, "mri-train")
        examples = self._sample_examples_random(
            examples, self.max_train_examples, seed=42
        )
        labels = self.examples_to_labels(examples)
        return load_or_build_dataset(
            cache_prefix=f"{model_name}_{self.name}_train",
            examples=examples,
            labels=labels,
            model=model,
            dataset_root=self.train_jsonl.parent,
            cache_root=self.cache_root,
            use_cache=use_cache,
            desc="mri-train",
            embed_example=self.embed_example,
        )

    def build_test_dataset(self, model, model_name, use_cache=True):
        examples = self._read_jsonl(self.test_jsonl, "mri-test")
        examples = self._sample_examples_random(
            examples, self.max_test_examples, seed=43
        )
        labels = self.examples_to_labels(examples)
        return load_or_build_dataset(
            cache_prefix=f"{model_name}_{self.name}_test",
            examples=examples,
            labels=labels,
            model=model,
            dataset_root=self.test_jsonl.parent,
            cache_root=self.cache_root,
            use_cache=use_cache,
            desc="mri-test",
            embed_example=self.embed_example,
        )
