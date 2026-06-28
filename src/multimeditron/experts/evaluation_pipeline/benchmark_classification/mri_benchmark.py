import os
from pathlib import Path

from .base import ClassificationBenchmark
from .datasets import load_or_build_dataset
from ..load_from_clip import encode_img


class MRIBenchmark(ClassificationBenchmark):
    """Brain tumor MRI classification benchmark (4 classes).

    Source: Brain Tumor MRI Dataset (Masoud Nickparvar)
    https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

    Reads directly from the dataset's folder structure — no preprocessing step needed.
    Set MRI_DATASET_ROOT to the images/ folder (which must contain train/ and test/
    subdirectories, each with one subfolder per class), or pass dataset_root explicitly.

    Expected layout:
        <root>/
            train/
                glioma/      meningioma/      no_tumor/      pituitary/
            test/
                glioma/      meningioma/      no_tumor/      pituitary/
    """

    name = "mri"
    num_classes = 4

    labels = ["glioma", "meningioma", "no_tumor", "pituitary"]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    _IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

    def __init__(
        self,
        dataset_root=None,
        cache_root=None,
        max_train_examples=None,
        max_test_examples=None,
    ):
        super().__init__(
            cache_root=cache_root,
            max_train_examples=max_train_examples,
            max_test_examples=max_test_examples,
        )
        env_root = os.environ.get("MRI_DATASET_ROOT")
        if dataset_root is not None:
            self.dataset_root = Path(dataset_root)
        elif env_root is not None:
            self.dataset_root = Path(env_root)
        else:
            raise ValueError(
                "MRIBenchmark requires a dataset root. "
                "Set the MRI_DATASET_ROOT environment variable or pass dataset_root explicitly.\n"
                "Expected layout: <root>/train/{glioma,meningioma,no_tumor,pituitary}/*.jpg"
            )

    def _scan_split(self, split_name):
        """Return (examples, int_labels) by scanning <dataset_root>/<split_name>/<label>/."""
        split_dir = self.dataset_root / split_name
        examples, int_labels = [], []
        for label in self.labels:
            class_dir = split_dir / label
            if not class_dir.is_dir():
                continue
            for img_path in sorted(class_dir.glob("*")):
                if img_path.suffix.lower() in self._IMAGE_SUFFIXES:
                    examples.append({"image_path": str(img_path), "label_idx": self.label_to_idx[label]})
                    int_labels.append(self.label_to_idx[label])
        if not examples:
            raise FileNotFoundError(
                f"No images found under {split_dir}. "
                "Check that MRI_DATASET_ROOT points to the folder containing train/ and test/."
            )
        return examples, int_labels

    def examples_to_labels(self, examples):
        return [ex["label_idx"] for ex in examples]

    def embed_example(self, example, _label, model, _dataset_root):
        return encode_img(model, example["image_path"])

    def build_train_dataset(self, model, model_name, use_cache=True):
        examples, labels = self._scan_split("train")
        examples = self._sample_examples_random(examples, self.max_train_examples, seed=42)
        labels = self.examples_to_labels(examples)
        return load_or_build_dataset(
            cache_prefix=f"{model_name}_{self.name}_train",
            examples=examples,
            labels=labels,
            model=model,
            dataset_root=self.dataset_root,
            cache_root=self.cache_root,
            use_cache=use_cache,
            desc="mri-train",
            embed_example=self.embed_example,
        )

    def build_test_dataset(self, model, model_name, use_cache=True):
        examples, labels = self._scan_split("test")
        examples = self._sample_examples_random(examples, self.max_test_examples, seed=43)
        labels = self.examples_to_labels(examples)
        return load_or_build_dataset(
            cache_prefix=f"{model_name}_{self.name}_test",
            examples=examples,
            labels=labels,
            model=model,
            dataset_root=self.dataset_root,
            cache_root=self.cache_root,
            use_cache=use_cache,
            desc="mri-test",
            embed_example=self.embed_example,
        )
