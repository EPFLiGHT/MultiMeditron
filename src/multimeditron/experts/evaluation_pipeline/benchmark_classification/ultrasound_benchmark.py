from __future__ import annotations

from pathlib import Path

from .base import ClassificationBenchmark
from .datasets import load_or_build_dataset, read_jsonl


class UltrasoundBenchmark(ClassificationBenchmark):
    name = "ultrasound"
    num_classes = 4

    dataset_root = Path("/lightscratch/users/deschryv/clipFineTune/ultrasound_evaluation")

    BREAST = 0
    OTHER = 1
    ABDOMEN = 2
    THYROID = 3

    train_sources = [
        ("classifier-breast-radiopedia-final_train.jsonl", BREAST),
        ("classifier-heart-radiopedia-final_train.jsonl", OTHER),
        ("classifier-lungs-radiopedia-final_train.jsonl", OTHER),
        ("classifier-abdomen-radiopedia-final_train.jsonl", ABDOMEN),
        ("classifier-thyroid-radiopedia-final_train.jsonl", THYROID),
    ]

    test_sources = [
        ("classifier-breast-radiopedia-final_test.jsonl", BREAST),
        ("classifier-heart-radiopedia-final_test.jsonl", OTHER),
        ("classifier-lungs-radiopedia-final_test.jsonl", OTHER),
        ("classifier-abdomen-radiopedia-final_test.jsonl", ABDOMEN),
        ("classifier-thyroid-radiopedia-final_test.jsonl", THYROID),
    ]

    def load_split_examples(self, split_sources: list[tuple[str, int]]) -> tuple[list[dict], list[int]]:
        all_examples = []
        all_labels = []

        for filename, label in split_sources:
            examples = read_jsonl(self.dataset_root / filename)
            all_examples.extend(examples)
            all_labels.extend([label] * len(examples))

        return all_examples, all_labels

    def build_train_dataset(self, model, model_name: str, use_cache: bool = True):
        train_examples, train_labels = self.load_split_examples(self.train_sources)

        return load_or_build_dataset(
            cache_prefix=f"{model_name}_{self.name}_train",
            examples=train_examples,
            labels=train_labels,
            model=model,
            dataset_root=self.dataset_root,
            cache_root=self.cache_root,
            use_cache=use_cache,
            desc="ultrasound-train",
        )

    def build_test_dataset(self, model, model_name: str, use_cache: bool = True):
        test_examples, test_labels = self.load_split_examples(self.test_sources)

        return load_or_build_dataset(
            cache_prefix=f"{model_name}_{self.name}_test",
            examples=test_examples,
            labels=test_labels,
            model=model,
            dataset_root=self.dataset_root,
            cache_root=self.cache_root,
            use_cache=use_cache,
            desc="ultrasound-test",
        )
