from collections import defaultdict
import json
from pathlib import Path
from random import Random

from .base import ClassificationBenchmark
from .datasets import load_or_build_dataset, read_jsonl
from .multimediset_manifest import DEFAULT_MANIFEST_ROOT, load_or_build_manifest_dataset


class CTBenchmark(ClassificationBenchmark):
    name = "ct"
    num_classes = 5

    dataset_root = Path("/lightscratch/users/nemo/datasets/CT_data/CT2D-glob")
    dataset_jsonl = Path("/lightscratch/users/tagemoua/processing-scripts/experts/CT2D-glob.jsonl")
    default_manifest_root = DEFAULT_MANIFEST_ROOT / "ct"

    labels = ["atherosoma", "Covid", "healthy", "glioblastoma", "tumor"]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    train_rate = 0.8
    split_seed = 42
    sep_by_patient = False
    max_train_examples = 50_000
    max_test_examples = 10_000
    balanced_sampling = True

    def find_label(self, example):
        text = example.get("text", "")
        if "tumor" in text:
            return "tumor"
        if "atherosoma" in text:
            return "atherosoma"
        if "glioblastoma" in text:
            return "glioblastoma"
        if "Covid" in text:
            return "Covid"
        return "healthy"

    def get_patient_id(self, example):
        return example["modalities"][0]["value"].split("_")[0].split("/")[-1]

    def __init__(
        self,
        dataset_root=None,
        dataset_jsonl=None,
        cache_root=None,
        max_train_examples=None,
        max_test_examples=None,
        balanced_sampling=None,
        manifest_root=None,
        use_manifest=True,
    ):
        super().__init__(cache_root=cache_root)
        self.dataset_root = Path(dataset_root) if dataset_root is not None else self.dataset_root
        self.dataset_jsonl = Path(dataset_jsonl) if dataset_jsonl is not None else self.dataset_jsonl
        self.max_train_examples = self.max_train_examples if max_train_examples is None else max_train_examples
        self.max_test_examples = self.max_test_examples if max_test_examples is None else max_test_examples
        self.balanced_sampling = self.balanced_sampling if balanced_sampling is None else balanced_sampling
        self.manifest_root = Path(manifest_root) if manifest_root is not None else self.default_manifest_root
        self.use_manifest = use_manifest

    def load_examples(self):
        examples = read_jsonl(self.dataset_jsonl)
        valid_labels = set(self.labels)

        filtered_examples = []
        for example in examples:
            label = self.find_label(example)
            if label in valid_labels:
                filtered_examples.append(example)

        return filtered_examples

    def _sample_examples(self, examples, max_examples, seed_offset):
        if max_examples is None or len(examples) <= max_examples:
            return examples

        rng = Random(self.split_seed + seed_offset)
        shuffled = examples.copy()
        rng.shuffle(shuffled)

        if not self.balanced_sampling:
            return shuffled[:max_examples]

        grouped_examples = defaultdict(list)
        for example in shuffled:
            grouped_examples[self.find_label(example)].append(example)

        class_names = sorted(grouped_examples)
        if not class_names:
            return []

        selected_counts = {class_name: 0 for class_name in class_names}
        selected_examples = []

        while len(selected_examples) < max_examples:
            made_progress = False
            for class_name in class_names:
                class_examples = grouped_examples[class_name]
                next_index = selected_counts[class_name]
                if next_index >= len(class_examples):
                    continue
                selected_examples.append(class_examples[next_index])
                selected_counts[class_name] += 1
                made_progress = True
                if len(selected_examples) >= max_examples:
                    break
            if not made_progress:
                break

        return selected_examples

    def split_examples(self, examples):
        rng = Random(self.split_seed)

        if self.sep_by_patient:
            patient_to_examples = {}
            for example in examples:
                patient_id = self.get_patient_id(example)
                patient_to_examples.setdefault(patient_id, []).append(example)

            patient_ids = list(patient_to_examples.keys())
            rng.shuffle(patient_ids)

            train_patient_count = int(self.train_rate * len(patient_ids))
            train_patient_ids = set(patient_ids[:train_patient_count])

            train_examples = []
            test_examples = []
            for patient_id, patient_examples in patient_to_examples.items():
                if patient_id in train_patient_ids:
                    train_examples.extend(patient_examples)
                else:
                    test_examples.extend(patient_examples)

            train_examples = self._sample_examples(train_examples, self.max_train_examples, seed_offset=101)
            test_examples = self._sample_examples(test_examples, self.max_test_examples, seed_offset=202)
            return train_examples, test_examples

        shuffled = examples.copy()
        rng.shuffle(shuffled)

        train_size = int(self.train_rate * len(shuffled))
        train_examples = shuffled[:train_size]
        test_examples = shuffled[train_size:]
        train_examples = self._sample_examples(train_examples, self.max_train_examples, seed_offset=101)
        test_examples = self._sample_examples(test_examples, self.max_test_examples, seed_offset=202)
        return train_examples, test_examples

    def examples_to_labels(self, examples):
        return [self.label_to_idx[self.find_label(example)] for example in examples]

    def build_train_dataset(self, model, model_name, use_cache=True):
        manifest_path = self.manifest_root / "mlp_train.jsonl"
        if self.use_manifest and manifest_path.exists():
            return load_or_build_manifest_dataset(
                manifest_path=manifest_path,
                cache_prefix=f"{model_name}_{self.name}_multimediset_mlp_train",
                model=model,
                cache_root=self.cache_root,
                use_cache=use_cache,
                desc="ct-manifest-mlp-train",
                max_examples=self.max_train_examples,
                seed=self.split_seed + 101,
            )

        examples = self.load_examples()
        train_examples, _ = self.split_examples(examples)
        train_labels = self.examples_to_labels(train_examples)

        return load_or_build_dataset(
            cache_prefix=f"{model_name}_{self.name}_train",
            examples=train_examples,
            labels=train_labels,
            model=model,
            dataset_root=self.dataset_root,
            cache_root=self.cache_root,
            use_cache=use_cache,
            desc="ct-train",
        )

    def build_test_dataset(self, model, model_name, use_cache=True):
        manifest_path = self.manifest_root / "benchmark_eval.jsonl"
        if self.use_manifest and manifest_path.exists():
            return load_or_build_manifest_dataset(
                manifest_path=manifest_path,
                cache_prefix=f"{model_name}_{self.name}_multimediset_benchmark_eval",
                model=model,
                cache_root=self.cache_root,
                use_cache=use_cache,
                desc="ct-manifest-benchmark-eval",
                max_examples=self.max_test_examples,
                seed=self.split_seed + 202,
            )

        examples = self.load_examples()
        _, test_examples = self.split_examples(examples)
        test_labels = self.examples_to_labels(test_examples)

        return load_or_build_dataset(
            cache_prefix=f"{model_name}_{self.name}_test",
            examples=test_examples,
            labels=test_labels,
            model=model,
            dataset_root=self.dataset_root,
            cache_root=self.cache_root,
            use_cache=use_cache,
            desc="ct-test",
        )
