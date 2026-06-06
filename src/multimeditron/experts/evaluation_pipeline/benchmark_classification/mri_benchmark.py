from collections import defaultdict
from pathlib import Path
from random import Random

from .base import ClassificationBenchmark
from .datasets import load_or_build_dataset, read_jsonl, resolve_image_path
from .multimediset_manifest import DEFAULT_MANIFEST_ROOT, load_or_build_manifest_dataset
from load_from_clip import encode_img, encode_img_bytes


class MRIBenchmark(ClassificationBenchmark):
    """MRI classification benchmark across 4 classes: brain tumor, crohn, healthy, bone infection.

    Images are resolved with a multi-path fallback strategy to handle the fragmented
    layout of the MRI-glob dataset. When no image file is found on disk, bytes are
    loaded directly from the parquet shards as a last resort.
    """

    name = "mri"
    num_classes = 4

    default_dataset_root = Path("/lightscratch/users/nemo/datasets/MRI_data/MRI-glob")
    default_dataset_jsonl = Path("/lightscratch/users/nemo/datasets/MRI_data/MRI-glob/MRI-glob.jsonl")
    default_parquet_paths = (
        Path("/lightscratch/users/nemo/datasets/MRI_data/MRI-glob/ignoreme/images.parquet"),
        Path("/lightscratch/users/nemo/datasets/MRI_data/MRI-glob/ignoreme/images-1.parquet"),
        Path("/lightscratch/users/nemo/datasets/MRI_data/MRI-glob/ignoreme/images-2.parquet"),
    )
    default_manifest_root = DEFAULT_MANIFEST_ROOT / "mri"

    labels = ["brain tumor", "crohn", "healthy", "Bone infection"]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    train_rate = 0.8
    split_seed = 42
    subset_fraction = 0.5
    sep_by_patient = False
    max_train_examples = 50_000
    max_test_examples = 10_000
    manifest_max_train_examples = 5_000
    manifest_max_test_examples = 3_000
    balanced_sampling = True

    def __init__(
        self,
        dataset_root=None,
        dataset_jsonl=None,
        cache_root=None,
        parquet_paths=None,
        subset_fraction=None,
        max_train_examples=None,
        max_test_examples=None,
        balanced_sampling=None,
        manifest_root=None,
        use_manifest=True,
    ):
        super().__init__(cache_root=cache_root)
        self.dataset_root = Path(dataset_root) if dataset_root is not None else self.default_dataset_root
        self.dataset_jsonl = Path(dataset_jsonl) if dataset_jsonl is not None else self.default_dataset_jsonl
        chosen_parquet_paths = parquet_paths if parquet_paths is not None else self.default_parquet_paths
        self.parquet_paths = tuple(Path(path) for path in chosen_parquet_paths)
        self.subset_fraction = self.subset_fraction if subset_fraction is None else subset_fraction
        self.balanced_sampling = self.balanced_sampling if balanced_sampling is None else balanced_sampling
        self.manifest_root = Path(manifest_root) if manifest_root is not None else self.default_manifest_root
        self.use_manifest = use_manifest
        manifest_available = (
            self.use_manifest
            and (self.manifest_root / "mlp_train.jsonl").exists()
            and (self.manifest_root / "benchmark_eval.jsonl").exists()
        )
        default_max_train_examples = (
            self.manifest_max_train_examples if manifest_available else self.max_train_examples
        )
        default_max_test_examples = (
            self.manifest_max_test_examples if manifest_available else self.max_test_examples
        )
        self.max_train_examples = (
            default_max_train_examples if max_train_examples is None else max_train_examples
        )
        self.max_test_examples = (
            default_max_test_examples if max_test_examples is None else max_test_examples
        )
        self.source_image_roots = (
            self.dataset_root / "images",
            self.dataset_root / "ignoreme" / "images",
            self.dataset_root / "ignoreme" / "images_small",
            self.dataset_root / "ignoreme" / "images_container",
            self.dataset_root / "old" / "images",
            self.dataset_root / "old" / "images-old",
        )
        self._prepared_image_bytes = {}

    def find_label(self, example):
        text = example.get("text", "")
        text_lower = text.lower()

        if "brain tumor" in text_lower:
            return "brain tumor"
        if "crohn" in text_lower:
            return "crohn"
        if "bone infection" in text_lower:
            return "Bone infection"
        return "healthy"

    def get_patient_id(self, example):
        return example["modalities"][0]["value"].split("_")[0].split("/")[-1]

    def load_examples(self):
        examples = read_jsonl(self.dataset_jsonl)
        valid_examples = []

        for example in examples:
            text = example.get("text", "").lower()
            if (
                "healthy" in text
                or "brain tumor" in text
                or "crohn" in text
                or "bone infection" in text
            ):
                valid_examples.append(example)

        return valid_examples

    def subset_examples(self, examples):
        if self.subset_fraction is None or self.subset_fraction >= 1.0:
            return examples

        rng = Random(self.split_seed)
        shuffled = examples.copy()
        rng.shuffle(shuffled)

        subset_size = int(self.subset_fraction * len(shuffled))
        return shuffled[:subset_size]

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

        # Round-robin over classes to keep the reduced benchmark roughly balanced.
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
        examples = self.subset_examples(examples)

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

    def _is_usable_image_path(self, image_path):
        return image_path.is_file() and image_path.stat().st_size > 0

    def _find_image_path(self, image_value, dataset_root):
        image_path = resolve_image_path(image_value, dataset_root)
        if self._is_usable_image_path(image_path):
            return image_path
        if image_path.exists():
            print(f"[mri] skipping empty image file: {image_path}")

        file_name = Path(image_value).name
        for root in self.source_image_roots:
            candidate = root / file_name
            if self._is_usable_image_path(candidate):
                return candidate
            if candidate.exists():
                print(f"[mri] skipping empty image file: {candidate}")

        return None

    def prepare_images(self, examples, dataset_root, desc):
        missing_targets = {}
        self._prepared_image_bytes = {}

        for example in examples:
            image_value = example["modalities"][0]["value"]
            image_path = self._find_image_path(image_value, dataset_root)
            if image_path is not None:
                continue
            missing_targets.setdefault(Path(image_value).name, resolve_image_path(image_value, dataset_root))

        if not missing_targets:
            return

        print(f"[{desc}] loading {len(missing_targets)} missing image(s) directly from MRI parquet files")
        unresolved = self._restore_missing_images(missing_targets)
        if unresolved:
            print(f"[{desc}] could not find {len(unresolved)} image(s) in MRI parquet files")

    def _restore_missing_images(self, missing_targets):
        import pyarrow.parquet as pq

        remaining = dict(missing_targets)

        for parquet_path in self.parquet_paths:
            if not parquet_path.exists() or not remaining:
                continue

            parquet_file = pq.ParquetFile(parquet_path)
            for batch in parquet_file.iter_batches(columns=["path", "bytes"], batch_size=4096):
                for row in batch.to_pylist():
                    file_name = Path(row["path"]).name
                    if file_name not in remaining:
                        continue

                    self._prepared_image_bytes[file_name] = row["bytes"]
                    remaining.pop(file_name, None)

                if not remaining:
                    break

        return remaining

    def embed_example(self, example, _label, model, dataset_root):
        image_value = example["modalities"][0]["value"]
        image_path = self._find_image_path(image_value, dataset_root)
        if image_path is not None:
            return encode_img(model, str(image_path))

        image_bytes = self._prepared_image_bytes.get(Path(image_value).name)
        if image_bytes is None:
            print(f"[mri] no usable image found for {image_value}")
            return None

        return encode_img_bytes(model, image_bytes)

    def build_train_dataset(self, model, model_name, use_cache=True):
        manifest_path = self.manifest_root / "mlp_train.jsonl"
        if self.use_manifest and manifest_path.exists():
            return load_or_build_manifest_dataset(
                manifest_path=manifest_path,
                cache_prefix=f"{model_name}_{self.name}_multimediset_mlp_train",
                model=model,
                cache_root=self.cache_root,
                use_cache=use_cache,
                desc="mri-manifest-mlp-train",
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
            desc="mri-train",
            prepare_images=self.prepare_images,
            embed_example=self.embed_example,
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
                desc="mri-manifest-benchmark-eval",
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
            desc="mri-test",
            prepare_images=self.prepare_images,
            embed_example=self.embed_example,
        )
