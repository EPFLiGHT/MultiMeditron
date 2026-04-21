from __future__ import annotations

import json
from pathlib import Path

from .base import ClassificationBenchmark
from .datasets import load_or_build_dataset
from load_from_clip import encode_img


class OphthalmologyDRBenchmark(ClassificationBenchmark):
    """Binary diabetic retinopathy benchmark built from coherent fundus datasets.

    This benchmark intentionally focuses on a single clinically meaningful task:
    distinguishing normal fundus images from any level of diabetic retinopathy.
    The current default sources are EyePACS and Messidor-2 because both are
    centered on diabetic retinopathy grading and ship with manifests whose image
    paths resolve cleanly on the shared storage.
    """

    name = "ophthalmology_dr"
    num_classes = 2

    default_train_jsonls = (
        Path('/lightscratch/users/turan/datasets/opthalmology_expert_datasets/eyepacs/eyepacs_train.jsonl'),
        Path('/lightscratch/users/turan/datasets/opthalmology_expert_datasets/messidor2_eval/messidor_train.jsonl'),
    )
    default_test_jsonls = (
        Path('/lightscratch/users/turan/datasets/opthalmology_expert_datasets/eyepacs/eyepacs_val.jsonl'),
        Path('/lightscratch/users/turan/datasets/opthalmology_expert_datasets/messidor2_eval/messidor_val.jsonl'),
    )
    default_image_roots = (
        Path('/lightscratch/users/turan/datasets/opthalmology_expert_datasets/eyepacs'),
        Path('/lightscratch/users/turan/datasets/opthalmology_expert_datasets/messidor2_eval'),
    )

    labels = ['normal', 'diabetic_retinopathy']
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    def __init__(
        self,
        train_jsonls: tuple[str | Path, ...] | None = None,
        test_jsonls: tuple[str | Path, ...] | None = None,
        image_roots: tuple[str | Path, ...] | None = None,
        cache_root: Path | None = None,
    ) -> None:
        super().__init__(cache_root=cache_root)
        chosen_train = train_jsonls if train_jsonls is not None else self.default_train_jsonls
        chosen_test = test_jsonls if test_jsonls is not None else self.default_test_jsonls
        chosen_roots = image_roots if image_roots is not None else self.default_image_roots
        self.train_jsonls = tuple(Path(path) for path in chosen_train)
        self.test_jsonls = tuple(Path(path) for path in chosen_test)
        self.image_roots = tuple(Path(path) for path in chosen_roots)
        self.dataset_root = self.image_roots[0] if self.image_roots else Path('/')

    def _read_examples(self, jsonl_paths: tuple[Path, ...], split_name: str) -> list[dict]:
        examples: list[dict] = []
        dropped_unlabeled = 0
        dropped_missing_images = 0

        for jsonl_path in jsonl_paths:
            with jsonl_path.open('r', encoding='utf-8') as f:
                for line in f:
                    example = json.loads(line)
                    image_path = self.resolve_example_image_path(example, jsonl_path.parent)
                    if image_path is None:
                        dropped_missing_images += 1
                        continue

                    label = self.find_label(example)
                    if label is None:
                        dropped_unlabeled += 1
                        continue

                    example = dict(example)
                    example['label'] = label
                    example['__image_path__'] = str(image_path)
                    example['__source_jsonl__'] = str(jsonl_path)
                    examples.append(example)

        print(
            f'[{split_name}] kept {len(examples)} example(s), '
            f'dropped {dropped_unlabeled} unlabeled and {dropped_missing_images} missing-image example(s)'
        )
        return examples

    def resolve_example_image_path(self, example: dict, source_root: Path) -> Path | None:
        image_value = example['modalities'][0]['value']
        image_path = Path(image_value)

        if image_path.is_absolute() and image_path.exists():
            return image_path

        candidates = [source_root / image_value]
        for root in self.image_roots:
            candidates.append(root / image_value)
            candidates.append(root / Path(image_value).name)

        seen: set[Path] = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                return candidate

        return None

    def find_label(self, example: dict) -> str | None:
        raw_label = example.get('label')
        if raw_label:
            normalized = str(raw_label).strip().lower()
            if normalized in {'normal', 'healthy', 'no diabetic retinopathy', 'no_dr', 'no_diabetic_retinopathy'}:
                return 'normal'
            if normalized in {
                'diabetic retinopathy',
                'mild diabetic retinopathy',
                'moderate diabetic retinopathy',
                'severe diabetic retinopathy',
                'proliferative diabetic retinopathy',
            }:
                return 'diabetic_retinopathy'

        text = str(example.get('text', '')).lower()
        if 'no diabetic retinopathy' in text:
            return 'normal'
        if 'normal fundus' in text or 'healthy fundus' in text or 'healthy retina' in text:
            return 'normal'
        if 'mild diabetic retinopathy' in text:
            return 'diabetic_retinopathy'
        if 'moderate diabetic retinopathy' in text:
            return 'diabetic_retinopathy'
        if 'severe diabetic retinopathy' in text:
            return 'diabetic_retinopathy'
        if 'proliferative diabetic retinopathy' in text:
            return 'diabetic_retinopathy'
        if 'diabetic retinopathy' in text:
            return 'diabetic_retinopathy'

        return None

    def load_train_examples(self) -> list[dict]:
        return self._read_examples(self.train_jsonls, 'ophthalmology-dr-train')

    def load_test_examples(self) -> list[dict]:
        return self._read_examples(self.test_jsonls, 'ophthalmology-dr-test')

    def examples_to_labels(self, examples: list[dict]) -> list[int]:
        return [self.label_to_idx[str(example['label'])] for example in examples]

    def embed_example(self, example: dict, _label: int, model, _dataset_root: Path):
        image_path = example.get('__image_path__')
        if image_path is None:
            return None
        return encode_img(model, str(image_path))

    def build_train_dataset(self, model, model_name: str, use_cache: bool = True):
        train_examples = self.load_train_examples()
        train_labels = self.examples_to_labels(train_examples)

        return load_or_build_dataset(
            cache_prefix=f'{model_name}_{self.name}_train',
            examples=train_examples,
            labels=train_labels,
            model=model,
            dataset_root=self.dataset_root,
            cache_root=self.cache_root,
            use_cache=use_cache,
            desc='ophthalmology-dr-train',
            embed_example=self.embed_example,
        )

    def build_test_dataset(self, model, model_name: str, use_cache: bool = True):
        test_examples = self.load_test_examples()
        test_labels = self.examples_to_labels(test_examples)

        return load_or_build_dataset(
            cache_prefix=f'{model_name}_{self.name}_test',
            examples=test_examples,
            labels=test_labels,
            model=model,
            dataset_root=self.dataset_root,
            cache_root=self.cache_root,
            use_cache=use_cache,
            desc='ophthalmology-dr-test',
            embed_example=self.embed_example,
        )
