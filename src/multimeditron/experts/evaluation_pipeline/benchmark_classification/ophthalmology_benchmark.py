import json
from pathlib import Path

from .base import ClassificationBenchmark
from .datasets import load_or_build_dataset
from .multimediset_manifest import DEFAULT_MANIFEST_ROOT, load_or_build_manifest_dataset
from load_from_clip import encode_img


class OphthalmologyBenchmark(ClassificationBenchmark):
    """Binary diabetic retinopathy benchmark built from fundus datasets.

    Distinguishes normal fundus images from any level of diabetic retinopathy.
    Default sources are EyePACS and Messidor-2. Accepts either a single
    dataset root + jsonl pair (used by train_multidomain_clip via build_benchmarks)
    or explicit tuples of multiple sources.
    """

    name = "ophthalmology"
    num_classes = 2
    labels = ['normal', 'diabetic_retinopathy']
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    default_manifest_root = DEFAULT_MANIFEST_ROOT / "eye"

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

    def __init__(
        self,
        dataset_root=None,
        train_jsonl=None,
        test_jsonl=None,
        cache_root=None,
        max_train_examples=None,
        max_test_examples=None,
        manifest_root=None,
        use_manifest=True,
    ):
        super().__init__(cache_root=cache_root)
        if train_jsonl is not None:
            self.train_jsonls = (Path(train_jsonl),)
            self.test_jsonls = (Path(test_jsonl),)
            self.image_roots = (Path(dataset_root),)
        else:
            self.train_jsonls = self.default_train_jsonls
            self.test_jsonls = self.default_test_jsonls
            self.image_roots = self.default_image_roots
        self.dataset_root = self.image_roots[0]
        self.max_train_examples = max_train_examples
        self.max_test_examples = max_test_examples
        self.manifest_root = Path(manifest_root) if manifest_root is not None else self.default_manifest_root
        self.use_manifest = use_manifest

    def _read_examples(self, jsonl_paths, split_name):
        examples = []
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

    def resolve_example_image_path(self, example, source_root):
        image_value = example['modalities'][0]['value']
        image_path = Path(image_value)

        if image_path.is_absolute() and image_path.exists():
            return image_path

        candidates = [source_root / image_value]
        for root in self.image_roots:
            candidates.append(root / image_value)
            candidates.append(root / Path(image_value).name)

        seen = set()
        for candidate in candidates:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate.exists():
                return candidate

        return None

    def find_label(self, example):
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
        if 'diabetic retinopathy' in text:
            return 'diabetic_retinopathy'

        return None

    def load_train_examples(self):
        examples = self._read_examples(self.train_jsonls, 'ophthalmology-train')
        return self._sample_examples_random(examples, self.max_train_examples, seed=42)

    def load_test_examples(self):
        examples = self._read_examples(self.test_jsonls, 'ophthalmology-test')
        return self._sample_examples_random(examples, self.max_test_examples, seed=43)

    def examples_to_labels(self, examples):
        return [self.label_to_idx[str(example['label'])] for example in examples]

    def embed_example(self, example, _label, model, _dataset_root):
        image_path = example.get('__image_path__')
        if image_path is None:
            return None
        return encode_img(model, str(image_path))

    def build_train_dataset(self, model, model_name, use_cache=True):
        manifest_path = self.manifest_root / 'mlp_train.jsonl'
        if self.use_manifest and manifest_path.exists():
            return load_or_build_manifest_dataset(
                manifest_path=manifest_path,
                cache_prefix=f'{model_name}_{self.name}_multimediset_mlp_train',
                model=model,
                cache_root=self.cache_root,
                use_cache=use_cache,
                desc='ophthalmology-manifest-mlp-train',
                max_examples=self.max_train_examples,
                seed=42,
            )

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
            desc='ophthalmology-train',
            embed_example=self.embed_example,
        )

    def build_test_dataset(self, model, model_name, use_cache=True):
        manifest_path = self.manifest_root / 'benchmark_eval.jsonl'
        if self.use_manifest and manifest_path.exists():
            return load_or_build_manifest_dataset(
                manifest_path=manifest_path,
                cache_prefix=f'{model_name}_{self.name}_multimediset_benchmark_eval',
                model=model,
                cache_root=self.cache_root,
                use_cache=use_cache,
                desc='ophthalmology-manifest-benchmark-eval',
                max_examples=self.max_test_examples,
                seed=43,
            )

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
            desc='ophthalmology-test',
            embed_example=self.embed_example,
        )
