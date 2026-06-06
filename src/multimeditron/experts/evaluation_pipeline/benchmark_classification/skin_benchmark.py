import json
from pathlib import Path

from .base import ClassificationBenchmark
from .datasets import load_or_build_dataset
from .multimediset_manifest import DEFAULT_MANIFEST_ROOT, load_or_build_manifest_dataset
from load_from_clip import encode_img


class SkinBenchmark(ClassificationBenchmark):
    """Integrated skin disease benchmark built from coherent source manifests.

    This benchmark avoids the broken aggregate SKIN_data manifests whose `text`
    field contains free-form descriptions. Instead it combines Skin10 and ISIC
    splits that still resolve images cleanly and can be mapped to a stable class
    taxonomy compatible with the current single-label evaluation pipeline.
    """

    name = 'skin'
    default_manifest_root = DEFAULT_MANIFEST_ROOT / 'skin'

    default_train_jsonls = (
        Path('/lightscratch/users/turan/datasets/skin_expert_datasets/skin_diseases_10/skin10_train.jsonl'),
        Path('/lightscratch/users/turan/datasets/skin_expert_datasets/isic/isic_train.jsonl'),
    )
    default_test_jsonls = (
        Path('/lightscratch/users/turan/datasets/skin_expert_datasets/skin_diseases_10/skin10_val.jsonl'),
        Path('/lightscratch/users/turan/datasets/skin_expert_datasets/isic/isic_val.jsonl'),
    )
    default_image_roots = (
        Path('/lightscratch/users/turan/datasets/skin_expert_datasets/skin_diseases_10'),
        Path('/lightscratch/users/turan/datasets/skin_expert_datasets/isic'),
    )

    labels = [
        'atopic-dermatitis',
        'basal-cell-carcinoma',
        'benign-keratosis-like-lesions',
        'eczema',
        'melanocytic-nevi',
        'melanoma',
        'psoriasis-pictures-lichen-planus-and-related-diseases',
        'seborrheic-keratoses-and-other-benign-tumors',
        'tinea-ringworm-candidiasis-and-other-fungal-infections',
        'warts-molluscum-and-other-viral-infections',
    ]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    num_classes = len(labels)

    def __init__(
        self,
        train_jsonls=None,
        test_jsonls=None,
        image_roots=None,
        cache_root=None,
        max_train_examples=None,
        max_test_examples=None,
        manifest_root=None,
        use_manifest=True,
    ):
        super().__init__(
            cache_root=cache_root,
            max_train_examples=max_train_examples,
            max_test_examples=max_test_examples,
        )
        chosen_train = train_jsonls if train_jsonls is not None else self.default_train_jsonls
        chosen_test = test_jsonls if test_jsonls is not None else self.default_test_jsonls
        chosen_roots = image_roots if image_roots is not None else self.default_image_roots
        self.train_jsonls = tuple(Path(path) for path in chosen_train)
        self.test_jsonls = tuple(Path(path) for path in chosen_test)
        self.image_roots = tuple(Path(path) for path in chosen_roots)
        self.dataset_root = self.image_roots[0] if self.image_roots else Path('/')
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
                    label = self.find_label(example, jsonl_path)
                    if label is None:
                        dropped_unlabeled += 1
                        continue

                    image_path = self.resolve_example_image_path(example, jsonl_path.parent)
                    if image_path is None:
                        dropped_missing_images += 1
                        continue

                    normalized = dict(example)
                    normalized['label'] = label
                    normalized['__image_path__'] = str(image_path)
                    normalized['__source_jsonl__'] = str(jsonl_path)
                    examples.append(normalized)

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

    def _extract_skin10_label(self, image_value):
        parts = Path(image_value).parts
        if len(parts) >= 2 and parts[0] == 'rebuilt':
            label = parts[1]
            if label in self.label_to_idx:
                return label
        return None

    def _extract_isic_label(self, text):
        text = text.lower()
        if 'basal cell carcinoma' in text:
            return 'basal-cell-carcinoma'
        if 'benign keratosis' in text:
            return 'benign-keratosis-like-lesions'
        if 'melanocytic nevus' in text or 'melanocytic nevi' in text:
            return 'melanocytic-nevi'
        if 'melanoma' in text:
            return 'melanoma'
        return None

    def find_label(self, example, source_jsonl):
        image_value = example['modalities'][0]['value']
        if 'skin_diseases_10' in str(source_jsonl):
            return self._extract_skin10_label(image_value)
        if 'isic' in str(source_jsonl):
            return self._extract_isic_label(str(example.get('text', '')))
        return None

    def load_train_examples(self):
        return self._read_examples(self.train_jsonls, 'skin-integrated-train')

    def load_test_examples(self):
        return self._read_examples(self.test_jsonls, 'skin-integrated-test')

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
                desc='skin-manifest-mlp-train',
                max_examples=self.max_train_examples,
                seed=42,
            )

        train_examples = self.load_train_examples()
        train_examples = self._sample_examples_random(train_examples, self.max_train_examples, seed=42)
        train_labels = self.examples_to_labels(train_examples)
        return load_or_build_dataset(
            cache_prefix=f'{model_name}_{self.name}_train',
            examples=train_examples,
            labels=train_labels,
            model=model,
            dataset_root=self.dataset_root,
            cache_root=self.cache_root,
            use_cache=use_cache,
            desc='skin-integrated-train',
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
                desc='skin-manifest-benchmark-eval',
                max_examples=self.max_test_examples,
                seed=43,
            )

        test_examples = self.load_test_examples()
        test_examples = self._sample_examples_random(test_examples, self.max_test_examples, seed=43)
        test_labels = self.examples_to_labels(test_examples)
        return load_or_build_dataset(
            cache_prefix=f'{model_name}_{self.name}_test',
            examples=test_examples,
            labels=test_labels,
            model=model,
            dataset_root=self.dataset_root,
            cache_root=self.cache_root,
            use_cache=use_cache,
            desc='skin-integrated-test',
            embed_example=self.embed_example,
        )
