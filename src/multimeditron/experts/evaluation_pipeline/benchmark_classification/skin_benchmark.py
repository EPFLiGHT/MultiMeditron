from pathlib import Path

from .base import ClassificationBenchmark
from .multimediset_manifest import DEFAULT_MANIFEST_ROOT, load_or_build_manifest_dataset


CLEAN_SUBDATASETS = {"skin_diseases_10", "isic"}


class SkinBenchmark(ClassificationBenchmark):
    """Skin disease classification benchmark via multimediset manifest.

    Uses only skin_diseases_10 and isic records (reliable folder-structure labels).
    DermNet, Fitzpatrick and SCIN are excluded from the benchmark splits due to
    noisy GPT-text-based label assignment; use SCINBenchmark for fairness evaluation.
    """

    name = "skin"
    default_manifest_root = DEFAULT_MANIFEST_ROOT / "skin"
    labels = [
        "atopic-dermatitis",
        "basal-cell-carcinoma",
        "benign-keratosis-like-lesions",
        "eczema",
        "melanocytic-nevi",
        "melanoma",
        "psoriasis",
        "seborrheic-keratoses",
        "tinea-ringworm-candidiasis",
        "warts-molluscum-viral",
    ]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    num_classes = len(labels)

    def __init__(
        self,
        cache_root=None,
        max_train_examples=None,
        max_test_examples=None,
        manifest_root=None,
    ):
        super().__init__(
            cache_root=cache_root,
            max_train_examples=max_train_examples,
            max_test_examples=max_test_examples,
        )
        self.manifest_root = (
            Path(manifest_root) if manifest_root is not None else self.default_manifest_root
        )

    def build_train_dataset(self, model, model_name, use_cache=True):
        return load_or_build_manifest_dataset(
            manifest_path=self.manifest_root / "mlp_train.jsonl",
            cache_prefix=f"{model_name}_{self.name}_clean_mlp_train",
            model=model,
            cache_root=self.cache_root,
            use_cache=use_cache,
            desc="skin-manifest-mlp-train",
            max_examples=self.max_train_examples,
            seed=42,
            allowed_subdatasets=CLEAN_SUBDATASETS,
        )

    def build_test_dataset(self, model, model_name, use_cache=True):
        return load_or_build_manifest_dataset(
            manifest_path=self.manifest_root / "benchmark_eval.jsonl",
            cache_prefix=f"{model_name}_{self.name}_clean_benchmark_eval",
            model=model,
            cache_root=self.cache_root,
            use_cache=use_cache,
            desc="skin-manifest-benchmark-eval",
            max_examples=self.max_test_examples,
            seed=43,
            allowed_subdatasets=CLEAN_SUBDATASETS,
        )
