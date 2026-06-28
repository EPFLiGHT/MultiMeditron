from pathlib import Path

from .base import ClassificationBenchmark
from .multimediset_manifest import DEFAULT_MANIFEST_ROOT, load_or_build_manifest_dataset


class CTBenchmark(ClassificationBenchmark):
    name = "ct"
    num_classes = 2

    default_manifest_root = DEFAULT_MANIFEST_ROOT / "ct"

    labels = ["covid-19 infection", "right lung"]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    max_train_examples = 5_000
    max_test_examples = 1_000

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
            Path(manifest_root)
            if manifest_root is not None
            else self.default_manifest_root
        )

    @classmethod
    def is_available(cls, manifest_root=None):
        root = (
            Path(manifest_root)
            if manifest_root is not None
            else cls.default_manifest_root
        )
        return (root / "mlp_train.jsonl").exists() and (
            root / "benchmark_eval.jsonl"
        ).exists()

    def build_train_dataset(self, model, model_name, use_cache=True):
        return load_or_build_manifest_dataset(
            manifest_path=self.manifest_root / "mlp_train.jsonl",
            cache_prefix=f"{model_name}_{self.name}_mlp_train",
            model=model,
            cache_root=self.cache_root,
            use_cache=use_cache,
            desc="ct-mlp-train",
            max_examples=self.max_train_examples,
        )

    def build_test_dataset(self, model, model_name, use_cache=True):
        return load_or_build_manifest_dataset(
            manifest_path=self.manifest_root / "benchmark_eval.jsonl",
            cache_prefix=f"{model_name}_{self.name}_benchmark_eval",
            model=model,
            cache_root=self.cache_root,
            use_cache=use_cache,
            desc="ct-benchmark-eval",
            max_examples=self.max_test_examples,
        )
