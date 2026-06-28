from pathlib import Path

from .base import ClassificationBenchmark
from .multimediset_manifest import DEFAULT_MANIFEST_ROOT, load_or_build_manifest_dataset


class UltrasoundBenchmark(ClassificationBenchmark):
    """Ultrasound body-part / pathology classification via multimediset manifest."""

    name = "ultrasound"
    num_classes = 13
    default_manifest_root = DEFAULT_MANIFEST_ROOT / "ultrasound"

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
            cache_prefix=f"{model_name}_ultrasound_multimediset_mlp_train",
            model=model,
            cache_root=self.cache_root,
            use_cache=use_cache,
            desc="ultrasound-manifest-mlp-train",
            max_examples=self.max_train_examples,
            seed=42,
        )

    def build_test_dataset(self, model, model_name, use_cache=True):
        return load_or_build_manifest_dataset(
            manifest_path=self.manifest_root / "benchmark_eval.jsonl",
            cache_prefix=f"{model_name}_ultrasound_multimediset_benchmark_eval",
            model=model,
            cache_root=self.cache_root,
            use_cache=use_cache,
            desc="ultrasound-manifest-benchmark-eval",
            max_examples=self.max_test_examples,
            seed=43,
        )
