from pathlib import Path

from .base import ClassificationBenchmark
from .multimediset_manifest import DEFAULT_MANIFEST_ROOT, load_or_build_manifest_dataset


class OphthalmologyBenchmark(ClassificationBenchmark):
    """Binary diabetic retinopathy benchmark built from fundus datasets.

    Distinguishes normal fundus images from any level of diabetic retinopathy.
    Source: EyeDataset val split via multimediset manifest.
    """

    name = "ophthalmology"
    num_classes = 2
    # Order matches the manifest label_map: diabetic_retinopathy=0, normal=1.
    labels = ["diabetic_retinopathy", "normal"]
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    default_manifest_root = DEFAULT_MANIFEST_ROOT / "eye"

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
            cache_prefix=f"{model_name}_{self.name}_multimediset_mlp_train",
            model=model,
            cache_root=self.cache_root,
            use_cache=use_cache,
            desc="ophthalmology-manifest-mlp-train",
            max_examples=self.max_train_examples,
            seed=42,
        )

    def build_test_dataset(self, model, model_name, use_cache=True):
        return load_or_build_manifest_dataset(
            manifest_path=self.manifest_root / "benchmark_eval.jsonl",
            cache_prefix=f"{model_name}_{self.name}_multimediset_benchmark_eval",
            model=model,
            cache_root=self.cache_root,
            use_cache=use_cache,
            desc="ophthalmology-manifest-benchmark-eval",
            max_examples=self.max_test_examples,
            seed=43,
        )
