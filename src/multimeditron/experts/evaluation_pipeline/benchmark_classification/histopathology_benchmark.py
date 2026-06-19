from pathlib import Path

from .base import ClassificationBenchmark
from .multimediset_manifest import DEFAULT_MANIFEST_ROOT, load_or_build_manifest_dataset


# 33 TCGA cancer types — labels are the snake-case filenames with _ → space,
# sorted alphabetically and indexed at manifest build time.
# The authoritative list and label_id mapping live in
# benchmark_splits/multimediset/histopathology/split_summary.json.
# The num_classes here is a safe upper bound; actual count depends on the
# manifest (run build_histopathology_splits.py to generate it).
_EXPECTED_NUM_CLASSES = 33


class HistopathologyBenchmark(ClassificationBenchmark):
    name = "histopathology"
    num_classes = _EXPECTED_NUM_CLASSES

    default_manifest_root = DEFAULT_MANIFEST_ROOT / "histopathology"

    max_train_examples = 10_000
    max_test_examples = 5_000

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
            cache_prefix=f"{model_name}_{self.name}_strat_mlp_train",
            model=model,
            cache_root=self.cache_root,
            use_cache=use_cache,
            desc="histopathology-mlp-train",
            max_examples=self.max_train_examples,
            stratify_by_label=True,
        )

    def build_test_dataset(self, model, model_name, use_cache=True):
        return load_or_build_manifest_dataset(
            manifest_path=self.manifest_root / "benchmark_eval.jsonl",
            cache_prefix=f"{model_name}_{self.name}_strat_benchmark_eval",
            model=model,
            cache_root=self.cache_root,
            use_cache=use_cache,
            desc="histopathology-benchmark-eval",
            max_examples=self.max_test_examples,
            stratify_by_label=True,
        )
