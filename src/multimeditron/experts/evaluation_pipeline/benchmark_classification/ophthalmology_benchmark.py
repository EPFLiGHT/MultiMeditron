from __future__ import annotations

from pathlib import Path

from .ophthalmology_dr_benchmark import OphthalmologyDRBenchmark


class OphthalmologyBenchmark(OphthalmologyDRBenchmark):
    """Backward-compatible ophthalmology benchmark.

    The current aggregated ophthalmology manifests used by the training pipeline
    are dominated by diabetic retinopathy data and contain many additional
    diagnoses outside the old 4-class schema. To keep ``train_new_pipeline``
    stable and make the benchmark statistically meaningful, the historical
    ``ophthalmology`` benchmark now evaluates the clinically coherent binary
    task implemented by ``OphthalmologyDRBenchmark``.
    """

    name = "ophthalmology"
    labels = ['normal', 'diabetic_retinopathy']
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    num_classes = 2

    def __init__(
        self,
        dataset_root: str | Path | None = None,
        train_jsonl: str | Path | None = None,
        test_jsonl: str | Path | None = None,
        cache_root: Path | None = None,
        max_train_examples: int | None = None,
        max_test_examples: int | None = None,
    ) -> None:
        train_jsonls = (train_jsonl,) if train_jsonl is not None else None
        test_jsonls = (test_jsonl,) if test_jsonl is not None else None
        image_roots = (dataset_root,) if dataset_root is not None else None
        super().__init__(
            train_jsonls=train_jsonls,
            test_jsonls=test_jsonls,
            image_roots=image_roots,
            cache_root=cache_root,
        )
        self.dataset_root = Path(dataset_root) if dataset_root is not None else self.dataset_root
        self.train_jsonl = Path(train_jsonl) if train_jsonl is not None else self.train_jsonls[0]
        self.test_jsonl = Path(test_jsonl) if test_jsonl is not None else self.test_jsonls[0]
        self.max_train_examples = max_train_examples
        self.max_test_examples = max_test_examples

    def load_train_examples(self) -> list[dict]:
        examples = super().load_train_examples()
        return self._sample_examples_random(examples, self.max_train_examples, seed=42)

    def load_test_examples(self) -> list[dict]:
        examples = super().load_test_examples()
        return self._sample_examples_random(examples, self.max_test_examples, seed=43)
