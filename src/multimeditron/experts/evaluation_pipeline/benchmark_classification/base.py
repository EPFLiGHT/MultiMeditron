from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from transformers import VisionTextDualEncoderModel

from ..mlp_eval import MLP_eval

from .datasets import build_class_weights


class ClassificationBenchmark(ABC):
    """Base class for benchmarks using frozen image embeddings plus an MLP head.

    Subclasses are responsible for defining how train/test datasets are built.
    This class keeps the shared evaluation flow in one place.
    """

    name = "benchmark_classification"
    num_classes: int | None = None

    def __init__(self, cache_root: Path | None = None) -> None:
        self.cache_root = cache_root

    @abstractmethod
    def build_train_dataset(
        self,
        model: VisionTextDualEncoderModel,
        model_name: str,
        use_cache: bool = True,
    ):
        """Return the training dataset for this benchmark."""

    @abstractmethod
    def build_test_dataset(
        self,
        model: VisionTextDualEncoderModel,
        model_name: str,
        use_cache: bool = True,
    ):
        """Return the test dataset for this benchmark."""

    def evaluate(
        self,
        model_path: str,
        use_cache: bool = True,
        mlp_kwargs: dict | None = None,
    ):
        model = VisionTextDualEncoderModel.from_pretrained(model_path)
        model_name = Path(model_path).name
        return self.evaluate_model(
            model=model,
            model_name=model_name,
            use_cache=use_cache,
            mlp_kwargs=mlp_kwargs,
        )

    def evaluate_model(
        self,
        model: VisionTextDualEncoderModel,
        model_name: str,
        use_cache: bool = True,
        mlp_kwargs: dict | None = None,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        train_dataset = self.build_train_dataset(model=model, model_name=model_name, use_cache=use_cache)
        test_dataset = self.build_test_dataset(model=model, model_name=model_name, use_cache=use_cache)

        class_weights = build_class_weights(train_dataset.labels)
        loss = nn.CrossEntropyLoss(weight=class_weights)

        benchmark = MLP_eval(
            output_dim=self.get_num_classes(),
            training_set=train_dataset,
            test_set=test_dataset,
            loss=loss,
            **(mlp_kwargs or {}),
        )
        return benchmark.evaluate()

    def get_num_classes(self) -> int:
        if self.num_classes is None:
            raise ValueError(f"{self.__class__.__name__} must define num_classes")
        return self.num_classes
