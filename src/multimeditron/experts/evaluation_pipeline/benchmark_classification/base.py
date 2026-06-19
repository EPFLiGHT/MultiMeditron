import random
from abc import ABC, abstractmethod
from pathlib import Path

import torch
import torch.nn as nn
from transformers import VisionTextDualEncoderModel

from mlp_eval import MLP_eval

from .datasets import build_class_weights


class ClassificationBenchmark(ABC):
    """Base class for benchmarks using frozen image embeddings plus an MLP head.

    Subclasses are responsible for defining how train/test datasets are built.
    This class keeps the shared evaluation flow in one place.
    """

    name = "benchmark_classification"
    num_classes = None

    def __init__(
        self,
        cache_root=None,
        max_train_examples=None,
        max_test_examples=None,
    ):
        self.cache_root = cache_root
        self.max_train_examples = (
            max_train_examples
            if max_train_examples is not None
            else getattr(self, "max_train_examples", None)
        )
        self.max_test_examples = (
            max_test_examples
            if max_test_examples is not None
            else getattr(self, "max_test_examples", None)
        )

    @staticmethod
    def _sample_examples_random(
        examples,
        max_n,
        seed=42,
    ):
        """Randomly subsample *examples* to at most *max_n* entries.

        Returns the original list unchanged when *max_n* is None or already
        smaller than the list length.  Uses a fixed seed so results are
        reproducible across runs.
        """
        if max_n is None or len(examples) <= max_n:
            return examples
        rng = random.Random(seed)
        shuffled = examples.copy()
        rng.shuffle(shuffled)
        return shuffled[:max_n]

    @abstractmethod
    def build_train_dataset(
        self,
        model,
        model_name,
        use_cache=True,
    ):
        """Return the training dataset for this benchmark."""

    @abstractmethod
    def build_test_dataset(
        self,
        model,
        model_name,
        use_cache=True,
    ):
        """Return the test dataset for this benchmark."""

    def evaluate(
        self,
        model_path,
        use_cache=True,
        mlp_kwargs=None,
    ):
        model = VisionTextDualEncoderModel.from_pretrained(model_path)
        model_name = Path(model_path).name
        return self.evaluate_model(
            model=model,
            model_name=model_name,
            use_cache=use_cache,
            mlp_kwargs=mlp_kwargs,
        )

    def build_loss(self, train_dataset):
        """Return the loss function for MLP training. Override for non-standard losses."""
        class_weights = build_class_weights(
            train_dataset.labels, num_classes=self.get_num_classes()
        )
        return nn.CrossEntropyLoss(weight=class_weights)

    def build_mlp_kwargs(self):
        """Return extra kwargs forwarded to MLP_eval. Override to change accuracy function etc."""
        return {}

    def evaluate_model(
        self,
        model,
        model_name,
        use_cache=True,
        mlp_kwargs=None,
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        model.eval()

        train_dataset = self.build_train_dataset(
            model=model, model_name=model_name, use_cache=use_cache
        )
        test_dataset = self.build_test_dataset(
            model=model, model_name=model_name, use_cache=use_cache
        )

        loss = self.build_loss(train_dataset)

        benchmark = MLP_eval(
            output_dim=self.get_num_classes(),
            training_set=train_dataset,
            test_set=test_dataset,
            loss=loss,
            **(mlp_kwargs if mlp_kwargs is not None else self.build_mlp_kwargs()),
        )
        return benchmark.evaluate()

    def get_num_classes(self):
        if self.num_classes is None:
            raise ValueError(f"{self.__class__.__name__} must define num_classes")
        return self.num_classes
