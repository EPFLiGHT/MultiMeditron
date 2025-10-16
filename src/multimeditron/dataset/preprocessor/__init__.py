import logging
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)

class BaseDatasetPreprocessor(ABC):
    @abstractmethod
    def _process(self, ds: "Dataset", num_processes: int, **kwargs) -> "Dataset":
        raise NotImplementedError

    def process(self, ds: "Dataset", num_processes: int, **kwargs) -> "Dataset":
        logger.debug(f"Running preprocessor: {self.name}")
        return self._process(ds, num_processes, **kwargs)

    def __call__(self, ds: "Dataset", num_processes: int, **kwargs) -> "Dataset":
        return self.process(ds, num_processes, **kwargs)

class AutoDatasetPreprocessor:
    _registry = {}

    @classmethod
    def register(c, name: str):
        def wrapper(cls):
            # Register the name as a static string
            if name in c._registry:
                raise ValueError(f"Processor with name {name} is already registered.")

            # Instantiate the processor class and store it in the registry
            processor = cls()
            setattr(cls, "name", name)
            setattr(processor, "name", name)
            c._registry[name] = processor
            return cls
        return wrapper

    @classmethod
    def get(c, name: str) -> BaseDatasetPreprocessor:
        if name not in c._registry:
            raise ValueError(f"Processor with name {name} is not registered. Available processors: {list(c._registry.keys())}")
        return c._registry[name]

def run_preprocessors(ds: "Dataset", num_processes: int, processors: list) -> "Dataset":
    from datasets import enable_caching, disable_caching, is_caching_enabled

    # Disable caching as it often causes desync issues
    was_caching_enabled = is_caching_enabled()
    disable_caching()

    # Run each processor in sequence
    for idx, proc in enumerate(processors):
        logger.info(f"Running processor [{idx+1}/{len(processors)}]: {proc.type} with args: {proc.kwargs}")
        processor = AutoDatasetPreprocessor.get(proc.type)
        ds = processor(ds, num_processes, **proc.kwargs)

    # Restore previous caching state
    if was_caching_enabled:
        enable_caching()
    return ds

from multimeditron.dataset.preprocessor.python import PythonProcessor, PythonFilterProcessor
from multimeditron.dataset.preprocessor.shuffle import ShuffleProcessor

__all__ = [
    "BaseDatasetPreprocessor",
    "AutoDatasetPreprocessor",
    "run_preprocessors",
    "PythonProcessor",
    "PythonFilterProcessor",
    "ShuffleProcessor",
]
