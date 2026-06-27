import logging
from typing import TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from datasets import Dataset

logger = logging.getLogger(__name__)

class BaseDatasetPreprocessor(ABC):
    """Abstract base class for dataset preprocessors.

    Subclasses must implement :meth:`_process` to define the transformation
    applied to a HuggingFace Dataset.  Instances are callable and can be
    chained via :func:`run_preprocessors`.
    """

    @abstractmethod
    def _process(self, ds: "Dataset", num_processes: int, **kwargs) -> "Dataset":
        """Apply the preprocessing transformation to a dataset.

        Args:
            ds (Dataset): The input HuggingFace dataset.
            num_processes (int): Number of parallel workers for map/filter.
            **kwargs: Processor-specific keyword arguments.

        Returns:
            Dataset: The transformed dataset.
        """
        raise NotImplementedError

    def process(self, ds: "Dataset", num_processes: int, **kwargs) -> "Dataset":
        """Run the preprocessor with debug logging.

        Args:
            ds (Dataset): The input HuggingFace dataset.
            num_processes (int): Number of parallel workers.
            **kwargs: Forwarded to :meth:`_process`.

        Returns:
            Dataset: The transformed dataset.
        """
        logger.debug(f"Running preprocessor: {self.name}")
        return self._process(ds, num_processes, **kwargs)

    def __call__(self, ds: "Dataset", num_processes: int, **kwargs) -> "Dataset":
        return self.process(ds, num_processes, **kwargs)

class AutoDatasetPreprocessor:
    """Registry for dataset preprocessors.

    Use the :meth:`register` decorator to add new preprocessors and
    :meth:`get` to retrieve them by name.
    """

    _registry = {}

    @classmethod
    def register(c, name: str):
        """Class decorator that registers a preprocessor under the given name.

        Args:
            name (str): Unique identifier for the preprocessor.

        Returns:
            A decorator that registers the class and returns it unchanged.

        Raises:
            ValueError: If a preprocessor with the same name is already registered.
        """
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
        """Retrieve a registered preprocessor by name.

        Args:
            name (str): The registered name of the preprocessor.

        Returns:
            BaseDatasetPreprocessor: The preprocessor instance.

        Raises:
            ValueError: If no preprocessor is registered under the given name.
        """
        if name not in c._registry:
            raise ValueError(f"Processor with name {name} is not registered. Available processors: {list(c._registry.keys())}")
        return c._registry[name]

def run_preprocessors(ds: "Dataset", num_processes: int, processors: list) -> "Dataset":
    """Run a sequence of registered preprocessors on a dataset.

    Temporarily disables HuggingFace dataset caching to avoid desync
    issues, then applies each processor in order.

    Args:
        ds (Dataset): The input HuggingFace dataset.
        num_processes (int): Number of parallel workers for each processor.
        processors (list): List of processor config objects with ``type``
            and ``kwargs`` attributes.

    Returns:
        Dataset: The dataset after all processors have been applied.
    """
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
