from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from multimeditron.model.prompt_tokenizers import MODALITIES_KEY, MODALITIES_TYPE_KEY


class BaseModalityLoader(ABC):
    """
    Abstract base class for modality loaders.
    """

    name: str # automatically set when registered

    @abstractmethod
    def load(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def __call__(self, *args, **kwds):
        return self.load(*args, **kwds)

    @staticmethod
    def merge_modalities(sample: Dict[str, Any], loaders: Dict[str, BaseModalityLoader]):
        if MODALITIES_KEY not in sample:
            return sample

        # Processed sample
        processed_sample = sample.copy()
        processed_sample[MODALITIES_KEY] = []

        # Add additional kwargs to modalities
        for modality in sample[MODALITIES_KEY]:
            modality_loader = loaders.get(modality[MODALITIES_TYPE_KEY], None)
            if modality_loader is None:
                raise ValueError(f"Modality loader for type '{modality[MODALITIES_TYPE_KEY]}' not found.")

            modality_preprocessed = modality.copy()
            modality_preprocessed["value"] = modality_loader(modality)
            processed_sample[MODALITIES_KEY].append(modality_preprocessed)

        return processed_sample

class AutoModalityLoader:
    _registry = {}

    def __init__(self):
        raise RuntimeError("AutoModalityLoader should not be instantiated directly. Please use the 'from_name' method.")

    @classmethod
    def register(c, name: str):
        def decorator(cls):
            if not issubclass(cls, BaseModalityLoader):
                raise ValueError(f"Class {cls.__name__} must inherit from AbstractModalityLoader to be registered.")
            modality_type = cls.name
            if modality_type in c._registry:
                raise ValueError(f"Modality type '{modality_type}' is already registered.")

            setattr(cls, "name", name)
            c._registry[modality_type] = cls

            return cls
        return decorator
    
    @classmethod
    def from_name(c, name: str, *args, **kwargs) -> BaseModalityLoader:
        if name not in c._registry:
            raise ValueError(f"Modality type '{name}' is not registered.")
        loader_class = c._registry[name]
        instance = loader_class(*args, **kwargs)
        setattr(instance, "name", name)
        return instance

from multimeditron.dataset.loader.image.bytes import RawImageLoader
from multimeditron.dataset.loader.image.fs import FileSystemImageLoader

__all__ = [
    BaseModalityLoader,
    AutoModalityLoader,
    RawImageLoader,
    FileSystemImageLoader,
]
