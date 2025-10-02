"""
This module provides classes and utilities for handling modality loaders in the MultiMeditron framework.

Main components:
- `BaseModalityLoader`: An abstract base class for defining the structure of modality loaders.
- `AutoModalityLoader`: A registry and factory for managing modality loaders.

Key Features:
- Centralized management of modality loaders.
- Dynamic registration and instantiation of modality loader classes.
- Support for merging multiple modalities in a structured way.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List
from multimeditron.model.constants import MODALITIES_KEY, MODALITY_TYPE_KEY, MODALITY_VALUE_KEY


class BaseModalityLoader(ABC):
    """
    Abstract base class for modality loaders.

    Attributes:
        name (str): Name of the modality loader. Automatically set when registered.
    """

    name: str # automatically set when registered

    @abstractmethod
    def load(self, *args, **kwargs) -> Any:
        """
        Abstract method to load modality data. This method must be implemented by subclasses.

        Args:
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            Any: The loaded modality data.
        """
        raise NotImplementedError

    def __call__(self, *args, **kwds):
        """
        Allows the instance to be called like a function, which internally calls the load method.

        Args:
            *args: Variable length argument list.
            **kwds: Arbitrary keyword arguments.

        Returns:
            Any: The result of the load method.
        """
        return self.load(*args, **kwds)

    @staticmethod
    def load_modalities(sample: Dict[str, Any], loaders: Dict[str, BaseModalityLoader]):
        """
        Load modalities in a sample using the provided modality loaders.
        This function loads the modality if needed using the loader defined in `loaders`
        For instance:
            If `loaders` is a dictionary containing an element `"image" : FileSystemImageLoader`,
            then this function assumes that every samples store images as path and load the image
            using the `FileSystemImageLoader`

        Note that each sample should contain a key `MODALITY_TYPE_KEY` defining what kind of modality this is
        (for instance "image"). And a key `MODALITY_VALUE_KEY` defining the value of the modality,
        this could be a path or the bytes of the images for instance.

        Args:
            sample (Dict[str, Any]): The input sample containing modalities.
            loaders (Dict[str, BaseModalityLoader]): A dictionary mapping modality types to their corresponding loaders.

        Returns:
            Dict[str, Any]: The processed sample with merged modalities.

        Raises:
            ValueError: If a modality type is not found in the loaders.
        """
        # Check if the sample contains the required key for modalities
        if MODALITIES_KEY not in sample:
            return sample

        # Create a copy of the sample to process
        processed_sample = sample.copy()
        processed_sample[MODALITIES_KEY] = []

        # Iterate through each modality in the sample
        for modality in sample[MODALITIES_KEY]:
            # Retrieve the corresponding modality loader
            modality_loader = loaders.get(modality[MODALITY_TYPE_KEY], None)
            if modality_loader is None:
                raise ValueError(f"Modality loader for type '{modality[MODALITY_TYPE_KEY]}' not found.")

            # Preprocess the modality using the loader
            modality_preprocessed = modality.copy()
            modality_preprocessed[MODALITY_VALUE_KEY] = modality_loader(modality)
            processed_sample[MODALITIES_KEY].append(modality_preprocessed)

        return processed_sample

class AutoModalityLoader:
    """
    Loader registry to automatically manage and instantiate modality loaders.

    The `AutoModalityLoader` acts as a central registry for different modality loader classes, enabling easy registration,
    instantiation, and retrieval of loaders. Each loader must inherit from `BaseModalityLoader` to be compatible.

    Attributes:
        _registry (Dict[str, Type[BaseModalityLoader]]): Internal registry of modality loader classes keyed by name.
    """
    _registry = {}

    def __init__(self):
        """
        Prevent instantiation of `AutoModalityLoader` directly.

        Raises:
            RuntimeError: Always raised to enforce usage of class methods.
        """
        raise RuntimeError("AutoModalityLoader should not be instantiated directly. Please use the 'from_name' method.")

    @classmethod
    def register(c, name: str):
        """
        Register a modality loader class under a specific name.

        Args:
            name (str): The name under which the loader class will be registered.

        Returns:
            Callable: A decorator to register the loader class.

        Raises:
            ValueError: If the class is not a subclass of `BaseModalityLoader` or if the name is already registered.
        """
        def decorator(clazz):
            if not issubclass(clazz, BaseModalityLoader):
                raise ValueError(f"Class {clazz.__name__} must inherit from AbstractModalityLoader to be registered.")
            if name in c._registry:
                raise ValueError(f"Modality type '{name}' is already registered.")

            setattr(clazz, "name", name)
            c._registry[name] = clazz

            return clazz
        return decorator
    
    @classmethod
    def from_name(cls, name: str, *args, **kwargs) -> BaseModalityLoader:
        """
        Retrieve and instantiate a registered loader class by name.

        Args:
            name (str): The registered name of the loader class.
            *args: Positional arguments to pass to the loader's initializer.
            **kwargs: Keyword arguments to pass to the loader's initializer.

        Returns:
            BaseModalityLoader: The instantiated loader.

        Raises:
            ValueError: If the name is not registered.
        """
        if name not in cls._registry:
            raise ValueError(f"Modality type '{name}' is not registered.")
        loader_class = cls._registry[name]
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

