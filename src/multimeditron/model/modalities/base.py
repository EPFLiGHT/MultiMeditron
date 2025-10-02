from __future__ import annotations
from transformers import PretrainedConfig, PreTrainedModel, ProcessorMixin
from typing import Any, Optional, Dict
from abc import ABC, abstractmethod
import torch
from transformers import AutoModel, AutoConfig, AutoProcessor, PretrainedConfig, PreTrainedModel

class BaseModalityConfig(PretrainedConfig):
    """
    Configuration class for defining modality parameters.

    This configuration is used as the base for all modality-specific configurations.

    Attributes:
        hidden_size (int): The size of the hidden layers' representation.
        modality_type (Optional[str]): The type of modality (e.g., 'ClipImage', 'ClipAudio').
        max_batch_size (int): The maximum batch size supported by the modality.
    """
    def __init__(self,
                 hidden_size: int = 1024,
                 modality_type: Optional[str] = None,
                 max_batch_size: int = 32,
                 **kwargs):
        """
        Initializes the BaseModalityConfig.

        Args:
            hidden_size (int): The size of the hidden layers' representation. Default is 1024.
            modality_type (Optional[str]): The type of modality (e.g., 'ClipImage', 'ClipAudio'). Default is None.
            max_batch_size (int): The maximum batch size supported by the modality. Default is 32.
            **kwargs: Additional keyword arguments passed to the PretrainedConfig initializer.
        """
        self.modality_type = modality_type  # e.g., 'ClipImage', 'ClipAudio'
        self.max_batch_size = max_batch_size
        self.hidden_size = hidden_size

        super().__init__(**kwargs)

class BaseModalityProcessor(ABC, ProcessorMixin):
    """
    Abstract base class for modality processors.

    The BaseModalityProcessor defines a standard interface for processing inputs of a specific modality.
    Subclasses must implement the abstract `process` method.

    Attributes:
        config (BaseModalityConfig): Configuration object for the processor.
    """
    def __init__(self, config: BaseModalityConfig):
        """
        Initializes the BaseModalityProcessor with the given configuration.

        Args:
            config (BaseModalityConfig): Configuration object for the processor.
        """
        self.config = config

    @abstractmethod
    def process(self, modality: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method for processing modality.

        Args:
            modality (Dict[str, Any]): Input data to be processed.

        Returns:
            Dict[str, Any]: The original sample with the processed modality
        """
        raise NotImplementedError

    def __call__(self, modality: Dict[str, Any]) -> Dict[str, Any]:
        """
        Makes the processor callable, processing the given modality.

        Args:
            modality (Dict[str, Any]): Input data to be processed.

        Returns:
            Dict[str, Any]: The original sample with the processed modality
        """
        return self.process(modality)

class BaseModality(ABC, PreTrainedModel):
    """
    Abstract base class for modality models.

    This base class defines the common interface and attributes for all modality models.
    Subclasses must implement the abstract methods `embedding_size`, `freeze_projection_only`, and `freeze_modality_only`.

    Attributes:
        config (BaseModalityConfig): Configuration object for the modality.
        config_class (type): Class reference for the configuration.
        tokenizer (Optional[Any]): Tokenizer associated with the model, if any.
        _dtype (torch.dtype): Data type for the model's tensors.
    """
    preprocessor_class: type = None

    def __init__(self, config: BaseModalityConfig, dtype: torch.dtype = torch.bfloat16):
        """
        Initializes the BaseModality with the given configuration and data type.

        Args:
            config (BaseModalityConfig): Configuration object for the modality.
            dtype (torch.dtype): Data type for the model's tensors. Default is torch.bfloat16.
        """
        super().__init__(config)

        self.config = config
        self.config_class = BaseModalityConfig
        self.tokenizer = None
        self._dtype = dtype

    @property
    @abstractmethod
    def embedding_size(self) -> int:
        """
        Abstract property that must be implemented to return the embedding size of the modality.

        Returns:
            int: The size of the embedding vector.
        """
        ...

    @property
    def num_patches_per_entry(self) -> Optional[int]:
        """
        Property that returns the number of patches per entry, if applicable.

        Returns:
            Optional[int]: Number of patches per entry, or None if not applicable.
        """
        return None

    def get_config(self) -> BaseModalityConfig:
        """
        Retrieve the configuration object associated with the modality.

        Returns:
            ModalityConfig: The configuration object.
        """
        return self.config
    
    @abstractmethod
    def freeze_projection_only(self):
        """
        Freeze the parameters of the projection layers, while keeping the modality trainable.
        """
        ...
    
    @abstractmethod
    def freeze_modality_only(self):
        """
        Freeze the parameters of the modality, while keeping the projection layers trainable.
        """
        ...
    
    def freeze_all(self):
        """
        Freeze all parameters in the model.
        """
        for params in self.parameters():
            params.requires_grad = False

    def unfreeze_all(self):
        """
        Unfreeze all parameters in the model.
        """
        for params in self.parameters():
            params.requires_grad = True


class AutoModality:
    """
    A class for managing modality registration and retrieval.

    The AutoModality class provides a centralized registry for modality subclasses. It handles the registration of modality classes,
    and allows users to retrieve pretrained models, processors, and configurations for specific modalities.

    Attributes:
        _registry (dict): Internal dictionary storing registered modality classes, indexed by name.
    """
    _registry = {}

    def __init__(self):
        raise RuntimeError("AutoModality should not be instantiated directly. Please use the 'from_name' method.")
    
    @classmethod
    def register(c, name: str):
        def decorator(cls):
            if not issubclass(cls, BaseModality):
                raise ValueError(f"Class {cls.__name__} must inherit from BaseModality to be registered.")
            if name in c._registry:
                raise ValueError(f"Modality name '{name}' is already registered.")
            if not hasattr(cls, "preprocessor_class") or cls.preprocessor_class is None:
                raise ValueError(f"Modality class '{cls.__name__}' must define a 'preprocessor_class' attribute.")
            c._registry[name] = cls
            setattr(cls.config_class, "model_type", name)

            AutoConfig.register(name, cls)
            AutoModel.register(cls.config_class, cls)
            AutoProcessor.register(cls.config_class, cls.preprocessor_class)

            return cls
        return decorator

    @classmethod
    def from_pretrained(c, *args, **kwargs) -> BaseModality:
        model = AutoModel.from_pretrained(*args, **kwargs)
        if not isinstance(model, BaseModality):
            raise ValueError(f"Model loaded is not an instance of BaseModality. Got {type(model)}. Available values are {list(c._registry.keys())}")
        return model
    
    @classmethod
    def preprocessor_from_name(c, name: str, *args, **kwargs) -> BaseModalityProcessor:
        if name not in c._registry:
            raise ValueError(f"Modality name '{name}' is not registered. Available values are {list(c._registry.keys())}")
        preprocessor_class = c._registry[name].preprocessor_class
        assert preprocessor_class is not None, f"Modality class '{name}' does not have a preprocessor_class defined."
        return preprocessor_class(*args, **kwargs)

    @classmethod
    def config_from_dict(c, config: dict, **kwargs) -> BaseModalityConfig:
        assert "model_type" in config, "Config dictionary must contain a 'model_type' key."
        if config["model_type"] not in c._registry:
            raise ValueError(f"Modality name '{config['model_type']}' is not registered. Available values are {list(c._registry.keys())}")
        config_class = c._registry[config["model_type"]].config_class
        assert config_class is not None, f"Modality class '{config['model_type']}' does not have a config_class defined."
        return config_class.from_dict(config, **kwargs)
