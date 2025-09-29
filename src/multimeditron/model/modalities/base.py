from __future__ import annotations
from transformers import PretrainedConfig, PreTrainedModel, ProcessorMixin
from typing import Any, Optional, Dict
from abc import ABC, abstractmethod
import torch
from transformers import AutoModel, AutoConfig, AutoProcessor, PretrainedConfig, PreTrainedModel

class BaseModalityConfig(PretrainedConfig):
    def __init__(self,
                 hidden_size: int,
                 modality_type: Optional[str] = None,
                 max_batch_size: int = 32,
                 **kwargs):
        self.modality_type = modality_type  # e.g., 'ClipImage', 'ClipAudio'
        self.max_batch_size = max_batch_size
        self.hidden_size = hidden_size

        super().__init__(**kwargs)

class BaseModalityProcessor(ABC, ProcessorMixin):
    def __init__(self, config: BaseModalityConfig):
        self.config = config

    @abstractmethod
    def process(self, inputs: Dict[str, Any]) -> torch.Tensor:
        raise NotImplementedError
    
    def __call__(self, inputs: Dict[str, Any]) -> torch.Tensor:
        return self.process(inputs)

class BaseModality(ABC, PreTrainedModel):
    preprocessor_class: type = None

    def __init__(self, config: BaseModalityConfig, dtype: torch.dtype = torch.bfloat16):
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

    @abstractmethod
    def modality_to_tensor(self, modality: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method to convert a modality into a tensor representation.

        Args:
            modality (Dict[str, Any]): Input modality data.

        Returns:
            Dict[str, Any]: Tensor representation of the modality.
        """
        ...

    def get_config(self) -> BaseModalityConfig:
        """
        Retrieve the configuration object associated with the modality.

        Returns:
            ModalityConfig: The configuration object.
        """
        return self.config

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Name of the modality.

        Returns:
            str: The name of the modality.
        """
        ...
    
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
            raise ValueError(f"Model loaded is not an instance of BaseModality. Got {type(model)}")
        return model
    
    @classmethod
    def preprocessor_from_name(c, name: str, *args, **kwargs) -> BaseModalityProcessor:
        if name not in c._registry:
            raise ValueError(f"Modality name '{name}' is not registered.")
        preprocessor_class = c._registry[name].preprocessor_class
        assert preprocessor_class is not None, f"Modality class '{name}' does not have a preprocessor_class defined."
        return preprocessor_class(*args, **kwargs)

    @classmethod
    def config_from_dict(c, config: dict, **kwargs) -> BaseModalityConfig:
        assert "model_type" in config, "Config dictionary must contain a 'model_type' key."
        if config["model_type"] not in c._registry:
            raise ValueError(f"Modality name '{config['model_type']}' is not registered.")
        config_class = c._registry[config["model_type"]].config_class
        assert config_class is not None, f"Modality class '{config['model_type']}' does not have a config_class defined."
        return config_class.from_dict(config, **kwargs)
