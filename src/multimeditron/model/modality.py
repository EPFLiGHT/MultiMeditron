from __future__ import annotations
import abc

from transformers import PretrainedConfig, PreTrainedModel
from typing import Any, Optional, OrderedDict, Dict
from dataclasses import dataclass
from abc import ABC, abstractmethod
from enum import Enum
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from transformers import AutoModel, AutoTokenizer, PretrainedConfig, PreTrainedTokenizer, PreTrainedModel

from multimeditron.dataset.registry.registry import ModalityRegistry


class ModalityWithProjectionConfig(PretrainedConfig):
    def __init__(self, hidden_size: int,
                 modality_name: Optional[str] = None,
                 max_batch_size: int = 32,
                 modality_type: Optional[str] = None,
                 **kwargs):
        self.modality_type = modality_type  # e.g., 'image', 'audio'
        self.modality_name = modality_name  # e.g., 'ClipImage', 'ClipAudio'

        self.max_batch_size = max_batch_size
        self.hidden_size = hidden_size

        super().__init__(**kwargs)


class ModalityWithProjection(ABC, PreTrainedModel):
    def __init__(self, config: ModalityWithProjectionConfig, dtype: torch.dtype = torch.bfloat16):
        super().__init__(config)

        self.config = config
        self.config_class = ModalityWithProjectionConfig
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

    def get_config(self) -> ModalityWithProjectionConfig:
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
    
    @abstractmethod
    def freeze_projection_only(self):
        """
        Freeze the parameters of the projection layers, while keeping the modality trainable.
        """
    
    @abstractmethod
    def freeze_modality_only(self):
        """
        Freeze the parameters of the modality, while keeping the projection layers trainable.
        """
    
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

