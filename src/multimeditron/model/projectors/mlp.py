from transformers import PreTrainedModel, PretrainedConfig
import torch.nn as nn
import torch

from .base import AbstractProjector

class MLPProjector(AbstractProjector):
    def __init__(self, modality_size: int, projected_size: int):
        self.projection = nn.Sequential(
                nn.Linear(modality_size, modality_size, dtype=self.dtype),
                nn.GELU(),
                nn.Linear(modality_size, projected_size, dtype=self.dtype),
                nn.GELU(),
                nn.Linear(projected_size, projected_size, dtype=self.dtype),
        )


    def forward(self, hidden_state: torch.Tensor) -> torch.FloatTensor:
        """
        Forward pass of the model for projection

        Args:
            value (Any): Input data to be processed by the modality.

        Returns:
            torch.FloatTensor: Projected tensor representation.
        """
        projection = self.projection(hidden_state)

        return projection

