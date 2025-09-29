from transformers import PreTrainedModel, PretrainedConfig
import torch.nn as nn
import torch

class MLPProjectorConfig(PretrainedConfig):
    def __init__(self, modality_size: int,
                 projected_size: int,
                 **kwargs):
        super().__init__(**kwargs)
        self.modality_size = modality_size
        self.projected_size = projected_size


class MLPProjector(PreTrainedModel):
    """

    """
    def __init__(self, config: MLPProjectorConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)


        self.projection = nn.Sequential(
                nn.Linear(config.modality_size, config.modality_size, dtype=self.dtype),
                nn.GELU(),
                nn.Linear(config.modality_size, config.projected_size, dtype=self.dtype),
                nn.GELU(),
                nn.Linear(config.projected_size, config.projected_size, dtype=self.dtype),
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

