import torch.nn as nn
import torch

class MLPProjector(nn.Module):
    """
    A multi-layer perceptron (MLP) projector for projecting input tensors output from modalities 
    to the embedding space of the transformer model.

    This class is designed to process input tensors of a specified size through a series of linear layers
    and activation functions, mapping them from the modality embedder embedding space to the LLM embedding space.

    Attributes:
        projection (nn.Sequential):
            A sequential container of linear layers and activation functions defining the projection pipeline.
    """
    def __init__(self, modality_size: int, projected_size: int, dtype: torch.dtype = torch.bfloat16):
        """
        Initialize the MLPProjector with the given parameters.

        Args:
            modality_size (int): The size of the input modality (number of input features).
            projected_size (int): The size of the projected output space (number of output features).
            dtype (torch.dtype, optional): The data type for the layers (default is torch.bfloat16).

        This constructor creates a sequential pipeline consisting of:
            - A linear layer that transforms the input features to the same size.
            - A GELU activation layer for non-linear transformation.
            - Another linear layer projecting to the specified output size.
            - A GELU activation layer.
            - A final linear layer that maintains the projected size.
        """
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(modality_size, modality_size, dtype=dtype),
            nn.GELU(),
            nn.Linear(modality_size, projected_size, dtype=dtype),
            nn.GELU(),
            nn.Linear(projected_size, projected_size, dtype=dtype),
        )

    def forward(self, hidden_state: torch.Tensor) -> torch.FloatTensor:
        """
        Perform the forward pass of the MLP projector.

        This method takes an input tensor, processes it through the projection pipeline
        (which consists of sequential linear layers and activation functions), and produces
        a tensor representation in the projected space.

        Args:
            hidden_state (torch.Tensor):
                A tensor of shape (batch_size, modality_size) representing the input features.

        Returns:
            torch.FloatTensor:
                A tensor of shape (batch_size, projected_size) representing the projected features.
        """
        projection = self.projection(hidden_state)

        return projection

