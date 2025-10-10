from typing import List, Optional, Tuple
import PIL
from transformers import PreTrainedModel, PretrainedConfig
from torchvision import models
import torch.nn as nn
import torch
from transformers import AutoImageProcessor

class GatingNetworkConfig(PretrainedConfig):
    """
    Configuration class for the Gating Network model.
    Attributes:
        num_classes (int): Number of output classes for the gating network (number of experts).
        top_k (int): Number of top predictions to consider for gating.
    """
    model_type = "gating_network"

    def __init__(self, num_classes: int = 2, 
                 top_k: int = 1, 
                 image_processor_path: str = "openai/clip-vit-large-patch14",
                 class_names : List[str] = [],
                 **kwargs):
        """
        Initializes the GatingNetworkConfig.
        Args:
            num_classes (int): Number of output classes for the gating network (number of experts).
            top_k (int): Number of top predictions to consider for gating.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.top_k = top_k
        self.image_processor_path = image_processor_path
        self.class_names = class_names


class GatingNetwork(PreTrainedModel):
    """
    A Gating Network model that uses a pretrained ResNet50 as the backbone.
    This model outputs logits for each expert, selects the top-k experts, and computes softmax weights.
    Attributes:
        config_class (GatingNetworkConfig): The configuration class for the model.
        resnet (nn.Module): The ResNet50 model used as the backbone.
        processor (AutoImageProcessor): The image processor for preprocessing input images.
        top_k (int): Number of top predictions to consider for gating.
    """

    config_class = GatingNetworkConfig

    def __init__(self, config: GatingNetworkConfig, resnet_path: Optional[str] = None):
        """
        Initializes the GatingNetwork model.
        Args:
            config (GatingNetworkConfig): The configuration for the model.
        """
        super().__init__(config)
        
        # Load pretrained gating weights
        if resnet_path is not None:
            resnet_weights = torch.load(resnet_path)
            self.resnet = models.resnet50(weights=None)
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, config.num_classes)
            self.resnet.load_state_dict(resnet_weights)
        else:
            self.resnet = models.resnet50(weights=None)
            self.resnet.fc = nn.Linear(self.resnet.fc.in_features, config.num_classes)

        self.top_k = config.top_k
        self.image_processor = AutoImageProcessor.from_pretrained(config.image_processor_path)

        self.post_init()

    def forward(self, pixel_values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass of the GatingNetwork.
        Args:
            pixel_values (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).

        Returns:
            logits (torch.Tensor): Logits for each expert of shape (batch_size, num_classes).
            topk_indices (torch.Tensor): Indices of the top-k experts of shape (batch_size, top_k).
            weights (torch.Tensor): Softmax weights for each expert of shape (batch_size, num_classes).
        """

        logits = self.resnet(pixel_values)
        topk_vals, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        weights = torch.nn.functional.softmax(logits, dim=-1)

        return logits, topk_indices, weights


    def preprocess_images(self, images: List[PIL.Image.Image]) -> torch.Tensor:
        """
        Preprocesses input images using the image processor.
        Args:
            images (List[PIL.Image] or torch.Tensor): List of input images or a tensor.

        Returns:
            torch.Tensor: Preprocessed image tensor.
        """
        if isinstance(images, torch.Tensor):
            return images
        else:
            return self.image_processor(images=images, return_tensors="pt")["pixel_values"]
