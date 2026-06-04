from typing import List, Optional, Tuple
import PIL
from transformers import PreTrainedModel, PretrainedConfig
from torchvision import models
import torch.nn as nn
import torch
from transformers import AutoImageProcessor

class GatingNetworkConfig(PretrainedConfig):
    """
    Configuration class for the ResNet50-based gating network used in MoE routing.
    """

    model_type = "gating_network"

    def __init__(self, num_classes: int = 2,
                 top_k: int = 1,
                 image_processor_path: str = "openai/clip-vit-base-patch32",
                 class_names: List[str] = [],
                 **kwargs):
        """
        Initialize the GatingNetworkConfig.

        Args:
            num_classes (int, optional): Number of output classes, i.e. number of
                expert slots. Defaults to 2.
            top_k (int, optional): Number of top experts selected per image during
                inference. Defaults to 1.
            image_processor_path (str, optional): HuggingFace model identifier (or
                local path) used to load the image preprocessor. Defaults to
                "openai/clip-vit-base-patch32".
            class_names (List[str], optional): Ordered list of expert class names
                corresponding to the output logits (e.g. ["CT", "MRI", ...]).
                Used to align gating indices with expert module indices. Defaults
                to an empty list.
            **kwargs: Additional arguments forwarded to PretrainedConfig.
        """
        super().__init__(**kwargs)
        self.num_classes = num_classes
        self.top_k = top_k
        self.image_processor_path = image_processor_path
        self.class_names = class_names


class GatingNetwork(PreTrainedModel):
    """
    ResNet50-based gating network that routes images to the appropriate expert(s).

    Given an input image, the network produces per-expert logits, derives softmax
    weights over all experts, and returns the top-k expert indices.  Used inside
    MOEImageModalityPEP to weight or select specialist CLIP encoders.
    """

    config_class = GatingNetworkConfig

    def __init__(self, config: GatingNetworkConfig, resnet_path: Optional[str] = None):
        """
        Initialize the GatingNetwork.

        Args:
            config (GatingNetworkConfig): Model configuration specifying number of
                classes, top-k, and image processor path.
            resnet_path (str, optional): Path to a raw ResNet50 state-dict file
                (.pt). When provided the weights are loaded directly instead of
                using HuggingFace's from_pretrained mechanism. Defaults to None.
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
