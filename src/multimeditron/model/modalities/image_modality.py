from multimeditron.model.constants import NUM_EMBEDDINGS_KEY, MODALITY_VALUE_KEY
from multimeditron.model.modalities.base import BaseModality, BaseModalityConfig, AutoModality, BaseModalityProcessor
from multimeditron.model.projectors.mlp import MLPProjector
import torch
from transformers import AutoImageProcessor, AutoModel, AutoConfig

from typing import Dict, Any


class ImageConfig(BaseModalityConfig):
    """
    Configuration class for the Image Modality. Extends the BaseModalityConfig.

    Attributes:
        hidden_size (int): Dimension of the hidden layer for the projection network.
        max_batch_size (int): Maximum batch size for processing.
        clip_name (str): Name of the CLIP model to use as the feature extractor.
        projection_type (str): Type of projection network (e.g., "mlp").

    Example:
        >>> config = ImageConfig(hidden_size=512, max_batch_size=16, clip_name="openai/clip-vit-base-patch32")
        >>> print(config.clip_name)
        openai/clip-vit-base-patch32
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        max_batch_size: int = 32,
        clip_name: str = "openai/clip-vit-large-patch14",
        projection_type: str = "mlp",
        **kwargs
    ):
        """
        Initializes the ImageConfig.

        Args:
            hidden_size (int): Dimension of the hidden layer for the projection network.
            max_batch_size (int): Maximum batch size for processing.
            clip_name (str): Name of the CLIP model to use as the feature extractor.
            projection_type (str): Type of projection network (e.g., "mlp").
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            max_batch_size=max_batch_size,
            modality_type="image",
            hidden_size=hidden_size,
            kwargs=kwargs
        )

        self.clip_name = clip_name
        self.projection_type = projection_type


class ImageProcessor(BaseModalityProcessor):
    """
    A processor for handling image data. It uses a pretrained CLIP model for processing image inputs into tensors.

    Attributes:
        image_processor (AutoImageProcessor): An instance of a pretrained image processor.
        _num_patches_per_entry (int): The number of patches per image entry, based on image and patch size.
    """

    def __init__(self, config):
        """
        Initializes the ImageProcessor with the specified configuration.

        Args:
            config (ImageConfig): The configuration object specifying CLIP model details and other parameters.

        Raises:
            AssertionError: If `clip_name` is not specified in the configuration.
        """
        super().__init__(config)
        assert config.clip_name is not None, "clip_name must be specified in the config"

        self.image_processor = AutoImageProcessor.from_pretrained(config.clip_name)

        feature_extractor_config = AutoConfig.from_pretrained(config.clip_name, trust_remote_code=True)
        self._num_patches_per_entry = (feature_extractor_config.vision_config.image_size // feature_extractor_config.vision_config.patch_size) ** 2

    def process(self, modality: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes the input image modality into a tensor suitable for model consumption.

        Args:
            modality (Dict[str, Any]): The input image data, where "value" is the key for image data.

        Returns:
            torch.Tensor: The processed tensor representation of the image.
        """
        processed_modality = modality.copy()
        image = modality[MODALITY_VALUE_KEY]

        processed_modality[MODALITY_VALUE_KEY] = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        processed_modality[NUM_EMBEDDINGS_KEY] = self._num_patches_per_entry

        return processed_modality


@AutoModality.register("meditron_clip")
class ImageModality(BaseModality):
    config_class = ImageConfig
    preprocessor_class = ImageProcessor

    def __init__(self, config: ImageConfig):
        super().__init__(config)

        self.vision_tower_name = config.clip_name
        assert self.vision_tower_name is not None, "vision_tower_name must be specified in the config"

        self.feature_extractor = AutoModel.from_pretrained(self.vision_tower_name, trust_remote_code=True)
        self._embedding_size = self.feature_extractor.vision_embed_dim
        self._num_patches_per_entry = (self.feature_extractor.vision_model.config.image_size // self.feature_extractor.vision_model.config.patch_size) ** 2

        self.projector = MLPProjector(self._embedding_size, config.hidden_size, dtype=self.dtype)

    def __call__(self, inputs) -> torch.FloatTensor:
        inputs = torch.stack(inputs, dim=0)
        inputs = inputs.to(self.feature_extractor.device)
        image_features = self.feature_extractor.vision_model(inputs).last_hidden_state[:, 1:, :]

        projected = self.projector(image_features)

        return projected

    @property
    def embedding_size(self) -> int:
        return self._embedding_size
    
    @classmethod
    def from_dict(cls, config_args, **kwargs):
        return ImageConfig.from_dict(config_args, **kwargs)

    def freeze_modality_only(self):
        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False
        for parameters in self.projector.parameters():
            parameters.requires_grad = True

    def freeze_projection_only(self):
        for parameters in self.projector.parameters():
            parameters.requires_grad = False
        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = True

