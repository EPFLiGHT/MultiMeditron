from multimeditron.model.constants import NUM_EMBEDDINGS_KEY, MODALITY_VALUE_KEY, POSITION_IDS_KEY
from multimeditron.model.modalities import BaseModality, BaseModalityConfig, AutoModality, BaseModalityProcessor
from multimeditron.model.projectors.mlp import MLPProjector
import torch
from transformers import AutoImageProcessor, AutoModel, AutoConfig
from multimeditron.model.projectors.pixel_shuffle import PixelShuffleProjector

from typing import Dict, Any


class ImageConfig(BaseModalityConfig):
    """
    Configuration class for the Image Modality. Extends the BaseModalityConfig.

    Attributes:
        hidden_size (int): Dimension of the hidden layer for the projection network.
        clip_name (str): Name of the CLIP model to use as the feature extractor.
        projection_type (str): Type of projection network (e.g., "mlp").
        use_2d_position_ids (bool): Whether to use the 2D positional embeddings adaptation for 1D llm without retraining.

    Example:
        >>> config = ImageConfig(hidden_size=512, clip_name="openai/clip-vit-base-patch32")
        >>> print(config.clip_name)
        openai/clip-vit-base-patch32
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        clip_name: str = "openai/clip-vit-large-patch14",
        projection_type: str = "mlp",
        pixel_shuffle_factor: int = 1,
        use_2d_position_ids: bool = False,
        **kwargs
    ):
        """
        Initializes the ImageConfig.

        Args:
            hidden_size (int): Dimension of the hidden layer for the projection network.
            clip_name (str): Name of the CLIP model to use as the feature extractor.
            projection_type (str): Type of projection network (e.g., "mlp").
            use_2d_position_ids (bool): Whether to use the 2D positional embeddings adaptation for 1D llm without retraining.
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            modality_type="image",
            hidden_size=hidden_size,
            kwargs=kwargs
        )

        self.clip_name = clip_name
        self.projection_type = projection_type
        self.pixel_shuffle_factor = pixel_shuffle_factor
        self.use_2d_position_ids = use_2d_position_ids


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
        self._image_size = (feature_extractor_config.vision_config.image_size // feature_extractor_config.vision_config.patch_size)
        
        # Adjust patch count if pixel shuffle is used
        raw_patches = self._image_size ** 2
        f = getattr(config, "pixel_shuffle_factor", 1)
        self._num_patches_per_entry = raw_patches // (f * f)

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

        # Force RGB conversion to prevent crashes with grayscale images from external evaluators
        if hasattr(image, "convert"):
            image = image.convert("RGB")
            
            # Hugging Face processor bug: if H or W is 1 or 3, infer_channel_dimension_format guesses C=1.
            # We fix this by ensuring the image is always at least 4x4 pixels before processing.
            if image.width < 4 or image.height < 4:
                image = image.resize((max(4, image.width), max(4, image.height)))

        processed_modality[MODALITY_VALUE_KEY] = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        processed_modality[NUM_EMBEDDINGS_KEY] = self._num_patches_per_entry

        if self.config.use_2d_position_ids:
            # Create a position ids tensor for 2D adaptation starting at 0 to image_size - 1 on both axis
            processed_modality[POSITION_IDS_KEY] = torch.stack(
                torch.meshgrid(
                    torch.arange(self._image_size, dtype=torch.long),
                    torch.arange(self._image_size, dtype=torch.long),
                    indexing="ij"
                ),
                dim=-1
            ).reshape(self._num_patches_per_entry, 2)  # (num_patches, 2)

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

        # Support both OpenAI CLIP (vision_embed_dim) and SigLIP2 (config.vision_config.hidden_size)
        if hasattr(self.feature_extractor, 'vision_embed_dim'):
            self.embedding_size = self.feature_extractor.vision_embed_dim
        else:
            self.embedding_size = self.feature_extractor.config.vision_config.hidden_size

        vision_cfg = self.feature_extractor.vision_model.config
        self._num_patches_per_entry = (vision_cfg.image_size // vision_cfg.patch_size) ** 2

        # SigLIP2 has no CLS token, OpenAI CLIP does
        self._has_cls_token = getattr(vision_cfg, 'cls_flag', not 'siglip' in self.vision_tower_name.lower())

        if config.projection_type == "pixel_shuffle":
            self.projector = PixelShuffleProjector(
                self.embedding_size, 
                config.hidden_size, 
                factor=config.pixel_shuffle_factor, 
                dtype=self.dtype
            )
        else:
            self.projector = MLPProjector(self.embedding_size, config.hidden_size, dtype=self.dtype)

    def forward(self, inputs) -> torch.FloatTensor:
        inputs = torch.stack(inputs, dim=0)
        inputs = inputs.to(self.feature_extractor.device)
        last_hidden = self.feature_extractor.vision_model(inputs).last_hidden_state
        # Skip CLS token for CLIP models that have one; SigLIP2 has no CLS token
        image_features = last_hidden[:, 1:, :] if self._has_cls_token else last_hidden

        projected = self.projector(image_features)

        return projected

    def freeze_modality_embedder(self):
        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = False

    def unfreeze_modality_embedder(self):
        for parameters in self.feature_extractor.parameters():
            parameters.requires_grad = True

    def unfreeze_projection(self):
        for parameters in self.projector.parameters():
            parameters.requires_grad = True


