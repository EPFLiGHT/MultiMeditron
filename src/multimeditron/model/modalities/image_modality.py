from multimeditron.model.constants import NUM_EMBEDDINGS_KEY, MODALITY_VALUE_KEY, POSITION_IDS_KEY
from multimeditron.model.modalities import BaseModality, BaseModalityConfig, AutoModality, BaseModalityProcessor
from multimeditron.model.projectors.mlp import MLPProjector
from multimeditron.model.projectors.pixel_shuffle import PixelShuffleProjector
import torch
from transformers import AutoImageProcessor, AutoModel, AutoConfig

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
            projection_type (str): Type of projection network ("mlp" or "pixel_shuffle").
            pixel_shuffle_factor (int): Spatial downscale factor used with "pixel_shuffle" projection.
                Token count is reduced by ``pixel_shuffle_factor ** 2``. Ignored for "mlp".
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
        factor = getattr(config, "pixel_shuffle_factor", 1)
        self._spatial_size = self._image_size // factor  # side length after pixel-unshuffle
        self._num_patches_per_entry = self._spatial_size ** 2

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

        if self.config.use_2d_position_ids:
            # Create a position ids tensor for the post-projection spatial grid.
            # When pixel_shuffle_factor > 1, the grid is (image_size/f) x (image_size/f).
            processed_modality[POSITION_IDS_KEY] = torch.stack(
                torch.meshgrid(
                    torch.arange(self._spatial_size, dtype=torch.long),
                    torch.arange(self._spatial_size, dtype=torch.long),
                    indexing="ij"
                ),
                dim=-1
            ).reshape(self._num_patches_per_entry, 2)  # (num_patches, 2)

        return processed_modality


@AutoModality.register("meditron_clip")
class ImageModality(BaseModality):
    """Single-CLIP image modality with an MLP projection to the LLM hidden space."""

    config_class = ImageConfig
    preprocessor_class = ImageProcessor

    def __init__(self, config: ImageConfig):
        """Initialize the ImageModality with a pretrained CLIP vision tower and projector.

        Args:
            config (ImageConfig): Configuration specifying the CLIP model name,
                hidden size, and projection type.
        """
        super().__init__(config)

        self.vision_tower_name = config.clip_name
        assert self.vision_tower_name is not None, "vision_tower_name must be specified in the config"

        self.feature_extractor = AutoModel.from_pretrained(self.vision_tower_name, trust_remote_code=True)
        if hasattr(self.feature_extractor, "vision_embed_dim"):
            self.embedding_size = self.feature_extractor.vision_embed_dim
        else:
            self.embedding_size = self.feature_extractor.vision_model.config.hidden_size
        self._num_patches_per_entry = (self.feature_extractor.vision_model.config.image_size // self.feature_extractor.vision_model.config.patch_size) ** 2

        if config.projection_type == "mlp":
            self.projector = MLPProjector(self.embedding_size, config.hidden_size, dtype=self.dtype)
        elif config.projection_type == "pixel_shuffle":
            self.projector = PixelShuffleProjector(
                self.embedding_size,
                config.hidden_size,
                factor=config.pixel_shuffle_factor,
                dtype=self.dtype,
            )
        else:
            raise ValueError(f"Unsupported projection_type: {config.projection_type!r}. Expected 'mlp' or 'pixel_shuffle'.")

    def forward(self, inputs) -> torch.FloatTensor:
        """Extract CLIP vision features from a batch of images and project to LLM hidden size.

        Args:
            inputs (List[torch.Tensor]): List of preprocessed image tensors, one per sample.

        Returns:
            torch.FloatTensor: Projected patch embeddings of shape (batch, num_patches, hidden_size).
        """
        inputs = torch.stack(inputs, dim=0)
        inputs = inputs.to(self.feature_extractor.device)
        features = self.feature_extractor.vision_model(inputs).last_hidden_state
        # SigLIP/SigLIP2 have no CLS token — all N tokens are patch tokens.
        # CLIP-style models prepend a CLS token at position 0 that must be dropped.
        # Detect by checking whether N is already a perfect square.
        N = features.shape[1]
        H = int(N ** 0.5)
        if H * H == N:
            image_features = features  # no CLS token
        else:
            image_features = features[:, 1:, :]  # skip CLS

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

