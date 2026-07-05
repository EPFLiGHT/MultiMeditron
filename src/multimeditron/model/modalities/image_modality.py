from multimeditron.model.constants import NUM_EMBEDDINGS_KEY, MODALITY_VALUE_KEY, POSITION_IDS_KEY
from multimeditron.model.modalities import BaseModality, BaseModalityConfig, AutoModality, BaseModalityProcessor
from multimeditron.model.projectors.mlp import MLPProjector
import torch
from transformers import AutoImageProcessor, AutoModel, AutoConfig
from multimeditron.model.projectors.pixel_shuffle import PixelShuffleProjector

from typing import Dict, Any


class ImageConfig(BaseModalityConfig):
    def __init__(
        self,
        hidden_size: int = 4096,
        clip_name: str = "openai/clip-vit-large-patch14",
        projection_type: str = "mlp",
        pixel_shuffle_factor: int = 1,
        use_2d_position_ids: bool = False,
        **kwargs
    ):
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
    def __init__(self, config):
        super().__init__(config)
        assert config.clip_name is not None, "clip_name must be specified in the config"

        self.image_processor = AutoImageProcessor.from_pretrained(config.clip_name)

        feature_extractor_config = AutoConfig.from_pretrained(config.clip_name, trust_remote_code=True)
        self._image_size = (feature_extractor_config.vision_config.image_size // feature_extractor_config.vision_config.patch_size)

        raw_patches = self._image_size ** 2
        if getattr(config, "projection_type", "mlp") == "pixel_shuffle":
            f = getattr(config, "pixel_shuffle_factor", 1)
            self._num_patches_per_entry = raw_patches // (f * f)
        else:
            self._num_patches_per_entry = raw_patches

    def process(self, modality: Dict[str, Any]) -> Dict[str, Any]:
        processed_modality = modality.copy()
        image = modality[MODALITY_VALUE_KEY]

        if hasattr(image, "convert"):
            image = image.convert("RGB")
            if image.width < 4 or image.height < 4:
                image = image.resize((max(4, image.width), max(4, image.height)))

        if torch.is_tensor(image):
            if image.dtype == torch.bfloat16:
                image = image.to(torch.float32)
        elif isinstance(image, (list, tuple)):
            image = [img.to(torch.float32) if (torch.is_tensor(img) and img.dtype == torch.bfloat16) else img for img in image]

        processed_modality[MODALITY_VALUE_KEY] = self.image_processor(images=image, return_tensors="pt")["pixel_values"]
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

        if hasattr(self.feature_extractor, 'vision_embed_dim'):
            self.embedding_size = self.feature_extractor.vision_embed_dim
        else:
            self.embedding_size = self.feature_extractor.config.vision_config.hidden_size

        vision_cfg = self.feature_extractor.vision_model.config
        self._num_patches_per_entry = (vision_cfg.image_size // vision_cfg.patch_size) ** 2
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
        batch_tensors = []
        for inp in inputs:
            if inp.ndim == 3:
                batch_tensors.append(inp.unsqueeze(0))
            else:
                batch_tensors.append(inp)

        inputs = torch.cat(batch_tensors, dim=0).to(self.feature_extractor.device)
        last_hidden = self.feature_extractor.vision_model(inputs).last_hidden_state

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

