import warnings

from multimeditron.model.prompt_tokenizers import NUM_EMBEDDINGS_KEY
from ..modality import ModalityWithProjectionConfig, ModalityWithProjection
import torch
from transformers import AutoImageProcessor, VisionTextDualEncoderModel, CLIPModel, AutoModel
from multimeditron.dataset.registry.registry import ModalityRegistry
import numpy as np
from PIL import Image
import io

from typing import Optional, Dict, Any


class ImageConfig(ModalityConfig):
    model_type = "meditron_clip"

    def __init__(
            self,
            modality_name: Optional[str] = None,
            max_batch_size: int = 32,
            use_bias_proj: bool = True,
            clip_name: str = "openai/clip-vit-large-patch14",
            **kwargs
            ):
        super().__init__(
                modality_name=modality_name,
                max_batch_size=max_batch_size,
                use_bias_proj=use_bias_proj,
                modality_type="image",
                kwargs=kwargs
                )

        self.clip_name = clip_name


class ImageModality(ModalityWithProjection):
    config_class = ImageConfig

    def __init__(self, config: ModalityWithProjectionConfig):
        super().__init__(config)

        self.vision_tower_name = config.clip_name

        try:
            self.image_processor = AutoImageProcessor.from_pretrained(
                    self.vision_tower_name)
        except:
            warnings.warn("No Image Processor found, using openai/clip-vit-large-patch14")
            self.image_processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14")

        self.feature_extractor = AutoModel.from_pretrained(self.vision_tower_name, trust_remote_code=True)
        self._embedding_size = self.feature_extractor.vision_embed_dim
        self._num_patches_per_entry = (self.feature_extractor.vision_model.config.image_size // self.feature_extractor.vision_model.config.patch_size) ** 2

    def __call__(self, inputs) -> torch.FloatTensor:
        inputs = torch.stack(inputs, dim=0)
        inputs = inputs.to(self.feature_extractor.device)
        image_features = self.feature_extractor.vision_model(inputs).last_hidden_state[:, 1:, :]
        return image_features

    def modality_to_tensor(self, modality) -> Dict[str, Any]:
        processed_modality = modality.copy()

        image = modality["value"]
        if isinstance(modality["value"], dict):
            image = Image.open(io.BytesIO(image["bytes"]))

        processed_modality["value"] = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        processed_modality[NUM_EMBEDDINGS_KEY] = self._num_patches_per_entry

        return processed_modality

    @property
    def embedding_size(self) -> int:
        return self._embedding_size
    
    @classmethod
    def from_dict(cls, config_args, **kwargs):
        return ImageConfig.from_dict(config_args, **kwargs)

