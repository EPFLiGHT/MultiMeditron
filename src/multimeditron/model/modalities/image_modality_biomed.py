from transformers import AutoConfig, AutoImageProcessor, AutoModel
from multimeditron.model.modalities.base import BaseModalityConfig

import torch
from typing import Dict, Any
from PIL import Image

from multimeditron.model.constants import (
    NUM_EMBEDDINGS_KEY,
    MODALITY_VALUE_KEY,
    POSITION_IDS_KEY,
)
from multimeditron.model.modalities.base import BaseModalityProcessor, BaseModality, AutoModality
from multimeditron.model.projectors.mlp import MLPProjector


class BioMedCLIPImageConfig(BaseModalityConfig):
    """
    Image modality config for OpenCLIP-based models (e.g. BiomedCLIP)
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        clip_name: str = "",
        # Set this to true (trust)
        trust_remote_code: bool = False,
        projection_type: str = "mlp",
        **kwargs,
    ):
        super().__init__(
            modality_type="image",
            hidden_size=hidden_size,
            kwargs=kwargs,
        )

        self.clip_name = clip_name
        self.projection_type = projection_type
        self.trust_remote_code = trust_remote_code


class BioMedCLIPImageProcessor(BaseModalityProcessor):
    """
    Image processor using OpenCLIP preprocessing (BiomedCLIP-compatible)
    """

    def __init__(self, config: BioMedCLIPImageConfig):
        super().__init__(config)
        assert config.clip_name is not None

        self.preprocess = AutoImageProcessor.from_pretrained(
                config.clip_name, 
                trust_remote_code=config.trust_remote_code
        )
        feature_extractor_config = AutoConfig.from_pretrained(
                config.clip_name, 
                trust_remote_code=config.trust_remote_code
        )

        vision_cfg = feature_extractor_config.vision_cfg
        self._num_patches_per_entry = (vision_cfg["image_size"] // vision_cfg["patch_size"]) ** 2

    def process(self, modality: Dict[str, Any]) -> Dict[str, Any]:
        processed = modality.copy()
        image: Image.Image = modality[MODALITY_VALUE_KEY]

        pixel_values = self.preprocess(image)
        processed[MODALITY_VALUE_KEY] = pixel_values
        processed[NUM_EMBEDDINGS_KEY] = self._num_patches_per_entry

        return processed


@AutoModality.register("meditron_biomedclip")
class BioMedCLIPImageModality(BaseModality):
    """
    Image modality backed by BiomedCLIP (OpenCLIP).
    """

    config_class = BioMedCLIPImageConfig
    preprocessor_class = BioMedCLIPImageProcessor

    def __init__(self, config: BioMedCLIPImageConfig):
        super().__init__(config)

        assert config.clip_name is not None

        self.feature_extractor = AutoModel.from_pretrained(
                config.clip_name,
                trust_remote_code=config.trust_remote_code
        )

        remote_config = AutoConfig.from_pretrained(
                config.clip_name,
                trust_remote_code=config.trust_remote_code
        )
        self.embedding_size = remote_config.vision_cfg["width"]

        self.projector = MLPProjector(
            self.embedding_size,
            config.hidden_size,
            dtype=self.dtype,
        )


    def forward(self, inputs) -> torch.FloatTensor:
        """
        inputs: list[Tensor] each (3, 224, 224)
        """
        x = torch.stack(inputs, dim=0).to(self.device)

        # OpenCLIP ViT output: (B, D, P, P)
        # D is the dimension of an embedding
        # P is the number of patches
        res = self.feature_extractor.forward_intermediates(vision_inputs=x, normalize_intermediates=True)
        features = res["image_intermediates"][-1]
        features = features.flatten(start_dim=-2, end_dim=-1)
        features = features.transpose(-1, -2)

        projected = self.projector(features)

        return projected

    def freeze_modality_embedder(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def unfreeze_modality_embedder(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = True

    def unfreeze_projection(self):
        for p in self.projector.parameters():
            p.requires_grad = True

