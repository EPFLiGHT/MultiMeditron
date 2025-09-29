import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModel, AutoTokenizer

from multimeditron.dataset.registry.registry import ModalityRegistry
from ..modality import AbstractModality, ModalityConfig
from typing import Optional, Dict, Any
import os
from PIL import Image
import io

from multimeditron.model.prompt_tokenizers import NUM_EMBEDDINGS_KEY


class CTConfig(ModalityConfig):
    model_type = "image_3d"

    def __init__(
        self,
        modality_name: Optional[str] = None,
        max_batch_size: int = 32,
        use_bias_proj: bool = True,
        **kwargs
    ):
        super().__init__(
            modality_name=modality_name,
            max_batch_size=max_batch_size,
            use_bias_proj=use_bias_proj,
            modality_type=CTConfig.model_type,
            kwargs=kwargs
        )

class CTModality(AbstractModality):
    config_class = CTConfig

    def __init__(self, config: CTConfig):
        super().__init__(config)
        self.clip_name = getattr(config, "clip_name", "GoodBaiBai88/M3D-CLIP")
        self.vision_tower_name = self.clip_name
        self.feature_extractor = AutoModel.from_pretrained(self.vision_tower_name, trust_remote_code=True)

    def preprocess(self, image, base_path = ""):
        # Prepare your 3D medical image:
        # 2. The image needs to be normalized to 0-1, considering Min-Max Normalization.
        # 3. The image format needs to be converted to .npy format.
        if isinstance(image, str):
            image_path = os.path.join(base_path, image)
            # Load and preprocess the image (expecting .npy format for 3D medical images)
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file {image_path} not found")
            image = np.load(image_path)
        elif isinstance(image, np.ndarray):
            image = image
        image = image / np.max(image)  # Min-Max normalization to 0-1
        # 1. The image shape needs to be processed as 1*32*256*256, considering resize and other methods.
        image = np.resize(image, (1, 32, 256, 256))
        image_tensor = torch.tensor(image).float()
        return image_tensor

    def __call__(self, image_tensors):
        image_tensors = image_tensors.to(self.feature_extractor.device).to(self.feature_extractor.dtype)
        outputs = self.feature_extractor.encode_image(image_tensors)[:, 0]

        return outputs


    def modality_to_tensor(self, modality: Dict[str, Any]) -> Dict[str, Any]:
        preprocessed, ar = self.preprocessor(Image.open(io.BytesIO(modality["value"]["bytes"])))
        modality[NUM_EMBEDDINGS_KEY] = preprocessed.shape[0] * self.num_patches

        return preprocessed

    @property
    def embedding_size(self) -> int:
        return self.feature_extractor.config.hidden_size
    
    @property
    def num_patches_per_entry(self) -> int:
        return 2048

    @classmethod
    def from_dict(cls, config_args, **kwargs):
        return CTConfig.from_dict(config_args, **kwargs)
