import warnings

from transformers.models import sam

from ..modality import AbstractModality, ModalityConfig
import torch
from transformers import AutoImageProcessor, VisionTextDualEncoderModel, CLIPModel, AutoModel
from multimeditron.dataset.registry.registry import ModalityRegistry
from multimeditron.model.prompt_tokenizers import NUM_EMBEDDINGS_KEY
import torch
import core.vision_encoder.pe as pe
import core.transforms.image_transform as transforms
from PIL import Image
import io

import numpy as np

from typing import Optional, Dict, Any


class PEImageConfig(ModalityConfig):
    model_type = "meditron_perception_encoder"

    def __init__(
            self,
            modality_name: Optional[str] = None,
            max_batch_size: int = 32,
            use_bias_proj: bool = True,
            pe_name: str = "PE-Core-B16-224",
            max_num_tiles: int = 1,
            **kwargs
            ):
        super().__init__(
            modality_name=modality_name,
            max_batch_size=max_batch_size,
            use_bias_proj=use_bias_proj,
            modality_type="image",
            kwargs=kwargs
        )

        self.pe_name = pe_name
        self.max_num_tiles = max_num_tiles
        

class PEImageModality(AbstractModality):
    config_class = PEImageConfig

    def __init__(self, config: ModalityConfig):
        super().__init__(config)

        self.pe_name = config.pe_name
        self.feature_extractor = pe.VisionTransformer.from_config(self.pe_name, pretrained=True)  # Downloads from HF

        self.preprocessor = transforms.get_image_transform(
                image_res=self.feature_extractor.image_size,
        )

        self.num_patches = (self.feature_extractor.image_size // 
                            self.feature_extractor.patch_size) ** 2
        self._embedding_size = self.feature_extractor.width
    
    def __call__(self, inputs) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]

        # inputs is a list of size batch that contains samples with shape:
        # (num_tiles, 3, image_width, image_height)
        num_samples = len(inputs)
        
        samples_num_tiles = [inputs[i].shape[0] 
                                  for i in range(num_samples)]
        
        limit_indices = np.cumsum([0] + samples_num_tiles)
        
        # concatenated_samples is a tensor of shape
        # (sum of num_of_tiles, 3, image_width, image_height)
        concatenated_samples = torch.cat(inputs).to(self.dtype).to(self.device)

        # sample_features is a tensor of shape
        # (sum of num_tiles, num_embeddings, embedding_size)
        sample_features = self.feature_extractor.forward_features(concatenated_samples, 
                    norm=True, strip_cls_token=True)

        outputs = []
        for start_idx, end_idx in zip(limit_indices, limit_indices[1:]):
            features = sample_features[start_idx:end_idx].flatten(0, 1)
            outputs.append(features)
            
        return torch.stack(outputs)
        
    def modality_to_tensor(self, modality: Dict[str, Any]) -> Dict[str, Any]:
        processed_modality = modality.copy()

        image = modality["value"]
        if isinstance(modality["value"], dict):
            image = Image.open(io.BytesIO(image["bytes"]))
        
        preprocessed, ar = self.preprocessor(image)
       
        processed_modality[NUM_EMBEDDINGS_KEY] = preprocessed.shape[0] * self.num_patches
        processed_modality["value"] = preprocessed

        return processed_modality

    @property
    def embedding_size(self) -> int:
        return self._embedding_size
    
    @classmethod
    def from_dict(cls, config_args, **kwargs):
        return PEImageConfig.from_dict(config_args, **kwargs)

