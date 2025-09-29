from transformers import PreTrainedModel, PretrainedConfig
from abc import ABC

from multimeditron.model.modality import ModalityWithProjectionConfig

class ModalityProjectorConfig(PretrainedConfig):
    def __init__(self, modality_size: int,
                 projected_size: int,
                 **kwargs):
        super().__init__(**kwargs)
        self.modality_size = modality_size
        self.projected_size = projected_size

class ModalityProjector(PreTrainedModel, ABC):
    def __init__(self, config: ModalityWithProjectionConfig, *inputs, **kwargs):
        super().__init__(config, *inputs, **kwargs)
        self.config = config


    def

