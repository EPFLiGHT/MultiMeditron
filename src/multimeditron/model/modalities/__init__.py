from .image_modality_pe import PEImageConfig, PEImageModality
from .image_modality import ImageModality, ImageConfig
from .three_d_modality import CTModality, CTConfig
from .image_modality_moe import ImageModality as MoEImageModality, ImageConfig as MoEImageConfig
from transformers import AutoConfig, AutoModel

MODALITY_FROM_NAME = {
    "ImageModality": ImageModality,
    "PerceptionImageModality": PEImageModality,
    "3DModality": CTModality,
    "MoEImageModality": MoEImageModality,
}

MODALITY_CONFIG_FROM_NAME = {
    "ImageModality": ImageConfig,
    "PerceptionImageModality": PEImageConfig,
    "3DModality": CTConfig,
    "MoEImageModality": MoEImageConfig,
}

MODALITY_CONFIG_FROM_MODEL_TYPE = {
    ImageConfig.model_type: ImageConfig,
    PEImageConfig.model_type: PEImageConfig,
    CTConfig.model_type: CTConfig,
    MoEImageConfig.model_type: MoEImageConfig,
}

for modality_name in MODALITY_FROM_NAME.keys():
    cls = MODALITY_FROM_NAME[modality_name]
    config_cls = MODALITY_CONFIG_FROM_NAME[modality_name]
    AutoConfig.register(getattr(config_cls, "model_type"), cls)
    AutoModel.register(config_cls, cls)
