import torch
import open_clip
from modeling_clip import OpenCLIPVisionTextDualEncoderModel, VisionTextDualEncoderConfig

model_id = "calpt/CLIP-ViT-B-32-xlm-roberta-base-laion5B-s13B-b90k"
config = VisionTextDualEncoderConfig.from_pretrained(model_id)
config.vision_config.hidden_act = "gelu"
model = OpenCLIPVisionTextDualEncoderModel.from_pretrained(model_id, config=config)
