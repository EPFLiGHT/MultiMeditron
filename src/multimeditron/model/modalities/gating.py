from transformers import PreTrainedModel, PretrainedConfig
from torchvision import models,transforms, datasets
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoImageProcessor


class GatingNetworkConfig(PretrainedConfig):
    model_type = "gating_network"

    def __init__(self, num_labels: int = 2, top_k: int = 1, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.top_k = top_k


class GatingNetwork(PreTrainedModel):
    config_class = GatingNetworkConfig

    def __init__(self, config: GatingNetworkConfig):
        super().__init__(config)
        self.resnet = models.resnet50()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, config.num_labels)
        self.processor = AutoImageProcessor.from_pretrained("openai/clip-vit-large-patch14")
        self.top_k = config.top_k

        self.post_init()

    def forward(self, pixel_values, labels=None):
        logits = self.resnet(pixel_values)
        topk_vals, topk_indices = torch.topk(logits, self.top_k, dim=-1)

        weights = torch.nn.functional.softmax(logits, dim=-1)

        return logits, topk_indices, weights


AutoConfig.register("gating_network", GatingNetworkConfig)
AutoModel.register(GatingNetworkConfig, GatingNetwork)

