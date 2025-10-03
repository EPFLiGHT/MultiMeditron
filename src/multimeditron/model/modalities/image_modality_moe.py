from multimeditron.model.constants import NUM_EMBEDDINGS_KEY, MODALITY_VALUE_KEY
from multimeditron.model.modalities.base import AutoModality, BaseModality, BaseModalityConfig, BaseModalityProcessor
from multimeditron.model.modalities.moe.gating import GatingNetwork
from multimeditron.model.projectors.mlp import MLPProjector
import torch
from transformers import AutoModel, AutoImageProcessor, AutoConfig
from typing import Dict, Any, List


class MOEImageConfig(BaseModalityConfig):
    def __init__(
        self,
        hidden_size: int = 1024,
        max_batch_size: int = 32,
        use_bias_proj: bool = True,
        expert_clip_names: List[str] = [
            "openai/clip-vit-large-patch14", 
            "openai/clip-vit-large-patch14"
        ],
        image_processor: str = "openai/clip-vit-large-patch14",
        gating_path: str = "",
        top_k_experts: int = 1,
        projection_type: str = "mlp",
        **kwargs,
    ):
        super().__init__(
            max_batch_size=max_batch_size,
            use_bias_proj=use_bias_proj,
            modality_type="image",
            hidden_size=hidden_size,
            kwargs=kwargs,
        )

        self.expert_clip_names = expert_clip_names
        self.top_k_experts = top_k_experts
        self.gating_path = gating_path
        self.projection_type = projection_type
        self.image_processor = image_processor


class MOEImageProcessor(BaseModalityProcessor):
    def __init__(self, config: MOEImageConfig):
        super().__init__(config)
        self.image_processor = AutoImageProcessor.from_pretrained(config.image_processor)

        processor_config = AutoConfig.from_pretrained(config.image_processor, trust_remote_code=True)
        self._num_patches_per_entry = (processor_config.vision_config.image_size // processor_config.vision_config.patch_size) ** 2


    def process(self, modality: Dict[str, Any]):
        processed_modality = modality.copy()

        image = modality[MODALITY_VALUE_KEY]

        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        processed_modality[MODALITY_VALUE_KEY] = pixel_values
        processed_modality[NUM_EMBEDDINGS_KEY] = self._num_patches_per_entry

        return processed_modality


@AutoModality.register("moe_meditron_clip")
class MOEImageModality(BaseModality):
    config_class = MOEImageConfig
    preprocessor_class = MOEImageProcessor

    def __init__(self, config: MOEImageConfig):
        super().__init__(config)

        self.experts = torch.nn.ModuleList()

        self._embedding_size = None
        for clip_name in config.expert_clip_names:
            expert_model = AutoModel.from_pretrained(clip_name, trust_remote_code=True)

            if self._embedding_size is None:
                self._embedding_size = expert_model.vision_embed_dim

            self.experts.append(expert_model.vision_model)

        self._num_patches_per_entry = (self.experts[0].config.image_size // self.experts[0].config.patch_size) ** 2

        self.gating_network = GatingNetwork.from_pretrained(config.gating_path)
        self.image_processor = self.gating_network.image_processor

        self.projector = MLPProjector(self._embedding_size, config.hidden_size)

    def forward(self, inputs) -> torch.Tensor:
        device = next(self.experts[0].parameters()).device
        inputs = torch.stack(inputs, dim=0).to(device)

        _logits, _topk_indices, weights = self.gating_network(inputs)
        
        if self.training:
            # Use all experts
            expert_outputs = []
            for expert in self.experts:
                expert_out = expert(inputs).last_hidden_state[:, 1:, :]

                expert_outputs.append(expert_out)

            # stacked_expert_outputs shape: (num_experts, batch_size, num_patches, embedding_size)
            stacked_expert_outputs = torch.stack(expert_outputs, dim=1)
            weights = weights.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, 1, 1, num_experts)

            weighted_output = (stacked_expert_outputs * weights).sum(dim=1)

            projected = self.projector(weighted_output)

            return projected

        else:
            # Evaluation mode
            raise NotImplementedError("Evaluation mode not implemented yet.")

    @property
    def embedding_size(self) -> int:
        return self._embedding_size

    def freeze_modality_only(self):
        for params in self.gating_network.parameters():
            params.requires_grad = False
        
        for expert in self.experts:
            for params in expert.parameters():
                params.requires_grad = False

    def freeze_projection_only(self):
        for params in self.projector.parameters():
            params.requires_grad = False


