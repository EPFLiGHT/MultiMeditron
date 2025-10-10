from multimeditron.model.constants import NUM_EMBEDDINGS_KEY, MODALITY_VALUE_KEY
from multimeditron.model.modalities.base import AutoModality, BaseModality, BaseModalityConfig, BaseModalityProcessor
from multimeditron.model.modalities.moe.gating import GatingNetwork
from multimeditron.model.projectors.mlp import MLPProjector
import torch
from transformers import AutoModel, AutoImageProcessor, AutoConfig
from typing import Dict, Any, List


class MOEImageConfigPEP(BaseModalityConfig):
    def __init__(
        self,
        hidden_size: int = 4096,
        use_bias_proj: bool = True,
        expert_clip_names: List[str] = [],
        image_processor: str = "openai/clip-vit-large-patch14",
        gating_path: str = "",
        top_k_experts: int = 1,
        projection_type: str = "mlp",
        **kwargs,
    ):
        """
        Config for Mixture of Experts (MoE) Image Modality using CLIP models as experts with per expert projection.
        Args:
            hidden_size (int): The hidden size of the output embeddings.
            use_bias_proj (bool): Whether to use bias in the projection layer.
            expert_clip_names (List[str]): List of pretrained CLIP model names to be used as experts.
            image_processor (str): Pretrained image processor name for preprocessing images.
            gating_path (str): Path to the pretrained gating network.
            top_k_experts (int): Number of top experts to select during inference.
            projection_type (str): Type of projection layer to use ("mlp" supported).
            **kwargs: Additional keyword arguments.
        """
        super().__init__(
            use_bias_proj=use_bias_proj,
            modality_type="image",
            hidden_size=hidden_size,
            **kwargs,
        )

        self.expert_clip_names = expert_clip_names
        self.top_k_experts = top_k_experts
        self.gating_path = gating_path
        self.projection_type = projection_type
        self.image_processor = image_processor


class MOEImageProcessorPEP(BaseModalityProcessor):
    """
    Processor for Mixture of Experts (MoE) Image Modality.
    Per Expert Projection (PEP) version.
    Uses a pretrained image processor to convert raw images into pixel values.
    """
    def __init__(self, config: MOEImageConfigPEP):
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


@AutoModality.register("moe_pep_meditron_clip")
class MOEImageModalityPEP(BaseModality):
    """
    Mixture of Experts (MoE) Image Modality using CLIP models as experts.
    Combines multiple pretrained CLIP models as experts and uses a gating network to select and weight their outputs.
    Uses Per Expert Projection (PEP) where each expert has its own projection layer.
    During training, all experts are used and their outputs are weighted by the gating network.
    During evaluation, only the top-k experts are used (not implemented yet).
    """

    config_class = MOEImageConfigPEP
    preprocessor_class = MOEImageProcessorPEP

    def __init__(self, config: MOEImageConfigPEP):
        super().__init__(config)

        self.experts = torch.nn.ModuleList()

        self._embedding_size = None
        for clip_name in config.expert_clip_names:
            expert_model = AutoModel.from_pretrained(clip_name, trust_remote_code=True)

            if self._embedding_size is None:
                self._embedding_size = expert_model.vision_embed_dim
            self.experts.append(expert_model.vision_model)
            

        self._num_patches_per_entry = (self.experts[0].config.image_size // self.experts[0].config.patch_size) ** 2
        assert len(self.experts) > 0, "No experts provided in config.expert_clip_names."

        # per-expert projectors
        def make_projector(in_dim: int, out_dim: int):
            if config.projection_type == "mlp":
                return MLPProjector(in_dim, out_dim)
            raise ValueError(f"Unsupported projection_type: {config.projection_type}")

        self.projectors = torch.nn.ModuleList(
            [make_projector(expert.vision_embed_dim, config.hidden_size)
             for expert in self.experts]
        )

        self.gating_network = GatingNetwork.from_pretrained(config.gating_path)

        
    def forward(self, inputs) -> torch.Tensor:
        device = next(self.experts[0].parameters()).device
        inputs = torch.stack(inputs, dim=0).to(device)

        _logits, _topk_indices, weights = self.gating_network(inputs)
        
        if self.training:
            # Use all experts
            expert_outputs = []
            expert_projector_outputs = []
            for expert, projector in zip(self.experts, self.projectors):
                expert_out = expert(inputs).last_hidden_state[:, 1:, :]
                expert_outputs.append(projector(expert_out))

            # stacked_expert_outputs shape: (num_experts, batch_size, num_patches, embedding_size)
            stacked_expert_outputs = torch.stack(expert_outputs, dim=1)
            weights = weights.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, 1, 1, num_experts)

            weighted_output = (stacked_expert_outputs * weights).sum(dim=1)

            return weighted_output

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
        for projector in self.projectors:
            for p in projector.parameters():
                p.requires_grad = False


