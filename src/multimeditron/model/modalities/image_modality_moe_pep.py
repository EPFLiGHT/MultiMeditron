import uuid
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
        image_processor: str = "openai/clip-vit-base-patch32",
        gating_path: str = "",
        top_k_experts: int = 5,
        projection_type: str = "mlp",
        fusion_method: str = "weighted_average",
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
        self.fusion_method = fusion_method

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
        self.top_k_experts = config.top_k_experts

        self.fusion_method = config.fusion_method


    def process(self, modality: Dict[str, Any]):
        processed_modality = modality.copy()

        image = modality[MODALITY_VALUE_KEY]

        pixel_values = self.image_processor(images=image, return_tensors="pt")["pixel_values"][0]
        processed_modality[MODALITY_VALUE_KEY] = pixel_values

        # Determine number of embeddings based on fusion method
        if self.fusion_method == "sequence_append":
            processed_modality[NUM_EMBEDDINGS_KEY] = self._num_patches_per_entry * self.top_k_experts
        elif self.fusion_method == "weighted_average":
            processed_modality[NUM_EMBEDDINGS_KEY] = self._num_patches_per_entry 
        else:
            raise ValueError(f"Unsupported fusion_method: {self.fusion_method}")

        return processed_modality


@AutoModality.register("moe_meditron_clip_pep")
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

        in_dims: List[int] = []  # collect in_dims for each expert
        self.expert_names: List[str] = list(config.expert_clip_names)
        self._native_embed_dim = None              # track experts’ native (pre-proj) dim
        self._embedding_size = config.hidden_size  # post-projection dim seen by the LLM

        for clip_name in config.expert_clip_names:
            expert_model = AutoModel.from_pretrained(clip_name, trust_remote_code=True)

            # robustly get the vision embedding dim across CLIP impls
            if hasattr(expert_model, "vision_embed_dim"):
                vision_dim = expert_model.vision_embed_dim
            else:
                vision_dim = getattr(expert_model.vision_model.config, "hidden_size")

            if self._native_embed_dim is None:
                self._native_embed_dim = vision_dim  # set once, for reference
            in_dims.append(vision_dim)               # append for EVERY expert

            self.experts.append(expert_model.vision_model)

        assert len(self.experts) > 0, "No experts provided in config.expert_clip_names."

        self._num_patches_per_entry = (
            self.experts[0].config.image_size // self.experts[0].config.patch_size
        ) ** 2

        # All experts must share the same patch grid for simple append.
        for e in self.experts[1:]:
            # ensure consistent patch/grid sizes across experts
            assert (
                e.config.patch_size == self.experts[0].config.patch_size
                and e.config.image_size == self.experts[0].config.image_size
            ), "sequence_append requires identical (image_size, patch_size) across experts."

        # per-expert projectors
        def make_projector(in_dim: int, out_dim: int):
            if config.projection_type == "mlp":
                return MLPProjector(in_dim, out_dim)
            raise ValueError(f"Unsupported projection_type: {config.projection_type}")

        # create one projector per expert
        self.projectors = torch.nn.ModuleList(
            [make_projector(in_dim, config.hidden_size) for in_dim in in_dims]
        )

        # check we do have one projector per expert
        assert (
            len(self.projectors) == len(self.experts)
        ), f"PEP expects one projector per expert, got {len(self.projectors)} vs {len(self.experts)}"

        self.fusion_method = config.fusion_method
        self.gating_network = GatingNetwork.from_pretrained(config.gating_path)

        self.modality_frozen = not self.training


    def forward(self, inputs) -> torch.Tensor:
        inputs = torch.stack(inputs, dim=0).to(self.device)  # (B, C, H, W)

        _logits, _topk_indices, weights = self.gating_network(inputs)  # weights: (B, E)
        
        # Use all experts: project per expert, then fuse
        expert_outputs = []
        for expert, projector in zip(self.experts, self.projectors):
            # expert_out: (B, 1+P, D_native); drop CLS → (B, P, D_native)
            expert_out = expert(inputs).last_hidden_state[:, 1:, :]
            # project to hidden_size per expert: (B, P, H)
            expert_outputs.append(projector(expert_out))

        # stacked_expert_outputs: (B, E, P, H)
        stacked_expert_outputs = torch.stack(expert_outputs, dim=1)

        if self.fusion_method == "sequence_append":
            # concat along the sequence axis → (B, E*P, H)
            concat = torch.flatten(stacked_expert_outputs, start_dim=1, end_dim=2)
            return concat

        elif self.fusion_method == "weighted_average":
            # weights: (B, E) → (B, E, 1, 1)
            weights = weights.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, num_experts, 1, 1)
            weighted_output = (stacked_expert_outputs * weights).sum(dim=1)  # (B, P, H)
            return weighted_output

        else:
            raise ValueError(f"Unsupported fusion_method: {self.fusion_method}")

    @property
    def embedding_size(self) -> int:
        return self._embedding_size  # post-projection dim 

    def train(self, mode: bool = True):
        super().train(mode)

        if self.modality_frozen:
            self.gating_network.eval()

        return self

    def freeze_modality_embedder(self):
        for params in self.gating_network.parameters():
            params.requires_grad = False

        for expert in self.experts:
            for params in expert.parameters():
                params.requires_grad = False

        self.gating_network.eval()
        self.modality_frozen = True
        
    def unfreeze_modality_embedder(self):
        for params in self.gating_network.parameters():
            params.requires_grad = True

        for expert in self.experts:
            for params in expert.parameters():
                params.requires_grad = True

        self.gating_network.train()
        self.modality_frozen = False

    
    def unfreeze_projection(self):
        for parameters in self.projectors.parameters():
            parameters.requires_grad = True

