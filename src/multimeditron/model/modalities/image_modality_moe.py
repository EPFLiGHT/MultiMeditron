import torch
from multimeditron.model.constants import NUM_EMBEDDINGS_KEY, MODALITY_VALUE_KEY
from multimeditron.model.modalities import AutoModality, BaseModality, BaseModalityConfig, BaseModalityProcessor
from multimeditron.model.modalities.moe.gating import GatingNetwork
from multimeditron.model.projectors.mlp import MLPProjector
from multimeditron.model.attention import CrossAttention
from transformers import AutoModel, AutoImageProcessor, AutoConfig, AutoImageProcessor, AutoConfig
from typing import Dict, Any, List


class MOEImageConfig(BaseModalityConfig):
    def __init__(
        self,
        hidden_size: int = 1024,
        use_bias_proj: bool = True,
        expert_clip_names: List[str] = [],
        image_processor: str = "openai/clip-vit-large-patch14",
        gating_path: str = "",
        top_k_experts: int = 1,
        projection_type: str = "mlp",
        generalist_idx: int = -1,
        fusion_method: str = "weighted_average",
        cross_attn_heads: int = 8, 
        **kwargs,
    ):
        """
        Config for Mixture of Experts (MoE) Image Modality using CLIP models as experts.
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
            kwargs=kwargs,
        )

        self.expert_clip_names = expert_clip_names
        self.top_k_experts = top_k_experts
        self.gating_path = gating_path
        self.projection_type = projection_type
        self.image_processor = image_processor
        self.generalist_idx = generalist_idx
        self.fusion_method = fusion_method
        self.cross_attn_heads = cross_attn_heads


class MOEImageProcessor(BaseModalityProcessor):
    """
    Processor for Mixture of Experts (MoE) Image Modality.
    Uses a pretrained image processor to convert raw images into pixel values.
    Prepares processing for the fusion method between experts' outputs.
    """
    def __init__(self, config: MOEImageConfig):
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
        elif self.fusion_method in ("weighted_average", "cross_attn"):
            processed_modality[NUM_EMBEDDINGS_KEY] = self._num_patches_per_entry
        else:
            raise ValueError(f"Unknown fusion_method: {self.fusion_method}")

        return processed_modality



@AutoModality.register("moe_meditron_clip")
class MOEImageModality(BaseModality):
    """
    Mixture of Experts (MoE) Image Modality using CLIP models as experts.
    Combines multiple pretrained CLIP models as experts and uses a gating network to select and weight their outputs.
    During training, all experts are used and their outputs are weighted by the gating network.
    During evaluation, only the top-k experts are used (not implemented yet).
    """

    config_class = MOEImageConfig
    preprocessor_class = MOEImageProcessor

    def __init__(self, config: MOEImageConfig):
        super().__init__(config)

        self.experts = torch.nn.ModuleList()
        self.expert_names: List[str] = list(config.expert_clip_names)
        assert len(self.expert_names) > 0, "config.expert_clip_names must be non-empty"

        # verify last expert is generalist
        # print(self.expert_names[-1] + " is assumed to be the generalist expert.")
        
        self.embedding_size = None
        for clip_name in config.expert_clip_names:
            expert_model = AutoModel.from_pretrained(clip_name, trust_remote_code=True)

            if self.embedding_size is None:
                self.embedding_size = expert_model.vision_embed_dim

            self.experts.append(expert_model.vision_model)

        self._num_patches_per_entry = (self.experts[0].config.image_size // self.experts[0].config.patch_size) ** 2
        self.generalist_idx = config.generalist_idx
        self.fusion_method = config.fusion_method
        self.gating_network = GatingNetwork.from_pretrained(config.gating_path)

        gate_class_names: List[str] = getattr(self.gating_network.config, "class_names", []) or []
        if gate_class_names:
            # build perm[class_idx] = expert_idx
            name_to_expert_idx = {name: i for i, name in enumerate(self.expert_names)}
            try:
                perm_list = [name_to_expert_idx[name] for name in gate_class_names]
            except KeyError as e:
                raise ValueError(f"Gating class name {e} not found in expert_clip_names: {self.expert_names}")
        else:
            num_experts = len(self.experts)
            perm_list = list(range(num_experts))

        # register permutation as a non-persistent buffer 
        self.register_buffer("_gating_to_expert_perm", torch.tensor(perm_list, dtype=torch.long), persistent=False)
        self.projector = MLPProjector(self.embedding_size, config.hidden_size)

        if self.fusion_method.replace("-", "_") == "cross_attn":
            self.cross_attn = CrossAttention(
                dim=self.embedding_size,
                num_heads=config.cross_attn_heads,
                qkv_bias=True,
                attn_drop=0.1,
                proj_drop=0.1,
            )

        self.modality_frozen = not self.training

    def forward(self, inputs) -> torch.Tensor:
        device = next(self.experts[0].parameters()).device
        inputs = torch.stack(inputs, dim=0).to(device)

        _logits, _topk_indices, weights = self.gating_network(inputs)

        # Use all experts
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_out = expert(inputs).last_hidden_state[:, 1:, :]
            expert_outputs.append(expert_out)

        # stacked_expert_outputs shape: (B, N_experts, P, H)
        stacked_expert_outputs = torch.stack(expert_outputs, dim=1)

        if self.fusion_method == "sequence_append":
            # as each expert has the same P (patch_size) -> if mix ViT experts with different P, need to handle differently
            # stacked_expert_outputs: (B, E, P, H)
            fused = torch.flatten(stacked_expert_outputs, start_dim=1, end_dim=2)  # (B, E*P, H)
        elif self.fusion_method == "weighted_average":
            # apply permutation to weights to align with expert order
            perm = self._gating_to_expert_perm  # shape (N,)

            weights = weights.index_select(dim=-1, index=perm)  # -> (B, N_experts)
            weights = weights.unsqueeze(-1).unsqueeze(-1)  # Shape: (batch_size, num_experts, 1, 1)
            fused = (stacked_expert_outputs * weights).sum(dim=1)
        elif self.fusion_method == "cross_attn":
            # query=generalist → cross-attn over specialists
            B, E, P, C = stacked_expert_outputs.shape
            # generalist tokens as queries: [B, P, C]
            q = stacked_expert_outputs[:, self.generalist_idx, :, :]

            # specialist indices (all except generalist)
            specialist_indices = [i for i in range(E) if i != self.generalist_idx] # just in case order changes

            # align gating weights to expert order
            perm = self._gating_to_expert_perm
            w_all = weights.index_select(dim=-1, index=perm)  # [B, E]

            # keep only specialists’ weights and softmax across them
            w_spec = w_all[:, specialist_indices]             # [B, E_spec] (E_spec = 4)
            w_spec = torch.softmax(w_spec, dim=-1)

            # build weighted expert contexts: list of [B, P, C]
            experts_ctx = []
            for j, e_idx in enumerate(specialist_indices):
                ctx = stacked_expert_outputs[:, e_idx, :, :]  # [B, P, C]
                # scale each specialist’s tokens by its gating weight
                wj = w_spec[:, j].view(B, 1, 1)               # [B, 1, 1]
                ctx = ctx * wj
                experts_ctx.append(ctx)
               
            # cross-attend: generalist queries over specialists
            fused = self.cross_attn(q, experts_ctx)           # [B, P, C]
        else:
            raise ValueError(f"Unsupported fusion_method: {self.fusion_method}")

        projected = self.projector(fused)
        return projected


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
        for parameters in self.projector.parameters():
            parameters.requires_grad = True
