import uuid
from multimeditron.model.constants import NUM_EMBEDDINGS_KEY, MODALITY_VALUE_KEY
from multimeditron.model.modalities.base import AutoModality, BaseModality, BaseModalityConfig, BaseModalityProcessor
from multimeditron.model.modalities.moe.gating import GatingNetwork
from multimeditron.model.projectors.mlp import MLPProjector
from multimeditron.model.attention import CrossAttention
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
        generalist_idx: int = -1,
        fusion_method: str = "weighted_average",
        cross_attn_heads: int = 8,
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
        elif self.fusion_method in ("weighted_average", "cross_attn"):  # CHANGED
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

        self.generalist_idx = config.generalist_idx
        self.fusion_method = config.fusion_method
        self.gating_network = GatingNetwork.from_pretrained(config.gating_path)

        # build perm[class_idx] = expert_idx so we can align gating → experts
        gate_class_names: List[str] = getattr(self.gating_network.config, "class_names", []) or []
        if gate_class_names:
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

        # cross-attention module (operates in projected space, dim = hidden_size)
        if self.fusion_method == "cross_attn":
            self.cross_attn = CrossAttention(
                dim=config.hidden_size,
                num_heads=config.cross_attn_heads,
                qkv_bias=True,
                attn_drop=0.1,
                proj_drop=0.1,
            )

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

        elif self.fusion_method == "cross_attn":
            # query=generalist → cross-attn over specialists
            B, E, P, C = stacked_expert_outputs.shape
            # generalist tokens as queries: [B, P, C]
            q = stacked_expert_outputs[:, self.generalist_idx, :, :]

            # specialist indices (all except generalist)
            specialist_indices = [i for i in range(E) if i != self.generalist_idx]  # just in case order changes

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
            return fused

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
