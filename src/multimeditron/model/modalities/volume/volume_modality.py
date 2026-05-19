from __future__ import annotations

from typing import List

import torch
import torch.nn.functional as F
from transformers import AutoModel

from multimeditron.model.modalities.base import AutoModality, BaseModality
from multimeditron.model.modalities.volume.volume_config import VolumeConfig
from multimeditron.model.modalities.volume.volume_processor import VolumeProcessor
from multimeditron.model.projectors.mlp import MLPProjector


@AutoModality.register("volume_3d")
class VolumeModality(BaseModality):
    """
    3D volume modality backed by a pretrained vision encoder.

    Expected encoder behavior:
    - expose `encode_image(x)` that returns token features, or
    - expose `vision_model(x).last_hidden_state`.
    """

    config_class = VolumeConfig
    preprocessor_class = VolumeProcessor

    def __init__(self, config: VolumeConfig):
        super().__init__(config)

        load_kwargs = {"trust_remote_code": config.trust_remote_code}
        if config.clip_revision is not None:
            load_kwargs["revision"] = config.clip_revision

        self.feature_extractor = AutoModel.from_pretrained(
            config.pretrain_vision_model,
            **load_kwargs,
        )

        remote_cfg = getattr(self.feature_extractor, "config", None)
        self.embedding_size = int(getattr(remote_cfg, "hidden_size", 768))
        self.projector = MLPProjector(
            self.embedding_size,
            config.hidden_size,
            dtype=self.dtype,
        )

        self._embedder_frozen = False

    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.feature_extractor, "encode_image"):
            tokens = self.feature_extractor.encode_image(x)
        elif hasattr(self.feature_extractor, "vision_model"):
            tokens = self.feature_extractor.vision_model(x).last_hidden_state
        else:
            raise ValueError(
                "Volume encoder must implement `encode_image` or `vision_model(...).last_hidden_state`."
            )

        if tokens.ndim != 3:
            raise ValueError(f"Expected token tensor of shape (B, N, D), got {tuple(tokens.shape)}")

        # M3D-CLIP style outputs include a CLS token. Remove it when possible.
        if tokens.shape[1] > self.config.proj_out_num:
            tokens = tokens[:, 1:, :]
        return tokens

    def _pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        target_tokens = int(self.config.proj_out_num)
        if tokens.shape[1] == target_tokens:
            return tokens

        pooled = F.adaptive_avg_pool1d(tokens.transpose(1, 2), target_tokens)
        return pooled.transpose(1, 2)

    def forward(self, inputs: List[torch.Tensor]) -> torch.FloatTensor:
        x = torch.stack(inputs, dim=0).to(self.device)
        tokens = self._encode_tokens(x)
        tokens = self._pool_tokens(tokens)
        return self.projector(tokens)

    def freeze_modality_embedder(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = False
        self.feature_extractor.eval()
        self._embedder_frozen = True

    def unfreeze_modality_embedder(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = True
        self._embedder_frozen = False

    def unfreeze_projection(self):
        for p in self.projector.parameters():
            p.requires_grad = True

    def train(self, mode: bool = True):
        super().train(mode)
        if self._embedder_frozen:
            self.feature_extractor.eval()
        return self
