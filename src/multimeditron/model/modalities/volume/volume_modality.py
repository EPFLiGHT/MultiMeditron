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
        self._patch_grid_pre = config.num_patches_pre
        self._patch_grid_post = config.num_patches_post

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

        expected_pre_tokens = (
            self._patch_grid_pre[0] * self._patch_grid_pre[1] * self._patch_grid_pre[2]
        )
        if tokens.shape[1] == expected_pre_tokens + 1:
            tokens = tokens[:, 1:, :]
        elif tokens.shape[1] != expected_pre_tokens:
            raise ValueError(
                "Unexpected number of encoder tokens. "
                f"Expected {expected_pre_tokens} (or {expected_pre_tokens + 1} with CLS), "
                f"got {tokens.shape[1]}"
            )
        return tokens

    def _spatial_pool_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, embed_dim = tokens.shape
        grid_d, grid_h, grid_w = self._patch_grid_pre
        expected_pre_tokens = grid_d * grid_h * grid_w
        if token_count != expected_pre_tokens:
            raise ValueError(
                "Token count does not match pre-pooling grid. "
                f"Expected {expected_pre_tokens}, got {token_count}"
            )

        x = tokens.transpose(1, 2).reshape(batch_size, embed_dim, grid_d, grid_h, grid_w)
        pool_d, pool_h, pool_w = self.config.pool_factor
        x = F.avg_pool3d(
            x,
            kernel_size=(pool_d, pool_h, pool_w),
            stride=(pool_d, pool_h, pool_w),
        )
        pooled_tokens = x.flatten(2).transpose(1, 2)

        expected_post_tokens = (
            self._patch_grid_post[0] * self._patch_grid_post[1] * self._patch_grid_post[2]
        )
        if pooled_tokens.shape[1] != expected_post_tokens:
            raise ValueError(
                "Unexpected post-pooling token count. "
                f"Expected {expected_post_tokens}, got {pooled_tokens.shape[1]}"
            )
        if pooled_tokens.shape[1] != self.config.proj_out_num:
            raise ValueError(
                "Post-pooling token count does not match proj_out_num. "
                f"Expected {self.config.proj_out_num}, got {pooled_tokens.shape[1]}"
            )
        return pooled_tokens

    def forward(self, inputs: List[torch.Tensor]) -> torch.FloatTensor:
        x = torch.stack(inputs, dim=0).to(self.device)
        tokens = self._encode_tokens(x)
        tokens = self._spatial_pool_tokens(tokens)
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
