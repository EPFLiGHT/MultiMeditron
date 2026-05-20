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
        self._validate_remote_config(remote_cfg, config)
        self.embedding_size = int(getattr(remote_cfg, "hidden_size", 768))
        self.projector = MLPProjector(
            self.embedding_size,
            config.hidden_size,
            dtype=self.dtype,
        )
        self._patch_grid_pre = config.num_patches_pre
        self._patch_grid_post = config.num_patches_post

        self._embedder_frozen = False

    @staticmethod
    def _as_tuple(value):
        if value is None:
            return None
        return tuple(int(x) for x in value)

    def _validate_remote_config(self, remote_cfg, config: VolumeConfig) -> None:
        if remote_cfg is None:
            return

        remote_img_size = self._as_tuple(getattr(remote_cfg, "img_size", None))
        if remote_img_size is not None and remote_img_size != config.volume_size:
            raise ValueError(
                "Loaded 3D encoder img_size does not match VolumeConfig.volume_size: "
                f"{remote_img_size} != {config.volume_size}"
            )

        remote_patch_size = self._as_tuple(getattr(remote_cfg, "patch_size", None))
        if remote_patch_size is not None and remote_patch_size != config.patch_size:
            raise ValueError(
                "Loaded 3D encoder patch_size does not match VolumeConfig.patch_size: "
                f"{remote_patch_size} != {config.patch_size}"
            )

        remote_in_channels = getattr(remote_cfg, "in_channels", None)
        if remote_in_channels is not None and int(remote_in_channels) != 1:
            raise ValueError(
                "This volume_3d implementation currently supports only single-channel "
                f"M3D-CLIP inputs; loaded encoder has in_channels={remote_in_channels}"
            )

    @staticmethod
    def _first_tensor(output):
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "last_hidden_state"):
            return output.last_hidden_state
        if isinstance(output, (tuple, list)) and len(output) > 0:
            return VolumeModality._first_tensor(output[0])
        raise ValueError(f"Could not extract token tensor from encoder output type {type(output)}")

    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self.feature_extractor, "vision_encoder"):
            tokens = self._first_tensor(self.feature_extractor.vision_encoder(x))
        elif hasattr(self.feature_extractor, "vision_model"):
            tokens = self._first_tensor(self.feature_extractor.vision_model(x))
        elif hasattr(self.feature_extractor, "encode_image"):
            # Fallback for encoders that expose only projected CLIP features.
            tokens = self.feature_extractor.encode_image(x)
        else:
            raise ValueError(
                "Volume encoder must implement `vision_encoder`, `vision_model`, or `encode_image`."
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
        target_dtype = next(self.feature_extractor.parameters()).dtype
        x = torch.stack(inputs, dim=0).to(device=self.device, dtype=target_dtype)
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
