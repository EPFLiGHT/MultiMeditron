from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoModel

from multimeditron.model.modalities.base import BaseModality, AutoModality
from multimeditron.model.modalities.volume.volume_config import VolumeConfig
from multimeditron.model.modalities.volume.volume_processor import VolumeProcessor
from multimeditron.model.projectors.mlp import MLPProjector


# M3D-CLIP's published patch size. Fixed by the pretrained weights.
_M3D_CLIP_PATCH_SIZE: Tuple[int, int, int] = (4, 16, 16)


def _patch_monai_for_m3dclip() -> None:
    """Reconcile M3D-CLIP's modeling code with MONAI >= 1.4.

    MONAI 1.4 renamed PatchEmbeddingBlock's ``pos_embed`` kwarg to
    ``proj_type``; M3D-CLIP's ``modeling_m3d_clip.py`` still passes the
    old name and crashes with a TypeError at load time.

    See: https://huggingface.co/GoodBaiBai88/M3D-CLIP/discussions/3
    """
    try:
        from monai.networks.blocks.patchembedding import PatchEmbeddingBlock
    except ImportError:
        return

    orig = PatchEmbeddingBlock.__init__
    if getattr(orig, "_m3dclip_patched", False):
        return

    def patched(self, *args, **kwargs):
        if "pos_embed" in kwargs and "proj_type" not in kwargs:
            kwargs["proj_type"] = kwargs.pop("pos_embed")
        return orig(self, *args, **kwargs)

    patched._m3dclip_patched = True
    PatchEmbeddingBlock.__init__ = patched


def _infer_patch_grid(
    volume_size: Tuple[int, int, int],
    patch_size: Tuple[int, int, int] = _M3D_CLIP_PATCH_SIZE,
) -> Tuple[int, int, int]:
    return tuple(v // p for v, p in zip(volume_size, patch_size))


def _infer_pool_factor(
    patch_grid: Tuple[int, int, int], proj_out_num: int
) -> Tuple[int, int, int]:
    n = patch_grid[0] * patch_grid[1] * patch_grid[2]
    if n == proj_out_num:
        return (1, 1, 1)
    ratio = round((n / proj_out_num) ** (1 / 3))
    if ratio < 1:
        ratio = 1
    return (ratio, ratio, ratio)


@AutoModality.register("volume_3d")
class VolumeModality(BaseModality):
    """
    3D Medical Volume Modality backed by a pretrained 3D vision encoder.

    Default backend: M3D-CLIP (``GoodBaiBai88/M3D-CLIP``, Apache-2.0).
    The encoder produces 1 + (D/4)*(H/16)*(W/16) tokens; the CLS token is
    dropped, the spatial grid is average-pooled to ``config.proj_out_num``
    tokens, and the result is projected to ``config.hidden_size`` via the
    existing ``MLPProjector``.

    Forward:
        inputs: List of (C, D, H, W) float tensors in [0, 1].
        returns: (B, proj_out_num, hidden_size).
    """

    config_class = VolumeConfig
    preprocessor_class = VolumeProcessor

    def __init__(self, config: VolumeConfig):
        super().__init__(config)

        _patch_monai_for_m3dclip()

        self.feature_extractor = AutoModel.from_pretrained(
            config.clip_name,
            trust_remote_code=config.trust_remote_code,
        )

        # M3D-CLIP ViT hidden size (768 for the published checkpoint).
        encoder_config = getattr(self.feature_extractor, "config", None)
        self.embedding_size = getattr(encoder_config, "hidden_size", 768)

        self._patch_grid = _infer_patch_grid(config.volume_size)
        self._pool_factor = _infer_pool_factor(self._patch_grid, config.proj_out_num)

        self.projector = MLPProjector(
            self.embedding_size,
            config.hidden_size,
            dtype=self.dtype,
        )

    def forward(self, inputs: List[torch.Tensor]) -> torch.FloatTensor:
        x = torch.stack(inputs, dim=0).to(self.device)

        # M3D-CLIP exposes encode_image -> (B, 1 + N_patches, D_vis).
        tokens = self.feature_extractor.encode_image(x)[:, 1:, :]

        b, n, d = tokens.shape
        gd, gh, gw = self._patch_grid
        expected = gd * gh * gw
        assert n == expected, (
            f"M3D-CLIP returned {n} patch tokens but config.volume_size "
            f"implies {gd}x{gh}x{gw}={expected}. Check volume_size."
        )

        # (B, N, D) -> (B, D, gd, gh, gw) -> avg-pool -> (B, N', D)
        grid = tokens.transpose(1, 2).reshape(b, d, gd, gh, gw)
        grid = F.avg_pool3d(grid, kernel_size=self._pool_factor, stride=self._pool_factor)
        tokens = grid.flatten(2).transpose(1, 2)

        return self.projector(tokens)

    def freeze_modality_embedder(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = False

    def unfreeze_modality_embedder(self):
        for p in self.feature_extractor.parameters():
            p.requires_grad = True

    def unfreeze_projection(self):
        for p in self.projector.parameters():
            p.requires_grad = True
