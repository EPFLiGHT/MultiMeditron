from __future__ import annotations

from math import prod
from typing import Tuple

from multimeditron.model.modalities.base import BaseModalityConfig


class VolumeConfig(BaseModalityConfig):
    """
    Configuration for the 3D volume modality.

    This config is used by:
    - VolumeProcessor (shape resize/normalization contract)
    - VolumeModality (vision encoder + projector settings)

    pretrain_vision_model accepts either:
    - a Hugging Face model id (e.g. GoodBaiBai88/M3D-CLIP), or
    - a local model path.
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        pretrain_vision_model: str = "GoodBaiBai88/M3D-CLIP",
        trust_remote_code: bool = True,
        projection_type: str = "mlp",
        volume_size: Tuple[int, int, int] = (32, 256, 256),
        patch_size: Tuple[int, int, int] = (4, 16, 16),
        proj_out_num: int = 256,
        pool_factor: Tuple[int, int, int] = (2, 2, 2),
        clip_revision: str | None = None,
        **kwargs,
    ):
        super().__init__(
            modality_type="image_3d",
            hidden_size=hidden_size,
            **kwargs,
        )

        self.pretrain_vision_model = pretrain_vision_model
        self.trust_remote_code = trust_remote_code
        self.projection_type = projection_type
        self.volume_size = tuple(int(x) for x in volume_size)
        self.patch_size = tuple(int(x) for x in patch_size)
        self.proj_out_num = int(proj_out_num)
        self.pool_factor = tuple(int(x) for x in pool_factor)
        self.clip_revision = clip_revision

        self.num_patches_pre: Tuple[int, int, int]
        self.num_patches_post: Tuple[int, int, int]

        self._validate()

    def _validate(self) -> None:
        if not self.pretrain_vision_model:
            raise ValueError("A pretrained 3D vision model must be provided via 'pretrain_vision_model'.")
        if len(self.volume_size) != 3:
            raise ValueError(
                f"volume_size must be (D, H, W), got: {self.volume_size}"
            )
        if any(v <= 0 for v in self.volume_size):
            raise ValueError(
                f"volume_size values must be > 0, got: {self.volume_size}"
            )
        if self.proj_out_num <= 0:
            raise ValueError(f"proj_out_num must be > 0, got: {self.proj_out_num}")

        if len(self.patch_size) != 3:
            raise ValueError(f"patch_size must be (d, h, w), got: {self.patch_size}")
        if any(p <= 0 for p in self.patch_size):
            raise ValueError(
                f"patch_size values must be > 0, got: {self.patch_size}"
            )

        if len(self.pool_factor) != 3:
            raise ValueError(
                f"pool_factor must be (fd, fh, fw), got: {self.pool_factor}"
            )
        if any(p <= 0 for p in self.pool_factor):
            raise ValueError(
                f"pool_factor values must be > 0, got: {self.pool_factor}"
            )

        if any(v % pch != 0 for v, pch in zip(self.volume_size, self.patch_size)):
            raise ValueError(
                "volume_size must be divisible by patch_size per axis, "
                f"got volume_size={self.volume_size}, patch_size={self.patch_size}"
            )
        self.num_patches_pre = tuple(
            v // pch for v, pch in zip(self.volume_size, self.patch_size)
        )

        if any(n % f != 0 for n, f in zip(self.num_patches_pre, self.pool_factor)):
            raise ValueError(
                "num_patches_pre must be divisible by pool_factor per axis, "
                f"got num_patches_pre={self.num_patches_pre}, pool_factor={self.pool_factor}"
            )
        self.num_patches_post = tuple(
            n // f for n, f in zip(self.num_patches_pre, self.pool_factor)
        )

        expected_proj_out_num = prod(self.num_patches_post)
        if self.proj_out_num != expected_proj_out_num:
            raise ValueError(
                "proj_out_num must match derived post-pooling token count, "
                f"expected {expected_proj_out_num} from volume_size={self.volume_size}, "
                f"patch_size={self.patch_size}, pool_factor={self.pool_factor}, "
                f"got proj_out_num={self.proj_out_num}"
            )
