from __future__ import annotations

from typing import Tuple

from multimeditron.model.modalities.base import BaseModalityConfig


class VolumeConfig(BaseModalityConfig):
    """
    Configuration for the 3D volume modality.

    This config is used by:
    - VolumeProcessor (shape resize/normalization contract)
    - VolumeModality (vision encoder + projector settings)
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        pretrain_vision_model: str | None = None,
        trust_remote_code: bool = True,
        projection_type: str = "mlp",
        volume_size: Tuple[int, int, int] = (32, 256, 256),
        proj_out_num: int = 256,
        pool_factor: Tuple[int, int, int] | None = None,
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
        self.proj_out_num = int(proj_out_num)
        self.pool_factor = tuple(int(x) for x in pool_factor) if pool_factor is not None else None
        self.clip_revision = clip_revision

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

        if self.pool_factor is not None:
            if len(self.pool_factor) != 3:
                raise ValueError(
                    f"pool_factor must be (fd, fh, fw), got: {self.pool_factor}"
                )
            if any(p <= 0 for p in self.pool_factor):
                raise ValueError(
                    f"pool_factor values must be > 0, got: {self.pool_factor}"
                )
