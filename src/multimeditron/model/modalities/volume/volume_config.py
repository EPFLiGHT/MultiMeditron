from __future__ import annotations

from typing import Tuple

from multimeditron.model.modalities.base import BaseModalityConfig


class VolumeConfig(BaseModalityConfig):
    """
    Configuration for the 3D Volume Modality.

    Loads a pretrained 3D vision encoder (M3D-CLIP by default) and projects
    its patch tokens into the LLM embedding space via a parameter-free
    spatial average pool followed by an MLP projector. The pool collapses
    M3D-CLIP's (D/4, H/16, W/16) patch grid down to ``proj_out_num`` tokens.

    Args:
        hidden_size: LLM hidden dim (e.g. 4096 for Llama-3-8B).
        clip_name: HuggingFace id of the pretrained 3D encoder. Default
            ``GoodBaiBai88/M3D-CLIP`` (Apache-2.0, 3D ViT + BERT text tower).
        trust_remote_code: Required for M3D-CLIP's custom modeling code.
        volume_size: Target (D, H, W) after preprocessing. Must match the
            pretrained encoder's expected input. Default (32, 256, 256)
            matches M3D-CLIP.
        proj_out_num: Number of LLM tokens after spatial pooling. Default
            256 matches the M3D-LaMed perceiver contract.
        num_channels: 1 for single-channel CT/MRI volumes.
        projection_type: Forwarded to ``MLPProjector``.
    """

    def __init__(
        self,
        hidden_size: int = 4096,
        clip_name: str = "GoodBaiBai88/M3D-CLIP",
        trust_remote_code: bool = True,
        volume_size: Tuple[int, int, int] = (32, 256, 256),
        proj_out_num: int = 256,
        num_channels: int = 1,
        projection_type: str = "mlp",
        **kwargs,
    ):
        super().__init__(
            modality_type="3d_volume",
            hidden_size=hidden_size,
            **kwargs,
        )
        self.clip_name = clip_name
        self.trust_remote_code = trust_remote_code
        self.volume_size = tuple(volume_size)
        self.proj_out_num = proj_out_num
        self.num_channels = num_channels
        self.projection_type = projection_type

    @property
    def num_patches(self) -> int:
        return self.proj_out_num
