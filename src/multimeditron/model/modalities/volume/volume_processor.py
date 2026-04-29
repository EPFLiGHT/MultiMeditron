from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from multimeditron.model.modalities.base import BaseModalityProcessor
from multimeditron.model.modalities.volume.volume_config import VolumeConfig
from multimeditron.model.constants import MODALITY_VALUE_KEY, NUM_EMBEDDINGS_KEY


class VolumeProcessor(BaseModalityProcessor):
    """
    Processor for 3D medical volumes (CT, MRI).

    Pipeline:
        1. ndarray -> float32 tensor
        2. Trilinear resize to ``config.volume_size`` (D, H, W)
        3. Per-volume min-max normalize to [0, 1]

    Min-max to [0, 1] matches M3D-CLIP's training distribution. Both
    M3D-CLIP and DCFormer were pretrained on [0, 1] inputs; skipping this
    yields out-of-distribution features (the issue is distributional, not
    floating-point — bf16 has more than enough range for HU values).

    Output: float32 tensor (C, D, H, W) plus
    ``NUM_EMBEDDINGS_KEY = config.proj_out_num``.
    """

    def __init__(self, config: VolumeConfig):
        super().__init__(config)
        self._num_patches = config.proj_out_num

    def process(self, modality: Dict[str, Any]) -> Dict[str, Any]:
        processed = modality.copy()
        volume = modality[MODALITY_VALUE_KEY]

        if isinstance(volume, torch.Tensor):
            tensor = volume.float()
        else:
            tensor = torch.tensor(np.asarray(volume), dtype=torch.float32)

        d, h, w = self.config.volume_size
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=(d, h, w),
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

        vmin = tensor.min()
        vmax = tensor.max()
        denom = (vmax - vmin).clamp_min(1e-6)
        tensor = (tensor - vmin) / denom

        processed[MODALITY_VALUE_KEY] = tensor
        processed[NUM_EMBEDDINGS_KEY] = self._num_patches
        return processed
