from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from multimeditron.model.constants import MODALITY_VALUE_KEY, NUM_EMBEDDINGS_KEY
from multimeditron.model.modalities.base import BaseModalityProcessor


class VolumeProcessor(BaseModalityProcessor):
    """
    Processor for 3D volumes.

    Responsibilities:
    - Accept numpy arrays from loaders
    - Normalize shape to (C, D, H, W)
    - Cast to float
    - Resize to target volume_size
    - Min-max normalize to [0, 1]
    """

    def __init__(self, config: Any):
        super().__init__(config)

        if not hasattr(config, "volume_size"):
            raise ValueError("VolumeProcessor requires config.volume_size")
        if len(config.volume_size) != 3:
            raise ValueError(
                f"config.volume_size must be a 3-tuple (D, H, W), got: {config.volume_size}"
            )
        self.volume_size: Tuple[int, int, int] = tuple(int(x) for x in config.volume_size)

        # Number of multimodal tokens inserted into prompt.
        # For volume models this is typically set by config (e.g. after pooling).
        self._num_embeddings = int(getattr(config, "proj_out_num", 1))
        if self._num_embeddings <= 0:
            raise ValueError("config.proj_out_num must be > 0")

    def _to_chw3d(self, arr: np.ndarray) -> np.ndarray:
        """
        Convert array to channel-first (C, D, H, W).

        Accepted inputs:
        - (D, H, W): single-channel volume
        - (C, D, H, W): channel-first volume
        - (D, H, W, C): channel-last volume (C must be small, e.g. 1/3/4)
        """
        if arr.ndim == 3:
            return arr[None, ...]

        if arr.ndim != 4:
            raise ValueError(
                f"Expected 3D or 4D volume array, got shape {arr.shape} (ndim={arr.ndim})"
            )

        c_first = arr.shape[0] in (1, 3, 4)
        c_last = arr.shape[-1] in (1, 3, 4)

        if c_first and not c_last:
            return arr
        if c_last and not c_first:
            return np.moveaxis(arr, -1, 0)
        if c_first and c_last:
            # Ambiguous case like (1, D, H, 1): keep as channel-first by convention.
            return arr

        raise ValueError(
            "For 4D input, expected channel dimension to be first or last "
            f"with size in (1,3,4), got shape {arr.shape}"
        )

    def process(self, modality: Dict[str, Any]) -> Dict[str, Any]:
        processed = modality.copy()
        volume = modality.get(MODALITY_VALUE_KEY, None)

        if not isinstance(volume, np.ndarray):
            raise ValueError(
                f"VolumeProcessor expects numpy.ndarray at '{MODALITY_VALUE_KEY}', got {type(volume)}"
            )

        volume = self._to_chw3d(volume)
        tensor = torch.from_numpy(volume).float()  # (C, D, H, W)

        if not torch.isfinite(tensor).all():
            raise ValueError("Volume contains non-finite values (NaN or Inf)")

        # Resize in 3D: interpolate expects (N, C, D, H, W).
        tensor = F.interpolate(
            tensor.unsqueeze(0),
            size=self.volume_size,
            mode="trilinear",
            align_corners=False,
        ).squeeze(0)

        # Min-max normalize each volume to [0, 1].
        v_min = torch.amin(tensor)
        v_max = torch.amax(tensor)
        if float(v_max - v_min) > 0:
            tensor = (tensor - v_min) / (v_max - v_min)
        else:
            tensor = torch.zeros_like(tensor)

        processed[MODALITY_VALUE_KEY] = tensor
        processed[NUM_EMBEDDINGS_KEY] = self._num_embeddings
        return processed
