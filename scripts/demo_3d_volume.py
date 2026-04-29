"""
Minimal demo: load M3D-CLIP via VolumeModality and run a synthetic volume.

Run:
    python scripts/demo_3d_volume.py

This downloads ~800 MB on first run (the M3D-CLIP weights). The test
suite (tests/test_volume_3d.py) covers the same path with a mocked
encoder for offline / CI use.
"""

from __future__ import annotations

import numpy as np
import torch

from multimeditron.model.constants import MODALITY_VALUE_KEY
from multimeditron.model.modalities.volume.volume_config import VolumeConfig
from multimeditron.model.modalities.volume.volume_modality import VolumeModality
from multimeditron.model.modalities.volume.volume_processor import VolumeProcessor


def main() -> None:
    cfg = VolumeConfig()
    print(f"loading {cfg.clip_name}")

    proc = VolumeProcessor(cfg)
    mod = VolumeModality(cfg).eval()

    raw = np.random.rand(1, 64, 64, 32).astype(np.float32)
    print(f"synthetic input: {raw.shape}")

    processed = proc.process({MODALITY_VALUE_KEY: raw, "type": "3d_volume"})
    tensor = processed[MODALITY_VALUE_KEY]
    print(f"after processor: {tuple(tensor.shape)} dtype={tensor.dtype}")

    with torch.no_grad():
        embeds = mod([tensor])
    print(f"after modality:  {tuple(embeds.shape)} dtype={embeds.dtype}")
    print(f"expected:        (1, {cfg.proj_out_num}, {cfg.hidden_size})")


if __name__ == "__main__":
    main()
