"""
Tests for the 3D Volume Modality (Issue #21).

Run with::

    pytest tests/test_volume_3d.py -v

Coverage:
  * config, loader, processor, registry, freeze/unfreeze — offline,
    using a synthetic NIfTI generated in ``tmp_path``.
  * forward pass — mocks ``AutoModel.from_pretrained`` so CI does not
    download the ~800 MB M3D-CLIP weights.
  * an optional integration test that hits the real weights, gated by
    ``RUN_M3DCLIP_WEIGHTS_TESTS=1``.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

nib = pytest.importorskip("nibabel")


@pytest.fixture
def synthetic_nifti(tmp_path: Path) -> Path:
    arr = np.random.rand(64, 64, 32).astype(np.float32)
    nii = nib.Nifti1Image(arr, np.eye(4))
    out = tmp_path / "synth.nii.gz"
    nib.save(nii, str(out))
    return out


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def test_volume_config_defaults():
    from multimeditron.model.modalities.volume.volume_config import VolumeConfig

    cfg = VolumeConfig()
    assert cfg.modality_type == "3d_volume"
    assert cfg.hidden_size == 4096
    assert cfg.clip_name == "GoodBaiBai88/M3D-CLIP"
    assert cfg.trust_remote_code is True
    assert cfg.volume_size == (32, 256, 256)
    assert cfg.proj_out_num == 256
    assert cfg.num_channels == 1
    assert cfg.num_patches == 256


def test_volume_config_custom():
    from multimeditron.model.modalities.volume.volume_config import VolumeConfig

    cfg = VolumeConfig(volume_size=(64, 128, 128), proj_out_num=64)
    assert cfg.volume_size == (64, 128, 128)
    assert cfg.proj_out_num == 64
    assert cfg.num_patches == 64


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def test_volume_loader(synthetic_nifti: Path):
    from multimeditron.dataset.loader.volume.volume_loader import (
        FileSystemVolumeLoader,
    )

    loader = FileSystemVolumeLoader(base_path=str(synthetic_nifti.parent))
    sample = {"value": synthetic_nifti.name, "type": "3d_volume"}
    vol = loader.load(sample)

    assert isinstance(vol, np.ndarray)
    assert vol.dtype == np.float32
    assert vol.ndim == 4
    assert vol.shape[0] == 1


def test_volume_loader_missing_file(tmp_path: Path):
    from multimeditron.dataset.loader.volume.volume_loader import (
        FileSystemVolumeLoader,
    )

    loader = FileSystemVolumeLoader(base_path=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        loader.load({"value": "does_not_exist.nii.gz", "type": "3d_volume"})


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------

def test_volume_processor(synthetic_nifti: Path):
    from multimeditron.model.constants import (
        MODALITY_VALUE_KEY,
        NUM_EMBEDDINGS_KEY,
    )
    from multimeditron.model.modalities.volume.volume_config import VolumeConfig
    from multimeditron.model.modalities.volume.volume_processor import (
        VolumeProcessor,
    )

    cfg = VolumeConfig()
    proc = VolumeProcessor(cfg)

    arr = nib.load(str(synthetic_nifti)).get_fdata().astype(np.float32)[None]
    out = proc.process({MODALITY_VALUE_KEY: arr, "type": "3d_volume"})
    tensor = out[MODALITY_VALUE_KEY]

    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (cfg.num_channels, *cfg.volume_size)
    assert tensor.dtype == torch.float32
    assert tensor.min().item() >= 0.0
    assert tensor.max().item() <= 1.0
    assert out[NUM_EMBEDDINGS_KEY] == cfg.proj_out_num


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def test_registry():
    from multimeditron.dataset.loader import AutoModalityLoader
    from multimeditron.model.modalities.base import AutoModality

    assert "volume_3d" in AutoModality._registry
    assert "fs-volume" in AutoModalityLoader._registry


# ---------------------------------------------------------------------------
# Mocked-encoder forward / freeze tests
# ---------------------------------------------------------------------------

class _StubEncoder(torch.nn.Module):
    """Mimics M3D-CLIP's ``encode_image`` contract without real weights."""

    def __init__(self, hidden_size: int = 768, n_patches: int = 2048):
        super().__init__()
        self._hidden_size = hidden_size
        self._n_patches = n_patches
        self.config = MagicMock(hidden_size=hidden_size)
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def encode_image(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return torch.zeros(
            b, 1 + self._n_patches, self._hidden_size, dtype=x.dtype
        )


def _build_modality_with_stub():
    from multimeditron.model.modalities.volume import volume_modality as vm
    from multimeditron.model.modalities.volume.volume_config import VolumeConfig

    stub = _StubEncoder()
    with patch.object(vm, "AutoModel") as mock_auto:
        mock_auto.from_pretrained.return_value = stub
        cfg = VolumeConfig()
        mod = vm.VolumeModality(cfg)
    return cfg, mod, stub


def test_modality_forward_mocked():
    cfg, mod, _ = _build_modality_with_stub()
    mod.eval()

    x = torch.randn(2, cfg.num_channels, *cfg.volume_size)
    with torch.no_grad():
        out = mod([x[i] for i in range(2)])

    assert out.shape == (2, cfg.proj_out_num, cfg.hidden_size)


def test_freeze_unfreeze_mocked():
    _, mod, _ = _build_modality_with_stub()

    mod.freeze_modality_embedder()
    assert all(not p.requires_grad for p in mod.feature_extractor.parameters())

    mod.unfreeze_modality_embedder()
    assert all(p.requires_grad for p in mod.feature_extractor.parameters())

    mod.unfreeze_projection()
    assert all(p.requires_grad for p in mod.projector.parameters())


# ---------------------------------------------------------------------------
# Optional integration test — pulls real M3D-CLIP weights (~800 MB).
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    os.environ.get("RUN_M3DCLIP_WEIGHTS_TESTS") != "1",
    reason="Set RUN_M3DCLIP_WEIGHTS_TESTS=1 to enable (downloads ~800 MB).",
)
def test_modality_forward_real_weights(synthetic_nifti: Path):
    from multimeditron.dataset.loader.volume.volume_loader import (
        FileSystemVolumeLoader,
    )
    from multimeditron.model.constants import MODALITY_VALUE_KEY
    from multimeditron.model.modalities.volume.volume_config import VolumeConfig
    from multimeditron.model.modalities.volume.volume_modality import (
        VolumeModality,
    )
    from multimeditron.model.modalities.volume.volume_processor import (
        VolumeProcessor,
    )

    cfg = VolumeConfig()
    loader = FileSystemVolumeLoader(base_path=str(synthetic_nifti.parent))
    proc = VolumeProcessor(cfg)
    mod = VolumeModality(cfg).eval()

    raw = loader.load({"value": synthetic_nifti.name, "type": "3d_volume"})
    out = proc.process({MODALITY_VALUE_KEY: raw, "type": "3d_volume"})

    with torch.no_grad():
        result = mod([out[MODALITY_VALUE_KEY]])

    assert result.shape == (1, cfg.proj_out_num, cfg.hidden_size)
