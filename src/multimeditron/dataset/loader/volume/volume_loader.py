from __future__ import annotations

import os
import pathlib
from typing import Any, Dict, Union

import numpy as np

from multimeditron.dataset.loader import BaseModalityLoader, AutoModalityLoader
from multimeditron.model.constants import MODALITY_VALUE_KEY


@AutoModalityLoader.register("fs-volume")
class FileSystemVolumeLoader(BaseModalityLoader):
    """
    Loader for 3D medical volume files (NIfTI format) from the filesystem.

    Expects the sample dictionary to have a "value" key containing the path to a NIfTI file (.nii or .nii.gz). 
    Returns a numpy array of shape (C, D, H, W).

    Example:

    .. code-block:: python

        loader = FileSystemVolumeLoader(base_path="/path/to/volumes")
        sample = {"value": "scan.nii.gz", "type": "3d_volume"}
        volume = loader.load(sample)
        # volume is a numpy array of shape (1, D, H, W)
    """

    def __init__(self, base_path: Union[str, pathlib.Path] = ""):
        """
        Args:
            base_path (Union[str, pathlib.Path]): The base directory where volume files are stored.
                Defaults to empty string (absolute paths in dataset).
        """
        super().__init__()
        self.base_path = base_path

    def load(self, sample: Dict[str, Any]) -> np.ndarray:
        """
        Load a 3D medical volume from the filesystem.

        Args:
            sample (Dict[str, Any]): A dictionary containing at least the "value" key
                with the path to the NIfTI file.

        Returns:
            np.ndarray: The loaded volume as a float32 array of shape (C, D, H, W),
                where C=1 for single-channel scans (CT/MRI).
        """
        try:
            import nibabel as nib
        except ImportError:
            raise ImportError(
                "nibabel is required to load NIfTI files. "
                "Install it with: pip install nibabel"
            )

        volume_path = os.path.join(self.base_path, sample[MODALITY_VALUE_KEY])

        if not os.path.exists(volume_path):
            raise FileNotFoundError(f"Volume file {volume_path} not found")

        nii = nib.load(volume_path)
        volume = nii.get_fdata().astype(np.float32)

        # Ensure shape is (C, D, H, W) — add channel dim if needed
        if volume.ndim == 3:
            volume = volume[np.newaxis, ...]  # (D, H, W) → (1, D, H, W)
        elif volume.ndim == 4:
            # NIfTI 4D volumes store channels last: (D, H, W, C)
            # Only transpose if last dim is small (likely channels, not spatial)
            if volume.shape[-1] <= 4:
                volume = np.transpose(volume, (3, 0, 1, 2))  # (D, H, W, C) → (C, D, H, W)
            # else: already (C, D, H, W), leave as-is
        else:
            raise ValueError(
                f"Expected 3D (D,H,W) or 4D (D,H,W,C) volume, got shape {volume.shape}"
            )

        return volume
