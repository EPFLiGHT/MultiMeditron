import io
import os
import pathlib
from typing import Any, Dict, Union

import numpy as np

from multimeditron.dataset.loader import AutoModalityLoader, BaseModalityLoader
from multimeditron.model.constants import MODALITY_VALUE_KEY


def _load_npy_from_path(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Volume file {path} not found")
    if not path.lower().endswith(".npy"):
        raise ValueError(f"Expected a .npy volume file, got: {path}")

    arr = np.load(path, allow_pickle=False)
    if not isinstance(arr, np.ndarray):
        raise ValueError(f"Loaded object from {path} is not a numpy array")
    return arr


def _load_npy_from_bytes(raw_bytes: bytes) -> np.ndarray:
    arr = np.load(io.BytesIO(raw_bytes), allow_pickle=False)
    if not isinstance(arr, np.ndarray):
        raise ValueError("Loaded object from raw bytes is not a numpy array")
    return arr


@AutoModalityLoader.register("fs-volume")
class FileSystemVolumeLoader(BaseModalityLoader):
    """
    Loader for volume files from the filesystem.
    Expects sample["value"] to be a relative path under base_path.
    Only .npy files are supported.
    """

    def __init__(self, base_path: Union[str, pathlib.Path]):
        super().__init__()
        self.base_path = base_path

    def load(self, sample: Dict[str, Any]) -> np.ndarray:
        value = sample.get(MODALITY_VALUE_KEY)
        if not isinstance(value, str):
            raise ValueError(
                f"Expected '{MODALITY_VALUE_KEY}' to be a relative path string for fs-volume"
            )
        volume_path = os.path.join(self.base_path, value)
        return _load_npy_from_path(volume_path)


@AutoModalityLoader.register("raw-volume")
class RawVolumeLoader(BaseModalityLoader):
    """
    Loader for in-memory/raw volumes.
    Supports either:
      - numpy.ndarray directly, or
      - dict-like Arrow payload with {"bytes": <raw .npy bytes>}.
    """

    def __init__(self):
        super().__init__()

    def load(self, sample: Dict[str, Any]) -> np.ndarray:
        value = sample.get(MODALITY_VALUE_KEY)

        if isinstance(value, np.ndarray):
            return value

        if isinstance(value, dict) and "bytes" in value:
            raw_bytes = value["bytes"]
            if not isinstance(raw_bytes, (bytes, bytearray)):
                raise ValueError(
                    "Expected raw-volume bytes payload to be 'bytes' or 'bytearray'"
                )
            return _load_npy_from_bytes(bytes(raw_bytes))

        raise ValueError(
            "Unsupported raw-volume value. Expected numpy.ndarray or {'bytes': ...} payload."
        )
