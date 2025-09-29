import os
from typing import Dict, Any, Union
from multimeditron.dataset.loader import BaseModalityLoader, AutoModalityLoader
import pathlib
import numpy as np
import PIL
import io
import warnings

warnings.simplefilter("error", PIL.Image.DecompressionBombWarning)

@AutoModalityLoader.register("raw-image")
class RawImageLoader(BaseModalityLoader):
    def __init__(self):
        super().__init__()

    def load(self, sample: Dict[str, Any]) -> np.ndarray:
        image_bytes = sample["value"]["bytes"]
        image = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image