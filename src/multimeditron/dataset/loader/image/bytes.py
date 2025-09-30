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
    """
    Loader for raw image bytes.
    Expects the sample dictionary to have a "value" key containing a dictionary with a "bytes" key holding the raw image bytes.
    Example:
        loader = RawImageLoader()
        sample = {"value": {"bytes": b'...'}, "type": "image"}
        image = loader.load(sample)
        # image is a PIL Image object
    """

    def __init__(self):
        """
        Initializes the RawImageLoader.
        """

        super().__init__()

    def load(self, sample: Dict[str, Any]) -> PIL.Image.Image:
        """
        Load an image from raw bytes.
        Args:
            sample (Dict[str, Any]): A dictionary containing at least the "value" key with a dictionary that has a "bytes" key holding the raw image bytes.
        Returns:
            PIL.Image.Image: The loaded image as a PIL Image object.
        """

        image_bytes = sample["value"]["bytes"]
        image = PIL.Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return image
