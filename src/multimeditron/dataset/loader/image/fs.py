import os
from typing import Dict, Any, Union
from multimeditron.dataset.loader import BaseModalityLoader, AutoModalityLoader
import pathlib
import numpy as np
import PIL
import warnings

warnings.simplefilter("error", PIL.Image.DecompressionBombWarning)

@AutoModalityLoader.register("fs-image")
class FileSystemImageLoader(BaseModalityLoader):
    """
    Loader for image files from the filesystem.
    Expects the sample dictionary to have a "value" key containing the relative path to the image file.
    The base_path parameter specifies the root directory where images are stored.
    Example:
        loader = FileSystemImageLoader(base_path="/path/to/images")
        sample = {"value": "image1.jpg", "type": "image"}
        image = loader.load(sample)
        # image is a PIL Image object
    """

    def __init__(self, base_path: Union[str, pathlib.Path]):
        """
        Args:
            base_path (Union[str, pathlib.Path]): The base directory where image files are stored.
        """
        super().__init__()
        self.base_path = base_path

    def load(self, sample: Dict[str, Any]) -> PIL.Image.Image:
        """
        Load an image from the filesystem.
        Args:
            sample (Dict[str, Any]): A dictionary containing at least the "value" key with the relative path to the image file.
        Returns:
            PIL.Image.Image: The loaded image as a PIL Image object.
        """
        image_path = os.path.join(self.base_path, sample["value"])

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found")
        
        # Load png/jpg/jpeg images
        image = PIL.Image.open(image_path).convert("RGB")
        return image
