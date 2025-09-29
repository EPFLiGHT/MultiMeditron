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
    def __init__(self, base_path: Union[str, pathlib.Path]):
        self.base_path = base_path

    def load(self, sample: Dict[str, Any]) -> np.ndarray:
        image_path = os.path.join(self.base_path, sample["value"])

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file {image_path} not found")
        
        # Load png/jpg/jpeg images
        image = PIL.Image.open(image_path).convert("RGB")
        return image