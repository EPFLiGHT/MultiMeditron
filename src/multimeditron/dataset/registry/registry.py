import abc
from typing import Dict, Any
import os
import numpy as np
import logging
from multimeditron.utils import print_warning_once


logger = logging.getLogger(__name__)

class ModalityRegistry(abc.ABC):
    registry_type: str

    @abc.abstractmethod
    def check_sample(self, sample: Dict[str, Any]) -> bool:
        ...

    @abc.abstractmethod
    def get_modality(self, modality: Dict[str, Any]) -> Any:
        ...

    def __exit__(self, exception_type, exception_value, exception_traceback):
        pass

    def __enter__(self):
        return self
            

def get_registry(registry_type: str) -> ModalityRegistry:
    from multimeditron.dataset.registry.fs_registry import FileSystemImageRegistry
    from multimeditron.dataset.registry.wids_registry import WIDSImageRegistry

    match registry_type:
        case "wids":
            return WIDSImageRegistry
        
        case "fs":
            return FileSystemImageRegistry

        case _:
            print_warning_once(
                f"Unrecognized registry type {registry_type}, current legacy behavior is to use FileSystemImageRegistry. This will raise an error in future versions."
            )
            return FileSystemImageRegistry
            # Future versions:
            # raise ValueError(f"Unrecognized registry type {registry_type}")

