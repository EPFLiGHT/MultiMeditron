from abc import ABC, abstractmethod
from typing import Any
import os

class AbstractModalityLoader(ABC):
    @abstractmethod
    @property
    def modality_type(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def load(self, *args, **kwargs) -> Any:
        raise NotImplementedError

class AutoModalityLoader(ABC):
    ...
