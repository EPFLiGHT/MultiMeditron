from abc import ABC
from transformers import PreTrainedModel

class AbstractProjector(ABC, PreTrainedModel):
    model_type: str

    def __init__(self, modality_size: int, projected_size: int):
        super().__init__()
        self.modality_size = modality_size
        self.projected_size = projected_size


_PROJECTOR_LOADER_REGISTRY = {}
class AutoProjectorLoader:
    def __init__(self):
        raise RuntimeError("AutoProjectorLoader should not be instantiated directly. Please use the 'from_model_type' method.")

    @classmethod
    def register(name: str):
        def decorator(cls):
            if not issubclass(cls, AbstractProjector):
                raise ValueError(f"Class {cls.__name__} must inherit from AutoProjectorLoader to be registered.")

            model_type = cls.model_type
            if model_type in _PROJECTOR_LOADER_REGISTRY:
                raise ValueError(f"Modality type '{model_type}' is already registered.")

            setattr(cls, "name", name)
            _PROJECTOR_LOADER_REGISTRY[model_type] = cls

            return cls
        return decorator
    
    @classmethod
    def from_model_type(model_type: str, *args, **kwargs) -> AbstractProjector:
        if model_type not in _PROJECTOR_LOADER_REGISTRY:
            raise ValueError(f"Modality type '{model_type}' is not registered.")

        loader_class = _PROJECTOR_LOADER_REGISTRY[model_type]
        instance = loader_class(*args, **kwargs)
        setattr(instance, "model_type", model_type)

        return instance
