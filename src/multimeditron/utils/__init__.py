import enum
from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema
from typing import Any, Optional
import torch
import functools
import logging

def get_torch_dtype(dtype: torch.dtype | str) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        dtype = getattr(torch, dtype)
        assert isinstance(dtype, torch.dtype)
    return dtype

def pydantic_enum[E: enum.Enum](enum_cls: type[E]) -> type[E]:
    def __str__(self) -> str:
        return self.name.lower().replace('_', '-')

    def __get_pydantic_core_schema__(cls: type[E], source_type: Any, handler: GetCoreSchemaHandler):
        assert source_type is cls
        
        def get_enum(value: Any, validate_next: core_schema.ValidatorFunctionWrapHandler):
            if isinstance(value, cls):
                return value
            else:
                name: str = validate_next(value)
                name = name.upper().replace('-', '_')
                return enum_cls[name]

        def serialize(enum: E):
            return enum.name.lower().replace('_', '-')
        
        expected = [member.name.lower().replace('_', '-') for member in cls]
        name_schema = core_schema.literal_schema(expected)
        
        return core_schema.no_info_wrap_validator_function(
            get_enum, name_schema, 
            ref=cls.__name__,
            serialization=core_schema.plain_serializer_function_ser_schema(serialize)
        )
    
    setattr(enum_cls, '__str__', __str__)
    setattr(enum_cls, '__get_pydantic_core_schema__', classmethod(__get_pydantic_core_schema__))
    return enum_cls

_global_cache = dict()
class _GlobalCache:
    @staticmethod
    def _is_in_cache(key: str) -> bool:
        return key in _global_cache

    @staticmethod
    def _update_cache(key: str, value: Any) -> Optional[Any]:
        o = None
        if key in _global_cache:
            o = _global_cache[key]
        _global_cache[key] = value
        return o
    
    @staticmethod
    def _set_cache(key: str, value: Any) -> bool:
        if key in _global_cache:
            return False
        else:
            _global_cache[key] = value
            return True

def print_warning_once(message: str, logger: Optional[logging.Logger] = None):
    message = str(message)
    if _GlobalCache._set_cache(message, 0):
        if logger is not None:
            logger.warning(message)
        else:
            logging.warning(message, stacklevel=2)