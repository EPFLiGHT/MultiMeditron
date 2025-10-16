from typing import Union
# from typing import Any, Optional
import torch
# import functools
# import logging

def get_torch_dtype(dtype: Union[torch.dtype, str]) -> torch.dtype:
    if not isinstance(dtype, torch.dtype):
        dtype = getattr(torch, dtype)
        assert isinstance(dtype, torch.dtype)
    return dtype

# _global_cache = dict()
# class _GlobalCache:
#     @staticmethod
#     def _is_in_cache(key: str) -> bool:
#         return key in _global_cache
#
#     @staticmethod
#     def _update_cache(key: str, value: Any) -> Optional[Any]:
#         o = None
#         if key in _global_cache:
#             o = _global_cache[key]
#         _global_cache[key] = value
#         return o
#
#     @staticmethod
#     def _set_cache(key: str, value: Any) -> bool:
#         if key in _global_cache:
#             return False
#         else:
#             _global_cache[key] = value
#             return True
#
# def print_warning_once(message: str, logger: Optional[logging.Logger] = None):
#     message = str(message)
#     if _GlobalCache._set_cache(message, 0):
#         if logger is not None:
#             logger.warning(message)
#         else:
#             logging.warning(message, stacklevel=2)
