import torch
import torch.jit

from typing import Optional, Tuple


def torch_get_default_tensor_type():
    return torch.empty(0).type()


def _torch_get_default_dtype() -> torch.dtype:
    '''A torchscript-compatible version of torch.get_default_dtype()'''
    return torch.empty(0).dtype


def torch_get_default_device() -> torch.device:
    return torch.empty(0).device


def explicit_default_types(dtype: Optional[torch.dtype], device: Optional[torch.device]) -> Tuple[torch.dtype, torch.device]:
    '''A torchscript-compatible type resolver'''
    if dtype is None:
        dtype = _torch_get_default_dtype()
    if device is None:
        device = torch_get_default_device()
    return dtype, device