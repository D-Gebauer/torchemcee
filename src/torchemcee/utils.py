# -*- coding: utf-8 -*-

from __future__ import annotations

import warnings
from functools import wraps
from typing import Any, Callable, Optional, Union

import torch

__all__ = [
    "deprecated",
    "resolve_device",
    "make_generator",
    "free_memory_bytes",
    "chain_bytes",
]

#: no good way to ask for free memory on CPU, so just assume 2 GB
CPU_MEMORY_CAP = 2 << 30


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def make_generator(seed: Optional[int], device: torch.device) -> torch.Generator:
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(int(seed))
    return generator


def free_memory_bytes(device: torch.device) -> int:
    if device.type == "cuda":
        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    return CPU_MEMORY_CAP


def chain_bytes(
    nsteps: int, ntargets: int, nwalkers: int, ndim: int, dtype: torch.dtype
) -> int:
    itemsize = torch.empty((), dtype=dtype).element_size()
    return int(nsteps) * int(ntargets) * int(nwalkers) * int(ndim) * itemsize


def deprecation_warning(msg: str) -> None:
    warnings.warn(msg, DeprecationWarning, stacklevel=3)


def deprecated(alternate: str) -> Callable[..., Any]:
    def wrapper(func: Callable[..., Any]) -> Callable[..., Any]:
        name = func.__name__

        @wraps(func)
        def f(*args: Any, **kwargs: Any) -> Any:
            deprecation_warning(
                f"The {name!r} attribute is deprecated. Use {alternate!r} instead."
            )
            return func(*args, **kwargs)

        return f

    return wrapper
