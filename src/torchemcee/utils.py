# -*- coding: utf-8 -*-

"""Some small helpers"""

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
    "next_pow_two",
]

#: no good way to ask for free memory on CPU, so just assume 2 GB
CPU_MEMORY_CAP = 2 << 30


def resolve_device(device: Optional[Union[str, torch.device]] = None) -> torch.device:
    """Resolve a device specification to a concrete ``torch.device``

    Args:
        device (Optional): A device, a device string, or ``None`` to select
            CUDA when it is available and CPU otherwise.

    Returns:
        torch.device: The resolved device.

    """
    if device is None:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def make_generator(seed: Optional[int], device: torch.device) -> torch.Generator:
    """Build a device-local random number generator

    Args:
        seed (Optional[int]): The seed, or ``None`` to seed from entropy.
        device (torch.device): The device the generator draws on.

    Returns:
        torch.Generator: The seeded generator.

    """
    generator = torch.Generator(device=device)
    if seed is None:
        generator.seed()
    else:
        generator.manual_seed(int(seed))
    return generator


def free_memory_bytes(device: torch.device) -> int:
    """Return the free memory on a device in bytes

    Args:
        device (torch.device): The device to query. On CPU a fixed cap is
            returned.

    Returns:
        int: The number of free bytes.

    """
    if device.type == "cuda":
        free, _total = torch.cuda.mem_get_info(device)
        return int(free)
    return CPU_MEMORY_CAP


def chain_bytes(
    nsteps: int, ntargets: int, nwalkers: int, ndim: int, dtype: torch.dtype
) -> int:
    """Return the number of bytes a chain of the given shape would occupy

    Args:
        nsteps (int): Number of stored steps.
        ntargets (int): Number of target distributions.
        nwalkers (int): Number of walkers per target.
        ndim (int): Number of parameters.
        dtype (torch.dtype): The storage dtype.

    Returns:
        int: The size of the chain in bytes.

    """
    itemsize = torch.empty((), dtype=dtype).element_size()
    return int(nsteps) * int(ntargets) * int(nwalkers) * int(ndim) * itemsize


def next_pow_two(n: int) -> int:
    """Returns the next power of two greater than or equal to `n`"""
    i = 1
    while i < int(n):
        i = i << 1
    return i


def deprecation_warning(msg: str) -> None:
    """Issue a ``DeprecationWarning`` from the caller's frame"""
    warnings.warn(msg, DeprecationWarning, stacklevel=3)


def deprecated(alternate: str) -> Callable[..., Any]:
    """Mark an attribute as deprecated in favour of ``alternate``

    Args:
        alternate (str): The name to use instead.

    Returns:
        callable: A decorator.

    """

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
