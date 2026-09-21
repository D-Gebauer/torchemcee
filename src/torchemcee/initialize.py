# -*- coding: utf-8 -*-

"""Helpers for setting up the initial walker positions"""

from __future__ import annotations

from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch

from .state import State
from .utils import resolve_device

__all__ = ["from_prior", "ball", "check_initial_state"]

LogProbFn = Callable[[torch.Tensor], torch.Tensor]


def from_prior(
    sample_fn: Callable[[int], torch.Tensor],
    nwalkers: int,
    ntargets: int = 1,
    *,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Draw the initial positions from a prior

    Args:
        sample_fn (callable): A function taking a number of samples ``n``
            and returning positions with shape ``(n, ndim)``. The ``sample``
            method of a ``torch.distributions.Distribution`` works directly.
        nwalkers (int): The number of walkers per target.
        ntargets (Optional[int]): The number of targets. (default: ``1``)
        device (Optional): The device to place the positions on.
            (default: CUDA when available, else CPU)
        dtype (Optional[torch.dtype]): The dtype of the positions.
            (default: ``torch.float32``)

    Returns:
        torch.Tensor: The initial positions, with shape
        ``(ntargets, nwalkers, ndim)``.

    Raises:
        ValueError: If ``sample_fn`` does not return a two-dimensional
            tensor with the requested number of rows.

    """
    device = resolve_device(device)
    n = int(ntargets) * int(nwalkers)
    samples = torch.as_tensor(sample_fn(n))
    if samples.dim() != 2 or samples.shape[0] != n:
        raise ValueError(
            f"sample_fn({n}) must return shape ({n}, ndim); got "
            f"{tuple(samples.shape)}."
        )
    ndim = samples.shape[1]
    return (
        samples.to(device=device, dtype=dtype)
        .reshape(ntargets, nwalkers, ndim)
        .contiguous()
    )


def ball(
    center: Union[torch.Tensor, np.ndarray],
    scale: Union[float, torch.Tensor, np.ndarray],
    nwalkers: int,
    *,
    log_prob_fn: Optional[LogProbFn] = None,
    max_resample: int = 100,
    device: Optional[Union[str, torch.device]] = None,
    dtype: torch.dtype = torch.float32,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Draw the initial positions in a Gaussian ball around a point

    If ``log_prob_fn`` is given, walkers with a non-finite log-probability
    (e.g. outside of the prior) are redrawn up to ``max_resample`` times.

    Args:
        center (torch.Tensor or numpy.ndarray): The centre of the ball, with
            shape ``(ndim,)`` for one target or ``(ntargets, ndim)`` for a
            separate centre per target.
        scale (float, torch.Tensor or numpy.ndarray): The standard deviation
            of the ball, either a scalar or per parameter with shape
            ``(ndim,)``.
        nwalkers (int): The number of walkers per target.
        log_prob_fn (Optional[callable]): The log-probability function used
            to reject invalid draws. When it is ``None`` no check is made.
        max_resample (Optional[int]): The maximum number of resampling
            rounds. (default: ``100``)
        device (Optional): The device to place the positions on.
            (default: CUDA when available, else CPU)
        dtype (Optional[torch.dtype]): The dtype of the positions.
            (default: ``torch.float32``)
        generator (Optional[torch.Generator]): The generator for the draws.

    Returns:
        torch.Tensor: The initial positions, with shape
        ``(ntargets, nwalkers, ndim)``.

    Raises:
        ValueError: If the shapes are inconsistent, or if walkers remain
            invalid after ``max_resample`` rounds.

    """
    device = resolve_device(device)
    center_t = torch.as_tensor(center, device=device, dtype=dtype)
    if center_t.dim() == 1:
        center_t = center_t.unsqueeze(0)
    if center_t.dim() != 2:
        raise ValueError(
            "center must have shape (ndim,) or (ntargets, ndim); got "
            f"{tuple(center_t.shape)}."
        )
    ntargets, ndim = center_t.shape

    scale_t = torch.as_tensor(scale, device=device, dtype=dtype)
    if scale_t.dim() == 0:
        scale_t = scale_t.expand(ndim)
    if tuple(scale_t.shape) != (ndim,):
        raise ValueError(
            f"scale must be a scalar or have shape ({ndim},); got "
            f"{tuple(scale_t.shape)}."
        )

    def draw(shape: Tuple[int, ...]) -> torch.Tensor:
        noise = torch.randn(shape, generator=generator, device=device, dtype=dtype)
        return center_t.unsqueeze(1) + noise * scale_t

    coords = draw((ntargets, nwalkers, ndim))
    if log_prob_fn is None:
        return coords

    invalid = ~torch.isfinite(log_prob_fn(coords))
    for _ in range(int(max_resample)):
        if not bool(invalid.any()):
            return coords
        candidate = draw((ntargets, nwalkers, ndim))
        coords = torch.where(invalid.unsqueeze(-1), candidate, coords)
        invalid = ~torch.isfinite(log_prob_fn(coords))

    if bool(invalid.any()):
        n_bad = int(invalid.sum())
        targets = torch.nonzero(invalid.any(dim=1)).flatten().tolist()
        raise ValueError(
            f"{n_bad} walkers still have a non-finite log-probability after "
            f"{max_resample} resampling rounds (affected targets: "
            f"{targets[:8]}{' ...' if len(targets) > 8 else ''}). The centre "
            "is probably outside the support, or the scale is far too large "
            "for it."
        )
    return coords


def check_initial_state(state: State, *, raise_on_invalid: bool = True) -> torch.Tensor:
    """Report the walkers with a non-finite initial log-probability

    Args:
        state (State): The state to check.
        raise_on_invalid (Optional[bool]): Raise instead of returning the
            indices when any walker is invalid. (default: ``True``)

    Returns:
        torch.Tensor: The ``(target, walker)`` indices of the invalid
        walkers, with shape ``(n_invalid, 2)``.

    Raises:
        ValueError: If any walker is invalid and ``raise_on_invalid`` is
            set.

    """
    if raise_on_invalid:
        state.validate()
        return torch.zeros((0, 2), dtype=torch.long, device=state.coords.device)
    return torch.nonzero(~torch.isfinite(state.log_prob))
