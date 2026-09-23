# -*- coding: utf-8 -*-

from __future__ import annotations

import logging
from typing import Union

import torch

__all__ = [
    "AutocorrError",
    "next_pow_two",
    "function_1d",
    "auto_window",
    "integrated_time",
]

logger = logging.getLogger(__name__)


class AutocorrError(Exception):
    """Raised if the chain is too short to estimate an autocorrelation time

    The current estimate of the autocorrelation time can be accessed via the
    ``tau`` attribute of this exception.

    """

    def __init__(self, tau: torch.Tensor, *args: object) -> None:
        self.tau = tau
        super().__init__(*args)


def next_pow_two(n: Union[int, float]) -> int:
    """Returns the next power of two greater than or equal to `n`"""
    i = 1
    while i < int(n):
        i = i << 1
    return i


def function_1d(x: torch.Tensor) -> torch.Tensor:
    """Estimate the normalized autocorrelation function of a time series

    Unlike in emcee, this takes the last axis as time and broadcasts over all
    the others.

    Args:
        x (torch.Tensor): The series, with time along the last axis.

    Returns:
        torch.Tensor: The autocorrelation function, the same shape as ``x``.

    """
    n = next_pow_two(x.shape[-1])

    # Compute the FFT and then (from that) the auto-correlation function
    centred = x - x.mean(dim=-1, keepdim=True)
    f = torch.fft.rfft(centred, n=2 * n, dim=-1)
    acf = torch.fft.irfft(f * torch.conj(f), n=2 * n, dim=-1)
    trimmed: torch.Tensor = acf[..., : x.shape[-1]]
    tiny = float(torch.finfo(trimmed.dtype).tiny)
    return trimmed / trimmed[..., :1].clamp_min(tiny)


def auto_window(taus: torch.Tensor, c: float) -> torch.Tensor:
    nsteps = taus.shape[-1]
    m = torch.arange(nsteps, device=taus.device, dtype=taus.dtype)
    below = m < c * taus
    any_window = below.logical_not().any(dim=-1)
    return torch.where(
        any_window,
        below.logical_not().to(torch.long).argmax(dim=-1),
        torch.full_like(any_window, nsteps - 1, dtype=torch.long),
    )


def integrated_time(
    x: torch.Tensor,
    c: float = 5.0,
    tol: float = 50.0,
    quiet: bool = False,
) -> torch.Tensor:
    """Estimate the integrated autocorrelation time of a batch of chains

    This uses the iterative procedure described on page 16 of `Sokal's notes
    <https://www.semanticscholar.org/paper/Monte-Carlo-Methods-in-Statistical-Mechanics%3A-and-Sokal/0bfe9e3db30605fe2d4d26e1a288a5e2997e7225>`_
    to determine a reasonable window size. Same as in emcee, the
    autocorrelation function is computed per walker and then averaged.

    Args:
        x (torch.Tensor): The chain. Four dimensions are interpreted as
            ``(nsteps, ntargets, nwalkers, ndim)``; three as the
            single-target ``(nsteps, nwalkers, ndim)``.
        c (Optional[float]): The step size for the window search.
            (default: ``5.0``)
        tol (Optional[float]): The minimum number of autocorrelation times
            needed to trust the estimate. (default: ``50.0``)
        quiet (Optional[bool]): If ``True``, log a warning instead of
            raising when the chain is too short. (default: ``False``)

    Returns:
        torch.Tensor: The autocorrelation time per target and parameter,
        with shape ``(ntargets, ndim)``.

    Raises:
        ValueError: If the chain does not have three or four dimensions.
        AutocorrError: If the chain is shorter than ``tol`` times the
            estimated autocorrelation time for any target and parameter,
            unless ``quiet`` is set.

    """
    if x.dim() == 3:
        x = x.unsqueeze(1)
    if x.dim() != 4:
        raise ValueError(
            "the chain must have shape (nsteps, ntargets, nwalkers, ndim) "
            f"or (nsteps, nwalkers, ndim); got {tuple(x.shape)}."
        )
    nsteps, ntargets, _nwalkers, ndim = x.shape

    chain = x.to(torch.float64).permute(1, 3, 2, 0)  # (B, ndim, W, nsteps)
    acf = function_1d(chain).mean(dim=-2)  # (B, ndim, nsteps)
    taus = 2.0 * torch.cumsum(acf, dim=-1) - 1.0

    window = auto_window(taus, c)
    tau = torch.gather(taus, -1, window.unsqueeze(-1)).squeeze(-1)

    # Check convergence
    too_short = tol * tau > nsteps
    if bool(too_short.any()):
        worst = float(tau.max())
        msg = (
            f"The chain is shorter than {tol:g} times the integrated "
            f"autocorrelation time for {int(too_short.sum())} of "
            f"{ntargets * ndim} (target, parameter) pairs. The longest tau "
            f"is {worst:.1f}, so at least {int(tol * worst)} steps are "
            f"needed and the chain has {nsteps}."
        )
        if quiet:
            logger.warning(msg)
        else:
            raise AutocorrError(tau, msg + " Pass quiet=True to ignore this.")
    return tau
