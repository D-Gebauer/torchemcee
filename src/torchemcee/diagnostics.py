# -*- coding: utf-8 -*-

"""Convergence diagnostics, batched over targets

Everything here takes a chain with shape ``(nsteps, ntargets, nwalkers,
ndim)`` and returns one value per target.
"""

from __future__ import annotations

from typing import Optional

import torch

from .autocorr import AutocorrError
from .autocorr import integrated_time as _integrated_time

__all__ = [
    "AutocorrError",
    "acceptance_fraction",
    "integrated_autocorr_time",
    "split_rhat",
    "effective_sample_size",
    "summarize",
]


def acceptance_fraction(accepted: torch.Tensor, nsteps: int) -> torch.Tensor:
    """Estimate the acceptance fraction of every walker

    This should be roughly between 0.2 and 0.5. Values close to zero usually
    mean stuck walkers.

    Args:
        accepted (torch.Tensor): The cumulative acceptance counts, with
            shape ``(ntargets, nwalkers)``.
        nsteps (int): The number of moves those counts accumulated over.

    Returns:
        torch.Tensor: The acceptance fraction per walker, with shape
        ``(ntargets, nwalkers)``.

    Raises:
        ValueError: If ``nsteps`` is not positive.

    """
    if nsteps < 1:
        raise ValueError(f"nsteps must be >= 1; got {nsteps}.")
    return accepted.to(torch.float64) / float(nsteps)


def integrated_autocorr_time(
    chain: torch.Tensor,
    *,
    c: float = 5.0,
    tol: float = 50.0,
    quiet: bool = False,
) -> torch.Tensor:
    """Estimate the integrated autocorrelation time of a batch of chains

    This just calls :func:`torchemcee.autocorr.integrated_time`.

    Args:
        chain (torch.Tensor): The chain, with shape
            ``(nsteps, ntargets, nwalkers, ndim)``.
        c (Optional[float]): The step size for the window search.
            (default: ``5.0``)
        tol (Optional[float]): The minimum number of autocorrelation times
            needed to trust the estimate. (default: ``50.0``)
        quiet (Optional[bool]): If ``True``, log a warning instead of
            raising when the chain is too short. (default: ``False``)

    Returns:
        torch.Tensor: The integrated autocorrelation time per target and
        parameter, with shape ``(ntargets, ndim)``.

    Raises:
        ValueError: If the chain does not have four dimensions.
        AutocorrError: If the chain is shorter than ``tol`` times the
            estimated autocorrelation time for any target and parameter,
            unless ``quiet`` is set.

    """
    return _integrated_time(chain, c=c, tol=tol, quiet=quiet)


def split_rhat(chain: torch.Tensor) -> torch.Tensor:
    """Estimate the split-R-hat of a batch of chains

    Each walker is treated as one chain and split in half. The walkers of an
    ensemble aren't independent, so take this with a grain of salt and also
    look at the autocorrelation time.

    Args:
        chain (torch.Tensor): The chain, with shape
            ``(nsteps, ntargets, nwalkers, ndim)``.

    Returns:
        torch.Tensor: R-hat per target and parameter, with shape
        ``(ntargets, ndim)``.

    Raises:
        ValueError: If the chain does not have four dimensions or is shorter
            than four steps.

    """
    if chain.dim() != 4:
        raise ValueError(
            "chain must have shape (nsteps, ntargets, nwalkers, ndim); got "
            f"{tuple(chain.shape)}."
        )
    nsteps, ntargets, nwalkers, ndim = chain.shape
    if nsteps < 4:
        raise ValueError(f"need at least 4 steps for split-R-hat; got {nsteps}.")

    half = nsteps // 2
    x = chain.to(torch.float64)[: 2 * half]
    # (2 * half, B, nwalkers, ndim) -> (half, B, 2 * nwalkers, ndim)
    x = torch.stack((x[:half], x[half:]), dim=2).reshape(
        half, ntargets, 2 * nwalkers, ndim
    )

    chain_mean = x.mean(dim=0)  # (B, 2W, ndim)
    chain_var = x.var(dim=0, unbiased=True)  # (B, 2W, ndim)

    w = chain_var.mean(dim=1)  # within-chain
    b = chain_mean.var(dim=1, unbiased=True) * half  # between-chain
    var_hat = (half - 1) / half * w + b / half
    return torch.sqrt(var_hat / w.clamp_min(torch.finfo(torch.float64).tiny))


def effective_sample_size(
    chain: torch.Tensor,
    *,
    c: float = 5.0,
    tol: float = 50.0,
    quiet: bool = True,
) -> torch.Tensor:
    """Estimate the effective sample size of a batch of chains

    Args:
        chain (torch.Tensor): The chain, with shape
            ``(nsteps, ntargets, nwalkers, ndim)``.
        c (Optional[float]): The step size for the window search.
            (default: ``5.0``)
        tol (Optional[float]): The minimum number of autocorrelation times
            needed to trust the estimate. (default: ``50.0``)
        quiet (Optional[bool]): If ``True``, log a warning instead of
            raising when the chain is too short. (default: ``True``)

    Returns:
        torch.Tensor: ``nsteps * nwalkers / tau`` per target and parameter,
        with shape ``(ntargets, ndim)``.

    """
    nsteps, _ntargets, nwalkers, _ndim = chain.shape
    tau = integrated_autocorr_time(chain, c=c, tol=tol, quiet=quiet)
    return (nsteps * nwalkers) / tau.clamp_min(torch.finfo(torch.float64).tiny)


def summarize(
    chain: torch.Tensor,
    accepted: torch.Tensor,
    *,
    c: float = 5.0,
    tol: float = 50.0,
    rhat_threshold: float = 1.05,
    min_acceptance: float = 0.05,
    nsteps: Optional[int] = None,
) -> dict:
    """Compute every diagnostic in one pass and flag the failed targets

    Handy when running many targets, ``failed`` gives the indices of the ones
    that didn't converge.

    Args:
        chain (torch.Tensor): The chain, with shape
            ``(nsteps, ntargets, nwalkers, ndim)``.
        accepted (torch.Tensor): The cumulative acceptance counts, with
            shape ``(ntargets, nwalkers)``.
        c (Optional[float]): The step size for the window search.
            (default: ``5.0``)
        tol (Optional[float]): The minimum number of autocorrelation times
            needed to trust a target's estimate. (default: ``50.0``)
        rhat_threshold (Optional[float]): The largest R-hat a converged
            target may have. (default: ``1.05``)
        min_acceptance (Optional[float]): The smallest mean acceptance
            fraction a converged target may have. (default: ``0.05``)
        nsteps (Optional[int]): The number of moves ``accepted`` was counted
            over, i.e. ``sampler.iteration``. (default: the chain length)

    Returns:
        dict: ``tau``, ``rhat`` and ``ess``, each with shape
        ``(ntargets, ndim)``; ``acceptance`` with shape
        ``(ntargets, nwalkers)``; ``converged`` with shape ``(ntargets,)``;
        and ``failed``, the indices of the targets that did not converge.

    """
    nstored = chain.shape[0]
    tau = integrated_autocorr_time(chain, c=c, tol=tol, quiet=True)
    rhat = split_rhat(chain)
    ess = (chain.shape[0] * chain.shape[2]) / tau.clamp_min(
        torch.finfo(torch.float64).tiny
    )
    acc = acceptance_fraction(accepted, nstored if nsteps is None else nsteps)

    converged = (
        (rhat < rhat_threshold).all(dim=-1)
        & (tol * tau <= nstored).all(dim=-1)
        & (acc.mean(dim=-1) > min_acceptance)
    )
    return {
        "tau": tau,
        "rhat": rhat,
        "ess": ess,
        "acceptance": acc,
        "converged": converged,
        "failed": torch.nonzero(~converged).flatten(),
    }
