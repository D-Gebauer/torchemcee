# -*- coding: utf-8 -*-

"""Adapters for ``sbi`` potentials and ``zuko`` flows, needs the ``sbi`` extra"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch

__all__ = ["flow_log_prob_fn", "potential_to_log_prob_fn"]

LogProbFn = Callable[[torch.Tensor], torch.Tensor]


class _ConditionCache:
    """Caches ``x_o`` expanded along the walker axis

    ``n`` only takes a couple of different values during a run, so this
    avoids redoing the expand in every step.

    Args:
        x_o (torch.Tensor): The observations, with shape
            ``(ntargets, x_dim)``.

    """

    def __init__(self, x_o: torch.Tensor) -> None:
        if x_o.dim() == 1:
            x_o = x_o.unsqueeze(0)
        if x_o.dim() != 2:
            raise ValueError(
                f"x_o must have shape (ntargets, x_dim); got " f"{tuple(x_o.shape)}."
            )
        self.x_o = x_o
        self.ntargets = int(x_o.shape[0])
        self._cache: Dict[int, torch.Tensor] = {}

    def get(self, n: int) -> torch.Tensor:
        """Returns ``x_o`` repeated ``n`` times per target, as ``(B * n, x_dim)``"""
        cached = self._cache.get(n)
        if cached is None:
            cached = (
                self.x_o.unsqueeze(1)
                .expand(self.ntargets, n, self.x_o.shape[1])
                .reshape(self.ntargets * n, self.x_o.shape[1])
                .contiguous()
            )
            self._cache[n] = cached
        return cached


def flow_log_prob_fn(
    density_estimator: Any,
    x_o: torch.Tensor,
    prior: Optional[Any] = None,
) -> LogProbFn:
    """Build a log-probability function from a conditional flow

    This evaluates the flow for each walker conditioned on its own
    observation and adds the log-prior if one is given.

    Args:
        density_estimator (ConditionalDensityEstimator): The trained
            likelihood estimator.
        x_o (torch.Tensor): The observations, with shape
            ``(ntargets, x_dim)``. Row ``b`` conditions target ``b``.
        prior (Optional[Distribution]): A prior whose ``log_prob`` is added
            to the likelihood. Pass ``None`` to sample the likelihood alone.

    Returns:
        callable: A function mapping ``(ntargets, n, ndim)`` to
        ``(ntargets, n)``, suitable for :class:`EnsembleSampler`.

    """
    cache = _ConditionCache(x_o)

    def log_prob_fn(theta: torch.Tensor) -> torch.Tensor:
        ntargets, n, ndim = theta.shape
        if ntargets != cache.ntargets:
            raise ValueError(
                f"theta has {ntargets} targets but x_o has " f"{cache.ntargets}."
            )
        flat = theta.reshape(ntargets * n, ndim)
        condition = cache.get(n).unsqueeze(0)  # (1, B * n, x_dim)
        raw: torch.Tensor = density_estimator.log_prob(input=condition, condition=flat)
        log_prob = raw.reshape(ntargets, n)
        if prior is not None:
            log_prob = log_prob + prior.log_prob(flat).reshape(ntargets, n)
        return log_prob

    return log_prob_fn


def potential_to_log_prob_fn(
    potential_fn: Any,
    x_o: torch.Tensor,
    *,
    track_gradients: bool = False,
) -> LogProbFn:
    """Build a log-probability function from an ``sbi`` potential

    ``x_o`` is expanded along the walker axis and set with
    ``x_is_iid=False``, so each theta is paired with its own x. The potential
    already includes the prior.

    Args:
        potential_fn (BasePotential): The potential, for instance from
            ``sbi.inference.likelihood_estimator_based_potential``.
        x_o (torch.Tensor): The observations, with shape
            ``(ntargets, x_dim)``. Row ``b`` conditions target ``b``.
        track_gradients (Optional[bool]): Whether the potential tracks
            gradients. Sampling does not need them. (default: ``False``)

    Returns:
        callable: A function mapping ``(ntargets, n, ndim)`` to
        ``(ntargets, n)``, suitable for :class:`EnsembleSampler`.

    """
    cache = _ConditionCache(x_o)

    def log_prob_fn(theta: torch.Tensor) -> torch.Tensor:
        ntargets, n, ndim = theta.shape
        if ntargets != cache.ntargets:
            raise ValueError(
                f"theta has {ntargets} targets but x_o has " f"{cache.ntargets}."
            )
        flat = theta.reshape(ntargets * n, ndim)
        potential_fn.set_x(cache.get(n), x_is_iid=False)
        out: torch.Tensor = potential_fn(flat, track_gradients=track_gradients)
        return out.reshape(ntargets, n)

    return log_prob_fn
