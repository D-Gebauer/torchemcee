# -*- coding: utf-8 -*-

"""The state of the ensemble"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

__all__ = ["State"]


@dataclass
class State:
    """The state of the ensemble during a run

    There is always a leading target axis, a normal single run just has
    ``ntargets == 1``.

    Args:
        coords (torch.Tensor): The current positions of the walkers,
            with shape ``(ntargets, nwalkers, ndim)``.
        log_prob (torch.Tensor): The log-probability at ``coords``, with
            shape ``(ntargets, nwalkers)``.
        accepted (torch.Tensor): The cumulative number of accepted proposals
            per walker as ``int64``, with shape ``(ntargets, nwalkers)``.
        blobs (Optional[torch.Tensor]): The metadata returned alongside the
            log-probability, with leading axes ``(ntargets, nwalkers)``, or
            ``None`` when the log-probability function returns no blobs.
            (default: ``None``)
        random_state (Optional[torch.Tensor]): The state of the sampler's
            generator at this step. (default: ``None``)
        step (int): The number of moves completed since the last call to
            :func:`EnsembleSampler.reset`. (default: ``0``)

    """

    coords: torch.Tensor
    log_prob: torch.Tensor
    accepted: torch.Tensor
    blobs: Optional[torch.Tensor] = None
    random_state: Optional[torch.Tensor] = None
    step: int = 0

    @property
    def ntargets(self) -> int:
        """int: The number of target distributions"""
        return int(self.coords.shape[0])

    @property
    def nwalkers(self) -> int:
        """int: The number of walkers per target"""
        return int(self.coords.shape[1])

    @property
    def ndim(self) -> int:
        """int: The number of parameters"""
        return int(self.coords.shape[2])

    def validate(self, *, allow_nonfinite: bool = False) -> None:
        """Check the shapes and that everything is finite

        A walker that starts at ``-inf`` gives ``-inf - (-inf) = nan`` in the
        acceptance ratio and then basically never moves again, so by default
        this raises.

        Args:
            allow_nonfinite (Optional[bool]): If ``True``, permit walkers
                with a non-finite log-probability. (default: ``False``)

        Raises:
            ValueError: If the shapes are inconsistent, ``nwalkers`` is odd,
                the coordinates are not finite, or any log-probability is
                not finite (unless ``allow_nonfinite`` is set).

        """
        if self.coords.dim() != 3:
            raise ValueError(
                "coords must have shape (ntargets, nwalkers, ndim); got "
                f"{tuple(self.coords.shape)}."
            )
        expected = (self.ntargets, self.nwalkers)
        if tuple(self.log_prob.shape) != expected:
            raise ValueError(
                f"log_prob must have shape {expected}; got "
                f"{tuple(self.log_prob.shape)}."
            )
        if tuple(self.accepted.shape) != expected:
            raise ValueError(
                f"accepted must have shape {expected}; got "
                f"{tuple(self.accepted.shape)}."
            )
        if self.nwalkers % 2 != 0:
            raise ValueError(f"nwalkers must be even; got {self.nwalkers}.")
        if not torch.isfinite(self.coords).all():
            raise ValueError("coords contains non-finite values.")

        if allow_nonfinite:
            return

        bad = ~torch.isfinite(self.log_prob)
        if bool(bad.any()):
            n_bad = int(bad.sum())
            targets = torch.nonzero(bad.any(dim=1)).flatten().tolist()
            walkers = torch.nonzero(bad)[:8].tolist()
            raise ValueError(
                f"{n_bad} of {self.log_prob.numel()} walkers have a "
                "non-finite initial log-probability and would get stuck. "
                f"Affected targets: {targets[:8]}"
                f"{' ...' if len(targets) > 8 else ''}; "
                f"first bad (target, walker) pairs: {walkers}. "
                "Start them inside the prior, e.g. with "
                "torchemcee.ball(..., log_prob_fn=...), or pass "
                "allow_nonfinite=True to run anyway."
            )

    @property
    def has_blobs(self) -> bool:
        """bool: Whether this state carries blobs"""
        return self.blobs is not None

    def clone(self) -> "State":
        """Returns a deep copy of this state"""
        return State(
            coords=self.coords.clone(),
            log_prob=self.log_prob.clone(),
            accepted=self.accepted.clone(),
            blobs=None if self.blobs is None else self.blobs.clone(),
            random_state=self.random_state,
            step=self.step,
        )

    def to(
        self,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "State":
        """Returns a copy of this state on another device or in another dtype

        Args:
            device (Optional[torch.device]): The target device, or ``None``
                to keep the current one.
            dtype (Optional[torch.dtype]): The target dtype for the floating
                point tensors, or ``None`` to keep the current one. The
                acceptance counts always stay integral.

        Returns:
            State: The converted state.

        """
        return State(
            coords=self.coords.to(device=device, dtype=dtype),
            log_prob=self.log_prob.to(device=device, dtype=dtype),
            accepted=self.accepted.to(device=device),
            blobs=(None if self.blobs is None else self.blobs.to(device=device)),
            random_state=self.random_state,
            step=self.step,
        )
