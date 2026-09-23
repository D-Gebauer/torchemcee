# -*- coding: utf-8 -*-

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple

import torch

from ..state import State

__all__ = ["Move"]

#: coords (ntargets, n, ndim) -> (log_prob (ntargets, n), blobs or None)
ComputeLogProb = Callable[[torch.Tensor], Tuple[torch.Tensor, Optional[torch.Tensor]]]


class Move(ABC):
    """The abstract base class for moves

    Unlike in emcee there is no model object, ``propose`` just gets the
    function that computes the log-probability.

    """

    @abstractmethod
    def propose(
        self,
        state: State,
        compute_log_prob: ComputeLogProb,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[State, torch.Tensor]:
        """Advance the ensemble by one sweep

        Args:
            state (State): The current state of the ensemble. This is updated
                in place.
            compute_log_prob (callable): Maps positions with shape
                ``(ntargets, n, ndim)`` to a tuple of the log-probability
                with shape ``(ntargets, n)`` and the blobs, or ``None``.
            generator (Optional[torch.Generator]): The generator used for
                every random draw in the move.

        Returns:
            tuple: The updated state and a boolean acceptance mask with
            shape ``(ntargets, nwalkers)``.

        """

    def tune(self, state: State, accepted: torch.Tensor) -> None:
        pass

    @staticmethod
    def _accept(
        state: State,
        active: torch.Tensor,
        proposal: torch.Tensor,
        new_log_prob: torch.Tensor,
        factors: torch.Tensor,
        new_blobs: Optional[torch.Tensor],
        generator: Optional[torch.Generator],
    ) -> torch.Tensor:
        """Apply the Metropolis-Hastings rule to a subset of the walkers

        Args:
            state (State): The state to update in place.
            active (torch.Tensor): The walker indices being updated, with
                shape ``(n,)``.
            proposal (torch.Tensor): The proposed positions, with shape
                ``(ntargets, n, ndim)``.
            new_log_prob (torch.Tensor): The log-probability at the
                proposal, with shape ``(ntargets, n)``.
            factors (torch.Tensor): The log of the proposal-density ratio,
                with shape ``(ntargets, n)``.
            new_blobs (Optional[torch.Tensor]): The blobs at the proposal.
            generator (Optional[torch.Generator]): The generator for the
                acceptance draw.

        Returns:
            torch.Tensor: The acceptance mask, with shape
            ``(ntargets, n)``.

        """
        dtype, device = state.coords.dtype, state.coords.device
        current = state.log_prob[:, active]
        log_accept = factors + new_log_prob.to(dtype) - current
        # nan can only come from -inf - -inf, just count that as a rejection
        log_accept = torch.nan_to_num(log_accept, nan=-float("inf"))

        threshold = torch.log(
            torch.rand(
                log_accept.shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
        )
        accept = threshold < log_accept

        state.coords[:, active] = torch.where(
            accept.unsqueeze(-1), proposal, state.coords[:, active]
        )
        state.log_prob[:, active] = torch.where(accept, new_log_prob.to(dtype), current)
        if new_blobs is not None and state.blobs is not None:
            mask = accept.reshape(accept.shape + (1,) * (new_blobs.dim() - 2))
            state.blobs[:, active] = torch.where(
                mask, new_blobs, state.blobs[:, active]
            )
        return accept
