# -*- coding: utf-8 -*-

"""A general Metropolis-Hastings move"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import torch

from ..state import State
from .move import ComputeLogProb, Move

__all__ = ["MHMove"]

#: (coords, generator) -> (new coords, log ratio of the proposal densities)
ProposalFunction = Callable[
    [torch.Tensor, Optional[torch.Generator]],
    Tuple[torch.Tensor, torch.Tensor],
]


class MHMove(Move):
    """A general Metropolis-Hastings proposal

    All walkers are updated at once and independently of each other, so this
    is not affine invariant.

    Args:
        proposal_function (callable): The proposal, taking the current
            positions and a generator and returning the proposed positions
            and the log of the proposal-density ratio. It is a ratio of one
            (a zero log-ratio) for a symmetric proposal.
        ndim (Optional[int]): If given, the dimension of the parameter
            space, checked against the state at every proposal.

    """

    def __init__(
        self,
        proposal_function: ProposalFunction,
        ndim: Optional[int] = None,
    ) -> None:
        self.ndim = ndim
        self.get_proposal = proposal_function

    def propose(
        self,
        state: State,
        compute_log_prob: ComputeLogProb,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[State, torch.Tensor]:
        """Advance every walker by one Metropolis-Hastings step"""
        # Check to make sure that the dimensions match.
        ntargets, nwalkers, ndim = state.coords.shape
        if self.ndim is not None and self.ndim != ndim:
            raise ValueError(
                f"dimension mismatch in proposal: the move expects "
                f"{self.ndim}, the state has {ndim}."
            )

        # Get the move-specific proposal.
        proposal, factors = self.get_proposal(state.coords, generator)

        # Compute the lnprobs of the proposed position.
        new_log_prob, new_blobs = compute_log_prob(proposal)

        active = torch.arange(nwalkers, device=state.coords.device)
        accepted = self._accept(
            state,
            active,
            proposal,
            new_log_prob,
            factors,
            new_blobs,
            generator,
        )

        state.accepted += accepted.to(state.accepted.dtype)
        state.step += 1
        return state, accepted
