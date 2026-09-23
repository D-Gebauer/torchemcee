# -*- coding: utf-8 -*-

from __future__ import annotations

from abc import abstractmethod
from typing import List, Optional, Tuple

import torch

from ..state import State
from .move import ComputeLogProb, Move

__all__ = ["RedBlueMove"]


class RedBlueMove(Move):
    """An abstract red-blue ensemble move with parallelization as described in
    `Foreman-Mackey et al. (2013) <https://arxiv.org/abs/1202.3665>`_

    The ensemble is split into ``nsplits`` sub-ensembles and each one is
    updated using only the others, so a whole sub-ensemble is a single
    batched call. The split is only done along the walker axis, so the
    targets never mix.

    Args:
        nsplits (Optional[int]): The number of sub-ensembles to use. In
            general, higher values give better performance per proposal but
            lower parallelism. (default: ``2``)
        randomize_split (Optional[bool]): Randomly shuffle walkers between
            sub-ensembles at every step. (default: ``True``)

    Raises:
        ValueError: If ``nsplits`` is smaller than two.

    """

    def __init__(self, nsplits: int = 2, randomize_split: bool = True) -> None:
        self.nsplits = int(nsplits)
        if self.nsplits < 2:
            raise ValueError(f"nsplits must be >= 2; got {self.nsplits}.")
        self.randomize_split = bool(randomize_split)

    def setup(self, coords: torch.Tensor) -> None:
        pass

    @abstractmethod
    def get_proposal(
        self,
        s: torch.Tensor,
        c: List[torch.Tensor],
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Propose new positions for one sub-ensemble

        Args:
            s (torch.Tensor): The positions of the sub-ensemble being
                updated, with shape ``(ntargets, ns, ndim)``.
            c (list): The positions of the complementary sub-ensembles, each
                with shape ``(ntargets, nc, ndim)``.
            generator (Optional[torch.Generator]): The generator for the
                random draws.

        Returns:
            tuple: The proposed positions, with shape
            ``(ntargets, ns, ndim)``, and the log of the proposal-density
            ratio, with shape ``(ntargets, ns)``.

        """

    def propose(
        self,
        state: State,
        compute_log_prob: ComputeLogProb,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[State, torch.Tensor]:
        """Advance the ensemble by one sweep over the sub-ensembles"""
        # Check that the dimensions are compatible.
        coords = state.coords
        ntargets, nwalkers, _ndim = coords.shape
        if nwalkers < 2 * self.nsplits:
            raise ValueError(
                f"nwalkers must be at least 2 * nsplits = "
                f"{2 * self.nsplits} for this move; got {nwalkers}."
            )
        device = coords.device

        # Run any move-specific setup.
        self.setup(coords)

        accepted_mask = torch.zeros(
            (ntargets, nwalkers), dtype=torch.bool, device=device
        )

        # Split the ensemble and iterate over the sub-ensembles.
        order = torch.arange(nwalkers, device=device)
        if self.randomize_split:
            order = order[torch.randperm(nwalkers, generator=generator, device=device)]
        splits = [order[k :: self.nsplits] for k in range(self.nsplits)]

        for k in range(self.nsplits):
            active = splits[k]
            complement = [splits[j] for j in range(self.nsplits) if j != k]

            s = coords[:, active]
            c = [coords[:, idx] for idx in complement]

            # Get the move-specific proposal.
            proposal, factors = self.get_proposal(s, c, generator)

            # Compute the lnprobs of the proposed position.
            new_log_prob, new_blobs = compute_log_prob(proposal)

            accept = self._accept(
                state,
                active,
                proposal,
                new_log_prob,
                factors,
                new_blobs,
                generator,
            )
            accepted_mask[:, active] = accept

        state.accepted += accepted_mask.to(state.accepted.dtype)
        state.step += 1
        return state, accepted_mask
