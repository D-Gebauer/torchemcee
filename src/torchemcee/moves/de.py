# -*- coding: utf-8 -*-

"""The differential evolution move"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from .red_blue import RedBlueMove

__all__ = ["DEMove"]


class DEMove(RedBlueMove):
    """A proposal using differential evolution

    This is the `Ter Braak (2006)
    <https://doi.org/10.1007/s11222-006-8769-1>`_ proposal with the
    modification to the scaling of gamma suggested by `Nelson et al. (2013)
    <https://doi.org/10.1088/0067-0049/210/1/11>`_.

    Args:
        sigma (Optional[float]): The standard deviation of the Gaussian used
            to scale gamma. (default: ``1.0e-5``)
        gamma0 (Optional[float]): The mean stretch factor for the proposal
            vector. By default this is ``2.38 / sqrt(2 ndim)``, as
            recommended by the reference.
        **kwargs: Passed to :class:`RedBlueMove`.

    """

    def __init__(
        self,
        sigma: float = 1.0e-5,
        gamma0: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        self.sigma = float(sigma)
        self.gamma0 = gamma0
        self.g0: float = 0.0
        super().__init__(**kwargs)

    def setup(self, coords: torch.Tensor) -> None:
        """Set gamma0, the default depends on ``ndim``"""
        if self.gamma0 is None:
            ndim = coords.shape[2]
            self.g0 = 2.38 / float(2 * ndim) ** 0.5
        else:
            self.g0 = float(self.gamma0)

    def get_proposal(
        self,
        s: torch.Tensor,
        c: List[torch.Tensor],
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Displace each walker along a difference vector of the complement"""
        c_cat = torch.cat(c, dim=1)
        ntargets, ns, ndim = s.shape
        nc = c_cat.shape[1]
        device, dtype = s.device, s.dtype
        if nc < 2:
            raise ValueError(
                "DEMove needs at least two walkers in the complement; got " f"{nc}."
            )

        # two different walkers from the complement: draw the second one
        # from nc - 1 slots and shift it past the first
        first = torch.randint(nc, (ntargets, ns), generator=generator, device=device)
        second = torch.randint(
            nc - 1, (ntargets, ns), generator=generator, device=device
        )
        second = second + (second >= first).to(second.dtype)

        idx = lambda i: torch.gather(  # noqa: E731
            c_cat, 1, i.unsqueeze(-1).expand(ntargets, ns, ndim)
        )
        diffs = idx(first) - idx(second)

        gamma = self.g0 * (
            1.0
            + self.sigma
            * torch.randn(
                (ntargets, ns, 1),
                generator=generator,
                device=device,
                dtype=dtype,
            )
        )
        proposal = s + gamma * diffs
        factors = torch.zeros((ntargets, ns), device=device, dtype=dtype)
        return proposal, factors
