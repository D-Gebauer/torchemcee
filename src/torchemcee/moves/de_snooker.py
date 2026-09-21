# -*- coding: utf-8 -*-

"""The snooker differential evolution move"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from .red_blue import RedBlueMove

__all__ = ["DESnookerMove"]


class DESnookerMove(RedBlueMove):
    """A snooker proposal using differential evolution

    Based on `Ter Braak & Vrugt (2008)
    <https://doi.org/10.1007/s11222-008-9104-9>`_, following the
    implementation in emcee.

    This always uses four sub-ensembles, since it needs three complements.

    Args:
        gammas (Optional[float]): The mean stretch factor for the proposal
            vector. (default: ``1.7``)
        **kwargs: Passed to :class:`RedBlueMove`.

    """

    def __init__(self, gammas: float = 1.7, **kwargs: Any) -> None:
        self.gammas = float(gammas)
        kwargs["nsplits"] = 4
        super().__init__(**kwargs)

    def get_proposal(
        self,
        s: torch.Tensor,
        c: List[torch.Tensor],
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Project a complement difference onto the snooker direction"""
        ntargets, ns, ndim = s.shape
        device, dtype = s.device, s.dtype

        # one walker from each of the three other splits, in random order
        picks = []
        for cj in c[:3]:
            nc = cj.shape[1]
            i = torch.randint(nc, (ntargets, ns), generator=generator, device=device)
            picks.append(
                torch.gather(cj, 1, i.unsqueeze(-1).expand(ntargets, ns, ndim))
            )
        w = torch.stack(picks, dim=2)  # (B, ns, 3, ndim)
        shuffle = torch.argsort(
            torch.rand(
                (ntargets, ns, 3),
                generator=generator,
                device=device,
                dtype=dtype,
            ),
            dim=2,
        )
        w = torch.gather(w, 2, shuffle.unsqueeze(-1).expand(ntargets, ns, 3, ndim))
        z, z1, z2 = w[:, :, 0], w[:, :, 1], w[:, :, 2]

        delta = s - z
        norm = torch.linalg.norm(delta, dim=-1, keepdim=True)
        norm = norm.clamp_min(torch.finfo(dtype).tiny)
        u = delta / torch.sqrt(norm)

        projection = (u * z1).sum(-1, keepdim=True) - (u * z2).sum(-1, keepdim=True)
        proposal = s + u * self.gammas * projection

        new_norm = torch.linalg.norm(proposal - z, dim=-1)
        factors = (0.5 * (ndim - 1.0)) * (
            torch.log(new_norm.clamp_min(torch.finfo(dtype).tiny))
            - torch.log(norm.squeeze(-1))
        )
        return proposal, factors
