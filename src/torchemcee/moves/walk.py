# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from .red_blue import RedBlueMove

__all__ = ["WalkMove"]


class WalkMove(RedBlueMove):
    """A `Goodman & Weare (2010)
    <https://msp.org/camcos/2010/5-1/p04.xhtml>`_ "walk move" with
    parallelization as described in `Foreman-Mackey et al. (2013)
    <https://arxiv.org/abs/1202.3665>`_

    The proposal is a Gaussian with the covariance of a random subset of the
    complement.

    Args:
        s (Optional[int]): The number of complementary walkers used to
            estimate the covariance. By default the whole complement is
            used, which is also the fastest since the covariance is then
            only computed once.
        **kwargs: Passed to :class:`RedBlueMove`.

    """

    def __init__(self, s: Optional[int] = None, **kwargs: Any) -> None:
        self.s = s
        super().__init__(**kwargs)

    def get_proposal(
        self,
        s: torch.Tensor,
        c: List[torch.Tensor],
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        c_cat = torch.cat(c, dim=1)
        ntargets, ns, ndim = s.shape
        nc = c_cat.shape[1]
        device, dtype = s.device, s.dtype

        if self.s is None:
            subset = c_cat.unsqueeze(1).expand(ntargets, ns, nc, ndim)
            nsub = nc
        else:
            nsub = int(self.s)
            if not 2 <= nsub <= nc:
                raise ValueError(
                    f"s must be between 2 and the complement size {nc}; " f"got {nsub}."
                )
            # random subset for each walker, without replacement
            order = torch.argsort(
                torch.rand(
                    (ntargets, ns, nc),
                    generator=generator,
                    device=device,
                    dtype=dtype,
                ),
                dim=-1,
            )[..., :nsub]
            subset = torch.gather(
                c_cat.unsqueeze(1).expand(ntargets, ns, nc, ndim),
                2,
                order.unsqueeze(-1).expand(ntargets, ns, nsub, ndim),
            )

        centred = subset - subset.mean(dim=2, keepdim=True)
        cov = torch.einsum("bnki,bnkj->bnij", centred, centred) / (nsub - 1)
        # small ridge so the Cholesky doesn't fail for degenerate subsets
        eye = torch.eye(ndim, device=device, dtype=dtype)
        jitter = 1e-10 * torch.diagonal(cov, dim1=-2, dim2=-1).mean(
            dim=-1, keepdim=True
        ).clamp_min(torch.finfo(dtype).tiny)
        chol = torch.linalg.cholesky(cov + jitter.unsqueeze(-1) * eye)

        noise = torch.randn(
            (ntargets, ns, ndim, 1),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        proposal = s + (chol @ noise).squeeze(-1)
        factors = torch.zeros((ntargets, ns), device=device, dtype=dtype)
        return proposal, factors
