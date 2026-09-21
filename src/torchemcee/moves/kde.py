# -*- coding: utf-8 -*-

"""A proposal using a kernel density estimate of the complement"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from .red_blue import RedBlueMove

__all__ = ["KDEMove"]


class KDEMove(RedBlueMove):
    """A proposal using a kernel density estimate of the complementary
    sub-ensemble

    Uses Scott's rule for the bandwidth by default, like
    ``scipy.stats.gaussian_kde``. The proposal isn't symmetric, so its
    density goes into the acceptance ratio.

    Args:
        bw_method (Optional): The bandwidth rule, ``"scott"`` or
            ``"silverman"``, or a scalar factor applied to the covariance.
            (default: ``"scott"``)
        **kwargs: Passed to :class:`RedBlueMove`.

    Raises:
        ValueError: If the bandwidth rule is not recognized.

    """

    def __init__(self, bw_method: Any = None, **kwargs: Any) -> None:
        self.bw_method = bw_method
        super().__init__(**kwargs)

    def _bandwidth_factor(self, n: int, ndim: int) -> float:
        method = self.bw_method if self.bw_method is not None else "scott"
        if method == "scott":
            return float(float(n) ** (-1.0 / (ndim + 4)))
        if method == "silverman":
            return float((n * (ndim + 2.0) / 4.0) ** (-1.0 / (ndim + 4)))
        if isinstance(method, (int, float)):
            return float(method)
        raise ValueError(
            f"{self.bw_method!r} is not a recognized bandwidth rule; "
            "expected 'scott', 'silverman' or a number."
        )

    @staticmethod
    def _log_pdf(
        x: torch.Tensor, centres: torch.Tensor, chol: torch.Tensor
    ) -> torch.Tensor:
        """The log density of a Gaussian mixture centred on ``centres``

        Args:
            x (torch.Tensor): The evaluation points, ``(B, n, ndim)``.
            centres (torch.Tensor): The kernel centres, ``(B, nc, ndim)``.
            chol (torch.Tensor): The Cholesky factor of the bandwidth
                matrix, ``(B, ndim, ndim)``.

        Returns:
            torch.Tensor: The log density, ``(B, n)``.

        """
        ndim = x.shape[-1]
        nc = centres.shape[1]
        diff = x.unsqueeze(2) - centres.unsqueeze(1)  # (B, n, nc, ndim)
        solved = torch.linalg.solve_triangular(
            chol.unsqueeze(1), diff.transpose(-1, -2), upper=False
        )  # (B, n, ndim, nc)
        maha = (solved**2).sum(dim=-2)  # (B, n, nc)
        log_det = torch.log(torch.diagonal(chol, dim1=-2, dim2=-1)).sum(-1)  # (B,)
        norm = 0.5 * ndim * torch.log(torch.tensor(2.0 * torch.pi)) + log_det.unsqueeze(
            -1
        )
        return torch.logsumexp(-0.5 * maha - norm.unsqueeze(-1), dim=-1) - (
            torch.log(torch.tensor(float(nc)))
        )

    def get_proposal(
        self,
        s: torch.Tensor,
        c: List[torch.Tensor],
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Draw an independent sample from a KDE of the complement"""
        c_cat = torch.cat(c, dim=1)
        ntargets, ns, ndim = s.shape
        nc = c_cat.shape[1]
        device, dtype = s.device, s.dtype

        centred = c_cat - c_cat.mean(dim=1, keepdim=True)
        cov = torch.einsum("bki,bkj->bij", centred, centred) / (nc - 1)
        cov = cov * self._bandwidth_factor(nc, ndim) ** 2
        eye = torch.eye(ndim, device=device, dtype=dtype)
        jitter = 1e-10 * torch.diagonal(cov, dim1=-2, dim2=-1).mean(dim=-1).clamp_min(
            torch.finfo(dtype).tiny
        )
        chol = torch.linalg.cholesky(cov + jitter.reshape(ntargets, 1, 1) * eye)

        # pick a random centre and add noise with the kernel covariance
        pick = torch.randint(nc, (ntargets, ns), generator=generator, device=device)
        centres = torch.gather(c_cat, 1, pick.unsqueeze(-1).expand(ntargets, ns, ndim))
        noise = torch.randn(
            (ntargets, ns, ndim, 1),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        proposal = centres + (chol.unsqueeze(1) @ noise).squeeze(-1)

        factors = self._log_pdf(s, c_cat, chol) - self._log_pdf(proposal, c_cat, chol)
        return proposal, factors
