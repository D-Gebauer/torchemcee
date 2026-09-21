# -*- coding: utf-8 -*-

"""The affine-invariant stretch move"""

from __future__ import annotations

from typing import Any, List, Optional, Tuple

import torch

from .red_blue import RedBlueMove

__all__ = ["StretchMove"]


class StretchMove(RedBlueMove):
    """The affine-invariant stretch move of `Goodman & Weare (2010)
    <https://msp.org/camcos/2010/5-1/p04.xhtml>`_

    Each walker is moved along the line to a random walker of the
    complement, by a factor ``z`` drawn from ``g(z) ∝ 1/sqrt(z)`` on
    ``[1/a, a]``.

    Args:
        a (Optional[float]): The stretch scale. (default: ``2.0``)
        **kwargs: Passed to :class:`RedBlueMove`.

    Raises:
        ValueError: If ``a`` is not greater than one.

    """

    def __init__(self, a: float = 2.0, **kwargs: Any) -> None:
        if a <= 1.0:
            raise ValueError(f"stretch scale a must be > 1; got {a}.")
        self.a = float(a)
        super().__init__(**kwargs)

    def get_proposal(
        self,
        s: torch.Tensor,
        c: List[torch.Tensor],
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Stretch each walker towards a random member of the complement"""
        c_cat = torch.cat(c, dim=1)
        ntargets, ns, ndim = s.shape
        nc = c_cat.shape[1]
        device, dtype = s.device, s.dtype

        u = torch.rand((ntargets, ns), generator=generator, device=device, dtype=dtype)
        zz = ((self.a - 1.0) * u + 1.0) ** 2.0 / self.a
        factors = (ndim - 1.0) * torch.log(zz)

        rint = torch.randint(nc, (ntargets, ns), generator=generator, device=device)
        partners = torch.gather(c_cat, 1, rint.unsqueeze(-1).expand(ntargets, ns, ndim))
        return partners - (partners - s) * zz.unsqueeze(-1), factors
