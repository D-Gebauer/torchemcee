# -*- coding: utf-8 -*-

"""A Metropolis-Hastings move with a Gaussian proposal"""

from __future__ import annotations

import math
from typing import Optional, Tuple, Union

import numpy as np
import torch

from .mh import MHMove

__all__ = ["GaussianMove"]


class GaussianMove(MHMove):
    """A Metropolis step with a Gaussian proposal function

    Args:
        cov: The covariance of the proposal function. This can be a scalar,
            a vector of length ``ndim`` for a diagonal covariance, or a full
            ``(ndim, ndim)`` matrix.
        mode (Optional): ``"vector"`` updates all dimensions simultaneously,
            ``"random"`` updates one randomly chosen dimension per walker,
            and ``"sequential"`` cycles through the dimensions. The latter
            two are only valid for a scalar or diagonal covariance.
            (default: ``"vector"``)
        factor (Optional[float]): If given, the proposal is made with a
            standard deviation multiplied by ``exp(U(-log(factor),
            log(factor)))``, which is invalid in ``"vector"`` mode with a
            full covariance matrix. Must be at least one.

    Raises:
        ValueError: If the covariance, the mode or the factor is invalid.

    """

    allowed_modes = ("vector", "random", "sequential")

    def __init__(
        self,
        cov: Union[float, np.ndarray, torch.Tensor],
        mode: str = "vector",
        factor: Optional[float] = None,
    ) -> None:
        if mode not in self.allowed_modes:
            raise ValueError(
                f"{mode!r} is not a recognized mode; expected one of "
                f"{list(self.allowed_modes)}."
            )
        self.mode = mode

        if factor is None:
            self._log_factor = None
        else:
            if factor < 1.0:
                raise ValueError(f"factor must be >= 1.0; got {factor}.")
            self._log_factor = math.log(float(factor))

        cov_t = torch.as_tensor(cov, dtype=torch.float64)
        if cov_t.dim() == 0:
            self._scale = cov_t.sqrt().reshape(1)
            self._chol: Optional[torch.Tensor] = None
        elif cov_t.dim() == 1:
            self._scale = cov_t.sqrt()
            self._chol = None
        elif cov_t.dim() == 2:
            if cov_t.shape[0] != cov_t.shape[1]:
                raise ValueError(
                    f"a covariance matrix must be square; got " f"{tuple(cov_t.shape)}."
                )
            if mode != "vector":
                raise ValueError(
                    "a full covariance matrix is only supported in " "'vector' mode."
                )
            self._scale = torch.empty(0)
            self._chol = torch.linalg.cholesky(cov_t)
        else:
            raise ValueError(
                "cov must be a scalar, a vector or a matrix; got shape "
                f"{tuple(cov_t.shape)}."
            )

        self._index = 0
        super().__init__(self._propose)

    def _get_factor(
        self,
        shape: Tuple[int, ...],
        generator: Optional[torch.Generator],
        device: torch.device,
        dtype: torch.dtype,
    ) -> Union[float, torch.Tensor]:
        if self._log_factor is None:
            return 1.0
        u = torch.rand(shape, generator=generator, device=device, dtype=dtype)
        return torch.exp((2.0 * u - 1.0) * self._log_factor)

    def _propose(
        self, coords: torch.Tensor, generator: Optional[torch.Generator]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        ntargets, nwalkers, ndim = coords.shape
        device, dtype = coords.device, coords.dtype
        noise = torch.randn(
            coords.shape, generator=generator, device=device, dtype=dtype
        )

        if self._chol is not None:
            step = noise @ self._chol.to(device=device, dtype=dtype).T
        else:
            scale = self._scale.to(device=device, dtype=dtype)
            step = noise * scale
        step = step * self._get_factor(
            (ntargets, nwalkers, 1), generator, device, dtype
        )

        if self.mode == "random":
            chosen = torch.randint(
                ndim,
                (ntargets, nwalkers, 1),
                generator=generator,
                device=device,
            )
            mask = torch.zeros_like(step, dtype=torch.bool)
            mask.scatter_(2, chosen, True)
            step = torch.where(mask, step, torch.zeros_like(step))
        elif self.mode == "sequential":
            index = self._index % ndim
            self._index += 1
            mask = torch.zeros_like(step, dtype=torch.bool)
            mask[..., index] = True
            step = torch.where(mask, step, torch.zeros_like(step))

        factors = torch.zeros((ntargets, nwalkers), device=device, dtype=dtype)
        return coords + step, factors
