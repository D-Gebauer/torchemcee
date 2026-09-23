# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pytest
import torch

DTYPE = torch.float64


@pytest.fixture
def seed() -> int:
    return 42


@pytest.fixture
def generator(seed: int) -> torch.Generator:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


@pytest.fixture(params=["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def device(request) -> torch.device:
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    return torch.device(request.param)


def correlated_cov(ndim: int, rho: float = 0.6) -> np.ndarray:
    i = np.arange(ndim)
    return rho ** np.abs(i[:, None] - i[None, :])


def gaussian_log_prob(mean, cov, *, dtype=DTYPE):
    mean_t = torch.as_tensor(mean, dtype=dtype)
    if mean_t.dim() == 1:
        mean_t = mean_t.unsqueeze(0)
    icov = torch.as_tensor(np.linalg.inv(np.asarray(cov)), dtype=dtype)

    def log_prob_fn(theta: torch.Tensor) -> torch.Tensor:
        d = theta - mean_t.unsqueeze(1).to(theta.device, theta.dtype)
        return -0.5 * torch.einsum(
            "bni,ij,bnj->bn", d, icov.to(theta.device, theta.dtype), d
        )

    return log_prob_fn


def rosenbrock_log_prob(a: float = 1.0, b: float = 100.0):
    def log_prob_fn(theta: torch.Tensor) -> torch.Tensor:
        x, y = theta[..., 0], theta[..., 1]
        return -((a - x) ** 2 + b * (y - x**2) ** 2) / 20.0

    return log_prob_fn


def bounded_gaussian_log_prob(mean, cov, low, high, *, dtype=DTYPE):
    inner = gaussian_log_prob(mean, cov, dtype=dtype)
    low_t = torch.as_tensor(low, dtype=dtype)
    high_t = torch.as_tensor(high, dtype=dtype)

    def log_prob_fn(theta: torch.Tensor) -> torch.Tensor:
        lo = low_t.to(theta.device, theta.dtype)
        hi = high_t.to(theta.device, theta.dtype)
        inside = ((theta >= lo) & (theta <= hi)).all(dim=-1)
        return torch.where(
            inside,
            inner(theta),
            torch.full_like(inside, -float("inf"), dtype=theta.dtype),
        )

    return log_prob_fn


def emcee_log_prob(mean, cov):
    icov = np.linalg.inv(np.asarray(cov))
    mean = np.asarray(mean)

    def log_prob(p):
        d = p - mean
        return -0.5 * float(d @ icov @ d)

    return log_prob
