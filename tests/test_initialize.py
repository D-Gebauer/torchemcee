# -*- coding: utf-8 -*-

"""Initial-state helpers"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchemcee

from .conftest import bounded_gaussian_log_prob, gaussian_log_prob


def test_from_prior_shapes():
    prior = torch.distributions.Normal(
        torch.zeros(3, dtype=torch.float64),
        torch.ones(3, dtype=torch.float64),
    )
    coords = torchemcee.from_prior(
        lambda n: prior.sample((n,)),
        nwalkers=16,
        ntargets=4,
        dtype=torch.float64,
    )
    assert tuple(coords.shape) == (4, 16, 3)


def test_from_prior_rejects_bad_sample_fn():
    with pytest.raises(ValueError, match="must return shape"):
        torchemcee.from_prior(lambda n: torch.zeros(n, 3, 2), 4, 2)


def test_ball_shapes():
    g = torch.Generator()
    g.manual_seed(0)
    single = torchemcee.ball(
        torch.zeros(3, dtype=torch.float64),
        0.1,
        16,
        dtype=torch.float64,
        generator=g,
    )
    assert tuple(single.shape) == (1, 16, 3)

    multi = torchemcee.ball(
        torch.zeros(5, 3, dtype=torch.float64),
        0.1,
        16,
        dtype=torch.float64,
        generator=g,
    )
    assert tuple(multi.shape) == (5, 16, 3)


def test_ball_accepts_per_parameter_scale():
    g = torch.Generator()
    g.manual_seed(0)
    coords = torchemcee.ball(
        torch.zeros(2, dtype=torch.float64),
        torch.tensor([0.01, 10.0], dtype=torch.float64),
        4096,
        dtype=torch.float64,
        generator=g,
    )
    spread = coords[0].std(dim=0)
    assert float(spread[0]) < 0.05
    assert float(spread[1]) > 5.0


def test_ball_resamples_invalid_walkers():
    """With a bounded target, no walker is returned outside the support"""
    ndim = 2
    low, high = [-1.0, -1.0], [1.0, 1.0]
    log_prob_fn = bounded_gaussian_log_prob(np.zeros(ndim), np.eye(ndim), low, high)
    g = torch.Generator()
    g.manual_seed(0)
    # scale way too large for the box, most first draws land outside
    coords = torchemcee.ball(
        torch.zeros(ndim, dtype=torch.float64),
        2.0,
        256,
        log_prob_fn=log_prob_fn,
        dtype=torch.float64,
        generator=g,
    )
    assert torch.isfinite(log_prob_fn(coords)).all()
    assert float(coords.min()) >= low[0]
    assert float(coords.max()) <= high[0]


def test_ball_raises_when_center_is_invalid():
    """A centre outside the support cannot be rescued by resampling"""
    ndim = 2
    log_prob_fn = bounded_gaussian_log_prob(
        np.zeros(ndim), np.eye(ndim), [-1.0, -1.0], [1.0, 1.0]
    )
    g = torch.Generator()
    g.manual_seed(0)
    with pytest.raises(ValueError, match="resampling rounds"):
        torchemcee.ball(
            torch.full((ndim,), 50.0, dtype=torch.float64),
            0.1,
            32,
            log_prob_fn=log_prob_fn,
            max_resample=5,
            dtype=torch.float64,
            generator=g,
        )


def test_ball_rejects_bad_shapes():
    with pytest.raises(ValueError, match="center must have shape"):
        torchemcee.ball(torch.zeros(2, 2, 2), 0.1, 8)
    with pytest.raises(ValueError, match="scale must be"):
        torchemcee.ball(torch.zeros(3), torch.zeros(2), 8)


def test_check_initial_state_reports_indices():
    ndim, nwalkers = 2, 8
    log_prob_fn = gaussian_log_prob(np.zeros(ndim), np.eye(ndim))
    coords = torch.zeros(1, nwalkers, ndim, dtype=torch.float64)
    log_prob = log_prob_fn(coords)
    log_prob[0, 3] = -float("inf")
    state = torchemcee.State(
        coords=coords,
        log_prob=log_prob,
        accepted=torch.zeros(1, nwalkers, dtype=torch.long),
    )

    bad = torchemcee.check_initial_state(state, raise_on_invalid=False)
    assert bad.tolist() == [[0, 3]]
    with pytest.raises(ValueError, match="non-finite"):
        torchemcee.check_initial_state(state)
