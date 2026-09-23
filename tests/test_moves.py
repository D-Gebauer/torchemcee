# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pytest
import torch

import torchemcee
from torchemcee.moves import RedBlueMove, StretchMove
from torchemcee.state import State

from .conftest import correlated_cov, gaussian_log_prob

ALL_ENSEMBLE_MOVES = [
    torchemcee.StretchMove,
    torchemcee.DEMove,
    torchemcee.DESnookerMove,
    torchemcee.WalkMove,
    torchemcee.KDEMove,
]


def _run(move, ndim=2, nwalkers=64, nsteps=4000, seed=4, ntargets=1):
    cov = correlated_cov(ndim)
    log_prob_fn = gaussian_log_prob(np.zeros(ndim), cov)
    g = torch.Generator()
    g.manual_seed(seed)
    s = torchemcee.EnsembleSampler(
        log_prob_fn,
        ndim,
        nwalkers,
        ntargets,
        moves=move,
        generator=g,
        dtype=torch.float64,
    )
    start = torch.randn(ntargets, nwalkers, ndim, generator=g, dtype=torch.float64)
    s.run_mcmc(start, nsteps, discard=nsteps // 4, progress=False)
    return s, cov


@pytest.mark.parametrize("move_cls", ALL_ENSEMBLE_MOVES)
def test_every_move_samples_a_gaussian(move_cls):
    s, cov = _run(move_cls())
    chain = s.get_chain(flat=True)[0]
    assert np.allclose(chain.mean(axis=0), 0.0, atol=0.12)
    # the snooker move comes out a few percent narrow on a run this short,
    # emcee does the same, so the tolerance is a bit loose
    assert np.allclose(np.cov(chain.T), cov, atol=0.12)


@pytest.mark.parametrize("move_cls", ALL_ENSEMBLE_MOVES)
def test_every_move_keeps_targets_independent(move_cls):
    ndim, nwalkers, ntargets = 2, 64, 3
    means = np.array([[-30.0, -30.0], [0.0, 0.0], [30.0, 30.0]])
    log_prob_fn = gaussian_log_prob(means, np.eye(ndim))
    g = torch.Generator()
    g.manual_seed(5)
    s = torchemcee.EnsembleSampler(
        log_prob_fn,
        ndim,
        nwalkers,
        ntargets,
        moves=move_cls(),
        generator=g,
        dtype=torch.float64,
    )
    start = torchemcee.ball(
        torch.as_tensor(means, dtype=torch.float64),
        0.5,
        nwalkers,
        log_prob_fn=log_prob_fn,
        dtype=torch.float64,
        generator=g,
    )
    s.run_mcmc(start, 2000, discard=500, progress=False)
    chain = s.get_chain(flat=True)
    for b in range(ntargets):
        assert np.allclose(
            chain[b].mean(axis=0), means[b], atol=0.3
        ), f"target {b} drifted to {chain[b].mean(axis=0)}"


def test_mh_move_samples_a_gaussian():
    def proposal(coords, generator):
        noise = torch.randn(
            coords.shape,
            generator=generator,
            device=coords.device,
            dtype=coords.dtype,
        )
        factors = torch.zeros(
            coords.shape[:2], device=coords.device, dtype=coords.dtype
        )
        return coords + 0.7 * noise, factors

    s, cov = _run(torchemcee.MHMove(proposal), nsteps=6000)
    chain = s.get_chain(flat=True)[0]
    assert np.allclose(chain.mean(axis=0), 0.0, atol=0.1)
    assert np.allclose(np.cov(chain.T), cov, atol=0.1)


def test_mh_move_checks_ndim():
    move = torchemcee.MHMove(lambda c, g: (c, torch.zeros(c.shape[:2])), ndim=5)
    state = State(
        coords=torch.zeros(1, 8, 2),
        log_prob=torch.zeros(1, 8),
        accepted=torch.zeros(1, 8, dtype=torch.long),
    )
    with pytest.raises(ValueError, match="dimension mismatch"):
        move.propose(state, lambda c: (torch.zeros(c.shape[:2]), None), None)


@pytest.mark.parametrize("mode", ["vector", "random", "sequential"])
def test_gaussian_move_modes(mode):
    s, cov = _run(torchemcee.GaussianMove(0.5, mode=mode), nsteps=8000)
    chain = s.get_chain(flat=True)[0]
    assert np.allclose(chain.mean(axis=0), 0.0, atol=0.15)


def test_gaussian_move_validates_arguments():
    with pytest.raises(ValueError, match="not a recognized mode"):
        torchemcee.GaussianMove(0.1, mode="diagonal")
    with pytest.raises(ValueError, match="factor must be"):
        torchemcee.GaussianMove(0.1, factor=0.5)
    with pytest.raises(ValueError, match="only supported in 'vector'"):
        torchemcee.GaussianMove(np.eye(2), mode="random")


def test_stretch_z_distribution():
    a = 2.0
    g = torch.Generator()
    g.manual_seed(0)
    u = torch.rand(200000, generator=g, dtype=torch.float64)
    z = ((a - 1.0) * u + 1.0) ** 2 / a

    assert float(z.min()) >= 1.0 / a - 1e-12
    assert float(z.max()) <= a + 1e-12

    # CDF of g(z) on [1/a, a] is (sqrt(a z) - 1) / (a - 1)
    zs = np.sort(z.numpy())
    empirical = np.arange(1, len(zs) + 1) / len(zs)
    analytic = (np.sqrt(a * zs) - 1.0) / (a - 1.0)
    assert np.max(np.abs(empirical - analytic)) < 0.01


def test_detailed_balance_on_gaussian():
    ndim, nwalkers = 2, 64
    cov = correlated_cov(ndim)
    log_prob_fn = gaussian_log_prob(np.zeros(ndim), cov)

    g = torch.Generator()
    g.manual_seed(4)
    chol = torch.linalg.cholesky(torch.as_tensor(cov, dtype=torch.float64))
    start = torch.randn(1, nwalkers, ndim, generator=g, dtype=torch.float64) @ chol.T

    s = torchemcee.EnsembleSampler(
        log_prob_fn, ndim, nwalkers, generator=g, dtype=torch.float64
    )
    s.run_mcmc(start, 4000, progress=False)
    chain = s.get_chain(flat=True)[0]

    # no burn-in needed, the ensemble starts from the target
    assert np.allclose(chain.mean(axis=0), 0.0, atol=0.06)
    assert np.allclose(np.cov(chain.T), cov, atol=0.08)


def test_jacobian_factor_present():
    class NoJacobianStretch(StretchMove):
        def get_proposal(self, s, c, generator):
            q, _factors = super().get_proposal(s, c, generator)
            return q, torch.zeros(s.shape[:2], device=s.device, dtype=s.dtype)

    good, _ = _run(StretchMove(), ndim=8, seed=6)
    bad, _ = _run(NoJacobianStretch(), ndim=8, seed=6)

    good_std = good.get_chain(flat=True)[0].std(axis=0)
    bad_std = bad.get_chain(flat=True)[0].std(axis=0)

    assert np.allclose(good_std, 1.0, atol=0.15), good_std
    # without the Jacobian the ensemble collapses towards the mode
    assert bad_std.mean() < 0.9, bad_std


def test_partners_come_from_the_complement():
    seen: List[Tuple[int, int]] = []

    class Recording(RedBlueMove):
        def get_proposal(self, s, c, generator):
            seen.append((s.shape[1], sum(cj.shape[1] for cj in c)))
            return s.clone(), torch.zeros(s.shape[:2], device=s.device, dtype=s.dtype)

    ndim, nwalkers = 2, 32
    s, _ = _run(Recording(), ndim=ndim, nwalkers=nwalkers, nsteps=4)
    # two equal splits, the complement is always the other half
    assert all(active + complement == nwalkers for active, complement in seen)
    assert all(active == nwalkers // 2 for active, _ in seen)


def test_nsplits_four_for_snooker():
    move = torchemcee.DESnookerMove()
    assert move.nsplits == 4


def test_randomize_split_can_be_disabled():
    move = StretchMove(randomize_split=False)
    s, cov = _run(move)
    chain = s.get_chain(flat=True)[0]
    assert np.allclose(chain.mean(axis=0), 0.0, atol=0.12)


def test_rejects_a_le_one():
    with pytest.raises(ValueError, match="a must be > 1"):
        StretchMove(a=1.0)


def test_rejects_bad_nsplits():
    with pytest.raises(ValueError, match="nsplits must be >= 2"):
        StretchMove(nsplits=1)


def test_rejects_too_few_walkers_for_nsplits():
    state = State(
        coords=torch.zeros(1, 6, 2),
        log_prob=torch.zeros(1, 6),
        accepted=torch.zeros(1, 6, dtype=torch.long),
    )
    move = torchemcee.DESnookerMove()
    with pytest.raises(ValueError, match="at least 2 \\* nsplits"):
        move.propose(state, lambda c: (torch.zeros(c.shape[:2]), None), None)
