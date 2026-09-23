# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchemcee

from .conftest import (
    bounded_gaussian_log_prob,
    correlated_cov,
    gaussian_log_prob,
)


def _sampler(log_prob_fn, ndim, nwalkers, ntargets, seed=3, **kwargs):
    g = torch.Generator()
    g.manual_seed(seed)
    return torchemcee.EnsembleSampler(
        log_prob_fn,
        ndim,
        nwalkers,
        ntargets,
        generator=g,
        dtype=torch.float64,
        **kwargs,
    )


def test_recovers_gaussian_moments():
    ndim = 3
    mean = np.array([1.0, -2.0, 0.5])
    cov = correlated_cov(ndim)
    s = _sampler(gaussian_log_prob(mean, cov), ndim, 64, 1)
    start = torch.as_tensor(
        mean + 0.1 * np.random.default_rng(0).standard_normal((1, 64, ndim)),
        dtype=torch.float64,
    )
    s.run_mcmc(start, 5000, discard=1000, progress=False)

    chain = s.get_chain(flat=True)[0]
    assert np.allclose(chain.mean(axis=0), mean, atol=0.05)
    assert np.allclose(np.cov(chain.T), cov, atol=0.08)


def test_ntargets_one_matches_unbatched_shape():
    ndim, nwalkers = 2, 32
    s = _sampler(gaussian_log_prob(np.zeros(ndim), np.eye(ndim)), ndim, nwalkers, 1)
    start = torch.randn(nwalkers, ndim, dtype=torch.float64)
    s.run_mcmc(start, 50, progress=False)
    assert s.get_chain().shape == (50, 1, nwalkers, ndim)
    assert s.get_chain(flat=True).shape == (1, 50 * nwalkers, ndim)


def test_targets_are_independent():
    # four well separated Gaussians, so any mixing between targets
    # shows up in the means
    ndim, nwalkers, ntargets = 3, 64, 4
    means = np.array([[-30.0] * ndim, [0.0] * ndim, [30.0] * ndim, [60.0] * ndim])
    log_prob_fn = gaussian_log_prob(means, np.eye(ndim))
    s = _sampler(log_prob_fn, ndim, nwalkers, ntargets)

    g = torch.Generator()
    g.manual_seed(1)
    start = torchemcee.ball(
        torch.as_tensor(means, dtype=torch.float64),
        0.5,
        nwalkers,
        log_prob_fn=log_prob_fn,
        dtype=torch.float64,
        generator=g,
    )
    s.run_mcmc(start, 3000, discard=500, progress=False)

    chain = s.get_chain(flat=True)
    for b in range(ntargets):
        assert np.allclose(chain[b].mean(axis=0), means[b], atol=0.1), (
            f"target {b} recovered {chain[b].mean(axis=0)}, " f"expected {means[b]}"
        )
        assert np.allclose(chain[b].std(axis=0), 1.0, atol=0.1)


def test_batched_matches_sequential():
    # only statistically, the random numbers are used in a different order
    ndim, nwalkers, ntargets = 2, 64, 3
    means = np.array([[-2.0, 1.0], [0.0, 0.0], [3.0, -1.0]])
    cov = correlated_cov(ndim)

    rng = np.random.default_rng(5)
    start = means[:, None, :] + 0.1 * rng.standard_normal((ntargets, nwalkers, ndim))

    batched = _sampler(gaussian_log_prob(means, cov), ndim, nwalkers, ntargets, seed=9)
    batched.run_mcmc(
        torch.as_tensor(start, dtype=torch.float64),
        4000,
        discard=500,
        progress=False,
    )
    batched_chain = batched.get_chain(flat=True)

    for b in range(ntargets):
        single = _sampler(gaussian_log_prob(means[b], cov), ndim, nwalkers, 1, seed=9)
        single.run_mcmc(
            torch.as_tensor(start[b], dtype=torch.float64),
            4000,
            discard=500,
            progress=False,
        )
        one = single.get_chain(flat=True)[0]
        assert np.allclose(batched_chain[b].mean(axis=0), one.mean(axis=0), atol=0.06)
        assert np.allclose(batched_chain[b].std(axis=0), one.std(axis=0), rtol=0.06)


def test_seeded_runs_are_reproducible():
    ndim, nwalkers = 2, 32
    start = torch.randn(1, nwalkers, ndim, dtype=torch.float64)
    log_prob_fn = gaussian_log_prob(np.zeros(ndim), np.eye(ndim))

    chains = []
    for _ in range(2):
        s = _sampler(log_prob_fn, ndim, nwalkers, 1, seed=1234)
        s.run_mcmc(start.clone(), 100, progress=False)
        chains.append(s.get_chain())
    assert np.array_equal(chains[0], chains[1])


def test_different_seeds_differ():
    ndim, nwalkers = 2, 32
    start = torch.randn(1, nwalkers, ndim, dtype=torch.float64)
    log_prob_fn = gaussian_log_prob(np.zeros(ndim), np.eye(ndim))

    a = _sampler(log_prob_fn, ndim, nwalkers, 1, seed=1)
    a.run_mcmc(start.clone(), 100, progress=False)
    b = _sampler(log_prob_fn, ndim, nwalkers, 1, seed=2)
    b.run_mcmc(start.clone(), 100, progress=False)
    assert not np.array_equal(a.get_chain(), b.get_chain())


def test_does_not_touch_global_rng():
    ndim, nwalkers = 2, 32
    torch.manual_seed(0)
    before = torch.randn(4)

    s = _sampler(gaussian_log_prob(np.zeros(ndim), np.eye(ndim)), ndim, nwalkers, 1)
    s.run_mcmc(
        torch.randn(1, nwalkers, ndim, dtype=torch.float64),
        50,
        progress=False,
    )

    torch.manual_seed(0)
    _ = torch.randn(4)
    after = torch.randn(4)

    torch.manual_seed(0)
    _ = torch.randn(4)
    expected = torch.randn(4)
    assert torch.equal(after, expected)
    assert before.shape == expected.shape


def test_rejects_odd_nwalkers():
    with pytest.raises(ValueError, match="even"):
        torchemcee.EnsembleSampler(lambda t: t.sum(-1), 2, 31)


def test_rejects_too_few_walkers():
    with pytest.raises(ValueError, match="2 \\* ndim"):
        torchemcee.EnsembleSampler(lambda t: t.sum(-1), 8, 16)


def test_raises_on_nonfinite_initial_log_prob():
    ndim, nwalkers = 2, 32
    log_prob_fn = bounded_gaussian_log_prob(
        np.zeros(ndim), np.eye(ndim), [-1.0, -1.0], [1.0, 1.0]
    )
    s = _sampler(log_prob_fn, ndim, nwalkers, 1)
    start = torch.full((1, nwalkers, ndim), 5.0, dtype=torch.float64)
    with pytest.raises(ValueError, match="non-finite initial log-probability"):
        s.run_mcmc(start, 10, progress=False)


def test_allow_nonfinite_opt_out():
    ndim, nwalkers = 2, 32
    log_prob_fn = bounded_gaussian_log_prob(
        np.zeros(ndim), np.eye(ndim), [-1.0, -1.0], [1.0, 1.0]
    )
    s = _sampler(log_prob_fn, ndim, nwalkers, 1)
    # all walkers but one at the same point, so skip the independence check too
    start = torch.zeros((1, nwalkers, ndim), dtype=torch.float64)
    start[0, 0] = 5.0
    s.run_mcmc(
        start,
        10,
        progress=False,
        allow_nonfinite=True,
        skip_initial_state_check=True,
    )
    assert s.get_chain().shape == (10, 1, nwalkers, ndim)


def test_bounded_target_stays_in_support():
    ndim, nwalkers = 2, 32
    low, high = [-1.0, -1.0], [1.0, 1.0]
    log_prob_fn = bounded_gaussian_log_prob(np.zeros(ndim), np.eye(ndim), low, high)
    s = _sampler(log_prob_fn, ndim, nwalkers, 1)
    g = torch.Generator()
    g.manual_seed(2)
    start = torchemcee.ball(
        torch.zeros(ndim, dtype=torch.float64),
        0.2,
        nwalkers,
        log_prob_fn=log_prob_fn,
        dtype=torch.float64,
        generator=g,
    )
    s.run_mcmc(start, 500, progress=False)
    chain = s.get_chain()
    assert chain.min() >= low[0] and chain.max() <= high[0]


def test_log_prob_fn_sees_varying_n():
    ndim, nwalkers = 2, 32
    seen = []
    inner = gaussian_log_prob(np.zeros(ndim), np.eye(ndim))

    def log_prob_fn(theta):
        seen.append(tuple(theta.shape))
        return inner(theta)

    s = _sampler(log_prob_fn, ndim, nwalkers, 1)
    s.run_mcmc(
        torch.randn(1, nwalkers, ndim, dtype=torch.float64),
        5,
        progress=False,
    )
    assert seen[0] == (1, nwalkers, ndim)
    assert all(shape == (1, nwalkers // 2, ndim) for shape in seen[1:])
    assert len(seen) == 1 + 2 * 5


def test_rejects_wrong_log_prob_shape():
    ndim, nwalkers = 2, 32
    s = _sampler(lambda theta: theta.sum(), ndim, nwalkers, 1)
    with pytest.raises(ValueError, match="must return shape"):
        s.run_mcmc(
            torch.randn(1, nwalkers, ndim, dtype=torch.float64),
            5,
            progress=False,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_roundtrip(dtype):
    ndim, nwalkers = 2, 32
    g = torch.Generator()
    g.manual_seed(0)
    s = torchemcee.EnsembleSampler(
        gaussian_log_prob(np.zeros(ndim), np.eye(ndim), dtype=dtype),
        ndim,
        nwalkers,
        generator=g,
        dtype=dtype,
        backend=torchemcee.MemoryBackend(dtype=dtype),
    )
    s.run_mcmc(torch.randn(1, nwalkers, ndim, dtype=dtype), 100, progress=False)
    assert s.get_chain(numpy=False).dtype == dtype
    assert np.isfinite(s.get_chain()).all()


def test_thin_by_and_discard_counts():
    ndim, nwalkers = 2, 32
    s = _sampler(
        gaussian_log_prob(np.zeros(ndim), np.eye(ndim)),
        ndim,
        nwalkers,
        1,
        backend=torchemcee.MemoryBackend(store="thinned", dtype=torch.float64),
    )
    s.run_mcmc(
        torch.randn(1, nwalkers, ndim, dtype=torch.float64),
        20,
        discard=50,
        thin_by=3,
        progress=False,
    )
    assert s.get_chain().shape[0] == 20
    assert s.state.step == 50 + 20 * 3


def test_acceptance_fraction_is_sane():
    ndim, nwalkers = 3, 64
    s = _sampler(
        gaussian_log_prob(np.zeros(ndim), correlated_cov(ndim)),
        ndim,
        nwalkers,
        1,
    )
    s.run_mcmc(
        torch.randn(1, nwalkers, ndim, dtype=torch.float64),
        1000,
        progress=False,
    )
    acc = s.acceptance_fraction
    assert acc.shape == (1, nwalkers)
    assert 0.15 < float(acc.mean()) < 0.8
