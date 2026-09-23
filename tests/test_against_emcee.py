# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pytest
import torch

emcee = pytest.importorskip("emcee")
from scipy import stats  # noqa: E402

import torchemcee  # noqa: E402

from .conftest import (  # noqa: E402
    correlated_cov,
    emcee_log_prob,
    gaussian_log_prob,
    rosenbrock_log_prob,
)

NDIM = 4
NWALKERS = 64
NSTEPS = 4000
DISCARD = 1000


def _run_both(seed=7):
    mean = np.linspace(-1.0, 1.0, NDIM)
    cov = correlated_cov(NDIM)

    rng = np.random.default_rng(seed)
    start = mean + 0.1 * rng.standard_normal((NWALKERS, NDIM))

    ref = emcee.EnsembleSampler(NWALKERS, NDIM, emcee_log_prob(mean, cov))
    # seed emcee as well, otherwise the comparisons get flaky
    ref.random_state = np.random.RandomState(seed).get_state()
    ref.run_mcmc(start, NSTEPS, progress=False)

    g = torch.Generator()
    g.manual_seed(seed)
    ours = torchemcee.EnsembleSampler(
        gaussian_log_prob(mean, cov),
        NDIM,
        NWALKERS,
        generator=g,
        dtype=torch.float64,
    )
    ours.run_mcmc(
        torch.as_tensor(start, dtype=torch.float64).unsqueeze(0),
        NSTEPS,
        progress=False,
    )
    return ref, ours, mean, cov


def test_gaussian_two_sample_agreement():
    ref, ours, mean, cov = _run_both()

    a = ref.get_chain(discard=DISCARD, flat=True)
    b = ours.get_chain(discard=DISCARD, flat=True)[0]

    # both should match the true moments
    assert np.allclose(b.mean(axis=0), mean, atol=0.05)
    assert np.allclose(np.cov(b.T), cov, atol=0.08)

    # and each other. Thin to roughly independent samples first, otherwise
    # the KS p-value doesn't mean much
    tau = int(max(ref.get_autocorr_time(quiet=True).max(), 1))
    for i in range(NDIM):
        p = stats.ks_2samp(a[::tau, i], b[::tau, i]).pvalue
        assert p > 1e-3, f"parameter {i}: KS p={p:.2e}"


def test_rosenbrock_agreement():
    # tau is several hundred here, so the means are noisy and are
    # compared within the combined MC error of both chains
    seed = 11
    rng = np.random.default_rng(seed)
    start = np.column_stack([rng.normal(1.0, 0.3, 64), rng.normal(1.0, 0.3, 64)])

    def ref_log_prob(p):
        x, y = p
        return -((1.0 - x) ** 2 + 100.0 * (y - x**2) ** 2) / 20.0

    ref = emcee.EnsembleSampler(64, 2, ref_log_prob)
    ref.random_state = np.random.RandomState(seed).get_state()
    ref.run_mcmc(start, 8000, progress=False)

    g = torch.Generator()
    g.manual_seed(seed)
    ours = torchemcee.EnsembleSampler(
        rosenbrock_log_prob(), 2, 64, generator=g, dtype=torch.float64
    )
    ours.run_mcmc(
        torch.as_tensor(start, dtype=torch.float64).unsqueeze(0),
        8000,
        progress=False,
    )

    a = ref.get_chain(discard=2000, flat=True)
    b = ours.get_chain(discard=2000, flat=True)[0]
    tau_a = ref.get_autocorr_time(quiet=True)
    tau_b = ours.get_autocorr_time(discard=2000, quiet=True).numpy()[0]

    for i in range(2):
        se = np.hypot(
            a[:, i].std() / np.sqrt(len(a) / max(tau_a[i], 1.0)),
            b[:, i].std() / np.sqrt(len(b) / max(tau_b[i], 1.0)),
        )
        deviation = abs(a[:, i].mean() - b[:, i].mean()) / se
        assert deviation < 4.0, (
            f"parameter {i}: means differ by {deviation:.1f} sigma "
            f"({a[:, i].mean():.3f} vs {b[:, i].mean():.3f})"
        )

    # the spread is much less noisy than the mean
    assert np.allclose(a.std(axis=0), b.std(axis=0), rtol=0.2)


def test_acceptance_fraction_agreement():
    ref, ours, _, _ = _run_both()
    assert np.isclose(
        float(ours.acceptance_fraction.mean()),
        float(ref.acceptance_fraction.mean()),
        atol=0.02,
    )


def test_autocorr_time_matches_emcee_estimator():
    ref, ours, _, _ = _run_both()

    for chain in (ref.get_chain(), ours.get_chain()[:, 0]):
        expected = emcee.autocorr.integrated_time(chain, quiet=True)
        got = torchemcee.integrated_autocorr_time(
            torch.as_tensor(chain, dtype=torch.float64).unsqueeze(1),
            quiet=True,
        )[0]
        assert np.allclose(got.numpy(), expected, rtol=1e-6, atol=1e-6)
