# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchemcee

from .conftest import correlated_cov, gaussian_log_prob


def _ar1_chain(tau, nsteps, nwalkers=32, ntargets=1, ndim=1, seed=0):
    # for x_{t+1} = rho x_t + noise, tau = (1 + rho) / (1 - rho)
    rho = (tau - 1.0) / (tau + 1.0)
    g = torch.Generator()
    g.manual_seed(seed)
    noise = torch.randn(
        nsteps, ntargets, nwalkers, ndim, generator=g, dtype=torch.float64
    ) * np.sqrt(1 - rho**2)
    chain = torch.empty_like(noise)
    chain[0] = noise[0]
    for t in range(1, nsteps):
        chain[t] = rho * chain[t - 1] + noise[t]
    return chain


def test_acceptance_fraction_counts():
    accepted = torch.tensor([[3, 0, 10]], dtype=torch.long)
    acc = torchemcee.acceptance_fraction(accepted, 10)
    assert torch.allclose(acc, torch.tensor([[0.3, 0.0, 1.0]], dtype=torch.float64))


def test_acceptance_fraction_rejects_zero_steps():
    with pytest.raises(ValueError, match="nsteps"):
        torchemcee.acceptance_fraction(torch.zeros(1, 2, dtype=torch.long), 0)


def test_autocorr_time_on_known_ar1_process():
    for tau in (5.0, 20.0):
        chain = _ar1_chain(tau, nsteps=200000, nwalkers=8)
        got = float(torchemcee.integrated_autocorr_time(chain, quiet=True)[0, 0])
        assert abs(got - tau) / tau < 0.15, f"tau={tau}, estimated {got}"


def test_autocorr_raises_when_chain_is_too_short():
    chain = _ar1_chain(50.0, nsteps=500, nwalkers=8)
    with pytest.raises(torchemcee.AutocorrError) as excinfo:
        torchemcee.integrated_autocorr_time(chain)
    # the estimate is still on the exception
    assert excinfo.value.tau.shape == (1, 1)


def test_autocorr_quiet_downgrades_to_warning(caplog):
    chain = _ar1_chain(50.0, nsteps=500, nwalkers=8)
    tau = torchemcee.integrated_autocorr_time(chain, quiet=True)
    assert tau.shape == (1, 1)
    assert "autocorrelation time" in caplog.text


def test_rhat_flags_non_mixing_ensemble():
    converged = _ar1_chain(5.0, nsteps=4000, nwalkers=16)
    assert float(torchemcee.split_rhat(converged).max()) < 1.05

    # walkers frozen at different places
    stuck = torch.zeros(4000, 1, 16, 1, dtype=torch.float64)
    stuck += torch.arange(16, dtype=torch.float64).reshape(1, 1, 16, 1)
    stuck += 0.01 * torch.randn(4000, 1, 16, 1, dtype=torch.float64)
    assert float(torchemcee.split_rhat(stuck).min()) > 2.0


def test_effective_sample_size():
    chain = _ar1_chain(10.0, nsteps=20000, nwalkers=8)
    ess = torchemcee.effective_sample_size(chain)
    expected = 20000 * 8 / 10.0
    assert abs(float(ess[0, 0]) - expected) / expected < 0.2


def test_diagnostics_are_per_target():
    good = _ar1_chain(3.0, nsteps=4000, nwalkers=16, seed=1)
    stuck = torch.zeros(4000, 1, 16, 1, dtype=torch.float64)
    stuck += torch.arange(16, dtype=torch.float64).reshape(1, 1, 16, 1)
    chain = torch.cat([good, stuck, good], dim=1)  # 3 targets
    accepted = torch.full((3, 16), 2000, dtype=torch.long)
    accepted[1] = 0  # stuck target

    summary = torchemcee.summarize(chain, accepted)
    assert summary["converged"].tolist() == [True, False, True]
    assert summary["failed"].tolist() == [1]
    assert summary["tau"].shape == (3, 1)
    assert summary["acceptance"].shape == (3, 16)


def test_diagnostics_reject_wrong_shape():
    # 3D is fine (emcee's shape), 2D is not
    assert torchemcee.integrated_autocorr_time(
        _ar1_chain(3.0, nsteps=2000, nwalkers=8)[:, 0], quiet=True
    ).shape == (1, 1)
    with pytest.raises(ValueError, match="shape"):
        torchemcee.integrated_autocorr_time(torch.zeros(10, 4))
    with pytest.raises(ValueError, match="shape"):
        torchemcee.split_rhat(torch.zeros(10, 4, 2))


def test_sampler_autocorr_rescales_for_thinning():
    ndim, nwalkers = 2, 64
    g = torch.Generator()
    g.manual_seed(0)
    s = torchemcee.EnsembleSampler(
        gaussian_log_prob(np.zeros(ndim), correlated_cov(ndim)),
        ndim,
        nwalkers,
        generator=g,
        dtype=torch.float64,
    )
    s.run_mcmc(
        torch.randn(1, nwalkers, ndim, dtype=torch.float64),
        20000,
        progress=False,
    )
    unthinned = s.get_autocorr_time(quiet=True)
    thinned = s.get_autocorr_time(thin=5, quiet=True)
    assert torch.allclose(unthinned, thinned, rtol=0.25)
