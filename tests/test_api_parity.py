# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pytest
import torch

emcee = pytest.importorskip("emcee")

import torchemcee  # noqa: E402

from .conftest import gaussian_log_prob  # noqa: E402

NDIM, NWALKERS = 2, 32


def _sampler(log_prob_fn=None, ntargets=1, seed=0, **kwargs):
    g = torch.Generator()
    g.manual_seed(seed)
    if log_prob_fn is None:
        log_prob_fn = gaussian_log_prob(np.zeros(NDIM), np.eye(NDIM))
    return torchemcee.EnsembleSampler(
        log_prob_fn,
        NDIM,
        NWALKERS,
        ntargets,
        generator=g,
        dtype=torch.float64,
        **kwargs,
    )


def _start(ntargets=1, seed=1):
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.randn(ntargets, NWALKERS, NDIM, generator=g, dtype=torch.float64)


def test_sampler_surface_covers_emcee():
    ours = {n for n in dir(torchemcee.EnsembleSampler) if not n.startswith("_")}
    theirs = {n for n in dir(emcee.EnsembleSampler) if not n.startswith("_")}
    assert not theirs - ours, f"missing: {sorted(theirs - ours)}"


def test_module_surface_covers_emcee():
    move_names = {
        n for n in dir(emcee.moves) if not n.startswith("_") and n[0].isupper()
    }
    assert not move_names - set(torchemcee.moves.__all__)

    autocorr_names = {
        "AutocorrError",
        "next_pow_two",
        "function_1d",
        "auto_window",
        "integrated_time",
    }
    assert not autocorr_names - set(torchemcee.autocorr.__all__)

    assert hasattr(torchemcee.backends, "HDFBackend")


def test_get_value_names():
    s = _sampler()
    s.run_mcmc(_start(), 20, progress=False)
    assert s.get_value("chain").shape == s.get_chain().shape
    assert s.get_value("log_prob").shape == s.get_log_prob().shape
    with pytest.raises(ValueError, match="not a stored quantity"):
        s.get_value("nonsense")


def test_iteration_and_get_last_sample():
    s = _sampler()
    assert s.iteration == 0
    final = s.run_mcmc(_start(), 25, progress=False)
    assert s.iteration == 25
    last = s.get_last_sample()
    assert torch.equal(last.coords, final.coords)
    assert np.allclose(s.get_chain()[-1], final.coords.cpu().numpy())


def test_continue_from_previous_state():
    s = _sampler()
    first = s.run_mcmc(_start(), 20, progress=False)
    second = s.run_mcmc(None, 20, progress=False)
    # the returned state is a copy
    assert not torch.equal(first.coords, second.coords)
    # continuing appends to the chain
    assert s.iteration == 40
    assert s.get_chain().shape[0] == 40
    assert np.allclose(s.get_chain()[19], first.coords.cpu().numpy())

    # a new initial state starts a new chain
    s.run_mcmc(_start(seed=2), 15, progress=False)
    assert s.iteration == 15
    assert s.get_chain().shape[0] == 15

    fresh = _sampler()
    with pytest.raises(ValueError, match="has not been run yet"):
        fresh.run_mcmc(None, 5, progress=False)


def test_random_state_roundtrip():
    s = _sampler()
    s.run_mcmc(_start(), 10, progress=False)
    saved = s.random_state
    a = s.run_mcmc(None, 10, progress=False).coords.clone()

    s.random_state = saved
    s._previous_state = s.get_last_sample()
    assert isinstance(saved, torch.Tensor)
    assert a.shape == (1, NWALKERS, NDIM)


def test_state_carries_random_state():
    s = _sampler()
    final = s.run_mcmc(_start(), 5, progress=False)
    assert final.random_state is not None


def test_compute_log_prob_is_public():
    s = _sampler()
    coords = _start()
    log_prob, blobs = s.compute_log_prob(coords)
    assert log_prob.shape == (1, NWALKERS)
    assert blobs is None


def test_compute_log_prob_rejects_nan():
    s = _sampler(log_prob_fn=lambda t: torch.full(t.shape[:2], float("nan")))
    with pytest.raises(ValueError, match="nan"):
        s.run_mcmc(_start(), 5, progress=False)


def test_blobs_roundtrip():
    inner = gaussian_log_prob(np.zeros(NDIM), np.eye(NDIM))

    def log_prob_fn(theta):
        return inner(theta), theta.sum(dim=-1)

    s = _sampler(log_prob_fn=log_prob_fn)
    s.run_mcmc(_start(), 30, progress=False)

    assert s.has_blobs()
    blobs = s.get_blobs()
    chain = s.get_chain()
    assert blobs.shape == chain.shape[:-1]
    # the blob should be the sum of the parameters
    assert np.allclose(blobs, chain.sum(axis=-1), atol=1e-10)
    assert s.get_blobs(flat=True).shape == s.get_chain(flat=True).shape[:-1]


def test_blobs_with_trailing_axes():
    inner = gaussian_log_prob(np.zeros(NDIM), np.eye(NDIM))

    def log_prob_fn(theta):
        return inner(theta), torch.stack([theta[..., 0], theta[..., 1]], -1)

    s = _sampler(log_prob_fn=log_prob_fn)
    s.run_mcmc(_start(), 10, progress=False)
    assert s.get_blobs().shape == (10, 1, NWALKERS, 2)


def test_no_blobs_raises_on_access():
    s = _sampler()
    s.run_mcmc(_start(), 10, progress=False)
    assert not s.has_blobs()
    with pytest.raises(RuntimeError, match="no blobs"):
        s.get_blobs()


def test_weighted_move_list():
    s = _sampler(moves=[(torchemcee.DEMove(), 0.8), (torchemcee.DESnookerMove(), 0.2)])
    assert len(s.moves) == 2
    assert np.isclose(sum(s._weights), 1.0)
    s.run_mcmc(_start(), 200, progress=False)
    assert np.isfinite(s.get_chain()).all()


def test_unweighted_move_list():
    s = _sampler(moves=[torchemcee.StretchMove(), torchemcee.DEMove()])
    assert np.allclose(s._weights, [0.5, 0.5])
    s.run_mcmc(_start(), 100, progress=False)
    assert np.isfinite(s.get_chain()).all()


def test_malformed_move_lists():
    with pytest.raises(ValueError, match="empty"):
        _sampler(moves=[])
    with pytest.raises(ValueError, match="\\(move, weight\\) pair"):
        _sampler(moves=[("not a move", 1.0)])
    with pytest.raises(ValueError, match="sum to"):
        _sampler(moves=[(torchemcee.DEMove(), 0.0)])


def test_walkers_independent_flags_degenerate_start():
    coords = torch.zeros(2, NWALKERS, NDIM, dtype=torch.float64)
    coords[1] = torch.randn(NWALKERS, NDIM, dtype=torch.float64)
    ok = torchemcee.walkers_independent(coords)
    assert ok.tolist() == [False, True]

    s = _sampler(ntargets=2)
    with pytest.raises(ValueError, match="condition number"):
        s.run_mcmc(coords, 5, progress=False)

    # and the check can be skipped
    s = _sampler(ntargets=2)
    s.run_mcmc(coords, 5, progress=False, skip_initial_state_check=True)


def test_deprecated_aliases_warn_and_work():
    s = _sampler()
    s.run_mcmc(_start(), 20, progress=False)

    with pytest.deprecated_call():
        assert s.flatchain.shape == s.get_chain(flat=True).shape
    with pytest.deprecated_call():
        assert s.chain.shape == (1, NWALKERS, 20, NDIM)
    with pytest.deprecated_call():
        assert s.lnprobability.shape == s.get_log_prob().shape
    with pytest.deprecated_call():
        assert s.flatlnprobability.shape == s.get_log_prob(flat=True).shape
    with pytest.deprecated_call():
        with pytest.raises(torchemcee.AutocorrError):
            s.acor


def test_move_singular_alias():
    move = torchemcee.DEMove()
    s = _sampler(move=move)
    assert s.move is move
    s = _sampler(moves=[torchemcee.DEMove(), torchemcee.StretchMove()])
    with pytest.raises(AttributeError, match="use .moves"):
        s.move


def test_autocorr_accepts_emcee_shaped_chain():
    s = _sampler()
    s.run_mcmc(_start(), 2000, progress=False)
    batched = torchemcee.integrated_time(s.get_chain(numpy=False), quiet=True)
    single = torchemcee.integrated_time(s.get_chain(numpy=False)[:, 0], quiet=True)
    assert torch.allclose(batched, single)


def test_tune_hook_is_called():
    calls = []

    class Tunable(torchemcee.StretchMove):
        def tune(self, state, accepted):
            calls.append(int(accepted.sum()))

    s = _sampler(moves=Tunable())
    s.run_mcmc(_start(), 7, progress=False, tune=True)
    assert len(calls) == 7

    calls.clear()
    s = _sampler(moves=Tunable())
    s.run_mcmc(_start(), 7, progress=False)
    assert calls == []
