# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pytest
import torch

import torchemcee

from .conftest import gaussian_log_prob

NDIM, NWALKERS = 2, 32


def _run(backend, seed=0, **kwargs):
    g = torch.Generator()
    g.manual_seed(seed)
    g_start = torch.Generator()
    g_start.manual_seed(seed + 1000)
    s = torchemcee.EnsembleSampler(
        gaussian_log_prob(np.zeros(NDIM), np.eye(NDIM)),
        NDIM,
        NWALKERS,
        generator=g,
        dtype=torch.float64,
        backend=backend,
    )
    start = torch.randn(1, NWALKERS, NDIM, generator=g_start, dtype=torch.float64)
    s.run_mcmc(start, progress=False, **kwargs)
    return s


def test_store_all_and_thinned_agree():
    kw = dict(nsteps=30, discard=20, thin_by=3)
    full = _run(torchemcee.MemoryBackend(store="all", dtype=torch.float64), **kw)
    thinned = _run(torchemcee.MemoryBackend(store="thinned", dtype=torch.float64), **kw)
    assert full.get_chain().shape[0] == 20 + 30 * 3
    assert thinned.get_chain().shape[0] == 30
    assert np.array_equal(full.get_chain(discard=20, thin=3), thinned.get_chain())


def test_auto_falls_back_to_thinned_under_budget():
    backend = torchemcee.MemoryBackend(store="auto", budget=1e-9, dtype=torch.float64)
    s = _run(backend, nsteps=10, discard=20, thin_by=2)
    assert backend.stores_all is False
    assert s.get_chain().shape[0] == 10


def test_auto_stores_all_when_it_fits():
    backend = torchemcee.MemoryBackend(store="auto", dtype=torch.float64)
    s = _run(backend, nsteps=10, discard=20, thin_by=2)
    assert backend.stores_all is True
    assert s.get_chain().shape[0] == 20 + 10 * 2


def test_buffer_reused_across_runs_of_same_shape():
    backend = torchemcee.MemoryBackend(store="all", dtype=torch.float64)
    _run(backend, nsteps=25)
    first = backend._chain.data_ptr()
    _run(backend, nsteps=25, seed=1)
    assert backend._chain.data_ptr() == first


def test_buffer_reallocated_when_shape_changes():
    backend = torchemcee.MemoryBackend(store="all", dtype=torch.float64)
    _run(backend, nsteps=25)
    first_shape = tuple(backend._chain.shape)
    _run(backend, nsteps=40, seed=1)
    assert tuple(backend._chain.shape) != first_shape


def test_offload_matches_device_storage():
    on_device = _run(
        torchemcee.MemoryBackend(store="all", dtype=torch.float64),
        seed=3,
        nsteps=40,
    )
    offloaded = _run(
        torchemcee.MemoryBackend(store="all", offload=True, dtype=torch.float64),
        seed=3,
        nsteps=40,
    )
    assert np.array_equal(on_device.get_chain(), offloaded.get_chain())
    assert offloaded.backend.storage_device.type == "cpu"


def test_flat_layout_groups_by_target():
    ntargets, nsteps = 3, 7
    backend = torchemcee.MemoryBackend(store="all", dtype=torch.float64)
    backend.reset(ntargets, NWALKERS, NDIM, nsteps)

    # tag every entry with its (step, target, walker)
    for step in range(nsteps):
        coords = torch.zeros(ntargets, NWALKERS, NDIM, dtype=torch.float64)
        for b in range(ntargets):
            for w in range(NWALKERS):
                coords[b, w, 0] = b
                coords[b, w, 1] = step * 100 + w
        backend.save_step(
            torchemcee.State(
                coords=coords,
                log_prob=torch.zeros(ntargets, NWALKERS, dtype=torch.float64),
                accepted=torch.zeros(ntargets, NWALKERS, dtype=torch.long),
            )
        )

    flat = backend.get_chain(flat=True)
    assert tuple(flat.shape) == (ntargets, nsteps * NWALKERS, NDIM)
    for b in range(ntargets):
        assert torch.all(flat[b, :, 0] == b), "targets are interleaved"
    # walkers of one step stay together
    assert torch.equal(
        flat[0, :NWALKERS, 1],
        torch.arange(NWALKERS, dtype=torch.float64),
    )


def test_log_prob_shapes_match_chain():
    s = _run(torchemcee.MemoryBackend(store="all", dtype=torch.float64), nsteps=15)
    assert s.get_log_prob().shape == s.get_chain().shape[:-1]
    assert s.get_log_prob(flat=True).shape == s.get_chain(flat=True).shape[:-1]


def test_save_log_prob_can_be_disabled():
    backend = torchemcee.MemoryBackend(
        store="all", save_log_prob=False, dtype=torch.float64
    )
    s = _run(backend, nsteps=10)
    with pytest.raises(RuntimeError, match="save_log_prob"):
        s.get_log_prob()


def test_invalid_store_policy():
    with pytest.raises(ValueError, match="store must be"):
        torchemcee.MemoryBackend(store="everything")


def test_discard_beyond_chain_raises():
    s = _run(torchemcee.MemoryBackend(store="all", dtype=torch.float64), nsteps=10)
    with pytest.raises(ValueError, match="leaves nothing"):
        s.get_chain(discard=999)
