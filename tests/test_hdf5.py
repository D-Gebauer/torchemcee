# -*- coding: utf-8 -*-

import numpy as np
import pytest
import torch

pytest.importorskip("h5py")

import torchemcee  # noqa: E402

from .conftest import gaussian_log_prob  # noqa: E402

NDIM, NWALKERS, NTARGETS = 3, 16, 2


def _sampler(backend, seed=3, blobs=False):
    lp = gaussian_log_prob(torch.zeros(NDIM), torch.eye(NDIM))

    def fn(theta):
        return (lp(theta), theta.sum(-1)) if blobs else lp(theta)

    return torchemcee.EnsembleSampler(
        fn, NDIM, NWALKERS, NTARGETS, backend=backend, device="cpu", seed=seed
    )


def _start():
    g = torch.Generator().manual_seed(0)
    return torch.randn(NTARGETS, NWALKERS, NDIM, generator=g)


@pytest.mark.parametrize("blobs", [False, True])
def test_matches_memory_backend(blobs):
    ref = _sampler(torchemcee.MemoryBackend(device="cpu"), blobs=blobs)
    ref.run_mcmc(_start(), 20, discard=5, thin_by=2, progress=False)
    with torchemcee.TempHDF5Backend() as backend:
        s = _sampler(backend, blobs=blobs)
        s.run_mcmc(_start(), 20, discard=5, thin_by=2, progress=False)
        assert s.get_chain().shape == (20, NTARGETS, NWALKERS, NDIM)
        for flat in (False, True):
            np.testing.assert_allclose(s.get_chain(flat=flat), ref.get_chain(flat=flat))
            np.testing.assert_allclose(
                s.get_log_prob(flat=flat), ref.get_log_prob(flat=flat)
            )
        assert s.has_blobs() == blobs
        if blobs:
            np.testing.assert_allclose(s.get_blobs(), ref.get_blobs())


def test_store_all_keeps_burn_in():
    with torchemcee.TempHDF5Backend(store="all") as backend:
        s = _sampler(backend)
        s.run_mcmc(_start(), 10, discard=5, progress=False)
        assert s.get_chain().shape[0] == 15
        assert s.get_chain(discard=5).shape[0] == 10


def test_continue_appends():
    with torchemcee.TempHDF5Backend() as backend:
        s = _sampler(backend)
        s.run_mcmc(_start(), 10, progress=False)
        first = s.get_chain()
        s.run_mcmc(None, 10, progress=False)
        chain = s.get_chain()
        assert chain.shape[0] == 20
        np.testing.assert_allclose(chain[:10], first)


def test_resume_from_file():
    with torchemcee.TempHDF5Backend() as backend:
        s = _sampler(backend)
        last = s.run_mcmc(_start(), 10, progress=False)

        again = torchemcee.HDF5Backend(backend.filename)
        assert again.iteration == 10
        assert again.shape == (NTARGETS, NWALKERS, NDIM)
        state = again.get_last_sample()
        np.testing.assert_allclose(state.coords, last.coords)
        assert torch.equal(state.accepted, last.accepted)

        s2 = _sampler(again, seed=4)
        s2.run_mcmc(None, 5, progress=False)
        assert s2.get_chain().shape[0] == 15
        assert s2.iteration == 15


def test_read_only():
    with torchemcee.TempHDF5Backend() as backend:
        _sampler(backend).run_mcmc(_start(), 5, progress=False)
        ro = torchemcee.HDF5Backend(backend.filename, read_only=True)
        assert ro.get_chain().shape[0] == 5
        with pytest.raises(RuntimeError):
            ro.reset(NTARGETS, NWALKERS, NDIM, 5)


def test_empty_file_raises():
    with torchemcee.TempHDF5Backend() as backend:
        with pytest.raises(RuntimeError):
            backend.get_chain()
        with pytest.raises(ValueError):
            _sampler(backend).run_mcmc(None, 5, progress=False)
