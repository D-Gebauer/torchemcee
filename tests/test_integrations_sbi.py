# -*- coding: utf-8 -*-

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("sbi")
pytestmark = pytest.mark.sbi

from sbi.inference import likelihood_estimator_based_potential  # noqa: E402
from sbi.neural_nets.net_builders import build_zuko_maf  # noqa: E402
from sbi.utils import BoxUniform  # noqa: E402

import torchemcee  # noqa: E402
from torchemcee.integrations.sbi import (  # noqa: E402
    flow_log_prob_fn,
    potential_to_log_prob_fn,
)

NDIM, XDIM = 2, 2


@pytest.fixture(scope="module")
def trained():
    # with a flat prior the posterior is a unit Gaussian centred on x
    torch.manual_seed(0)
    n = 4000
    prior = BoxUniform(low=-3 * torch.ones(NDIM), high=3 * torch.ones(NDIM))
    theta = prior.sample((n,))
    x = theta + torch.randn(n, XDIM)

    estimator = build_zuko_maf(batch_x=x, batch_y=theta, hidden_features=32)
    opt = torch.optim.Adam(estimator.parameters(), lr=1e-3)
    for _ in range(300):
        idx = torch.randint(n, (256,))
        loss = estimator.loss(x[idx], condition=theta[idx]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    estimator.eval()
    return estimator, prior


def test_flow_log_prob_fn_shapes(trained):
    estimator, prior = trained
    x_o = torch.zeros(3, XDIM)
    fn = flow_log_prob_fn(estimator, x_o, prior=prior)
    for n in (4, 7):
        out = fn(torch.zeros(3, n, NDIM))
        assert tuple(out.shape) == (3, n)
        assert torch.isfinite(out).all()


def test_conditioning_matches_target_index(trained):
    estimator, prior = trained
    x_o = torch.tensor([[-2.0, -2.0], [2.0, 2.0]])
    fn = flow_log_prob_fn(estimator, x_o, prior=prior)

    theta = torch.tensor([[-2.0, -2.0], [2.0, 2.0]]).unsqueeze(1)  # (2, 1, 2)
    out = fn(theta)

    # swapping the thetas should lower both log-probs
    swapped = fn(theta.flip(0))
    out, swapped = out.detach(), swapped.detach()
    assert float(out[0, 0]) > float(swapped[0, 0])
    assert float(out[1, 0]) > float(swapped[1, 0])


def test_flow_rejects_target_mismatch(trained):
    estimator, prior = trained
    fn = flow_log_prob_fn(estimator, torch.zeros(3, XDIM), prior=prior)
    with pytest.raises(ValueError, match="targets"):
        fn(torch.zeros(2, 4, NDIM))


def test_potential_adapter_matches_flow_adapter(trained):
    estimator, prior = trained
    x_o = torch.tensor([[0.5, -0.5], [1.0, 1.0]])

    potential, _ = likelihood_estimator_based_potential(estimator, prior, x_o=None)
    via_potential = potential_to_log_prob_fn(potential, x_o)
    via_flow = flow_log_prob_fn(estimator, x_o, prior=prior)

    theta = torch.randn(2, 5, NDIM).clamp(-2.5, 2.5)
    assert torch.allclose(via_potential(theta), via_flow(theta), atol=1e-5)


def test_posterior_is_centred_on_the_observation(trained):
    estimator, prior = trained
    x_o = torch.tensor([[-1.0, 0.5], [1.0, -0.5]])
    log_prob_fn = flow_log_prob_fn(estimator, x_o, prior=prior)

    g = torch.Generator()
    g.manual_seed(0)
    start = torchemcee.ball(
        x_o.to(torch.float32),
        0.3,
        64,
        log_prob_fn=log_prob_fn,
        generator=g,
    )
    sampler = torchemcee.EnsembleSampler(log_prob_fn, NDIM, 64, ntargets=2, generator=g)
    sampler.run_mcmc(start, 2000, discard=500, progress=False)
    chain = sampler.get_chain(flat=True)

    for b in range(2):
        assert np.allclose(
            chain[b].mean(axis=0), x_o[b].numpy(), atol=0.25
        ), f"target {b}: {chain[b].mean(axis=0)} vs {x_o[b].numpy()}"
