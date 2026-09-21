# torchemcee

A PyTorch version of the [emcee](https://github.com/dfm/emcee) ensemble sampler that runs on the GPU and can sample many posteriors at once.

I wrote this for simulation-based inference, where you have one trained flow and want posteriors for hundreds of observations. Running emcee on those one by one is slow and barely uses the GPU. Here all of them are sampled in the same run, with one batched call to the log-prob per step.

## Install

```bash
pip install git+https://github.com/D-Gebauer/torchemcee
```

or clone the repo and `pip install -e ".[dev]"`. There are also `[sbi]` and `[hdf5]` extras for the `sbi`/`zuko` helpers and the HDF5 backend.

## Usage

It works pretty much like emcee. The two differences are that the log-prob is batched and that everything has an extra leading `ntargets` axis:

```python
import torch
import torchemcee

def log_prob_fn(theta):
    # theta: (ntargets, n, ndim) -> (ntargets, n)
    return -0.5 * (theta**2).sum(-1)

sampler = torchemcee.EnsembleSampler(log_prob_fn, ndim=5, nwalkers=64, ntargets=32)

start = torchemcee.ball(torch.zeros(32, 5), 0.1, nwalkers=64)
sampler.run_mcmc(start, 5000, discard=500)

samples = sampler.get_chain(flat=True)   # (32, 5000 * 64, 5)
tau = sampler.get_autocorr_time()        # (32, 5)
```

A few things to keep in mind:

- `n` is not always `nwalkers`, since the moves only update part of the ensemble at a time. So don't hardcode it.
- Return `-inf` outside of the prior. Walkers are not allowed to *start* there though, otherwise they get stuck. `torchemcee.ball(..., log_prob_fn=log_prob_fn)` redraws the ones that do.
- `discard` in `run_mcmc` drops the burn-in while running, so it doesn't have to be stored. For big runs, `MemoryBackend(offload=True)` keeps the chain on the CPU.
- For reproducible runs pass `seed=` or your own `torch.Generator`.
- With `ntargets=1` (the default) you can just pass `(nwalkers, ndim)` positions like in emcee.

The rest of the API follows emcee: all the moves (`StretchMove`, `DEMove`, `DESnookerMove`, `WalkMove`, `KDEMove`, `MHMove`, `GaussianMove`), weighted move lists, blobs, `get_chain` / `get_log_prob` / `get_autocorr_time`, `run_mcmc(None, n)` to continue a run, and so on. What's missing is `pool` / `vectorize` / `args`, which aren't needed here.

To write the chain to disk while running (needs `h5py`, or the `[hdf5]` extra):

```python
backend = torchemcee.HDF5Backend("chain.h5")
sampler = torchemcee.EnsembleSampler(log_prob_fn, ndim=5, nwalkers=64, ntargets=32, backend=backend)
sampler.run_mcmc(start, 5000)
```

If the job dies, set up the same sampler again and call `sampler.run_mcmc(None, nsteps)` to continue from the file.

With many targets it's annoying to check convergence by hand, so there is

```python
chain = sampler.get_chain(numpy=False)
res = torchemcee.summarize(chain, sampler.state.accepted, nsteps=sampler.iteration)
res["failed"]   # indices of the targets that didn't converge
```

which gives the autocorrelation time, R-hat, ESS and acceptance fraction per target.

## sbi

For a likelihood flow trained with `sbi`:

```python
from torchemcee.integrations.sbi import flow_log_prob_fn

log_prob_fn = flow_log_prob_fn(density_estimator, x_o, prior=prior)   # x_o: (ntargets, x_dim)
```

`potential_to_log_prob_fn` does the same for an `sbi` potential.

## Tests

The tests compare against emcee (same moments, acceptance fractions and autocorrelation times), so you need it installed:

```bash
pip install -e ".[dev]"
pytest tests/
```

## License

MIT. If you use this, please also cite [emcee](https://arxiv.org/abs/1202.3665), this is just a port of it.
