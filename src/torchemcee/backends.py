# -*- coding: utf-8 -*-

"""Backends for storing the chain"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from tempfile import NamedTemporaryFile
from typing import Any, Optional, Tuple, Union

import numpy as np
import torch

from .state import State
from .utils import chain_bytes, free_memory_bytes, resolve_device

try:
    import h5py
except ImportError:  # pragma: no cover
    h5py = None

__all__ = [
    "Backend",
    "MemoryBackend",
    "HDF5Backend",
    "HDFBackend",
    "TempHDF5Backend",
    "TempHDFBackend",
]


class Backend(ABC):
    """The abstract base class for chain storage

    Note that ``discard`` and ``thin`` always refer to what was actually
    stored. If the burn-in was already dropped while running, ``discard=0``
    gives you everything.

    """

    @abstractmethod
    def plan(
        self,
        ntargets: int,
        nwalkers: int,
        ndim: int,
        total_steps: int,
        discard: int,
        thin_by: int,
    ) -> bool:
        """Decide whether every step of the run is stored

        Called by the sampler before :func:`Backend.reset`.

        Args:
            ntargets (int): The number of target distributions.
            nwalkers (int): The number of walkers per target.
            ndim (int): The number of parameters.
            total_steps (int): The total number of moves the run will take.
            discard (int): The number of burn-in moves.
            thin_by (int): The thinning factor applied while running.

        Returns:
            bool: ``True`` to store all ``total_steps`` moves, ``False`` to
            store only the post-``discard`` thinned ones.

        """

    @abstractmethod
    def reset(self, ntargets: int, nwalkers: int, ndim: int, nsteps: int) -> None:
        """Size (or resize) the storage for ``nsteps`` stored steps

        Args:
            ntargets (int): The number of target distributions.
            nwalkers (int): The number of walkers per target.
            ndim (int): The number of parameters.
            nsteps (int): The number of steps that will be stored.

        """

    @abstractmethod
    def grow(self, ntargets: int, nwalkers: int, ndim: int, nsteps: int) -> None:
        """Make room for ``nsteps`` more steps, keeping what is stored

        Used when a run is continued.

        Args:
            ntargets (int): The number of target distributions.
            nwalkers (int): The number of walkers per target.
            ndim (int): The number of parameters.
            nsteps (int): The number of additional steps to make room for.

        """

    @abstractmethod
    def save_step(self, state: State) -> None:
        """Append one retained step

        Args:
            state (State): The state to store.

        """

    @abstractmethod
    def get_value(
        self,
        name: str,
        *,
        discard: int = 0,
        thin: int = 1,
        flat: bool = False,
    ) -> torch.Tensor:
        """Get a stored quantity by name

        Args:
            name (str): One of ``"chain"``, ``"log_prob"`` or ``"blobs"``.
            discard (Optional[int]): Discard this many steps from the
                beginning of the stored chain. (default: ``0``)
            thin (Optional[int]): Use only every ``thin`` steps.
                (default: ``1``)
            flat (Optional[bool]): Flatten the steps and walkers into a
                single axis per target. (default: ``False``)

        Returns:
            torch.Tensor: The stored quantity.

        """

    def get_chain(
        self, *, discard: int = 0, thin: int = 1, flat: bool = False
    ) -> torch.Tensor:
        """Get the stored chain of walker positions"""
        return self.get_value("chain", discard=discard, thin=thin, flat=flat)

    def get_log_prob(
        self, *, discard: int = 0, thin: int = 1, flat: bool = False
    ) -> torch.Tensor:
        """Get the stored chain of log-probabilities"""
        return self.get_value("log_prob", discard=discard, thin=thin, flat=flat)

    def get_blobs(
        self, *, discard: int = 0, thin: int = 1, flat: bool = False
    ) -> torch.Tensor:
        """Get the stored chain of blobs"""
        return self.get_value("blobs", discard=discard, thin=thin, flat=flat)

    @abstractmethod
    def get_last_sample(self) -> State:
        """Returns the last stored state of the chain

        Raises:
            RuntimeError: If nothing has been stored yet.

        """

    def has_blobs(self) -> bool:
        """Returns ``True`` if the backend stores blobs"""
        return False

    @property
    @abstractmethod
    def shape(self) -> Tuple[int, int, int]:
        """tuple: The ``(ntargets, nwalkers, ndim)`` shape of the ensemble"""

    def get_autocorr_time(
        self, *, discard: int = 0, thin: int = 1, **kwargs: Any
    ) -> torch.Tensor:
        """Estimate the autocorrelation time of the stored chain

        Args:
            discard (Optional[int]): Discard this many steps from the
                beginning of the stored chain. (default: ``0``)
            thin (Optional[int]): Use only every ``thin`` steps.
                (default: ``1``)
            **kwargs: Passed to
                :func:`torchemcee.autocorr.integrated_time`.

        Returns:
            torch.Tensor: The autocorrelation time per target and
            parameter, with shape ``(ntargets, ndim)``.

        """
        from .autocorr import integrated_time

        chain = self.get_chain(discard=discard, thin=thin)
        return integrated_time(chain, **kwargs) * thin


def _slice(buf: torch.Tensor, discard: int, thin: int, nstored: int) -> torch.Tensor:
    if discard >= nstored:
        raise ValueError(
            f"discard={discard} leaves nothing of the {nstored} stored steps."
        )
    if thin < 1:
        raise ValueError(f"thin must be >= 1; got {thin}.")
    return buf[discard:nstored:thin]


def _flatten(out: torch.Tensor) -> torch.Tensor:
    # permute first so the targets don't get mixed up
    nsteps, ntargets, nwalkers = out.shape[:3]
    return out.permute(1, 0, 2, *range(3, out.dim())).reshape(
        ntargets, nsteps * nwalkers, *out.shape[3:]
    )


class MemoryBackend(Backend):
    """A chain held in memory, on the sampling device or on the host

    Args:
        store (Optional[str]): ``"thinned"`` only keeps the retained steps,
            i.e. what is left after ``discard`` and ``thin_by``. ``"all"``
            keeps every step including the burn-in, so you can still pick
            ``discard`` when reading the chain. ``"auto"`` does ``"all"`` if
            it fits into memory and ``"thinned"`` otherwise.
            (default: ``"thinned"``)
        budget (Optional[float]): The fraction of the free device memory the
            full chain may use with ``"auto"``. On CPU a fixed cap of 2 GB is
            assumed. (default: ``0.5``)
        offload (Optional[bool]): Store the chain in host memory instead of
            on the GPU. A bit slower, but saves a lot of GPU memory for big
            runs. (default: ``False``)
        device (Optional): The device the chain is stored on, ignored if
            ``offload`` is set. (default: the sampling device)
        dtype (Optional[torch.dtype]): The storage dtype.
            (default: ``torch.float32``)
        save_log_prob (Optional[bool]): Whether to store the
            log-probabilities alongside the positions. (default: ``True``)

    Raises:
        ValueError: If ``store`` is not one of the three policies or
            ``budget`` is outside ``(0, 1]``.

    """

    def __init__(
        self,
        store: str = "thinned",
        *,
        budget: float = 0.5,
        offload: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        save_log_prob: bool = True,
    ) -> None:
        if store not in ("auto", "all", "thinned"):
            raise ValueError(
                f"store must be 'auto', 'all' or 'thinned'; got {store!r}."
            )
        if not 0.0 < budget <= 1.0:
            raise ValueError(f"budget must be in (0, 1]; got {budget}.")
        self.store = store
        self.budget = float(budget)
        self.offload = bool(offload)
        self.device = resolve_device(device)
        self.dtype = dtype
        self.save_log_prob = bool(save_log_prob)

        self.stores_all: Optional[bool] = None
        self._chain: Optional[torch.Tensor] = None
        self._log_prob: Optional[torch.Tensor] = None
        self._blobs: Optional[torch.Tensor] = None
        self._accepted: Optional[torch.Tensor] = None
        self._nstored = 0

    @property
    def storage_device(self) -> torch.device:
        """torch.device: The device the chain is written to"""
        return torch.device("cpu") if self.offload else self.device

    @property
    def nstored(self) -> int:
        """int: The number of steps stored so far"""
        return self._nstored

    @property
    def iteration(self) -> int:
        """int: The number of steps stored so far

        This is emcee's name for :attr:`nstored`.
        """
        return self._nstored

    @property
    def shape(self) -> Tuple[int, int, int]:
        """tuple: The ``(ntargets, nwalkers, ndim)`` shape of the ensemble"""
        if self._chain is None:
            raise RuntimeError("reset() must be called before shape.")
        return tuple(self._chain.shape[1:])  # type: ignore[return-value]

    @property
    def accepted(self) -> torch.Tensor:
        """torch.Tensor: The acceptance counts of the last stored step"""
        if self._accepted is None:
            raise RuntimeError("no samples stored; run the sampler first.")
        return self._accepted

    def has_blobs(self) -> bool:
        """Returns ``True`` if this backend stored blobs"""
        return self._blobs is not None

    def plan(
        self,
        ntargets: int,
        nwalkers: int,
        ndim: int,
        total_steps: int,
        discard: int,
        thin_by: int,
    ) -> bool:
        """Apply the storage policy to a planned run

        Returns:
            bool: Whether every step of the run will be stored.

        """
        if self.store == "all":
            self.stores_all = True
        elif self.store == "thinned":
            self.stores_all = False
        else:
            needed = chain_bytes(total_steps, ntargets, nwalkers, ndim, self.dtype)
            available = free_memory_bytes(self.storage_device)
            self.stores_all = needed < self.budget * available
        return self.stores_all

    def reset(self, ntargets: int, nwalkers: int, ndim: int, nsteps: int) -> None:
        """Allocate (or reuse) the buffers for a run

        The buffers are reused if shape, dtype and device didn't change, so
        looping over many runs doesn't reallocate every time.

        """
        shape = (nsteps, ntargets, nwalkers, ndim)
        dev = self.storage_device
        if (
            self._chain is None
            or tuple(self._chain.shape) != shape
            or self._chain.dtype != self.dtype
            or self._chain.device != dev
        ):
            self._chain = torch.empty(shape, dtype=self.dtype, device=dev)
        if self.save_log_prob:
            lp_shape = (nsteps, ntargets, nwalkers)
            if (
                self._log_prob is None
                or tuple(self._log_prob.shape) != lp_shape
                or self._log_prob.dtype != self.dtype
                or self._log_prob.device != dev
            ):
                self._log_prob = torch.empty(lp_shape, dtype=self.dtype, device=dev)
        else:
            self._log_prob = None
        self._blobs = None
        self._accepted = None
        self._nstored = 0

    def grow(self, ntargets: int, nwalkers: int, ndim: int, nsteps: int) -> None:
        """Make room for ``nsteps`` more steps, keeping what is stored"""
        if self._chain is None or self._nstored == 0:
            self.reset(ntargets, nwalkers, ndim, nsteps)
            return

        shape = (self._nstored + nsteps, ntargets, nwalkers, ndim)
        if tuple(self._chain.shape[1:]) != (ntargets, nwalkers, ndim):
            raise ValueError(
                "cannot grow a chain of shape "
                f"{tuple(self._chain.shape[1:])} to "
                f"{(ntargets, nwalkers, ndim)}."
            )
        dev = self.storage_device

        grown = torch.empty(shape, dtype=self.dtype, device=dev)
        grown[: self._nstored] = self._chain[: self._nstored]
        self._chain = grown

        if self._log_prob is not None:
            grown_lp = torch.empty(shape[:3], dtype=self.dtype, device=dev)
            grown_lp[: self._nstored] = self._log_prob[: self._nstored]
            self._log_prob = grown_lp

        if self._blobs is not None:
            grown_blobs = torch.empty(
                (shape[0],) + tuple(self._blobs.shape[1:]),
                dtype=self._blobs.dtype,
                device=dev,
            )
            grown_blobs[: self._nstored] = self._blobs[: self._nstored]
            self._blobs = grown_blobs

    def save_step(self, state: State) -> None:
        """Append one retained step

        Args:
            state (State): The state to store.

        Raises:
            RuntimeError: If called before :func:`MemoryBackend.reset`, or
                if the storage is already full.

        """
        if self._chain is None:
            raise RuntimeError("reset() must be called before save_step().")
        nsteps = self._chain.shape[0]
        if self._nstored >= nsteps:
            raise RuntimeError(
                f"storage is full ({nsteps} steps), call reset() or grow()."
            )
        i = self._nstored
        dev = self._chain.device
        self._chain[i] = state.coords.to(device=dev, dtype=self.dtype)
        if self._log_prob is not None:
            self._log_prob[i] = state.log_prob.to(device=dev, dtype=self.dtype)
        if state.blobs is not None:
            if self._blobs is None:
                self._blobs = torch.empty(
                    (nsteps,) + tuple(state.blobs.shape),
                    dtype=state.blobs.dtype,
                    device=dev,
                )
            self._blobs[i] = state.blobs.to(device=dev)
        self._accepted = state.accepted.clone()
        self._nstored += 1

    def get_value(
        self,
        name: str,
        *,
        discard: int = 0,
        thin: int = 1,
        flat: bool = False,
    ) -> torch.Tensor:
        """Get a stored quantity by name

        Args:
            name (str): One of ``"chain"``, ``"log_prob"`` or ``"blobs"``.
            discard (Optional[int]): Discard this many steps from the
                beginning of the stored chain. (default: ``0``)
            thin (Optional[int]): Use only every ``thin`` steps.
                (default: ``1``)
            flat (Optional[bool]): Flatten the steps and walkers into a
                single axis per target. (default: ``False``)

        Returns:
            torch.Tensor: The stored quantity, with shape
            ``(nsteps, ntargets, nwalkers, ...)``, or
            ``(ntargets, nsteps * nwalkers, ...)`` when flattened.

        Raises:
            ValueError: If ``name`` is not a stored quantity.
            RuntimeError: If nothing has been stored, or the quantity was
                not saved for this run.

        """
        if self._nstored == 0:
            raise RuntimeError("no samples stored; run the sampler first.")
        if name == "chain":
            buf = self._chain
        elif name == "log_prob":
            if self._log_prob is None:
                raise RuntimeError(
                    "log-probabilities were not stored (save_log_prob=False)."
                )
            buf = self._log_prob
        elif name == "blobs":
            if self._blobs is None:
                raise RuntimeError(
                    "no blobs were stored; the log-probability function "
                    "returned none."
                )
            buf = self._blobs
        else:
            raise ValueError(
                f"{name!r} is not a stored quantity; expected 'chain', "
                "'log_prob' or 'blobs'."
            )

        assert buf is not None
        out = _slice(buf, discard, thin, self._nstored)
        if flat:
            out = _flatten(out)
        return out

    def get_last_sample(self) -> State:
        """Returns the last stored state of the chain

        Raises:
            RuntimeError: If nothing has been stored yet.

        """
        if self._chain is None or self._nstored == 0:
            raise RuntimeError("no samples stored; run the sampler first.")
        i = self._nstored - 1
        return State(
            coords=self._chain[i],
            log_prob=(
                self._log_prob[i] if self._log_prob is not None else torch.empty(0)
            ),
            accepted=(
                self._accepted
                if self._accepted is not None
                else torch.zeros(
                    self._chain.shape[1:3],
                    dtype=torch.long,
                    device=self._chain.device,
                )
            ),
            blobs=None if self._blobs is None else self._blobs[i],
            step=self._nstored,
        )


class HDF5Backend(Backend):
    """A backend that writes the chain to an HDF5 file

    Every retained step is written to disk right away, so a run that dies
    can be continued from the file with ``sampler.run_mcmc(None, nsteps)``.
    This needs ``h5py``.

    Args:
        filename (str): The name of the HDF5 file.
        name (Optional[str]): The name of the group the chain is saved in.
            (default: ``"mcmc"``)
        read_only (Optional[bool]): Open the file read only, which is useful
            for looking at a chain while it is still running.
            (default: ``False``)
        store (Optional[str]): ``"thinned"`` only keeps the retained steps,
            ``"all"`` keeps every step including the burn-in.
            (default: ``"thinned"``)
        dtype (Optional[torch.dtype]): The storage dtype.
            (default: ``torch.float32``)
        compression (Optional[str]): Passed to ``h5py``, e.g. ``"gzip"``.
        compression_opts (Optional): Passed to ``h5py``.

    """

    def __init__(
        self,
        filename: str,
        *,
        name: str = "mcmc",
        read_only: bool = False,
        store: str = "thinned",
        dtype: torch.dtype = torch.float32,
        compression: Optional[str] = None,
        compression_opts: Any = None,
    ) -> None:
        if h5py is None:
            raise ImportError("you must install 'h5py' to use the HDF5Backend")
        if store not in ("all", "thinned"):
            raise ValueError(f"store must be 'all' or 'thinned'; got {store!r}.")
        self.filename = filename
        self.name = name
        self.read_only = read_only
        self.store = store
        self.dtype = dtype
        self.compression = compression
        self.compression_opts = compression_opts

    @property
    def _np_dtype(self) -> Any:
        return torch.empty((), dtype=self.dtype).numpy().dtype

    @property
    def initialized(self) -> bool:
        """bool: Whether the file already contains this chain"""
        if not os.path.exists(self.filename):
            return False
        try:
            with self.open() as f:
                return self.name in f
        except OSError:
            return False

    def open(self, mode: str = "r") -> Any:
        """Open the file, the caller has to close it again"""
        if self.read_only and mode != "r":
            raise RuntimeError(
                "The backend has been loaded in read-only mode. Set "
                "read_only=False to make changes."
            )
        return h5py.File(self.filename, mode)

    @property
    def nstored(self) -> int:
        """int: The number of steps stored so far"""
        if not self.initialized:
            return 0
        with self.open() as f:
            return int(f[self.name].attrs["iteration"])

    @property
    def iteration(self) -> int:
        """int: The number of steps stored so far"""
        return self.nstored

    @property
    def shape(self) -> Tuple[int, int, int]:
        """tuple: The ``(ntargets, nwalkers, ndim)`` shape of the ensemble"""
        with self.open() as f:
            g = f[self.name]
            return (
                int(g.attrs["ntargets"]),
                int(g.attrs["nwalkers"]),
                int(g.attrs["ndim"]),
            )

    @property
    def accepted(self) -> torch.Tensor:
        """torch.Tensor: The acceptance counts of the last stored step"""
        with self.open() as f:
            return torch.from_numpy(f[self.name]["accepted"][...])

    def has_blobs(self) -> bool:
        """Returns ``True`` if this backend stored blobs"""
        if not self.initialized:
            return False
        with self.open() as f:
            return "blobs" in f[self.name]

    def plan(
        self,
        ntargets: int,
        nwalkers: int,
        ndim: int,
        total_steps: int,
        discard: int,
        thin_by: int,
    ) -> bool:
        """Returns ``True`` if every step of the run is stored"""
        return self.store == "all"

    def reset(self, ntargets: int, nwalkers: int, ndim: int, nsteps: int) -> None:
        """Clear the group and set up empty datasets for a new chain"""
        with self.open("a") as f:
            if self.name in f:
                del f[self.name]
            g = f.create_group(self.name)
            g.attrs["ntargets"] = ntargets
            g.attrs["nwalkers"] = nwalkers
            g.attrs["ndim"] = ndim
            g.attrs["iteration"] = 0
            g.create_dataset(
                "accepted", data=np.zeros((ntargets, nwalkers), dtype=np.int64)
            )
            for key, tail in (
                ("chain", (ntargets, nwalkers, ndim)),
                ("log_prob", (ntargets, nwalkers)),
            ):
                g.create_dataset(
                    key,
                    (nsteps,) + tail,
                    maxshape=(None,) + tail,
                    dtype=self._np_dtype,
                    compression=self.compression,
                    compression_opts=self.compression_opts,
                )

    def grow(self, ntargets: int, nwalkers: int, ndim: int, nsteps: int) -> None:
        """Make room for ``nsteps`` more steps, keeping what is stored"""
        if not self.initialized:
            self.reset(ntargets, nwalkers, ndim, nsteps)
            return
        if self.shape != (ntargets, nwalkers, ndim):
            raise ValueError(
                f"cannot grow a chain of shape {self.shape} to "
                f"{(ntargets, nwalkers, ndim)}."
            )
        with self.open("a") as f:
            g = f[self.name]
            n = int(g.attrs["iteration"]) + nsteps
            for key in ("chain", "log_prob", "blobs"):
                if key in g:
                    g[key].resize(n, axis=0)

    def save_step(self, state: State) -> None:
        """Append one retained step

        Args:
            state (State): The state to store.

        """
        with self.open("a") as f:
            g = f[self.name]
            i = int(g.attrs["iteration"])
            if i >= g["chain"].shape[0]:
                raise RuntimeError(
                    f"storage is full ({i} steps), call reset() or grow()."
                )
            g["chain"][i] = state.coords.to("cpu", self.dtype).numpy()
            g["log_prob"][i] = state.log_prob.to("cpu", self.dtype).numpy()
            if state.blobs is not None:
                blobs = state.blobs.cpu().numpy()
                if "blobs" not in g:
                    g.create_dataset(
                        "blobs",
                        (g["chain"].shape[0],) + blobs.shape,
                        maxshape=(None,) + blobs.shape,
                        dtype=blobs.dtype,
                        compression=self.compression,
                        compression_opts=self.compression_opts,
                    )
                g["blobs"][i] = blobs
            g["accepted"][...] = state.accepted.cpu().numpy()
            g.attrs["step"] = state.step
            g.attrs["iteration"] = i + 1

    def get_value(
        self,
        name: str,
        *,
        discard: int = 0,
        thin: int = 1,
        flat: bool = False,
    ) -> torch.Tensor:
        """Get a stored quantity by name, see :func:`Backend.get_value`"""
        if name not in ("chain", "log_prob", "blobs"):
            raise ValueError(
                f"{name!r} is not a stored quantity; expected 'chain', "
                "'log_prob' or 'blobs'."
            )
        nstored = self.nstored
        if nstored == 0:
            raise RuntimeError("no samples stored; run the sampler first.")
        if discard >= nstored:
            raise ValueError(
                f"discard={discard} leaves nothing of the {nstored} stored steps."
            )
        if thin < 1:
            raise ValueError(f"thin must be >= 1; got {thin}.")
        with self.open() as f:
            g = f[self.name]
            if name not in g:
                raise RuntimeError(f"no {name} were stored.")
            out = torch.from_numpy(g[name][discard:nstored:thin])
        return _flatten(out) if flat else out

    def get_last_sample(self) -> State:
        """Returns the last stored state of the chain

        Raises:
            RuntimeError: If nothing has been stored yet.

        """
        nstored = self.nstored
        if nstored == 0:
            raise RuntimeError("no samples stored; run the sampler first.")
        with self.open() as f:
            g = f[self.name]
            return State(
                coords=torch.from_numpy(g["chain"][nstored - 1]),
                log_prob=torch.from_numpy(g["log_prob"][nstored - 1]),
                accepted=torch.from_numpy(g["accepted"][...]),
                blobs=(
                    torch.from_numpy(g["blobs"][nstored - 1]) if "blobs" in g else None
                ),
                step=int(g.attrs.get("step", nstored)),
            )


class TempHDF5Backend:
    """A :class:`HDF5Backend` in a temporary file, mostly for testing"""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.filename: Optional[str] = None

    def __enter__(self) -> HDF5Backend:
        f = NamedTemporaryFile(prefix="torchemcee-", suffix=".h5", delete=False)
        f.close()
        self.filename = f.name
        return HDF5Backend(f.name, **self.kwargs)

    def __exit__(self, *args: Any) -> None:
        if self.filename is not None:
            os.remove(self.filename)


#: the name emcee uses
HDFBackend = HDF5Backend
TempHDFBackend = TempHDF5Backend
