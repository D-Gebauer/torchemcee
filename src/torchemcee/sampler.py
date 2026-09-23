# -*- coding: utf-8 -*-

"""The ensemble sampler"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import torch

from .autocorr import integrated_time
from .backends import Backend, MemoryBackend
from .diagnostics import acceptance_fraction
from .moves import Move, StretchMove
from .state import State
from .utils import deprecated, make_generator, resolve_device

try:
    from tqdm.auto import trange
except ImportError:  # pragma: no cover
    trange = None

__all__ = ["EnsembleSampler", "walkers_independent"]

LogProbFn = Callable[[torch.Tensor], Any]


def walkers_independent(coords: torch.Tensor) -> torch.Tensor:
    """Check that the walkers of each target span the parameter space

    Same check as in emcee, done per target. If the walkers are linearly
    dependent (e.g. all started at the same point) the ensemble can never
    leave the subspace they span.

    Args:
        coords (torch.Tensor): The positions, with shape
            ``(ntargets, nwalkers, ndim)``.

    Returns:
        torch.Tensor: A boolean per target, with shape ``(ntargets,)``.

    """
    if not bool(torch.isfinite(coords).all()):
        return torch.zeros(coords.shape[0], dtype=torch.bool, device=coords.device)
    c = coords.to(torch.float64)
    c = c - c.mean(dim=1, keepdim=True)
    colmax = c.abs().amax(dim=1, keepdim=True)
    if bool((colmax == 0).any()):
        degenerate = (colmax == 0).any(dim=-1).squeeze(-1)
    else:
        degenerate = torch.zeros(
            coords.shape[0], dtype=torch.bool, device=coords.device
        )
    c = c / colmax.clamp_min(torch.finfo(torch.float64).tiny)
    colsum = torch.sqrt((c**2).sum(dim=1, keepdim=True))
    c = c / colsum.clamp_min(torch.finfo(torch.float64).tiny)
    cond: torch.Tensor = torch.linalg.cond(c)
    independent: torch.Tensor = (cond <= 1e8) & ~degenerate
    return independent


class EnsembleSampler:
    """An ensemble MCMC sampler that runs many targets at once

    This works like ``emcee.EnsembleSampler``, but everything is a torch
    tensor with an extra leading ``ntargets`` axis. So you can sample e.g. one
    posterior per observation in a single run on the GPU.

    Args:
        log_prob_fn (callable): A function that takes positions with shape
            ``(ntargets, n, ndim)`` and returns the log-probability (up to a
            constant) with shape ``(ntargets, n)``. Careful: ``n`` is not
            always ``nwalkers``, since the moves only evaluate part of the
            ensemble at a time. Row ``b`` should only depend on target ``b``.
            Return ``-inf`` outside of the prior. It can also return a tuple
            ``(log_prob, blobs)``, where the blobs have leading axes
            ``(ntargets, n)``.
        ndim (int): Number of dimensions in the parameter space.
        nwalkers (int): The number of walkers per target. Must be even and
            larger than ``2 * ndim``.
        ntargets (Optional[int]): The number of independent targets that are
            sampled in parallel. (default: ``1``)
        moves (Optional): A single move, a list of moves, or a weighted list
            of the form ``[(StretchMove(), 0.1), ...]``. When running, a
            move is drawn from the list at each proposal.
            (default: :class:`StretchMove`)
        backend (Optional): A :class:`backends.Backend` subclass used to
            store the chain. (default: :class:`backends.MemoryBackend` on
            ``device``)
        device (Optional): The device the chain runs on.
            (default: CUDA when available, else CPU)
        dtype (Optional[torch.dtype]): The dtype of the positions and
            log-probabilities. The stored chain uses the dtype of the
            backend. (default: ``torch.float32``)
        seed (Optional[int]): Seed for the generator, ignored if ``generator``
            is given.
        generator (Optional[torch.Generator]): The generator used for all
            random draws. Pass one to get reproducible runs.
        move (Optional): Deprecated alias for ``moves``.

    Raises:
        ValueError: If ``nwalkers`` is odd, is not greater than
            ``2 * ndim``, ``ntargets`` is not positive, or the move list is
            malformed.

    """

    def __init__(
        self,
        log_prob_fn: LogProbFn,
        ndim: int,
        nwalkers: int,
        ntargets: int = 1,
        *,
        moves: Optional[Any] = None,
        backend: Optional[Backend] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
        seed: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        move: Optional[Move] = None,
    ) -> None:
        if nwalkers % 2 != 0:
            raise ValueError(f"nwalkers must be even; got {nwalkers}.")
        if nwalkers <= 2 * ndim:
            raise ValueError(
                f"nwalkers must be larger than 2 * ndim = {2 * ndim}; got "
                f"{nwalkers}."
            )
        if ntargets < 1:
            raise ValueError(f"ntargets must be >= 1; got {ntargets}.")

        if moves is None and move is not None:
            moves = move
        self._moves, self._weights = _parse_moves(moves)

        self.log_prob_fn = log_prob_fn
        self.ndim = int(ndim)
        self.nwalkers = int(nwalkers)
        self.ntargets = int(ntargets)
        self.device = resolve_device(device)
        self.dtype = dtype
        self.generator = (
            generator if generator is not None else make_generator(seed, self.device)
        )
        self.backend = (
            backend
            if backend is not None
            else MemoryBackend(device=self.device, dtype=dtype)
        )
        self._state: Optional[State] = None
        self._previous_state: Optional[State] = None

    @property
    def moves(self) -> List[Move]:
        """list: The moves the sampler draws from"""
        return self._moves

    @property
    def move(self) -> Move:
        """Move: The only move, when the sampler was given exactly one"""
        if len(self._moves) != 1:
            raise AttributeError(
                f"this sampler has {len(self._moves)} moves; use .moves."
            )
        return self._moves[0]

    @property
    def random_state(self) -> torch.Tensor:
        """torch.Tensor: The state of the sampler's generator

        Can be set to restore an earlier generator state.
        """
        return self.generator.get_state()

    @random_state.setter
    def random_state(self, state: torch.Tensor) -> None:
        try:
            self.generator.set_state(state)
        except Exception:  # pragma: no cover
            pass

    @property
    def iteration(self) -> int:
        """int: The number of moves taken in the current chain

        This is reset when a run is started from a new initial state.
        """
        return 0 if self._state is None else self._state.step

    @property
    def state(self) -> Optional[State]:
        """Optional[State]: The current state, or ``None`` before a run"""
        return self._state

    def compute_log_prob(
        self, coords: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Evaluate the log-probability at a set of positions

        This runs under ``torch.no_grad``, nothing here needs gradients.

        Args:
            coords (torch.Tensor): The positions, with shape
                ``(ntargets, n, ndim)``.

        Returns:
            tuple: The log-probability with shape ``(ntargets, n)``, and the
            blobs or ``None``.

        Raises:
            ValueError: If the function returns the wrong shape, or any
                log-probability is nan.

        """
        with torch.no_grad():
            result = self.log_prob_fn(coords)

        blobs: Optional[torch.Tensor] = None
        if isinstance(result, tuple):
            if len(result) != 2:
                raise ValueError(
                    "log_prob_fn must return the log-probability, or a "
                    f"tuple of it and the blobs; got {len(result)} values."
                )
            result, raw_blobs = result
            blobs = torch.as_tensor(raw_blobs, device=self.device).detach()

        log_prob = torch.as_tensor(result, device=self.device).detach()
        expected = tuple(coords.shape[:2])
        if tuple(log_prob.shape) != expected:
            raise ValueError(
                "log_prob_fn must return shape (ntargets, n) = "
                f"{expected} for input {tuple(coords.shape)}; got "
                f"{tuple(log_prob.shape)}."
            )
        if bool(torch.isnan(log_prob).any()):
            raise ValueError("log_prob_fn returned nan.")
        if blobs is not None and tuple(blobs.shape[:2]) != expected:
            raise ValueError(
                f"the blobs must have leading axes {expected}; got "
                f"{tuple(blobs.shape)}."
            )
        return log_prob.to(self.dtype), blobs

    def _coerce_state(
        self,
        initial_state: Union[State, torch.Tensor, np.ndarray, None],
        *,
        allow_nonfinite: bool = False,
        skip_initial_state_check: bool = False,
    ) -> State:
        """Turn whatever was passed as initial state into a checked State"""
        continuing = initial_state is None
        if initial_state is None:
            if self._previous_state is None:
                # maybe the backend has a chain from an earlier session
                try:
                    self._previous_state = self.backend.get_last_sample()
                except RuntimeError:
                    raise ValueError(
                        "cannot continue without an initial state: the "
                        "sampler has not been run yet."
                    ) from None
            initial_state = self._previous_state

        if isinstance(initial_state, State):
            # clone it, the moves write into the state in place
            state = initial_state.to(device=self.device, dtype=self.dtype).clone()
            if state.log_prob.numel() == 0:
                state.log_prob, state.blobs = self.compute_log_prob(state.coords)
        else:
            coords = torch.as_tensor(
                initial_state, device=self.device, dtype=self.dtype
            )
            if coords.dim() == 2:
                coords = coords.unsqueeze(0)
            if coords.dim() != 3:
                raise ValueError(
                    "initial_state must have shape (ntargets, nwalkers, "
                    f"ndim) or (nwalkers, ndim); got {tuple(coords.shape)}."
                )
            coords = coords.clone()
            log_prob, blobs = self.compute_log_prob(coords)
            state = State(
                coords=coords,
                log_prob=log_prob,
                accepted=torch.zeros(
                    coords.shape[:2], device=self.device, dtype=torch.long
                ),
                blobs=blobs,
            )

        expected = (self.ntargets, self.nwalkers, self.ndim)
        if tuple(state.coords.shape) != expected:
            raise ValueError(
                f"initial_state must have shape {expected}; got "
                f"{tuple(state.coords.shape)}."
            )
        if not continuing:
            state.accepted = torch.zeros_like(state.accepted)
            state.step = 0
        state.validate(allow_nonfinite=allow_nonfinite)

        if not skip_initial_state_check:
            independent = walkers_independent(state.coords)
            if not bool(independent.all()):
                bad = torch.nonzero(~independent).flatten().tolist()
                raise ValueError(
                    "The initial state has a large condition number for "
                    f"targets {bad[:8]}"
                    f"{' ...' if len(bad) > 8 else ''}. Make sure that the "
                    "walkers are linearly independent, or pass "
                    "skip_initial_state_check=True to run anyway."
                )
        return state

    def run_mcmc(
        self,
        initial_state: Union[State, torch.Tensor, np.ndarray, None],
        nsteps: int,
        **kwargs: Any,
    ) -> State:
        """Iterate :func:`EnsembleSampler.sample` and return the final state

        ``nsteps`` is the number of *kept* steps, so in total
        ``discard + nsteps * thin_by`` moves are taken.

        Args:
            initial_state (State, torch.Tensor, numpy.ndarray or None): The
                initial positions, with shape ``(ntargets, nwalkers, ndim)``
                or ``(nwalkers, ndim)`` for a single target. Pass ``None``
                to continue from where the previous run finished.
            nsteps (int): The number of retained steps.
            **kwargs: Passed to :func:`EnsembleSampler.sample`.

        Returns:
            State: The final state of the ensemble.

        """
        state: Optional[State] = None
        for state in self.sample(initial_state, nsteps, **kwargs):
            pass
        if state is None:  # pragma: no cover
            raise RuntimeError("the sampler yielded no states.")
        self._previous_state = state
        return state

    def sample(
        self,
        initial_state: Union[State, torch.Tensor, np.ndarray, None],
        nsteps: int,
        *,
        discard: int = 0,
        thin_by: int = 1,
        store: bool = True,
        progress: bool = True,
        progress_kwargs: Optional[Dict[str, Any]] = None,
        tune: bool = False,
        allow_nonfinite: bool = False,
        skip_initial_state_check: bool = False,
    ) -> Iterator[State]:
        """Advance the ensemble, yielding the state after each retained move

        Args:
            initial_state (State, torch.Tensor, numpy.ndarray or None): The
                initial positions. ``None`` continues the previous run and
                appends to the stored chain (like emcee), anything else
                starts a new chain.
            nsteps (int): The number of retained steps.
            discard (Optional[int]): The number of burn-in moves to take
                before keeping anything. emcee doesn't have this, but this
                way the burn-in never has to be stored. (default: ``0``)
            thin_by (Optional[int]): Retain one step every ``thin_by`` moves.
                (default: ``1``)
            store (Optional[bool]): Save the chain to the backend.
                (default: ``True``)
            progress (Optional[bool]): Show a progress bar.
                (default: ``True``)
            progress_kwargs (Optional[dict]): Passed to ``tqdm``.
            tune (Optional[bool]): Call ``move.tune`` after every proposal.
                (default: ``False``)
            allow_nonfinite (Optional[bool]): Permit an initial state with
                non-finite log-probabilities. (default: ``False``)
            skip_initial_state_check (Optional[bool]): Skip the check that
                the walkers of each target are linearly independent.
                (default: ``False``)

        Yields:
            State: A copy of the state of the ensemble after each retained
            step.

        Raises:
            ValueError: If ``nsteps`` or ``thin_by`` is not positive, or
                ``discard`` is negative.

        """
        if nsteps < 1:
            raise ValueError(f"nsteps must be >= 1; got {nsteps}.")
        if thin_by < 1:
            raise ValueError(f"thin_by must be >= 1; got {thin_by}.")
        if discard < 0:
            raise ValueError(f"discard must be >= 0; got {discard}.")

        continuing = initial_state is None
        state = self._coerce_state(
            initial_state,
            allow_nonfinite=allow_nonfinite,
            skip_initial_state_check=skip_initial_state_check,
        )
        self._state = state

        total_steps = discard + nsteps * thin_by
        if store:
            stores_all = self.backend.plan(
                self.ntargets,
                self.nwalkers,
                self.ndim,
                total_steps,
                discard,
                thin_by,
            )
            to_store = total_steps if stores_all else nsteps
            # continuing appends to the chain, a new start overwrites it
            if continuing:
                self.backend.grow(self.ntargets, self.nwalkers, self.ndim, to_store)
            else:
                self.backend.reset(self.ntargets, self.nwalkers, self.ndim, to_store)
        else:
            stores_all = False

        step_range: Any = range(total_steps)
        if progress and trange is not None:
            kw = dict(progress_kwargs or {})
            kw.setdefault(
                "desc",
                (
                    "ensemble MCMC"
                    if self.ntargets == 1
                    else f"ensemble MCMC (ntargets={self.ntargets})"
                ),
            )
            step_range = trange(total_steps, **kw)

        for step in step_range:
            move = self._pick_move()
            with torch.no_grad():
                state, accepted = move.propose(
                    state, self.compute_log_prob, self.generator
                )
            if tune:
                move.tune(state, accepted)

            state.random_state = self.random_state
            retained = step >= discard and (step - discard) % thin_by == 0
            if store and (stores_all or retained):
                self.backend.save_step(state)
            if retained:
                # yield a copy, the live state gets overwritten by the next move
                snapshot = state.clone()
                self._previous_state = snapshot
                yield snapshot

    def _pick_move(self) -> Move:
        """Draw one move from the weighted list"""
        if len(self._moves) == 1:
            return self._moves[0]
        weights = torch.as_tensor(
            self._weights, dtype=torch.float64, device=self.generator.device
        )
        i = int(torch.multinomial(weights, 1, generator=self.generator).item())
        return self._moves[i]

    def reset(self) -> None:
        """Clear the stored chain and the acceptance counters"""
        if self._state is not None:
            self._state.accepted.zero_()
            self._state.step = 0
        self._previous_state = None
        self.backend.reset(self.ntargets, self.nwalkers, self.ndim, 0)

    def get_value(
        self,
        name: str,
        *,
        discard: int = 0,
        thin: int = 1,
        flat: bool = False,
        numpy: bool = True,
    ) -> Any:
        """Get a stored quantity by name

        Args:
            name (str): One of ``"chain"``, ``"log_prob"`` or ``"blobs"``.
            discard (Optional[int]): Discard this many steps from the
                beginning of the stored chain. (default: ``0``)
            thin (Optional[int]): Use only every ``thin`` steps.
                (default: ``1``)
            flat (Optional[bool]): Flatten the steps and walkers into a
                single axis per target. (default: ``False``)
            numpy (Optional[bool]): Return a ``numpy.ndarray``. Pass
                ``False`` to get the torch tensor without a device round
                trip. (default: ``True``)

        Returns:
            numpy.ndarray or torch.Tensor: The stored quantity.

        """
        out = self.backend.get_value(name, discard=discard, thin=thin, flat=flat)
        return out.cpu().numpy() if numpy else out

    def get_chain(self, **kwargs: Any) -> Any:
        """Get the stored chain of walker positions

        The arguments are those of :func:`EnsembleSampler.get_value`.

        Returns:
            numpy.ndarray or torch.Tensor: The chain, with shape
            ``(nsteps, ntargets, nwalkers, ndim)``, or
            ``(ntargets, nsteps * nwalkers, ndim)`` when flattened.

        """
        return self.get_value("chain", **kwargs)

    def get_log_prob(self, **kwargs: Any) -> Any:
        """Get the stored chain of log-probabilities

        The arguments are those of :func:`EnsembleSampler.get_value`.

        Returns:
            numpy.ndarray or torch.Tensor: The log-probabilities, shaped
            like the chain without its parameter axis.

        """
        return self.get_value("log_prob", **kwargs)

    def get_blobs(self, **kwargs: Any) -> Any:
        """Get the stored chain of blobs

        The arguments are those of :func:`EnsembleSampler.get_value`.

        Returns:
            numpy.ndarray or torch.Tensor: The blobs.

        """
        return self.get_value("blobs", **kwargs)

    def get_last_sample(self) -> State:
        """Returns the most recent state of the chain"""
        if self._previous_state is None:
            raise RuntimeError("run the sampler first.")
        return self._previous_state

    def has_blobs(self) -> bool:
        """Returns ``True`` if the chain stores blobs"""
        return self.backend.has_blobs()

    @property
    @deprecated("get_chain()")
    def chain(self) -> Any:  # pragma: no cover
        """numpy.ndarray: Deprecated alias, ``(ntargets, nwalkers, nsteps, ndim)``"""
        return self.get_chain().transpose(1, 2, 0, 3)

    @property
    @deprecated("get_chain(flat=True)")
    def flatchain(self) -> Any:  # pragma: no cover
        """numpy.ndarray: Deprecated alias for the flattened chain"""
        return self.get_chain(flat=True)

    @property
    @deprecated("get_log_prob()")
    def lnprobability(self) -> Any:  # pragma: no cover
        """numpy.ndarray: Deprecated alias for the log-probabilities"""
        return self.get_log_prob()

    @property
    @deprecated("get_log_prob(flat=True)")
    def flatlnprobability(
        self,
    ) -> Any:  # pragma: no cover
        """numpy.ndarray: Deprecated alias for the flat log-probabilities"""
        return self.get_log_prob(flat=True)

    @property
    @deprecated("get_blobs()")
    def blobs(self) -> Any:  # pragma: no cover
        """numpy.ndarray: Deprecated alias for the blobs"""
        return self.get_blobs()

    @property
    @deprecated("get_blobs(flat=True)")
    def flatblobs(self) -> Any:  # pragma: no cover
        """numpy.ndarray: Deprecated alias for the flattened blobs"""
        return self.get_blobs(flat=True)

    @property
    @deprecated("get_autocorr_time()")
    def acor(self) -> torch.Tensor:  # pragma: no cover
        """torch.Tensor: Deprecated alias for the autocorrelation time"""
        return self.get_autocorr_time()

    @property
    def acceptance_fraction(self) -> torch.Tensor:
        """torch.Tensor: The acceptance fraction of every walker"""
        if self._state is None:
            raise RuntimeError("run the sampler first.")
        return acceptance_fraction(self._state.accepted, self._state.step)

    def get_autocorr_time(
        self,
        *,
        discard: int = 0,
        thin: int = 1,
        c: float = 5.0,
        tol: float = 50.0,
        quiet: bool = False,
    ) -> torch.Tensor:
        """Estimate the integrated autocorrelation time of the chain

        Args:
            discard (Optional[int]): Discard this many steps from the
                beginning of the stored chain. (default: ``0``)
            thin (Optional[int]): Use only every ``thin`` steps. The result
                is rescaled accordingly. (default: ``1``)
            c (Optional[float]): The step size for the window search.
                (default: ``5.0``)
            tol (Optional[float]): The minimum number of autocorrelation
                times needed to trust the estimate. (default: ``50.0``)
            quiet (Optional[bool]): If ``True``, log a warning instead of
                raising when the chain is too short. (default: ``False``)

        Returns:
            torch.Tensor: The autocorrelation time per target and parameter,
            with shape ``(ntargets, ndim)``.

        """
        chain = self.backend.get_chain(discard=discard, thin=thin)
        return integrated_time(chain, c=c, tol=tol, quiet=quiet) * thin


def _parse_moves(
    moves: Optional[Any],
) -> Tuple[List[Move], List[float]]:
    """Normalize the ``moves`` argument into a list and a weight vector"""
    if moves is None:
        return [StretchMove()], [1.0]
    if isinstance(moves, Move):
        return [moves], [1.0]
    if not isinstance(moves, Iterable):
        raise ValueError(
            "moves must be a Move, a list of moves, or a list of "
            f"(move, weight) pairs; got {type(moves).__name__}."
        )

    entries: Sequence[Any] = list(moves)
    if not entries:
        raise ValueError("the move list is empty.")

    if all(isinstance(entry, Move) for entry in entries):
        return list(entries), [1.0 / len(entries)] * len(entries)

    parsed: List[Move] = []
    weights: List[float] = []
    for entry in entries:
        if (
            not isinstance(entry, (tuple, list))
            or len(entry) != 2
            or not isinstance(entry[0], Move)
        ):
            raise ValueError(
                "each entry of a weighted move list must be a "
                f"(move, weight) pair; got {entry!r}."
            )
        parsed.append(entry[0])
        weights.append(float(entry[1]))

    total = sum(weights)
    if total <= 0:
        raise ValueError(f"the move weights must sum to > 0; got {total}.")
    return parsed, [w / total for w in weights]
