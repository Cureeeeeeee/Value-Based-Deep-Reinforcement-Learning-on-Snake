"""Uniform experience replay (Lin 1992; Mnih et al. 2015).

A fixed-size FIFO store of transitions ``(s, a, r, s', terminated)`` in
preallocated numpy arrays. The agent learns from uniform samples of the store
instead of from the transitions it has just produced: a sample mixes many
boards and many past versions of the policy, and every transition is used for
several gradient steps. Why that matters is stated in ``agent.py``.

What is stored
--------------
``next_obs`` is whatever the caller passes. The trainer passes
``info["final_obs"]`` of ``VecSnakeEnv.step``, the true successor: the
observation that ``step`` returns for a finished sub-env already belongs to
the next episode. ``terminated`` is the environment's flag only; ``truncated``
(starvation) is not stored, because such a transition is learned from like any
other (bootstrapped from its successor).

Order and randomness
--------------------
``add`` of ``N`` transitions is equivalent to ``N`` single appends: the write
position advances by ``N`` around the ring and the oldest transitions are
overwritten first. The buffer owns no RNG; ``sample`` draws from the generator
it is given with one fixed call, so a run is reproducible from its seed and a
test can replicate the stream.

Prioritised replay (M7; Schaul et al. 2016, proportional variant)
-----------------------------------------------------------------
:class:`PrioritizedReplayBuffer` keeps the same ring and gives every slot a
priority ``p_i = (|td_i| + eps) ** alpha``, the TD error the transition had
the last time it was trained on (new transitions get the largest priority
assigned so far, so each is trained on at least once). ``sample`` draws slot
``i`` with probability ``p_i / sum_j p_j`` by stratified sampling: the mass
``[0, total)`` is cut into ``batch_size`` equal segments and one point is
drawn uniformly in each, so a batch spreads over the priority range instead
of clustering on the top. Sampling by TD error changes the data distribution
the regression sees; the batch therefore carries importance-sampling weights
``w_i = (n * P(i)) ** (-beta)``, normalised by their maximum over the buffer
so that they only scale updates down, with which the trainer weights each
transition's loss. ``beta`` is annealed from ``per_beta_start`` to 1 over the
run (the training loop does the schedule): early on the bias of the sampling
is tolerated, at the end the expected update is the uniform-replay one.

No sum tree. The paper's tree makes one sample ``O(log n)``; here a sample
is a cumulative sum over the stored priorities (``O(n)``), about 0.1 ms for
the shipped capacity of 100,000 - below the cost of the gradient step it
feeds, and vectorised numpy rather than a Python loop over the batch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Batch:
    """``B`` transitions as numpy arrays (copies, independent of the buffer)."""

    #: float32 ``(B, *obs_shape)``
    obs: np.ndarray
    #: int64 ``(B,)``
    actions: np.ndarray
    #: float32 ``(B,)``
    rewards: np.ndarray
    #: float32 ``(B, *obs_shape)``: the true successor observation.
    next_obs: np.ndarray
    #: bool ``(B,)``: the environment's ``terminated`` flag (never ``done``).
    terminated: np.ndarray


class ReplayBuffer:
    """Ring buffer of the most recent ``capacity`` transitions. See the module docstring."""

    def __init__(self, capacity: int, obs_shape: tuple[int, ...]) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, (int, np.integer)) or capacity < 1:
            raise ValueError(f"capacity must be an integer >= 1, got {capacity!r}")
        self.capacity = int(capacity)
        self.obs_shape = tuple(int(d) for d in obs_shape)
        self._obs = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self._actions = np.zeros(self.capacity, dtype=np.int64)
        self._rewards = np.zeros(self.capacity, dtype=np.float32)
        self._next_obs = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self._terminated = np.zeros(self.capacity, dtype=bool)
        self._pos = 0  # slot of the next write = slot of the oldest transition once full
        self._size = 0

    def __len__(self) -> int:
        """Number of stored transitions (at most ``capacity``)."""
        return self._size

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        terminated: np.ndarray,
    ) -> None:
        """Append ``N`` transitions (leading axis ``N``), copying them into the storage.

        ``obs`` / ``next_obs`` are ``(N, *obs_shape)``, ``actions`` /
        ``rewards`` / ``terminated`` are ``(N,)``. The caller keeps ownership
        of its arrays and may reuse them. Once the buffer is full the oldest
        transitions are overwritten; if ``N > capacity`` only the last
        ``capacity`` transitions are kept. ``ValueError`` on inconsistent
        shapes.
        """
        obs = np.asarray(obs)
        next_obs = np.asarray(next_obs)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        terminated = np.asarray(terminated)
        n = actions.shape[0] if actions.ndim == 1 else -1
        if not (
            n >= 0
            and obs.shape == next_obs.shape == (n, *self.obs_shape)
            and rewards.shape == terminated.shape == (n,)
        ):
            raise ValueError(
                f"inconsistent transitions for obs_shape {self.obs_shape}: obs {obs.shape}, actions {actions.shape}, "
                f"rewards {rewards.shape}, next_obs {next_obs.shape}, terminated {terminated.shape}"
            )

        # Equivalent to N single appends: if the batch is longer than the ring,
        # its first N - capacity transitions would be overwritten by its own tail.
        skip = max(0, n - self.capacity)
        start = (self._pos + skip) % self.capacity
        count = n - skip
        first = min(count, self.capacity - start)  # transitions that fit before the end of the storage
        fields = (
            (self._obs, obs),
            (self._actions, actions),
            (self._rewards, rewards),
            (self._next_obs, next_obs),
            (self._terminated, terminated),
        )
        for store, values in fields:
            store[start : start + first] = values[skip : skip + first]
            store[: count - first] = values[skip + first :]  # the part that wraps around (possibly empty)
        self._pos = (start + count) % self.capacity
        self._size = min(self.capacity, self._size + n)

    def sample(self, batch_size: int, rng: np.random.Generator) -> Batch:
        """``batch_size`` transitions drawn uniformly **with** replacement.

        RNG protocol (fixed): exactly one call
        ``idx = rng.integers(0, len(self), size=batch_size)``; ``idx`` indexes
        the storage slots, which hold the transitions in insertion order until
        the buffer wraps around. ``ValueError`` if the buffer is empty.
        """
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        idx = rng.integers(0, self._size, size=batch_size)
        return Batch(*self._gather(idx))

    def _gather(self, idx: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The five transition arrays of the slots ``idx`` (fancy indexing copies)."""
        return self._obs[idx], self._actions[idx], self._rewards[idx], self._next_obs[idx], self._terminated[idx]


@dataclass(frozen=True)
class PrioritizedBatch(Batch):
    """A :class:`Batch` drawn by priority, with what the trainer needs to
    weight the loss and to write the new priorities back."""

    #: int64 ``(B,)``: the storage slots the transitions were drawn from.
    indices: np.ndarray
    #: float32 ``(B,)``: importance-sampling weights, each in ``(0, 1]``.
    weights: np.ndarray


class PrioritizedReplayBuffer(ReplayBuffer):
    """Ring buffer with proportional prioritisation. See the module docstring.

    ``alpha`` is the priority exponent (0 = uniform sampling, every weight
    1), ``eps`` the constant added to ``|td|`` so no priority is 0.
    ``max_priority`` is the largest priority ever assigned (1.0 at the start):
    the priority of a newly added transition.
    """

    def __init__(self, capacity: int, obs_shape: tuple[int, ...], alpha: float = 0.6, eps: float = 1e-6) -> None:
        super().__init__(capacity, obs_shape)
        if not (isinstance(alpha, (int, float)) and not isinstance(alpha, bool) and 0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be a number in [0, 1], got {alpha!r}")
        if not (isinstance(eps, (int, float)) and not isinstance(eps, bool) and eps > 0.0):
            raise ValueError(f"eps must be a positive number, got {eps!r}")
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.max_priority = 1.0
        self._priority = np.zeros(self.capacity, dtype=np.float64)

    @property
    def priorities(self) -> np.ndarray:
        """Copy of the priorities of the stored slots ``(len(self),)``, in slot order."""
        return self._priority[: self._size].copy()

    def add(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        terminated: np.ndarray,
    ) -> None:
        """As :meth:`ReplayBuffer.add`; every slot written gets priority ``max_priority``."""
        pos_before = self._pos
        super().add(obs, actions, rewards, next_obs, terminated)  # validates the shapes, writes the ring
        # The same slots the base class wrote: the last min(N, capacity)
        # transitions, from the (skipped-ahead) old write position onwards.
        n = np.asarray(actions).shape[0]
        skip = max(0, n - self.capacity)
        slots = ((pos_before + skip) + np.arange(n - skip)) % self.capacity
        self._priority[slots] = self.max_priority

    def sample(self, batch_size: int, rng: np.random.Generator, beta: float) -> PrioritizedBatch:  # type: ignore[override]
        """``batch_size`` transitions drawn by priority, stratified.

        RNG protocol (fixed): exactly one call ``u = rng.random(batch_size)``.
        Sample ``k`` is the slot whose cumulative priority mass covers
        ``(k + u_k) / batch_size * total``. ``weights`` are
        ``(p_i / p_min) ** (-beta)``, i.e. ``(n P(i)) ** (-beta)`` divided by
        its maximum over the stored slots. ``ValueError`` if the buffer is
        empty or ``beta`` is outside ``[0, 1]``.
        """
        if self._size == 0:
            raise ValueError("cannot sample from an empty replay buffer")
        if not (isinstance(beta, (int, float)) and not isinstance(beta, bool) and 0.0 <= beta <= 1.0):
            raise ValueError(f"beta must be a number in [0, 1], got {beta!r}")
        p = self._priority[: self._size]
        total = float(p.sum())
        u = rng.random(batch_size)
        mass = (np.arange(batch_size) + u) / batch_size * total
        idx = np.minimum(np.searchsorted(np.cumsum(p), mass, side="right"), self._size - 1)
        weights = (p[idx] / p.min()) ** (-float(beta))
        return PrioritizedBatch(
            *self._gather(idx),
            indices=idx.astype(np.int64, copy=False),
            weights=weights.astype(np.float32),
        )

    def update_priorities(self, indices: np.ndarray, td_abs: np.ndarray) -> None:
        """Set the priority of the slots ``indices`` to ``(|td_abs| + eps) ** alpha``.

        Duplicate indices: the last value wins. ``ValueError`` for arrays that
        are not 1-D of the same shape, an index outside the stored slots, or
        a non-finite ``td_abs``.
        """
        indices = np.asarray(indices)
        td_abs = np.asarray(td_abs, dtype=np.float64)
        if indices.ndim != 1 or indices.shape != td_abs.shape:
            raise ValueError(f"indices and td_abs must be 1-D of the same shape, got {indices.shape} and {td_abs.shape}")
        if indices.size and (indices.min() < 0 or indices.max() >= self._size):
            raise ValueError(f"indices must lie in 0..{self._size - 1} (the stored slots), got {indices.tolist()}")
        if not np.isfinite(td_abs).all():
            raise ValueError(f"td_abs must be finite, got {td_abs.tolist()}")
        p = (np.abs(td_abs) + self.eps) ** self.alpha
        self._priority[indices] = p
        if p.size:
            self.max_priority = max(self.max_priority, float(p.max()))
