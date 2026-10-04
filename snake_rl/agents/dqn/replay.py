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
        return Batch(
            obs=self._obs[idx],  # fancy indexing copies
            actions=self._actions[idx],
            rewards=self._rewards[idx],
            next_obs=self._next_obs[idx],
            terminated=self._terminated[idx],
        )
