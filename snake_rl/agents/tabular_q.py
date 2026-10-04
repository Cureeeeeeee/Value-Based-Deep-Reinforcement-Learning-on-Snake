"""Tabular Q-learning on observation A (PLAN.md milestone M4, Part B phase 1).

The reference point "before function approximation": one table entry per
(state, action); no network, no replay buffer, no target network. The DQN on
the same 11 features (M5) is compared with this agent.

State
-----
Observation A is 11 binary features, so a state is the 11-bit integer
``sum_k obs[k] * 2**k`` and the table is ``q[2048, n_actions]``. At most
``8 * 4 * 8 = 256`` of the 2048 codes can occur as a state the agent acts in
(three free danger bits; a one-hot heading; the food is above / below / level
and left / right / level, but never on the head). The unused rows cost 50 kB
and keep the index a plain bit pattern.

Update rule (Watkins 1989; Sutton & Barto 2018, eq. 6.8)
--------------------------------------------------------
For one transition ``(s, a, r, s', terminated)``::

    target   = r + gamma * (0 if terminated else max_a' Q[s', a'])
    Q[s, a] += alpha * (target - Q[s, a])

The max over the successor's actions makes the method off-policy: the table
estimates the value of the greedy policy while an epsilon-greedy policy
collects the data.

terminated is not truncated
---------------------------
``terminated`` (wall, self collision, win) means there is no future, so the
successor is worth zero by definition. ``truncated`` (starvation: ``H`` steps
without food) is a time limit the agent cannot observe - ``steps_since_food``
is not part of the observation. The successor of a truncated transition is an
ordinary state from which the game would have continued, so the target
bootstraps from it like any other transition. Treating truncation as terminal
would write "the future is worth nothing here" into entries shared with every
other visit of the same state, driven by a hidden clock. Bootstrapping keeps
the value of looping behaviour honest: a policy that circles forever gets the
discounted return of circling forever (zero under R0, the accumulated step
penalty under R1, the accumulated shaping bonus under R3), not a value cut
off after ``H`` steps. Whether looping pays is then decided by the reward
design alone, which is what the reward experiments measure.

The caller therefore passes the environment's ``terminated`` flag (never
``done``) and takes ``next_obs`` from ``info["final_obs"]`` of
``VecSnakeEnv.step``: the observation that ``step`` returns for a finished
sub-env already belongs to the next episode.

Synchronous batch form
----------------------
Training steps ``N`` environments at once, so :meth:`TabularQAgent.update`
receives ``N`` transitions. All ``N`` targets and TD errors are computed from
the table as it was before the call; then every distinct cell ``(s, a)`` in
the batch moves by ``alpha`` times the *mean* TD error of its transitions.
Duplicates are averaged, not summed: ``k`` transitions of one cell make one
step of size ``alpha`` towards their mean target, so the effective step size
does not grow with the number of environments (a sum would step by
``k * alpha``). With ``N = 1`` every cell occurs once and the rule is exactly
the textbook update above.

What the table can and cannot learn
-----------------------------------
Observation A is not Markov: it discards the body layout, all distances and
the board size, so many different boards share one code. The table does not
learn the optimal action values of the game; it learns one value per
*aggregate* of boards (state aggregation with a fixed hard partition),
averaged over the boards the behaviour policy happens to visit under that
code. The convergence guarantee of tabular Q-learning does not carry over to
the aggregated process, and no amount of training recovers what the features
removed - the agent cannot see the dead end it is entering. That ceiling
belongs to the representation, not to the learner, which is exactly why this
agent is the reference point for the DQN on the same features: the network
sees the same 2048 codes, and every greedy policy it can express on them is
also a table. A difference between the two on observation A is therefore a
difference in learning dynamics (replay, target network, optimiser), not in
what can be represented; what the features removed can only come back with a
richer observation (B).

Exploration
-----------
Epsilon-greedy with an explicit generator and a fixed draw order
(:meth:`TabularQAgent.act_epsilon`): a run is reproducible from its seed and
there is no hidden RNG. Greedy ties go to the lowest action index
(``np.argmax``) everywhere, so the single-observation and the batched greedy
policy always agree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

from snake_rl.env.core import SnakeEnv
from snake_rl.env.obs import N_FEATURES_A

#: Number of observation-A codes (rows of the table).
N_STATES_A = 2**N_FEATURES_A

#: Weight of feature ``k`` in the state index: ``2**k``. Floating point so
#: that the index is one matrix product; exact, because every partial sum is
#: an integer below 2048. The float32 copy avoids a cast of the environment's
#: float32 observations on the training hot path.
_BIT_WEIGHTS = 2.0 ** np.arange(N_FEATURES_A)
_BIT_WEIGHTS.setflags(write=False)
_BIT_WEIGHTS_F32 = _BIT_WEIGHTS.astype(np.float32)
_BIT_WEIGHTS_F32.setflags(write=False)


def state_index(obs: np.ndarray) -> np.ndarray:
    """Table row of observation(s) A: ``sum_k obs[..., k] * 2**k``.

    ``obs`` has shape ``(..., 11)`` and holds exactly 0 / 1 (any real or bool
    dtype; the values are not validated). Returns an int64 array of shape
    ``obs.shape[:-1]`` (0-d for a single observation).
    """
    obs = np.asarray(obs)
    if obs.ndim == 0 or obs.shape[-1] != N_FEATURES_A:
        raise ValueError(f"observation A must have a last axis of {N_FEATURES_A}, got shape {obs.shape}")
    weights = _BIT_WEIGHTS_F32 if obs.dtype == np.float32 else _BIT_WEIGHTS
    return np.asarray(obs @ weights).astype(np.int64)


@dataclass(frozen=True)
class TabularQConfig:
    #: Step size.
    alpha: float = 0.1
    #: Discount.
    gamma: float = 0.99
    eps_start: float = 1.0
    eps_end: float = 0.05
    #: Epsilon decays linearly over this fraction of ``train.total_env_steps``
    #: (used by the training loop, not by the agent).
    eps_decay_fraction: float = 0.2
    #: Initial value of every table entry.
    q_init: float = 0.0


class TabularQAgent:
    """Q-table over the 2048 observation-A codes. See the module docstring.

    ``q[s, a]`` are the action values (float64), ``visits[s, a]`` the number
    of transitions that updated each cell (int64). Both are public; the agent
    holds no other state and no RNG.
    """

    name = "tabular_q"

    def __init__(self, n_actions: int, cfg: TabularQConfig | None = None) -> None:
        if isinstance(n_actions, bool) or not isinstance(n_actions, (int, np.integer)) or n_actions < 1:
            raise ValueError(f"n_actions must be a positive integer, got {n_actions!r}")
        self.cfg = cfg if cfg is not None else TabularQConfig()
        self.n_actions = int(n_actions)
        self.q: np.ndarray = np.full((N_STATES_A, self.n_actions), self.cfg.q_init, dtype=np.float64)
        self.visits: np.ndarray = np.zeros((N_STATES_A, self.n_actions), dtype=np.int64)

    # ------------------------------------------------------------- acting
    def reset(self) -> None:
        """Policy protocol (start of an evaluation episode). Nothing to reset."""

    def act(self, obs: np.ndarray, env: SnakeEnv | None = None) -> int:
        """Greedy action for one observation ``(11,)`` (Policy protocol; ``env`` is unused)."""
        obs = np.asarray(obs)
        if obs.ndim != 1:
            raise ValueError(f"act() takes one observation of shape ({N_FEATURES_A},), got shape {obs.shape}")
        return int(np.argmax(self.q[state_index(obs)]))

    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        """Greedy actions ``(N,)`` int64 for a batch of observations ``(N, 11)``."""
        obs = np.asarray(obs)
        if obs.ndim != 2:
            raise ValueError(f"act_batch() takes observations of shape (N, {N_FEATURES_A}), got shape {obs.shape}")
        return self.q[state_index(obs)].argmax(axis=1).astype(np.int64, copy=False)

    def act_epsilon(self, obs: np.ndarray, epsilon: float, rng: np.random.Generator) -> np.ndarray:
        """Epsilon-greedy actions ``(N,)`` int64 for a batch ``(N, 11)``.

        RNG protocol (fixed, so that runs are reproducible and tests can
        replicate the stream): first ``rng.random(N) < epsilon`` decides who
        explores, then ``rng.integers(0, n_actions, size=N)`` proposes a
        random action for everyone. Both draws happen on every call, in this
        order, whatever ``epsilon`` is.
        """
        greedy = self.act_batch(obs)
        n = greedy.shape[0]
        explore = rng.random(n) < epsilon
        random_actions = rng.integers(0, self.n_actions, size=n)
        return np.where(explore, random_actions, greedy)

    # ----------------------------------------------------------- learning
    def update(
        self,
        obs: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_obs: np.ndarray,
        terminated: np.ndarray,
    ) -> float:
        """One synchronous batch of Q-learning updates from ``N`` transitions.

        ``obs`` / ``next_obs`` are ``(N, 11)``; ``actions`` ``(N,)`` integers;
        ``rewards`` ``(N,)`` of any real dtype (the arithmetic is float64);
        ``terminated`` ``(N,)`` bool - the environment's ``terminated`` flag
        only. A starvation-truncated transition has ``terminated = False`` and
        bootstraps from ``next_obs``, which must be the true successor
        (``info["final_obs"]``), not the post-reset observation.

        With ``q`` as it stands before the call for every ``n``::

            target_n = r_n + gamma * (0 if terminated_n else max_a' q[s'_n, a'])
            td_n     = target_n - q[s_n, a_n]
            q[s, a] += alpha * mean(td_n over the transitions of cell (s, a))
            visits[s, a] += number of transitions of cell (s, a)

        Returns ``mean(|td_n|)`` over the batch.
        """
        obs = np.asarray(obs)
        next_obs = np.asarray(next_obs)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards, dtype=np.float64)
        terminated = np.asarray(terminated, dtype=bool)
        n = actions.shape[0] if actions.ndim == 1 else -1
        if n < 1:
            raise ValueError(f"actions must have shape (N,) with N >= 1, got {actions.shape}")
        if not (
            obs.shape == next_obs.shape == (n, N_FEATURES_A)
            and rewards.shape == terminated.shape == (n,)
        ):
            raise ValueError(
                f"inconsistent batch for N = {n}: obs {obs.shape}, next_obs {next_obs.shape}, "
                f"rewards {rewards.shape}, terminated {terminated.shape}"
            )
        if not np.issubdtype(actions.dtype, np.integer):
            raise ValueError(f"actions must be integers, got dtype {actions.dtype}")
        if actions.min() < 0 or actions.max() >= self.n_actions:
            raise ValueError(f"actions must lie in [0, {self.n_actions})")

        q = self.q
        s = state_index(obs)
        next_max = q[state_index(next_obs)].max(axis=1)
        target = rewards + self.cfg.gamma * np.where(terminated, 0.0, next_max)
        q_sa = q[s, actions]
        td = target - q_sa

        # Group the transitions by cell (flat index s * n_actions + a): the
        # number of transitions and the mean TD error of each transition's cell.
        cell = s * self.n_actions + actions
        count = np.bincount(cell)[cell]
        td_mean = np.bincount(cell, weights=td)[cell] / count

        # One write. All transitions of a cell carry the same new value (old
        # value + alpha * mean TD error of the cell) and the same new visit
        # count, so the duplicate indices are harmless: each distinct cell
        # moves exactly once.
        q[s, actions] = q_sa + self.cfg.alpha * td_mean
        self.visits[s, actions] += count
        return float(np.abs(td).mean())

    # ---------------------------------------------------------------- I/O
    def save(self, path: str | Path) -> Path:
        """Write ``q``, ``visits``, ``n_actions`` and the config fields as a numpy
        ``.npz`` archive to exactly ``path`` (no suffix is appended)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            np.savez(f, q=self.q, visits=self.visits, n_actions=np.int64(self.n_actions), **asdict(self.cfg))
        return path

    @classmethod
    def load(cls, path: str | Path) -> "TabularQAgent":
        """Exact inverse of :meth:`save` (``q``, ``visits`` and ``cfg`` round-trip)."""
        with np.load(Path(path)) as data:
            cfg = TabularQConfig(**{f.name: data[f.name].item() for f in fields(TabularQConfig)})
            agent = cls(int(data["n_actions"]), cfg)
            q, visits = data["q"], data["visits"]
        if q.shape != agent.q.shape or visits.shape != agent.visits.shape:
            raise ValueError(
                f"{path}: table shapes {q.shape} / {visits.shape} do not match "
                f"({N_STATES_A}, {agent.n_actions})"
            )
        agent.q = q.astype(np.float64, copy=False)
        agent.visits = visits.astype(np.int64, copy=False)
        return agent
