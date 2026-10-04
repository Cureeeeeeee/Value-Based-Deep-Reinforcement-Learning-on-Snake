"""Deep Q-Network agent (PLAN.md milestone M5; Mnih et al. 2015).

Q-learning with a neural network ``Q(s, a; theta)`` in place of the table. The
agent holds two networks of the same architecture - the online network
``theta``, which acts and is trained, and the target network ``theta^-``, a
periodically refreshed copy - and an Adam optimiser. It holds no RNG: the
initial weights are a function of ``seed``, exploration and replay sampling
draw from generators the caller passes in.

One gradient step (:meth:`DQNAgent.update`)
-------------------------------------------
For a batch of transitions ``(s, a, r, s', terminated)``::

    q_sa   = Q(s, a; theta)
    target = r + gamma * (0 if terminated else max_a' Q(s', a'; theta^-))
    loss   = mean over the batch of Huber(q_sa - target)

    Huber(d) = d^2 / 2 if |d| <= 1 else |d| - 1 / 2

followed by one Adam step on ``theta`` with the global gradient norm clipped
to ``grad_clip``. The target is treated as a constant: no gradient flows
through it (a semi-gradient method, like the tabular update). The Huber loss
is quadratic for small TD errors and linear for large ones, so the derivative
of a transition's loss with respect to ``q_sa`` is at most 1 in magnitude
however wrong the estimate is (the "error clipping" of Mnih et al. 2015).

This is the tabular update of ``tabular_q.py`` with the table lookup replaced
by a network and the step ``alpha * td`` replaced by a gradient step on a
regression loss. One consequence of the replacement motivates both stabilisers
below: a gradient step on one input changes the values of other inputs as
well. The table also computes its target from the values it is updating, but
an update to one cell leaves every other cell unchanged; in the network the
update for ``(s, a)`` also moves the value of the successor ``s'`` from which
its own target is computed.

Replay buffer - what it is for
------------------------------
Stochastic gradient descent assumes samples that are roughly independent and
come from one distribution. Consecutive transitions of a game are neither:
successive positions of one game differ by the head cell and the tail cell,
and the distribution of visited states changes whenever the greedy policy
changes. Training on the stream as it
arrives pulls the network towards the most recent moments of play, at the
expense of what it learned before. The buffer (``replay.py``) keeps the most recent
``buffer_size`` transitions and every gradient step draws a uniform batch from
it: a batch mixes many episodes and many past policies, and a transition is
drawn many times before it leaves the buffer (64 times on average in the
shipped configurations: 32 batches of 256 per 128 collected transitions). The
``n_envs`` parallel boards already decorrelate one vec-step across boards;
they do not remove the correlation over time or the drift with the current
policy.

Sampling old transitions is admissible because Q-learning is off-policy: the
target of a stored transition takes the max over the successor's actions and
does not involve the action the old policy took next, so the transition does
not go stale (a SARSA target would). The data distribution still matters. With
function approximation the weights the regression settles on depend on which
states and actions the buffer holds - the off-policy element of the deadly
triad below. And on observation A a code is not a Markov state, so the average
reward and successor of a (code, action) pair depend on the policy that
collected the data.

Target network - what it is for
-------------------------------
With a single network the target ``r + gamma * max_a' Q(s', a'; theta)`` moves
with every step on ``theta``, and often in the same direction as ``q_sa``,
because ``s`` and ``s'`` are similar inputs. The regression chases a target
that it moves itself, which can oscillate or diverge. The target network
freezes the target: ``theta^-`` is an exact copy of ``theta`` made every
``target_update_every`` gradient steps, so between two copies the target
function is fixed and the problem is a regression towards it (the buffer keeps
changing). ``target_update_every = 1`` copies after every step: no separate
target network in effect.

DQN combines the three elements of the "deadly triad" (Sutton & Barto 2018,
section 11.3) - function approximation, bootstrapping and off-policy training
- which together can make value estimates diverge. Neither device removes an
element. The target network keeps the bootstrap but takes it from a delayed
copy, so an update no longer moves its own target at once. Replay addresses a
different problem, correlated and non-stationary samples, and makes the
training data more off-policy, not less. Mnih et al. (2015) introduce the pair
against the instability of Q-learning with a neural network, which they
attribute to correlated observations, to a data distribution that shifts with
the policy, and to the correlation between action values and their targets;
replay itself goes back to Lin (1992). Neither comes with a convergence
guarantee, and nothing here claims one. The project measures what each
contributes on Snake by removing them (M6: +-replay, +-target network).

terminated is not truncated
---------------------------
``terminated`` (wall, self collision, win) means there is no future: the
successor is worth zero by definition. ``truncated`` (starvation) is a time
limit that the agent cannot observe; the successor is an ordinary state from
which the game would have continued, so the target bootstraps from it like any
other transition. Zeroing it would teach the network that ordinary states are
worthless whenever a hidden clock happens to run out. The caller therefore
stores the environment's ``terminated`` flag (never ``done``) and takes
``next_obs`` from ``info["final_obs"]`` of ``VecSnakeEnv.step``: the
observation that ``step`` returns for a finished sub-env is the first
observation of the next episode, not the successor.

Double DQN (``cfg.double``; van Hasselt, Guez & Silver 2016)
------------------------------------------------------------
::

    a*     = argmax_a' Q(s', a'; theta)                      # online network selects
    target = r + gamma * (0 if terminated else Q(s', a*; theta^-))   # target network evaluates

The plain target uses one network both to choose the best successor action and
to value it. The maximum over noisy estimates is biased upwards
(``E[max_a X_a] >= max_a E[X_a]``): the action whose value happens to be
overestimated is the one that gets selected, and bootstrapping carries the
excess backwards. Choosing with the online network and valuing with the target
network means an action is no longer scored by the same errors that selected
it. This is meant to reduce the overestimation; it cannot remove it, since
the two networks are correlated (one is an old copy of the other). M7 measures
the effect on Snake.

Relation to the tabular agent on the same features
--------------------------------------------------
On observation A the network sees the same 11 binary features as
``TabularQAgent``, hence the same codes (at most 256 occur as a state the
agent acts in), and every greedy policy it expresses is a function of that
code - a policy the table can express too. A difference in score between the
two is therefore a difference in how the action values are represented and
learned (shared weights against one number per code, replay, target network,
optimiser, loss, training data), not in the state representation: what
observation A discards (body layout, distances) is as invisible to the network
as to the table, so neither can represent the optimal action values of the
game. If the network scores lower, limited capacity is not excluded by this
argument.

Acting
------
Greedy ties go to the lowest action index (``np.argmax`` on the action values
copied to the host). :meth:`DQNAgent.act` is :meth:`DQNAgent.act_batch` on a
batch of one, so the two share one code path and one tie rule. The action
values of one observation can differ in the last few bits between batch sizes
(the linear-algebra library picks its kernel by matrix size; measured on the
CPU: 5e-8 for an untrained network with ``|Q| < 0.2``, up to 1e-6 for the
trained 8x8 networks with ``|Q|`` up to 2), so the two can disagree only where
two action values coincide to within that rounding. No greedy action changed
in the five final 8x8 models (smallest gap between the two best actions
3.4e-4) or in their 205 checkpoints.
"""

from __future__ import annotations

import copy
import math
from dataclasses import asdict, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from snake_rl.agents.dqn.config import DQNConfig
from snake_rl.agents.dqn.networks import build_q_network
from snake_rl.agents.dqn.replay import Batch
from snake_rl.env.core import SnakeEnv


def resolve_device(name: str) -> torch.device:
    """``"auto"`` -> cuda if available, else cpu; ``"cpu"`` / ``"cuda"`` as named."""
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError('device "cuda" requested but torch.cuda.is_available() is False')
        return torch.device("cuda")
    raise ValueError(f'device must be "auto", "cpu" or "cuda", got {name!r}')


class DQNAgent:
    """DQN with a hard-synchronised target network. See the module docstring.

    Public state: ``q_net`` (online), ``target_net``, ``optimizer`` and
    ``grad_steps``, the number of gradient steps taken so far.
    """

    name = "dqn"

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        cfg: DQNConfig | None = None,
        seed: int = 0,
    ) -> None:
        if isinstance(n_actions, bool) or not isinstance(n_actions, (int, np.integer)) or n_actions < 1:
            raise ValueError(f"n_actions must be a positive integer, got {n_actions!r}")
        self.cfg = cfg if cfg is not None else DQNConfig()
        self.obs_shape = tuple(int(d) for d in obs_shape)
        self.n_actions = int(n_actions)
        self.device = resolve_device(self.cfg.device)

        # The initial weights are a function of ``seed`` alone: built on the CPU
        # from the freshly seeded default generator, whose state is restored
        # afterwards, so creating an agent leaves the global torch RNG untouched.
        # (Seeding the generator object, not torch.manual_seed, keeps the CUDA
        # generators out of it.)
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(int(seed))
            q_net = build_q_network(self.obs_shape, self.n_actions, self.cfg.hidden)
        self.q_net = q_net.to(self.device)
        self.target_net = copy.deepcopy(self.q_net)
        self.target_net.requires_grad_(False)  # only ever written by a copy of q_net
        self.optimizer = torch.optim.Adam(
            self.q_net.parameters(), lr=self.cfg.lr, fused=self.device.type == "cuda"
        )
        self.grad_steps = 0

    # ------------------------------------------------------------- acting
    def reset(self) -> None:
        """Policy protocol (start of an evaluation episode). Nothing to reset."""

    def q_values(self, obs: np.ndarray) -> np.ndarray:
        """Action values ``(N, n_actions)`` float32 of the online network for
        observations ``(N, *obs_shape)``. No gradient is recorded."""
        obs = np.asarray(obs)
        if obs.ndim != len(self.obs_shape) + 1 or obs.shape[1:] != self.obs_shape:
            raise ValueError(f"q_values() takes observations of shape (N, {self._shape_text()}), got shape {obs.shape}")
        with torch.no_grad():
            q = self.q_net(torch.as_tensor(obs, dtype=torch.float32, device=self.device))
        return q.cpu().numpy()

    def act(self, obs: np.ndarray, env: SnakeEnv | None = None) -> int:
        """Greedy action for one observation ``obs_shape`` (Policy protocol; ``env`` is unused)."""
        obs = np.asarray(obs)
        if obs.shape != self.obs_shape:
            raise ValueError(f"act() takes one observation of shape ({self._shape_text()}), got shape {obs.shape}")
        return int(self.act_batch(obs[None])[0])

    def act_batch(self, obs: np.ndarray) -> np.ndarray:
        """Greedy actions ``(N,)`` int64 for observations ``(N, *obs_shape)``;
        ties go to the lowest action index."""
        return self.q_values(obs).argmax(axis=1).astype(np.int64, copy=False)

    def act_epsilon(self, obs: np.ndarray, epsilon: float, rng: np.random.Generator) -> np.ndarray:
        """Epsilon-greedy actions ``(N,)`` int64 for observations ``(N, *obs_shape)``.

        RNG protocol (fixed, the one of ``TabularQAgent.act_epsilon``): first
        ``rng.random(N) < epsilon`` decides who explores, then
        ``rng.integers(0, n_actions, size=N)`` proposes a random action for
        everyone. Both draws happen on every call, in this order, whatever
        ``epsilon`` is.
        """
        greedy = self.act_batch(obs)
        n = greedy.shape[0]
        explore = rng.random(n) < epsilon
        random_actions = rng.integers(0, self.n_actions, size=n)
        return np.where(explore, random_actions, greedy)

    # ----------------------------------------------------------- learning
    def update(self, batch: Batch) -> dict[str, float]:
        """One gradient step on a batch of transitions (module docstring).

        ``batch.terminated`` is the environment's ``terminated`` flag only: a
        starvation-truncated transition has ``terminated = False`` and
        bootstraps from ``batch.next_obs``, which must be the true successor
        (``info["final_obs"]``).

        Returns ``loss``, ``mean_q`` (mean of ``q_sa``), ``mean_target`` and
        ``grad_norm`` (global gradient norm before clipping) as Python floats.
        A non-finite loss or gradient norm raises ``FloatingPointError``
        before the optimiser step: the networks, the optimiser state and
        ``grad_steps`` are left as they were.
        """
        cfg, device = self.cfg, self.device
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=device)
        terminated = torch.as_tensor(batch.terminated, dtype=torch.bool, device=device)

        with torch.no_grad():  # the target is a constant of the regression
            next_q = self.target_net(next_obs)
            if cfg.double:
                # Double DQN: the online network selects, the target network evaluates.
                best = self.q_net(next_obs).argmax(dim=1, keepdim=True)
                next_value = next_q.gather(1, best).squeeze(1)
            else:
                next_value = next_q.max(dim=1).values
            # Bootstrap unless terminated; a truncated transition keeps its future value.
            target = rewards + cfg.gamma * next_value.masked_fill(terminated, 0.0)

        q_sa = self.q_net(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.smooth_l1_loss(q_sa, target, beta=1.0)  # Huber, mean over the batch

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), cfg.grad_clip)

        # One host transfer for all four statistics, taken before the step so
        # that a NaN / inf aborts with the weights still intact.
        loss_value, mean_q, mean_target, norm_value = torch.stack(
            (loss.detach(), q_sa.detach().mean(), target.mean(), grad_norm)
        ).tolist()
        if not (math.isfinite(loss_value) and math.isfinite(norm_value)):
            raise FloatingPointError(
                f"non-finite DQN update at gradient step {self.grad_steps + 1}: "
                f"loss {loss_value}, gradient norm {norm_value}"
            )
        self.optimizer.step()
        self.grad_steps += 1
        if self.grad_steps % cfg.target_update_every == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return {"loss": loss_value, "mean_q": mean_q, "mean_target": mean_target, "grad_norm": norm_value}

    # ---------------------------------------------------------------- I/O
    def save(self, path: str | Path, include_optimizer: bool = True) -> Path:
        """Write a checkpoint to exactly ``path`` (parent directories are created).

        The file holds plain tensors and primitives only, so it loads with
        ``torch.load(path, weights_only=True)``: ``q_net`` / ``target_net``
        (state dicts), ``optimizer`` (state dict, only when
        ``include_optimizer``), ``grad_steps``, ``obs_shape``, ``n_actions``
        and ``config`` (the ``DQNConfig`` fields). Tensors are saved as CPU
        tensors, so a checkpoint written on the GPU opens on any machine.
        """
        payload: dict[str, Any] = {
            "q_net": _to_cpu(self.q_net.state_dict()),
            "target_net": _to_cpu(self.target_net.state_dict()),
            "grad_steps": self.grad_steps,
            "obs_shape": list(self.obs_shape),
            "n_actions": self.n_actions,
            "config": {**asdict(self.cfg), "hidden": list(self.cfg.hidden)},
        }
        if include_optimizer:
            payload["optimizer"] = _to_cpu(self.optimizer.state_dict())
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:  # a file object: torch.save writes exactly here
            torch.save(payload, f)
        return path

    @classmethod
    def load(cls, path: str | Path, device: str | None = None) -> "DQNAgent":
        """Inverse of :meth:`save`: networks, ``grad_steps`` and ``cfg`` round-trip exactly.

        ``device`` (``"auto"`` / ``"cpu"`` / ``"cuda"``) replaces ``cfg.device``
        of the checkpoint, e.g. to open a GPU run on a CPU-only machine. A
        checkpoint saved without the optimiser gets a fresh one.
        """
        payload = torch.load(Path(path), map_location="cpu", weights_only=True)
        # A field added to DQNConfig after the checkpoint was written takes its default.
        known = {f.name for f in fields(DQNConfig)}
        cfg = DQNConfig(**{key: value for key, value in payload["config"].items() if key in known})
        if device is not None:
            cfg = replace(cfg, device=device)
        agent = cls(tuple(payload["obs_shape"]), payload["n_actions"], cfg)
        agent.q_net.load_state_dict(payload["q_net"])
        agent.target_net.load_state_dict(payload["target_net"])
        agent.grad_steps = int(payload["grad_steps"])
        if "optimizer" in payload:
            state = payload["optimizer"]
            # The fused implementation is a property of the device in use now,
            # not of the device the checkpoint was written on.
            for group in state["param_groups"]:
                group["fused"] = agent.device.type == "cuda"
            agent.optimizer.load_state_dict(state)
        return agent

    def _shape_text(self) -> str:
        return ", ".join(str(d) for d in self.obs_shape)


def _to_cpu(obj: Any) -> Any:
    """Copy of a (nested) state dict with every tensor detached on the CPU."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().clone()
    if isinstance(obj, dict):
        return {key: _to_cpu(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(value) for value in obj)
    return obj
