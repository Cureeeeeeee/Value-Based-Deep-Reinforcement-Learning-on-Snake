"""Evaluation protocol (PLAN.md section 4, Part B "evaluation protocol").

``evaluate(policy, cfg, n_episodes, seed)`` plays ``n_episodes`` greedy
episodes on the scalar environment, episode ``i`` reset with
``seed + i``. The seeds are the same for every player, so all agents and
baselines are judged on the same sequence of boards. Per episode it records
score, steps, length and the end cause; the summary gives mean score with a
95 % normal-approximation confidence interval, mean steps and the cause
breakdown. Results serialise to JSON with the environment config and the git
hash so that tables are regenerated from files, never typed by hand.

``evaluate_batched(policy, cfg, n_episodes, seed)`` is the same protocol on
the vectorised environment, for learned agents that act on a batch of
observations (``policy.act_batch(obs) -> actions``). It steps
``VecSnakeEnv(cfg, n_envs=n_episodes, seed=seed)`` and keeps, for every
sub-env ``i``, only its **first** episode. Sub-env ``i`` owns the generator
``default_rng(seed + i)``, the one behind ``reset(seed=seed + i)`` of the
scalar environment, so a deterministic policy that reads nothing but the
observation gets the same ``n_episodes`` boards and returns the same episode
records as ``evaluate``. Sub-envs that finish early keep auto-resetting until
the slowest one is done; those later episodes are discarded. Keeping "the
first ``n_episodes`` episodes to finish" instead would over-represent short
episodes (a board that dies early contributes again and again while a long
episode is still running) and bias the mean score downwards.

Value estimates against realised returns (M7; van Hasselt et al. 2016)
----------------------------------------------------------------------
With ``gamma`` given, both functions also record, for a policy that exposes
``q_values(obs) -> (N, n_actions)``, what the network claims and what the
policy then collects. For every step ``t`` of a recorded episode,
``v_t = max_a Q(s_t, a)`` is the value estimate of the state the action is
chosen from (for a greedy policy ``Q(s_t, a_t)``), ``r_t`` the reward the
environment returns (the training reward of ``cfg.reward``), and after the
episode ``G_t = r_t + gamma * G_{t+1}`` with ``G_T = 0`` is the realised
discounted return from ``s_t``. ``G`` stops at the end of the episode
whether it terminated or was truncated by starvation: it is what the policy
actually collected, with no bootstrap; at gamma 0.9 the tail beyond the
100-step starvation limit weighs under 3e-5 in any case. Per episode the
record keeps ``value_start = v_0``, ``return_start = G_0`` and the means of
``v_t`` and ``G_t`` over the episode's steps (``value_visited``,
``return_visited``); ``EvalResult.value_summary`` averages them over the
episodes, the visited-state pair weighted by episode length, so that it is
the mean over all visited states. The difference ``value - return`` is the
overestimation the Double DQN study measures. In the batched protocol the
action values come from one ``q_values`` call per vec-step on the whole
batch, before ``act_batch`` chooses the actions.

CLI::

    python -m snake_rl.evaluate --policy random bfs hamiltonian --W 10 --episodes 100

writes ``runs/baselines/<policy>_W<W>_<mode>.json`` and prints a markdown
table.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from snake_rl.env.common import CAUSE_NAMES
from snake_rl.env.core import EnvConfig, SnakeEnv
from snake_rl.env.obs import ObsConfig
from snake_rl.env.rewards import RewardConfig
from snake_rl.env.vec_env import VecSnakeEnv
from snake_rl.utils import git_hash

DEFAULT_EVAL_SEED = 10_000
#: The per-episode value fields (M7), present only when evaluated with ``gamma``.
VALUE_FIELDS = ("value_start", "return_start", "value_visited", "return_visited")


@dataclass(frozen=True)
class EpisodeRecord:
    seed: int
    score: int
    steps: int
    length: int
    cause: str
    #: ``max_a Q(s_0, a)`` of the first state (M7; ``None`` without ``gamma``).
    value_start: float | None = None
    #: Realised discounted return from the first state.
    return_start: float | None = None
    #: Mean of ``max_a Q(s_t, a)`` over the episode's steps.
    value_visited: float | None = None
    #: Mean of the realised discounted return ``G_t`` over the episode's steps.
    return_visited: float | None = None


@dataclass
class EvalResult:
    policy: str
    env: dict[str, Any]
    seed: int
    episodes: list[EpisodeRecord] = field(default_factory=list)
    git: str = "unknown"
    wall_time_s: float = 0.0

    # ----------------------------------------------------------- summaries
    @property
    def scores(self) -> np.ndarray:
        return np.array([e.score for e in self.episodes], dtype=np.float64)

    @property
    def steps(self) -> np.ndarray:
        return np.array([e.steps for e in self.episodes], dtype=np.float64)

    @property
    def mean_score(self) -> float:
        return float(self.scores.mean())

    @property
    def ci95(self) -> float:
        """Half-width of the 95 % CI of the mean score (normal approximation)."""
        n = len(self.episodes)
        if n < 2:
            return float("nan")
        return float(1.96 * self.scores.std(ddof=1) / np.sqrt(n))

    @property
    def cause_fractions(self) -> dict[str, float]:
        n = max(len(self.episodes), 1)
        return {c: sum(e.cause == c for e in self.episodes) / n for c in CAUSE_NAMES if c != "none"}

    def summary(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "n_episodes": len(self.episodes),
            "mean_score": self.mean_score,
            "ci95": self.ci95,
            "median_score": float(np.median(self.scores)) if self.episodes else float("nan"),
            "max_score": int(self.scores.max()) if self.episodes else 0,
            "mean_steps": float(self.steps.mean()) if self.episodes else float("nan"),
            "cause_fractions": self.cause_fractions,
        }

    def value_summary(self) -> dict[str, float] | None:
        """The value estimates against the realised returns (module docstring).

        ``value_start`` / ``return_start``: means over the episodes of the
        first-state quantities. ``value_visited`` / ``return_visited``: means
        over all visited states (the per-episode means weighted by the
        episode's steps). ``None`` when there are no episodes or an episode
        was evaluated without ``gamma``.
        """
        if not self.episodes or any(getattr(e, name) is None for e in self.episodes for name in VALUE_FIELDS):
            return None
        steps = self.steps
        total_steps = float(steps.sum())
        return {
            "value_start": float(np.mean([e.value_start for e in self.episodes])),
            "return_start": float(np.mean([e.return_start for e in self.episodes])),
            "value_visited": float(np.dot([e.value_visited for e in self.episodes], steps) / total_steps),
            "return_visited": float(np.dot([e.return_visited for e in self.episodes], steps) / total_steps),
        }

    # --------------------------------------------------------------- I/O
    def to_dict(self) -> dict[str, Any]:
        d = {
            "policy": self.policy,
            "env": self.env,
            "seed": self.seed,
            "git": self.git,
            "wall_time_s": self.wall_time_s,
            "summary": self.summary(),
        }
        values = self.value_summary()
        if values is not None:
            d["values"] = values
        d["episodes"] = [_episode_dict(e) for e in self.episodes]
        return d

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "EvalResult":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            policy=d["policy"],
            env=d["env"],
            seed=d["seed"],
            episodes=[EpisodeRecord(**e) for e in d["episodes"]],
            git=d.get("git", "unknown"),
            wall_time_s=d.get("wall_time_s", 0.0),
        )


def _episode_dict(e: EpisodeRecord) -> dict[str, Any]:
    """The JSON form of a record: the value fields only when present, so a
    file written without ``gamma`` is what it was before M7."""
    d = asdict(e)
    for name in VALUE_FIELDS:
        if d[name] is None:
            del d[name]
    return d


def env_config_to_dict(cfg: EnvConfig) -> dict[str, Any]:
    return asdict(cfg)


def env_config_from_dict(d: dict[str, Any]) -> EnvConfig:
    return EnvConfig(
        W=d["W"],
        action_mode=d["action_mode"],
        starve_limit=d["starve_limit"],
        obs=ObsConfig(**d["obs"]),
        reward=RewardConfig(**d["reward"]),
    )


# ------------------------------------------------------- value bookkeeping
def _check_gamma(policy, gamma: float | None) -> None:
    if gamma is None:
        return
    if isinstance(gamma, bool) or not isinstance(gamma, (int, float, np.floating)) or not 0.0 <= gamma <= 1.0:
        raise ValueError(f"gamma must be None or a number in [0, 1], got {gamma!r}")
    if not callable(getattr(policy, "q_values", None)):
        raise TypeError(
            f"value estimates need a policy with q_values(obs) -> (N, n_actions); {type(policy).__name__} has none"
        )


def _value_fields(values: list[float], rewards: list[float], gamma: float) -> dict[str, float]:
    """The four per-episode value fields from the per-step ``max_a Q`` and rewards."""
    returns = np.empty(len(rewards), dtype=np.float64)
    g = 0.0
    for t in range(len(rewards) - 1, -1, -1):  # G_t = r_t + gamma * G_{t+1}, G_T = 0
        g = rewards[t] + gamma * g
        returns[t] = g
    return {
        "value_start": float(values[0]),
        "return_start": float(returns[0]),
        "value_visited": float(np.mean(values)),
        "return_visited": float(returns.mean()),
    }


def evaluate(
    policy, cfg: EnvConfig, n_episodes: int = 100, seed: int = DEFAULT_EVAL_SEED, gamma: float | None = None
) -> EvalResult:
    """Play ``n_episodes`` greedy episodes; episode ``i`` uses ``reset(seed=seed + i)``.

    With ``gamma`` the records also carry the value estimates against the
    realised returns (module docstring); the policy must have ``q_values``.
    """
    _check_gamma(policy, gamma)
    env = SnakeEnv(cfg)
    result = EvalResult(policy=policy.name, env=env_config_to_dict(cfg), seed=seed, git=git_hash())
    t0 = time.perf_counter()
    for i in range(n_episodes):
        ep_seed = seed + i
        obs, info = env.reset(seed=ep_seed)
        if hasattr(policy, "reset"):
            policy.reset()
        values: list[float] = []
        rewards: list[float] = []
        while True:
            if gamma is not None:
                values.append(float(policy.q_values(obs[None])[0].max()))
            obs, reward, terminated, truncated, info = env.step(policy.act(obs, env))
            if gamma is not None:
                rewards.append(float(reward))
            if terminated or truncated:
                break
        extra = _value_fields(values, rewards, gamma) if gamma is not None else {}
        result.episodes.append(
            EpisodeRecord(
                seed=ep_seed, score=info["score"], steps=info["steps"], length=info["length"], cause=info["cause"],
                **extra,
            )
        )
    result.wall_time_s = time.perf_counter() - t0
    return result


def evaluate_batched(
    policy, cfg: EnvConfig, n_episodes: int = 100, seed: int = DEFAULT_EVAL_SEED, gamma: float | None = None
) -> EvalResult:
    """Greedy evaluation on the vectorised env: the first episode of each of
    ``n_episodes`` sub-envs, sub-env ``i`` seeded ``seed + i``.

    ``policy`` needs ``name`` and ``act_batch(obs (N, *obs_shape)) -> (N,)``
    integer actions and must be a deterministic function of the observation;
    the result then equals ``evaluate(policy, cfg, n_episodes, seed)`` episode
    by episode (see the module docstring). Episodes are ordered by ``i``.
    ``policy.reset()``, if it exists, is called once before the first step
    (all first episodes start together). With ``gamma`` (module docstring)
    the policy must also have ``q_values``; it is called once per vec-step on
    the whole batch, before ``act_batch``.
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be at least 1, got {n_episodes}")
    _check_gamma(policy, gamma)
    env = VecSnakeEnv(cfg, n_envs=n_episodes, seed=seed)
    result = EvalResult(policy=policy.name, env=env_config_to_dict(cfg), seed=seed, git=git_hash())
    t0 = time.perf_counter()
    if hasattr(policy, "reset"):
        policy.reset()
    obs = env.reset()
    # ``info`` reports the counters of the episode that just ended (the state
    # arrays of the env already hold the next one), so the record of a sub-env
    # is copied from ``info`` on the step that ends its first episode.
    keys = ("score", "steps", "length", "cause")
    first = {key: np.zeros(n_episodes, dtype=np.int64) for key in keys}
    pending = np.ones(n_episodes, dtype=bool)  # first episode still running
    step_values: list[np.ndarray] = []  # per vec-step: max_a Q of every sub-env (with gamma)
    step_rewards: list[np.ndarray] = []
    while pending.any():
        if gamma is not None:
            step_values.append(policy.q_values(obs).max(axis=1))
        obs, reward, _, _, info = env.step(policy.act_batch(obs))
        if gamma is not None:
            step_rewards.append(np.asarray(reward, dtype=np.float64))
        ended = pending & info["done"]
        for key in keys:
            first[key][ended] = info[key][ended]
        pending &= ~ended
    columns = [first[key].tolist() for key in keys]  # Python ints: JSON-serialisable
    if gamma is not None:
        all_values = np.stack(step_values)  # (vec-steps, n_episodes)
        all_rewards = np.stack(step_rewards)
    for i, (score, steps, length, cause) in enumerate(zip(*columns)):
        extra = {}
        if gamma is not None:
            # The first episode of sub-env i occupies its first ``steps`` vec-steps.
            extra = _value_fields(all_values[:steps, i].tolist(), all_rewards[:steps, i].tolist(), gamma)
        result.episodes.append(
            EpisodeRecord(seed=seed + i, score=score, steps=steps, length=length, cause=CAUSE_NAMES[cause], **extra)
        )
    result.wall_time_s = time.perf_counter() - t0
    return result


def markdown_table(results: list[EvalResult]) -> str:
    causes = [c for c in CAUSE_NAMES if c != "none"]
    head = "| policy | episodes | score mean +- 95% CI | median | max | mean steps | " + " | ".join(causes) + " |"
    sep = "|" + "---|" * (6 + len(causes))
    rows = [head, sep]
    for r in results:
        s = r.summary()
        frac = " | ".join(f"{s['cause_fractions'][c]:.2f}" for c in causes)
        rows.append(
            f"| {s['policy']} | {s['n_episodes']} | {s['mean_score']:.2f} +- {s['ci95']:.2f} | "
            f"{s['median_score']:.1f} | {s['max_score']} | {s['mean_steps']:.1f} | {frac} |"
        )
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    from snake_rl.agents.baselines import BASELINE_NAMES, make_policy

    p = argparse.ArgumentParser(description="Greedy evaluation of baseline players.")
    p.add_argument("--policy", nargs="+", choices=BASELINE_NAMES, default=list(BASELINE_NAMES))
    p.add_argument("--W", type=int, default=10)
    p.add_argument("--action-mode", choices=["relative", "absolute"], default="relative")
    p.add_argument("--episodes", type=int, default=100)
    p.add_argument("--seed", type=int, default=DEFAULT_EVAL_SEED, help="evaluation seed; episode i uses seed+i")
    p.add_argument("--out-dir", default="runs/baselines")
    args = p.parse_args(argv)

    cfg = EnvConfig(W=args.W, action_mode=args.action_mode)
    results = []
    for name in args.policy:
        res = evaluate(make_policy(name, seed=args.seed), cfg, args.episodes, args.seed)
        path = res.save(Path(args.out_dir) / f"{name}_W{args.W}_{args.action_mode}.json")
        s = res.summary()
        print(f"{name:<12} score {s['mean_score']:.2f} +- {s['ci95']:.2f}  steps {s['mean_steps']:.0f}  "
              f"causes {s['cause_fractions']}  ({res.wall_time_s:.1f}s) -> {path}")
        results.append(res)
    print()
    print(markdown_table(results))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
