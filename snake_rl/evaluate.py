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


@dataclass(frozen=True)
class EpisodeRecord:
    seed: int
    score: int
    steps: int
    length: int
    cause: str


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

    # --------------------------------------------------------------- I/O
    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "env": self.env,
            "seed": self.seed,
            "git": self.git,
            "wall_time_s": self.wall_time_s,
            "summary": self.summary(),
            "episodes": [asdict(e) for e in self.episodes],
        }

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


def evaluate(policy, cfg: EnvConfig, n_episodes: int = 100, seed: int = DEFAULT_EVAL_SEED) -> EvalResult:
    """Play ``n_episodes`` greedy episodes; episode ``i`` uses ``reset(seed=seed + i)``."""
    env = SnakeEnv(cfg)
    result = EvalResult(policy=policy.name, env=env_config_to_dict(cfg), seed=seed, git=git_hash())
    t0 = time.perf_counter()
    for i in range(n_episodes):
        ep_seed = seed + i
        obs, info = env.reset(seed=ep_seed)
        if hasattr(policy, "reset"):
            policy.reset()
        while True:
            obs, _, terminated, truncated, info = env.step(policy.act(obs, env))
            if terminated or truncated:
                break
        result.episodes.append(
            EpisodeRecord(seed=ep_seed, score=info["score"], steps=info["steps"], length=info["length"], cause=info["cause"])
        )
    result.wall_time_s = time.perf_counter() - t0
    return result


def evaluate_batched(policy, cfg: EnvConfig, n_episodes: int = 100, seed: int = DEFAULT_EVAL_SEED) -> EvalResult:
    """Greedy evaluation on the vectorised env: the first episode of each of
    ``n_episodes`` sub-envs, sub-env ``i`` seeded ``seed + i``.

    ``policy`` needs ``name`` and ``act_batch(obs (N, *obs_shape)) -> (N,)``
    integer actions and must be a deterministic function of the observation;
    the result then equals ``evaluate(policy, cfg, n_episodes, seed)`` episode
    by episode (see the module docstring). Episodes are ordered by ``i``.
    ``policy.reset()``, if it exists, is called once before the first step
    (all first episodes start together).
    """
    if n_episodes < 1:
        raise ValueError(f"n_episodes must be at least 1, got {n_episodes}")
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
    while pending.any():
        obs, _, _, _, info = env.step(policy.act_batch(obs))
        ended = pending & info["done"]
        for key in keys:
            first[key][ended] = info[key][ended]
        pending &= ~ended
    columns = [first[key].tolist() for key in keys]  # Python ints: JSON-serialisable
    for i, (score, steps, length, cause) in enumerate(zip(*columns)):
        result.episodes.append(
            EpisodeRecord(seed=seed + i, score=score, steps=steps, length=length, cause=CAUSE_NAMES[cause])
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
