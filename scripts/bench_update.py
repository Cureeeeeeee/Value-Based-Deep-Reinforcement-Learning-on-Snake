"""Time one DQN gradient step and one ``act_epsilon`` call of a configuration
on a device (PLAN.md milestone M6: the device decision for the CNN).

Usage, from the repo root::

    python scripts/bench_update.py --config configs/dqn_B_8x8.yaml --device cuda
        [--set key=value ...] [--threads T] [--updates N] [--warmup M] [--seed S]

The environment and the agent are built from the configuration (``--device``
replaces ``agent.device``, ``--threads`` replaces ``agent.torch_threads``). A
replay buffer is filled with ``max(batch_size, 4096)`` transitions of the
uniform random policy; ``--warmup`` updates on sampled batches are discarded,
then ``--updates`` updates are timed (``torch.cuda.synchronize`` before the
clock is read on cuda), then ``--updates`` calls of ``act_epsilon`` on a batch
of ``n_envs`` observations. One line per measurement::

    update: 2.13 ms  act: 0.41 ms  (dqn_B8_W8, cuda, 2 threads, batch 256, n_envs 128)

Nothing is written under ``runs/``. The numbers depend on what else the
machine is doing: run it on an idle machine and say so when quoting them.
The M5 numbers for the MLP (docs/PROGRESS.md) were taken with probe scripts;
this script replaces them for M6 onwards.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from snake_rl.agents.dqn import ReplayBuffer  # noqa: E402
from snake_rl.agents.dqn.agent import DQNAgent  # noqa: E402
from snake_rl.config import env_seed_for, load_config  # noqa: E402
from snake_rl.env.vec_env import VecSnakeEnv  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python scripts/bench_update.py", description=__doc__.split("\n\n")[0])
    p.add_argument("--config", required=True, help="YAML experiment file (agent.kind dqn)")
    p.add_argument("--set", dest="overrides", nargs="*", action="append", default=[], metavar="KEY=VALUE")
    p.add_argument("--device", choices=("cpu", "cuda"), required=True)
    p.add_argument("--threads", type=int, default=None, help="torch threads (default: agent.torch_threads of the file)")
    p.add_argument("--updates", type=int, default=200, help="timed gradient steps (default 200)")
    p.add_argument("--warmup", type=int, default=20, help="untimed gradient steps first (default 20)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    if args.updates < 1 or args.warmup < 0:
        p.error("--updates must be >= 1 and --warmup >= 0")

    overrides = [item for group in args.overrides for item in group]
    overrides.append(f"agent.device={args.device}")
    if args.threads is not None:
        overrides.append(f"agent.torch_threads={args.threads}")
    cfg = load_config(args.config, overrides)
    if cfg.agent_kind != "dqn":
        p.error(f"agent.kind must be dqn, got {cfg.agent_kind}")
    ac, tc = cfg.agent, cfg.train
    if ac.torch_threads is not None:
        torch.set_num_threads(ac.torch_threads)

    env = VecSnakeEnv(cfg.env, tc.n_envs, seed=env_seed_for(args.seed, tc.n_envs))
    agent = DQNAgent(env.obs_shape, env.n_actions, ac, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    n_fill = max(ac.batch_size, 4096)
    buffer = ReplayBuffer(n_fill, env.obs_shape)
    obs = env.reset()
    while len(buffer) < n_fill:  # uniform random policy
        actions = rng.integers(0, env.n_actions, size=tc.n_envs)
        next_obs, reward, terminated, truncated, info = env.step(actions)
        buffer.add(obs, actions, reward, info["final_obs"], terminated)
        obs = next_obs

    def sync() -> None:
        if agent.device.type == "cuda":
            torch.cuda.synchronize()

    for _ in range(args.warmup):
        agent.update(buffer.sample(ac.batch_size, rng))
    sync()
    t0 = time.perf_counter()
    for _ in range(args.updates):
        agent.update(buffer.sample(ac.batch_size, rng))
    sync()
    update_ms = (time.perf_counter() - t0) / args.updates * 1e3

    agent.act_epsilon(obs, 0.1, rng)  # warm-up
    sync()
    t0 = time.perf_counter()
    for _ in range(args.updates):
        agent.act_epsilon(obs, 0.1, rng)
    sync()  # act_epsilon ends on the host anyway; kept for symmetry
    act_ms = (time.perf_counter() - t0) / args.updates * 1e3

    threads = torch.get_num_threads()
    print(
        f"update: {update_ms:.2f} ms  act: {act_ms:.2f} ms  "
        f"({cfg.exp}, {agent.device.type}, {threads} threads, batch {ac.batch_size}, n_envs {tc.n_envs})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
