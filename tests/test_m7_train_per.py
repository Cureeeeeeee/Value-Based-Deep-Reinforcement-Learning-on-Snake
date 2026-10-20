"""The prioritised-replay training loop (M7, spec D6) of ``snake_rl.train``,
and the uniform loop left as it was.

The PER loop, from the spec::

    buffer = PrioritizedReplayBuffer(buffer_size, obs_shape, per_alpha, per_eps)
    for k = 0, 1, ...:                                     # env_steps_before = k * n_envs
        epsilon = linear_schedule(k * n_envs, eps_start, eps_end, eps_decay_fraction * total_env_steps)
        beta    = linear_schedule(k * n_envs, per_beta_start, 1.0, total_env_steps)
        actions = agent.act_epsilon(obs, epsilon, explore_rng)
        next_obs, reward, terminated, truncated, info = env.step(actions)
        buffer.add(obs, actions, reward, info["final_obs"], terminated)
        obs = next_obs
        if (k + 1) * n_envs >= learning_starts:
            G times:
                batch = buffer.sample(batch_size, sample_rng, beta)
                stats = agent.update(batch, batch.weights)
                buffer.update_priorities(batch.indices, stats["td_abs"])

A tiny PER run (``TINY`` of ``tests/test_train_dqn.py`` plus ``agent.per=true``
and a visibly annealing ``per_beta_start``) is trained once with spies on
``PrioritizedReplayBuffer.sample`` / ``update_priorities`` and
``DQNAgent.update`` (none changes what the run does); an independent loop
written from the pseudo-code above reproduces the run exactly. Conventions:
CPU, one torch thread, pytest temporary directories, ``git_hash`` patched.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

import snake_rl.evaluate as evaluate_module
import snake_rl.train as train_module
from snake_rl.agents.dqn import PrioritizedBatch, PrioritizedReplayBuffer, ReplayBuffer
from snake_rl.agents.dqn.agent import DQNAgent
from snake_rl.agents.dqn.replay import Batch
from snake_rl.agents.schedules import linear_schedule
from snake_rl.config import load_config
from snake_rl.env import VecSnakeEnv
from snake_rl.evaluate import EvalResult, evaluate_batched
from snake_rl.train import DQNWindow, train
from tests.test_train_dqn import (
    FAKE_GIT,
    HEADER,
    RUN_ENTRIES,
    SMOKE,
    TINY,
    UPDATE_COLUMNS,
    VALUE_COLUMNS,
    VALUE_FIELDS,
    assert_same_run,
    checkpoint_names,
    expected_grad_steps,
    final_episodes,
    flat,
    metrics_without_times,
    read_metrics,
    read_run_config,
)

#: TINY (8 boards, 2400 env steps, learning from 1000 on, 2 gradient steps per vec-step) with prioritised
#: replay; beta starts at 0.5 so that the annealing to 1.0 is visible, alpha and eps away from their defaults.
PER = (*TINY, "exp=per_dqn", "agent.per=true", "agent.per_beta_start=0.5", "agent.per_alpha=0.7", "agent.per_eps=1.0e-5")
PER_SEED = 5
#: A short uniform run for the "nothing changed without PER" checks.
PLAIN = (*TINY, "exp=plain_dqn", "train.total_env_steps=1600")


# ----------------------------------------------------------------- fixtures
@pytest.fixture(scope="module", autouse=True)
def fast_git_hash():
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(train_module, "git_hash", lambda: FAKE_GIT)
        patch.setattr(evaluate_module, "git_hash", lambda: FAKE_GIT)
        yield


@pytest.fixture(scope="module", autouse=True)
def restore_torch_threads():
    before = torch.get_num_threads()
    yield
    torch.set_num_threads(before)


@pytest.fixture(autouse=True)
def isolated_cwd(tmp_path, monkeypatch):
    """A relative ``runs`` directory can only ever land in ``tmp_path``."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(scope="module")
def per_run(tmp_path_factory):
    """The PER run, with spies on the prioritised buffer (construction, ``sample``, ``update_priorities``)
    and on ``DQNAgent.update``; ``events`` lists every call in order."""
    runs = tmp_path_factory.mktemp("per_runs")
    cfg = load_config(SMOKE, PER)
    events, buffers = [], []
    real_init = PrioritizedReplayBuffer.__init__
    real_sample = PrioritizedReplayBuffer.sample
    real_update_priorities = PrioritizedReplayBuffer.update_priorities
    real_update = DQNAgent.update

    def spy_init(self, *args, **kwargs):
        real_init(self, *args, **kwargs)
        buffers.append(self)

    def spy_sample(self, batch_size, rng, beta):
        state_before = rng.bit_generator.state
        batch = real_sample(self, batch_size, rng, beta)
        events.append(SimpleNamespace(kind="sample", buffer=self, batch_size=batch_size, rng=rng, state_before=state_before,
                                      beta=beta, batch=batch, indices=np.array(batch.indices), weights=np.array(batch.weights),
                                      size=len(self)))
        return batch

    def spy_update(self, batch, weights=None):
        out = real_update(self, batch, weights)
        events.append(SimpleNamespace(kind="update", batch=batch, weights=None if weights is None else np.array(weights),
                                      grad_steps=self.grad_steps,
                                      out={k: (np.array(v) if isinstance(v, np.ndarray) else v) for k, v in out.items()}))
        return out

    def spy_update_priorities(self, indices, td_abs):
        events.append(SimpleNamespace(kind="priorities", buffer=self, indices=np.array(indices), td_abs=np.array(td_abs)))
        return real_update_priorities(self, indices, td_abs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(PrioritizedReplayBuffer, "__init__", spy_init)
        patch.setattr(PrioritizedReplayBuffer, "sample", spy_sample)
        patch.setattr(PrioritizedReplayBuffer, "update_priorities", spy_update_priorities)
        patch.setattr(DQNAgent, "update", spy_update)
        run_dir = train(cfg, PER_SEED, runs, quiet=True)
    return SimpleNamespace(cfg=cfg, seed=PER_SEED, runs=runs, dir=run_dir, events=events, buffers=buffers)


def by_kind(run, kind: str) -> list:
    return [e for e in run.events if e.kind == kind]


# ---------------------------------------------------------------- run files
def test_per_run_writes_the_run_files_and_the_metrics_shape(per_run):
    cfg = per_run.cfg
    assert cfg.agent.per and cfg.agent.replay and cfg.agent.per_beta_start == 0.5 and cfg.agent.per_alpha == 0.7
    assert per_run.dir == per_run.runs / "per_dqn" / str(PER_SEED)
    assert sorted(p.name for p in per_run.dir.iterdir()) == RUN_ENTRIES
    assert checkpoint_names(per_run.dir) == [f"step_{s:08d}.pt" for s in (0, 800, 1_600, 2_400)]
    recorded = read_run_config(per_run.dir)
    assert recorded["agent"]["per"] is True and recorded["agent"]["per_beta_start"] == 0.5
    assert load_config(per_run.dir / "config.yaml") == cfg
    lines = (per_run.dir / "metrics.csv").read_text(encoding="utf-8").splitlines()
    assert lines[0] == HEADER
    rows = read_metrics(per_run.dir)
    assert [r["env_steps"] for r in rows] == [0, 800, 1_600, 2_400]
    assert [r["grad_steps"] for r in rows] == [0, 0, 152, 352]  # 8 boards: 2 gradient steps per vec-step from 1000 on
    assert [r["grad_steps"] for r in rows] == [expected_grad_steps(cfg, r["vec_steps"]) for r in rows]
    assert [r["buffer_size"] for r in rows] == [0, 800, 1_000, 1_000]
    for row in rows[:2]:
        assert all(row[name] is None for name in UPDATE_COLUMNS)
    for row in rows[2:]:
        assert all(np.isfinite(row[name]) for name in UPDATE_COLUMNS) and row["loss"] > 0 and row["grad_norm"] > 0
    agent = DQNAgent.load(per_run.dir / "model.pt")
    assert agent.grad_steps == 352 and agent.cfg == cfg.agent and agent.cfg.per is True


def test_per_run_uses_one_prioritised_buffer_with_the_configured_alpha_and_eps(per_run):
    cfg = per_run.cfg
    assert len(per_run.buffers) == 1
    buffer = per_run.buffers[0]
    assert isinstance(buffer, PrioritizedReplayBuffer)
    assert buffer.capacity == cfg.agent.buffer_size == 1000 and buffer.obs_shape == (11,)
    assert buffer.alpha == cfg.agent.per_alpha == 0.7 and buffer.eps == cfg.agent.per_eps == 1e-5
    assert len(buffer) == 1000 and buffer.max_priority >= 1.0
    assert all(e.buffer is buffer for e in per_run.events if e.kind in ("sample", "priorities"))


# ------------------------------------------------------------ the loop
def test_every_sample_gets_the_beta_of_its_vec_step_and_the_sampling_stream(per_run):
    ac, tc = per_run.cfg.agent, per_run.cfg.train
    n = tc.n_envs
    samples = by_kind(per_run, "sample")
    G = max(1, n // 4)
    first_learning_k = -(-ac.learning_starts // n) - 1  # the first vec-step k with (k + 1) * n >= learning_starts
    assert first_learning_k == 124 and G == 2
    assert len(samples) == 352
    betas = []
    for j, s in enumerate(samples):
        k = first_learning_k + j // G
        want = linear_schedule(k * n, ac.per_beta_start, 1.0, tc.total_env_steps)
        assert s.beta == want, (j, k)
        assert s.batch_size == ac.batch_size == 16
        assert s.rng is samples[0].rng  # one sampling stream for the whole run ...
        assert s.size == min(k * n + n, ac.buffer_size)  # the buffer holds the vec-steps so far (the ring is 1000)
        betas.append(s.beta)
    assert samples[0].state_before == np.random.default_rng([PER_SEED, 2]).bit_generator.state  # ... default_rng([seed, 2])
    # The G samples of one vec-step share beta; beta rises from the first learning vec-step towards 1.0 and
    # stays below it: the last vec-step uses linear_schedule(total - n_envs, ...).
    assert betas == sorted(betas) and all(betas[2 * i] == betas[2 * i + 1] for i in range(len(betas) // 2))
    assert betas[0] == 0.5 + (1.0 - 0.5) * min(1.0, 992 / 2400)  # the schedule's own expression at 124 * 8 steps
    assert betas[-1] == linear_schedule(tc.total_env_steps - n, ac.per_beta_start, 1.0, tc.total_env_steps) < 1.0
    assert ac.per_beta_start < betas[0] < betas[-1]


def test_update_receives_the_batch_weights_and_update_priorities_the_indices_and_td_abs(per_run):
    events = per_run.events
    assert [e.kind for e in events] == ["sample", "update", "priorities"] * 352
    for j in range(352):
        sample, update, priorities = events[3 * j : 3 * j + 3]
        assert isinstance(sample.batch, PrioritizedBatch) and isinstance(sample.batch, Batch)
        assert update.batch is sample.batch
        assert update.weights is not None and update.weights.dtype == np.float32
        assert np.array_equal(update.weights, sample.weights) and np.array_equal(update.weights, sample.batch.weights)
        assert ((update.weights > 0) & (update.weights <= 1)).all()
        assert update.grad_steps == j + 1
        assert "td_abs" in update.out and update.out["td_abs"].shape == (16,) and update.out["td_abs"].dtype == np.float32
        assert np.array_equal(priorities.indices, sample.indices) and priorities.indices.dtype == np.int64
        assert np.array_equal(priorities.td_abs, update.out["td_abs"])
        assert ((sample.indices >= 0) & (sample.indices < sample.size)).all()
    # Prioritisation did something: not every batch is drawn with equal weights.
    assert any(len(set(e.weights.tolist())) > 1 for e in by_kind(per_run, "sample")[10:])


def test_per_metrics_rows_average_the_four_float_statistics_only(per_run):
    rows = read_metrics(per_run.dir)
    updates = by_kind(per_run, "update")
    windows = {2: updates[:152], 3: updates[152:352]}
    for i, window in windows.items():
        for name in UPDATE_COLUMNS:
            assert rows[i][name] == pytest.approx(float(np.mean([u.out[name] for u in window])), rel=1e-9), (i, name)
    assert all(type(u.out[name]) is float for u in updates for name in UPDATE_COLUMNS)


def test_dqn_window_add_update_sums_the_four_statistics_and_ignores_td_abs():
    window = DQNWindow()
    window.add_update({"loss": 1.0, "mean_q": 2.0, "mean_target": 3.0, "grad_norm": 4.0, "td_abs": np.array([1.0, 9.0])})
    window.add_update({"loss": 3.0, "mean_q": 4.0, "mean_target": 5.0, "grad_norm": 6.0, "td_abs": np.array([7.0])})
    assert window.updates == 2
    assert set(window.stat_sums) == set(UPDATE_COLUMNS)
    assert [window.mean(name) for name in UPDATE_COLUMNS] == [2.0, 3.0, 4.0, 5.0]
    assert DQNWindow().mean("loss") is None


@pytest.mark.parametrize("which", ["per_run"])
def test_per_value_columns_are_reproduced_from_the_checkpoints_and_the_values_block(which, request):
    run = request.getfixturevalue(which)
    cfg, tc = run.cfg, run.cfg.train
    rows = read_metrics(run.dir)
    for name, row in zip(checkpoint_names(run.dir), rows):
        agent = DQNAgent.load(run.dir / "checkpoints" / name)
        values = evaluate_batched(agent, cfg.env, tc.eval_episodes, tc.eval_seed, gamma=cfg.agent.gamma).value_summary()
        assert {key: row[column] for column, key in zip(VALUE_COLUMNS, VALUE_FIELDS)} == values
    stored = EvalResult.load(run.dir / "final_eval.json")
    assert stored.value_summary() == {key: rows[-1][column] for column, key in zip(VALUE_COLUMNS, VALUE_FIELDS)}


# ------------------------------------------------- an independent loop
def reference_per_run(cfg, seed: int, tmp_path):
    """The PER loop of D6 written out from its pseudo-code with its own ``PrioritizedReplayBuffer``,
    without ``snake_rl.train``, ``env_seed_for``, ``linear_schedule`` or ``act_epsilon``. Returns (model
    file, metrics rows, final evaluation)."""
    ac, tc = cfg.agent, cfg.train
    n = tc.n_envs
    torch.set_num_threads(ac.torch_threads)
    env = VecSnakeEnv(cfg.env, n_envs=n, seed=100_000 * (seed + 1))
    agent = DQNAgent(env.obs_shape, env.n_actions, ac, seed=seed)
    buffer = PrioritizedReplayBuffer(ac.buffer_size, env.obs_shape, ac.per_alpha, ac.per_eps)
    explore_rng = np.random.default_rng([seed, 1])
    sample_rng = np.random.default_rng([seed, 2])
    per_vec_step = ac.grad_steps_per_vec_step or max(1, n // 4)
    decay_steps = ac.eps_decay_fraction * tc.total_env_steps
    rows: list[dict] = []
    results: list[EvalResult] = []

    def evaluation(env_steps, epsilon, scores, n_truncated, stats):
        result = evaluate_batched(agent, cfg.env, tc.eval_episodes, tc.eval_seed, gamma=ac.gamma)
        results.append(result)
        values = result.value_summary()
        rows.append({
            "env_steps": env_steps,
            "vec_steps": env_steps // n,
            "grad_steps": agent.grad_steps,
            "epsilon": epsilon,
            "eval_mean_score": float(result.scores.mean()),
            "eval_max_score": int(result.scores.max()),
            "eval_mean_steps": float(result.steps.mean()),
            "eval_frac_starve": result.cause_fractions["starve"],
            **{f"eval_{name}": values[name] for name in VALUE_FIELDS},
            "train_episodes": len(scores),
            "train_mean_score": float(np.mean(scores)) if scores else None,
            "train_truncated": n_truncated,
            **{name: float(np.mean([s[name] for s in stats])) if stats else None for name in UPDATE_COLUMNS},
            "buffer_size": len(buffer),
        })

    obs = env.reset()
    evaluation(0, ac.eps_start, [], 0, [])
    scores, n_truncated, stats = [], 0, []
    for k in range(tc.total_env_steps // n):
        steps_before = k * n
        epsilon = ac.eps_start + (ac.eps_end - ac.eps_start) * min(1.0, steps_before / decay_steps)
        beta = ac.per_beta_start + (1.0 - ac.per_beta_start) * min(1.0, steps_before / tc.total_env_steps)
        explore = explore_rng.random(n) < epsilon
        random_actions = explore_rng.integers(0, env.n_actions, size=n)
        actions = np.where(explore, random_actions, agent.act_batch(obs))
        next_obs, reward, terminated, truncated, info = env.step(actions)
        buffer.add(obs, actions, reward, info["final_obs"], terminated)  # true successor, terminated only
        obs = next_obs
        scores.extend(info["score"][info["done"]].tolist())
        n_truncated += int(truncated.sum())
        if steps_before + n >= ac.learning_starts:
            for _ in range(per_vec_step):
                batch = buffer.sample(ac.batch_size, sample_rng, beta)
                out = agent.update(batch, batch.weights)
                buffer.update_priorities(batch.indices, out["td_abs"])
                stats.append(out)
        if (steps_before + n) % tc.eval_every == 0:
            evaluation(steps_before + n, epsilon, scores, n_truncated, stats)
            scores, n_truncated, stats = [], 0, []
    return agent.save(tmp_path / "reference_per_model.pt"), rows, results[-1]


def test_independent_per_loop_reproduces_train_exactly(per_run, tmp_path):
    model, rows, final = reference_per_run(per_run.cfg, per_run.seed, tmp_path)
    assert flat(per_run.dir / "model.pt") == flat(model)  # both networks, the optimiser state, grad_steps, the config
    assert final_episodes(per_run.dir) == final.episodes  # the trainer's protocol: value fields included
    written = read_metrics(per_run.dir)
    assert len(written) == len(rows) == 4
    for got, want in zip(written, rows):
        for name, value in want.items():
            if name in (*UPDATE_COLUMNS, "train_mean_score") and value is not None:
                assert got[name] == pytest.approx(value, rel=1e-9), name  # summation order is free
            else:
                assert got[name] == value, (name, want["env_steps"])
    assert rows[-1]["grad_steps"] == 352 and sum(r["train_episodes"] for r in rows) > 50


# ---------------------------------------------------------- determinism
def test_per_run_is_deterministic_and_differs_from_the_uniform_run(per_run, tmp_path):
    again = train(per_run.cfg, per_run.seed, tmp_path / "again", quiet=True)
    assert_same_run(per_run.dir, again)
    other_seed = train(per_run.cfg, per_run.seed + 1, tmp_path / "other", quiet=True)
    assert flat(other_seed / "model.pt")["/q_net/net.0.weight"] != flat(per_run.dir / "model.pt")["/q_net/net.0.weight"]
    # The same configuration with uniform replay is another run: same initial network, other updates.
    uniform = load_config(SMOKE, [*PER, "agent.per=false", "exp=uniform_dqn"])
    assert uniform.agent.per is False and uniform.agent.replay is True
    other = train(uniform, per_run.seed, tmp_path / "uniform", quiet=True)
    rows = read_metrics(other)
    assert [r["grad_steps"] for r in rows] == [0, 0, 152, 352] and [r["buffer_size"] for r in rows] == [0, 800, 1_000, 1_000]
    a, b = flat(per_run.dir / "model.pt"), flat(other / "model.pt")
    assert a["/config/per"] is True and b["/config/per"] is False
    assert a["/q_net/net.0.weight"] != b["/q_net/net.0.weight"]

    def network(path) -> dict:
        return {key: value for key, value in flat(path).items() if key.startswith("/q_net/")}

    assert network(per_run.dir / "checkpoints" / "step_00000000.pt") == network(other / "checkpoints" / "step_00000000.pt")
    assert network(per_run.dir / "checkpoints" / "step_00001600.pt") != network(other / "checkpoints" / "step_00001600.pt")
    assert metrics_without_times(per_run.dir) != metrics_without_times(other)


# ------------------------------------------------- the uniform loop is untouched
def test_uniform_loop_is_unchanged_without_per(tmp_path, monkeypatch):
    """Without ``agent.per`` the trainer builds a plain ``ReplayBuffer``, calls ``sample(batch_size, rng)``
    and ``update(batch)`` without weights; no prioritised buffer is constructed or sampled."""
    per_inits, per_samples, samples, updates = [], [], [], []
    real_per_init = PrioritizedReplayBuffer.__init__
    real_sample = ReplayBuffer.sample
    real_update = DQNAgent.update

    def spy_per_init(self, *args, **kwargs):
        per_inits.append(args)
        real_per_init(self, *args, **kwargs)

    def spy_per_sample(self, *args, **kwargs):
        per_samples.append(args)
        raise AssertionError("a uniform run must not sample a prioritised buffer")

    def spy_sample(self, batch_size, rng):  # the M6 signature: a third argument would be a TypeError
        samples.append((type(self), batch_size, rng))
        return real_sample(self, batch_size, rng)

    def spy_update(self, batch, weights=None):
        updates.append((type(batch), weights, self.grad_steps))
        return real_update(self, batch, weights)

    monkeypatch.setattr(PrioritizedReplayBuffer, "__init__", spy_per_init)
    monkeypatch.setattr(PrioritizedReplayBuffer, "sample", spy_per_sample)
    monkeypatch.setattr(ReplayBuffer, "sample", spy_sample)
    monkeypatch.setattr(DQNAgent, "update", spy_update)
    cfg = load_config(SMOKE, PLAIN)
    assert cfg.agent.per is False
    run_dir = train(cfg, 0, tmp_path, quiet=True)
    assert per_inits == [] and per_samples == []
    assert len(samples) == len(updates) == expected_grad_steps(cfg, cfg.train.total_env_steps // cfg.train.n_envs) == 152
    assert all(kind is ReplayBuffer and batch_size == 16 for kind, batch_size, _ in samples)
    assert all(rng is samples[0][2] for _, _, rng in samples)
    assert all(kind is Batch and kind is not PrioritizedBatch and weights is None for kind, weights, _ in updates)
    assert [steps for _, _, steps in updates] == list(range(152))
    assert [r["buffer_size"] for r in read_metrics(run_dir)] == [0, 800, 1_000]


def test_per_without_replay_is_refused_before_anything_is_written(tmp_path):
    with pytest.raises(ValueError, match="agent.per"):
        load_config(SMOKE, [*PER, "agent.replay=false", "agent.grad_steps_per_vec_step=1"])
    assert list(tmp_path.iterdir()) == []
