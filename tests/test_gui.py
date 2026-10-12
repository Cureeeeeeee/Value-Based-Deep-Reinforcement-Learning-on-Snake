"""pygame GUI (``snake_rl/gui/app.py``, PLAN.md milestone M9), headless.

Written from the M9 specification, independently of the implementation. The
SDL dummy video driver is selected before pygame is imported, so no window is
ever opened. Covers:

* the players: ``load_player`` for baseline names, run directories, bare
  checkpoints (``*.pt`` / ``*.npz``) with and without a ``config.yaml``;
  ``action_values`` for the DQN, the table, a baseline and a ``RecordedPlayer``;
* ``Recording`` (JSON round trip, re-simulation) and ``RecordedPlayer``
  (playback, exhaustion, rewind);
* ``GameSession``: same environment and boards as training, same episode
  records as ``snake_rl.evaluate``, counters, statistics, history, the
  next-seed rule, human control (heading -> action, reversal ignored);
* ``App``: every key, the ``tick`` clock (pause, one step per ``1 / speed``
  seconds, capped accumulation, auto reset in agent mode only), the end-of-
  episode messages, pixel colours of the board, screenshots, the panel text;
* the command line for the three modes and its exit code 2 cases.

All boards are 8x8 and all networks tiny, so the file runs in a few seconds.
"""

from __future__ import annotations

import os

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

import struct
from pathlib import Path

import numpy as np
import pygame
import pytest

from snake_rl.agents.baselines import make_policy
from snake_rl.agents.dqn import DQNConfig
from snake_rl.agents.dqn.agent import DQNAgent
from snake_rl.agents.tabular_q import TabularQAgent, state_index
from snake_rl.config import load_config
from snake_rl.env import EnvConfig, SnakeEnv
from snake_rl.env.common import DOWN, LEFT, RIGHT, STRAIGHT, TURN_LEFT, TURN_RIGHT, UP, heading_to_action
from snake_rl.evaluate import env_config_to_dict, evaluate
from snake_rl.gui import app as gui
from snake_rl.gui.app import (
    ACTION_LABELS,
    AUTO_RESET_DELAY,
    BASELINE_PLAYERS,
    COLOR_ARROW,
    COLOR_DEATH,
    COLOR_EMPTY,
    COLOR_FOOD,
    COLOR_HEAD,
    DEFAULT_CELL,
    DEFAULT_FPS,
    DEFAULT_SEED,
    FOOTER_H,
    HEADER_H,
    KEY_HEADINGS,
    MARGIN,
    MAX_FPS,
    MIN_FPS,
    MODES,
    PANEL_W,
    SPEED_FACTOR,
    App,
    GameSession,
    LoadedPlayer,
    RecordedPlayer,
    Recording,
    action_values,
    board_cell_rect,
    load_player,
    main,
    window_size,
)
from tests.helpers import FAR_FOOD, HOOK_COLLIDING_ACTION, HOOK_SELF_COLLISION, serpentine_almost_full

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
W = 8
CELL = 24  # small frames keep the pixel tests fast
MAX_EPISODE_STEPS = 5_000  # safety cap for "play to the end" loops


# ------------------------------------------------------------------ helpers
def cfg8(**kwargs) -> EnvConfig:
    return EnvConfig(W=W, **kwargs)


def key(k: int) -> pygame.event.Event:
    return pygame.event.Event(pygame.KEYDOWN, key=k, mod=0, unicode="")


def play_to_end(session: GameSession) -> None:
    for _ in range(MAX_EPISODE_STEPS):
        if session.done:
            return
        session.agent_step()
    raise AssertionError("episode did not end")


def tick_to_end(app: App) -> None:
    """Advance an unpaused app one step per tick until the episode ends."""
    for _ in range(MAX_EPISODE_STEPS):
        if app.session.done:
            return
        app.tick(1.0 / app.speed)
    raise AssertionError("episode did not end")


def seed_with_food_off_row(cfg: EnvConfig, row: int, first: int = DEFAULT_SEED) -> int:
    """A seed whose first food is not on ``row``: straight moves along that row never eat."""
    env = SnakeEnv(cfg)
    for seed in range(first, first + 500):
        env.reset(seed=seed)
        if env.food[0] != row:
            return seed
    raise AssertionError("no such seed")


def png_size(path: Path) -> tuple[int, int]:
    head = path.read_bytes()[:24]
    assert head[:8] == b"\x89PNG\r\n\x1a\n", path
    return struct.unpack(">II", head[16:24])


def exit_code(argv: list[str]) -> int:
    try:
        return main(argv)
    except SystemExit as err:  # a parser error also means exit code 2
        code = err.code
        return code if isinstance(code, int) else (0 if code is None else 1)


def centre_colour(surface: pygame.Surface, r: int, c: int, cell: int = CELL) -> tuple[int, int, int]:
    x, y = board_cell_rect(r, c, cell).center
    colour = surface.get_at((x, y))
    return (colour.r, colour.g, colour.b)


def write_run(run_dir: Path, template: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(template.read_text(encoding="utf-8"), encoding="utf-8")
    return run_dir


# ----------------------------------------------------------------- fixtures
@pytest.fixture
def dqn() -> DQNAgent:
    return DQNAgent((11,), 3, DQNConfig(hidden=(8,), device="cpu"), seed=0)


@pytest.fixture
def table() -> TabularQAgent:
    agent = TabularQAgent(3)
    agent.q[:] = np.random.default_rng(0).normal(size=agent.q.shape)
    return agent


@pytest.fixture
def bfs_session() -> GameSession:
    return GameSession(cfg8(), player=make_policy("bfs"), seed=DEFAULT_SEED)


@pytest.fixture
def human_session() -> GameSession:
    return GameSession(cfg8(), player=None, seed=DEFAULT_SEED, mode="human")


@pytest.fixture
def make_app():
    apps: list[App] = []

    def _make(session: GameSession, **kwargs) -> App:
        kwargs.setdefault("cell", CELL)
        app = App(session, headless=True, **kwargs)
        apps.append(app)
        return app

    yield _make
    for app in apps:
        app.close()


@pytest.fixture
def bfs_recording(tmp_path: Path) -> tuple[Recording, Path]:
    session = GameSession(cfg8(), player=make_policy("bfs"), seed=DEFAULT_SEED)
    play_to_end(session)
    path = session.recording.save(tmp_path / "recordings" / "bfs.json")
    return session.recording, path


# ---------------------------------------------------------------- constants
def test_constants():
    assert MODES == ("human", "agent", "replay")
    assert BASELINE_PLAYERS == ("random", "bfs", "hamiltonian")
    assert DEFAULT_SEED == 10_000
    assert MIN_FPS < DEFAULT_FPS < MAX_FPS
    assert SPEED_FACTOR > 1
    assert AUTO_RESET_DELAY > 0
    assert len(ACTION_LABELS["relative"]) == 3
    assert len(ACTION_LABELS["absolute"]) == 4
    assert KEY_HEADINGS[pygame.K_UP] == UP and KEY_HEADINGS[pygame.K_w] == UP
    assert KEY_HEADINGS[pygame.K_RIGHT] == RIGHT and KEY_HEADINGS[pygame.K_d] == RIGHT
    assert KEY_HEADINGS[pygame.K_DOWN] == DOWN and KEY_HEADINGS[pygame.K_s] == DOWN
    assert KEY_HEADINGS[pygame.K_LEFT] == LEFT and KEY_HEADINGS[pygame.K_a] == LEFT


def test_window_size_formula():
    for w, cell in ((10, 48), (8, 24), (16, 20)):
        assert window_size(w, cell) == (
            MARGIN + w * cell + MARGIN + PANEL_W + MARGIN,
            HEADER_H + MARGIN + w * cell + FOOTER_H,
        )


def test_board_cell_rect_geometry():
    rect = board_cell_rect(2, 3, CELL)
    assert (rect.x, rect.y) == (MARGIN + 3 * CELL, HEADER_H + MARGIN + 2 * CELL)
    assert (rect.width, rect.height) == (CELL, CELL)


# ------------------------------------------------------------------ players
def test_load_player_baselines():
    for name in BASELINE_PLAYERS:
        loaded = load_player(name)
        assert isinstance(loaded, LoadedPlayer)
        assert loaded.kind == "baseline"
        assert loaded.env_cfg is None
        assert loaded.player.name == name
        assert isinstance(loaded.label, str) and loaded.label


def test_load_player_tabular_run_dir(tmp_path: Path):
    run = write_run(tmp_path / "tabular_q_W8" / "0", CONFIGS / "tabular_q_8x8.yaml")
    agent = TabularQAgent(3)
    agent.q[:] = np.random.default_rng(1).normal(size=agent.q.shape)
    agent.save(run / "q_table.npz")

    loaded = load_player(run)
    assert loaded.kind == "tabular_q"
    assert isinstance(loaded.player, TabularQAgent)
    np.testing.assert_array_equal(loaded.player.q, agent.q)
    assert loaded.env_cfg == load_config(run / "config.yaml").env
    assert loaded.env_cfg.W == 8
    assert "tabular_q_W8" in loaded.label and "q_table" in loaded.label


def test_load_player_dqn_checkpoint_finds_parent_config(tmp_path: Path, dqn: DQNAgent):
    run = write_run(tmp_path / "dqn_A_W8" / "0", CONFIGS / "dqn_A_8x8.yaml")
    ckpt = dqn.save(run / "checkpoints" / "step_00000100.pt", include_optimizer=False)

    loaded = load_player(ckpt)
    assert loaded.kind == "dqn"
    assert isinstance(loaded.player, DQNAgent)
    assert loaded.env_cfg == load_config(run / "config.yaml").env
    assert loaded.env_cfg.W == 8
    assert "dqn_A_W8" in loaded.label and "step_00000100" in loaded.label
    obs = SnakeEnv(loaded.env_cfg).reset(seed=0)[0]
    np.testing.assert_allclose(loaded.player.q_values(obs[None]), dqn.q_values(obs[None]), rtol=1e-6)


def test_load_player_dqn_run_dir(tmp_path: Path, dqn: DQNAgent):
    run = write_run(tmp_path / "dqn_A_W8" / "1", CONFIGS / "dqn_A_8x8.yaml")
    dqn.save(run / "model.pt")
    loaded = load_player(run)
    assert loaded.kind == "dqn"
    assert loaded.env_cfg == load_config(run / "config.yaml").env
    assert "model.pt" in loaded.label


def test_load_player_bare_checkpoint_without_config(tmp_path: Path, dqn: DQNAgent):
    ckpt = dqn.save(tmp_path / "loose" / "weights.pt", include_optimizer=False)
    loaded = load_player(ckpt)
    assert loaded.kind == "dqn"
    assert loaded.env_cfg is None
    assert "weights.pt" in loaded.label


def test_load_player_errors(tmp_path: Path):
    with pytest.raises(ValueError, match="nonsense"):
        load_player("nonsense")
    with pytest.raises(FileNotFoundError):
        load_player(tmp_path / "missing.pt")
    (tmp_path / "empty").mkdir()
    with pytest.raises((ValueError, FileNotFoundError)):
        load_player(tmp_path / "empty")


def test_action_values(dqn: DQNAgent, table: TabularQAgent):
    env = SnakeEnv(cfg8())
    obs, _ = env.reset(seed=3)
    q_dqn = action_values(dqn, obs)
    assert q_dqn.shape == (3,)
    np.testing.assert_array_equal(q_dqn, dqn.q_values(obs[None])[0])
    assert int(np.argmax(q_dqn)) == dqn.act(obs, env)

    q_tab = action_values(table, obs)
    assert q_tab.shape == (3,)
    np.testing.assert_array_equal(q_tab, table.q[state_index(obs)])
    assert int(np.argmax(q_tab)) == table.act(obs, env)

    assert action_values(make_policy("bfs"), obs) is None


# ---------------------------------------------------------------- recording
def test_recording_round_trip_and_resimulation(bfs_recording, tmp_path: Path):
    rec, path = bfs_recording
    assert path.is_file() and path.parent.name == "recordings"
    assert rec.env == env_config_to_dict(cfg8())
    assert rec.seed == DEFAULT_SEED
    assert isinstance(rec.player, str) and rec.player
    assert rec.q_values is None  # BFS has no action values
    assert rec.cause in ("wall", "self", "starve", "win")
    assert len(rec.actions) == rec.steps

    loaded = Recording.load(path)
    assert loaded == rec
    assert loaded.env_config() == cfg8()

    env = SnakeEnv(loaded.env_config())
    env.reset(seed=loaded.seed)
    for action in loaded.actions:
        _, _, terminated, truncated, info = env.step(action)
    assert terminated or truncated
    assert (info["score"], info["steps"], info["cause"]) == (loaded.score, loaded.steps, loaded.cause)


def test_recording_with_q_values_round_trip(dqn: DQNAgent, tmp_path: Path):
    session = GameSession(cfg8(), player=dqn, seed=DEFAULT_SEED)
    for _ in range(5):
        if session.done:
            break
        session.agent_step()
    rec = session.recording
    assert rec.q_values is not None and len(rec.q_values) == len(rec.actions)
    assert all(len(q) == 3 for q in rec.q_values)
    path = rec.save(tmp_path / "dqn.json")
    assert Recording.load(path) == rec


def test_recorded_player_playback_exhaustion_and_rewind(bfs_recording):
    rec, _ = bfs_recording
    short = Recording(env=rec.env, seed=rec.seed, player=rec.player, actions=rec.actions[:2],
                      q_values=None, score=0, steps=2, cause="none")
    player = RecordedPlayer(short)
    assert player.name == "replay"
    env = SnakeEnv(short.env_config())
    obs, _ = env.reset(seed=short.seed)
    assert player.act(obs, env) == short.actions[0]
    assert player.act(obs, env) == short.actions[1]
    with pytest.raises(IndexError):
        player.act(obs, env)
    player.reset()
    assert player.act(obs, env) == short.actions[0]


def test_recorded_player_action_values(dqn: DQNAgent):
    session = GameSession(cfg8(), player=dqn, seed=DEFAULT_SEED)
    for _ in range(3):
        session.agent_step()
    rec = session.recording
    player = RecordedPlayer(rec)
    env = SnakeEnv(rec.env_config())
    obs, _ = env.reset(seed=rec.seed)
    for i in range(3):
        np.testing.assert_allclose(action_values(player, obs), np.asarray(rec.q_values[i]))
        player.act(obs, env)
    assert action_values(player, obs) is None  # exhausted

    without = Recording(env=rec.env, seed=rec.seed, player=rec.player, actions=rec.actions,
                        q_values=None, score=rec.score, steps=rec.steps, cause=rec.cause)
    assert action_values(RecordedPlayer(without), obs) is None


# -------------------------------------------------------------- GameSession
def test_session_uses_the_training_environment(bfs_session: GameSession):
    cfg = cfg8()
    assert isinstance(bfs_session.env, SnakeEnv)
    assert bfs_session.env.cfg == cfg
    reference = SnakeEnv(cfg)
    ref_obs, _ = reference.reset(seed=DEFAULT_SEED)
    np.testing.assert_array_equal(bfs_session.env.body_age, reference.body_age)
    assert bfs_session.env.food == reference.food
    assert bfs_session.env.heading == reference.heading
    np.testing.assert_array_equal(bfs_session.obs, ref_obs)

    bfs_session.reset(seed=42)
    reference.reset(seed=42)
    np.testing.assert_array_equal(bfs_session.env.body_age, reference.body_age)
    assert bfs_session.env.food == reference.food


def test_session_initial_state(bfs_session: GameSession):
    s = bfs_session
    assert s.mode == "agent"
    assert s.seed == DEFAULT_SEED and s.episode == 1
    assert s.done is False and s.cause == "none"
    assert s.episode_return == 0.0 and not s.last_reward and s.last_action is None
    assert s.q is None and s.chosen is None
    assert s.stats == {"wall": 0, "self": 0, "starve": 0, "win": 0}
    assert s.history == []
    assert s.recording.seed == DEFAULT_SEED and s.recording.actions == []
    assert s.recording.cause == "none"
    assert s.score == 0 and s.steps == 0 and s.length == 3 and s.steps_since_food == 0
    assert s.starve_limit == s.env.H == 100
    assert s.heading == RIGHT
    assert s.W == W and s.action_mode == "relative" and s.n_actions == 3


def test_session_agent_episode_matches_evaluate(bfs_session: GameSession):
    s = bfs_session
    play_to_end(s)
    record = evaluate(make_policy("bfs"), cfg8(), n_episodes=1, seed=DEFAULT_SEED).episodes[0]
    assert (s.score, s.steps, s.cause) == (record.score, record.steps, record.cause)
    assert s.history == [(record.score, record.steps, record.cause)]
    assert s.stats[record.cause] == 1 and sum(s.stats.values()) == 1
    rec = s.recording
    assert (rec.score, rec.steps, rec.cause) == (record.score, record.steps, record.cause)
    assert len(rec.actions) == record.steps
    assert s.last_action == rec.actions[-1]
    expected_return = float(record.score) - (1.0 if record.cause in ("wall", "self") else 0.0)
    assert s.episode_return == pytest.approx(expected_return)
    if record.cause in ("wall", "self"):
        assert s.last_reward == pytest.approx(-1.0)
    with pytest.raises(RuntimeError):
        s.step(STRAIGHT)


def test_session_replays_its_own_recording(bfs_session: GameSession):
    play_to_end(bfs_session)
    rec = bfs_session.recording
    replay = GameSession(rec.env_config(), player=RecordedPlayer(rec), seed=rec.seed, mode="replay")
    assert replay.mode == "replay"
    play_to_end(replay)
    assert (replay.score, replay.steps, replay.cause) == (rec.score, rec.steps, rec.cause)
    assert replay.cause != "end"
    assert replay.stats[rec.cause] == 1


def test_session_replay_runs_out_of_actions(bfs_session: GameSession):
    play_to_end(bfs_session)
    rec = bfs_session.recording
    short = Recording(env=rec.env, seed=rec.seed, player=rec.player, actions=rec.actions[:5],
                      q_values=None, score=0, steps=5, cause="none")
    replay = GameSession(short.env_config(), player=RecordedPlayer(short), seed=short.seed, mode="replay")
    for _ in range(5):
        assert replay.agent_step() == short.actions[replay.steps - 1]
    assert replay.done is False
    assert replay.agent_step() == -1
    assert replay.done is True and replay.cause == "end"
    assert replay.history == [] and sum(replay.stats.values()) == 0


def test_session_next_seed_rule(bfs_session: GameSession):
    s = bfs_session
    s.agent_step()
    s.reset()
    assert (s.seed, s.episode) == (DEFAULT_SEED + 1, 2)
    assert s.steps == 0 and s.done is False and s.recording.actions == [] and s.recording.seed == s.seed
    s.reset(seed=3)
    assert (s.seed, s.episode) == (3, 3)
    s.reset()
    assert (s.seed, s.episode) == (4, 4)


def test_session_agent_step_requires_player(human_session: GameSession):
    with pytest.raises(RuntimeError):
        human_session.agent_step()


def test_session_human_action_relative(human_session: GameSession):
    s = human_session
    assert s.heading == RIGHT
    assert s.human_action(UP) == heading_to_action(RIGHT, UP, "relative") == TURN_LEFT
    assert s.human_action(DOWN) == heading_to_action(RIGHT, DOWN, "relative") == TURN_RIGHT
    assert s.human_action(RIGHT) == STRAIGHT
    assert s.human_action(LEFT) == STRAIGHT  # the reversal keeps going
    assert s.straight_action() == STRAIGHT
    s.step(s.human_action(UP))
    assert s.heading == UP and s.steps == 1 and s.last_action == TURN_LEFT


def test_session_human_action_absolute():
    s = GameSession(cfg8(action_mode="absolute"), player=None, seed=DEFAULT_SEED, mode="human")
    assert s.n_actions == 4 and s.action_mode == "absolute"
    assert s.human_action(UP) == UP
    assert s.human_action(DOWN) == DOWN
    assert s.human_action(RIGHT) == RIGHT
    assert s.human_action(LEFT) == RIGHT  # the reversal keeps the current heading
    assert s.straight_action() == RIGHT
    s.step(s.human_action(DOWN))
    assert s.heading == DOWN and s.steps == 1


def test_session_learned_player_values_and_choice(dqn: DQNAgent):
    s = GameSession(cfg8(), player=dqn, seed=DEFAULT_SEED)
    assert s.q is not None and s.q.shape == (3,)
    np.testing.assert_array_equal(s.q, action_values(dqn, s.obs))
    assert s.chosen == int(np.argmax(s.q)) == dqn.act(s.obs, s.env)
    q_before, chosen_before = s.q.copy(), s.chosen
    action = s.agent_step()
    assert action == chosen_before == s.last_action
    np.testing.assert_allclose(np.asarray(s.recording.q_values[0]), q_before, rtol=1e-6)
    np.testing.assert_array_equal(s.q, action_values(dqn, s.obs))  # recomputed for the new state


def test_session_tabular_player(table: TabularQAgent):
    s = GameSession(cfg8(), player=table, seed=DEFAULT_SEED)
    np.testing.assert_array_equal(s.q, table.q[state_index(s.obs)])
    assert s.chosen == table.act(s.obs, s.env)
    play_to_end(s)
    assert s.cause in ("wall", "self", "starve", "win")


def test_session_human_with_advisor(dqn: DQNAgent):
    s = GameSession(cfg8(), player=None, seed=DEFAULT_SEED, advisor=dqn, mode="human")
    assert s.mode == "human"
    np.testing.assert_array_equal(s.q, action_values(dqn, s.obs))
    assert s.chosen == dqn.act(s.obs, s.env)
    s.step(STRAIGHT)  # the human decides, whatever the advisor prefers
    assert s.last_action == STRAIGHT and s.steps == 1
    assert len(s.recording.q_values) == 1
    np.testing.assert_array_equal(s.q, action_values(dqn, s.obs))


def test_session_wall_self_starve_win_causes():
    # wall: straight from the centre of an 8x8 board, heading right, dies on the 4th step
    s = GameSession(cfg8(), player=None, seed=DEFAULT_SEED, mode="human")
    for _ in range(3):
        s.step(STRAIGHT)
        assert not s.done
    s.step(STRAIGHT)
    assert s.done and s.cause == "wall" and s.steps == 4
    assert s.stats["wall"] == 1 and s.history[-1][2] == "wall"

    # self: the hook position of tests/helpers.py and its colliding action
    s.reset()
    s.env.set_state(HOOK_SELF_COLLISION, FAR_FOOD)
    s.step(HOOK_COLLIDING_ACTION)
    assert s.done and s.cause == "self" and s.stats["self"] == 1

    # starvation: limit 2, food off the head's row
    cfg = cfg8(starve_limit=2)
    seed = seed_with_food_off_row(cfg, row=W // 2)
    s = GameSession(cfg, player=None, seed=seed, mode="human")
    s.step(STRAIGHT)
    assert not s.done
    s.step(STRAIGHT)
    assert s.done and s.cause == "starve" and s.stats["starve"] == 1
    assert s.last_reward == pytest.approx(0.0)  # truncation is not a death

    # win: all but one cell covered, the food on it, straight ahead
    s = GameSession(cfg8(), player=None, seed=DEFAULT_SEED, mode="human")
    cells, food = serpentine_almost_full(W)
    s.env.set_state(cells, food)
    s.step(STRAIGHT)
    assert s.done and s.cause == "win" and s.stats["win"] == 1


# --------------------------------------------------------------------- App
def test_app_initial_state(make_app, bfs_session: GameSession, human_session: GameSession):
    agent_app = make_app(bfs_session)
    assert agent_app.session is bfs_session and agent_app.mode == "agent"
    assert agent_app.paused is False
    assert agent_app.speed == pytest.approx(DEFAULT_FPS)
    assert agent_app.quit_requested is False
    assert agent_app.pending_heading is None
    assert agent_app.message == ""
    assert isinstance(agent_app.screen, pygame.Surface)
    assert agent_app.screen.get_size() == window_size(W, CELL)

    human_app = make_app(human_session, fps=3.0)
    assert human_app.mode == "human"
    assert human_app.paused is True
    assert human_app.speed == pytest.approx(3.0)


def test_app_quit_keys(make_app, bfs_session: GameSession):
    for event in (key(pygame.K_ESCAPE), key(pygame.K_q), pygame.event.Event(pygame.QUIT)):
        app = make_app(bfs_session)
        assert app.quit_requested is False
        app.handle_event(event)
        assert app.quit_requested is True


def test_app_space_toggles_pause(make_app, bfs_session: GameSession):
    app = make_app(bfs_session)
    app.handle_event(key(pygame.K_SPACE))
    assert app.paused is True
    app.handle_event(key(pygame.K_SPACE))
    assert app.paused is False


def test_app_speed_keys(make_app, bfs_session: GameSession):
    app = make_app(bfs_session)
    app.handle_event(key(pygame.K_PLUS))
    assert app.speed == pytest.approx(DEFAULT_FPS * SPEED_FACTOR)
    app.handle_event(key(pygame.K_MINUS))
    assert app.speed == pytest.approx(DEFAULT_FPS)
    app.handle_event(key(pygame.K_EQUALS))
    assert app.speed == pytest.approx(DEFAULT_FPS * SPEED_FACTOR)
    app.handle_event(key(pygame.K_KP_MINUS))
    assert app.speed == pytest.approx(DEFAULT_FPS)
    app.handle_event(key(pygame.K_KP_PLUS))
    assert app.speed == pytest.approx(DEFAULT_FPS * SPEED_FACTOR)
    for _ in range(50):
        app.handle_event(key(pygame.K_PLUS))
    assert app.speed == pytest.approx(MAX_FPS)
    for _ in range(100):
        app.handle_event(key(pygame.K_MINUS))
    assert app.speed == pytest.approx(MIN_FPS)


def test_app_tick_clock_agent_mode(make_app, bfs_session: GameSession):
    app = make_app(bfs_session, fps=4.0)
    interval = 1.0 / app.speed
    app.tick(0.5 * interval)
    assert bfs_session.steps == 0
    app.tick(0.5 * interval)
    assert bfs_session.steps == 1  # time accumulates across ticks
    app.tick(interval)
    assert bfs_session.steps == 2
    app.tick(100.0)
    assert bfs_session.steps == 3  # a huge dt yields at most one step
    app.tick(0.0)
    assert bfs_session.steps == 3
    app.handle_event(key(pygame.K_SPACE))
    assert app.paused
    for _ in range(20):
        app.tick(interval)
    assert bfs_session.steps == 3  # nothing moves while paused
    app.handle_event(key(pygame.K_n))
    assert bfs_session.steps == 4  # single step while paused
    app.handle_event(key(pygame.K_PERIOD))
    assert bfs_session.steps == 5
    app.handle_event(key(pygame.K_SPACE))
    app.handle_event(key(pygame.K_n))
    assert bfs_session.steps == 6  # and while running


def test_app_heading_keys_ignored_in_agent_mode(make_app, bfs_session: GameSession):
    app = make_app(bfs_session)
    for k in KEY_HEADINGS:
        app.handle_event(key(k))
    assert app.pending_heading is None
    assert bfs_session.steps == 0


def test_app_human_paused_steps_once_per_key(make_app, human_session: GameSession):
    app = make_app(human_session)
    assert app.paused
    app.handle_event(key(pygame.K_UP))
    assert human_session.steps == 1 and human_session.heading == UP
    assert human_session.last_action == TURN_LEFT
    assert app.paused is True
    app.handle_event(key(pygame.K_DOWN))  # reversal of UP: keeps going up
    assert human_session.steps == 2 and human_session.heading == UP
    assert human_session.last_action == STRAIGHT
    app.handle_event(key(pygame.K_LEFT))
    assert human_session.steps == 3 and human_session.heading == LEFT
    for _ in range(50):
        app.tick(1.0 / app.speed)
    assert human_session.steps == 3  # paused: the clock does not move the snake
    app.handle_event(key(pygame.K_n))  # one step straight
    assert human_session.steps == 4 and human_session.heading == LEFT


def test_app_human_running_applies_pending_heading_once(make_app, human_session: GameSession):
    app = make_app(human_session, fps=5.0)
    interval = 1.0 / app.speed
    app.handle_event(key(pygame.K_SPACE))
    assert app.paused is False
    app.handle_event(key(pygame.K_DOWN))
    assert app.pending_heading == DOWN
    assert human_session.steps == 0  # applied at the next tick, not at once
    app.tick(interval)
    assert human_session.steps == 1 and human_session.heading == DOWN
    assert app.pending_heading is None
    app.tick(interval)
    assert human_session.steps == 2 and human_session.heading == DOWN  # straight on
    assert human_session.last_action == STRAIGHT


def test_app_wall_death_message_and_no_auto_reset_in_human_mode(make_app, human_session: GameSession):
    app = make_app(human_session, fps=10.0)
    app.handle_event(key(pygame.K_SPACE))
    tick_to_end(app)
    assert human_session.cause == "wall" and human_session.steps == 4
    assert "wall" in app.message.lower()
    episode = human_session.episode
    for _ in range(int(AUTO_RESET_DELAY * 10) + 20):
        app.tick(0.1)
    assert human_session.done and human_session.episode == episode
    steps = human_session.steps
    app.handle_event(key(pygame.K_UP))
    app.handle_event(key(pygame.K_n))
    assert human_session.done and human_session.steps == steps  # keys do nothing after the end
    app.handle_event(key(pygame.K_r))
    assert human_session.done is False and human_session.episode == episode + 1
    assert human_session.seed == DEFAULT_SEED + 1
    assert app.message == ""


def test_app_self_death_message(make_app, human_session: GameSession):
    app = make_app(human_session, fps=10.0)
    human_session.env.set_state(HOOK_SELF_COLLISION, FAR_FOOD)  # head (5, 4) heading left
    app.handle_event(key(pygame.K_SPACE))
    app.handle_event(key(pygame.K_UP))  # turn right -> (4, 4), age 2 -> self collision
    app.tick(1.0 / app.speed)
    app.tick(0.0)
    assert human_session.done and human_session.cause == "self"
    assert "self" in app.message.lower()


def test_app_starvation_message():
    cfg = cfg8(starve_limit=2)
    seed = seed_with_food_off_row(cfg, row=W // 2)
    session = GameSession(cfg, player=None, seed=seed, mode="human")
    app = App(session, headless=True, cell=CELL, fps=10.0)
    try:
        app.handle_event(key(pygame.K_SPACE))
        tick_to_end(app)
        assert session.cause == "starve"
        assert "starv" in app.message.lower()
    finally:
        app.close()


def test_app_win_message(make_app, human_session: GameSession):
    app = make_app(human_session)
    cells, food = serpentine_almost_full(W)
    human_session.env.set_state(cells, food)  # head heading left, food straight ahead
    app.handle_event(key(pygame.K_LEFT))  # paused: one step with the current heading
    app.tick(0.0)
    assert human_session.done and human_session.cause == "win"
    assert "win" in app.message.lower()


def test_app_auto_reset_in_agent_mode(make_app, bfs_session: GameSession):
    app = make_app(bfs_session, fps=20.0)
    tick_to_end(app)
    assert bfs_session.done and app.message
    app.tick(AUTO_RESET_DELAY / 2)
    assert bfs_session.done  # not yet
    for _ in range(int(AUTO_RESET_DELAY * 10) + 5):
        app.tick(0.1)
    assert bfs_session.done is False
    assert bfs_session.episode == 2 and bfs_session.seed == DEFAULT_SEED + 1
    assert app.message == ""
    assert sum(bfs_session.stats.values()) == 1


def test_app_no_auto_reset_while_paused(make_app, bfs_session: GameSession):
    app = make_app(bfs_session, fps=20.0)
    tick_to_end(app)
    app.handle_event(key(pygame.K_SPACE))
    for _ in range(int(AUTO_RESET_DELAY * 10) + 20):
        app.tick(0.1)
    assert bfs_session.done and bfs_session.episode == 1


def test_app_replay_mode_keys_and_end(make_app, bfs_recording):
    rec, _ = bfs_recording
    session = GameSession(rec.env_config(), player=RecordedPlayer(rec), seed=rec.seed, mode="replay")
    app = make_app(session, fps=20.0)
    assert app.mode == "replay" and app.paused is False
    tick_to_end(app)
    assert (session.score, session.steps, session.cause) == (rec.score, rec.steps, rec.cause)
    for _ in range(int(AUTO_RESET_DELAY * 10) + 20):
        app.tick(0.1)
    assert session.done and session.episode == 1  # no auto reset in replay mode
    app.handle_event(key(pygame.K_r))  # the same recorded episode again
    assert session.done is False and session.seed == rec.seed and session.episode == 2 and session.steps == 0
    assert app.message == ""
    tick_to_end(app)
    assert (session.score, session.steps, session.cause) == (rec.score, rec.steps, rec.cause)


def test_app_replay_end_of_recording_message(make_app, bfs_recording):
    rec, _ = bfs_recording
    short = Recording(env=rec.env, seed=rec.seed, player=rec.player, actions=rec.actions[:3],
                      q_values=None, score=0, steps=3, cause="none")
    session = GameSession(short.env_config(), player=RecordedPlayer(short), seed=short.seed, mode="replay")
    app = make_app(session, fps=20.0)
    tick_to_end(app)
    assert session.cause == "end" and session.steps == 3
    assert "recording" in app.message.lower()


def test_app_save_recording(make_app, bfs_session: GameSession, tmp_path: Path):
    record_dir = tmp_path / "rec"
    app = make_app(bfs_session, record_dir=record_dir)
    for _ in range(4):
        app.handle_event(key(pygame.K_n))
    app.handle_event(key(pygame.K_s))
    files = sorted(record_dir.glob("*.json"))
    assert len(files) == 1
    assert files[0].name.startswith(f"bfs_seed{DEFAULT_SEED}_")
    assert app.message.startswith("Saved")
    saved = Recording.load(files[0])
    assert saved.actions == bfs_session.recording.actions
    assert len(saved.actions) == 4 and saved.seed == DEFAULT_SEED

    tick_to_end(app)
    later = make_app(bfs_session, record_dir=tmp_path / "rec2")  # a second directory: the names carry a 1 s timestamp
    later.handle_event(key(pygame.K_s))  # the finished episode
    files = sorted((tmp_path / "rec2").glob("*.json"))
    assert len(files) == 1
    finished = Recording.load(files[0])
    assert (finished.score, finished.steps, finished.cause) == (bfs_session.score, bfs_session.steps, bfs_session.cause)
    assert finished.actions == bfs_session.recording.actions


# ---------------------------------------------------------------- rendering
def test_render_board_colours(make_app, bfs_session: GameSession):
    app = make_app(bfs_session)
    frame = app.render()
    assert isinstance(frame, pygame.Surface)
    assert frame.get_size() == window_size(W, CELL)
    env = bfs_session.env
    hr, hc = env.head
    assert centre_colour(frame, hr, hc) in (tuple(COLOR_HEAD), tuple(COLOR_ARROW))
    assert centre_colour(frame, *env.food) == tuple(COLOR_FOOD)
    for r, c in env.snake_cells()[:-1]:
        assert centre_colour(frame, r, c) != tuple(COLOR_EMPTY)
    empties = [(r, c) for r in range(W) for c in range(W) if env.body_age[r, c] == 0 and (r, c) != env.food]
    assert centre_colour(frame, *empties[0]) == tuple(COLOR_EMPTY)
    assert centre_colour(frame, *empties[-1]) == tuple(COLOR_EMPTY)


def test_render_after_deaths(make_app, human_session: GameSession):
    app = make_app(human_session)
    human_session.env.set_state(HOOK_SELF_COLLISION, FAR_FOOD)
    human_session.step(HOOK_COLLIDING_ACTION)
    assert human_session.cause == "self"
    frame = app.render()
    assert centre_colour(frame, 4, 4) == tuple(COLOR_DEATH)  # the cell the head ran into

    human_session.reset()
    for _ in range(4):
        human_session.step(STRAIGHT)
    assert human_session.cause == "wall"
    frame = app.render()  # nothing to draw outside the board; must not raise
    assert frame.get_size() == window_size(W, CELL)


def test_render_with_action_values(make_app, dqn: DQNAgent):
    session = GameSession(cfg8(), player=dqn, seed=DEFAULT_SEED)
    app = make_app(session)
    frame = app.render()
    assert frame.get_size() == window_size(W, CELL)
    lines = "\n".join(app.panel_lines())
    for label in ACTION_LABELS["relative"]:
        assert label in lines


def test_screenshot_writes_png(make_app, bfs_session: GameSession, tmp_path: Path):
    app = make_app(bfs_session)
    path = app.screenshot(tmp_path / "shots" / "frame.png")
    assert path == tmp_path / "shots" / "frame.png" and path.is_file()
    assert png_size(path) == window_size(W, CELL)


def test_panel_lines(make_app, bfs_session: GameSession):
    app = make_app(bfs_session)
    lines = app.panel_lines()
    assert isinstance(lines, list) and all(isinstance(line, str) for line in lines)
    joined = "\n".join(lines)
    assert "Score" in joined
    assert "bfs" in joined.lower()
    assert f"{bfs_session.steps_since_food} / {bfs_session.starve_limit}" in joined
    tick_to_end(app)
    joined = "\n".join(app.panel_lines())
    assert f"{bfs_session.cause} 1" in joined.replace("starved", "starve")
    assert str(bfs_session.score) in joined


# ---------------------------------------------------------------------- CLI
def test_cli_agent_mode(tmp_path: Path):
    shot = tmp_path / "agent.png"
    assert exit_code(["--mode", "agent", "--player", "bfs", "--W", str(W), "--cell", str(CELL), "--headless",
                      "--steps", "20", "--screenshot", str(shot)]) == 0
    assert png_size(shot) == window_size(W, CELL)


def test_cli_player_implies_agent_mode(tmp_path: Path):
    shot = tmp_path / "hamiltonian.png"
    assert exit_code(["--player", "hamiltonian", "--W", str(W), "--cell", str(CELL), "--headless",
                      "--steps", "5", "--screenshot", str(shot)]) == 0
    assert png_size(shot) == window_size(W, CELL)


def test_cli_human_mode(tmp_path: Path):
    shot = tmp_path / "human.png"
    assert exit_code(["--mode", "human", "--W", str(W), "--cell", str(CELL), "--headless", "--steps", "0",
                      "--screenshot", str(shot)]) == 0
    assert png_size(shot) == window_size(W, CELL)


def test_cli_replay_mode(tmp_path: Path, bfs_recording):
    _, rec_path = bfs_recording
    shot = tmp_path / "replay.png"
    assert exit_code(["--replay", str(rec_path), "--cell", str(CELL), "--headless", "--steps", "10",
                      "--screenshot", str(shot)]) == 0
    assert png_size(shot) == window_size(W, CELL)
    shot2 = tmp_path / "replay2.png"
    assert exit_code(["--mode", "replay", "--replay", str(rec_path), "--cell", str(CELL), "--headless",
                      "--steps", "3", "--screenshot", str(shot2)]) == 0
    assert shot2.is_file()


def test_cli_learned_player_from_run_dir(tmp_path: Path, dqn: DQNAgent):
    run = write_run(tmp_path / "dqn_A_W8" / "0", CONFIGS / "dqn_A_8x8.yaml")
    dqn.save(run / "model.pt")
    shot = tmp_path / "dqn.png"
    assert exit_code(["--player", str(run), "--cell", str(CELL), "--headless", "--steps", "5",
                      "--screenshot", str(shot)]) == 0
    assert png_size(shot) == window_size(8, CELL)  # the board size comes from the run's config


def test_cli_exit_code_2(tmp_path: Path, bfs_recording, capsys):
    _, rec_path = bfs_recording
    assert exit_code(["--mode", "agent", "--headless", "--steps", "1"]) == 2
    assert exit_code(["--mode", "replay", "--headless", "--steps", "1"]) == 2
    assert exit_code(["--player", "nonsense", "--headless", "--steps", "1"]) == 2
    assert exit_code(["--replay", str(rec_path), "--W", "10", "--headless", "--steps", "1"]) == 2
    assert capsys.readouterr().err.strip()


def test_build_parser_defaults():
    args = gui.build_parser().parse_args([])
    assert args.seed == DEFAULT_SEED
    assert args.cell == DEFAULT_CELL
    assert args.player is None and args.replay is None and args.mode in (None, "human")
