"""pygame GUI (PLAN.md milestone M9; Part B "Interface"; Part D of the submission).

Three modes, all on the scalar reference environment ``SnakeEnv`` with the
same rules, observations and rewards that training uses:

* ``human``  - keyboard control (arrow keys / WASD give the desired heading;
  the 180-degree reversal is ignored). Optionally an *advisor* - a trained
  agent whose action values for the human's current state are shown in the
  side panel without acting.
* ``agent``  - a loaded player (``random`` / ``bfs`` / ``hamiltonian``, a
  tabular Q table or a DQN checkpoint) plays live. The side panel shows the
  action values of the current state as a bar chart with the chosen action,
  the 11 features of observation A, score, steps since the last food against
  the starvation limit, the episode return, and the running death-cause
  statistics of the session. Episodes follow the evaluation seeds
  (``DEFAULT_SEED`` = 10000, then 10001, ...), so the examiner can play a
  board and then watch the agent on the very same board.
* ``replay`` - a recorded episode (JSON: environment config, seed, actions and
  the panel values of every step) is re-simulated; the environment is
  deterministic given the seed, so the actions alone reproduce the game.

Resource-first note (``docs/RESOURCE_BASELINE.md``, M9 row): the turn loop is
organised like the ``run_game`` cells of ``resources/week08`` Part C (cell 22,
``run_game_sb3``) and ``resources/week09`` Part B (cell 28, ``run_game``):
reset -> loop { human input | argmax of the model's action values ->
``env.step`` -> render } -> game-over message. Those cells are console loops
over ``input()`` for Tic-Tac-Toe and nothing from them runs here verbatim;
the pygame rendering, the action-value panel, the recordings and the command
line are new (L3, no resource equivalent).

Architecture: :class:`GameSession` is the headless game model (environment,
player, counters, recording; no pygame; what the tests exercise), :class:`App`
is the pygame view and event loop around it. Headless operation
(``SDL_VIDEODRIVER=dummy``) renders to an off-screen surface, which is how the
tests run and how the Part D screenshots are produced
(``scripts/gui_screenshots.py``).

Command line::

    python -m snake_rl.gui.app                                   # human play, 10x10
    python -m snake_rl.gui.app --player runs/dqn_A_W10/0         # the DQN of that run plays
    python -m snake_rl.gui.app --player runs/dqn_A_W10/0/checkpoints/step_00012800.pt
    python -m snake_rl.gui.app --player bfs --seed 10000
    python -m snake_rl.gui.app --mode human --player runs/tabular_q_W10/0   # human plays, table advises
    python -m snake_rl.gui.app --replay runs/recordings/<file>.json

Keys: arrows / WASD steer (human); Space pause; N or . single step; R new
game (replay: restart); S save the episode as a recording; + / - speed;
Q / Esc quit.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Sequence

import numpy as np

os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
import pygame  # noqa: E402  (after the environment variable on purpose)

from snake_rl.agents.baselines import make_policy  # noqa: E402
from snake_rl.agents.tabular_q import TabularQAgent, state_index  # noqa: E402
from snake_rl.env.common import (  # noqa: E402
    DELTAS,
    DOWN,
    LEFT,
    RIGHT,
    STRAIGHT,
    UP,
    heading_to_action,
)
from snake_rl.env.core import EnvConfig, SnakeEnv  # noqa: E402
from snake_rl.evaluate import env_config_from_dict, env_config_to_dict  # noqa: E402

# ------------------------------------------------------------------ constants
MODES = ("human", "agent", "replay")
BASELINE_PLAYERS = ("random", "bfs", "hamiltonian")
#: First episode = evaluation board 0 (``evaluate.DEFAULT_EVAL_SEED``); then seed + 1, ...
DEFAULT_SEED = 10_000
DEFAULT_FPS = 6.0
MIN_FPS, MAX_FPS = 0.5, 60.0
SPEED_FACTOR = 1.5
AUTO_RESET_DELAY = 1.2
DEFAULT_CELL = 48
HEADER_H, MARGIN, PANEL_W, FOOTER_H = 48, 24, 408, 40
ACTION_LABELS = {
    "relative": ("Straight", "Turn left", "Turn right"),
    "absolute": ("Up", "Right", "Down", "Left"),
}
KEY_HEADINGS = {
    pygame.K_UP: UP, pygame.K_w: UP,
    pygame.K_RIGHT: RIGHT, pygame.K_d: RIGHT,
    pygame.K_DOWN: DOWN, pygame.K_s: DOWN,
    pygame.K_LEFT: LEFT, pygame.K_a: LEFT,
}
MODE_TITLES = {"human": "Human play", "agent": "Agent play", "replay": "Replay"}
KEY_HELP = "Space pause . N step . R new game . S save . +/- speed . Q quit"

# colours (RGB)
COLOR_BG = (242, 244, 247)
COLOR_HEADER = (38, 48, 66)
COLOR_HEADER_TEXT = (240, 242, 245)
COLOR_CHIP = (84, 96, 118)
COLOR_CHIP_ACTIVE = (58, 160, 92)
COLOR_BOARD_EDGE = (28, 32, 40)
COLOR_GRID = (66, 72, 86)
COLOR_EMPTY = (52, 58, 70)
COLOR_TAIL = (150, 215, 160)
COLOR_BODY = (84, 190, 110)
COLOR_HEAD = (30, 130, 70)
COLOR_ARROW = (255, 255, 255)
COLOR_FOOD = (231, 76, 60)
COLOR_DEATH = (220, 40, 40)
COLOR_PANEL = (255, 255, 255)
COLOR_PANEL_EDGE = (210, 214, 220)
COLOR_TEXT = (30, 34, 40)
COLOR_MUTED = (110, 116, 128)
COLOR_BAR_TRACK = (232, 235, 240)
COLOR_BAR_CHOSEN = (52, 140, 230)
COLOR_BAR_OTHER = (150, 158, 170)
COLOR_BAR_NEGATIVE = (231, 110, 90)
COLOR_FEATURE_ON = (52, 140, 230)
COLOR_FEATURE_OFF = (226, 229, 234)
COLOR_OVERLAY = (20, 22, 28)

CAUSE_LABELS = {"wall": "wall", "self": "self", "starve": "starved", "win": "win"}


def window_size(W: int, cell: int) -> tuple[int, int]:
    """Window ``(width, height)`` for a ``W x W`` board at ``cell`` pixels per cell."""
    return (MARGIN + W * cell + MARGIN + PANEL_W + MARGIN, HEADER_H + MARGIN + W * cell + FOOTER_H)


def board_cell_rect(r: int, c: int, cell: int) -> pygame.Rect:
    """Pixel rectangle of board cell ``(r, c)``."""
    return pygame.Rect(MARGIN + c * cell, HEADER_H + MARGIN + r * cell, cell, cell)


# ------------------------------------------------------------------- players
@dataclass
class LoadedPlayer:
    """A player built from a command-line spec, with where it came from."""

    player: Any
    kind: str  # "baseline" | "tabular_q" | "dqn"
    label: str
    env_cfg: EnvConfig | None


def _find_run_config(path: Path) -> tuple[Path | None, Any]:
    """``(run_dir, ExperimentConfig)`` of the run that owns ``path``: the first
    of the file's directory and its parent that holds ``config.yaml``
    (checkpoints live in ``<run>/checkpoints/``); ``(None, None)`` if neither."""
    from snake_rl.config import load_config

    for directory in (path.parent, path.parent.parent):
        config = directory / "config.yaml"
        if config.is_file():
            return directory, load_config(config)
    return None, None


def _checkpoint_steps(path: Path, exp_cfg: Any) -> int | None:
    """Environment steps of a checkpoint: from ``step_XXXXXXXX.pt`` or, for the
    final ``model.pt`` / ``q_table.npz``, the run's total budget."""
    match = re.fullmatch(r"step_(\d+)", path.stem)
    if match:
        return int(match.group(1))
    if exp_cfg is not None and path.name in ("model.pt", "q_table.npz"):
        return int(exp_cfg.train.total_env_steps)
    return None


def load_player(spec: str | Path, device: str = "cpu") -> LoadedPlayer:
    """Build the player named by ``spec``.

    ``spec`` is a baseline name (``random`` / ``bfs`` / ``hamiltonian``), a run
    directory (its ``model.pt`` or ``q_table.npz``), a DQN checkpoint ``*.pt``
    or a Q table ``*.npz``. For files the run's ``config.yaml`` (same directory
    or the parent) supplies ``env_cfg`` and the label; without one ``env_cfg``
    is None. Unknown names raise ``ValueError``; a missing checkpoint
    ``FileNotFoundError``. ``device`` is passed to ``DQNAgent.load``.
    """
    text = str(spec)
    if text in BASELINE_PLAYERS:
        return LoadedPlayer(make_policy(text), "baseline", text, None)
    path = Path(spec)
    if path.is_dir():
        for name in ("model.pt", "q_table.npz"):
            if (path / name).is_file():
                path = path / name
                break
        else:
            raise ValueError(f"{path}: not a run directory (no model.pt or q_table.npz)")
    if path.suffix not in (".pt", ".npz"):
        raise ValueError(
            f"unknown player {text!r}: expected one of {BASELINE_PLAYERS}, a run directory, a *.pt checkpoint or a *.npz table"
        )
    if not path.is_file():
        raise FileNotFoundError(f"no such checkpoint: {path}")
    run_dir, exp_cfg = _find_run_config(path)
    if path.suffix == ".pt":
        from snake_rl.agents.dqn.agent import DQNAgent  # torch, only when needed

        player: Any = DQNAgent.load(path, device=device)
        kind, title = "dqn", "DQN"
    else:
        player = TabularQAgent.load(path)
        kind, title = "tabular_q", "Tabular Q"
    steps = _checkpoint_steps(path, exp_cfg)
    where = f"{run_dir.parent.name}/{run_dir.name}  " if run_dir is not None else ""
    label = f"{title}  {where}{path.name}" + (f" ({steps:,} env steps)" if steps is not None else "")
    return LoadedPlayer(player, kind, label, exp_cfg.env if exp_cfg is not None else None)


def action_values(player: Any, obs: np.ndarray) -> np.ndarray | None:
    """Action values of one (unbatched) observation for the panel, or ``None``
    for players that have none (baselines, human)."""
    if player is None:
        return None
    if isinstance(player, RecordedPlayer):
        return player.action_values()
    if hasattr(player, "q_values"):  # DQNAgent
        return np.asarray(player.q_values(np.asarray(obs)[None])[0])
    table = getattr(player, "q", None)
    if isinstance(table, np.ndarray):  # TabularQAgent
        return np.asarray(table[state_index(obs)])
    return None


# ----------------------------------------------------------------- recording
@dataclass
class Recording:
    """One episode as JSON: enough to re-simulate it (the environment is a
    deterministic function of the seed and the actions) and to show the panel
    values that were on screen before every step."""

    env: dict[str, Any]
    seed: int
    player: str
    actions: list[int] = field(default_factory=list)
    q_values: list[list[float]] | None = None
    score: int = 0
    steps: int = 0
    cause: str = "none"

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "Recording":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            env=data["env"],
            seed=int(data["seed"]),
            player=str(data["player"]),
            actions=[int(a) for a in data["actions"]],
            q_values=None if data.get("q_values") is None else [[float(v) for v in row] for row in data["q_values"]],
            score=int(data.get("score", 0)),
            steps=int(data.get("steps", 0)),
            cause=str(data.get("cause", "none")),
        )

    def env_config(self) -> EnvConfig:
        return env_config_from_dict(self.env)


class RecordedPlayer:
    """Plays a :class:`Recording` back: ``act`` returns the next recorded
    action (``IndexError`` past the end); ``action_values`` the recorded panel
    values of the step about to be played."""

    name = "replay"

    def __init__(self, recording: Recording) -> None:
        self.recording = recording
        self.index = 0

    def reset(self) -> None:
        self.index = 0

    @property
    def exhausted(self) -> bool:
        return self.index >= len(self.recording.actions)

    def act(self, obs: np.ndarray, env: SnakeEnv | None = None) -> int:
        if self.exhausted:
            raise IndexError("end of recording")
        action = self.recording.actions[self.index]
        self.index += 1
        return action

    def action_values(self) -> np.ndarray | None:
        values = self.recording.q_values
        if values is None or self.index >= len(values):
            return None
        return np.asarray(values[self.index], dtype=np.float64)


# ------------------------------------------------------------------- session
class GameSession:
    """Headless game model: one ``SnakeEnv``, an optional player, the counters
    the panel shows and the recording of the current episode. No pygame.

    ``player=None`` is human control; ``advisor`` is a learned player whose
    action values are shown in human mode without acting. ``mode`` is derived
    from the player unless given: human (no player), replay (a
    :class:`RecordedPlayer`) or agent.
    """

    def __init__(
        self,
        env_cfg: EnvConfig,
        player: Any | None = None,
        seed: int = DEFAULT_SEED,
        advisor: Any | None = None,
        mode: str | None = None,
    ) -> None:
        if player is None:
            mode = "human"
        elif mode is None or mode == "human":
            mode = "replay" if isinstance(player, RecordedPlayer) else "agent"
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.env = SnakeEnv(env_cfg)
        self.player = player
        self.advisor = advisor
        self.mode = mode
        self.player_label = player.name if player is not None else "human"
        if advisor is not None:
            self.player_label = f"human (advisor: {getattr(advisor, 'name', 'agent')})"
        self.seed = int(seed)
        self.episode = 0
        self.stats: dict[str, int] = {"wall": 0, "self": 0, "starve": 0, "win": 0}
        self.history: list[tuple[int, int, str]] = []
        self.obs: np.ndarray
        self.q: np.ndarray | None = None
        self.prev_q: np.ndarray | None = None
        self.chosen: int | None = None
        self.recording: Recording
        self.reset(self.seed)

    # --------------------------------------------------------- properties
    @property
    def W(self) -> int:
        return self.env.W

    @property
    def action_mode(self) -> str:
        return self.env.cfg.action_mode

    @property
    def n_actions(self) -> int:
        return int(self.env.action_space.n)

    @property
    def score(self) -> int:
        return self.env.score

    @property
    def steps(self) -> int:
        return self.env.steps

    @property
    def length(self) -> int:
        return self.env.length

    @property
    def steps_since_food(self) -> int:
        return self.env.steps_since_food

    @property
    def starve_limit(self) -> int:
        return self.env.H

    @property
    def heading(self) -> int:
        return self.env.heading

    # ------------------------------------------------------------ control
    def reset(self, seed: int | None = None) -> None:
        """Start a new episode: ``seed`` or, for ``None``, the next seed."""
        self.seed = int(seed) if seed is not None else self.seed + 1
        self.episode += 1
        self.obs, _ = self.env.reset(seed=self.seed)
        self.done = False
        self.cause = "none"
        self.episode_return = 0.0
        self.last_reward = 0.0
        self.last_action: int | None = None
        self.prev_q = None
        for policy in (self.player, self.advisor):
            if policy is not None and hasattr(policy, "reset"):
                policy.reset()
        self.recording = Recording(env=env_config_to_dict(self.env.cfg), seed=self.seed, player=self.player_label)
        self._update_values()

    def step(self, action: int) -> None:
        """One environment step with ``action``; raises if the episode is over."""
        if self.done:
            raise RuntimeError("the episode is over; call reset() first")
        action = int(action)
        q_before = self.q
        self.obs, reward, terminated, truncated, info = self.env.step(action)
        self.last_reward = float(reward)
        self.last_action = action
        self.episode_return += float(reward)
        self.prev_q = q_before
        rec = self.recording
        rec.actions.append(action)
        if q_before is not None:
            if rec.q_values is None:
                rec.q_values = []
            rec.q_values.append([float(v) for v in q_before])
        rec.score, rec.steps = int(info["score"]), int(info["steps"])
        if terminated or truncated:
            self.done = True
            self.cause = str(info["cause"])
            rec.cause = self.cause
            self.history.append((int(info["score"]), int(info["steps"]), self.cause))
            self.stats[self.cause] += 1
            self.q, self.chosen = None, None
        else:
            self._update_values()

    def agent_step(self) -> int:
        """Let the player act once; returns the action (``-1`` when a replay
        has run out of actions, which ends the episode with cause ``end``)."""
        if self.player is None:
            raise RuntimeError("no player in human mode")
        if self.done:
            raise RuntimeError("the episode is over; call reset() first")
        if isinstance(self.player, RecordedPlayer) and self.player.exhausted:
            self.done = True
            self.cause = "end"
            self.q, self.chosen = None, None
            return -1
        action = int(self.player.act(self.obs, self.env))
        self.step(action)
        return action

    def straight_action(self) -> int:
        """The action that keeps the current heading."""
        return STRAIGHT if self.action_mode == "relative" else int(self.env.heading)

    def human_action(self, desired_heading: int) -> int:
        """Action that turns the snake towards ``desired_heading``; the reversal
        keeps the current heading (``heading_to_action`` would raise)."""
        desired = int(desired_heading) % 4
        if (desired + 2) % 4 == self.env.heading:
            return self.straight_action()
        return heading_to_action(self.env.heading, desired, self.action_mode)

    # ----------------------------------------------------------- helpers
    def _update_values(self) -> None:
        source = self.player if self.player is not None else self.advisor
        values = action_values(source, self.obs) if source is not None else None
        self.q = None if values is None else np.asarray(values, dtype=np.float64)
        self.chosen = None if self.q is None else int(np.argmax(self.q))


# ----------------------------------------------------------------------- app
def _lerp(a: tuple[int, int, int], b: tuple[int, int, int], t: float) -> tuple[int, int, int]:
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))  # type: ignore[return-value]


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", text).strip("_")[:40] or "player"


class App:
    """pygame view and event loop around a :class:`GameSession`."""

    def __init__(
        self,
        session: GameSession,
        *,
        fps: float = DEFAULT_FPS,
        cell: int = DEFAULT_CELL,
        headless: bool = False,
        record_dir: str | Path | None = None,
        title: str | None = None,
    ) -> None:
        if headless:
            os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        self.session = session
        self.mode = session.mode
        self.headless = bool(headless)
        self.speed = float(min(max(fps, MIN_FPS), MAX_FPS))
        self.cell = int(cell)
        self.paused = session.mode == "human"
        self.quit_requested = False
        self.pending_heading: int | None = None
        self.message = ""
        self.record_dir = Path(record_dir) if record_dir is not None else Path("runs") / "recordings"
        self.steps_taken = 0
        self._accum = 0.0
        self._since_done = 0.0
        pygame.display.init()
        pygame.font.init()
        self.size = window_size(session.W, self.cell)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption(title or f"snake-rl - {MODE_TITLES[self.mode]}")
        self._fonts: dict[tuple[int, bool], pygame.font.Font] = {}

    # ------------------------------------------------------------- events
    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self.quit_requested = True
            return
        if event.type != pygame.KEYDOWN:
            return
        key = event.key
        s = self.session
        if key in (pygame.K_ESCAPE, pygame.K_q):
            self.quit_requested = True
        elif key == pygame.K_SPACE:
            self.paused = not self.paused
        elif key in (pygame.K_n, pygame.K_PERIOD):
            if not s.done:
                self._advance()
        elif key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
            self.speed = min(self.speed * SPEED_FACTOR, MAX_FPS)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed = max(self.speed / SPEED_FACTOR, MIN_FPS)
        elif key == pygame.K_r:
            self.new_episode()
        elif key == pygame.K_s:
            self.save_recording()
        elif key in KEY_HEADINGS and self.mode == "human":
            if s.done:
                return
            self.pending_heading = KEY_HEADINGS[key]
            if self.paused:
                # paused = turn-by-turn play: one step per key press
                self._advance()

    def new_episode(self) -> None:
        """``R``: the next evaluation board (replay: the same episode again)."""
        s = self.session
        s.reset(s.seed if self.mode == "replay" else None)
        self.message = ""
        self.pending_heading = None
        self._accum = 0.0
        self._since_done = 0.0
        if self.mode == "human":
            self.paused = True

    def save_recording(self) -> Path:
        """``S``: write the current episode's recording into ``record_dir``."""
        rec = self.session.recording
        name = f"{_slug(rec.player)}_seed{rec.seed}_{time.strftime('%Y%m%d-%H%M%S')}.json"
        path = rec.save(self.record_dir / name)
        self.message = f"Saved {path}"
        return path

    # --------------------------------------------------------------- time
    def tick(self, dt: float) -> None:
        """Advance game time by ``dt`` seconds: at most one step per call."""
        s = self.session
        if s.done:
            if self.mode == "agent" and not self.paused:
                self._since_done += dt
                if self._since_done >= AUTO_RESET_DELAY:
                    self.new_episode()
            return
        if self.paused:
            return
        period = 1.0 / self.speed
        self._accum = min(self._accum + dt, period)
        if self._accum >= period:
            self._accum = 0.0
            self._advance()

    def _advance(self) -> None:
        s = self.session
        if self.mode == "human":
            action = s.human_action(self.pending_heading) if self.pending_heading is not None else s.straight_action()
            s.step(action)
            self.pending_heading = None
        else:
            s.agent_step()
        self.steps_taken += 1
        if s.done:
            self._since_done = 0.0
            self.message = self._end_message()

    def _end_message(self) -> str:
        s = self.session
        if s.cause == "wall":
            return "Game over: hit the wall"
        if s.cause == "self":
            return "Game over: self collision"
        if s.cause == "starve":
            return f"Starved (truncated at {s.starve_limit} steps without food)"
        if s.cause == "win":
            return "Board full: win"
        if s.cause == "end":
            return "End of recording"
        return ""

    # ----------------------------------------------------------- drawing
    def _font(self, size: int, bold: bool = False) -> pygame.font.Font:
        key = (size, bold)
        if key not in self._fonts:
            self._fonts[key] = pygame.font.SysFont("segoeui,dejavusans,arial", size, bold=bold)
        return self._fonts[key]

    def _text(self, text: str, pos: tuple[int, int], size: int = 16, color=COLOR_TEXT, bold: bool = False,
              right: bool = False) -> pygame.Rect:
        surface = self._font(size, bold).render(text, True, color)
        rect = surface.get_rect()
        if right:
            rect.topright = pos
        else:
            rect.topleft = pos
        self.screen.blit(surface, rect)
        return rect

    def _wrap(self, text: str, size: int, max_width: int, bold: bool = False) -> list[str]:
        """Word-wrap ``text`` to ``max_width`` pixels (a word longer than the
        width stands on its own line)."""
        font = self._font(size, bold)
        lines: list[str] = []
        current = ""
        for word in text.split():
            trial = f"{current} {word}".strip()
            if current and font.size(trial)[0] > max_width:
                lines.append(current)
                current = word
            else:
                current = trial
        if current:
            lines.append(current)
        return lines or [""]

    def _paragraph(self, text: str, pos: tuple[int, int], size: int, max_width: int, color=COLOR_TEXT,
                   bold: bool = False, line_h: int | None = None) -> int:
        """Draw wrapped text; returns the y below it."""
        x, y = pos
        step = line_h if line_h is not None else size + 4
        for line in self._wrap(text, size, max_width, bold):
            self._text(line, (x, y), size, color, bold)
            y += step
        return y

    def panel_lines(self) -> list[str]:
        """The text of the side panel, top to bottom (what the tests inspect)."""
        s = self.session
        labels = ACTION_LABELS[s.action_mode]
        lines: list[str] = []
        values, chosen = s.q, s.chosen
        if s.done and s.prev_q is not None:
            values, chosen = s.prev_q, s.last_action
        if values is None:
            source = "human" if s.player is None else getattr(s.player, "name", "player")
            lines.append(f"No action values: {source}")
        else:
            title = "Action values of the current state"
            if s.mode == "human" and s.advisor is not None:
                title = "Advisor's values: what the agent would do here"
            if s.done:
                title = "Action values before the last move"
            lines.append(title)
            for a, label in enumerate(labels):
                mark = "  (chosen)" if a == chosen else ""
                lines.append(f"{label}  {values[a]:+.2f}{mark}")
        if s.env.cfg.obs.kind == "A":
            bits = "".join("1" if v else "0" for v in s.obs)
            lines.append(f"Observation A: danger S L R | heading | food U D L R = {bits}")
        else:
            lines.append(f"Observation B: {s.env.cfg.obs.n_planes} planes")
        lines.append(f"Score {s.score}")
        lines.append(f"Steps since last food {s.steps_since_food} / {s.starve_limit}")
        lines.append(f"Episode steps {s.steps}")
        lines.append(f"Return ({s.env.cfg.reward.variant}) {s.episode_return:+.2f}")
        lines.append(f"Player: {s.player_label}")
        lines.append(f"Episode {s.episode} (seed {s.seed})")
        ended = " . ".join(f"{CAUSE_LABELS[c]} {s.stats[c]}" for c in ("wall", "self", "starve", "win"))
        lines.append(f"Episodes ended: {ended} ({len(s.history)} episodes)")
        state = "paused" if self.paused else "running"
        lines.append(f"Speed {self.speed:.1f} steps/s . {state}")
        if self.message:
            lines.append(self.message)
        lines.append(KEY_HELP)
        return lines

    def render(self) -> pygame.Surface:
        """Draw the whole frame onto ``screen`` and return it."""
        scr = self.screen
        scr.fill(COLOR_BG)
        self._draw_header()
        self._draw_board()
        self._draw_panel()
        self._draw_footer()
        return scr

    def _draw_header(self) -> None:
        width = self.size[0]
        pygame.draw.rect(self.screen, COLOR_HEADER, (0, 0, width, HEADER_H))
        self._text(f"snake-rl  -  {MODE_TITLES[self.mode]}", (MARGIN, 12), 20, COLOR_HEADER_TEXT, bold=True)
        x = width - MARGIN
        for mode in reversed(MODES):
            label = MODE_TITLES[mode].split()[0]
            font = self._font(14, bold=True)
            w = font.size(label)[0] + 22
            x -= w
            color = COLOR_CHIP_ACTIVE if mode == self.mode else COLOR_CHIP
            pygame.draw.rect(self.screen, color, (x, 12, w, HEADER_H - 24), border_radius=6)
            self._text(label, (x + 11, 15), 14, COLOR_HEADER_TEXT, bold=True)
            x -= 8

    def _draw_board(self) -> None:
        s, cell, scr = self.session, self.cell, self.screen
        W = s.W
        ox, oy = MARGIN, HEADER_H + MARGIN
        pygame.draw.rect(scr, COLOR_BOARD_EDGE, (ox - 3, oy - 3, W * cell + 6, W * cell + 6), border_radius=4)
        for r in range(W):
            for c in range(W):
                rect = board_cell_rect(r, c, cell)
                pygame.draw.rect(scr, COLOR_EMPTY, rect)
                pygame.draw.rect(scr, COLOR_GRID, rect, 1)
        cells = s.env.snake_cells()  # tail -> head
        n = len(cells)
        for i, (r, c) in enumerate(cells[:-1]):
            color = _lerp(COLOR_TAIL, COLOR_BODY, (i + 1) / max(n - 1, 1))
            pygame.draw.rect(scr, color, board_cell_rect(r, c, cell).inflate(-4, -4), border_radius=max(4, cell // 8))
        hr, hc = s.env.head
        head_rect = board_cell_rect(hr, hc, cell).inflate(-2, -2)
        pygame.draw.rect(scr, COLOR_HEAD, head_rect, border_radius=max(4, cell // 8))
        # death markers: the fatal cell (self) or the fatal edge (wall)
        if s.done and s.cause in ("wall", "self"):
            dr, dc = (int(v) for v in DELTAS[s.env.heading])
            tr, tc = hr + dr, hc + dc
            if s.cause == "self" and 0 <= tr < W and 0 <= tc < W:
                pygame.draw.rect(scr, COLOR_DEATH, board_cell_rect(tr, tc, cell).inflate(-4, -4), border_radius=max(4, cell // 8))
            else:
                pygame.draw.rect(scr, COLOR_DEATH, head_rect, width=4, border_radius=max(4, cell // 8))
        # heading arrow on the head
        cx, cy = head_rect.center
        k = max(4, cell // 5)
        arrows = {
            UP: [(cx, cy - k), (cx - k, cy + k // 2), (cx + k, cy + k // 2)],
            DOWN: [(cx, cy + k), (cx - k, cy - k // 2), (cx + k, cy - k // 2)],
            LEFT: [(cx - k, cy), (cx + k // 2, cy - k), (cx + k // 2, cy + k)],
            RIGHT: [(cx + k, cy), (cx - k // 2, cy - k), (cx - k // 2, cy + k)],
        }
        pygame.draw.polygon(scr, COLOR_ARROW, arrows[int(s.env.heading)])
        # food (not after a win: the board is full)
        fr, fc = s.env.food
        if s.env.body_age[fr, fc] == 0:
            pygame.draw.circle(scr, COLOR_FOOD, board_cell_rect(fr, fc, cell).center, max(3, int(cell * 0.3)))
        # end-of-episode banner over the board
        if s.done and self.message:
            band_h = 64
            band = pygame.Surface((W * cell, band_h), pygame.SRCALPHA)
            band.fill((*COLOR_OVERLAY, 215))
            scr.blit(band, (ox, oy + W * cell // 2 - band_h // 2))
            cx = ox + W * cell // 2
            text = self._font(18, bold=True).render(self.message, True, COLOR_HEADER_TEXT)
            scr.blit(text, text.get_rect(center=(cx, oy + W * cell // 2 - 12)))
            hint = "R: new game" if self.mode != "replay" else "R: replay again"
            if self.mode == "agent":
                hint = "next game in a moment (R: now, Space: hold)"
            hint_text = self._font(13).render(hint, True, COLOR_HEADER_TEXT)
            scr.blit(hint_text, hint_text.get_rect(center=(cx, oy + W * cell // 2 + 14)))

    def _draw_bars(self, x: int, y: int, width: int) -> int:
        """The action-value bar chart; returns the y below it."""
        s = self.session
        labels = ACTION_LABELS[s.action_mode]
        values, chosen = s.q, s.chosen
        title = "Action values of the current state"
        if s.mode == "human" and s.advisor is not None:
            title = "Advisor's values: what the agent would do here"
        if s.done and s.prev_q is not None:
            values, chosen, title = s.prev_q, s.last_action, "Action values before the last move"
        if values is None:
            source = "human" if s.player is None else getattr(s.player, "name", "player")
            self._text(f"No action values: {source}", (x, y), 17, COLOR_TEXT, bold=True)
            self._text("(planner / keyboard; nothing to show)", (x, y + 24), 13, COLOR_MUTED)
            return y + 52
        y = self._paragraph(title, (x, y), 17, width, COLOR_TEXT, bold=True, line_h=24) + 6
        label_w, value_w, marker_w = 84, 52, 66
        bar_x = x + label_w
        bar_w = width - label_w - value_w - marker_w - 8
        half = bar_w // 2
        scale = max(float(np.max(np.abs(values))), 1e-6)
        row_h = 26
        for a, label in enumerate(labels):
            v = float(values[a])
            self._text(label, (x, y + 4), 14, COLOR_TEXT)
            track = pygame.Rect(bar_x, y + 3, bar_w, row_h - 8)
            pygame.draw.rect(self.screen, COLOR_BAR_TRACK, track, border_radius=4)
            length = int(round(abs(v) / scale * half))
            color = COLOR_BAR_CHOSEN if a == chosen else (COLOR_BAR_NEGATIVE if v < 0 else COLOR_BAR_OTHER)
            if v >= 0:
                bar = pygame.Rect(bar_x + half, y + 3, length, row_h - 8)
            else:
                bar = pygame.Rect(bar_x + half - length, y + 3, length, row_h - 8)
            if length > 0:
                pygame.draw.rect(self.screen, color, bar, border_radius=4)
            pygame.draw.line(self.screen, COLOR_MUTED, (bar_x + half, y + 1), (bar_x + half, y + row_h - 3), 1)
            self._text(f"{v:+.2f}", (bar_x + bar_w + 8, y + 4), 14, COLOR_TEXT)
            if a == chosen:
                mx = bar_x + bar_w + 8 + value_w
                pygame.draw.polygon(self.screen, COLOR_BAR_CHOSEN, [(mx, y + 6), (mx, y + 18), (mx + 8, y + 12)])
                self._text("chosen", (mx + 12, y + 5), 12, COLOR_BAR_CHOSEN)
            y += row_h
        return y + 6

    def _draw_features(self, x: int, y: int) -> int:
        """Observation A as 11 lit / unlit squares; returns the y below."""
        s = self.session
        if s.env.cfg.obs.kind != "A":
            self._text(f"Observation B: {s.env.cfg.obs.n_planes} planes (head, body, food, age, heading)", (x, y), 13, COLOR_MUTED)
            return y + 22
        self._text("Observation A (what the agent sees)", (x, y), 13, COLOR_MUTED)
        y += 20
        size, gap = 18, 4
        groups = (("danger", ("S", "L", "R")), ("heading", ("U", "R", "D", "L")), ("food", ("U", "D", "L", "R")))
        k = 0
        gx = x
        for name, keys in groups:
            for key in keys:
                on = bool(s.obs[k] > 0.5)
                rect = pygame.Rect(gx, y, size, size)
                pygame.draw.rect(self.screen, COLOR_FEATURE_ON if on else COLOR_FEATURE_OFF, rect, border_radius=3)
                self._text(key, (gx + 4, y + 1), 12, COLOR_HEADER_TEXT if on else COLOR_MUTED)
                gx += size + gap
                k += 1
            self._text(name, (gx - len(keys) * (size + gap), y + size + 2), 11, COLOR_MUTED)
            gx += 14
        return y + size + 20

    def _draw_panel(self) -> None:
        s = self.session
        x0 = MARGIN + s.W * self.cell + MARGIN
        y0 = HEADER_H + MARGIN
        height = s.W * self.cell
        panel = pygame.Rect(x0, y0, PANEL_W, height)
        pygame.draw.rect(self.screen, COLOR_PANEL, panel, border_radius=8)
        pygame.draw.rect(self.screen, COLOR_PANEL_EDGE, panel, width=1, border_radius=8)
        self.screen.set_clip(panel.inflate(-2, -2))
        x, y = x0 + 16, y0 + 14
        inner = PANEL_W - 32
        y = self._draw_bars(x, y, inner)
        pygame.draw.line(self.screen, COLOR_PANEL_EDGE, (x, y), (x + inner, y))
        y += 8
        y = self._draw_features(x, y)
        pygame.draw.line(self.screen, COLOR_PANEL_EDGE, (x, y), (x + inner, y))
        y += 8
        col2 = x + inner // 2
        self._text("Score", (x, y), 12, COLOR_MUTED)
        self._text("Steps since last food", (col2, y), 12, COLOR_MUTED)
        y += 15
        self._text(str(s.score), (x, y), 17, COLOR_TEXT, bold=True)
        self._text(f"{s.steps_since_food} / {s.starve_limit}", (col2, y), 17, COLOR_TEXT, bold=True)
        y += 26
        self._text("Episode steps", (x, y), 12, COLOR_MUTED)
        self._text(f"Return ({s.env.cfg.reward.variant})", (col2, y), 12, COLOR_MUTED)
        y += 15
        self._text(str(s.steps), (x, y), 17, COLOR_TEXT, bold=True)
        self._text(f"{s.episode_return:+.2f}", (col2, y), 17, COLOR_TEXT, bold=True)
        y += 28
        y = self._paragraph(f"Player: {s.player_label}", (x, y), 13, inner, COLOR_TEXT, line_h=17) + 1
        self._text(f"Episode {s.episode} (seed {s.seed})", (x, y), 13, COLOR_TEXT)
        y += 20
        ended = "   ".join(f"{CAUSE_LABELS[c]} {s.stats[c]}" for c in ("wall", "self", "starve", "win"))
        self._text(f"Episodes ended ({len(s.history)}):", (x, y), 12, COLOR_MUTED)
        y += 15
        self._text(ended, (x, y), 14, COLOR_TEXT)
        y += 22
        state = "paused" if self.paused else "running"
        if self.mode == "human" and self.paused:
            state = "paused: arrow keys step, Space runs"
        self._text(f"Speed {self.speed:.1f} steps/s  .  {state}", (x, y), 13, COLOR_TEXT)
        y += 18
        if self.message and not s.done:
            y = self._paragraph(self.message, (x, y), 12, inner, COLOR_MUTED, line_h=15)
        self._text(KEY_HELP, (x, panel.bottom - 22), 11, COLOR_MUTED)
        self.screen.set_clip(None)

    def _draw_footer(self) -> None:
        s = self.session
        text = (f"{s.W} x {s.W} grid . {s.action_mode} actions . white arrow = heading . "
                f"reward {s.env.cfg.reward.variant} . starvation limit {s.starve_limit}")
        self._text(text, (MARGIN, self.size[1] - FOOTER_H + 11), 13, COLOR_MUTED)

    # ---------------------------------------------------------------- run
    def screenshot(self, path: str | Path) -> Path:
        """Render one frame and save it (PNG by extension)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pygame.image.save(self.render(), str(path))
        return path

    def run(self, max_steps: int | None = None, screenshot: str | Path | None = None) -> int:
        """Event loop at 60 frames per second. ``max_steps`` ends it after that
        many environment steps; ``screenshot`` saves the last frame. Headless
        runs advance one step per frame and stop at an episode end that the
        mode does not restart by itself."""
        clock = pygame.time.Clock()
        start = self.steps_taken
        while not self.quit_requested:
            for event in pygame.event.get():
                self.handle_event(event)
            if self.headless:
                if self.mode == "human" and max_steps:
                    self.paused = False  # scripted run: nobody presses a key, the snake goes straight
                dt = 1.0 / self.speed
            else:
                dt = clock.tick(60) / 1000.0
            self.tick(dt)
            self.render()
            pygame.display.flip()
            if max_steps is not None and self.steps_taken - start >= max_steps:
                break
            if self.headless and self.session.done and self.mode != "agent":
                break
        if screenshot is not None:
            self.screenshot(screenshot)
        return 0

    def close(self) -> None:
        pygame.display.quit()


# ------------------------------------------------------------------------ CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="snake-rl GUI: human play, agent play with live action values, replay.")
    p.add_argument("--mode", choices=MODES, default=None,
                   help="default: replay if --replay, else agent if --player, else human")
    p.add_argument("--player", default=None,
                   help="random | bfs | hamiltonian | run directory | *.pt checkpoint | *.npz table (human mode: the advisor)")
    p.add_argument("--replay", default=None, help="recording JSON written with the S key")
    p.add_argument("--W", type=int, default=None, help="board size (default: the player's run config, else 10)")
    p.add_argument("--action-mode", choices=["relative", "absolute"], default=None)
    p.add_argument("--starve-limit", type=int, default=None)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED, help="seed of the first episode (default: evaluation board 0)")
    p.add_argument("--fps", type=float, default=DEFAULT_FPS, help="steps per second")
    p.add_argument("--cell", type=int, default=DEFAULT_CELL, help="pixels per cell")
    p.add_argument("--steps", type=int, default=None, help="exit after this many environment steps")
    p.add_argument("--screenshot", default=None, help="save the last frame to this PNG")
    p.add_argument("--headless", action="store_true", help="no window (SDL dummy driver)")
    p.add_argument("--record-dir", default=None, help="where the S key writes recordings (default runs/recordings)")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    return p


def _build_app(args: argparse.Namespace) -> App:
    mode = args.mode or ("replay" if args.replay else "agent" if args.player else "human")
    if mode == "agent" and not args.player:
        raise ValueError("--mode agent needs --player")
    if mode == "replay" and not args.replay:
        raise ValueError("--mode replay needs --replay FILE")

    player: Any = None
    advisor: Any = None
    env_cfg: EnvConfig | None = None
    label: str | None = None
    if mode == "replay":
        recording = Recording.load(args.replay)
        env_cfg = recording.env_config()
        if args.W is not None and args.W != env_cfg.W:
            raise ValueError(f"the recording is on a {env_cfg.W}x{env_cfg.W} board; --W {args.W} does not match")
        player = RecordedPlayer(recording)
        label = f"replay of {recording.player} (seed {recording.seed})"
    elif args.player:
        loaded = load_player(args.player, device=args.device)
        env_cfg = loaded.env_cfg
        label = loaded.label
        if mode == "human":
            advisor = loaded.player
        else:
            player = loaded.player
    if env_cfg is None:
        env_cfg = EnvConfig()
    overrides: dict[str, Any] = {}
    if args.W is not None:
        overrides["W"] = args.W
    if args.action_mode is not None:
        overrides["action_mode"] = args.action_mode
    if args.starve_limit is not None:
        overrides["starve_limit"] = args.starve_limit
    if overrides:
        env_cfg = replace(env_cfg, **overrides)
    for policy in (player, advisor):
        shape = getattr(policy, "obs_shape", None)
        if shape is not None and tuple(shape) != env_cfg.obs.shape(env_cfg.W):
            raise ValueError(
                f"the checkpoint takes observations of shape {tuple(shape)}, the environment gives {env_cfg.obs.shape(env_cfg.W)}"
            )
        n = getattr(policy, "n_actions", None)
        if n is not None and int(n) != (3 if env_cfg.action_mode == "relative" else 4):
            raise ValueError(f"the checkpoint has {n} actions; the {env_cfg.action_mode} action mode needs another count")
    session = GameSession(env_cfg, player, seed=args.seed, advisor=advisor, mode=mode)
    if label is not None:
        session.player_label = label if mode != "human" else f"human (advisor: {label})"
        session.recording.player = session.player_label
    return App(session, fps=args.fps, cell=args.cell, headless=args.headless, record_dir=args.record_dir)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        app = _build_app(args)
    except (ValueError, FileNotFoundError, KeyError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    try:
        return app.run(max_steps=args.steps, screenshot=args.screenshot)
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
