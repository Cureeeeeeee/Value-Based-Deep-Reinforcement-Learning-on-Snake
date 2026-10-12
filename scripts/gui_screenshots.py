"""Render the GUI screenshots of the Part D document (M9) without a window.

The frames are produced by the real application (``snake_rl.gui.app``) under
the SDL dummy video driver, so they are what the window shows. Written files
(``--out-dir``, default ``runs/figures``):

    partD_gui_human.png      human play: the keyboard is driven by the BFS
                             player for ``--human-steps`` steps so that the
                             board shows a game in progress
    partD_gui_advisor.png    human play with the tabular Q table as advisor
    partD_gui_agent.png      agent play: the DQN of ``--dqn-run`` after
                             ``--agent-steps`` steps of evaluation board 0
    partD_gui_agent_end.png  agent play at the end of its third episode
                             (death marker, values before the last move,
                             death-cause statistics)
    partD_gui_replay.png     replay of a recorded episode of the DQN on the
                             next evaluation board, two thirds through
    partD_recording.json     that recording

Usage::

    python scripts/gui_screenshots.py [--dqn-run runs/dqn_A_W10/0]
        [--tabular-run runs/tabular_q_W10/0] [--out-dir runs/figures] [--seed 10000]
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")

import pygame  # noqa: E402

from snake_rl.agents.baselines import make_policy  # noqa: E402
from snake_rl.env.common import DOWN, LEFT, RIGHT, UP, next_heading  # noqa: E402
from snake_rl.env.core import EnvConfig  # noqa: E402
from snake_rl.gui.app import (  # noqa: E402
    DEFAULT_SEED,
    App,
    GameSession,
    RecordedPlayer,
    load_player,
)

HEADING_KEYS = {UP: pygame.K_UP, RIGHT: pygame.K_RIGHT, DOWN: pygame.K_DOWN, LEFT: pygame.K_LEFT}


def key(code: int) -> pygame.event.Event:
    return pygame.event.Event(pygame.KEYDOWN, key=code, mod=0, unicode="")


def human_play(app: App, steps: int) -> None:
    """Press the arrow keys a BFS player would press, one step per press."""
    s = app.session
    bfs = make_policy("bfs")
    for _ in range(steps):  # human play starts paused: one step per key press
        if s.done:
            break
        action = bfs.act(s.obs, s.env)
        heading = int(next_heading(s.heading, action, s.action_mode))
        app.handle_event(key(HEADING_KEYS[heading]))


def play_until_done(app: App, episodes: int) -> None:
    """Agent play through ``episodes`` episodes; stops at the end of the last one."""
    s = app.session
    period = 1.0 / app.speed
    finished = 0
    while True:
        app.tick(period)
        if s.done:
            finished += 1
            if finished >= episodes:
                return
            app.new_episode()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--dqn-run", default="runs/dqn_A_W10/0")
    p.add_argument("--tabular-run", default="runs/tabular_q_W10/0")
    p.add_argument("--out-dir", default="runs/figures")
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--human-steps", type=int, default=48)
    p.add_argument("--agent-steps", type=int, default=90)
    args = p.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    dqn = load_player(args.dqn_run)
    table = load_player(args.tabular_run)
    env_cfg = dqn.env_cfg or EnvConfig()

    # 1. human play
    app = App(GameSession(env_cfg, seed=args.seed), headless=True)
    human_play(app, args.human_steps)
    print(app.screenshot(out / "partD_gui_human.png"), "score", app.session.score, "steps", app.session.steps)
    app.close()

    # 2. human play with the tabular Q table as advisor
    session = GameSession(table.env_cfg or env_cfg, seed=args.seed, advisor=table.player)
    session.player_label = f"human (advisor: {table.label})"
    app = App(session, headless=True)
    human_play(app, args.human_steps)
    print(app.screenshot(out / "partD_gui_advisor.png"), "score", session.score)
    app.close()

    # 3. agent play, mid-episode
    session = GameSession(env_cfg, dqn.player, seed=args.seed)
    session.player_label = dqn.label
    app = App(session, headless=True)
    for _ in range(args.agent_steps):
        if session.done:
            break
        app.tick(1.0 / app.speed)
    print(app.screenshot(out / "partD_gui_agent.png"), "score", session.score, "steps", session.steps)
    app.close()

    # 4. agent play at the end of the third episode
    session = GameSession(env_cfg, dqn.player, seed=args.seed)
    session.player_label = dqn.label
    app = App(session, headless=True)
    play_until_done(app, 3)
    print(app.screenshot(out / "partD_gui_agent_end.png"), "history", session.history)
    app.close()

    # 5. a recorded episode of the DQN on the next evaluation board, replayed to two thirds
    session = GameSession(env_cfg, dqn.player, seed=args.seed + 1)
    session.player_label = dqn.label
    session.recording.player = dqn.label
    app = App(session, headless=True)
    play_until_done(app, 1)
    recording = session.recording
    rec_path = recording.save(out / "partD_recording.json")
    app.close()
    replay = GameSession(recording.env_config(), RecordedPlayer(recording), seed=recording.seed)
    replay.player_label = f"replay of {recording.player} (seed {recording.seed})"
    app = App(replay, headless=True)
    for _ in range(max(1, len(recording.actions) * 2 // 3)):
        app.tick(1.0 / app.speed)
    print(app.screenshot(out / "partD_gui_replay.png"), "of", rec_path, "steps", replay.steps, "/", len(recording.actions),
          "recorded", (recording.score, recording.steps, recording.cause))
    app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
