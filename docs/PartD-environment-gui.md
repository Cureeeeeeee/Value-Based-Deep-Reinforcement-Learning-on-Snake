# Assignment 3, Part D — Environment/Interface Design (text of the submission)

This file is the source text of the Part D document
(`43008-Project-Part-D-TheDeadlyTriad.docx`). The screenshots are rendered by
the application itself under the SDL dummy video driver by
`python scripts/gui_screenshots.py` (files `runs/figures/partD_gui_*.png`), so
every frame can be regenerated; the scores and values quoted in the captions
are read off those frames. The numbers of the environment section come from
`docs/PROGRESS.md` (sections "M2 验收", "M3 验收") and from Table 1 of the Part C
document.

**Project title and number:** Value-Based Deep Reinforcement Learning on Snake: A Controlled Study of State Representation, Reward Design, and DQN Stabilisation — Project number: 48

**Team name:** The Deadly Triad

**Team members:** Yue Liu (25211451, Yue.Liu-14@student.uts.edu.au); Jiahao Liu (14191136, Jiahao.Liu-4@student.uts.edu.au); Siqin Jin (26145324, siqin.jin@student.uts.edu.au)

## Environment chosen

A Snake environment written by us with the Gymnasium API (package `snake_rl.env`, registered as `Snake-v0`), used as the carrier of the controlled study described in Parts A–C. Code, configurations, the interface below and 1,293 automated tests: https://github.com/Cureeeeeeee/Value-Based-Deep-Reinforcement-Learning-on-Snake

- **Board and dynamics.** A W × W grid; 10 × 10 is the main board and 8 × 8 the quick one. The snake starts with length 3 in the centre, heading right. One food item is on the board at any time; it reappears uniformly at random on an empty cell when eaten, and the score is the number of food items eaten. The internal state is one integer array, body_age: 0 = empty, k > 0 = the cell stays occupied for k more steps (tail = 1, head = length). The one array gives body, head, tail and heading, and makes the rule "the tail cell may be entered because it vacates during the move" fall out of the update order.
- **Actions.** Three relative actions, straight / turn left / turn right (default). A four-direction absolute mode (up, right, down, left; the reversal is treated as straight) is kept as an ablation.
- **Episode end.** Terminated: the head leaves the grid, hits the body, or the board is full (win). Truncated: 100 steps without food (the starvation limit, max(100, W²)). Truncation is a time-out and not a death: it gives no penalty, and the learners bootstrap from the next state, because the agent cannot observe the hidden step counter.
- **Observations (first experimental axis).** A: 11 binary features — danger straight / left / right, heading (one-hot, 4), food above / below / left / right of the head. Compact and standard in the Snake literature, but not Markov: the body layout is discarded. B: a multi-plane W × W grid — head, body, food, body_age / length, and four broadcast heading planes (8 planes). The age plane is what makes the state Markov; a switch removes it for the non-Markov ablation.
- **Rewards (second axis).** R0: +1 food, −1 death, 0 otherwise. R1: R0 − 0.01 per step. R2: R0 + γ Φ(s′) − Φ(s) with Φ(s) = −(Manhattan distance to the food) / (2 (W − 1)), the potential-based shaping that leaves the optimal policy unchanged. R3: R0 + 0.01 whenever the move reduces the distance to the food — naive distance shaping, kept deliberately exploitable.
- **Two implementations of one rule set.** A scalar reference environment (readability first; used by the tests, the evaluation protocol and the interface) and a vectorised NumPy environment that steps 128–256 boards at once with automatic reset (training). Rules, observations and rewards are shared code, and a regression test compares the two step by step on 19 configurations (58,580 transitions). Throughput of the vectorised environment with 128–256 boards: about 0.6–1.1 million environment steps per second on observation A.
- **Players shipped with the environment.** Three reference players — random, BFS-greedy (shortest path to the food, ignores self-trapping) and a Hamiltonian cycle that fills the board — and the two learners of Part C, tabular Q-learning and a DQN on observation A. Mean scores on 10 × 10 over the 100 evaluation episodes (learners: also over five training seeds; Part C, Table 1): random 0.2, tabular Q-learning 18.5, DQN 20.3, BFS-greedy 30.6, Hamiltonian 97.

## GUI/Environment Type

Custom environment with a game interface of our own: a pygame application (`python -m snake_rl.gui.app`, pygame 2.6.1) on the same environment class that training and evaluation use; no robot, no external simulator. It has the three modes planned in Part B — human play, agent play with a live action-value panel, and replay of recorded episodes (Figures 1–4) — and is the demonstration tool of Parts F and G.

## Screenshots

@figure partD_gui_advisor.png
Figure 1. Human play, with the tabular Q table of Part C as advisor. The arrow keys (or WASD) set the desired heading; a reversal is ignored. The game starts paused, so each key press plays one step (turn-by-turn play, which suits the advisor panel); Space switches to real time and + / − change the speed. The panel shows the advisor's values for the human's current position and the action it would take (here: straight, +1.15; turning right into the body, −1.00), the 11 features of observation A as lit squares ("what the agent sees": danger to the right, heading up, food up and to the left), the score, the steps since the last food against the starvation limit, the episode steps, the episode return under the configured reward, and the end-of-episode counts of the session. Without a `--player` the panel has no values and the mode is plain human play. Episodes use the evaluation seeds (10000, 10001, …), so the examiner can play a board and then watch the agent on the same board.

@figure partD_gui_agent.png
Figure 2. Agent play: the DQN of Part C (`runs/dqn_A_W10/0`, 1,024,000 environment steps) at step 90 of evaluation board 10000, score 13. The selected player — a reference player, a Q table or any checkpoint of a DQN run — plays live at an adjustable speed. The bar chart is the "what is the agent thinking" panel of the Part B mock-up: the three action values of the current state, centred on zero, with the greedy choice marked. Here the network values "turn left" at −0.97, because the cell to the left is its own body (the danger-left feature is lit), and prefers straight (+0.82) to right (+0.76) because the food is straight ahead.

@figure partD_gui_agent_end.png
Figure 3. End of an episode in agent mode (the third episode of the session, board 10002, score 12 after 109 steps). The fatal cell is marked in red, the panel keeps the action values of the state before the last move, and the counters at the bottom accumulate the end causes over the episodes of the session (three self-collisions so far); in agent mode the next board starts after a short pause. The frame shows the finding of Part C on screen: all three values are about −1 — the agent sees that every move is fatal — but the 11 features gave it no earlier warning that it was enclosing itself.

@figure partD_gui_replay.png
Figure 4. Replay of a recorded episode (the same DQN on board 10001, shown at step 112 of 169). Pressing S in human or agent mode writes the episode to a JSON file — environment configuration, seed, the sequence of actions and the panel values of every step; `--replay FILE` re-simulates it, which is exact because the environment is deterministic given seed and actions, and shows the recorded values in the panel. The early / middle / late clips of the demonstration (Part F) are recordings of this kind from checkpoints of one training run.

**Implementation and testing.** The interface is about 1,000 lines of Python (`snake_rl/gui/app.py`). Its game logic (environment, player, counters, recording) is separate from the drawing and runs without a window, so it is covered by 56 automated tests under the SDL dummy video driver: a session in agent mode ends with the same score, steps and cause as the evaluation protocol on the same seed; a recording replays to the same result; the key mapping, the pause / step / speed / reset / save keys, the pixel colours of head, body, food and the fatal cell, and the command line are checked. The four screenshots above were produced by the application under that driver (`scripts/gui_screenshots.py`), so they can be regenerated.
