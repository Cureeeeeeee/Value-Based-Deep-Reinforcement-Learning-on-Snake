# snake-rl

**Value-Based Deep Reinforcement Learning on Snake: A Controlled Study of State
Representation, Reward Design, and DQN Stabilisation**

UTS 43008 Reinforcement Learning, Spring 2026, Assignment 3, project 48, team
*The Deadly Triad* (Yue Liu, Jiahao Liu, Siqin Jin).

A self-built Gymnasium-style Snake environment used as the carrier for a
controlled study: tabular Q-learning -> DQN -> Double / Dueling / PER, comparing
state representations (11 binary features vs multi-plane grid), reward designs
(R0-R3, potential-based vs naive distance shaping) and stabilisation components
(+-replay, +-target network), against non-learning baselines (random, BFS greedy,
Hamiltonian cycle). Every configuration runs with 5 seeds and is reported with
confidence intervals and a death-cause breakdown.

## Installation (Windows 11, Python 3.13, RTX 5090)

Order matters: the PyPI Windows wheel of `torch` is CPU-only, so torch is
installed first from the CUDA 12.8 index and is deliberately absent from
`requirements.txt`.

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cu128   # Blackwell (sm_120) needs torch >= 2.7
pip install -r requirements.txt
pip install -e .
python scripts/check_env.py     # versions, CUDA device, GPU matmul timing; exit 1 on any failure
pytest -q
```

Verified 2026-09-30: Python 3.13.7, torch 2.11.0+cu128, NVIDIA driver 616.92,
RTX 5090 reported as compute capability (12, 0). `pygame` 2.6.1 ships a
Python 3.13 wheel; if a future Python version lacks one, substitute
`pygame-ce` (same API).

The environment itself only needs `numpy` and `gymnasium`; a CPU-only `torch`
is enough for small configurations and for Colab.

## Layout

```
snake_rl/
  env/
    common.py    rules shared by both environments (headings, actions, food sampling, body_age helpers)
    core.py      scalar reference environment (gymnasium.Env), readability first
    vec_env.py   vectorised (N, W, W) environment for training, throughput first
    obs.py       observation A (11 features) / B (multi-plane grid), written once
    rewards.py   reward variants R0-R3, written once
  agents/
    baselines.py random / BFS greedy (time-aware) / Hamiltonian cycle
    tabular_q.py tabular Q-learning on observation A (synchronous batch form)
    schedules.py linear epsilon schedule
    dqn/         config.py (torch-free), networks.py (MLP on observation A, CNN on observation B),
                 replay.py (uniform), agent.py (DQN with target network; Double DQN switch;
                 the training loop's replay switch is agent.replay in the config)
  config.py      YAML experiment files: loading, --set overrides, validation, seed layout
  train.py       config-driven training, writes runs/<exp>/<seed>/ (tabular Q and DQN)
  evaluate.py    greedy evaluation protocol with death-cause breakdown, scalar and batched (+ CLI)
  utils.py       git hash and other reproducibility helpers
  analysis/      aggregate.py (results table and learning curves from run directories),
                 plots.py (evaluation-curve figure)
  gui/app.py     pygame: human play (optionally with an advising agent) / agent play with the
                 live action-value panel / replay of recorded episodes; headless screenshots
tests/           pytest; the scalar-vs-vectorised equivalence test is the regression gate
configs/         one yaml per experiment
scripts/         check_env.py, bench_vec_env.py (env throughput), bench_update.py (one gradient step,
                 cpu vs cuda), run_matrix.py (seeds x grid batch runner),
                 gui_screenshots.py (the Part D figures, rendered by the GUI without a window)
notebooks/       Colab notebook regenerating the main figures from logs/checkpoints
runs/            training outputs (not in git)
```

`PLAN.md` holds the frozen environment contract (section 3) and the milestone
list with acceptance criteria (section 4); `docs/PROGRESS.md` tracks progress.

## Environment in one paragraph

`W x W` grid (10 default). State is a single `body_age` array: 0 = empty,
`k > 0` = the cell stays occupied for `k` more steps (tail = 1, head = length).
Relative actions {straight, left, right} (absolute 4-direction mode as an
ablation). Step order: move -> wall? -> eat (tail stays, food re-spawns
uniformly on an empty cell) or move (tail vacates first, then self-collision
check, so entering the vacating tail cell is legal) -> starvation truncation
after `H = max(100, W^2)` steps without food. Wall / self / full-board are
`terminated`; starvation is `truncated` and must be bootstrapped by the learner.

```python
from snake_rl.env import SnakeEnv, EnvConfig, ObsConfig, RewardConfig

env = SnakeEnv(EnvConfig(W=10, obs=ObsConfig(kind="B"), reward=RewardConfig(variant="R2")))
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(0)   # info: cause, score, length, steps
```

## Baselines and evaluation

```powershell
python -m snake_rl.evaluate --policy random bfs hamiltonian --W 10 --episodes 100
```

plays 100 greedy episodes per player on fixed evaluation seeds (episode `i`
uses `10000 + i`, the same boards for every agent), writes
`runs/baselines/<policy>_W10_relative.json` (config, seed, git hash, per-episode
records, summary) and prints a markdown table with mean score, 95 % CI and the
death-cause breakdown. On 10x10 the Hamiltonian player wins every game (97),
BFS greedy scores about 31 and dies mostly by trapping itself, random scores
about 0.2.

## Training, batches and tables

```powershell
python -m snake_rl.train --config configs/tabular_q_10x10.yaml --seed 0          # one run, about 9 s
python -m snake_rl.train --config configs/smoke_8x8.yaml --seed 0                # DQN smoke run, seconds, CPU
python -m snake_rl.train --config configs/smoke_B_8x8.yaml --seed 0              # the same with the CNN on observation B
python scripts/run_matrix.py --config configs/dqn_A_10x10.yaml --seeds 0 1 2 3 4 --jobs 5   # about 10 min on the CPU
python scripts/run_matrix.py --config configs/dqn_B_10x10.yaml --seeds 0 1 2 3 4 --jobs 5   # CNN on observation B, GPU
python scripts/bench_update.py --config configs/dqn_B_10x10.yaml --device cuda   # ms per gradient step (device choice)
python scripts/run_matrix.py --config configs/tabular_q_10x10.yaml --seeds 0 1 2 3 4 --jobs 5
python -m snake_rl.analysis.aggregate --baseline runs/baselines/random_W10_relative.json runs/baselines/bfs_W10_relative.json runs/baselines/hamiltonian_W10_relative.json --exp runs/tabular_q_W10 --out runs/tables/m4_W10.md --curves-dir runs/tables/curves
```

A run writes `runs/<exp>/<seed>/` with `config.yaml` (full configuration, seed,
git hash), `metrics.csv` (the greedy evaluation curve), the checkpoint and
`final_eval.json` (per-episode records of the last evaluation). A run is never
overwritten, and one experiment directory holds seeds of one configuration
only: use `--exp` (or `--exp-prefix` / `--grid` with `run_matrix.py`) whenever
`--set` changes a value. Tables and curves are regenerated from the run
directories by `snake_rl.analysis.aggregate`; learner rows carry a Student-t
interval over training seeds, baseline rows an interval over the 100
evaluation episodes (column `CI over`).

```powershell
python -m snake_rl.analysis.plots --exp "runs/tabular_q_W10=tabular Q-learning" "runs/dqn_A_W10=DQN (MLP)" --baseline "runs/baselines/random_W10_relative.json=random" "runs/baselines/bfs_W10_relative.json=BFS greedy" --out runs/figures/m5_W10_curve.png
```

draws the evaluation curves (mean over seeds with a Student-t band; baselines as
dashed lines). `aggregate --last-k 5` adds the mean of the last five evaluations
per seed, which is less noisy than the last snapshot.

Results on the 11-feature observation (5 seeds, 100 greedy episodes each on the
baselines' boards; last evaluation):

| board | random | tabular Q | tabular Q, gamma = 0 | DQN (MLP) | BFS greedy |
|---|---|---|---|---|---|
| 8x8 | 0.23 | 15.67 +- 0.87 | 16.07 +- 0.10 | 16.11 +- 0.88 | 23.07 |
| 10x10 | 0.18 | 18.51 +- 1.10 | 19.69 +- 0.00 | 20.29 +- 1.55 | 30.61 |

A table trained with gamma = 0 (no look-ahead) reaches the same level, so on
these features the score reflects a reactive policy (avoid the blocked move,
turn to the food), not multi-step value learning; `docs/PROGRESS.md` has the
full analysis.

## GUI (Part D)

```powershell
python -m snake_rl.gui.app                                        # human play on 10x10 (arrows / WASD; starts paused: one step per key, Space runs)
python -m snake_rl.gui.app --player runs/dqn_A_W10/0              # the DQN of that run plays, live action values in the panel
python -m snake_rl.gui.app --player runs/dqn_A_W10/0/checkpoints/step_00012800.pt   # any checkpoint
python -m snake_rl.gui.app --mode human --player runs/tabular_q_W10/0   # human plays, the table advises
python -m snake_rl.gui.app --player bfs --seed 10000               # reference players: random | bfs | hamiltonian
python -m snake_rl.gui.app --replay runs/recordings/<file>.json    # re-simulate an episode saved with S
python scripts/gui_screenshots.py                                  # the Part D figures, headless -> runs/figures/partD_gui_*.png
```

The GUI runs the scalar `SnakeEnv` with the configuration of the player's run
(board, observation, action mode, reward). Episodes use the evaluation seeds
(10000, 10001, ...), so a human and an agent can be compared on the same
boards. Keys: Space pause, N step, R new game, S save the episode as a
recording, + / - speed, Q quit. Headless operation (`--headless`, SDL dummy
driver) is what the tests and the screenshot script use.
