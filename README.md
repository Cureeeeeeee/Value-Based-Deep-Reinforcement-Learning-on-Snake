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

## Setup (Windows 11, Python 3.13, RTX 5090)

```powershell
py -3.13 -m venv .venv
.venv\Scripts\activate
pip install torch --index-url https://download.pytorch.org/whl/cu128   # Blackwell needs torch >= 2.7
pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # must print True
pytest
```

The environment itself only needs `numpy` and `gymnasium`; a CPU-only `torch`
is enough for small configurations.

## Layout

```
snake_rl/
  env/
    common.py    rules shared by both environments (headings, step order helpers, food sampling)
    core.py      scalar reference environment (gymnasium.Env), readability first
    vec_env.py   vectorised (N, W, W) environment for training, throughput first   [M3]
    obs.py       observation A (11 features) / B (multi-plane grid), written once
    rewards.py   reward variants R0-R3, written once
  agents/        baselines, tabular Q-learning, DQN (networks / replay / agent)    [M2+]
  analysis/      aggregation over seeds, plots                                      [M10]
  gui/           pygame app: human play / agent play with live Q-values / replay    [M9]
tests/           pytest; the scalar-vs-vectorised equivalence test is the regression gate
configs/         one yaml per experiment
scripts/         batch runners (seeds x configs)
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
