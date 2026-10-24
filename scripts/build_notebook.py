"""Write ``notebooks/results.ipynb``, the Part E notebook, from the package sources.

The notebook carries the code of the study: the environment, the players, the
DQN, the experiment configuration, the evaluation protocol and the training loop
are the modules of ``snake_rl`` inlined one module per code cell (module docstring,
import statements and the command-line wrappers removed; every import gathered in
one cell), so the code in the notebook is the code that trained the experiments,
character for character. The notebook then trains the 8 x 8 tabular agent and two
DQN runs, rebuilds every table and figure of the report from the training logs
and checks the tables against the report source.

    python scripts/build_notebook.py                 # writes notebooks/results.ipynb
    python scripts/build_notebook.py --check         # exit 1 if the file differs from the sources

``tests/test_m10_notebook.py`` runs the check, so the notebook cannot drift from
the modules. The layout follows the Assignment 2 notebooks: numbered sections,
one descriptive comment line at the top of every code cell, no assert cells.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "notebooks" / "results.ipynb"
PACKAGE = "snake_rl"

#: Imports the notebook's own cells need beyond those of the modules.
EXTRA_IMPORTS = (
    "import glob",
    "import shutil",
    "import urllib.request",
    "import zipfile",
    "from IPython.display import Image, Markdown, display",
    "from snake_rl.analysis import report_tables as rt",
    "from snake_rl.analysis.plots import plot_learning_curves",
    "from snake_rl.utils import git_hash",
)
THIRD_PARTY = ("numpy", "torch", "yaml", "gymnasium", "IPython", "pandas", "matplotlib")

#: The modules inlined, in dependency order: (path, descriptive comment line, functions dropped).
MODULES = (
    ("snake_rl/env/common.py", "Rules shared by both environments: headings, actions, food sampling and the body_age grid", ()),
    ("snake_rl/env/obs.py", "Observation A (11 binary features) and observation B (multi-plane grid), batched encoders", ()),
    ("snake_rl/env/rewards.py", "Reward variants R0 to R3, computed from the transition in batched form", ()),
    ("snake_rl/env/core.py", "The scalar reference environment (gymnasium.Env): one board, the step order is the rule set", ()),
    ("snake_rl/env/vec_env.py", "The vectorised training environment: N boards as one (N, W, W) batch, auto-reset, final_obs", ()),
    ("snake_rl/agents/__init__.py", "The Policy protocol every player implements: act(obs, env) -> action, reset()", ()),
    ("snake_rl/agents/schedules.py", "The linear epsilon schedule, a pure function of the step counter", ()),
    ("snake_rl/agents/baselines.py", "Reference players: random, BFS greedy (time-aware) and the Hamiltonian cycle", ()),
    ("snake_rl/agents/tabular_q.py", "Tabular Q-learning on observation A: 11-bit state index, synchronous batch update, epsilon-greedy", ()),
    ("snake_rl/agents/dqn/config.py", "DQNConfig: the hyper-parameters of the agent and of its training loop", ()),
    ("snake_rl/agents/dqn/replay.py", "Experience replay: the uniform FIFO buffer and the prioritised buffer with importance weights", ()),
    ("snake_rl/agents/dqn/networks.py", "Q-networks: the MLP on observation A, the CNN on observation B, the Dueling head", ()),
    ("snake_rl/agents/dqn/agent.py", "DQNAgent: acting, the Huber gradient step, Double DQN target, target network, checkpoints", ()),
    ("snake_rl/config.py", "Experiment configuration: YAML file -> validated frozen dataclasses, --set overrides, seed layout", ()),
    ("snake_rl/evaluate.py", "Evaluation protocol: greedy episodes on fixed boards, death causes, value estimate against realised return", ("main",)),
    ("snake_rl/train.py", "Config-driven training: the tabular and the DQN trainers, the run directory and its files", ("main",)),
)


# ------------------------------------------------------------------ source handling
def _is_internal(node: ast.stmt) -> bool:
    if isinstance(node, ast.ImportFrom):
        return node.level > 0 or (node.module or "").split(".")[0] == PACKAGE
    if isinstance(node, ast.Import):
        return all(alias.name.split(".")[0] == PACKAGE for alias in node.names)
    return False


def _is_future(node: ast.stmt) -> bool:
    return isinstance(node, ast.ImportFrom) and node.module == "__future__"


def _is_main_guard(node: ast.stmt) -> bool:
    return (isinstance(node, ast.If) and isinstance(node.test, ast.Compare)
            and isinstance(node.test.left, ast.Name) and node.test.left.id == "__name__")


def _start(node: ast.AST) -> int:
    return min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])


def _tidy(lines: list[str]) -> str:
    """Trim blank lines at both ends and collapse runs of blank lines (two at module level, one inside a block)."""
    out: list[str] = []
    blanks = 0
    for line in lines:
        if line.strip() == "":
            blanks += 1
            continue
        if out:
            out.extend([""] * min(blanks, 1 if line.startswith((" ", "\t")) else 2))
        blanks = 0
        out.append(line.rstrip())
    return "\n".join(out)


def strip_module(source: str, drop: tuple[str, ...] = ()) -> tuple[str, list[ast.stmt]]:
    """The module without its docstring, its import statements, its ``__main__`` guard and the
    functions in ``drop``; returns the code and the external import statements that were removed."""
    tree = ast.parse(source)
    removed: set[int] = set()
    external: list[ast.stmt] = []

    def mark(node: ast.AST) -> None:
        removed.update(range(_start(node), node.end_lineno + 1))

    body = tree.body
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        mark(body[0])
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if not _is_internal(node) and not _is_future(node):
                external.append(node)
            mark(node)
        elif _is_main_guard(node):
            mark(node)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node.name in drop:
            mark(node)
    for node in ast.walk(tree):
        if isinstance(node, ast.If) and node.body and not node.orelse and all(_is_internal(s) for s in node.body):
            mark(node)  # an ``if TYPE_CHECKING:`` block that only imported package names
        elif _is_internal(node):
            mark(node)  # an import inside a function
    kept = [line for i, line in enumerate(source.splitlines(), 1) if i not in removed]
    return _tidy(kept), external


def top_level_names(code: str) -> set[str]:
    names: set[str] = set()
    for node in ast.parse(code).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                for leaf in ast.walk(target):
                    if isinstance(leaf, ast.Name):
                        names.add(leaf.id)
    return names


def _import_group(module: str) -> int:
    top = module.split(".")[0]
    if top == PACKAGE:
        return 2
    if top in THIRD_PARTY:
        return 1
    return 0


def import_cell(external: list[ast.stmt]) -> tuple[str, set[str]]:
    """One import cell from the modules' external imports plus EXTRA_IMPORTS; also the names it binds."""
    plain: set[str] = set()
    froms: dict[str, set[str]] = {}
    bound: set[str] = set()
    for node in external + [ast.parse(line).body[0] for line in EXTRA_IMPORTS]:
        if isinstance(node, ast.Import):
            for alias in node.names:
                plain.add(f"import {alias.name}" + (f" as {alias.asname}" if alias.asname else ""))
                bound.add(alias.asname or alias.name.split(".")[0])
        else:
            for alias in node.names:
                froms.setdefault(node.module, set()).add(alias.name + (f" as {alias.asname}" if alias.asname else ""))
                bound.add(alias.asname or alias.name)
    lines = ["from __future__ import annotations"]
    for group in (0, 1, 2):
        block = sorted(line for line in plain if _import_group(line.split()[1]) == group)
        block += [f"from {module} import {', '.join(sorted(names))}" for module, names in sorted(froms.items())
                  if _import_group(module) == group]
        if block:
            lines.append("")
            lines.extend(block)
    return "\n".join(lines), bound


def module_cell(path: str, comment: str, drop: tuple[str, ...]) -> tuple[str, list[ast.stmt]]:
    code, external = strip_module((REPO / path).read_text(encoding="utf-8"), drop)
    gap = "\n\n\n" if code.startswith(("@", "class ", "def ")) else "\n\n"
    return f"# {comment}\nfrom __future__ import annotations{gap}{code}", external


# ------------------------------------------------------------------ the notebook
class Notebook:
    def __init__(self) -> None:
        self.cells: list[dict] = []

    def md(self, text: str) -> None:
        self.cells.append({"cell_type": "markdown", "metadata": {}, "source": _source(text)})

    def code(self, text: str) -> None:
        self.cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                           "source": _source(text)})

    def to_dict(self) -> dict:
        for i, cell in enumerate(self.cells):
            cell["id"] = f"cell-{i:02d}"
        return {
            "cells": self.cells,
            "metadata": {
                "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                "language_info": {"name": "python"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }


def _source(text: str) -> list[str]:
    return text.strip("\n").splitlines(keepends=True)


def _notebook_names(code: str) -> set[str]:
    """Top-level names a notebook cell binds (IPython magics and shell lines skipped)."""
    plain = "\n".join(line[:len(line) - len(line.lstrip())] + "pass" if line.lstrip().startswith(("!", "%")) else line
                      for line in code.splitlines())
    return top_level_names(plain)


def build() -> dict:
    nb = Notebook()
    module_code: dict[str, str] = {}
    external: list[ast.stmt] = []
    for path, comment, drop in MODULES:
        module_code[path], ext = module_cell(path, comment, drop)
        external.extend(ext)
    imports, bound = import_cell(external)

    defined: dict[str, str] = {}
    for path, code in module_code.items():
        for name in top_level_names(code.split("\n", 2)[2]):
            if name in defined:
                raise SystemExit(f"{name} is defined in {defined[name]} and in {path}: the modules cannot share one namespace")
            defined[name] = path
    clash = bound & set(defined)
    if clash:
        raise SystemExit(f"imported names shadow module names: {sorted(clash)}")

    nb.md("""
# 43008 Reinforcement Learning -- Assignment 3, Part E: Value-Based Deep Reinforcement Learning on Snake

**The Deadly Triad (project 48): Yue Liu (25211451), Jiahao Liu (14191136), Siqin Jin (26145324)**
""")

    nb.md("""
**Project**: a controlled study of state representation, reward design and DQN stabilisation on a self-built Snake
environment (8 x 8 and 10 x 10 boards): tabular Q-learning; DQN on an 11-feature vector (observation A, MLP) or an
8-plane grid (observation B8, CNN); the discount factor; the replay buffer and the target network; Double DQN, Dueling
and prioritised replay; four reward designs (R0 sparse, R1 step penalty, R2 potential-based shaping, R3 naive distance
bonus) and a reward-hacking probe; random, BFS-greedy and Hamiltonian-cycle reference players. Every configuration is
trained with five seeds, the seed is the unit of replication and every interval is a 95 % Student-t interval over seeds.

**The code**: Sections 2 to 4 are the code that trained the study. Each code cell is one module of the `snake_rl`
package (repository: https://github.com/Cureeeeeeee/Value-Based-Deep-Reinforcement-Learning-on-Snake), inlined
character for character apart from the module docstring, the import statements (gathered in Section 1) and the
command-line wrappers: the environment and its two observation encodings and four reward variants, the reference
players, tabular Q-learning, the replay buffers, the Q-networks, the DQN agent, the experiment configuration, the
evaluation protocol and the training loop. `scripts/build_notebook.py` writes this notebook from those modules and
`tests/test_m10_notebook.py` fails if the two ever differ. The analysis modules (`snake_rl.analysis`: aggregation
over seeds, figures, behaviour analysis, the experiment matrix, `snake_rl.analysis.report` and
`snake_rl.analysis.report_tables`), `snake_rl.utils` and the pygame GUI (`snake_rl.gui`) are imported from the
repository and not repeated here.

**What the notebook does**: Section 5 trains with the code above, in this session: the 8 x 8 tabular agent of the
report and one DQN run on each observation (short configurations by default, the report's 8 x 8 configurations with
`TRAIN_MODE = "full"`), evaluated against the three reference players on the same boards. Sections 6 to 10 then take
the training logs of the whole study (64 configurations, 5 seeds each) and regenerate every table and figure of the
report, show them in the report's order with the report's captions, and check every computed table against the
report source cell by cell: no number of the report is typed by hand. Section 11 runs the behaviour analysis behind
Table 9 and Figures 11 to 13 on one trained network.

**Training logs**: `runs/` is not in the repository (about 1 GB with checkpoints). Three ways to get the logs into
this session: (1) upload `runs_logs.zip`, made with `python scripts/export_logs.py --include-behaviour` on the machine
that trained, when the cell of Section 6 asks for it; (2) set `LOGS_URL` in that cell to a link to the zip (for
example a GitHub release asset); (3) run the notebook inside a checkout that already has `runs/`. The zip holds the
per-run logs, the reference-player evaluations, the saved behaviour analyses (Table 9, Figures 11 to 13) and the
screenshot of Figure 1; it carries no weights.
""")

    nb.md("### 1. Repository, Imports and Settings")

    nb.code("""
# Clone the repository and install the package on Google Colab; a local checkout is used as it is, also when opened from notebooks/
import os
import subprocess
import sys

IN_COLAB = "google.colab" in sys.modules
REPO_URL = "https://github.com/Cureeeeeeee/Value-Based-Deep-Reinforcement-Learning-on-Snake.git"

if os.path.isdir(os.path.join("..", "snake_rl")):
    os.chdir("..")
elif not os.path.isdir("snake_rl"):
    if not os.path.isdir("snake-rl"):
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "snake-rl"], check=True)
    os.chdir("snake-rl")
if IN_COLAB:
    %pip install -q -e .
print("working directory:", os.getcwd())
""")

    nb.code(f"""
# Every import of the modules below, gathered here, plus the display helpers and the analysis modules of the repository
{imports}

REPORT_MD = "docs/PartE-final-report.md"   # the report source: captions, figure files and the tables to check against
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"python {{sys.version.split()[0]}}  torch {{torch.__version__}}  numpy {{np.__version__}}  device {{DEVICE}}")
""")

    # ---------------------------------------------------------------- Section 2: environment
    nb.md("""
### 2. The Snake Environment (`snake_rl/env`)

The game is written once: the rules in `common.py`, the observation encoders in `obs.py` and the reward variants in
`rewards.py` are shared by the scalar reference environment (`core.py`, a `gymnasium.Env` used by the tests, the GUI
and the evaluation) and by the vectorised training engine (`vec_env.py`), which must agree with the scalar one step by
step (`tests/test_vec_env.py` is the regression gate).
""")

    nb.md("""
#### 2.1 Rules shared by both environments

Grid cell `(r, c)` with `r` growing downwards; headings clockwise (0 = up, 1 = right, 2 = down, 3 = left); two action
modes (relative: straight / left / right, the mode of the study; absolute: the four headings). The snake is one
`W x W` integer grid `body_age`: 0 = empty, `k > 0` = this cell stays occupied for `k` more steps, so the tail is 1,
the neck `length - 1` and the head `length`. Every helper takes arrays with a leading `N` axis; the scalar environment
calls them with `N = 1`.
""")
    nb.code(module_code["snake_rl/env/common.py"])

    nb.md("""
#### 2.2 Observations A and B

Observation **A** is the community-standard hand-crafted vector of 11 binary features: danger straight / left / right,
heading one-hot (4), food above / below / left / right of the head (4); it discards the body layout and is not Markov.
Observation **B** is a multi-plane grid `(C, W, W)`: head, body, food, `body_age / length`, and four broadcast heading
planes (8 planes; switches drop the age plane and the heading planes). The age plane is what makes the state Markov:
it recovers the neck and the tail.
""")
    nb.code(module_code["snake_rl/env/obs.py"])

    nb.md("""
#### 2.3 Reward variants R0 to R3

R0: +1 food, -1 death (wall or self), 0 otherwise; starvation truncation is not a death and gets no -1, the learner
must bootstrap it. R1: R0 minus a step penalty. R2: R0 plus potential-based shaping `gamma * Phi(s') - Phi(s)` with
`Phi = -manhattan(head, food) / (2 (W - 1))` (Ng et al. 1999: the optimal policy is unchanged). R3: R0 plus a naive
distance bonus (`closer` when the move reduced the distance to the food, else `farther`); the default asymmetric
version (+0.01 / 0) makes oscillating a net gain, which is the reward-hacking probe of the report.
""")
    nb.code(module_code["snake_rl/env/rewards.py"])

    nb.md("""
#### 2.4 The scalar reference environment

One board behind the Gymnasium API. The step order is the rule set: the new heading from the action; off-grid ends
the episode (`wall`); the food cell is eaten (the tail stays, `length += 1`, new food on a free cell; a full board
is a `win`); otherwise every positive cell is decremented so the tail vacates, a positive head cell ends the episode
(`self`), moving into the vacating tail cell is legal; `H` steps without food truncate the episode (`starve`), which
is not a terminal state because `steps_since_food` is not observable.
""")
    nb.code(module_code["snake_rl/env/core.py"])

    nb.md("""
#### 2.5 The vectorised training environment

`N` boards advance together as one `(N, W, W)` batch of `body_age` grids; every rule is a masked array operation.
Sub-environment `i` behaves bitwise like a scalar `SnakeEnv` reset with `seed + i`. Finished sub-environments reset at
once: `step` returns the first observation of the new episode and keeps the true successor of the transition in
`info["final_obs"]`, which is what the learners store.
""")
    nb.code(module_code["snake_rl/env/vec_env.py"])

    nb.md("#### 2.6 A first look at the environment")
    nb.code("""
# One 8 x 8 board under both observations, and the vectorised engine checked against the scalar one for 300 random steps on four boards
board_A = SnakeEnv(EnvConfig(W=8), render_mode="ansi")
obs_A, _ = board_A.reset(seed=0)
print(board_A.render(), "\\n")
print("observation A:", obs_A.astype(int), "-> danger straight/left/right, heading one-hot, food up/down/left/right")
board_B = SnakeEnv(EnvConfig(W=8, obs=ObsConfig(kind="B")))
obs_B, _ = board_B.reset(seed=0)
print("observation B:", obs_B.shape, "planes (head, body, food, age, heading x 4); food plane:\\n", obs_B[2].astype(int))

check_cfg = EnvConfig(W=8)
vec = VecSnakeEnv(check_cfg, n_envs=4, seed=100)
vec_obs = vec.reset()
scalars = [SnakeEnv(check_cfg) for _ in range(4)]
scalar_obs = np.stack([env.reset(seed=100 + i)[0] for i, env in enumerate(scalars)])
rng = np.random.default_rng(0)
mismatches = 0
for _ in range(300):
    actions = rng.integers(0, vec.n_actions, size=4)
    vec_obs, vec_reward, vec_term, vec_trunc, vec_info = vec.step(actions)
    for i, env in enumerate(scalars):
        s_obs, s_reward, s_term, s_trunc, _ = env.step(int(actions[i]))
        mismatches += int(not np.array_equal(s_obs, vec_info["final_obs"][i]) or s_reward != vec_reward[i] or s_term != vec_term[i])
        if s_term or s_trunc:
            s_obs, _ = env.reset()
        mismatches += int(not np.array_equal(s_obs, vec_obs[i]))
print("vectorised against scalar environment, 4 boards x 300 steps:", "identical" if mismatches == 0 else f"{mismatches} mismatches")
""")

    # ---------------------------------------------------------------- Section 3: players
    nb.md("""
### 3. The Players (`snake_rl/agents`)

Every player implements the `Policy` protocol used by the evaluation: `act(obs, env) -> action` and `reset()`. The
reference players read the whole environment through `env`; the learned agents use only `obs`.
""")

    nb.md("#### 3.1 The Policy protocol and the epsilon schedule")
    nb.code(module_code["snake_rl/agents/__init__.py"])
    nb.code(module_code["snake_rl/agents/schedules.py"])

    nb.md("""
#### 3.2 Reference players

`RandomPolicy` is the floor. `BFSGreedyPolicy` follows the shortest path to the food on the grid (time-aware: a body
cell of age `k` is passable if the head arrives after at least `k` steps, the rule `step` applies), ignores
self-trapping and falls back to the first safe action. `HamiltonianPolicy` follows a fixed Hamiltonian cycle (even
`W`), never collides and never starves, so it wins every game: the upper bound in score and the lower bound in speed.
""")
    nb.code(module_code["snake_rl/agents/baselines.py"])

    nb.md("""
#### 3.3 Tabular Q-learning on observation A

The reference point before function approximation: a state is the 11-bit integer of observation A, the table is
`q[2048, n_actions]`, and the update is `Q[s, a] += alpha * (r + gamma * max_a' Q[s', a'] - Q[s, a])` with the target
zero only when the transition `terminated` (a starvation truncation bootstraps). Training steps `N` boards at once,
so the update takes `N` transitions and moves every distinct cell by `alpha` times the mean TD error of its
transitions (duplicates are averaged, not summed). Exploration is epsilon-greedy from an explicit generator.
""")
    nb.code(module_code["snake_rl/agents/tabular_q.py"])

    nb.md("""
#### 3.4 The DQN hyper-parameters

A torch-free dataclass of values (discount, learning rate, batch and buffer sizes, `learning_starts`, the replay
switch, `target_update_every`, gradient clipping, the epsilon schedule, the network layout, Double / Dueling / PER
switches, checkpoints, device); ranges are validated once, in the experiment configuration of Section 4.1.
""")
    nb.code(module_code["snake_rl/agents/dqn/config.py"])

    nb.md("""
#### 3.5 Experience replay

`ReplayBuffer`: a fixed-size FIFO store of `(s, a, r, s', terminated)` in preallocated numpy arrays, sampled
uniformly from a generator the caller passes in, so a batch mixes many boards and many past policies.
`PrioritizedReplayBuffer` (Schaul et al. 2016, proportional variant): priority `(|td| + eps) ** alpha`, stratified
sampling, importance-sampling weights `(n P(i)) ** (-beta)` normalised to at most 1, `beta` annealed to 1 by the
training loop; a cumulative sum over the priorities instead of a sum tree.
""")
    nb.code(module_code["snake_rl/agents/dqn/replay.py"])

    nb.md("""
#### 3.6 The Q-networks

`MLP` on the 11 features of observation A, `CNN` on the `(C, W, W)` planes of observation B, and the `DuelingMLP`
head `Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a')` (Wang et al. 2016), used in place of the plain head on A and as
the fully connected part of the CNN on B. `build_q_network` picks the architecture from the observation shape, so the
agent never names one.
""")
    nb.code(module_code["snake_rl/agents/dqn/networks.py"])

    nb.md("""
#### 3.7 The DQN agent

Online network `theta` and target network `theta^-` of the same architecture, Adam. One gradient step on a batch:
`target = r + gamma * (0 if terminated else max_a' Q(s', a'; theta^-))`, Huber loss of `Q(s, a; theta) - target`,
global gradient norm clipped; the target is a constant (semi-gradient), and `theta^-` is copied from `theta` every
`target_update_every` gradient steps. Double DQN selects `a*` with the online network and values it with the target
network; prioritised replay passes importance weights into the loss and gets the new TD errors back as priorities.
Greedy ties go to the lowest action index; `act` is `act_batch` on a batch of one.
""")
    nb.code(module_code["snake_rl/agents/dqn/agent.py"])

    # ---------------------------------------------------------------- Section 4: config / evaluate / train
    nb.md("### 4. Experiment Configuration, Evaluation Protocol and Training Loop")

    nb.md("""
#### 4.1 Experiment configuration

One YAML file (`configs/*.yaml`) becomes validated, frozen dataclasses (`EnvConfig`, `ObsConfig`, `RewardConfig`,
`TabularQConfig` or `DQNConfig`, `TrainConfig`); `--set key=value` overrides change single values, and one experiment
name is one configuration. The training environment of run seed `s` is seeded `100000 * (s + 1)`, so the generators
of different runs never overlap.
""")
    nb.code(module_code["snake_rl/config.py"])

    nb.md("""
#### 4.2 Evaluation protocol

`evaluate(policy, cfg, n_episodes, seed)` plays greedy episodes on the scalar environment, episode `i` reset with
`seed + i`: the same boards for every agent and every reference player. Per episode: score, steps, length and the end
cause; the summary gives the mean score with its 95 % interval, the mean steps and the cause breakdown.
`evaluate_batched` is the same protocol on the vectorised environment (the first episode of each sub-environment),
used by the training loop; with `gamma` both also record the value estimate `max_a Q` against the realised discounted
return, the overestimation measurement of the Double DQN study. Results serialise to JSON with the environment
configuration and the git hash.
""")
    nb.code(module_code["snake_rl/evaluate.py"])

    nb.md("""
#### 4.3 Training

`train(cfg, seed, runs_dir)` writes `<runs_dir>/<exp>/<seed>/`: `config.yaml` (every field, the git hash and the
software versions, written before training starts), `metrics.csv` (one row per evaluation), `q_table.npz` or
`checkpoints/` and `model.pt`, and `final_eval.json`, whose presence marks a finished run; a run is never overwritten.
The tabular loop: epsilon from the linear schedule, `act_epsilon` on `N` boards, one `step` of the vectorised
environment, `update` on the true successors `info["final_obs"]` with the `terminated` flags, an evaluation on the
fixed boards every `eval_every` steps. The DQN loop puts the replay buffer between acting and learning: each vec-step
adds `N` transitions and, after `learning_starts`, is followed by `G` gradient steps on sampled batches (default one
gradient step per four environment steps); without replay (`agent.replay: false`) exactly one gradient step on the
vec-step's own transitions. A non-finite loss aborts the run (`aborted.json`). The command-line wrapper `main` is
left out; it parses `--config --seed --exp --set --runs-dir` and calls `train`.
""")
    nb.code(module_code["snake_rl/train.py"])

    # ---------------------------------------------------------------- Section 5: training here
    nb.md("""
### 5. Training in This Notebook

The code above trains here. `TRAIN_MODE = "demo"` runs the 8 x 8 tabular configuration of the report (500,000
environment steps, about ten seconds) and the two short DQN configurations of the repository's smoke tests (16,000
steps on observation A with 64-unit layers; 48,000 steps on observation B with a 16 / 32-channel CNN), about two
minutes on a CPU in all. `TRAIN_MODE = "full"` runs one seed of the report's 8 x 8 DQN configurations instead
(512,000 steps, 125,504 gradient steps each; about 5 minutes for the MLP and 40 minutes for the CNN on the training
machine's CPU). The runs are written under `runs_notebook/`, apart from the logs of the study; a finished run is
reused when the notebook is run again.
""")

    nb.md("#### 5.1 Settings")
    nb.code("""
# "demo" = the quick configurations (about two minutes on a CPU); "full" = one seed of the 8 x 8 configurations of the report
TRAIN_MODE = "demo"
NOTEBOOK_RUNS = "runs_notebook"          # the runs of this notebook, apart from the logs of the study under runs/
GPU = ("agent.device=cuda",) if torch.cuda.is_available() else ()

CONFIGS = {
    "demo": {"tabular": ("configs/tabular_q_8x8.yaml", ()),
             "dqn_A": ("configs/smoke_8x8.yaml", ()),
             "dqn_B": ("configs/smoke_B_8x8.yaml", ())},
    "full": {"tabular": ("configs/tabular_q_8x8.yaml", ()),
             "dqn_A": ("configs/dqn_A_8x8.yaml", GPU),
             "dqn_B": ("configs/dqn_B_8x8.yaml", GPU)},
}[TRAIN_MODE]


def train_or_reuse(config_file, overrides=(), seed=0):
    # one run of config_file under NOTEBOOK_RUNS: a finished run is reused, an unfinished one is removed and repeated
    exp_cfg = load_config(config_file, overrides)
    run_dir = os.path.join(NOTEBOOK_RUNS, exp_cfg.exp, str(seed))
    if os.path.exists(os.path.join(run_dir, FINAL_EVAL_FILE)):
        print(f"[{exp_cfg.exp}/{seed}] finished run found -> {run_dir}")
        return run_dir
    shutil.rmtree(run_dir, ignore_errors=True)
    return str(train(exp_cfg, seed, NOTEBOOK_RUNS))


print(f"TRAIN_MODE = {TRAIN_MODE}:", ", ".join(f"{k} <- {v[0]}" for k, v in CONFIGS.items()))
""")

    nb.md("#### 5.2 The reference players on the evaluation boards")
    nb.code("""
# Random, BFS-greedy and Hamiltonian players on the 100 evaluation boards of the 8 x 8 study (episode i plays the board of seed 10000 + i)
board_cfg = load_config(CONFIGS["tabular"][0]).env
baseline_results = [evaluate(make_policy(name), board_cfg, n_episodes=100, seed=DEFAULT_EVAL_SEED) for name in BASELINE_NAMES]
os.makedirs(os.path.join(NOTEBOOK_RUNS, "baselines"), exist_ok=True)
baseline_files = [str(r.save(os.path.join(NOTEBOOK_RUNS, "baselines", f"{r.policy}_W8.json"))) for r in baseline_results]
display(Markdown(markdown_table(baseline_results)))
""")

    nb.md("#### 5.3 Tabular Q-learning")
    nb.code("""
# Tabular Q-learning on observation A, 8 x 8: the configuration of the report (500,000 environment steps on 50 boards, one evaluation every 25,000 steps)
run_dirs = {}
run_dirs["tabular"] = train_or_reuse(*CONFIGS["tabular"])
""")

    nb.md("#### 5.4 DQN on observation A")
    nb.code("""
# DQN with the MLP on the 11 features: the smoke configuration in demo mode, the report's 8 x 8 configuration in full mode
run_dirs["dqn_A"] = train_or_reuse(*CONFIGS["dqn_A"])
""")

    nb.md("#### 5.5 DQN on observation B")
    nb.code("""
# DQN with the CNN on the 8-plane grid: the smoke configuration in demo mode, the report's 8 x 8 configuration in full mode
run_dirs["dqn_B"] = train_or_reuse(*CONFIGS["dqn_B"])
""")

    nb.md("#### 5.6 Learning curves and the final evaluation")
    nb.code("""
# Evaluation curves of the three runs against the reference players, then the final greedy evaluation of every player on the same boards
labels = {"tabular": "tabular Q-learning (A)", "dqn_A": "DQN (A, MLP)", "dqn_B": "DQN (B8, CNN)"}
baselines = [(r.policy, f) for r, f in zip(baseline_results, baseline_files)]
for keys, png in ((("tabular",), "curve_tabular.png"), (("dqn_A", "dqn_B"), "curve_dqn.png")):
    out = plot_learning_curves([(labels[k], os.path.dirname(run_dirs[k])) for k in keys], baselines,
                               out=os.path.join(NOTEBOOK_RUNS, "figures", png), ylabel="mean score (greedy evaluation)")
    display(Image(filename=str(out)))

final_results = [EvalResult.load(os.path.join(d, FINAL_EVAL_FILE)) for d in run_dirs.values()]
for r, k in zip(final_results, run_dirs):
    r.policy = labels[k]
display(Markdown(markdown_table(final_results + baseline_results)))
""")

    nb.md("#### 5.7 One greedy episode of the trained DQN")
    nb.code("""
# The DQN trained on observation A plays the first evaluation board greedily; the board every ten steps and at the end (H head, o body, * food)
agent_A = DQNAgent.load(os.path.join(run_dirs["dqn_A"], MODEL_FILE), device="cpu")
board = SnakeEnv(load_config(CONFIGS["dqn_A"][0]).env, render_mode="ansi")
obs, info = board.reset(seed=DEFAULT_EVAL_SEED)
frames, done = [], False
while not done:
    if board.steps % 10 == 0:
        frames.append(f"step {board.steps}, score {board.score}\\n{board.render()}")
    obs, reward, terminated, truncated, info = board.step(agent_A.act(obs))
    done = terminated or truncated
frames.append(f"step {board.steps}, score {board.score}, end: {info['cause']}\\n{board.render()}")
print("\\n\\n".join(frames[:4] + frames[-1:]))
""")

    # ---------------------------------------------------------------- Sections 6-11: the report
    nb.md("### 6. Training Logs of the Full Study")

    nb.code("""
# Training logs: the local runs/ directory, or runs_logs.zip (scripts/export_logs.py) from LOGS_URL or the Colab upload dialog
LOGS_URL = ""               # link to runs_logs.zip; empty: upload it on Colab, or use the local runs/
LOGS_ZIP = "runs_logs.zip"


def experiment_dirs():
    return sorted(d for d in glob.glob("runs/*/") if glob.glob(os.path.join(d, "*", "final_eval.json")))


if not experiment_dirs():
    if LOGS_URL:
        urllib.request.urlretrieve(LOGS_URL, LOGS_ZIP)
    elif IN_COLAB and not os.path.exists(LOGS_ZIP):
        from google.colab import files

        LOGS_ZIP = next(iter(files.upload()))
    elif not os.path.exists(LOGS_ZIP):
        raise FileNotFoundError("no runs/ and no runs_logs.zip: set LOGS_URL or copy the zip next to this notebook")
    with zipfile.ZipFile(LOGS_ZIP) as zf:
        zf.extractall(".")  # entries are runs/<experiment>/<seed>/<file>

print(len(experiment_dirs()), "experiment directories with finished runs under runs/")
""")

    nb.md("""
### 7. Experiment Matrix

The 64 configurations of the study are defined in `snake_rl/analysis/matrix.py`; the status table lists the finished
seeds of every row and checks each run's recorded `config.yaml` against the row that launched it.
""")

    nb.code("""
# Finished seeds per configuration, and every run's config.yaml checked against the matrix
!{sys.executable} -m snake_rl.analysis.matrix --status
""")

    nb.md("### 8. Building the Report Outputs")

    nb.md("#### 8.1 Generated tables and figures")
    nb.code("""
# Every generated table and figure (runs/report/); items whose experiments are missing are skipped and listed in runs/report/index.md
!{sys.executable} -m snake_rl.analysis.report --runs-dir runs --out-dir runs/report
""")

    nb.md("#### 8.2 Composite figures")
    nb.code("""
# Figures 2, 9, 11 and 12 of the report: score distributions and death causes from the run directories, heat-maps from the saved behaviour analyses
sys.path.insert(0, "scripts")
from report_extra_figures import PANELS

needed = sorted({name for _, _, panels in PANELS for _, name in panels})
missing = [name for name in needed if not os.path.isdir(os.path.join("runs/behaviour", name))]
if missing:
    print("behaviour analyses missing under runs/behaviour:", ", ".join(missing))
    print("export the logs with --include-behaviour, or run snake_rl.analysis.behaviour for these players first")
else:
    !{sys.executable} scripts/report_extra_figures.py --out-dir runs/report/figures
""")

    nb.md("#### 8.3 The statistics of the report's tables")
    nb.code("""
# Tables 3 to 9 and C1 of the report from the run directories; a table is skipped when its experiments are missing
tables, skipped = rt.build("runs")
for name in rt.TABLES:
    print(f"{name}: {'computed' if name in tables else 'skipped (' + skipped[name] + ')'}")
""")

    nb.md("#### 8.4 The interface screenshot")
    nb.code("""
# Figure 1 is a screenshot of the GUI (scripts/gui_screenshots.py); rendered here when the two runs it plays are present with their weights
FIGURE1 = "runs/figures/partD_gui_agent.png"
if os.path.exists(FIGURE1):
    print("present:", FIGURE1)
elif all(os.path.exists(p) for p in ("runs/dqn_A_W10/0/model.pt", "runs/tabular_q_W10/0/q_table.npz")):
    !{sys.executable} scripts/gui_screenshots.py --out-dir runs/figures
else:
    print("not rendered: the screenshot needs the weights of runs/dqn_A_W10/0 and runs/tabular_q_W10/0")
""")

    nb.md("""
### 9. The Report, Table by Table and Figure by Figure

Each item below is one table or figure of the report, with the report's caption. Tables 1 and 2 are text (the
training settings and the summary of the answers) and are shown as the report source has them; every other table is
computed from the logs in Section 8.3, and the figures are the files that Section 8 wrote.
""")

    nb.code("""
# Captions, figure files and source tables of the report, and the two display helpers
captions = rt.report_captions(REPORT_MD)
figure_files = rt.report_figures(REPORT_MD)
source_tables = rt.report_tables(REPORT_MD)


def show_table(name, markdown=None):
    # a computed table (markdown) or, without one, the table of the report source; with the report's caption
    if markdown is None:
        if name not in tables and name in skipped:
            print(f"{name}: not computed ({skipped[name]})")
            return
        rows = source_tables[name]
        markdown = rt.markdown_table(rows[0], rows[1:])
    display(Markdown(f"**{name}.** {captions[name]}\\n\\n{markdown}"))


def show_figure(name, files=None):
    # the file(s) of a figure of the report with the report's caption
    files = figure_files[name] if files is None else files
    for f in files:
        if os.path.exists(f):
            display(Image(filename=f))
        else:
            print(f"{name}: missing {f}")
    display(Markdown(f"**{name}.** {captions[name]}"))
""")

    items = (
        ("9.1 The interface (Section 3.6): Figure 1", 'show_figure("Figure 1")'),
        ("9.2 Training settings (Section 4): Table 1", 'show_table("Table 1")'),
        ("9.3 The questions and the answers (Section 6): Table 2", 'show_table("Table 2")'),
        ("9.4 The scale and the tabular reference (Section 6.1): Table 3 and Figure 2",
         'show_table("Table 3", tables.get("Table 3"))\nshow_figure("Figure 2")'),
        ("9.5 State representation (Section 6.2): Table 4 and Figure 3",
         'show_table("Table 4", tables.get("Table 4"))\nshow_figure("Figure 3")'),
        ("9.6 Discount factor (Section 6.3): Table 5 and Figure 4",
         'show_table("Table 5", tables.get("Table 5"))\nshow_figure("Figure 4")'),
        ("9.7 The stabilisers (Section 6.4): Table 6, Figures 5 and 6",
         'show_table("Table 6", tables.get("Table 6"))\nshow_figure("Figure 5")\nshow_figure("Figure 6")'),
        ("9.8 Double DQN, Dueling and prioritised replay (Section 6.5): Table 7 and Figure 7",
         'show_table("Table 7", tables.get("Table 7"))\nshow_figure("Figure 7")'),
        ("9.9 Reward design (Section 6.6): Table 8, Figures 8 and 9",
         'show_table("Table 8", tables.get("Table 8"))\nshow_figure("Figure 8")\nshow_figure("Figure 9")'),
        ("9.10 Reward hacking, the probe (Section 6.7): Figure 10", 'show_figure("Figure 10")'),
        ("9.11 Behaviour (Section 6.8): Table 9, Figures 11 to 13",
         'show_table("Table 9", tables.get("Table 9"))\nshow_figure("Figure 11")\nshow_figure("Figure 12")\nshow_figure("Figure 13")'),
        ("9.12 Every configuration (Appendix B): Tables B1 and B2",
         '# The columns of the report from the generated summary tables of both boards\n'
         'for name, W in (("Table B1", 8), ("Table B2", 10)):\n'
         '    show_table(name, rt.appendix_table(f"runs/report/tables/summary_W{W}.md"))'),
        ("9.13 The tabular sweep (Appendix C): Table C1", 'show_table("Table C1", tables.get("Table C1"))'),
    )
    for heading, cell in items:
        nb.md(f"#### {heading}")
        nb.code(cell)

    nb.md("""
### 10. Check Against the Report

Every computed table is compared with the table in the report source cell by cell, and every figure file of the
report must exist.
""")

    nb.code("""
# Computed tables against the report source, and the figure files of the report
problems = 0
for name in rt.TABLES:
    if name not in tables:
        print(f"{name}: not computed ({skipped[name]})")
        problems += 1
        continue
    diffs = rt.compare_with_report(name, tables[name], REPORT_MD)
    problems += len(diffs)
    print(f"{name}: {'identical to the report' if not diffs else str(len(diffs)) + ' differences'}")
    for d in diffs:
        print("  " + d)
for name, files in figure_files.items():
    missing_files = [f for f in files if not os.path.exists(f)]
    if missing_files:
        problems += 1
        print(f"{name}: missing {', '.join(missing_files)}")
    else:
        print(f"{name}: {', '.join(files)}")
print("RESULT:", "every table and figure of the report is reproduced" if problems == 0 else f"{problems} problem(s)")
""")

    nb.md("""
### 11. Behaviour Analysis of One Run

`snake_rl.analysis.behaviour` plays 100 evaluation episodes with a trained network and writes the visit heat-map, the
loop / shaping statistics per episode and GUI-replayable recordings of the longest loops
(`python -m snake_rl.gui.app --replay <file>`). The saved analyses behind Table 9 and Figures 11 to 13 were made this
way. The run analysed is the gamma = 0.99 DQN of the report when its weights are present (the logs zip carries none),
otherwise the DQN trained on observation B in Section 5.
""")

    nb.code("""
# Behaviour analysis of one run: 100 greedy episodes, heat-map, loop statistics and recordings of the longest loops
candidates = ("runs/dqn_B8_W8_gamma099/0", run_dirs["dqn_B"])
RUN_DIR = next(d for d in candidates if os.path.exists(os.path.join(d, MODEL_FILE)))
OUT_DIR = os.path.join(NOTEBOOK_RUNS, "behaviour", os.path.basename(os.path.dirname(RUN_DIR)))

!{sys.executable} -m snake_rl.analysis.behaviour --player {RUN_DIR} --checkpoint final --device {DEVICE} --out-dir {OUT_DIR}
display(Image(filename=os.path.join(OUT_DIR, "heatmap.png")))
display(Image(filename=os.path.join(OUT_DIR, "last_steps.png")))
with open(os.path.join(OUT_DIR, "summary.json"), encoding="utf-8") as f:
    print(json.dumps(json.load(f), indent=2)[:2000])
""")

    # the notebook's own cells must not rebind a module-level name of the inlined code
    module_sources = set(module_code.values())
    for cell in nb.cells:
        if cell["cell_type"] != "code" or "".join(cell["source"]) in module_sources:
            continue
        clash = _notebook_names("".join(cell["source"])) & set(defined)
        if clash:
            raise SystemExit(f"notebook cell rebinds module names {sorted(clash)}: {''.join(cell['source'])[:60]!r}")
    return nb.to_dict()


def render(nb: dict) -> str:
    return json.dumps(nb, indent=1, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Write notebooks/results.ipynb from the package sources.")
    p.add_argument("--out", default=str(OUT))
    p.add_argument("--check", action="store_true", help="compare with the file instead of writing it")
    args = p.parse_args(argv)
    text = render(build())
    out = Path(args.out)
    if args.check:
        if not out.is_file() or out.read_text(encoding="utf-8") != text:
            print(f"{out} differs from the sources: run python scripts/build_notebook.py")
            return 1
        print(f"{out} is up to date")
        return 0
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8", newline="\n")
    nb = json.loads(text)
    print(f"written {out}: {len(nb['cells'])} cells ({sum(c['cell_type'] == 'code' for c in nb['cells'])} code)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
