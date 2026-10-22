"""The experiment matrix of the report as data (M10).

Every named configuration of the study - the experiment directory name under
``runs/``, the config file it was trained from and the ``--set`` overrides -
is listed once, here, so that a table or figure can name its rows by
experiment and the provenance of every number is in the code, not in a shell
history. ``G1_t31`` (one update per vectorised step, target period 31
updates) is a named configuration of the ``budget`` group.

What the module does with the matrix:

* :func:`status` scans a runs directory and says, per experiment, which seeds
  are finished, aborted or unfinished, and whether every recorded
  ``config.yaml`` is the configuration the matrix says (``mismatches``).
  ``agent.keep_checkpoints`` is not compared (``aggregate.IGNORED_SETTINGS``):
  whether weight files were kept decides nothing about the training.
* :func:`commands` writes the ``scripts/run_matrix.py`` command lines for the
  experiments that lack seeds, with ``agent.keep_checkpoints=false`` for DQN
  rows (a run with checkpoints is 24-47 MB; ``run_matrix`` skips the seeds
  that are done, so a command always lists all five seeds).

CLI::

    python -m snake_rl.analysis.matrix --status   [--runs-dir runs] [--W 8|10] [--group G]
    python -m snake_rl.analysis.matrix --commands [--all] [--keep-checkpoints] [--jobs N] [--python P] ...

``--status`` prints a Markdown table and exits 1 if any run mismatches its
row; ``--commands`` prints one launch line per experiment with missing seeds
(``--all``: every selected experiment).
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import yaml

from snake_rl.analysis.aggregate import ABORTED_FILE, CONFIG_FILE, FINAL_FILE, IGNORED_SETTINGS, _flat_settings
from snake_rl.config import ExperimentConfig, config_to_dict, load_config
from snake_rl.utils import REPO_ROOT

#: Experiment groups, in report order.
GROUPS = ("tabular", "main", "obs", "gamma", "components", "budget", "capacity", "variants", "reward", "probe")
#: Seeds of every configuration (project rule: five, never fewer).
SEEDS = (0, 1, 2, 3, 4)
#: The override that stops a DQN run from writing a weight file per evaluation.
NO_CHECKPOINTS = "agent.keep_checkpoints=false"

_BOARD = re.compile(r"_W(\d+)(?:_|$)")


@dataclass(frozen=True)
class Experiment:
    """One row of the matrix: a configuration trained with the five seeds."""

    name: str
    config: str
    overrides: tuple[str, ...]
    group: str
    label: str

    @property
    def W(self) -> int:
        """The board size, the integer after ``_W`` in the name."""
        match = _BOARD.search(self.name)
        if match is None:
            raise ValueError(f"{self.name!r} has no _W<size> component")
        return int(match.group(1))

    @property
    def kind(self) -> str:
        """``"tabular"`` for a tabular Q-learning row, else ``"dqn"``."""
        return "tabular" if Path(self.config).name.startswith("tabular_q") else "dqn"


@dataclass(frozen=True)
class Baseline:
    """A non-learning player evaluated once per board (``snake_rl.evaluate``)."""

    name: str
    W: int
    label: str

    @property
    def path(self) -> str:
        return f"runs/baselines/{self.name}_W{self.W}_relative.json"


def _row(name: str, config: str, overrides: str, group: str, label: str) -> Experiment:
    return Experiment(name, config, tuple(overrides.split()), group, label)


_A8, _B8 = "configs/dqn_A_8x8.yaml", "configs/dqn_B_8x8.yaml"
_A10, _B10 = "configs/dqn_A_10x10.yaml", "configs/dqn_B_10x10.yaml"
_T8, _T10 = "configs/tabular_q_8x8.yaml", "configs/tabular_q_10x10.yaml"

MATRIX: tuple[Experiment, ...] = (
    # ---------------------------------------------------------------- 8x8
    _row("tabular_q_W8", _T8, "", "tabular", "tabular Q"),
    _row("tabular_q_W8_gamma0", _T8, "agent.gamma=0.0", "tabular", "tabular Q, gamma 0"),
    _row("dqn_A_W8", _A8, "", "main", "A (MLP)"),
    _row("dqn_B8_W8", _B8, "", "main", "B8 (CNN)"),
    _row("dqn_B3_W8", _B8, "env.obs.age_plane=false env.obs.heading_planes=false", "obs", "B3 (CNN)"),
    _row("dqn_B7_W8", _B8, "env.obs.age_plane=false", "obs", "B7 (CNN)"),
    _row("dqn_A_W8_gamma099", _A8, "agent.gamma=0.99", "gamma", "A, gamma 0.99"),
    _row("dqn_B8_W8_gamma099", _B8, "agent.gamma=0.99", "gamma", "B8, gamma 0.99"),
    _row("dqn_A_W8_noreplay", _A8, "agent.replay=false", "components", "A, no replay"),
    _row("dqn_A_W8_noreplay_t31", _A8, "agent.replay=false agent.target_update_every=31", "components",
         "A, no replay, target 31"),
    _row("dqn_A_W8_notarget", _A8, "agent.target_update_every=1", "components", "A, no target"),
    _row("dqn_A_W8_noreplay_notarget", _A8, "agent.replay=false agent.target_update_every=1", "components",
         "A, no replay, no target"),
    _row("dqn_B8_W8_noreplay", _B8, "agent.replay=false", "components", "B8, no replay"),
    _row("dqn_B8_W8_noreplay_t31", _B8, "agent.replay=false agent.target_update_every=31", "components",
         "B8, no replay, target 31"),
    _row("dqn_B8_W8_notarget", _B8, "agent.target_update_every=1", "components", "B8, no target"),
    _row("dqn_B8_W8_noreplay_notarget", _B8, "agent.replay=false agent.target_update_every=1", "components",
         "B8, no replay, no target"),
    _row("dqn_A_W8_G1", _A8, "agent.grad_steps_per_vec_step=1", "budget", "A, G=1"),
    _row("dqn_A_W8_G4", _A8, "agent.grad_steps_per_vec_step=4", "budget", "A, G=4"),
    _row("dqn_A_W8_G8_b1024", _A8, "agent.grad_steps_per_vec_step=8 agent.batch_size=1024", "budget",
         "A, G=8, batch 1024"),
    _row("dqn_B8_W8_G1", _B8, "agent.grad_steps_per_vec_step=1", "budget", "B8, G=1"),
    _row("dqn_B8_W8_G1_t31", _B8, "agent.grad_steps_per_vec_step=1 agent.target_update_every=31", "budget",
         "B8, G=1, target 31"),
    _row("dqn_A_W8_h16", _A8, "agent.hidden=[16]", "capacity", "A, hidden 16"),
    _row("dqn_A_W8_linear", _A8, "agent.hidden=[]", "capacity", "A, linear"),
    _row("dqn_A_W8_double", _A8, "agent.double=true", "variants", "A, Double"),
    _row("dqn_A_W8_dueling", _A8, "agent.dueling=true", "variants", "A, Dueling"),
    _row("dqn_A_W8_per", _A8, "agent.per=true", "variants", "A, PER"),
    _row("dqn_A_W8_gamma099_double", _A8, "agent.gamma=0.99 agent.double=true", "variants", "A, gamma 0.99, Double"),
    _row("dqn_B8_W8_double", _B8, "agent.double=true", "variants", "B8, Double"),
    _row("dqn_B8_W8_dueling", _B8, "agent.dueling=true", "variants", "B8, Dueling"),
    _row("dqn_B8_W8_per", _B8, "agent.per=true", "variants", "B8, PER"),
    _row("dqn_B8_W8_gamma099_double", _B8, "agent.gamma=0.99 agent.double=true", "variants",
         "B8, gamma 0.99, Double"),
    _row("dqn_A_W8_R1", _A8, "env.reward.variant=R1", "reward", "A, R1"),
    _row("dqn_A_W8_R2", _A8, "env.reward.variant=R2 env.reward.gamma=0.9", "reward", "A, R2"),
    _row("dqn_A_W8_R3", _A8, "env.reward.variant=R3", "reward", "A, R3"),
    _row("dqn_A_W8_gamma099_R1", _A8, "agent.gamma=0.99 env.reward.variant=R1", "reward", "A, gamma 0.99, R1"),
    _row("dqn_A_W8_gamma099_R2", _A8, "agent.gamma=0.99 env.reward.variant=R2 env.reward.gamma=0.99", "reward",
         "A, gamma 0.99, R2"),
    _row("dqn_A_W8_gamma099_R3", _A8, "agent.gamma=0.99 env.reward.variant=R3", "reward", "A, gamma 0.99, R3"),
    _row("dqn_A_W8_gamma099_R3sym", _A8, "agent.gamma=0.99 env.reward.variant=R3 env.reward.farther=-0.01",
         "reward", "A, gamma 0.99, R3 symmetric"),
    _row("dqn_A_W8_gamma099_R3_H1000", _A8, "agent.gamma=0.99 env.reward.variant=R3 env.starve_limit=1000",
         "reward", "A, gamma 0.99, R3, H=1000"),
    _row("dqn_B8_W8_R1", _B8, "env.reward.variant=R1", "reward", "B8, R1"),
    _row("dqn_B8_W8_R2", _B8, "env.reward.variant=R2 env.reward.gamma=0.9", "reward", "B8, R2"),
    _row("dqn_B8_W8_R3", _B8, "env.reward.variant=R3", "reward", "B8, R3"),
    _row("dqn_B8_W8_gamma099_R1", _B8, "agent.gamma=0.99 env.reward.variant=R1", "reward", "B8, gamma 0.99, R1"),
    _row("dqn_B8_W8_gamma099_R2", _B8, "agent.gamma=0.99 env.reward.variant=R2 env.reward.gamma=0.99", "reward",
         "B8, gamma 0.99, R2"),
    _row("dqn_B8_W8_gamma099_R3", _B8, "agent.gamma=0.99 env.reward.variant=R3", "reward", "B8, gamma 0.99, R3"),
    _row("dqn_B8_W8_gamma099_R3sym", _B8, "agent.gamma=0.99 env.reward.variant=R3 env.reward.farther=-0.01",
         "reward", "B8, gamma 0.99, R3 symmetric"),
    _row("dqn_B8_W8_gamma099_R3_H1000", _B8, "agent.gamma=0.99 env.reward.variant=R3 env.starve_limit=1000",
         "reward", "B8, gamma 0.99, R3, H=1000"),
    _row("dqn_B8_W8_gamma099_R3_c01", _B8, "agent.gamma=0.99 env.reward.variant=R3 env.reward.closer=0.1",
         "probe", "B8, gamma 0.99, R3, closer 0.1"),
    # -------------------------------------------------------------- 10x10
    _row("tabular_q_W10", _T10, "", "tabular", "tabular Q"),
    _row("tabular_q_W10_gamma0", _T10, "agent.gamma=0.0", "tabular", "tabular Q, gamma 0"),
    _row("dqn_A_W10", _A10, "", "main", "A (MLP)"),
    _row("dqn_B8_W10", _B10, "", "main", "B8 (CNN)"),
    _row("dqn_B3_W10", _B10, "env.obs.age_plane=false env.obs.heading_planes=false", "obs", "B3 (CNN)"),
    _row("dqn_B7_W10", _B10, "env.obs.age_plane=false", "obs", "B7 (CNN)"),
    _row("dqn_B8_W10_gamma099", _B10, "agent.gamma=0.99", "gamma", "B8, gamma 0.99"),
    _row("dqn_A_W10_noreplay_t31", _A10, "agent.replay=false agent.target_update_every=31", "components",
         "A, no replay, target 31"),
    _row("dqn_A_W10_notarget", _A10, "agent.target_update_every=1", "components", "A, no target"),
    _row("dqn_B8_W10_noreplay", _B10, "agent.replay=false", "components", "B8, no replay"),
    _row("dqn_B8_W10_noreplay_t31", _B10, "agent.replay=false agent.target_update_every=31", "components",
         "B8, no replay, target 31"),
    _row("dqn_B8_W10_notarget", _B10, "agent.target_update_every=1", "components", "B8, no target"),
    _row("dqn_B8_W10_G1_t31", _B10, "agent.grad_steps_per_vec_step=1 agent.target_update_every=31", "budget",
         "B8, G=1, target 31"),
    _row("dqn_B8_W10_gamma099_R1", _B10, "agent.gamma=0.99 env.reward.variant=R1", "reward", "B8, gamma 0.99, R1"),
    _row("dqn_B8_W10_gamma099_R2", _B10, "agent.gamma=0.99 env.reward.variant=R2 env.reward.gamma=0.99", "reward",
         "B8, gamma 0.99, R2"),
    _row("dqn_B8_W10_gamma099_R3", _B10, "agent.gamma=0.99 env.reward.variant=R3", "reward", "B8, gamma 0.99, R3"),
)

BASELINES: tuple[Baseline, ...] = tuple(
    Baseline(name, W, label)
    for W in (8, 10)
    for name, label in (("random", "random"), ("bfs", "BFS greedy"), ("hamiltonian", "Hamiltonian"))
)

_BY_NAME = {exp.name: exp for exp in MATRIX}


# ------------------------------------------------------------------ lookups
def experiment(name: str) -> Experiment:
    """The matrix row called ``name``; ``ValueError`` if there is none."""
    try:
        return _BY_NAME[name]
    except KeyError:
        raise ValueError(f"unknown experiment {name!r}: not a row of the matrix") from None


def experiments(W: int | None = None, group: str | None = None) -> list[Experiment]:
    """The rows of the matrix, in order, restricted to a board and / or a group."""
    if group is not None and group not in GROUPS:
        raise ValueError(f"unknown group {group!r}; groups are {', '.join(GROUPS)}")
    return [e for e in MATRIX if (W is None or e.W == W) and (group is None or e.group == group)]


def baselines(W: int | None = None) -> list[Baseline]:
    """The baseline players, restricted to a board."""
    return [b for b in BASELINES if W is None or b.W == W]


def expected_config(exp: Experiment, repo_root: str | Path = REPO_ROOT) -> ExperimentConfig:
    """The configuration the row stands for: its config file with its overrides."""
    return load_config(Path(repo_root) / exp.config, [*exp.overrides, f"exp={exp.name}"])


# ------------------------------------------------------------------- status
@dataclass(frozen=True)
class ExperimentStatus:
    """What a runs directory holds for one row of the matrix."""

    experiment: Experiment
    finished: tuple[int, ...]
    aborted: tuple[str, ...]
    unfinished: tuple[str, ...]
    mismatches: tuple[str, ...]

    @property
    def n_finished(self) -> int:
        return len(self.finished)


def _settings(cfg: ExperimentConfig) -> dict[str, Any]:
    """Flat ``dotted.key -> value`` of a configuration without ``exp``, the
    ``run`` block and the ignored settings."""
    flat = _flat_settings(config_to_dict(cfg))
    return {
        key: value
        for key, value in flat.items()
        if key != "exp" and not key.startswith("run.") and key not in IGNORED_SETTINGS
    }


def _mismatch(run_dir: Path, expected: dict[str, Any]) -> str | None:
    """``None`` if the run's recorded configuration is ``expected``, else a note."""
    try:
        recorded = _settings(load_config(run_dir / CONFIG_FILE))
    except (OSError, ValueError) as err:
        return f"{run_dir.name}: {CONFIG_FILE} does not load ({err})"
    differences = [
        f"{key} {recorded.get(key, 'absent')!r} recorded, {expected.get(key, 'absent')!r} expected"
        for key in sorted(recorded.keys() | expected.keys())
        if recorded.get(key) != expected.get(key)
    ]
    return f"{run_dir.name}: {'; '.join(differences)}" if differences else None


def _recorded_seed(run_dir: Path) -> int | None:
    raw = yaml.safe_load((run_dir / CONFIG_FILE).read_text(encoding="utf-8"))
    run = raw.get("run") if isinstance(raw, dict) else None
    seed = run.get("seed") if isinstance(run, dict) else None
    return seed if isinstance(seed, int) and not isinstance(seed, bool) else None


def status(
    runs_dir: str | Path = "runs",
    experiments: Sequence[Experiment] | None = None,
    repo_root: str | Path = REPO_ROOT,
) -> list[ExperimentStatus]:
    """Per row (default: the whole matrix): finished seeds, aborted and
    unfinished run directories, and the runs whose recorded configuration is
    not the row's (``aggregate.scan_experiment`` has the file rules; here
    nothing is raised, a missing experiment directory is simply empty)."""
    rows = list(MATRIX if experiments is None else experiments)
    runs_dir = Path(runs_dir)
    out = []
    for exp in rows:
        exp_dir = runs_dir / exp.name
        finished: list[int] = []
        aborted: list[str] = []
        unfinished: list[str] = []
        mismatches: list[str] = []
        expected: dict[str, Any] | None = None
        for sub in sorted(p for p in exp_dir.iterdir() if p.is_dir()) if exp_dir.is_dir() else []:
            has_config = (sub / CONFIG_FILE).is_file()
            has_final = (sub / FINAL_FILE).is_file()
            if has_config and has_final:
                seed = _recorded_seed(sub)
                if seed is None:
                    mismatches.append(f"{sub.name}: no run.seed in {CONFIG_FILE}")
                else:
                    finished.append(seed)
            elif (sub / ABORTED_FILE).is_file():
                aborted.append(sub.name)
            elif has_config or has_final:
                unfinished.append(sub.name)
            if has_config:
                if expected is None:
                    expected = _settings(expected_config(exp, repo_root))
                note = _mismatch(sub, expected)
                if note is not None:
                    mismatches.append(note)
        out.append(
            ExperimentStatus(exp, tuple(sorted(finished)), tuple(aborted), tuple(unfinished), tuple(mismatches))
        )
    return out


# ----------------------------------------------------------------- commands
def commands(
    statuses: Sequence[ExperimentStatus],
    seeds: Sequence[int] = SEEDS,
    jobs: int = 5,
    keep_checkpoints: bool = False,
    python: str = "python",
    missing_only: bool = True,
) -> list[str]:
    """``scripts/run_matrix.py`` command lines, one per row that lacks seeds
    (``missing_only=False``: every row). DQN rows get
    ``agent.keep_checkpoints=false`` unless ``keep_checkpoints``."""
    lines = []
    for st in statuses:
        if missing_only and st.n_finished >= len(seeds):
            continue
        exp = st.experiment
        items = list(exp.overrides)
        if exp.kind == "dqn" and not keep_checkpoints:
            items.append(NO_CHECKPOINTS)
        line = (
            f"{python} scripts/run_matrix.py --config {exp.config} --seeds {' '.join(str(s) for s in seeds)} "
            f"--jobs {jobs} --exp-prefix {exp.name}"
        )
        if items:
            line += " --set " + " ".join(items)
        lines.append(line)
    return lines


def status_table(statuses: Sequence[ExperimentStatus], n_seeds: int = len(SEEDS)) -> str:
    """Markdown table of :func:`status` results."""
    lines = [
        "| experiment | board | group | config | overrides | finished | aborted | unfinished | configuration |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for st in statuses:
        exp = st.experiment
        overrides = " ".join(exp.overrides) if exp.overrides else "-"
        configuration = f"MISMATCH ({len(st.mismatches)} run(s))" if st.mismatches else "ok"
        lines.append(
            f"| {exp.name} | {exp.W}x{exp.W} | {exp.group} | {exp.config} | {overrides} | "
            f"{st.n_finished}/{n_seeds} | {len(st.aborted)} | {len(st.unfinished)} | {configuration} |"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------- CLI
def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="The experiment matrix: status against runs/, and launch commands.")
    p.add_argument("--runs-dir", default="runs", metavar="DIR")
    p.add_argument("--W", type=int, default=None, choices=(8, 10), help="only this board")
    p.add_argument("--group", default=None, choices=GROUPS, help="only this group")
    what = p.add_mutually_exclusive_group(required=True)
    what.add_argument("--status", action="store_true", help="print the status table; exit 1 on any mismatch")
    what.add_argument("--commands", action="store_true", help="print the launch commands of the missing rows")
    p.add_argument("--all", action="store_true", help="with --commands: every selected row, not only the missing")
    p.add_argument("--keep-checkpoints", action="store_true",
                   help=f"with --commands: do not append {NO_CHECKPOINTS}")
    p.add_argument("--jobs", type=int, default=None, metavar="N", help="with --commands: processes at a time (default 5)")
    p.add_argument("--python", default=None, metavar="PATH", help='with --commands: the interpreter (default "python")')
    args = p.parse_args(argv)
    if not args.commands and (args.all or args.keep_checkpoints or args.jobs is not None or args.python is not None):
        p.error("--all, --keep-checkpoints, --jobs and --python only apply to --commands")
    if args.jobs is not None and args.jobs < 1:
        p.error("--jobs must be at least 1")

    statuses = status(args.runs_dir, experiments(args.W, args.group))
    if args.status:
        print(status_table(statuses))
        mismatched = [st for st in statuses if st.mismatches]
        for st in mismatched:
            for note in st.mismatches:
                print(f"mismatch in {st.experiment.name}: {note}")
        return 1 if mismatched else 0
    for line in commands(
        statuses,
        jobs=args.jobs if args.jobs is not None else 5,
        keep_checkpoints=args.keep_checkpoints,
        python=args.python if args.python is not None else "python",
        missing_only=not args.all,
    ):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
