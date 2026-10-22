"""The experiment matrix as data (``snake_rl/analysis/matrix.py``, M10 spec
section 3): the 64 named configurations, their status against a runs
directory, the launch commands for what is missing, and the CLI.

The expected table below is the matrix of the specification, row for row
(name, config file, ``--set`` overrides, group, label): 48 rows for the 8x8
board and 16 for 10x10. Run directories for the status checks are written
by hand under ``tmp_path`` from ``config_to_dict(expected_config(exp))``
plus a ``run`` block, so a recorded run matches its row exactly unless a
test edits one key.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from snake_rl.analysis import matrix
from snake_rl.analysis.aggregate import IGNORED_SETTINGS
from snake_rl.config import ExperimentConfig, config_to_dict, load_config
from snake_rl.utils import REPO_ROOT
from tests.test_aggregate import FIVE_SEEDS, make_result

#: name | config | overrides | group | label, the 8x8 rows of the specification in order.
SPEC_W8 = """
tabular_q_W8 | configs/tabular_q_8x8.yaml | | tabular | tabular Q
tabular_q_W8_gamma0 | configs/tabular_q_8x8.yaml | agent.gamma=0.0 | tabular | tabular Q, gamma 0
dqn_A_W8 | configs/dqn_A_8x8.yaml | | main | A (MLP)
dqn_B8_W8 | configs/dqn_B_8x8.yaml | | main | B8 (CNN)
dqn_B3_W8 | configs/dqn_B_8x8.yaml | env.obs.age_plane=false env.obs.heading_planes=false | obs | B3 (CNN)
dqn_B7_W8 | configs/dqn_B_8x8.yaml | env.obs.age_plane=false | obs | B7 (CNN)
dqn_A_W8_gamma099 | configs/dqn_A_8x8.yaml | agent.gamma=0.99 | gamma | A, gamma 0.99
dqn_B8_W8_gamma099 | configs/dqn_B_8x8.yaml | agent.gamma=0.99 | gamma | B8, gamma 0.99
dqn_A_W8_noreplay | configs/dqn_A_8x8.yaml | agent.replay=false | components | A, no replay
dqn_A_W8_noreplay_t31 | configs/dqn_A_8x8.yaml | agent.replay=false agent.target_update_every=31 | components | A, no replay, target 31
dqn_A_W8_notarget | configs/dqn_A_8x8.yaml | agent.target_update_every=1 | components | A, no target
dqn_A_W8_noreplay_notarget | configs/dqn_A_8x8.yaml | agent.replay=false agent.target_update_every=1 | components | A, no replay, no target
dqn_B8_W8_noreplay | configs/dqn_B_8x8.yaml | agent.replay=false | components | B8, no replay
dqn_B8_W8_noreplay_t31 | configs/dqn_B_8x8.yaml | agent.replay=false agent.target_update_every=31 | components | B8, no replay, target 31
dqn_B8_W8_notarget | configs/dqn_B_8x8.yaml | agent.target_update_every=1 | components | B8, no target
dqn_B8_W8_noreplay_notarget | configs/dqn_B_8x8.yaml | agent.replay=false agent.target_update_every=1 | components | B8, no replay, no target
dqn_A_W8_G1 | configs/dqn_A_8x8.yaml | agent.grad_steps_per_vec_step=1 | budget | A, G=1
dqn_A_W8_G4 | configs/dqn_A_8x8.yaml | agent.grad_steps_per_vec_step=4 | budget | A, G=4
dqn_A_W8_G8_b1024 | configs/dqn_A_8x8.yaml | agent.grad_steps_per_vec_step=8 agent.batch_size=1024 | budget | A, G=8, batch 1024
dqn_B8_W8_G1 | configs/dqn_B_8x8.yaml | agent.grad_steps_per_vec_step=1 | budget | B8, G=1
dqn_B8_W8_G1_t31 | configs/dqn_B_8x8.yaml | agent.grad_steps_per_vec_step=1 agent.target_update_every=31 | budget | B8, G=1, target 31
dqn_A_W8_h16 | configs/dqn_A_8x8.yaml | agent.hidden=[16] | capacity | A, hidden 16
dqn_A_W8_linear | configs/dqn_A_8x8.yaml | agent.hidden=[] | capacity | A, linear
dqn_A_W8_double | configs/dqn_A_8x8.yaml | agent.double=true | variants | A, Double
dqn_A_W8_dueling | configs/dqn_A_8x8.yaml | agent.dueling=true | variants | A, Dueling
dqn_A_W8_per | configs/dqn_A_8x8.yaml | agent.per=true | variants | A, PER
dqn_A_W8_gamma099_double | configs/dqn_A_8x8.yaml | agent.gamma=0.99 agent.double=true | variants | A, gamma 0.99, Double
dqn_B8_W8_double | configs/dqn_B_8x8.yaml | agent.double=true | variants | B8, Double
dqn_B8_W8_dueling | configs/dqn_B_8x8.yaml | agent.dueling=true | variants | B8, Dueling
dqn_B8_W8_per | configs/dqn_B_8x8.yaml | agent.per=true | variants | B8, PER
dqn_B8_W8_gamma099_double | configs/dqn_B_8x8.yaml | agent.gamma=0.99 agent.double=true | variants | B8, gamma 0.99, Double
dqn_A_W8_R1 | configs/dqn_A_8x8.yaml | env.reward.variant=R1 | reward | A, R1
dqn_A_W8_R2 | configs/dqn_A_8x8.yaml | env.reward.variant=R2 env.reward.gamma=0.9 | reward | A, R2
dqn_A_W8_R3 | configs/dqn_A_8x8.yaml | env.reward.variant=R3 | reward | A, R3
dqn_A_W8_gamma099_R1 | configs/dqn_A_8x8.yaml | agent.gamma=0.99 env.reward.variant=R1 | reward | A, gamma 0.99, R1
dqn_A_W8_gamma099_R2 | configs/dqn_A_8x8.yaml | agent.gamma=0.99 env.reward.variant=R2 env.reward.gamma=0.99 | reward | A, gamma 0.99, R2
dqn_A_W8_gamma099_R3 | configs/dqn_A_8x8.yaml | agent.gamma=0.99 env.reward.variant=R3 | reward | A, gamma 0.99, R3
dqn_A_W8_gamma099_R3sym | configs/dqn_A_8x8.yaml | agent.gamma=0.99 env.reward.variant=R3 env.reward.farther=-0.01 | reward | A, gamma 0.99, R3 symmetric
dqn_A_W8_gamma099_R3_H1000 | configs/dqn_A_8x8.yaml | agent.gamma=0.99 env.reward.variant=R3 env.starve_limit=1000 | reward | A, gamma 0.99, R3, H=1000
dqn_B8_W8_R1 | configs/dqn_B_8x8.yaml | env.reward.variant=R1 | reward | B8, R1
dqn_B8_W8_R2 | configs/dqn_B_8x8.yaml | env.reward.variant=R2 env.reward.gamma=0.9 | reward | B8, R2
dqn_B8_W8_R3 | configs/dqn_B_8x8.yaml | env.reward.variant=R3 | reward | B8, R3
dqn_B8_W8_gamma099_R1 | configs/dqn_B_8x8.yaml | agent.gamma=0.99 env.reward.variant=R1 | reward | B8, gamma 0.99, R1
dqn_B8_W8_gamma099_R2 | configs/dqn_B_8x8.yaml | agent.gamma=0.99 env.reward.variant=R2 env.reward.gamma=0.99 | reward | B8, gamma 0.99, R2
dqn_B8_W8_gamma099_R3 | configs/dqn_B_8x8.yaml | agent.gamma=0.99 env.reward.variant=R3 | reward | B8, gamma 0.99, R3
dqn_B8_W8_gamma099_R3sym | configs/dqn_B_8x8.yaml | agent.gamma=0.99 env.reward.variant=R3 env.reward.farther=-0.01 | reward | B8, gamma 0.99, R3 symmetric
dqn_B8_W8_gamma099_R3_H1000 | configs/dqn_B_8x8.yaml | agent.gamma=0.99 env.reward.variant=R3 env.starve_limit=1000 | reward | B8, gamma 0.99, R3, H=1000
dqn_B8_W8_gamma099_R3_c01 | configs/dqn_B_8x8.yaml | agent.gamma=0.99 env.reward.variant=R3 env.reward.closer=0.1 | probe | B8, gamma 0.99, R3, closer 0.1
"""

#: The 10x10 rows.
SPEC_W10 = """
tabular_q_W10 | configs/tabular_q_10x10.yaml | | tabular | tabular Q
tabular_q_W10_gamma0 | configs/tabular_q_10x10.yaml | agent.gamma=0.0 | tabular | tabular Q, gamma 0
dqn_A_W10 | configs/dqn_A_10x10.yaml | | main | A (MLP)
dqn_B8_W10 | configs/dqn_B_10x10.yaml | | main | B8 (CNN)
dqn_B3_W10 | configs/dqn_B_10x10.yaml | env.obs.age_plane=false env.obs.heading_planes=false | obs | B3 (CNN)
dqn_B7_W10 | configs/dqn_B_10x10.yaml | env.obs.age_plane=false | obs | B7 (CNN)
dqn_B8_W10_gamma099 | configs/dqn_B_10x10.yaml | agent.gamma=0.99 | gamma | B8, gamma 0.99
dqn_A_W10_noreplay_t31 | configs/dqn_A_10x10.yaml | agent.replay=false agent.target_update_every=31 | components | A, no replay, target 31
dqn_A_W10_notarget | configs/dqn_A_10x10.yaml | agent.target_update_every=1 | components | A, no target
dqn_B8_W10_noreplay | configs/dqn_B_10x10.yaml | agent.replay=false | components | B8, no replay
dqn_B8_W10_noreplay_t31 | configs/dqn_B_10x10.yaml | agent.replay=false agent.target_update_every=31 | components | B8, no replay, target 31
dqn_B8_W10_notarget | configs/dqn_B_10x10.yaml | agent.target_update_every=1 | components | B8, no target
dqn_B8_W10_G1_t31 | configs/dqn_B_10x10.yaml | agent.grad_steps_per_vec_step=1 agent.target_update_every=31 | budget | B8, G=1, target 31
dqn_B8_W10_gamma099_R1 | configs/dqn_B_10x10.yaml | agent.gamma=0.99 env.reward.variant=R1 | reward | B8, gamma 0.99, R1
dqn_B8_W10_gamma099_R2 | configs/dqn_B_10x10.yaml | agent.gamma=0.99 env.reward.variant=R2 env.reward.gamma=0.99 | reward | B8, gamma 0.99, R2
dqn_B8_W10_gamma099_R3 | configs/dqn_B_10x10.yaml | agent.gamma=0.99 env.reward.variant=R3 | reward | B8, gamma 0.99, R3
"""

GROUPS = ("tabular", "main", "obs", "gamma", "components", "budget", "capacity", "variants", "reward", "probe")
TABULAR = {"tabular_q_W8", "tabular_q_W8_gamma0", "tabular_q_W10", "tabular_q_W10_gamma0"}
SEEDS = (0, 1, 2, 3, 4)
STATUS_HEAD = "| experiment | board | group | config | overrides | finished | aborted | unfinished | configuration |"


def parse(spec: str) -> list[tuple[str, str, tuple[str, ...], str, str]]:
    rows = []
    for line in spec.strip().splitlines():
        name, config, overrides, group, label = (cell.strip() for cell in line.split("|"))
        rows.append((name, config, tuple(overrides.split()), group, label))
    return rows


EXPECTED_W8, EXPECTED_W10 = parse(SPEC_W8), parse(SPEC_W10)
EXPECTED = EXPECTED_W8 + EXPECTED_W10


def as_tuple(exp) -> tuple[str, str, tuple[str, ...], str, str]:
    return (exp.name, exp.config, tuple(exp.overrides), exp.group, exp.label)


# ------------------------------------------------------------------- builders
def write_matrix_run(runs_dir: Path, name: str, seed: int, *, finished: bool = True, aborted: bool = False,
                     run_block: bool = True, edit: dict | None = None, dir_name: str | None = None) -> Path:
    """One run directory of matrix experiment ``name`` as ``snake_rl.train`` records it."""
    config = config_to_dict(matrix.expected_config(matrix.experiment(name)))
    for dotted, value in (edit or {}).items():
        node = config
        *parents, last = dotted.split(".")
        for part in parents:
            node = node[part]
        node[last] = value
    if run_block:
        config["run"] = {"seed": seed, "env_seed": 100_000 * (seed + 1), "git": "abc1234"}
    run_dir = runs_dir / name / (dir_name if dir_name is not None else str(seed))
    run_dir.mkdir(parents=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "metrics.csv").write_text("env_steps,eval_mean_score\n0,0.0\n", encoding="utf-8")
    if finished:
        make_result(FIVE_SEEDS[1], policy="dqn").save(run_dir / "final_eval.json")
    if aborted:
        (run_dir / "aborted.json").write_text(json.dumps({"error": "non-finite DQN update"}), encoding="utf-8")
    return run_dir


def status_of(runs_dir: Path, name: str) -> "matrix.ExperimentStatus":
    (found,) = matrix.status(runs_dir, experiments=[matrix.experiment(name)])
    return found


def st(name: str, finished=(), aborted=(), unfinished=(), mismatches=()) -> "matrix.ExperimentStatus":
    """A status without a runs directory (``ExperimentStatus`` is plain data)."""
    return matrix.ExperimentStatus(experiment=matrix.experiment(name), finished=tuple(finished),
                                   aborted=tuple(aborted), unfinished=tuple(unfinished), mismatches=tuple(mismatches))


def cli(argv: list[str], capsys) -> tuple[int, str, str]:
    try:
        code = matrix.main(argv)
    except SystemExit as exit_:
        code = exit_.code
    captured = capsys.readouterr()
    return code, captured.out, captured.err


# ------------------------------------------------------------------ the rows
def test_the_matrix_is_the_table_of_the_specification():
    assert isinstance(matrix.MATRIX, tuple)
    assert [as_tuple(e) for e in matrix.MATRIX] == EXPECTED
    assert len(matrix.MATRIX) == 64
    assert all(isinstance(e, matrix.Experiment) and isinstance(e.overrides, tuple) for e in matrix.MATRIX)


def test_counts_names_and_labels():
    assert len(matrix.experiments(W=8)) == 48 and len(matrix.experiments(W=10)) == 16
    names = [e.name for e in matrix.MATRIX]
    assert len(set(names)) == 64
    for W in (8, 10):
        labels = [e.label for e in matrix.experiments(W=W)]
        assert len(set(labels)) == len(labels), W
    assert matrix.GROUPS == GROUPS
    assert all(e.group in matrix.GROUPS for e in matrix.MATRIX)
    assert {e.group for e in matrix.MATRIX} == set(GROUPS)  # no empty group


def test_W_and_kind():
    assert {e.name for e in matrix.MATRIX if e.kind == "tabular"} == TABULAR
    assert all(e.kind in ("tabular", "dqn") for e in matrix.MATRIX)
    assert matrix.experiment("tabular_q_W10_gamma0").W == 10
    assert matrix.experiment("dqn_B8_W8_gamma099_R3_H1000").W == 8  # the H1000 is not the board
    assert matrix.experiment("dqn_A_W10").W == 10
    assert [e.W for e in matrix.MATRIX] == [8] * len(EXPECTED_W8) + [10] * len(EXPECTED_W10)
    assert all(type(e.W) is int for e in matrix.MATRIX)


def test_every_row_has_a_config_file_and_loads_with_its_overrides():
    for exp in matrix.MATRIX:
        assert (REPO_ROOT / exp.config).is_file(), exp.name
        assert "\\" not in exp.config and exp.config.startswith("configs/")
        cfg = matrix.expected_config(exp)
        assert isinstance(cfg, ExperimentConfig), exp.name
        assert cfg.exp == exp.name
        assert cfg.env.W == exp.W, exp.name
        assert cfg.agent_kind == ("tabular_q" if exp.kind == "tabular" else "dqn"), exp.name
        assert cfg == load_config(REPO_ROOT / exp.config, [*exp.overrides, f"exp={exp.name}"])


def test_expected_config_reads_the_configs_of_the_given_repo_root(tmp_path):
    shutil.copytree(REPO_ROOT / "configs", tmp_path / "configs")
    exp = matrix.experiment("dqn_A_W8_gamma099_R2")
    assert matrix.expected_config(exp, repo_root=tmp_path) == matrix.expected_config(exp)
    assert matrix.expected_config(exp, str(tmp_path)).agent.gamma == 0.99
    with pytest.raises((OSError, ValueError)):
        matrix.expected_config(exp, repo_root=tmp_path / "nowhere")


def test_a_few_rows_by_hand():
    r2 = matrix.experiment("dqn_A_W8_R2")
    assert (r2.config, r2.overrides, r2.group, r2.label) == (
        "configs/dqn_A_8x8.yaml", ("env.reward.variant=R2", "env.reward.gamma=0.9"), "reward", "A, R2")
    cfg = matrix.expected_config(r2)
    assert cfg.env.reward.variant == "R2" and cfg.env.reward.gamma == cfg.agent.gamma == 0.9
    t31 = matrix.expected_config(matrix.experiment("dqn_B8_W10_G1_t31"))
    assert t31.agent.grad_steps_per_vec_step == 1 and t31.agent.target_update_every == 31 and t31.env.W == 10
    linear = matrix.expected_config(matrix.experiment("dqn_A_W8_linear"))
    assert linear.agent.hidden == ()
    b3 = matrix.expected_config(matrix.experiment("dqn_B3_W8"))
    assert b3.env.obs.kind == "B" and not b3.env.obs.age_plane and not b3.env.obs.heading_planes
    gamma0 = matrix.expected_config(matrix.experiment("tabular_q_W10_gamma0"))
    assert gamma0.agent.gamma == 0.0 and gamma0.agent_kind == "tabular_q"


def test_experiment_and_experiments():
    assert matrix.experiment("dqn_A_W8") == matrix.MATRIX[2]
    assert matrix.experiment("dqn_B8_W10_gamma099_R3") == matrix.MATRIX[-1]
    with pytest.raises(ValueError, match="unknown experiment"):
        matrix.experiment("dqn_A_W9")
    with pytest.raises(ValueError, match="unknown experiment"):
        matrix.experiment("")
    assert matrix.experiments() == list(matrix.MATRIX)
    assert [e.name for e in matrix.experiments(group="probe")] == ["dqn_B8_W8_gamma099_R3_c01"]
    assert [e.name for e in matrix.experiments(W=10, group="reward")] == [
        "dqn_B8_W10_gamma099_R1", "dqn_B8_W10_gamma099_R2", "dqn_B8_W10_gamma099_R3"]
    assert [e.name for e in matrix.experiments(W=8, group="tabular")] == ["tabular_q_W8", "tabular_q_W8_gamma0"]
    assert [e.name for e in matrix.experiments(group="main")] == ["dqn_A_W8", "dqn_B8_W8", "dqn_A_W10", "dqn_B8_W10"]
    assert matrix.experiments(W=8) == [e for e in matrix.MATRIX if e.W == 8]  # matrix order
    with pytest.raises(ValueError):
        matrix.experiments(group="rewards")
    with pytest.raises(ValueError):
        matrix.experiments(W=8, group="")


def test_baselines():
    assert isinstance(matrix.BASELINES, tuple) and len(matrix.BASELINES) == 6
    assert {(b.name, b.W) for b in matrix.BASELINES} == {(n, W) for n in ("random", "bfs", "hamiltonian") for W in (8, 10)}
    assert [b.name for b in matrix.baselines(W=8)] == ["random", "bfs", "hamiltonian"]
    assert [b.label for b in matrix.baselines(W=8)] == ["random", "BFS greedy", "Hamiltonian"]
    assert [b.path for b in matrix.baselines(W=10)] == [
        "runs/baselines/random_W10_relative.json", "runs/baselines/bfs_W10_relative.json",
        "runs/baselines/hamiltonian_W10_relative.json"]
    assert matrix.baselines() == list(matrix.BASELINES)
    assert set(matrix.baselines(8)) | set(matrix.baselines(10)) == set(matrix.BASELINES)
    assert all(isinstance(b, matrix.Baseline) for b in matrix.BASELINES)
    assert matrix.Baseline(name="bfs", W=8, label="BFS greedy").path == "runs/baselines/bfs_W8_relative.json"


def test_the_rows_are_frozen():
    with pytest.raises(AttributeError):
        matrix.MATRIX[0].name = "x"  # type: ignore[misc]
    with pytest.raises(AttributeError):
        matrix.BASELINES[0].W = 12  # type: ignore[misc]


# ---------------------------------------------------------------------- status
def test_status_of_an_absent_experiment_is_empty(tmp_path):
    runs = tmp_path / "runs"
    runs.mkdir()
    s = status_of(runs, "dqn_A_W8")
    assert isinstance(s, matrix.ExperimentStatus)
    assert s.experiment == matrix.experiment("dqn_A_W8")
    assert (s.finished, s.aborted, s.unfinished, s.mismatches) == ((), (), (), ())
    (runs / "dqn_B8_W8").mkdir()  # an empty experiment directory is the same
    assert status_of(runs, "dqn_B8_W8").finished == ()
    assert len(matrix.status(runs)) == 64  # experiments=None means the whole matrix, in matrix order
    assert [s.experiment.name for s in matrix.status(runs)] == [e.name for e in matrix.MATRIX]
    assert len(matrix.status(str(runs), experiments=matrix.experiments(W=10))) == 16


def test_status_classifies_the_run_directories(tmp_path):
    runs = tmp_path / "runs"
    for seed in (10, 2, 0):  # creation order is not seed order
        write_matrix_run(runs, "dqn_A_W8", seed)
    write_matrix_run(runs, "dqn_A_W8", 3, finished=False, aborted=True)  # config + aborted.json
    write_matrix_run(runs, "dqn_A_W8", 4, finished=False)  # config.yaml only
    (runs / "dqn_A_W8" / "5").mkdir()
    make_result(FIVE_SEEDS[1]).save(runs / "dqn_A_W8" / "5" / "final_eval.json")  # final_eval.json only
    (runs / "dqn_A_W8" / "plots").mkdir()  # none of the files: not a run
    (runs / "dqn_A_W8" / "notes.txt").write_text("not a directory", encoding="utf-8")
    s = status_of(runs, "dqn_A_W8")
    assert s.finished == (0, 2, 10) and all(type(v) is int for v in s.finished)
    assert s.aborted == ("3",)
    assert s.unfinished == ("4", "5")
    assert s.mismatches == ()
    assert isinstance(s.finished, tuple) and isinstance(s.aborted, tuple) and isinstance(s.unfinished, tuple)


def test_status_does_not_raise_when_nothing_is_finished(tmp_path):
    runs = tmp_path / "runs"
    write_matrix_run(runs, "tabular_q_W8", 0, finished=False)
    write_matrix_run(runs, "tabular_q_W8", 1, finished=False, aborted=True)
    s = status_of(runs, "tabular_q_W8")
    assert (s.finished, s.aborted, s.unfinished, s.mismatches) == ((), ("1",), ("0",), ())


def test_status_reports_a_run_of_another_configuration(tmp_path):
    runs = tmp_path / "runs"
    write_matrix_run(runs, "dqn_A_W8", 0, edit={"agent.gamma": 0.5})
    write_matrix_run(runs, "dqn_A_W8", 1)
    s = status_of(runs, "dqn_A_W8")
    assert s.finished == (0, 1)  # the seed is still counted: the two facts are independent
    assert s.mismatches == ("0: agent.gamma 0.5 recorded, 0.9 expected",)
    # several keys of one run: one string, "; "-joined; several runs: one string each
    write_matrix_run(runs, "dqn_A_W8", 2, edit={"agent.gamma": 0.99, "agent.double": True})
    write_matrix_run(runs, "dqn_A_W8", 7, finished=False, edit={"train.total_env_steps": 256_000})
    s = status_of(runs, "dqn_A_W8")
    assert len(s.mismatches) == 3
    assert s.mismatches[0] == "0: agent.gamma 0.5 recorded, 0.9 expected"
    assert s.mismatches[1].startswith("2: ") and "agent.gamma 0.99 recorded, 0.9 expected" in s.mismatches[1]
    assert "agent.double" in s.mismatches[1] and "; " in s.mismatches[1]
    assert s.mismatches[2].startswith("7: ") and "train.total_env_steps 256000 recorded, 512000 expected" in s.mismatches[2]
    assert s.finished == (0, 1, 2) and s.unfinished == ("7",)


def test_status_ignores_exp_the_run_block_and_keep_checkpoints(tmp_path):
    runs = tmp_path / "runs"
    write_matrix_run(runs, "dqn_A_W8_gamma099", 0, edit={"agent.keep_checkpoints": False, "exp": "renamed"})
    run_dir = write_matrix_run(runs, "dqn_A_W8_gamma099", 1)
    config = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    config["run"]["git"] = "fffffff+dirty"
    config["run"]["argv"] = ["--config", "x.yaml"]
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    s = status_of(runs, "dqn_A_W8_gamma099")
    assert s.mismatches == () and s.finished == (0, 1)
    assert "agent.keep_checkpoints" in IGNORED_SETTINGS  # the same tuple as aggregate
    write_matrix_run(runs, "dqn_A_W8_gamma099", 2, edit={"agent.keep_checkpoints": False, "agent.gamma": 0.9})
    s = status_of(runs, "dqn_A_W8_gamma099")
    assert s.mismatches == ("2: agent.gamma 0.9 recorded, 0.99 expected",)  # keep_checkpoints is not mentioned


def test_status_lists_a_config_without_run_seed_as_a_mismatch(tmp_path):
    runs = tmp_path / "runs"
    write_matrix_run(runs, "dqn_B8_W8", 0, run_block=False, dir_name="no_seed")
    s = status_of(runs, "dqn_B8_W8")
    assert s.finished == ()
    assert len(s.mismatches) == 1 and "no_seed" in s.mismatches[0] and "no run.seed" in s.mismatches[0]
    write_matrix_run(runs, "dqn_B8_W8", 1)
    s = status_of(runs, "dqn_B8_W8")
    assert s.finished == (1,) and len(s.mismatches) == 1 and "no run.seed" in s.mismatches[0]


def test_status_works_on_a_tabular_row(tmp_path):
    runs = tmp_path / "runs"
    for seed in SEEDS:
        write_matrix_run(runs, "tabular_q_W8_gamma0", seed)
    s = status_of(runs, "tabular_q_W8_gamma0")
    assert s.finished == SEEDS and s.mismatches == ()
    write_matrix_run(runs, "tabular_q_W8_gamma0", 5, edit={"agent.gamma": 0.9})
    assert status_of(runs, "tabular_q_W8_gamma0").mismatches == ("5: agent.gamma 0.9 recorded, 0.0 expected",)


# -------------------------------------------------------------------- commands
def test_commands_for_the_missing_rows_only():
    statuses = [st("dqn_A_W8", finished=SEEDS), st("dqn_A_W8_gamma099", finished=(0, 1, 2)),
                st("tabular_q_W8"), st("dqn_B8_W8_G1_t31", finished=(0, 1, 2, 3), aborted=("4",))]
    assert matrix.commands(statuses) == [
        "python scripts/run_matrix.py --config configs/dqn_A_8x8.yaml --seeds 0 1 2 3 4 --jobs 5 "
        "--exp-prefix dqn_A_W8_gamma099 --set agent.gamma=0.99 agent.keep_checkpoints=false",
        "python scripts/run_matrix.py --config configs/tabular_q_8x8.yaml --seeds 0 1 2 3 4 --jobs 5 "
        "--exp-prefix tabular_q_W8",
        "python scripts/run_matrix.py --config configs/dqn_B_8x8.yaml --seeds 0 1 2 3 4 --jobs 5 "
        "--exp-prefix dqn_B8_W8_G1_t31 --set agent.grad_steps_per_vec_step=1 agent.target_update_every=31 "
        "agent.keep_checkpoints=false",
    ]
    assert matrix.commands([]) == []
    assert matrix.commands([st("dqn_A_W8", finished=SEEDS)]) == []
    assert matrix.commands([st("dqn_A_W8", finished=(0, 1, 2, 3, 4, 5))]) == []  # more than enough seeds


def test_commands_with_missing_only_false_lists_every_row():
    statuses = [st("dqn_A_W8", finished=SEEDS), st("tabular_q_W8_gamma0", finished=SEEDS)]
    assert matrix.commands(statuses, missing_only=False) == [
        "python scripts/run_matrix.py --config configs/dqn_A_8x8.yaml --seeds 0 1 2 3 4 --jobs 5 "
        "--exp-prefix dqn_A_W8 --set agent.keep_checkpoints=false",
        "python scripts/run_matrix.py --config configs/tabular_q_8x8.yaml --seeds 0 1 2 3 4 --jobs 5 "
        "--exp-prefix tabular_q_W8_gamma0 --set agent.gamma=0.0",
    ]


def test_commands_keep_checkpoints_item_placement():
    statuses = [st("dqn_A_W8"), st("dqn_A_W8_gamma099"), st("tabular_q_W8"), st("tabular_q_W8_gamma0")]
    without = matrix.commands(statuses)
    assert without[0].endswith(" --exp-prefix dqn_A_W8 --set agent.keep_checkpoints=false")
    assert without[1].endswith(" --exp-prefix dqn_A_W8_gamma099 --set agent.gamma=0.99 agent.keep_checkpoints=false")
    assert without[2].endswith(" --exp-prefix tabular_q_W8")  # a tabular row never gets the item
    assert without[3].endswith(" --exp-prefix tabular_q_W8_gamma0 --set agent.gamma=0.0")
    kept = matrix.commands(statuses, keep_checkpoints=True)
    assert kept[0].endswith(" --exp-prefix dqn_A_W8") and "--set" not in kept[0]
    assert kept[1].endswith(" --exp-prefix dqn_A_W8_gamma099 --set agent.gamma=0.99")
    assert kept[2:] == without[2:]
    assert all("keep_checkpoints" not in line for line in kept)
    for line in [*without, *kept]:
        assert line.count("--set") <= 1 and line.count("--exp-prefix") == 1


def test_commands_custom_python_jobs_and_seeds():
    statuses = [st("dqn_B8_W10", finished=(0,)), st("dqn_A_W10", finished=(0, 1)), st("tabular_q_W10")]
    lines = matrix.commands(statuses, seeds=(7, 3), jobs=2, python="C:/venv/Scripts/python.exe")
    # dqn_A_W10 has 2 finished seeds: not fewer than the 2 requested
    assert lines == [
        "C:/venv/Scripts/python.exe scripts/run_matrix.py --config configs/dqn_B_10x10.yaml --seeds 7 3 --jobs 2 "
        "--exp-prefix dqn_B8_W10 --set agent.keep_checkpoints=false",
        "C:/venv/Scripts/python.exe scripts/run_matrix.py --config configs/tabular_q_10x10.yaml --seeds 7 3 --jobs 2 "
        "--exp-prefix tabular_q_W10",
    ]
    assert matrix.commands(statuses, seeds=[0, 1, 2], jobs=1)[0].startswith(
        "python scripts/run_matrix.py --config configs/dqn_B_10x10.yaml --seeds 0 1 2 --jobs 1 ")


def test_commands_from_a_runs_directory(tmp_path):
    runs = tmp_path / "runs"
    for seed in SEEDS:
        write_matrix_run(runs, "tabular_q_W8", seed)
    write_matrix_run(runs, "tabular_q_W8_gamma0", 0)
    lines = matrix.commands(matrix.status(runs, experiments=matrix.experiments(W=8, group="tabular")))
    assert lines == ["python scripts/run_matrix.py --config configs/tabular_q_8x8.yaml --seeds 0 1 2 3 4 --jobs 5 "
                     "--exp-prefix tabular_q_W8_gamma0 --set agent.gamma=0.0"]


# ---------------------------------------------------------------- status_table
def test_status_table_exact_format():
    statuses = [
        st("tabular_q_W8", finished=SEEDS),
        st("dqn_A_W8_gamma099", finished=(0, 1), aborted=("2",), unfinished=("3",),
           mismatches=("1: agent.gamma 0.9 recorded, 0.99 expected",)),
        st("dqn_B8_W10_G1_t31"),
        st("dqn_B3_W10", finished=(0, 1, 2), mismatches=("0: x", "1: y")),
    ]
    table = matrix.status_table(statuses)
    lines = table.splitlines()
    assert lines[0] == STATUS_HEAD
    assert lines[1].count("|") == 10 and set(lines[1]) <= set("|-: ")
    assert lines[2:] == [
        "| tabular_q_W8 | 8x8 | tabular | configs/tabular_q_8x8.yaml | - | 5/5 | 0 | 0 | ok |",
        "| dqn_A_W8_gamma099 | 8x8 | gamma | configs/dqn_A_8x8.yaml | agent.gamma=0.99 | 2/5 | 1 | 1 | MISMATCH (1 run(s)) |",
        "| dqn_B8_W10_G1_t31 | 10x10 | budget | configs/dqn_B_10x10.yaml | "
        "agent.grad_steps_per_vec_step=1 agent.target_update_every=31 | 0/5 | 0 | 0 | ok |",
        "| dqn_B3_W10 | 10x10 | obs | configs/dqn_B_10x10.yaml | "
        "env.obs.age_plane=false env.obs.heading_planes=false | 3/5 | 0 | 0 | MISMATCH (2 run(s)) |",
    ]
    assert matrix.status_table(statuses[:1], n_seeds=3).splitlines()[2].split(" | ")[5] == "5/3"
    assert len(matrix.status_table([]).splitlines()) == 2


# ------------------------------------------------------------------------- CLI
def test_cli_status_prints_the_table_and_exits_0_without_mismatches(tmp_path, capsys):
    runs = tmp_path / "runs"
    runs.mkdir()
    code, out, err = cli(["--status", "--runs-dir", str(runs)], capsys)
    assert code == 0
    lines = out.splitlines()
    assert lines[0] == STATUS_HEAD and len(lines) == 2 + 64
    assert [line.split(" | ")[0].lstrip("| ") for line in lines[2:]] == [e.name for e in matrix.MATRIX]
    assert all(line.endswith("| 0/5 | 0 | 0 | ok |") for line in lines[2:])
    code, out, err = cli(["--status", "--runs-dir", str(runs), "--W", "10", "--group", "reward"], capsys)
    assert code == 0
    assert [line.split(" | ")[0].lstrip("| ") for line in out.splitlines()[2:]] == [
        "dqn_B8_W10_gamma099_R1", "dqn_B8_W10_gamma099_R2", "dqn_B8_W10_gamma099_R3"]
    code, out, err = cli(["--status", "--runs-dir", str(runs), "--group", "probe"], capsys)
    assert code == 0 and len(out.splitlines()) == 3


def test_cli_status_exits_1_on_a_mismatch(tmp_path, capsys):
    runs = tmp_path / "runs"
    for seed in SEEDS:
        write_matrix_run(runs, "tabular_q_W8", seed)
    write_matrix_run(runs, "dqn_A_W8", 0, edit={"agent.gamma": 0.5})
    code, out, err = cli(["--status", "--runs-dir", str(runs), "--W", "8"], capsys)
    assert code == 1
    table = matrix.status_table(matrix.status(runs, experiments=matrix.experiments(W=8)))
    assert out.startswith(table + "\n")  # the table first; what follows (if anything) spells the mismatches out
    assert "| tabular_q_W8 | 8x8 | tabular | configs/tabular_q_8x8.yaml | - | 5/5 | 0 | 0 | ok |" in out.splitlines()
    assert "| dqn_A_W8 | 8x8 | main | configs/dqn_A_8x8.yaml | - | 1/5 | 0 | 0 | MISMATCH (1 run(s)) |" in out.splitlines()
    assert "agent.gamma 0.5 recorded, 0.9 expected" in out + err
    code, out, err = cli(["--status", "--runs-dir", str(runs), "--group", "tabular"], capsys)
    assert code == 0  # the mismatching row is not selected


def test_cli_commands(tmp_path, capsys):
    runs = tmp_path / "runs"
    for seed in SEEDS:
        write_matrix_run(runs, "tabular_q_W8", seed)
    code, out, err = cli(["--commands", "--runs-dir", str(runs), "--W", "8", "--group", "tabular"], capsys)
    assert code == 0
    assert out.splitlines() == ["python scripts/run_matrix.py --config configs/tabular_q_8x8.yaml --seeds 0 1 2 3 4 "
                                "--jobs 5 --exp-prefix tabular_q_W8_gamma0 --set agent.gamma=0.0"]
    code, out, err = cli(["--commands", "--all", "--runs-dir", str(runs), "--W", "8", "--group", "tabular"], capsys)
    assert code == 0 and len(out.splitlines()) == 2
    assert out.splitlines()[0].endswith("--exp-prefix tabular_q_W8")
    code, out, err = cli(["--commands", "--runs-dir", str(runs), "--group", "main", "--jobs", "2",
                          "--python", "C:/venv/python.exe"], capsys)
    assert code == 0
    assert out.splitlines() == [
        "C:/venv/python.exe scripts/run_matrix.py --config configs/dqn_A_8x8.yaml --seeds 0 1 2 3 4 --jobs 2 "
        "--exp-prefix dqn_A_W8 --set agent.keep_checkpoints=false",
        "C:/venv/python.exe scripts/run_matrix.py --config configs/dqn_B_8x8.yaml --seeds 0 1 2 3 4 --jobs 2 "
        "--exp-prefix dqn_B8_W8 --set agent.keep_checkpoints=false",
        "C:/venv/python.exe scripts/run_matrix.py --config configs/dqn_A_10x10.yaml --seeds 0 1 2 3 4 --jobs 2 "
        "--exp-prefix dqn_A_W10 --set agent.keep_checkpoints=false",
        "C:/venv/python.exe scripts/run_matrix.py --config configs/dqn_B_10x10.yaml --seeds 0 1 2 3 4 --jobs 2 "
        "--exp-prefix dqn_B8_W10 --set agent.keep_checkpoints=false",
    ]
    code, out, err = cli(["--commands", "--keep-checkpoints", "--runs-dir", str(runs), "--group", "main", "--W", "10"],
                         capsys)
    assert code == 0
    assert out.splitlines() == [
        "python scripts/run_matrix.py --config configs/dqn_A_10x10.yaml --seeds 0 1 2 3 4 --jobs 5 --exp-prefix dqn_A_W10",
        "python scripts/run_matrix.py --config configs/dqn_B_10x10.yaml --seeds 0 1 2 3 4 --jobs 5 --exp-prefix dqn_B8_W10",
    ]
    code, out, err = cli(["--commands", "--runs-dir", str(runs)], capsys)
    assert code == 0 and len(out.splitlines()) == 63  # everything but the finished tabular_q_W8


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--runs-dir", "runs"],
        ["--status", "--commands"],
        ["--status", "--all"],
        ["--status", "--keep-checkpoints"],
        ["--status", "--jobs", "2"],
        ["--status", "--python", "python3"],
        ["--all"],
        ["--status", "--W", "9"],
        ["--status", "--W", "eight"],
        ["--status", "--group", "rewards"],
        ["--commands", "--jobs", "two"],
        ["--sttaus"],
    ],
    ids=lambda argv: " ".join(argv) or "nothing",
)
def test_cli_argument_errors(tmp_path, capsys, argv):
    runs = tmp_path / "runs"
    runs.mkdir()
    code, out, err = cli([*argv, "--runs-dir", str(runs)] if "--runs-dir" not in argv else argv, capsys)
    assert code == 2, (argv, err)
    assert out == ""
    assert err != ""


def test_module_entry_point(tmp_path):
    runs = tmp_path / "runs"
    for seed in SEEDS:
        write_matrix_run(runs, "tabular_q_W8", seed)
    proc = subprocess.run(
        [sys.executable, "-m", "snake_rl.analysis.matrix", "--status", "--runs-dir", str(runs), "--group", "tabular"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == matrix.status_table(matrix.status(runs, experiments=matrix.experiments(group="tabular")))
    proc = subprocess.run(
        [sys.executable, "-m", "snake_rl.analysis.matrix", "--commands", "--runs-dir", str(runs), "--group", "tabular"],
        cwd=REPO_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert len(proc.stdout.strip().splitlines()) == 3  # tabular_q_W8_gamma0, tabular_q_W10, tabular_q_W10_gamma0
