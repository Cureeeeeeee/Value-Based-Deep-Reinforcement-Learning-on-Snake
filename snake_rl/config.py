"""Experiment configuration: one YAML file -> validated, frozen dataclasses.

Layout. Every key is optional except ``exp`` and ``agent.kind``; a missing key
takes the default of its dataclass (``EnvConfig``, ``ObsConfig``,
``RewardConfig``, ``TabularQConfig``, ``TrainConfig``)::

    exp: tabular_q_W10           # experiment name = directory under the runs dir
    env:
      W: 10
      action_mode: relative      # relative | absolute
      starve_limit: null         # null -> max(100, W * W)
      obs: {kind: A}             # A | B (B also takes age_plane, heading_planes)
      reward: {variant: R0}      # R0..R3 (plus food, death, step_penalty, gamma, closer, farther)
    agent:
      kind: tabular_q            # selects the agent and its config dataclass
      alpha: 0.1
      gamma: 0.99
      eps_start: 1.0
      eps_end: 0.05
      eps_decay_fraction: 0.2
      q_init: 0.0
    train:
      n_envs: 50
      total_env_steps: 1000000
      eval_every: 50000
      eval_episodes: 100
      eval_seed: 10000

``load_config(path, overrides)`` reads the file, applies the
``dotted.key=value`` overrides (value parsed as YAML, e.g. ``agent.alpha=0.2``,
``env.reward.variant=R1``, ``exp=my_name``) and validates the result.
``config_to_dict`` is the inverse: the nested dict in this layout with every
field spelled out. A top-level ``run`` block is accepted and ignored; it is
what ``snake_rl.train`` adds to the ``config.yaml`` of a run directory, so the
file of a finished run can be loaded again to repeat the run.

Validation happens here, once, before anything is built, because the
environment classes accept bad values silently (``int(3.9)`` is 3, a
misspelt reward variant fails only at the first ``step``). Every error is a
``ValueError`` whose message names the offending key by its dotted path:

* unknown key at any level (a typo must not fall back to a default);
* types: integer fields take integers only (``128.0`` and ``true`` are
  rejected), number fields take finite numbers, flags take booleans;
* ranges: ``W >= 4``; ``starve_limit`` null or ``>= 1``; ``0 < alpha <= 1``;
  ``0 <= gamma <= 1``; ``0 <= eps_end <= eps_start <= 1``;
  ``0 <= eps_decay_fraction <= 1``; ``1 <= n_envs <= 100000``;
  ``eval_episodes >= 2`` (a confidence interval needs two episodes);
* evaluations land on exact step counts: ``eval_every`` is a positive
  multiple of ``n_envs`` and ``total_env_steps`` a positive multiple of
  ``eval_every``;
* ``tabular_q`` needs observation A (the table is indexed by the 11 bits);
* reward R2 needs ``env.reward.gamma == agent.gamma``: potential-based shaping
  (Ng et al. 1999) preserves the optimal policy only for the discount the
  agent actually uses;
* ``exp`` is a directory name made of letters, digits and ``. _ - = +``.

Seeds (``env_seed_for``). ``VecSnakeEnv`` gives sub-env ``i`` the generator
``default_rng(seed + i)``, so two base seeds closer than ``n_envs`` share
food streams. Run seed ``s`` therefore trains on the block starting at
``100000 * (s + 1)``; with ``n_envs <= 100000`` the blocks of different runs
are disjoint. The evaluation boards ``eval_seed .. eval_seed + eval_episodes
- 1`` (10000 .. 10099 by default) must end below 100000, the start of the
first training block, so no evaluation board is ever a training stream.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Sequence, get_args

import yaml

from snake_rl.agents.tabular_q import TabularQConfig
from snake_rl.env.common import ActionMode
from snake_rl.env.core import EnvConfig
from snake_rl.env.obs import ObsConfig
from snake_rl.env.rewards import RewardConfig, RewardVariant

#: Distance between the training seed blocks of two consecutive run seeds, and
#: therefore the largest ``n_envs`` (see the module docstring).
ENV_SEED_SPACING = 100_000
MAX_N_ENVS = ENV_SEED_SPACING

ACTION_MODES: tuple[str, ...] = get_args(ActionMode)
OBS_KINDS: tuple[str, ...] = ("A", "B")
REWARD_VARIANTS: tuple[str, ...] = get_args(RewardVariant)


@dataclass(frozen=True)
class TrainConfig:
    """Training budget and evaluation protocol, shared by every agent kind."""

    #: Parallel boards of the training ``VecSnakeEnv``. (50, not a power of
    #: two, so that the round evaluation interval below is a multiple of it.)
    n_envs: int = 50
    #: Environment steps in total (``n_envs`` per vec-step).
    total_env_steps: int = 1_000_000
    #: Environment steps between two evaluations.
    eval_every: int = 50_000
    #: Greedy episodes per evaluation.
    eval_episodes: int = 100
    #: Evaluation episode ``i`` plays the board of seed ``eval_seed + i``
    #: (``evaluate.DEFAULT_EVAL_SEED``): the same boards for every agent.
    eval_seed: int = 10_000


@dataclass(frozen=True)
class ExperimentConfig:
    """One experiment: what is trained, on which environment, for how long.
    Together with a run seed it determines a run completely."""

    #: Experiment name = directory under the runs dir.
    exp: str
    env: EnvConfig
    #: ``"tabular_q"`` now; ``"dqn"`` is added in M5.
    agent_kind: str
    #: Config dataclass of that agent kind.
    agent: TabularQConfig
    train: TrainConfig


# ------------------------------------------------------------------ seeds
def env_seed_for(run_seed: int, n_envs: int) -> int:
    """Base seed of the training environment of run ``run_seed``:
    ``100000 * (run_seed + 1)``. See the module docstring for the spacing."""
    if not _is_int(run_seed) or run_seed < 0:
        raise ValueError(f"the run seed must be an integer >= 0, got {run_seed!r}")
    if not _is_int(n_envs) or not 1 <= n_envs <= MAX_N_ENVS:
        raise ValueError(f"n_envs must be an integer in 1..{MAX_N_ENVS} (the seed spacing between runs), got {n_envs!r}")
    return ENV_SEED_SPACING * (run_seed + 1)


# ------------------------------------------------------------ dict <-> config
def config_to_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """Nested plain dict in the YAML layout, every field included.

    ``config_from_dict(config_to_dict(cfg)) == cfg`` for every valid config.
    """
    return {
        "exp": cfg.exp,
        "env": asdict(cfg.env),
        "agent": {"kind": cfg.agent_kind, **asdict(cfg.agent)},
        "train": asdict(cfg.train),
    }


def config_from_dict(raw: Mapping[str, Any]) -> ExperimentConfig:
    """Validate a nested dict in the YAML layout and build the config."""
    top = _Section(raw, "", ("exp", "env", "agent", "train", "run"))
    exp = _exp_name(top.raw.get("exp"))
    env = _env_config(top.section("env", _field_names(EnvConfig)))

    agent_raw = top.raw.get("agent")
    kind = agent_raw.get("kind") if isinstance(agent_raw, Mapping) else None
    if agent_raw is not None and not isinstance(agent_raw, Mapping):
        raise ValueError(f"agent must be a mapping, got {agent_raw!r}")
    if kind is None:
        raise ValueError(f"agent.kind is required (one of: {', '.join(_AGENT_KINDS)})")
    if not isinstance(kind, str) or kind not in _AGENT_KINDS:
        raise ValueError(f"agent.kind must be one of: {', '.join(_AGENT_KINDS)}; got {kind!r}")
    config_type, parse_agent = _AGENT_KINDS[kind]
    agent = parse_agent(top.section("agent", ("kind", *_field_names(config_type))))

    train = _train_config(top.section("train", _field_names(TrainConfig)))
    cfg = ExperimentConfig(exp=exp, env=env, agent_kind=kind, agent=agent, train=train)
    _check_combination(cfg)
    return cfg


class _UniqueKeyLoader(yaml.SafeLoader):
    """``yaml.safe_load`` that refuses a repeated key. PyYAML keeps the last
    one silently, so a second ``train:`` block would drop the keys of the
    first one back to their defaults."""

    def construct_mapping(self, node: yaml.MappingNode, deep: bool = False) -> dict[Any, Any]:
        seen: set[Any] = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=True)
            if key in seen:
                raise yaml.constructor.ConstructorError(None, None, f"duplicate key {key!r}", key_node.start_mark)
            seen.add(key)
        return super().construct_mapping(node, deep)


def load_config(path: str | Path, overrides: Sequence[str] = ()) -> ExperimentConfig:
    """Read a YAML experiment file, apply ``dotted.key=value`` overrides in
    order, validate. ``ValueError`` (prefixed with the path) on any problem."""
    path = Path(path)
    if isinstance(overrides, str):
        overrides = (overrides,)
    try:
        raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_UniqueKeyLoader)
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raise ValueError(f"the top level must be a mapping, got {raw!r}")
        for item in overrides:
            _apply_override(raw, item)
        return config_from_dict(raw)
    except yaml.YAMLError as err:
        raise ValueError(f"{path}: not valid YAML: {err}") from err
    except ValueError as err:
        raise ValueError(f"{path}: {err}") from err


def _apply_override(raw: dict[str, Any], item: str) -> None:
    """Set ``dotted.key=value`` in the nested dict; the value is parsed as YAML."""
    key, sep, text = str(item).partition("=")
    parts = key.strip().split(".")
    if not sep or not all(parts):
        raise ValueError(f"override {item!r} is not of the form dotted.key=value")
    try:
        value = yaml.safe_load(text)
    except yaml.YAMLError as err:
        raise ValueError(f"override {item!r}: the value is not valid YAML: {err}") from err
    node = raw
    for depth, part in enumerate(parts[:-1], start=1):
        child = node.get(part)
        if child is None:
            child = node[part] = {}
        elif not isinstance(child, dict):
            raise ValueError(f"override {item!r}: {'.'.join(parts[:depth])} is not a section")
        node = child
    node[parts[-1]] = value


# ----------------------------------------------------------------- parsing
def _is_int(value: object) -> bool:
    """A real integer: ``True`` is not one, and neither is ``128.0``."""
    return isinstance(value, int) and not isinstance(value, bool)


def _field_names(config_type: type) -> tuple[str, ...]:
    return tuple(f.name for f in fields(config_type))


def _require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


class _Section:
    """One mapping of the configuration with its dotted path. The typed
    getters return the value of a key, or ``default`` when the key is absent,
    and name the key in every error."""

    def __init__(self, raw: object, path: str, allowed: Sequence[str]) -> None:
        if raw is None:  # "env:" with every line commented out
            raw = {}
        if not isinstance(raw, Mapping):
            raise ValueError(f"{path or 'the top level'} must be a mapping, got {raw!r}")
        self.raw = raw
        self.path = path
        for name in raw:
            if name not in allowed:
                raise ValueError(f"unknown key {self.key(name)} (known keys here: {', '.join(allowed)})")

    def key(self, name: object) -> str:
        return f"{self.path}.{name}" if self.path else str(name)

    def section(self, name: str, allowed: Sequence[str]) -> "_Section":
        return _Section(self.raw.get(name), self.key(name), allowed)

    def integer(self, name: str, default: int) -> int:
        value = self.raw.get(name, default)
        if not _is_int(value):
            raise ValueError(f"{self.key(name)} must be an integer, got {value!r}")
        return value

    def number(self, name: str, default: float) -> float:
        value = self.raw.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            hint = ""
            if isinstance(value, str) and re.fullmatch(r"[+-]?\d+[eE][+-]?\d+", value.strip()):
                hint = " (YAML needs a dot in the mantissa: write 1.0e-3, not 1e-3)"
            raise ValueError(f"{self.key(name)} must be a finite number, got {value!r}{hint}")
        return float(value)

    def boolean(self, name: str, default: bool) -> bool:
        value = self.raw.get(name, default)
        if not isinstance(value, bool):
            raise ValueError(f"{self.key(name)} must be true or false, got {value!r}")
        return value

    def choice(self, name: str, default: str, allowed: Sequence[str]) -> str:
        value = self.raw.get(name, default)
        if not isinstance(value, str) or value not in allowed:
            raise ValueError(f"{self.key(name)} must be one of: {', '.join(allowed)}; got {value!r}")
        return value


_EXP_NAME = re.compile(r"[A-Za-z0-9._=+-]+")
#: Windows refuses these as file or directory names, with or without a suffix.
_WINDOWS_DEVICE_NAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))]
)


def _exp_name(value: object) -> str:
    if value is None:
        raise ValueError("exp is required: the experiment name, used as the directory under the runs dir")
    if not isinstance(value, str):
        raise ValueError(f"exp must be a string, got {value!r} (quote it)")
    usable = (
        _EXP_NAME.fullmatch(value) is not None
        and not value.endswith(".")  # ".", ".." and names Windows would silently shorten
        and value.split(".")[0].upper() not in _WINDOWS_DEVICE_NAMES
    )
    _require(
        usable,
        f"exp must be a non-empty directory name made of letters, digits and . _ - = + only "
        f"(not ending in a dot, not a Windows device name), got {value!r}",
    )
    return value


def _env_config(s: _Section) -> EnvConfig:
    default = EnvConfig()
    W = s.integer("W", default.W)
    _require(W >= 4, f"{s.key('W')} must be at least 4, got {W}")
    starve_limit = s.raw.get("starve_limit", default.starve_limit)
    if starve_limit is not None:
        starve_limit = s.integer("starve_limit", 0)
        _require(starve_limit >= 1, f"{s.key('starve_limit')} must be null or at least 1, got {starve_limit}")
    action_mode = s.choice("action_mode", default.action_mode, ACTION_MODES)

    o = s.section("obs", _field_names(ObsConfig))
    obs = ObsConfig(
        kind=o.choice("kind", default.obs.kind, OBS_KINDS),
        age_plane=o.boolean("age_plane", default.obs.age_plane),
        heading_planes=o.boolean("heading_planes", default.obs.heading_planes),
    )
    r = s.section("reward", _field_names(RewardConfig))
    numbers = [name for name in _field_names(RewardConfig) if name != "variant"]
    reward = RewardConfig(
        variant=r.choice("variant", default.reward.variant, REWARD_VARIANTS),
        **{name: r.number(name, getattr(default.reward, name)) for name in numbers},
    )
    return EnvConfig(W=W, action_mode=action_mode, starve_limit=starve_limit, obs=obs, reward=reward)


def _tabular_q_config(s: _Section) -> TabularQConfig:
    default = TabularQConfig()
    cfg = TabularQConfig(**{name: s.number(name, getattr(default, name)) for name in _field_names(TabularQConfig)})
    _require(0 < cfg.alpha <= 1, f"{s.key('alpha')} must be in (0, 1], got {cfg.alpha}")
    _require(0 <= cfg.gamma <= 1, f"{s.key('gamma')} must be in [0, 1], got {cfg.gamma}")
    _require(
        0 <= cfg.eps_end <= cfg.eps_start <= 1,
        f"need 0 <= {s.key('eps_end')} <= {s.key('eps_start')} <= 1, "
        f"got eps_end {cfg.eps_end} and eps_start {cfg.eps_start}",
    )
    _require(
        0 <= cfg.eps_decay_fraction <= 1,
        f"{s.key('eps_decay_fraction')} must be in [0, 1], got {cfg.eps_decay_fraction}",
    )
    return cfg


def _train_config(s: _Section) -> TrainConfig:
    default = TrainConfig()
    cfg = TrainConfig(**{name: s.integer(name, getattr(default, name)) for name in _field_names(TrainConfig)})
    n_envs, every, total = cfg.n_envs, cfg.eval_every, cfg.total_env_steps
    _require(1 <= n_envs <= MAX_N_ENVS, f"{s.key('n_envs')} must be in 1..{MAX_N_ENVS}, got {n_envs}")
    _require(
        every >= n_envs and every % n_envs == 0,
        f"{s.key('eval_every')} ({every}) must be a positive multiple of {s.key('n_envs')} ({n_envs}), "
        f"so that evaluations land on exact step counts",
    )
    _require(
        total >= every and total % every == 0,
        f"{s.key('total_env_steps')} ({total}) must be a positive multiple of {s.key('eval_every')} ({every})",
    )
    _require(cfg.eval_episodes >= 2, f"{s.key('eval_episodes')} must be at least 2, got {cfg.eval_episodes}")
    _require(
        cfg.eval_seed >= 0 and cfg.eval_seed + cfg.eval_episodes <= ENV_SEED_SPACING,
        f"{s.key('eval_seed')} ({cfg.eval_seed}): the evaluation boards eval_seed .. eval_seed + eval_episodes - 1 "
        f"must lie in 0..{ENV_SEED_SPACING - 1}, below the first training seed block",
    )
    return cfg


#: ``agent.kind`` -> (config dataclass, parser of the ``agent`` section).
#: M5 registers ``"dqn"`` here.
_AGENT_KINDS: dict[str, tuple[type, Callable[[_Section], Any]]] = {
    "tabular_q": (TabularQConfig, _tabular_q_config),
}


def _check_combination(cfg: ExperimentConfig) -> None:
    """Rules that tie two sections together."""
    env, agent = cfg.env, cfg.agent
    if cfg.agent_kind == "tabular_q":
        _require(
            env.obs.kind == "A",
            f"env.obs.kind must be A for agent.kind tabular_q (the table is indexed by the 11 binary "
            f"features), got {env.obs.kind!r}",
        )
    if env.reward.variant == "R2":
        _require(
            env.reward.gamma == agent.gamma,
            f"env.reward.gamma ({env.reward.gamma}) must equal agent.gamma ({agent.gamma}) under reward R2: "
            f"potential-based shaping keeps the optimal policy only for the agent's discount; set both",
        )
