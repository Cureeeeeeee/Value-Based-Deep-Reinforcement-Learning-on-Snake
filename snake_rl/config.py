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

``agent.kind: dqn`` (M5) selects ``DQNConfig`` (``snake_rl/agents/dqn/config.py``,
which documents every field); its ``agent`` block, with the defaults::

    agent:
      kind: dqn
      gamma: 0.99
      lr: 2.5e-4                     # Adam
      batch_size: 256
      buffer_size: 100000            # transitions in the replay buffer
      replay: true                   # false: one gradient step per vec-step on its own transitions (M6)
      learning_starts: 10000         # env steps collected before the first gradient step
      grad_steps_per_vec_step: null  # null -> max(1, n_envs // 4); must be null or 1 without replay
      target_update_every: 1000      # gradient steps between two hard target syncs (1 = no target network)
      grad_clip: 10.0                # max global gradient norm
      eps_start: 1.0
      eps_end: 0.05
      eps_decay_fraction: 0.2
      hidden: [256, 256]             # fully connected widths; [] = linear Q-function (observation A)
      conv_channels: [32, 64, 64]    # 3x3 convolutions for observation B; [] = none (M6)
      double: false                  # Double DQN target (M7)
      keep_checkpoints: true         # weights-only checkpoint at every evaluation
      device: auto                   # auto | cpu | cuda
      torch_threads: null            # null -> torch's own setting

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
* ``dqn``: ``lr > 0``; ``batch_size >= 1``; ``buffer_size >= batch_size``;
  ``batch_size <= learning_starts <= train.total_env_steps``;
  ``grad_steps_per_vec_step`` and ``torch_threads`` null or ``>= 1``;
  ``target_update_every >= 1``; ``grad_clip > 0``; ``hidden`` and
  ``conv_channels`` lists of positive integers (possibly empty); ``replay``
  a boolean; ``device`` one of auto / cpu / cuda; gamma and the epsilon
  schedule as for ``tabular_q``; observation A (MLP) or B (CNN);
* ``dqn`` with ``replay: true``: ``buffer_size >= train.n_envs`` (a smaller
  buffer would drop part of every vec-step); with ``replay: false``:
  ``grad_steps_per_vec_step`` null or 1 (there is exactly one gradient step
  per vec-step, on that step's own transitions);
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

from snake_rl.agents.dqn.config import DQNConfig  # torch-free: validating a file never imports torch
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
DEVICES: tuple[str, ...] = ("auto", "cpu", "cuda")


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
    #: ``"tabular_q"`` or ``"dqn"``.
    agent_kind: str
    #: Config dataclass of that agent kind.
    agent: TabularQConfig | DQNConfig
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
    A tuple field (``hidden`` of ``DQNConfig``) is written as a list, the form
    YAML and JSON give back.
    """
    agent = {name: list(value) if isinstance(value, tuple) else value for name, value in asdict(cfg.agent).items()}
    return {
        "exp": cfg.exp,
        "env": asdict(cfg.env),
        "agent": {"kind": cfg.agent_kind, **agent},
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
    try:
        agent_section = top.section("agent", ("kind", *_field_names(config_type)))
    except ValueError as err:  # an unknown key: the keys that exist depend on the kind, so say which one it is
        raise ValueError(f"{err} for agent.kind {kind}") from None
    agent = parse_agent(agent_section)

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

    def optional_integer(self, name: str, default: int | None) -> int | None:
        """An integer or null (``None``)."""
        value = self.raw.get(name, default)
        if value is not None and not _is_int(value):
            raise ValueError(f"{self.key(name)} must be null or an integer, got {value!r}")
        return value

    def number(self, name: str, default: float) -> float:
        value = self.raw.get(name, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            hint = ""
            # PyYAML reads YAML 1.1: a float needs a dot in the mantissa AND a
            # sign in the exponent; 1e-3 and 1.0e9 are both strings to it.
            if isinstance(value, str) and re.fullmatch(r"[+-]?\d+(\.\d*)?[eE][+-]?\d+", value.strip()):
                hint = " (YAML needs a dot in the mantissa and a sign in the exponent: write 1.0e-3 or 1.0e+9)"
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


def _dqn_config(s: _Section) -> DQNConfig:
    default = DQNConfig()
    hidden = s.raw.get("hidden", list(default.hidden))
    _require(
        isinstance(hidden, (list, tuple)) and all(_is_int(width) and width >= 1 for width in hidden),
        f"{s.key('hidden')} must be a list of positive integers (possibly empty: a linear Q-function), got {hidden!r}",
    )
    conv_channels = s.raw.get("conv_channels", list(default.conv_channels))
    _require(
        isinstance(conv_channels, (list, tuple)) and all(_is_int(c) and c >= 1 for c in conv_channels),
        f"{s.key('conv_channels')} must be a list of positive integers (possibly empty: no convolution), "
        f"got {conv_channels!r}",
    )
    cfg = DQNConfig(
        gamma=s.number("gamma", default.gamma),
        lr=s.number("lr", default.lr),
        batch_size=s.integer("batch_size", default.batch_size),
        buffer_size=s.integer("buffer_size", default.buffer_size),
        replay=s.boolean("replay", default.replay),
        learning_starts=s.integer("learning_starts", default.learning_starts),
        grad_steps_per_vec_step=s.optional_integer("grad_steps_per_vec_step", default.grad_steps_per_vec_step),
        target_update_every=s.integer("target_update_every", default.target_update_every),
        grad_clip=s.number("grad_clip", default.grad_clip),
        eps_start=s.number("eps_start", default.eps_start),
        eps_end=s.number("eps_end", default.eps_end),
        eps_decay_fraction=s.number("eps_decay_fraction", default.eps_decay_fraction),
        hidden=tuple(hidden),
        conv_channels=tuple(conv_channels),
        double=s.boolean("double", default.double),
        keep_checkpoints=s.boolean("keep_checkpoints", default.keep_checkpoints),
        device=s.choice("device", default.device, DEVICES),
        torch_threads=s.optional_integer("torch_threads", default.torch_threads),
    )
    _require(0 <= cfg.gamma <= 1, f"{s.key('gamma')} must be in [0, 1], got {cfg.gamma}")
    _require(cfg.lr > 0, f"{s.key('lr')} must be positive, got {cfg.lr}")
    _require(cfg.batch_size >= 1, f"{s.key('batch_size')} must be at least 1, got {cfg.batch_size}")
    _require(
        cfg.buffer_size >= cfg.batch_size,
        f"{s.key('buffer_size')} ({cfg.buffer_size}) must be at least {s.key('batch_size')} ({cfg.batch_size})",
    )
    _require(
        cfg.learning_starts >= cfg.batch_size,
        f"{s.key('learning_starts')} ({cfg.learning_starts}) must be at least {s.key('batch_size')} "
        f"({cfg.batch_size}): the first gradient step needs a batch of transitions",
    )
    _require(
        cfg.grad_steps_per_vec_step is None or cfg.grad_steps_per_vec_step >= 1,
        f"{s.key('grad_steps_per_vec_step')} must be null or at least 1, got {cfg.grad_steps_per_vec_step}",
    )
    _require(
        cfg.replay or cfg.grad_steps_per_vec_step in (None, 1),
        f"{s.key('grad_steps_per_vec_step')} must be null or 1 when {s.key('replay')} is false: without replay "
        f"there is exactly one gradient step per vec-step, on that step's own transitions; "
        f"got {cfg.grad_steps_per_vec_step}",
    )
    _require(
        cfg.target_update_every >= 1,
        f"{s.key('target_update_every')} must be at least 1, got {cfg.target_update_every}",
    )
    _require(cfg.grad_clip > 0, f"{s.key('grad_clip')} must be positive, got {cfg.grad_clip}")
    _require(
        0 <= cfg.eps_end <= cfg.eps_start <= 1,
        f"need 0 <= {s.key('eps_end')} <= {s.key('eps_start')} <= 1, "
        f"got eps_end {cfg.eps_end} and eps_start {cfg.eps_start}",
    )
    _require(
        0 <= cfg.eps_decay_fraction <= 1,
        f"{s.key('eps_decay_fraction')} must be in [0, 1], got {cfg.eps_decay_fraction}",
    )
    _require(
        cfg.torch_threads is None or cfg.torch_threads >= 1,
        f"{s.key('torch_threads')} must be null or at least 1, got {cfg.torch_threads}",
    )
    return cfg


#: ``agent.kind`` -> (config dataclass, parser of the ``agent`` section).
_AGENT_KINDS: dict[str, tuple[type, Callable[[_Section], Any]]] = {
    "tabular_q": (TabularQConfig, _tabular_q_config),
    "dqn": (DQNConfig, _dqn_config),
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
    if cfg.agent_kind == "dqn":
        _require(
            agent.learning_starts <= cfg.train.total_env_steps,
            f"agent.learning_starts ({agent.learning_starts}) must not exceed train.total_env_steps "
            f"({cfg.train.total_env_steps}): the run would end before the first gradient step",
        )
        _require(
            not agent.replay or agent.buffer_size >= cfg.train.n_envs,
            f"agent.buffer_size ({agent.buffer_size}) must be at least train.n_envs ({cfg.train.n_envs}) with "
            f"replay: a smaller buffer would drop part of every vec-step's transitions",
        )
    if env.reward.variant == "R2":
        _require(
            env.reward.gamma == agent.gamma,
            f"env.reward.gamma ({env.reward.gamma}) must equal agent.gamma ({agent.gamma}) under reward R2: "
            f"potential-based shaping keeps the optimal policy only for the agent's discount; set both",
        )
