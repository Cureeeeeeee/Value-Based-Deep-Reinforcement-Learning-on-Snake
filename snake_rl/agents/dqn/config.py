"""Hyper-parameters of the DQN agent and of its training loop (PLAN.md M5).

This module is deliberately free of torch: ``snake_rl.config`` imports it to
validate experiment files, and neither a tabular run nor
``python -m snake_rl.train --help`` may pay for the torch import.

The dataclass only holds values. Ranges and types are validated once, in
``snake_rl.config`` (the ``agent`` block with ``kind: dqn``), with error
messages that name the offending key.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DQNConfig:
    #: Discount.
    gamma: float = 0.99
    #: Adam step size.
    lr: float = 2.5e-4
    #: Transitions per gradient step.
    batch_size: int = 256
    #: Capacity of the replay buffer, in transitions.
    buffer_size: int = 100_000
    #: Environment steps collected before the first gradient step.
    learning_starts: int = 10_000
    #: Gradient steps after every vec-step (``n_envs`` environment steps).
    #: ``None`` means ``max(1, n_envs // 4)``: one gradient step per four
    #: environment steps, the update-to-data ratio of Mnih et al. (2015).
    grad_steps_per_vec_step: int | None = None
    #: Gradient steps between two hard target syncs. 1 = the target network is
    #: synchronised after every step, i.e. no separate target network in effect.
    target_update_every: int = 1_000
    #: Maximum global gradient norm (clipping threshold).
    grad_clip: float = 10.0
    eps_start: float = 1.0
    eps_end: float = 0.05
    #: Epsilon decays linearly over this fraction of ``train.total_env_steps``
    #: (used by the training loop, not by the agent).
    eps_decay_fraction: float = 0.2
    #: Widths of the hidden layers of the MLP. ``()`` = a linear Q-function.
    hidden: tuple[int, ...] = (256, 256)
    #: Double DQN target (van Hasselt et al. 2016); studied in M7.
    double: bool = False
    #: Training loop: write a weights-only checkpoint at every evaluation.
    keep_checkpoints: bool = True
    #: ``"auto"`` (cuda if available, else cpu) | ``"cpu"`` | ``"cuda"``.
    device: str = "auto"
    #: Training loop: ``torch.set_num_threads`` at trainer start; ``None``
    #: leaves torch's setting as it is.
    torch_threads: int | None = None

    def __post_init__(self) -> None:
        # A YAML file or a checkpoint delivers ``hidden`` as a list; the frozen
        # (hashable, comparable) config keeps a tuple. Anything else is left as
        # it is for the validation in ``snake_rl.config`` to report.
        if isinstance(self.hidden, list):
            object.__setattr__(self, "hidden", tuple(self.hidden))
