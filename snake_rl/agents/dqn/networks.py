"""Q-networks: observation -> one action value per action.

M5 has the MLP for observation A (11 binary features). The CNN for the grid
observation B arrives in M6 and the Dueling head in M7; both will be built by
:func:`build_q_network`, so the agent never names an architecture.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn


class MLP(nn.Module):
    """Fully connected Q-network.

    ``Linear -> ReLU`` for every width in ``hidden``, then a ``Linear`` layer
    to ``n_actions`` without activation (action values are unbounded).
    ``hidden = ()`` gives a linear Q-function ``Q(s, .) = W s + b``. The
    parameters take torch's default initialisation, drawn from the global CPU
    generator at construction (the agent seeds that draw).
    """

    def __init__(self, in_features: int, n_actions: int, hidden: Sequence[int]) -> None:
        super().__init__()
        widths = [int(in_features), *(int(h) for h in hidden)]
        if min(widths) < 1 or n_actions < 1:
            raise ValueError(
                f"layer sizes must be positive: in_features {in_features}, hidden {tuple(hidden)}, n_actions {n_actions}"
            )
        layers: list[nn.Module] = []
        for fan_in, fan_out in zip(widths[:-1], widths[1:]):
            layers += [nn.Linear(fan_in, fan_out), nn.ReLU()]
        layers.append(nn.Linear(widths[-1], int(n_actions)))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` float32 ``(N, in_features)`` -> action values ``(N, n_actions)``."""
        return self.net(x)


def build_q_network(obs_shape: tuple[int, ...], n_actions: int, hidden: Sequence[int]) -> nn.Module:
    """The Q-network for observations of shape ``obs_shape``.

    A 1-D shape (observation A) gives an :class:`MLP`. Any other shape is the
    grid observation B, whose CNN is milestone M6.
    """
    if len(obs_shape) == 1:
        return MLP(obs_shape[0], n_actions, hidden)
    raise NotImplementedError(
        f"no Q-network for observations of shape {tuple(obs_shape)}: the CNN for observation B is milestone M6"
    )
