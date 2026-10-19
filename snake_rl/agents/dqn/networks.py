"""Q-networks: observation -> one action value per action.

M5: the :class:`MLP` for observation A (11 binary features). M6: the
:class:`CNN` for the grid observation B ``(C, W, W)``. Both are built by
:func:`build_q_network`, so the agent never names an architecture. The
Dueling head arrives in M7.
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


class CNN(nn.Module):
    """Convolutional Q-network for the grid observation B.

    Adapted from ``resources/week09/43008-Week9-Part-D-DQN-Atari-CustomCNN-
    Solution-Colab.ipynb`` cell 13 (``CustomCNN``): a ``Sequential`` stack of
    ``Conv2d -> ReLU`` ending in ``Flatten``, the flattened size found by one
    dummy forward pass, then the fully connected part. Changed for this
    repository (docs/RESOURCE_BASELINE.md section 2): the base class is
    ``nn.Module`` (no stable-baselines3); the convolutions are 3x3, stride 1,
    padding 1 (the resource's Atari kernels 8x8/4, 4x4/2 leave nothing of a
    10x10 board; padding keeps the board size, so ``n_flatten`` is
    ``conv_channels[-1] * W * W``); the dummy input is ``torch.zeros(1, *in_shape)``
    (there is no ``observation_space`` here); and the resource's
    ``Linear(n_flatten, features_dim) -> ReLU`` plus the Q head
    ``Linear(features_dim, n_actions)`` that SB3 adds behind it are this
    repository's :class:`MLP` with ``hidden = (features_dim,)`` (the shipped
    configurations use the resource's ``features_dim = 512``).

    ``conv_channels = ()`` gives no convolution at all: ``Flatten`` followed
    by the MLP, i.e. a fully connected network on the raw grid (an ablation
    of the convolutional prior). Parameters take torch's default
    initialisation from the global CPU generator, as in :class:`MLP`.
    """

    def __init__(
        self,
        in_shape: tuple[int, int, int],
        n_actions: int,
        conv_channels: Sequence[int],
        hidden: Sequence[int],
    ) -> None:
        super().__init__()
        shape = tuple(int(d) for d in in_shape)
        if len(shape) != 3 or min(shape) < 1:
            raise ValueError(f"in_shape must be three positive integers (C, H, W), got {tuple(in_shape)}")
        channels = [int(c) for c in conv_channels]
        if any(c < 1 for c in channels) or n_actions < 1:
            raise ValueError(f"conv_channels must be positive, got {tuple(conv_channels)}; n_actions {n_actions}")
        layers: list[nn.Module] = []
        previous = shape[0]
        for c in channels:
            layers += [nn.Conv2d(previous, c, kernel_size=3, stride=1, padding=1), nn.ReLU()]
            previous = c
        self.cnn = nn.Sequential(*layers, nn.Flatten())
        # Compute the size of the output tensor after passing through the CNN
        # (resource cell 13, with a zero tensor in place of observation_space.sample()).
        with torch.no_grad():
            self.n_flatten = int(self.cnn(torch.zeros(1, *shape)).shape[1])
        self.head = MLP(self.n_flatten, n_actions, hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x`` float32 ``(N, C, H, W)`` -> action values ``(N, n_actions)``."""
        return self.head(self.cnn(x))


def build_q_network(
    obs_shape: tuple[int, ...],
    n_actions: int,
    hidden: Sequence[int],
    conv_channels: Sequence[int] = (32, 64, 64),
) -> nn.Module:
    """The Q-network for observations of shape ``obs_shape``.

    A 1-D shape (observation A) gives an :class:`MLP` and ignores
    ``conv_channels``; a 3-D shape ``(C, H, W)`` (observation B) gives a
    :class:`CNN`. Any other shape is a ``ValueError``.
    """
    shape = tuple(obs_shape)
    if len(shape) == 1:
        return MLP(shape[0], n_actions, hidden)
    if len(shape) == 3:
        return CNN(shape, n_actions, conv_channels, hidden)
    raise ValueError(
        f"no Q-network for observations of shape {shape}: expected (features,) for observation A "
        f"or (channels, height, width) for observation B"
    )
