"""DQN core (PLAN.md milestones M5 and M6): the oracle for ``snake_rl/agents/dqn``.

Everything here is derived from the M5 specification (sections D0-D3) and
the M6 specification (D0-D2: the CNN on observation B, the ``replay`` and
``conv_channels`` fields), from the textbook algorithm (Mnih et al. 2015; van
Hasselt et al. 2016 for the Double DQN target) and from the environments -
never from the implementation under test (``config.py``, ``networks.py``,
``replay.py``, ``agent.py``).

Contract under test.

* ``DQNConfig``: a frozen dataclass with the defaults of D0; importing it (or
  the package ``snake_rl.agents.dqn``) does not import torch.
* ``MLP`` / ``build_q_network``: Linear -> ReLU per hidden layer, then a Linear
  to the action values without activation; ``hidden = ()`` is a linear
  Q-function.
* ``CNN`` (M6, for the grid observation B): ``Conv2d(3x3, stride 1, padding 1)
  -> ReLU`` per width in ``conv_channels``, a ``Flatten``, then the ``MLP`` of
  ``hidden`` on ``n_flatten = conv_channels[-1] * H * W`` features;
  ``conv_channels = ()`` feeds the flattened grid straight into the MLP.
  ``build_q_network`` dispatches on the observation shape: 1-D -> ``MLP``,
  3-D -> ``CNN``, anything else -> ``ValueError``.
* ``ReplayBuffer``: a FIFO ring over preallocated storage. ``add`` copies its
  arguments; ``sample`` draws ``idx = rng.integers(0, len(buffer),
  size=batch_size)`` - one call, uniform with replacement over the stored
  transitions - and returns copies with fixed dtypes.
* ``DQNAgent.update`` is one gradient step on::

      q_sa   = Q_online(s)[a]
      target = r + gamma * (0 if terminated else max_a' Q_target(s', a'))
      target = r + gamma * (0 if terminated else Q_target(s', argmax_a' Q_online(s', a')))   [double]
      loss   = mean Huber(q_sa - target)       (quadratic below 1, linear above)

  with no gradient through the target, the global gradient norm clipped to
  ``grad_clip``, Adam, and a hard copy of the online network into the target
  network whenever ``grad_steps`` reaches a multiple of
  ``target_update_every``. A non-finite loss raises ``FloatingPointError``
  before anything changes.
* truncated != terminated: the caller passes the env ``terminated`` flag only
  and stores ``info["final_obs"]`` as ``next_obs``, so a starvation truncation
  bootstraps from its true successor.
* Greedy actions break ties towards the lowest index, the same in ``act`` and
  ``act_batch``. ``act_epsilon`` has the RNG protocol of the tabular agent:
  ``explore = rng.random(N) < epsilon``, then ``random_actions =
  rng.integers(0, n_actions, size=N)``, both drawn on every call.
* ``save`` / ``load``: an exact round trip through a file of plain tensors and
  primitives (``torch.load(weights_only=True)``).

Exactness. Integer results, copies, round trips and CPU determinism are
compared without tolerance. Network arithmetic is float32: the hand-built
cases use dyadic numbers (exact in float32) and are compared at 1e-5 or
tighter; the textbook loop evaluates the same formulas with other operations
(explicit Huber, explicit clipping), so its weights are compared at 1e-4
after up to 310 updates (measured distance: see ``TEXTBOOK_ATOL``).

How the tests look inside the implementation. Only through the attributes of
the spec (``q_net``, ``target_net``, ``optimizer``, ``grad_steps``, ``cfg``,
``device``; ``cnn``, ``n_flatten`` and ``head`` of the ``CNN``) and through
torch itself: the ``torch.nn.Linear`` and ``torch.nn.Conv2d`` modules of a
network in registration order (to set and read weights), and the first-moment
buffer of ``torch.optim.Adam`` after its first step, which is ``(1 - beta1)``
times the gradient the optimiser consumed (to read the clipped gradient).
The replay storage is read through the sampling protocol itself, with a
generator whose ``integers`` returns scripted indices.

Layout: config, networks (MLP, then the CNN of M6), replay buffer, agent
construction, action selection, update on hand-built transitions, target
network, failure and purity, textbook equivalence (the central test), the
agent on grid observations (M6), save / load, CUDA, learning sanity.
"""

from __future__ import annotations

import copy
import math
import re
import subprocess
import sys
import time
from collections import deque
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from snake_rl.agents import Policy
from snake_rl.agents.baselines import RandomPolicy
from snake_rl.agents.dqn.agent import DQNAgent
from snake_rl.agents.dqn.config import DQNConfig
from snake_rl.agents.dqn.networks import CNN, MLP, build_q_network
from snake_rl.agents.dqn.replay import Batch, ReplayBuffer
from snake_rl.agents.schedules import linear_schedule
from snake_rl.env.common import RIGHT
from snake_rl.env.obs import A_DANGER_STRAIGHT, A_HEADING, N_FEATURES_A
from snake_rl.env.vec_env import VecSnakeEnv
from snake_rl.evaluate import DEFAULT_EVAL_SEED, evaluate, evaluate_batched
from tests.helpers import FAR_FOOD, TURN_LEFT, EnvConfig, ObsConfig, RewardConfig

REPO_ROOT = Path(__file__).resolve().parents[1]
OBS_DIM = N_FEATURES_A  # 11
N_ACTIONS = 3
BATCH_FIELDS = ("obs", "actions", "rewards", "next_obs", "terminated")
STAT_KEYS = {"loss", "mean_q", "mean_target", "grad_norm"}

needs_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")


# ------------------------------------------------------------- helpers
def cpu_cfg(**kwargs) -> DQNConfig:
    """Config on the CPU; the default device ``auto`` would pick the GPU."""
    return DQNConfig(device="cpu", **kwargs)


def linear_layers(net: torch.nn.Module) -> list[torch.nn.Linear]:
    """The ``Linear`` modules of a network in registration (= forward) order."""
    return [m for m in net.modules() if isinstance(m, torch.nn.Linear)]


def numpy_forward(net: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    """Spec D1 in float64 numpy: Linear -> ReLU per hidden layer, then a Linear."""
    layers = linear_layers(net)
    h = np.asarray(x, dtype=np.float64)
    for i, layer in enumerate(layers):
        h = h @ layer.weight.detach().cpu().numpy().astype(np.float64).T + layer.bias.detach().cpu().numpy()
        if i < len(layers) - 1:
            h = np.maximum(h, 0.0)
    return h


def set_linear(net: torch.nn.Module, weight, bias) -> None:
    """Set the weights of a network that is a single ``Linear`` layer."""
    (layer,) = linear_layers(net)
    with torch.no_grad():
        layer.weight.copy_(torch.tensor(weight, dtype=torch.float32))
        layer.bias.copy_(torch.tensor(bias, dtype=torch.float32))


def weights(net: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Detached CPU copy of a network's state dict."""
    return {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}


def same(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> bool:
    """Bit-for-bit equality of two state dicts."""
    return a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def arrays(batch: Batch) -> list[np.ndarray]:
    return [getattr(batch, name) for name in BATCH_FIELDS]


def random_obs(rng: np.random.Generator, n: int, dim: int = OBS_DIM) -> np.ndarray:
    return rng.integers(0, 2, size=(n, dim)).astype(np.float32)


def random_batch(
    rng: np.random.Generator,
    n: int = 32,
    rewards: tuple[float, ...] = (-1.0, 0.0, 0.0, 1.0),
    p_terminated: float = 0.2,
) -> Batch:
    """Synthetic transitions on binary 11-feature observations."""
    return Batch(
        obs=random_obs(rng, n),
        actions=rng.integers(0, N_ACTIONS, size=n).astype(np.int64),
        rewards=rng.choice(rewards, size=n).astype(np.float32),
        next_obs=random_obs(rng, n),
        terminated=rng.random(n) < p_terminated,
    )


def small_agent(seed: int = 0, **kwargs) -> DQNAgent:
    """Agent on observation A with a small MLP, on the CPU."""
    settings = dict(hidden=(16, 16), lr=1e-2)
    settings.update(kwargs)
    return DQNAgent((OBS_DIM,), N_ACTIONS, cpu_cfg(**settings), seed=seed)


def randomise(net: torch.nn.Module, seed: int) -> None:
    """Standard-normal weights and biases from a local generator: action
    values that vary with the observation, whatever the initialisation was."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in linear_layers(net):
            layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator))
            layer.bias.copy_(torch.randn(layer.bias.shape, generator=generator))


# ------------------------------------------------------------- D0: config
#: Field order and defaults of spec D0.
CONFIG_DEFAULTS = {
    "gamma": 0.99,
    "lr": 2.5e-4,
    "batch_size": 256,
    "buffer_size": 100_000,
    "replay": True,
    "learning_starts": 10_000,
    "grad_steps_per_vec_step": None,
    "target_update_every": 1_000,
    "grad_clip": 10.0,
    "eps_start": 1.0,
    "eps_end": 0.05,
    "eps_decay_fraction": 0.2,
    "hidden": (256, 256),
    "conv_channels": (32, 64, 64),
    "double": False,
    "keep_checkpoints": True,
    "device": "auto",
    "torch_threads": None,
}


def test_config_defaults_and_immutability():
    cfg = DQNConfig()
    assert [f.name for f in fields(DQNConfig)] == list(CONFIG_DEFAULTS)
    for name, value in CONFIG_DEFAULTS.items():
        got = getattr(cfg, name)
        assert got == value and type(got) is type(value), (name, got)
    with pytest.raises(FrozenInstanceError):
        cfg.gamma = 0.5
    with pytest.raises(FrozenInstanceError):
        cfg.double = True
    with pytest.raises(FrozenInstanceError):
        cfg.replay = False
    assert DQNConfig(gamma=0.9, hidden=(64,)) == DQNConfig(gamma=0.9, hidden=(64,)) != cfg
    assert DQNConfig(hidden=()).hidden == ()
    assert DQNConfig(replay=False).replay is False and DQNConfig(replay=False) != cfg
    # M6: a list (YAML, checkpoint) becomes a tuple for conv_channels as for hidden; () = no convolution.
    assert DQNConfig(conv_channels=[8, 8]).conv_channels == (8, 8)
    assert type(DQNConfig(conv_channels=[8, 8]).conv_channels) is tuple
    assert DQNConfig(conv_channels=()).conv_channels == () and DQNConfig(conv_channels=[]).conv_channels == ()
    assert DQNConfig(conv_channels=(16,)) == DQNConfig(conv_channels=[16]) != cfg


def test_config_and_package_import_without_torch():
    """Tabular runs and ``--help`` must not pay the torch import: the config
    module and the package are importable (and the config usable) without it."""
    code = (
        "import sys\n"
        "import snake_rl.agents.dqn.config\n"
        "import snake_rl.agents.dqn\n"
        "from snake_rl.agents.dqn.config import DQNConfig\n"
        "assert DQNConfig().batch_size == 256\n"
        "assert DQNConfig().replay is True and DQNConfig().conv_channels == (32, 64, 64)\n"
        "print('torch-loaded' if 'torch' in sys.modules else 'torch-free')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "torch-free"


# ------------------------------------------------------------- D1: networks
def test_mlp_shapes_dtype_and_layers():
    net = MLP(OBS_DIM, N_ACTIONS, (16, 8))
    assert isinstance(net, torch.nn.Module)
    assert [tuple(layer.weight.shape) for layer in linear_layers(net)] == [(16, 11), (8, 16), (3, 8)]
    assert sum(p.numel() for p in net.parameters()) == (11 * 16 + 16) + (16 * 8 + 8) + (8 * 3 + 3)
    x = torch.from_numpy(random_obs(np.random.default_rng(0), 5))
    out = net(x)
    assert out.shape == (5, N_ACTIONS) and out.dtype == torch.float32
    assert net(x[:1]).shape == (1, N_ACTIONS)
    # ``hidden`` is any sequence of ints.
    assert [tuple(layer.weight.shape) for layer in linear_layers(MLP(4, 2, [5]))] == [(5, 4), (2, 5)]


@pytest.mark.parametrize("hidden", [(), (16,), (16, 8), (8, 8, 8)])
def test_mlp_forward_is_linear_relu_linear(hidden):
    """The forward pass against the formula in float64 numpy, on seeded random weights."""
    net = MLP(OBS_DIM, 4, hidden)
    assert len(linear_layers(net)) == len(hidden) + 1
    randomise(net, seed=len(hidden))
    rng = np.random.default_rng(len(hidden))
    x = rng.normal(size=(64, OBS_DIM)).astype(np.float32)
    with torch.no_grad():
        out = net(torch.from_numpy(x)).numpy()
    want = numpy_forward(net, x)
    assert out.shape == (64, 4)
    assert np.allclose(out, want, rtol=1e-5, atol=1e-5 * np.abs(want).max())
    if hidden:
        assert (want < 0).any()  # the action values are not rectified


def test_mlp_without_hidden_layers_is_one_linear_map():
    net = MLP(2, 3, ())
    assert len(linear_layers(net)) == 1 and len(list(net.parameters())) == 2
    set_linear(net, [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [0.0, 0.5, -1.0])
    with torch.no_grad():
        out = net(torch.tensor([[1.0, 2.0], [-3.0, 0.5], [0.0, 0.0]]))
    # Q(s) = [s0, s1 + 0.5, s0 + s1 - 1]: no activation, negative values pass.
    assert out.tolist() == [[1.0, 2.5, 2.0], [-3.0, 1.0, -3.5], [0.0, 0.5, -1.0]]


def test_mlp_applies_relu_between_layers_and_not_after_the_last():
    net = MLP(2, 2, (2,))
    first, last = linear_layers(net)
    with torch.no_grad():
        first.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
        first.bias.zero_()
        last.weight.copy_(torch.tensor([[1.0, -1.0], [2.0, 3.0]]))
        last.bias.copy_(torch.tensor([0.5, -0.5]))
        out = net(torch.tensor([[1.0, -2.0], [-1.0, 3.0], [-4.0, -4.0]]))
    # hidden = relu(x): [1, 0], [0, 3], [0, 0]. Without the ReLU the first row
    # would be [3.5, -4.5]; with a ReLU on the output the second would be [0, 8.5].
    assert out.tolist() == [[1.5, 1.5], [-2.5, 8.5], [0.5, -0.5]]


def test_build_q_network_gives_an_mlp_for_observation_a():
    net = build_q_network((OBS_DIM,), N_ACTIONS, (32, 16))
    assert isinstance(net, MLP)
    assert [tuple(layer.weight.shape) for layer in linear_layers(net)] == [(32, 11), (16, 32), (3, 16)]
    linear = build_q_network((OBS_DIM,), 4, ())
    assert [tuple(layer.weight.shape) for layer in linear_layers(linear)] == [(4, 11)]


# ------------------------------------------------------------- D1 (M6): CNN
#: Grid observations of the spec: 8 planes on 8x8, 3 planes (no age plane, no
#: heading planes) on 10x10, 7 planes (no age plane) on 8x8, 8 planes on 10x10.
GRID_SHAPES = [(8, 8, 8), (3, 10, 10), (7, 8, 8), (8, 10, 10)]
DEFAULT_CONV = (32, 64, 64)
#: (in_shape, conv_channels, hidden) of the functional-reference cases.
CNN_CASES = [
    ((8, 8, 8), (4, 6), (16,)),
    ((3, 10, 10), (5,), (8, 8)),
    ((7, 8, 8), (3, 3, 3), ()),
    ((8, 10, 10), (), (16,)),
    ((8, 8, 8), (), ()),
]


def conv_layers(net: torch.nn.Module) -> list[torch.nn.Conv2d]:
    """The ``Conv2d`` modules of a network in registration (= forward) order."""
    return [m for m in net.modules() if isinstance(m, torch.nn.Conv2d)]


def randomise_all(net: torch.nn.Module, seed: int) -> None:
    """Normal weights scaled by the fan-in and normal biases for every ``Conv2d``
    and ``Linear`` layer, from a local generator: action values that vary with
    the observation and stay of order one through several layers."""
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for layer in [m for m in net.modules() if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))]:
            fan_in = layer.weight[0].numel()
            layer.weight.copy_(torch.randn(layer.weight.shape, generator=generator) / math.sqrt(fan_in))
            layer.bias.copy_(0.1 * torch.randn(layer.bias.shape, generator=generator))


def cnn_reference(net: torch.nn.Module, x: np.ndarray) -> np.ndarray:
    """Spec D1 (M6) in float64 with torch's functional operations: ``conv2d``
    (stride 1, padding 1) -> ``relu`` per convolution layer, flatten, then the
    MLP formula (Linear -> ReLU per hidden layer, a Linear at the end)."""
    h = torch.from_numpy(np.asarray(x, dtype=np.float64))
    for conv in conv_layers(net):
        h = F.relu(F.conv2d(h, conv.weight.detach().double(), conv.bias.detach().double(), stride=1, padding=1))
    h = h.flatten(1)
    linears = linear_layers(net)
    for i, layer in enumerate(linears):
        h = h @ layer.weight.detach().double().T + layer.bias.detach().double()
        if i < len(linears) - 1:
            h = F.relu(h)
    return h.numpy()


def random_grid(rng: np.random.Generator, n: int, shape: tuple[int, int, int] = (8, 8, 8)) -> np.ndarray:
    """Synthetic grid observations ``(n, C, H, W)`` float32, standard normal.
    (The planes of B lie in [0, 1]; a zero-mean input keeps the action values
    of a randomised network from sharing one common mode across observations,
    which would make every greedy action the same.)"""
    return rng.normal(size=(n, *shape)).astype(np.float32)


@pytest.mark.parametrize("in_shape", GRID_SHAPES)
def test_cnn_structure_is_conv_relu_per_channel_count_then_flatten_then_the_mlp(in_shape):
    C, H, W = in_shape
    net = CNN(in_shape, N_ACTIONS, DEFAULT_CONV, (512,))
    assert isinstance(net, torch.nn.Module)
    # self.cnn = Sequential(Conv2d, ReLU, Conv2d, ReLU, Conv2d, ReLU, Flatten).
    assert isinstance(net.cnn, torch.nn.Sequential)
    modules = list(net.cnn)
    assert len(modules) == 2 * len(DEFAULT_CONV) + 1 and isinstance(modules[-1], torch.nn.Flatten)
    previous = C
    for i, c in enumerate(DEFAULT_CONV):
        conv, relu = modules[2 * i], modules[2 * i + 1]
        assert isinstance(conv, torch.nn.Conv2d) and isinstance(relu, torch.nn.ReLU), i
        assert conv.in_channels == previous and conv.out_channels == c
        assert conv.kernel_size == (3, 3) and conv.stride == (1, 1) and conv.padding == (1, 1)
        assert conv.dilation == (1, 1) and conv.groups == 1 and conv.bias is not None
        assert tuple(conv.weight.shape) == (c, previous, 3, 3) and tuple(conv.bias.shape) == (c,)
        previous = c
    assert [tuple(layer.weight.shape) for layer in conv_layers(net)] == [(32, C, 3, 3), (64, 32, 3, 3), (64, 64, 3, 3)]
    # Padding keeps H and W: n_flatten = last channel count * H * W.
    assert net.n_flatten == 64 * H * W and type(net.n_flatten) is int
    # self.head = MLP(n_flatten, n_actions, hidden).
    assert isinstance(net.head, MLP)
    assert [tuple(layer.weight.shape) for layer in linear_layers(net.head)] == [(512, 64 * H * W), (N_ACTIONS, 512)]
    assert [tuple(layer.weight.shape) for layer in linear_layers(net)] == [(512, 64 * H * W), (N_ACTIONS, 512)]
    n_params = (
        (32 * C * 9 + 32) + (64 * 32 * 9 + 64) + (64 * 64 * 9 + 64)
        + (64 * H * W * 512 + 512) + (512 * N_ACTIONS + N_ACTIONS)
    )
    assert sum(p.numel() for p in net.parameters()) == n_params
    assert all(key.startswith(("cnn.", "head.")) for key in net.state_dict())
    x = torch.from_numpy(random_grid(np.random.default_rng(0), 5, in_shape))
    out = net(x)
    assert out.shape == (5, N_ACTIONS) and out.dtype == torch.float32
    assert net(x[:1]).shape == (1, N_ACTIONS)
    # ``conv_channels`` and ``hidden`` are any sequences of ints.
    small = CNN(in_shape, 4, [2, 3], [5])
    assert [tuple(layer.weight.shape) for layer in conv_layers(small)] == [(2, C, 3, 3), (3, 2, 3, 3)]
    assert [tuple(layer.weight.shape) for layer in linear_layers(small)] == [(5, 3 * H * W), (4, 5)]
    assert small.n_flatten == 3 * H * W


@pytest.mark.parametrize("in_shape", GRID_SHAPES)
def test_cnn_without_convolutions_flattens_the_grid_into_the_mlp(in_shape):
    C, H, W = in_shape
    net = CNN(in_shape, 4, (), (16, 8))
    assert isinstance(net.cnn, torch.nn.Sequential) and len(list(net.cnn)) == 1
    assert isinstance(list(net.cnn)[0], torch.nn.Flatten) and conv_layers(net) == []
    assert net.n_flatten == C * H * W and type(net.n_flatten) is int
    assert isinstance(net.head, MLP)
    assert [tuple(layer.weight.shape) for layer in linear_layers(net)] == [(16, C * H * W), (8, 16), (4, 8)]
    assert sum(p.numel() for p in net.parameters()) == (C * H * W * 16 + 16) + (16 * 8 + 8) + (8 * 4 + 4)
    randomise(net, seed=1)
    x = random_grid(np.random.default_rng(2), 6, in_shape)
    with torch.no_grad():
        out = net(torch.from_numpy(x)).numpy()
    want = numpy_forward(net, x.reshape(6, -1))  # the MLP on the row-major flattened grid
    assert out.shape == (6, 4)
    assert np.allclose(out, want, rtol=1e-5, atol=1e-5 * np.abs(want).max())
    # Without any hidden layer: a linear Q-function of the C * H * W cells.
    linear = CNN(in_shape, 3, (), ())
    assert conv_layers(linear) == [] and len(linear_layers(linear)) == 1 and len(list(linear.parameters())) == 2
    assert tuple(linear_layers(linear)[0].weight.shape) == (3, C * H * W)


@pytest.mark.parametrize("case", CNN_CASES, ids=lambda case: f"{case[0]}-conv{case[1]}-hidden{case[2]}")
def test_cnn_forward_matches_the_functional_reference(case):
    """``forward`` against ``F.conv2d`` / ``relu`` / flatten / the MLP formula
    on randomised weights, in float64, at 1e-5 of the output scale."""
    in_shape, conv_channels, hidden = case
    net = CNN(in_shape, 4, conv_channels, hidden)
    assert len(conv_layers(net)) == len(conv_channels) and len(linear_layers(net)) == len(hidden) + 1
    randomise_all(net, seed=len(conv_channels) + 10 * len(hidden))
    x = random_grid(np.random.default_rng(len(conv_channels)), 32, in_shape)
    with torch.no_grad():
        out = net(torch.from_numpy(x)).numpy()
    want = cnn_reference(net, x)
    assert out.shape == want.shape == (32, 4) and out.dtype == np.float32
    scale = np.abs(want).max()
    assert scale > 0.01, scale
    assert np.allclose(out, want, rtol=1e-5, atol=1e-5 * scale)
    assert (want < 0).any()  # the action values are not rectified
    if conv_channels:  # the reference is not trivially satisfied: the convolutions do change the output
        with torch.no_grad():
            conv_layers(net)[0].weight.mul_(-1.0)
            flipped = net(torch.from_numpy(x)).numpy()
        assert np.abs(flipped - out).max() > 1e-3 * scale


def test_cnn_applies_relu_after_every_convolution_and_hidden_layer_but_not_after_the_last():
    # One convolution of one channel on a 1x2x2 grid with a known kernel, no hidden layer.
    net = CNN((1, 2, 2), 2, (1,), ())
    (conv,) = conv_layers(net)
    (last,) = linear_layers(net)
    with torch.no_grad():
        conv.weight.zero_()
        conv.weight[0, 0, 1, 1] = 1.0  # the centre tap: the convolution is the identity ...
        conv.bias.fill_(-0.5)  # ... minus 0.5, then ReLU
        last.weight.copy_(torch.tensor([[1.0, 1.0, 1.0, 1.0], [-1.0, 0.0, 0.0, -1.0]]))
        last.bias.copy_(torch.tensor([0.25, -0.75]))
        out = net(torch.tensor([[[[1.0, 0.0], [2.0, 0.25]]], [[[0.0, 0.0], [0.0, 0.0]]]]))
    # relu(x - 0.5) = [0.5, 0, 1.5, 0] and [0, 0, 0, 0]: Q = [2.25, -1.25] and [0.25, -0.75]. Without the
    # ReLU after the convolution the first row would be [1.5, -1.0]; with a ReLU on the output no action
    # value would be negative.
    assert out.tolist() == [[2.25, -1.25], [0.25, -0.75]]
    # The head keeps the MLP rule: ReLU between its layers, none after the last.
    net = CNN((1, 2, 2), 2, (), (2,))
    first, last = linear_layers(net)
    with torch.no_grad():
        first.weight.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]]))
        first.bias.zero_()
        last.weight.copy_(torch.tensor([[1.0, -1.0], [2.0, 3.0]]))
        last.bias.copy_(torch.tensor([0.5, -0.5]))
        out = net(torch.tensor([[[[1.0, 9.0], [9.0, -2.0]]], [[[-1.0, 9.0], [9.0, 3.0]]]]))
    assert out.tolist() == [[1.5, 1.5], [-2.5, 8.5]]


@pytest.mark.parametrize(
    "bad",
    [
        dict(in_shape=(8, 8)), dict(in_shape=(2, 3, 4, 5)), dict(in_shape=(8,)), dict(in_shape=(0, 8, 8)),
        dict(in_shape=(8, 0, 8)), dict(in_shape=(8, 8, -1)), dict(conv_channels=(0,)), dict(conv_channels=(16, -1)),
        dict(conv_channels=(8, 0, 8)), dict(n_actions=0), dict(n_actions=-3), dict(hidden=(0,)), dict(hidden=(16, -8)),
    ],
    ids=lambda bad: ",".join(f"{k}={v}" for k, v in bad.items()),
)
def test_cnn_rejects_bad_sizes(bad):
    settings = dict(in_shape=(8, 8, 8), n_actions=3, conv_channels=(4,), hidden=(8,))
    settings.update(bad)
    with pytest.raises(ValueError):
        CNN(settings["in_shape"], settings["n_actions"], settings["conv_channels"], settings["hidden"])


def test_build_q_network_dispatches_on_the_observation_shape():
    # 1-D: the MLP, whatever conv_channels says.
    mlp = build_q_network((OBS_DIM,), N_ACTIONS, (32, 16), (4, 4))
    assert isinstance(mlp, MLP) and not isinstance(mlp, CNN) and conv_layers(mlp) == []
    assert [tuple(layer.weight.shape) for layer in linear_layers(mlp)] == [(32, 11), (16, 32), (3, 16)]
    # 3-D: the CNN with the channel counts given ...
    cnn = build_q_network((8, 8, 8), N_ACTIONS, (16,), (4, 6))
    assert isinstance(cnn, CNN)
    assert [tuple(layer.weight.shape) for layer in conv_layers(cnn)] == [(4, 8, 3, 3), (6, 4, 3, 3)]
    assert [tuple(layer.weight.shape) for layer in linear_layers(cnn)] == [(16, 6 * 64), (N_ACTIONS, 16)]
    # ... (32, 64, 64) when none are given ...
    default = build_q_network((3, 10, 10), 4, (512,))
    assert isinstance(default, CNN) and default.n_flatten == 64 * 100
    assert [tuple(layer.weight.shape) for layer in conv_layers(default)] == [(32, 3, 3, 3), (64, 32, 3, 3), (64, 64, 3, 3)]
    assert [tuple(layer.weight.shape) for layer in linear_layers(default)] == [(512, 6400), (4, 512)]
    # ... and the "MLP on B" ablation with conv_channels = ().
    flat = build_q_network((7, 8, 8), N_ACTIONS, (16,), ())
    assert isinstance(flat, CNN) and conv_layers(flat) == [] and flat.n_flatten == 7 * 64
    assert [tuple(layer.weight.shape) for layer in linear_layers(flat)] == [(16, 7 * 64), (N_ACTIONS, 16)]
    # Any other shape: a ValueError that names the shape (no NotImplementedError "M6" any more).
    for shape in ((8, 8), (2, 3, 4, 5)):
        with pytest.raises(ValueError, match=re.escape(str(shape))):
            build_q_network(shape, N_ACTIONS, (16,))
        with pytest.raises(ValueError):
            build_q_network(shape, N_ACTIONS, (16,), (4,))


def grid_agent(seed: int = 0, in_shape: tuple[int, int, int] = (8, 8, 8), **kwargs) -> DQNAgent:
    """Agent on a grid observation with a small CNN (two layers of four
    channels, one hidden layer of 16), on the CPU."""
    settings = dict(conv_channels=(4, 4), hidden=(16,), lr=1e-2)
    settings.update(kwargs)
    return DQNAgent(in_shape, N_ACTIONS, cpu_cfg(**settings), seed=seed)


def test_cnn_agent_initial_weights_are_a_function_of_the_seed_alone():
    a = grid_agent(seed=5)
    assert a.obs_shape == (8, 8, 8) and isinstance(a.q_net, CNN) and isinstance(a.target_net, CNN)
    assert [tuple(layer.weight.shape) for layer in conv_layers(a.q_net)] == [(4, 8, 3, 3), (4, 4, 3, 3)]
    assert [tuple(layer.weight.shape) for layer in linear_layers(a.q_net)] == [(16, 256), (N_ACTIONS, 16)]
    assert same(weights(a.q_net), weights(a.target_net)) and a.target_net is not a.q_net
    (group,) = a.optimizer.param_groups
    assert [id(p) for p in group["params"]] == [id(p) for p in a.q_net.parameters()]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(999)
        torch.rand(17)  # wherever the global generator stands ...
        b = grid_agent(seed=5)
    assert same(weights(a.q_net), weights(b.q_net))  # ... seed 5 gives these weights
    others = [weights(grid_agent(seed=s).q_net) for s in (0, 6)]
    assert not any(same(weights(a.q_net), other) for other in others) and not same(others[0], others[1])
    # The convolutions are seeded too, not only the head, and nothing is zero or constant.
    assert not torch.equal(conv_layers(a.q_net)[0].weight, conv_layers(grid_agent(seed=0).q_net)[0].weight)
    for layer in conv_layers(a.q_net) + linear_layers(a.q_net):
        assert layer.weight.std() > 0.01


def test_constructing_a_cnn_agent_leaves_the_global_torch_rng_alone():
    before = torch.random.get_rng_state()
    grid_agent(seed=3)
    DQNAgent((3, 10, 10), 4, cpu_cfg(conv_channels=(), hidden=()), seed=8)
    DQNAgent((7, 8, 8), 4, cpu_cfg(conv_channels=(2,), hidden=(8, 8)), seed=8)
    assert torch.equal(torch.random.get_rng_state(), before)


# ------------------------------------------------------------- D2: replay buffer
TAG_SHAPE = (3,)


def tagged(ids, obs_shape: tuple[int, ...] = TAG_SHAPE) -> tuple[np.ndarray, ...]:
    """Transitions that carry their id ``t`` in every field: ``obs`` is all
    ``t``, ``action = t``, ``reward = t + 0.5``, ``next_obs`` all ``t + 0.25``,
    ``terminated = (t % 3 == 0)`` (exact in float32 for the ids used here)."""
    ids = np.asarray(list(ids), dtype=np.int64)
    obs = np.empty((len(ids), *obs_shape), dtype=np.float32)
    obs[...] = ids.astype(np.float32).reshape(-1, *[1] * len(obs_shape))
    return obs, ids.copy(), ids.astype(np.float32) + np.float32(0.5), obs + np.float32(0.25), ids % 3 == 0


def ids_of(batch: Batch) -> list[int]:
    """Ids of the tagged transitions in a batch; every row must be one intact transition."""
    assert isinstance(batch, Batch)
    for got, want in zip(arrays(batch), tagged(batch.actions, batch.obs.shape[1:])):
        assert got.shape == want.shape and np.array_equal(got, want)
    return batch.actions.tolist()


class ScriptedGenerator(np.random.Generator):
    """A generator whose ``integers`` returns scripted indices and records its
    calls: reads chosen storage slots through the sampling protocol of D2."""

    def __init__(self, indices) -> None:
        super().__init__(np.random.PCG64(0))
        self.indices = np.asarray(indices, dtype=np.int64)
        self.calls: list[tuple[int, int, int | None]] = []

    def integers(self, low, high=None, size=None, dtype=np.int64, endpoint=False):
        if high is None:
            low, high = 0, low
        self.calls.append((int(low), int(high), None if size is None else int(np.prod(size))))
        return self.indices.astype(dtype)


def stored(buffer: ReplayBuffer) -> Batch:
    """Every stored transition, by index ``0 .. len - 1`` of the sampling protocol."""
    n = len(buffer)
    rng = ScriptedGenerator(np.arange(n))
    batch = buffer.sample(n, rng)
    assert rng.calls == [(0, n, n)], rng.calls  # one draw: integers(0, len, size=batch_size)
    return batch


def test_replay_capacity_len_and_errors():
    buffer = ReplayBuffer(5, TAG_SHAPE)
    assert buffer.capacity == 5 and len(buffer) == 0
    with pytest.raises(ValueError):
        buffer.sample(4, np.random.default_rng(0))  # empty
    for capacity in (0, -1, -100):
        with pytest.raises(ValueError):
            ReplayBuffer(capacity, TAG_SHAPE)

    sizes = []
    for first, n in ((0, 2), (2, 1), (3, 2), (5, 3), (8, 1)):
        buffer.add(*tagged(range(first, first + n)))
        sizes.append(len(buffer))
    assert sizes == [2, 3, 5, 5, 5] and buffer.capacity == 5

    one = ReplayBuffer(1, TAG_SHAPE)  # the smallest buffer holds the latest transition
    one.add(*tagged([4]))
    one.add(*tagged([5, 6]))
    assert len(one) == 1 and ids_of(stored(one)) == [6]
    assert ids_of(one.sample(3, np.random.default_rng(0))) == [6, 6, 6]


@pytest.mark.parametrize("obs_shape", [(11,), (3, 8, 8)])
def test_sample_shapes_and_dtypes(obs_shape):
    rng = np.random.default_rng(0)
    n = 6
    buffer = ReplayBuffer(20, obs_shape)
    obs = rng.random((n, *obs_shape)).astype(np.float32)
    next_obs = rng.random((n, *obs_shape)).astype(np.float32)
    actions = rng.integers(0, 3, size=n)
    rewards = rng.normal(size=n).astype(np.float32)
    terminated = rng.random(n) < 0.5
    buffer.add(obs, actions, rewards, next_obs, terminated)

    everything = stored(buffer)
    assert isinstance(everything, Batch)
    for got, want in zip(arrays(everything), (obs, actions, rewards, next_obs, terminated)):
        assert isinstance(got, np.ndarray) and np.array_equal(got, want)

    for batch_size in (1, 4, 50):  # more than the buffer holds: with replacement
        batch = buffer.sample(batch_size, np.random.default_rng(batch_size))
        assert isinstance(batch, Batch)
        assert batch.obs.shape == (batch_size, *obs_shape) and batch.obs.dtype == np.float32
        assert batch.next_obs.shape == (batch_size, *obs_shape) and batch.next_obs.dtype == np.float32
        assert batch.actions.shape == (batch_size,) and batch.actions.dtype == np.int64
        assert batch.rewards.shape == (batch_size,) and batch.rewards.dtype == np.float32
        assert batch.terminated.shape == (batch_size,) and batch.terminated.dtype == np.bool_


def test_sample_dtypes_do_not_depend_on_the_dtypes_added():
    buffer = ReplayBuffer(4, TAG_SHAPE)
    obs, ids, rewards, next_obs, terminated = tagged(range(3))
    buffer.add(obs.astype(np.float64), ids.astype(np.int32), rewards.astype(np.float64), next_obs.astype(np.float64),
               terminated)
    batch = stored(buffer)
    assert ids_of(batch) == [0, 1, 2]
    assert [a.dtype for a in arrays(batch)] == [np.float32, np.int64, np.float32, np.float32, np.bool_]


def test_add_copies_its_arguments():
    buffer = ReplayBuffer(8, TAG_SHAPE)
    args = tagged(range(4))
    buffer.add(*args)
    for array in args:  # the caller scribbles over its arrays ...
        array[...] = 1
    assert ids_of(stored(buffer)) == [0, 1, 2, 3]
    for array, fresh in zip(args, tagged(range(4, 8))):  # ... and reuses them for the next step
        array[...] = fresh
    buffer.add(*args)
    for array in args:
        array[...] = 0
    assert ids_of(stored(buffer)) == [0, 1, 2, 3, 4, 5, 6, 7]


def test_add_accepts_one_array_as_obs_and_next_obs():
    buffer = ReplayBuffer(4, (2,))
    obs = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    buffer.add(obs, np.array([0, 1]), np.array([0.5, -1.0], dtype=np.float32), obs, np.array([False, True]))
    obs[:] = -7.0
    batch = stored(buffer)
    assert batch.obs.tolist() == [[1.0, 2.0], [3.0, 4.0]] and batch.next_obs.tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert not np.shares_memory(batch.obs, batch.next_obs)
    assert batch.actions.tolist() == [0, 1] and batch.rewards.tolist() == [0.5, -1.0]
    assert batch.terminated.tolist() == [False, True]


def test_replay_is_a_ring_transition_t_lives_in_slot_t_mod_capacity():
    """Adds of at most ``capacity`` transitions: the write position wraps and a
    batch that crosses the end of the storage is split."""
    capacity = 5
    buffer = ReplayBuffer(capacity, TAG_SHAPE)
    layout: dict[int, int] = {}
    t = 0
    for n in (3, 4, 1, 5, 2, 5, 4, 1, 1, 3):  # 3 + 4 wraps inside one add; 5 is a full lap
        ids = range(t, t + n)
        buffer.add(*tagged(ids))
        for i in ids:
            layout[i % capacity] = i
        t += n
        assert len(buffer) == min(t, capacity)
        assert ids_of(stored(buffer)) == [layout[slot] for slot in range(len(layout))], f"after {t} transitions"
    assert sorted(layout.values()) == list(range(t - capacity, t))  # the five most recent


def test_add_of_more_than_capacity_keeps_the_last_transitions():
    # Into an empty buffer ...
    buffer = ReplayBuffer(4, TAG_SHAPE)
    buffer.add(*tagged(range(10)))
    assert len(buffer) == 4 and sorted(ids_of(stored(buffer))) == [6, 7, 8, 9]
    # ... and FIFO order survives it: the next adds overwrite the oldest.
    buffer.add(*tagged([10, 11]))
    assert sorted(ids_of(stored(buffer))) == [8, 9, 10, 11]
    buffer.add(*tagged([12]))
    assert sorted(ids_of(stored(buffer))) == [9, 10, 11, 12]

    # Into a partly filled buffer, at an offset.
    buffer = ReplayBuffer(4, TAG_SHAPE)
    buffer.add(*tagged(range(3)))
    buffer.add(*tagged(range(3, 12)))
    assert len(buffer) == 4 and sorted(ids_of(stored(buffer))) == [8, 9, 10, 11]
    buffer.add(*tagged([12, 13, 14]))
    assert sorted(ids_of(stored(buffer))) == [11, 12, 13, 14]


def test_replay_holds_the_most_recent_transitions_of_any_stream():
    """Against ``collections.deque(maxlen=capacity)``: batch sizes below, at and
    above the capacity, in any mix."""
    capacity = 7
    rng = np.random.default_rng(3)
    buffer, model = ReplayBuffer(capacity, TAG_SHAPE), deque(maxlen=capacity)
    t = 0
    sizes = []
    for _ in range(150):
        n = int(rng.integers(1, 2 * capacity + 2))
        ids = range(t, t + n)
        t += n
        sizes.append(n)
        buffer.add(*tagged(ids))
        model.extend(ids)
        assert len(buffer) == len(model)
        assert sorted(ids_of(stored(buffer))) == list(model), f"after {t} transitions"
    assert min(sizes) == 1 and capacity in sizes and max(sizes) > capacity


def test_add_rejects_inconsistent_shapes():
    buffer = ReplayBuffer(8, (OBS_DIM,))

    def good() -> dict[str, np.ndarray]:
        return dict(
            obs=np.zeros((3, OBS_DIM), dtype=np.float32),
            actions=np.zeros(3, dtype=np.int64),
            rewards=np.zeros(3, dtype=np.float32),
            next_obs=np.zeros((3, OBS_DIM), dtype=np.float32),
            terminated=np.zeros(3, dtype=bool),
        )

    wrong = {
        "observation shape": dict(obs=np.zeros((3, 10), np.float32), next_obs=np.zeros((3, 10), np.float32)),
        "next_obs shape": dict(next_obs=np.zeros((3, 12), np.float32)),
        "next_obs rows": dict(next_obs=np.zeros((2, OBS_DIM), np.float32)),
        "actions rows": dict(actions=np.zeros(2, np.int64)),
        "rewards rows": dict(rewards=np.zeros(4, np.float32)),
        "terminated rows": dict(terminated=np.zeros(2, bool)),
    }
    for what, change in wrong.items():
        with pytest.raises(ValueError):
            buffer.add(**{**good(), **change})
        assert len(buffer) == 0, what
    buffer.add(**good())
    assert len(buffer) == 3


@pytest.mark.parametrize("batch_size", [1, 32, 500])
def test_sample_follows_the_rng_protocol(batch_size):
    """``idx = rng.integers(0, len(buffer), size=batch_size)``, one call per
    ``sample``: the batch is the twin generator's indices and both generators
    end in the same state."""
    # While filling: index i is the i-th transition added.
    buffer = ReplayBuffer(50, TAG_SHAPE)
    buffer.add(*tagged(range(4)))
    buffer.add(*tagged(range(4, 7)))
    rng, twin = np.random.default_rng(5), np.random.default_rng(5)
    for _ in range(3):
        batch = buffer.sample(batch_size, rng)
        idx = twin.integers(0, 7, size=batch_size)
        assert ids_of(batch) == idx.tolist()
        assert rng.bit_generator.state == twin.bit_generator.state

    # Full and wrapped: index = storage slot (ids 5 6 2 3 4 after 3 + 4 adds into 5 slots).
    ring = ReplayBuffer(5, TAG_SHAPE)
    ring.add(*tagged(range(3)))
    ring.add(*tagged(range(3, 7)))
    slots = np.array([5, 6, 2, 3, 4])
    for _ in range(3):
        batch = ring.sample(batch_size, rng)
        idx = twin.integers(0, 5, size=batch_size)
        assert ids_of(batch) == slots[idx].tolist()
        assert rng.bit_generator.state == twin.bit_generator.state


def test_sample_draws_only_stored_transitions_while_filling():
    buffer = ReplayBuffer(1000, TAG_SHAPE)
    buffer.add(*tagged(range(1, 8)))  # ids 1 .. 7: an unwritten slot (all zeros) is not a valid transition
    ids = ids_of(buffer.sample(3000, np.random.default_rng(0)))
    assert set(ids) == set(range(1, 8))
    # With replacement: a batch larger than the buffer, and repeats inside a small one.
    small = ids_of(buffer.sample(6, np.random.default_rng(1)))
    assert len(small) == 6 and len(set(small)) < 6
    assert np.array_equal(small, np.arange(1, 8)[np.random.default_rng(1).integers(0, 7, size=6)])


def test_sample_is_uniform_over_the_stored_transitions():
    buffer = ReplayBuffer(10, TAG_SHAPE)
    for first in (0, 6, 12):  # 18 transitions through 10 slots: ids 8 .. 17 remain
        buffer.add(*tagged(range(first, first + 6)))
    n = 40_000
    ids = np.array(ids_of(buffer.sample(n, np.random.default_rng(11))))
    assert set(ids.tolist()) == set(range(8, 18))
    frequency = np.bincount(ids - 8, minlength=10) / n
    assert np.abs(frequency - 0.1).max() < 0.007, frequency  # 0.0015 standard error


def test_sample_returns_fresh_copies():
    buffer = ReplayBuffer(8, TAG_SHAPE)
    buffer.add(*tagged(range(8)))
    first = buffer.sample(16, np.random.default_rng(1))
    kept = [array.copy() for array in arrays(first)]
    second = buffer.sample(16, np.random.default_rng(2))  # a later sample leaves the earlier batch alone
    for array, copy_ in zip(arrays(first), kept):
        assert np.array_equal(array, copy_)
    for a in arrays(first):
        for b in arrays(second):
            assert not np.shares_memory(a, b)
    assert not np.shares_memory(first.obs, first.next_obs)

    for array in arrays(first):  # writing into a batch does not reach the storage
        array[...] = 1
    again = buffer.sample(16, np.random.default_rng(1))
    for array, copy_ in zip(arrays(again), kept):
        assert np.array_equal(array, copy_)
    assert ids_of(stored(buffer)) == list(range(8))


#: Starvation limit of the hand-built truncation (the scenario of the tabular oracle).
TRUNCATION_H = 4


def starvation_truncation() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One real ``VecSnakeEnv`` step that ends by starvation truncation.

    Snake (1, 2) -> (1, 4) heading right, food far away, ``H - 1`` steps since
    the last food; turning left moves the head to (0, 4): alive, heading up
    with the wall ahead. Returns ``(obs, actions, reward, final_obs,
    terminated, reset_obs)``.
    """
    vec = VecSnakeEnv(EnvConfig(W=8, starve_limit=TRUNCATION_H), n_envs=1, seed=3)
    vec.reset()
    vec.set_state(0, [(1, 2), (1, 3), (1, 4)], FAR_FOOD, steps_since_food=TRUNCATION_H - 1)
    obs = vec.observe()
    actions = np.array([TURN_LEFT])
    reset_obs, reward, terminated, truncated, info = vec.step(actions)
    assert truncated.tolist() == [True] and terminated.tolist() == [False] and reward.tolist() == [0.0]
    final_obs = info["final_obs"]
    # The true successor sees the wall ahead; the first observation of the next episode does not.
    assert final_obs[0, A_DANGER_STRAIGHT] == 1.0 and reset_obs[0, A_DANGER_STRAIGHT] == 0.0
    assert final_obs[0, A_HEADING.start + RIGHT] == 0.0 and reset_obs[0, A_HEADING.start + RIGHT] == 1.0
    return obs, actions, reward, final_obs, terminated, reset_obs


def test_replay_keeps_final_obs_and_terminated_false_of_a_starvation_truncation():
    obs, actions, reward, final_obs, terminated, reset_obs = starvation_truncation()
    buffer = ReplayBuffer(4, (OBS_DIM,))
    buffer.add(obs, actions, reward, final_obs, terminated)
    batch = stored(buffer)
    assert np.array_equal(batch.obs, obs) and batch.actions.tolist() == [TURN_LEFT]
    assert batch.rewards.tolist() == [0.0]
    assert np.array_equal(batch.next_obs, final_obs) and not np.array_equal(batch.next_obs, reset_obs)
    assert batch.terminated.tolist() == [False] and batch.terminated.dtype == np.bool_


# ------------------------------------------------------------- D3: construction
def test_new_agent_attributes():
    cfg = cpu_cfg(hidden=(16, 8), lr=3e-3, gamma=0.9)
    agent = DQNAgent((OBS_DIM,), N_ACTIONS, cfg, seed=4)
    assert agent.cfg == cfg and agent.obs_shape == (OBS_DIM,) and agent.n_actions == N_ACTIONS
    assert agent.grad_steps == 0 and type(agent.grad_steps) is int
    assert isinstance(agent.device, torch.device) and agent.device == torch.device("cpu")
    assert isinstance(agent.q_net, MLP) and isinstance(agent.target_net, torch.nn.Module)
    assert [tuple(layer.weight.shape) for layer in linear_layers(agent.q_net)] == [(16, 11), (8, 16), (3, 8)]
    for net in (agent.q_net, agent.target_net):
        assert all(p.device.type == "cpu" and p.dtype == torch.float32 for p in net.parameters())

    # Adam on the online network only, with the configured step size and torch's defaults.
    assert isinstance(agent.optimizer, torch.optim.Adam)
    (group,) = agent.optimizer.param_groups
    assert group["lr"] == 3e-3 and tuple(group["betas"]) == (0.9, 0.999) and group["eps"] == 1e-8
    assert group["weight_decay"] == 0 and not group["amsgrad"]
    assert [id(p) for p in group["params"]] == [id(p) for p in agent.q_net.parameters()]


def test_target_network_starts_as_a_separate_exact_copy():
    agent = small_agent(seed=2)
    assert agent.target_net is not agent.q_net
    assert same(weights(agent.q_net), weights(agent.target_net))
    online, target = list(agent.q_net.parameters()), list(agent.target_net.parameters())
    assert len(online) == len(target) == 6
    for p, q in zip(online, target):
        assert p is not q and p.data_ptr() != q.data_ptr()
    with torch.no_grad():  # writing to one network does not reach the other
        online[0].add_(1.0)
    assert not same(weights(agent.q_net), weights(agent.target_net))


def test_initial_weights_are_a_function_of_the_seed_alone():
    a = small_agent(seed=5)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(999)
        torch.rand(17)  # wherever the global generator stands ...
        b = small_agent(seed=5)
    assert same(weights(a.q_net), weights(b.q_net))  # ... seed 5 gives these weights
    assert same(weights(a.target_net), weights(b.target_net))
    others = [weights(small_agent(seed=s).q_net) for s in (0, 6, 7)]
    assert not any(same(weights(a.q_net), other) for other in others)
    assert not same(others[0], others[1])
    # A random initialisation, not zeros or constants.
    for layer in linear_layers(a.q_net):
        assert layer.weight.std() > 0.01


def test_constructing_an_agent_leaves_the_global_torch_rng_alone():
    before = torch.random.get_rng_state()
    small_agent(seed=3)
    DQNAgent((OBS_DIM,), 4, cpu_cfg(hidden=()), seed=8)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_policy_protocol():
    agent = small_agent()
    assert isinstance(agent, Policy)
    assert agent.name == "dqn" and DQNAgent.name == "dqn"
    assert agent.reset() is None
    assert same(weights(agent.q_net), weights(small_agent().q_net))  # reset is a no-op


def test_untrained_agent_plays_through_both_evaluation_protocols():
    """``act`` on the scalar env and ``act_batch`` on the vectorised env play the same episodes."""
    cfg = EnvConfig(W=8, starve_limit=30)
    agent = small_agent(seed=1)
    scalar = evaluate(agent, cfg, n_episodes=10, seed=DEFAULT_EVAL_SEED)
    batched = evaluate_batched(agent, cfg, n_episodes=10, seed=DEFAULT_EVAL_SEED)
    assert scalar.policy == batched.policy == "dqn"
    assert len(scalar.episodes) == 10 and scalar.episodes == batched.episodes


# ------------------------------------------------------------- D3: action selection
def test_q_values_are_the_online_network():
    agent = small_agent(seed=3)
    randomise(agent.q_net, seed=1)
    randomise(agent.target_net, seed=2)  # a target network that differs: q_values must not read it
    obs = random_obs(np.random.default_rng(0), 40)
    q = agent.q_values(obs)
    assert isinstance(q, np.ndarray) and q.shape == (40, N_ACTIONS) and q.dtype == np.float32
    want = numpy_forward(agent.q_net, obs)
    scale = np.abs(want).max()
    assert np.allclose(q, want, rtol=1e-5, atol=1e-5 * scale)
    assert np.abs(q - numpy_forward(agent.target_net, obs)).max() > 0.1 * scale
    one = agent.q_values(obs[:1])
    assert one.shape == (1, N_ACTIONS) and np.allclose(one, q[:1], rtol=1e-5, atol=1e-5 * scale)


@pytest.mark.parametrize("n_actions", [3, 4])
def test_act_agrees_with_act_batch_row_by_row(n_actions):
    agent = DQNAgent((OBS_DIM,), n_actions, cpu_cfg(hidden=(16, 16)), seed=n_actions)
    randomise(agent.q_net, seed=n_actions)
    obs = random_obs(np.random.default_rng(1), 200)
    batch = agent.act_batch(obs)
    assert isinstance(batch, np.ndarray) and batch.shape == (200,) and batch.dtype == np.int64
    assert np.array_equal(batch, numpy_forward(agent.q_net, obs).argmax(axis=1))
    assert np.array_equal(batch, agent.q_values(obs).argmax(axis=1))
    assert len(set(batch.tolist())) > 1  # the greedy action depends on the observation
    for i in range(200):
        action = agent.act(obs[i])
        assert type(action) is int and action == batch[i], f"row {i}"
    assert agent.act(obs[0], None) == agent.act(obs[0], env=None) == batch[0]
    one = agent.act_batch(obs[7:8])
    assert one.shape == (1,) and one.dtype == np.int64 and one[0] == batch[7]


def tie_agent() -> tuple[DQNAgent, np.ndarray, list[int]]:
    """Linear Q-function with ``Q(e_k) = column k`` (zero bias): exact ties.
    Returns the agent, the five one-hot observations and the lowest best action of each."""
    columns = [
        ([1.0, 1.0, 0.0, 0.0], 0),
        ([0.0, 2.0, 2.0, 2.0], 1),
        ([-1.0, -1.0, -1.0, -1.0], 0),
        ([0.0, 1.0, 3.0, 3.0], 2),
        ([7.0, 0.0, 0.0, 7.0], 0),
    ]
    agent = DQNAgent((5,), 4, cpu_cfg(hidden=()), seed=0)
    set_linear(agent.q_net, np.array([column for column, _ in columns]).T, np.zeros(4))
    return agent, np.eye(5, dtype=np.float32), [best for _, best in columns]


def test_ties_pick_the_lowest_index():
    agent, obs, expected = tie_agent()
    assert agent.q_values(obs).T.tolist() == [[1, 0, -1, 0, 7], [1, 2, -1, 1, 0], [0, 2, -1, 3, 0], [0, 2, -1, 3, 7]]
    assert agent.act_batch(obs).tolist() == expected
    assert [agent.act(row) for row in obs] == expected
    assert agent.act_epsilon(obs, 0.0, np.random.default_rng(0)).tolist() == expected
    # All action values equal (a zero network): action 0 everywhere.
    flat = DQNAgent((OBS_DIM,), N_ACTIONS, cpu_cfg(hidden=()), seed=0)
    set_linear(flat.q_net, np.zeros((N_ACTIONS, OBS_DIM)), np.full(N_ACTIONS, 0.5))
    some = random_obs(np.random.default_rng(2), 50)
    assert not flat.act_batch(some).any() and flat.act(some[0]) == 0


def protocol_actions(greedy: np.ndarray, epsilon: float, n_actions: int, rng: np.random.Generator) -> np.ndarray:
    """The RNG protocol of ``act_epsilon``, replicated: two draws, in this order, on every call."""
    explore = rng.random(len(greedy)) < epsilon
    random_actions = rng.integers(0, n_actions, size=len(greedy))
    return np.where(explore, random_actions, greedy)


@pytest.mark.parametrize("n_actions", [3, 4])
@pytest.mark.parametrize("epsilon", [0.0, 0.05, 0.3, 1.0])
def test_act_epsilon_follows_the_rng_protocol(n_actions, epsilon):
    agent = DQNAgent((OBS_DIM,), n_actions, cpu_cfg(hidden=(16, 16)), seed=20)
    randomise(agent.q_net, seed=20)
    rng, twin = np.random.default_rng(99), np.random.default_rng(99)
    obs_rng = np.random.default_rng(0)
    for n in (1, 64, 1, 7, 300):  # the stream is shared by consecutive calls of any size
        obs = random_obs(obs_rng, n)
        greedy = numpy_forward(agent.q_net, obs).argmax(axis=1)
        actions = agent.act_epsilon(obs, epsilon, rng)
        assert isinstance(actions, np.ndarray) and actions.shape == (n,) and actions.dtype == np.int64
        assert np.array_equal(actions, protocol_actions(greedy, epsilon, n_actions, twin)), f"batch of {n}"
        assert rng.bit_generator.state == twin.bit_generator.state, f"batch of {n}"


def test_act_epsilon_consumes_the_same_stream_whatever_epsilon():
    obs = random_obs(np.random.default_rng(0), 64)
    agent = small_agent(seed=30)
    states = []
    for epsilon in (0.0, 0.05, 0.5, 1.0):
        rng = np.random.default_rng(1234)
        for _ in range(3):
            agent.act_epsilon(obs, epsilon, rng)
        states.append((rng.bit_generator.state, rng.random(4).tolist()))
    assert all(state == states[0] for state in states[1:])
    assert states[0][0] != np.random.default_rng(1234).bit_generator.state  # both draws happened at epsilon 0


def test_act_epsilon_zero_is_greedy_and_one_is_uniform():
    agent = small_agent(seed=9)
    randomise(agent.q_net, seed=9)
    obs = random_obs(np.random.default_rng(3), 500)
    greedy_actions = agent.act_epsilon(obs, 0.0, np.random.default_rng(3))
    assert np.array_equal(greedy_actions, numpy_forward(agent.q_net, obs).argmax(axis=1))
    assert np.array_equal(greedy_actions, agent.act_batch(obs)) and len(set(greedy_actions.tolist())) == N_ACTIONS

    n = 30_000
    obs = np.repeat(obs[:1], n, axis=0)
    greedy = agent.act(obs[0])
    actions = agent.act_epsilon(obs, 1.0, np.random.default_rng(42))
    frequency = np.bincount(actions, minlength=N_ACTIONS) / n
    assert np.abs(frequency - 1.0 / N_ACTIONS).max() < 0.01, frequency
    rng = np.random.default_rng(42)  # exactly the second draw of the protocol
    rng.random(n)
    assert np.array_equal(actions, rng.integers(0, N_ACTIONS, size=n))

    # epsilon = 0.3: P(greedy) = 1 - 0.3 + 0.3 / 3 = 0.8, P(each other action) = 0.1.
    actions = agent.act_epsilon(obs, 0.3, np.random.default_rng(7))
    frequency = np.bincount(actions, minlength=N_ACTIONS) / n
    want = np.full(N_ACTIONS, 0.1)
    want[greedy] = 0.8
    assert np.abs(frequency - want).max() < 0.01, frequency


def test_acting_creates_no_gradients_and_changes_nothing():
    agent = small_agent(seed=6)
    online, target = weights(agent.q_net), weights(agent.target_net)
    obs = random_obs(np.random.default_rng(0), 32)
    obs_before = obs.copy()
    agent.q_values(obs)
    agent.act(obs[0])
    agent.act_batch(obs)
    agent.act_epsilon(obs, 0.5, np.random.default_rng(0))
    assert all(p.grad is None for p in agent.q_net.parameters())
    assert all(p.grad is None for p in agent.target_net.parameters())
    assert all(p.requires_grad for p in agent.q_net.parameters())  # still trainable
    assert torch.is_grad_enabled()  # no global switch left behind
    assert same(weights(agent.q_net), online) and same(weights(agent.target_net), target)
    assert agent.grad_steps == 0 and np.array_equal(obs, obs_before)


# ------------------------------------------------------------- D3: update on hand-built transitions
# A linear Q-function on two features, three actions, gamma = 0.5 and dyadic
# numbers: every value below is exact in float32.
#
#     Q_online(s) = [s0, s1 + 0.5, s0 + s1 - 1]
#     Q_target(s) = [2 s1 + 1, s0 - s1, (s0 + s1) / 2]
W_ONLINE, B_ONLINE = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [0.0, 0.5, -1.0]
W_TARGET, B_TARGET = [[0.0, 2.0], [1.0, -1.0], [0.5, 0.5]], [1.0, 0.0, 0.0]
HAND_GAMMA = 0.5
HAND_LR = 0.125

#:  n  s       a  r      s'      terminated   q_sa  Q_target(s')   Q_online(s')    target   double target
#:  0  (1, 2)  1  0.5    (3, 0)  no           2.5   [1, 3, 1.5]    [3, 0.5, 2]     2.0      0.5 + 0.5 * 1  =  1.0
#:  1  (2, 0)  0  -1     (4, 4)  yes          2.0   [9, 0, 4]      [4, 4.5, 7]     -1.0     -1.0
#:  2  (0, 1)  2  0.5    (1, 3)  no (*)       0.0   [7, -2, 2]     [1, 3.5, 3]     4.0      0.5 + 0.5 * -2 = -0.5
#:  3  (1, 1)  0  0.25   (0, 0)  no           1.0   [1, 0, 0]      [0, 0.5, -1]    0.75     0.25 + 0.5 * 0 =  0.25
#: (*) stands for a starvation truncation: the episode ended, ``terminated`` is False, the target bootstraps.
HAND_BATCH = Batch(
    obs=np.array([[1, 2], [2, 0], [0, 1], [1, 1]], dtype=np.float32),
    actions=np.array([1, 0, 2, 0], dtype=np.int64),
    rewards=np.array([0.5, -1.0, 0.5, 0.25], dtype=np.float32),
    next_obs=np.array([[3, 0], [4, 4], [1, 3], [0, 0]], dtype=np.float32),
    terminated=np.array([False, True, False, False]),
)
HAND_Q = [2.5, 2.0, 0.0, 1.0]
#: Per variant: the targets, and the gradient of the mean Huber loss.
#: td = q_sa - target; d loss / d q_sa = clip(td, -1, 1) / 4; d q_sa / d W[a] = s, d q_sa / d b[a] = 1.
HAND = {
    False: dict(
        targets=[2.0, -1.0, 4.0, 0.75],
        # td = [0.5, 3, -4, 0.25] -> Huber [0.125, 2.5, 3.5, 0.03125], d loss / d q = [0.125, 0.25, -0.25, 0.0625]
        loss=(0.125 + 2.5 + 3.5 + 0.03125) / 4,
        grad_w=[[0.25 * 2 + 0.0625 * 1, 0.0625 * 1], [0.125 * 1, 0.125 * 2], [0.0, -0.25 * 1]],
        grad_b=[0.25 + 0.0625, 0.125, -0.25],
    ),
    True: dict(
        targets=[1.0, -1.0, -0.5, 0.25],
        # td = [1.5, 3, 0.5, 0.75] -> Huber [1.0, 2.5, 0.125, 0.28125], d loss / d q = [0.25, 0.25, 0.125, 0.1875]
        loss=(1.0 + 2.5 + 0.125 + 0.28125) / 4,
        grad_w=[[0.25 * 2 + 0.1875 * 1, 0.1875 * 1], [0.25 * 1, 0.25 * 2], [0.0, 0.125 * 1]],
        grad_b=[0.25 + 0.1875, 0.25, 0.125],
    ),
}


def hand_agent(double: bool = False, **overrides) -> DQNAgent:
    """The linear Q-functions above; the target network differs from the online one."""
    settings = dict(gamma=HAND_GAMMA, lr=HAND_LR, hidden=(), double=double)
    settings.update(overrides)
    agent = DQNAgent((2,), 3, cpu_cfg(**settings), seed=0)
    set_linear(agent.q_net, W_ONLINE, B_ONLINE)
    set_linear(agent.target_net, W_TARGET, B_TARGET)
    return agent


def huber(td: float) -> float:
    return 0.5 * td * td if abs(td) < 1.0 else abs(td) - 0.5


def consumed_gradient(agent: DQNAgent) -> tuple[np.ndarray, np.ndarray]:
    """Gradient (weight, bias) the optimiser consumed in its FIRST step on a
    linear Q-function: Adam's first moment is ``(1 - beta1) * g`` then."""
    (layer,) = linear_layers(agent.q_net)
    beta1 = agent.optimizer.param_groups[0]["betas"][0]
    state = agent.optimizer.state
    weight, bias = (
        state[p]["exp_avg"].detach().cpu().numpy().astype(np.float64) / (1.0 - beta1) for p in (layer.weight, layer.bias)
    )
    return weight, bias


def hand_rows(rows: list[int], terminated: list[bool] | None = None) -> Batch:
    """Some rows of ``HAND_BATCH``, optionally with other terminated flags."""
    batch = Batch(*(array[rows].copy() for array in arrays(HAND_BATCH)))
    if terminated is not None:
        batch = Batch(batch.obs, batch.actions, batch.rewards, batch.next_obs, np.array(terminated, dtype=bool))
    return batch


def test_hand_built_q_functions_are_what_the_comments_say():
    agent = hand_agent()
    assert agent.q_values(HAND_BATCH.obs)[np.arange(4), HAND_BATCH.actions].tolist() == HAND_Q
    assert agent.q_values(HAND_BATCH.next_obs).tolist() == [[3, 0.5, 2], [4, 4.5, 7], [1, 3.5, 3], [0, 0.5, -1]]
    assert numpy_forward(agent.target_net, HAND_BATCH.next_obs).tolist() == [
        [1, 3, 1.5], [9, 0, 4], [7, -2, 2], [1, 0, 0]]


@pytest.mark.parametrize("double", [False, True], ids=["dqn", "double"])
@pytest.mark.parametrize("row", [0, 1, 2, 3], ids=["non-terminal", "terminal", "truncated", "non-terminal-quadratic"])
def test_update_target_of_one_transition(double, row):
    agent = hand_agent(double=double)
    out = agent.update(hand_rows([row]))
    target = HAND[double]["targets"][row]
    assert out["mean_q"] == pytest.approx(HAND_Q[row], rel=1e-6, abs=1e-7)
    assert out["mean_target"] == pytest.approx(target, rel=1e-6, abs=1e-7)
    assert out["loss"] == pytest.approx(huber(HAND_Q[row] - target), rel=1e-6, abs=1e-7)


@pytest.mark.parametrize("double", [False, True], ids=["dqn", "double"])
def test_update_terminated_flag_alone_decides_the_bootstrap(double):
    # Row 1 not terminated: -1 + 0.5 * max [9, 0, 4] = 3.5; double: argmax [4, 4.5, 7] = 2 -> -1 + 0.5 * 4 = 1.
    out = hand_agent(double=double).update(hand_rows([1], terminated=[False]))
    assert out["mean_target"] == pytest.approx(1.0 if double else 3.5, rel=1e-6)
    # Row 2 terminated: the reward alone, whatever the networks say about s'.
    out = hand_agent(double=double).update(hand_rows([2], terminated=[True]))
    assert out["mean_target"] == pytest.approx(0.5, rel=1e-6)
    assert out["loss"] == pytest.approx(huber(0.0 - 0.5), rel=1e-6)
    # Per transition, inside one batch.
    out = hand_agent(double=double).update(hand_rows([1, 2, 1, 2], terminated=[False, True, True, False]))
    targets = [1.0 if double else 3.5, 0.5, -1.0, HAND[double]["targets"][2]]
    assert out["mean_target"] == pytest.approx(sum(targets) / 4, rel=1e-6)


@pytest.mark.parametrize("double", [False, True], ids=["dqn", "double"])
def test_update_matches_hand_computed_numbers(double):
    """Targets, Huber loss (rows 0 and 3 quadratic, rows 1 and 2 linear), the
    gradient against the manual derivation, and Adam's first step."""
    agent = hand_agent(double=double)
    want = HAND[double]
    out = agent.update(HAND_BATCH)

    assert set(out) == STAT_KEYS and all(type(v) is float for v in out.values())
    assert out["mean_q"] == pytest.approx(sum(HAND_Q) / 4, rel=1e-6)  # 1.375
    assert out["mean_target"] == pytest.approx(sum(want["targets"]) / 4, rel=1e-6, abs=1e-7)
    assert out["loss"] == pytest.approx(want["loss"], rel=1e-6)
    assert agent.grad_steps == 1

    grad_w, grad_b = consumed_gradient(agent)
    assert np.allclose(grad_w, want["grad_w"], rtol=1e-5, atol=1e-7)
    assert np.allclose(grad_b, want["grad_b"], rtol=1e-5, atol=1e-7)
    norm = math.sqrt(sum(g * g for g in np.ravel(want["grad_w"]).tolist() + list(want["grad_b"])))
    assert norm < agent.cfg.grad_clip  # not clipped: the optimiser consumed the gradient itself
    assert out["grad_norm"] == pytest.approx(norm, rel=1e-5)

    # Adam's first step moves every weight by lr against the sign of its
    # gradient (m / sqrt(v) = g / |g|); a zero gradient moves nothing.
    (layer,) = linear_layers(agent.q_net)
    assert np.allclose(layer.weight.detach().numpy(), W_ONLINE - HAND_LR * np.sign(want["grad_w"]), rtol=1e-5, atol=1e-6)
    assert np.allclose(layer.bias.detach().numpy(), B_ONLINE - HAND_LR * np.sign(want["grad_b"]), rtol=1e-5, atol=1e-6)
    assert layer.weight[2, 0].item() == 1.0  # d loss / d W[2, 0] = 0

    # The target network: untouched, and no gradient reached it.
    (target_layer,) = linear_layers(agent.target_net)
    assert target_layer.weight.tolist() == W_TARGET and target_layer.bias.tolist() == B_TARGET
    assert all(p.grad is None or not p.grad.any() for p in agent.target_net.parameters())


def test_double_dqn_target_differs_from_the_plain_target():
    plain, double = hand_agent(double=False).update(HAND_BATCH), hand_agent(double=True).update(HAND_BATCH)
    assert plain["mean_q"] == double["mean_q"] == 1.375
    assert plain["mean_target"] == pytest.approx(1.4375) and double["mean_target"] == pytest.approx(-0.0625)
    assert plain["loss"] == pytest.approx(1.5390625) and double["loss"] == pytest.approx(0.9765625)
    # Double DQN selects with the online network and evaluates with the target
    # network, so its bootstrap value never exceeds the plain max.
    assert all(d <= p for d, p in zip(HAND[True]["targets"], HAND[False]["targets"]))


def test_update_gradient_matches_autograd_on_the_written_out_loss():
    """Independent of the agent's code path: the loss as a formula of the
    weights, differentiated by torch, with the target a constant."""
    w = torch.tensor(W_ONLINE, requires_grad=True)
    b = torch.tensor(B_ONLINE, requires_grad=True)
    s, a = torch.from_numpy(HAND_BATCH.obs), torch.from_numpy(HAND_BATCH.actions)
    q_sa = (s @ w.T + b)[torch.arange(4), a]
    td = q_sa - torch.tensor(HAND[False]["targets"])
    loss = torch.where(td.abs() < 1.0, 0.5 * td * td, td.abs() - 0.5).mean()
    grad_w, grad_b = torch.autograd.grad(loss, (w, b))
    assert loss.item() == HAND[False]["loss"]
    assert grad_w.tolist() == HAND[False]["grad_w"] and grad_b.tolist() == HAND[False]["grad_b"]

    agent = hand_agent()
    agent.update(HAND_BATCH)
    got_w, got_b = consumed_gradient(agent)
    assert np.allclose(got_w, grad_w.numpy(), rtol=1e-5, atol=1e-7)
    assert np.allclose(got_b, grad_b.numpy(), rtol=1e-5, atol=1e-7)


def test_no_gradient_flows_through_the_target():
    """With the target network equal to the online network (a fresh agent) the
    semi-gradient still differentiates ``q_sa`` only. For one non-terminal
    transition with td > 1 it is ``s`` on row ``a``; a gradient through
    ``gamma * max Q(s')`` would subtract ``gamma * s'`` on the greedy row."""
    agent = DQNAgent((2,), 3, cpu_cfg(gamma=HAND_GAMMA, lr=HAND_LR, hidden=()), seed=0)
    set_linear(agent.q_net, W_ONLINE, B_ONLINE)
    set_linear(agent.target_net, W_ONLINE, B_ONLINE)
    # s = (4, 1), a = 0: q_sa = 4. s' = (1, 3): Q = [1, 3.5, 3] -> target = 0 + 0.5 * 3.5 = 1.75, td = 2.25.
    batch = Batch(
        obs=np.array([[4, 1]], dtype=np.float32),
        actions=np.array([0], dtype=np.int64),
        rewards=np.array([0.0], dtype=np.float32),
        next_obs=np.array([[1, 3]], dtype=np.float32),
        terminated=np.array([False]),
    )
    out = agent.update(batch)
    assert out["mean_target"] == pytest.approx(1.75) and out["loss"] == pytest.approx(1.75)
    grad_w, grad_b = consumed_gradient(agent)
    assert np.allclose(grad_w, [[4, 1], [0, 0], [0, 0]], rtol=1e-5, atol=1e-7)  # row 1 would be -0.5 * (1, 3)
    assert np.allclose(grad_b, [1, 0, 0], rtol=1e-5, atol=1e-7)
    assert out["grad_norm"] == pytest.approx(math.sqrt(18.0), rel=1e-5)
    assert all(p.grad is None or not p.grad.any() for p in agent.target_net.parameters())
    (target_layer,) = linear_layers(agent.target_net)
    assert target_layer.weight.tolist() == W_ONLINE and target_layer.bias.tolist() == B_ONLINE


#: One terminal transition with a large gradient: s = (30, 40), a = 0, r = 0.
#: q_sa = 30, target = 0 -> Huber is linear (loss 29.5), d loss / d q_sa = 1,
#: gradient = (30, 40) on W[0] and 1 on b[0]: global norm sqrt(2501) = 50.01.
BIG_BATCH = Batch(
    obs=np.array([[30, 40]], dtype=np.float32),
    actions=np.array([0], dtype=np.int64),
    rewards=np.array([0.0], dtype=np.float32),
    next_obs=np.array([[0, 0]], dtype=np.float32),
    terminated=np.array([True]),
)
BIG_NORM = math.sqrt(30.0**2 + 40.0**2 + 1.0)


@pytest.mark.parametrize("grad_clip", [10.0, 2.0, 0.5, 100.0])
def test_gradient_norm_is_reported_before_clipping_and_the_step_uses_the_clipped_gradient(grad_clip):
    agent = hand_agent(grad_clip=grad_clip)
    out = agent.update(BIG_BATCH)
    assert out["loss"] == pytest.approx(29.5) and out["mean_q"] == pytest.approx(30.0) and out["mean_target"] == 0.0
    assert out["grad_norm"] == pytest.approx(BIG_NORM, rel=1e-5)  # the norm BEFORE clipping

    grad_w, grad_b = consumed_gradient(agent)
    scale = min(1.0, grad_clip / BIG_NORM)  # the direction is kept; never scaled up
    assert np.allclose(grad_w, scale * np.array([[30, 40], [0, 0], [0, 0]]), rtol=1e-4, atol=1e-7)
    assert np.allclose(grad_b, scale * np.array([1, 0, 0]), rtol=1e-4, atol=1e-7)
    consumed_norm = math.sqrt((grad_w**2).sum() + (grad_b**2).sum())
    assert consumed_norm == pytest.approx(min(grad_clip, BIG_NORM), rel=1e-4)


# ------------------------------------------------------------- D3: target network
@pytest.mark.parametrize("every", [1, 3, 4])
def test_target_network_is_synced_exactly_every_k_gradient_steps(every):
    agent = small_agent(seed=1, target_update_every=every)
    rng = np.random.default_rng(every)
    last_sync = weights(agent.target_net)
    for step in range(1, 10):
        before = weights(agent.q_net)
        agent.update(random_batch(rng))
        assert agent.grad_steps == step
        online, target = weights(agent.q_net), weights(agent.target_net)
        assert not same(online, before), f"step {step}: the online network did not move"
        if step % every == 0:
            assert same(target, online), f"step {step}: no sync"
            last_sync = target
        else:
            assert same(target, last_sync), f"step {step}: the target network moved between syncs"
            assert not same(target, online)
    # A sync copies values: the networks stay two objects with their own memory.
    assert agent.target_net is not agent.q_net
    for p, q in zip(agent.q_net.parameters(), agent.target_net.parameters()):
        assert p is not q and p.data_ptr() != q.data_ptr()


def test_update_uses_the_target_network_as_it_was_before_the_sync():
    """``target_update_every = 1``: the copy happens after the step, so every
    update bootstraps from the online weights of the previous step."""
    agent = hand_agent(target_update_every=1)  # target = the hand-set Q_target for the first update only
    first = agent.update(HAND_BATCH)
    assert first["mean_target"] == pytest.approx(1.4375)
    assert same(weights(agent.target_net), weights(agent.q_net))
    # Second update: the target network is the online network after step 1.
    q_next = numpy_forward(agent.q_net, HAND_BATCH.next_obs)
    targets = HAND_BATCH.rewards + HAND_GAMMA * np.where(HAND_BATCH.terminated, 0.0, q_next.max(axis=1))
    second = agent.update(HAND_BATCH)
    assert second["mean_target"] == pytest.approx(targets.mean(), rel=1e-5)
    assert agent.grad_steps == 2 and same(weights(agent.target_net), weights(agent.q_net))


# ------------------------------------------------------------- D3: failure and purity
@pytest.mark.parametrize("bad", [np.inf, -np.inf, np.nan], ids=["inf", "-inf", "nan"])
def test_non_finite_loss_raises_before_anything_changes(bad):
    agent, twin = small_agent(seed=2, target_update_every=2), small_agent(seed=2, target_update_every=2)
    rng = np.random.default_rng(0)
    for _ in range(3):
        batch = random_batch(rng)
        agent.update(batch)
        twin.update(batch)
    online, target = weights(agent.q_net), weights(agent.target_net)

    poisoned = random_batch(rng)
    poisoned.rewards[5] = bad
    with pytest.raises(FloatingPointError):
        agent.update(poisoned)
    assert same(weights(agent.q_net), online) and same(weights(agent.target_net), target)
    assert agent.grad_steps == 3

    # The optimiser did not step either: training continues exactly like the twin's.
    for _ in range(3):
        batch = random_batch(rng)
        assert agent.update(batch) == twin.update(batch)
    assert same(weights(agent.q_net), weights(twin.q_net)) and same(weights(agent.target_net), weights(twin.target_net))
    assert agent.grad_steps == twin.grad_steps == 6


def test_update_does_not_modify_its_batch():
    agent = small_agent(seed=3)
    batch = random_batch(np.random.default_rng(1))
    kept = [array.copy() for array in arrays(batch)]
    out = agent.update(batch)
    for array, copy_ in zip(arrays(batch), kept):
        assert np.array_equal(array, copy_) and array.dtype == copy_.dtype
    assert set(out) == STAT_KEYS and all(type(v) is float and math.isfinite(v) for v in out.values())
    assert out["grad_norm"] > 0.0


def test_updates_are_deterministic_on_cpu():
    def run(seed: int) -> tuple[list[dict[str, float]], dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        agent = small_agent(seed=seed, target_update_every=7, double=True)
        rng = np.random.default_rng(5)
        stats = [agent.update(random_batch(rng)) for _ in range(40)]
        return stats, weights(agent.q_net), weights(agent.target_net)

    a, b, other = run(0), run(0), run(1)
    assert a[0] == b[0] and same(a[1], b[1]) and same(a[2], b[2])  # bit for bit
    assert a[0] != other[0] and not same(a[1], other[1])


def test_starvation_truncation_bootstraps_from_final_obs():
    """Real env -> replay buffer -> update: the truncated transition bootstraps
    from the true successor, not from zero and not from the post-reset observation."""
    obs, actions, reward, final_obs, terminated, reset_obs = starvation_truncation()
    buffer = ReplayBuffer(8, (OBS_DIM,))
    buffer.add(obs, actions, reward, final_obs, terminated)

    agent = DQNAgent((OBS_DIM,), N_ACTIONS, cpu_cfg(gamma=0.5, hidden=()), seed=0)
    set_linear(agent.q_net, np.zeros((N_ACTIONS, OBS_DIM)), np.zeros(N_ACTIONS))
    w_target = np.zeros((N_ACTIONS, OBS_DIM))
    w_target[:, A_DANGER_STRAIGHT] = [2.0, 8.0, 4.0]  # set in final_obs only
    w_target[:, A_HEADING.start + RIGHT] = 64.0  # set in the post-reset observation only
    set_linear(agent.target_net, w_target, np.zeros(N_ACTIONS))
    assert numpy_forward(agent.target_net, final_obs).tolist() == [[2.0, 8.0, 4.0]]
    assert numpy_forward(agent.target_net, reset_obs).tolist() == [[64.0, 64.0, 64.0]]

    out = agent.update(buffer.sample(4, np.random.default_rng(0)))  # four copies of the one transition
    # target = 0 + 0.5 * 8 = 4. As a terminal transition it would be 0; from the post-reset observation 32.
    assert out["mean_q"] == 0.0 and out["mean_target"] == pytest.approx(4.0)
    assert out["loss"] == pytest.approx(3.5)


# ------------------------------------------------------------- textbook equivalence
class TextbookDQN:
    """DQN (Mnih et al. 2015) and Double DQN (van Hasselt et al. 2016), written
    out step by step on plain torch: its own network, its own Adam, its own
    target copy, explicit formulas for target, Huber loss and clipping.
    """

    def __init__(self, layers: list[tuple[torch.Tensor, torch.Tensor]], cfg: DQNConfig) -> None:
        modules: list[torch.nn.Module] = []
        for i, (weight, bias) in enumerate(layers):
            linear = torch.nn.Linear(weight.shape[1], weight.shape[0])
            with torch.no_grad():
                linear.weight.copy_(weight)
                linear.bias.copy_(bias)
            modules.append(linear)
            if i < len(layers) - 1:
                modules.append(torch.nn.ReLU())
        self.online = torch.nn.Sequential(*modules)
        self.target = copy.deepcopy(self.online)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=cfg.lr)
        self.cfg = cfg
        self.steps = 0

    def step(self, batch: Batch) -> dict[str, float]:
        cfg = self.cfg
        s, s2 = torch.from_numpy(batch.obs.copy()), torch.from_numpy(batch.next_obs.copy())
        a, r = torch.from_numpy(batch.actions.copy()), torch.from_numpy(batch.rewards.copy())
        terminated = torch.from_numpy(batch.terminated.copy())
        rows = torch.arange(len(a))

        q_sa = self.online(s)[rows, a]
        with torch.no_grad():
            q_next = self.target(s2)
            if cfg.double:
                bootstrap = q_next[rows, self.online(s2).argmax(dim=1)]
            else:
                bootstrap = q_next.max(dim=1).values
            y = r + cfg.gamma * torch.where(terminated, torch.zeros_like(bootstrap), bootstrap)
        td = q_sa - y
        loss = torch.where(td.abs() < 1.0, 0.5 * td * td, td.abs() - 0.5).mean()

        self.optimizer.zero_grad()
        loss.backward()
        parameters = list(self.online.parameters())
        norm = math.sqrt(sum(float((p.grad.double() ** 2).sum()) for p in parameters))
        if norm > cfg.grad_clip:
            for p in parameters:
                p.grad.mul_(cfg.grad_clip / norm)
        self.optimizer.step()
        self.steps += 1
        if self.steps % cfg.target_update_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return {
            "loss": loss.item(),
            "mean_q": q_sa.mean().item(),
            "mean_target": y.mean().item(),
            "grad_norm": norm,
            "linear": int((td.abs() > 1.0).sum()),
        }


#: name -> (double, grad_clip, number of updates). With grad_clip = 10 no step
#: is clipped; with 0.2 about half of them are (the gradient norms of this run
#: lie between 0.09 and 0.49). Plain DQN is compared after 310 updates (15
#: syncs). The Double DQN target is discontinuous where the two best online
#: action values of a successor tie, so two correct implementations that
#: differ in the last float32 bit can part ways at a near-tie; its runs stop
#: after 130 updates, up to where the smallest gap between the two best values
#: is 5.7e-4 for these batches (measured), far above float32 noise.
TEXTBOOK_CASES = {
    "dqn": (False, 10.0, 310),
    "dqn-clipped": (False, 0.2, 310),
    "double": (True, 10.0, 130),
    "double-clipped": (True, 0.2, 130),
}
#: Largest weight difference agent - textbook loop at the end of the run,
#: measured with a reference implementation that calls ``smooth_l1_loss`` and
#: ``clip_grad_norm_``: 0 (bit-identical) for the unclipped cases, 2e-7 for
#: the clipped ones (``clip_grad_norm_`` divides by ``norm + 1e-6``, the loop
#: below by ``norm``). A wrong target, loss, clipping rule or sync schedule
#: moves the weights by 1e-2 or more.
TEXTBOOK_ATOL = 1e-4


@pytest.mark.parametrize("case", list(TEXTBOOK_CASES))
def test_agent_equals_a_textbook_dqn_loop(case):
    double, grad_clip, n_updates = TEXTBOOK_CASES[case]
    cfg = cpu_cfg(gamma=0.9, lr=1e-3, hidden=(32, 32), target_update_every=20, grad_clip=grad_clip, double=double)
    agent = DQNAgent((OBS_DIM,), N_ACTIONS, cfg, seed=7)
    textbook = TextbookDQN(
        [(layer.weight.detach().clone(), layer.bias.detach().clone()) for layer in linear_layers(agent.q_net)], cfg)
    rng = np.random.default_rng(100)
    clipped = linear = 0
    for step in range(1, n_updates + 1):
        batch = random_batch(rng, n=32, rewards=(-3.0, -1.0, 0.0, 0.0, 1.0, 3.0))
        want = textbook.step(batch)
        got = agent.update(batch)
        for key in STAT_KEYS:
            assert got[key] == pytest.approx(want[key], rel=1e-3, abs=1e-4), f"step {step}: {key}"
        clipped += want["grad_norm"] > grad_clip
        linear += want["linear"]
    assert agent.grad_steps == n_updates

    for net, reference in ((agent.q_net, textbook.online), (agent.target_net, textbook.target)):
        for layer, ref_layer in zip(linear_layers(net), linear_layers(reference), strict=True):
            assert torch.allclose(layer.weight, ref_layer.weight, rtol=0.0, atol=TEXTBOOK_ATOL)
            assert torch.allclose(layer.bias, ref_layer.bias, rtol=0.0, atol=TEXTBOOK_ATOL)
    # The run is only evidence if it went through every branch: clipped and
    # unclipped steps, both Huber regions, and a target network that lags
    # (the runs end 10 updates after a sync).
    if grad_clip < 1.0:
        assert clipped >= 30 and n_updates - clipped >= 30, clipped
    else:
        assert clipped == 0
    assert 100 <= linear <= 32 * n_updates - 100, linear
    assert not same(weights(agent.q_net), weights(agent.target_net))


def test_double_flag_changes_the_training_trajectory():
    def run(double: bool) -> dict[str, torch.Tensor]:
        agent = small_agent(seed=7, target_update_every=5, double=double)
        rng = np.random.default_rng(0)
        for _ in range(30):
            agent.update(random_batch(rng))
        return weights(agent.q_net)

    assert not same(run(False), run(True))


# ------------------------------------------------------------- M6: the agent on grid observations
def test_cnn_agent_action_selection_on_grid_observations():
    """``q_values`` / ``act`` / ``act_batch`` / ``act_epsilon`` on ``(N, C, W, W)``
    observations: the online CNN, the argmax, the RNG protocol."""
    agent = grid_agent(seed=3)
    randomise_all(agent.q_net, seed=1)
    randomise_all(agent.target_net, seed=2)  # a target network that differs: q_values must not read it
    obs = random_grid(np.random.default_rng(0), 40)
    q = agent.q_values(obs)
    assert isinstance(q, np.ndarray) and q.shape == (40, N_ACTIONS) and q.dtype == np.float32
    want = cnn_reference(agent.q_net, obs)
    scale = np.abs(want).max()
    assert np.allclose(q, want, rtol=1e-5, atol=1e-5 * scale)
    assert np.abs(q - cnn_reference(agent.target_net, obs)).max() > 0.1 * scale
    one = agent.q_values(obs[:1])
    assert one.shape == (1, N_ACTIONS) and np.allclose(one, q[:1], rtol=1e-5, atol=1e-5 * scale)

    batch = agent.act_batch(obs)
    assert isinstance(batch, np.ndarray) and batch.shape == (40,) and batch.dtype == np.int64
    assert np.array_equal(batch, want.argmax(axis=1)) and np.array_equal(batch, q.argmax(axis=1))
    assert len(set(batch.tolist())) > 1  # the greedy action depends on the observation
    for i in range(40):
        action = agent.act(obs[i])
        assert type(action) is int and action == batch[i], f"row {i}"
    assert agent.act(obs[0], None) == agent.act(obs[0], env=None) == batch[0]

    rng, twin = np.random.default_rng(99), np.random.default_rng(99)
    obs_rng = np.random.default_rng(1)
    for n, epsilon in ((1, 0.3), (16, 0.3), (7, 0.0), (64, 1.0)):
        some = random_grid(obs_rng, n)
        actions = agent.act_epsilon(some, epsilon, rng)
        assert isinstance(actions, np.ndarray) and actions.shape == (n,) and actions.dtype == np.int64
        greedy = cnn_reference(agent.q_net, some).argmax(axis=1)
        assert np.array_equal(actions, protocol_actions(greedy, epsilon, N_ACTIONS, twin)), f"batch of {n}"
        assert rng.bit_generator.state == twin.bit_generator.state

    # Acting records no gradient and changes nothing.
    assert all(p.grad is None for p in agent.q_net.parameters()) and agent.grad_steps == 0
    # A batch of the wrong shape is refused rather than reshaped.
    with pytest.raises(ValueError):
        agent.q_values(obs[:, 0])  # (40, 8, 8)
    with pytest.raises(ValueError):
        agent.act(obs[:1])  # a batch where one observation is expected


def test_cnn_agent_ties_pick_the_lowest_index():
    # No convolution, no hidden layer: a linear Q-function of the 3 * 2 * 2 = 12 flattened cells,
    # with Q(e_k) = column k for the first five cells (zero bias): exact ties.
    columns = [
        ([1.0, 1.0, 0.0, 0.0], 0),
        ([0.0, 2.0, 2.0, 2.0], 1),
        ([-1.0, -1.0, -1.0, -1.0], 0),
        ([0.0, 1.0, 3.0, 3.0], 2),
        ([7.0, 0.0, 0.0, 7.0], 0),
    ]
    agent = DQNAgent((3, 2, 2), 4, cpu_cfg(conv_channels=(), hidden=()), seed=0)
    assert isinstance(agent.q_net, CNN)
    weight = np.zeros((4, 12))
    weight[:, :5] = np.array([column for column, _ in columns]).T
    set_linear(agent.q_net, weight, np.zeros(4))
    obs = np.eye(12, dtype=np.float32)[:5].reshape(5, 3, 2, 2)  # one-hot cells 0 .. 4 as grids
    expected = [best for _, best in columns]
    assert agent.q_values(obs).T.tolist() == [[1, 0, -1, 0, 7], [1, 2, -1, 1, 0], [0, 2, -1, 3, 0], [0, 2, -1, 3, 7]]
    assert agent.act_batch(obs).tolist() == expected
    assert [agent.act(row) for row in obs] == expected
    assert agent.act_epsilon(obs, 0.0, np.random.default_rng(0)).tolist() == expected
    # All action values equal (zero convolutions, zero head, one bias): action 0 everywhere.
    flat = grid_agent(seed=0)
    with torch.no_grad():
        for layer in conv_layers(flat.q_net) + linear_layers(flat.q_net):
            layer.weight.zero_()
            layer.bias.zero_()
        linear_layers(flat.q_net)[-1].bias.fill_(0.5)
    some = random_grid(np.random.default_rng(2), 50)
    assert flat.q_values(some).tolist() == [[0.5, 0.5, 0.5]] * 50
    assert not flat.act_batch(some).any() and flat.act(some[0]) == 0


@pytest.mark.parametrize(
    "obs_cfg, W",
    [
        (ObsConfig(kind="B"), 8),
        (ObsConfig(kind="B", age_plane=False, heading_planes=False), 10),
        (ObsConfig(kind="B", age_plane=False), 8),
    ],
    ids=["B8-W8", "B3-W10", "B7-W8"],
)
def test_cnn_agent_update_on_real_grid_transitions_is_finite_and_moves_the_weights(obs_cfg, W):
    """Transitions of a real ``VecSnakeEnv`` on observation B, passed as the
    trainer passes them (``info["final_obs"]``, ``terminated`` only), through
    a ``Batch`` and through a ``ReplayBuffer`` of the grid shape."""
    env = VecSnakeEnv(EnvConfig(W=W, starve_limit=30, obs=obs_cfg), n_envs=16, seed=5)
    assert env.obs_shape == (obs_cfg.n_planes, W, W)
    agent = DQNAgent(env.obs_shape, env.n_actions, cpu_cfg(conv_channels=(4, 4), hidden=(16,), lr=1e-3,
                                                              target_update_every=5), seed=1)
    assert isinstance(agent.q_net, CNN) and agent.obs_shape == env.obs_shape
    buffer = ReplayBuffer(500, env.obs_shape)
    rng = np.random.default_rng(0)
    obs = env.reset()
    n_terminated = n_truncated = 0
    for step in range(1, 11):
        actions = rng.integers(0, env.n_actions, size=16)
        next_obs, reward, terminated, truncated, info = env.step(actions)
        assert obs.shape == (16, *env.obs_shape) and obs.dtype == np.float32
        batch = Batch(obs=obs, actions=actions.astype(np.int64), rewards=reward.astype(np.float32),
                      next_obs=info["final_obs"], terminated=terminated)
        buffer.add(obs, actions, reward, info["final_obs"], terminated)
        before, target_before = weights(agent.q_net), weights(agent.target_net)
        first_conv = conv_layers(agent.q_net)[0].weight.detach().clone()
        out = agent.update(batch)
        assert set(out) == STAT_KEYS and all(type(v) is float and math.isfinite(v) for v in out.values()), step
        assert out["grad_norm"] > 0.0 and agent.grad_steps == step
        after = weights(agent.q_net)
        assert not same(after, before)
        assert not torch.equal(conv_layers(agent.q_net)[0].weight, first_conv)  # the convolutions train too
        if step % 5 == 0:
            assert same(weights(agent.target_net), after), f"step {step}: no sync"
        else:
            assert same(weights(agent.target_net), target_before), f"step {step}: the target network moved"
        n_terminated += int(terminated.sum())
        n_truncated += int(truncated.sum())
        obs = next_obs
    assert n_terminated > 0  # random play on a small board dies within 160 environment steps
    # Replayed batches of the grid shape train as well.
    for _ in range(3):
        out = agent.update(buffer.sample(32, rng))
        assert all(math.isfinite(v) for v in out.values())
    assert agent.grad_steps == 13
    q = agent.q_values(obs)
    assert q.shape == (16, env.n_actions) and q.dtype == np.float32 and np.isfinite(q).all()
    assert all(torch.isfinite(v).all() for v in weights(agent.q_net).values())


# ------------------------------------------------------------- save / load
#: Every field away from its default, with values float32 cannot hold.
AWKWARD_CFG = DQNConfig(
    gamma=1.0 / 3.0, lr=1.2345678e-3, batch_size=48, buffer_size=5_000, replay=False, learning_starts=96,
    grad_steps_per_vec_step=1, target_update_every=4, grad_clip=0.7, eps_start=0.9, eps_end=0.01,
    eps_decay_fraction=0.35, hidden=(12, 8), conv_channels=(5,), double=True, keep_checkpoints=False, device="cpu",
    torch_threads=1,
)


def trained_agent(n_updates: int = 6) -> tuple[DQNAgent, np.random.Generator]:
    """Agent after ``n_updates`` updates (6: last sync at step 4, so target != online)."""
    agent = DQNAgent((OBS_DIM,), 4, AWKWARD_CFG, seed=11)
    rng = np.random.default_rng(21)
    for _ in range(n_updates):
        agent.update(random_batch(rng))
    return agent, rng


def test_save_load_round_trip_is_exact(tmp_path):
    agent, rng = trained_agent()
    online, target = weights(agent.q_net), weights(agent.target_net)
    assert not same(online, target)
    path = tmp_path / "deep" / "er" / "model.checkpoint"  # parents are created; no suffix is appended
    out = agent.save(path)
    assert isinstance(out, Path) and out.resolve() == path.resolve() and path.is_file()
    assert [p.name for p in path.parent.iterdir()] == ["model.checkpoint"]  # nothing else was written
    assert same(weights(agent.q_net), online) and same(weights(agent.target_net), target)  # saving changes nothing
    assert agent.grad_steps == 6

    loaded = DQNAgent.load(path)
    assert isinstance(loaded, DQNAgent) and loaded is not agent
    assert same(weights(loaded.q_net), online) and same(weights(loaded.target_net), target)
    assert loaded.grad_steps == 6 and type(loaded.grad_steps) is int
    assert loaded.cfg == AWKWARD_CFG and isinstance(loaded.cfg.hidden, tuple)
    assert tuple(loaded.obs_shape) == (OBS_DIM,) and loaded.n_actions == 4
    assert loaded.device == torch.device("cpu")
    assert loaded.target_net is not loaded.q_net

    # The optimiser state came along: both agents continue identically, through the next sync (step 8).
    for step in (7, 8, 9):
        batch = random_batch(rng)
        assert loaded.update(batch) == agent.update(batch), f"step {step}"
        assert same(weights(loaded.q_net), weights(agent.q_net)), f"step {step}"
        assert same(weights(loaded.target_net), weights(agent.target_net)), f"step {step}"
    assert loaded.grad_steps == agent.grad_steps == 9


def test_saved_file_holds_plain_tensors_and_primitives(tmp_path):
    agent, _ = trained_agent()
    for include_optimizer in (True, False):
        path = agent.save(str(tmp_path / f"model_{include_optimizer}.pt"), include_optimizer=include_optimizer)
        assert isinstance(path, Path) and path.is_file()
        payload = torch.load(path, map_location="cpu", weights_only=True)  # refuses pickled objects
        assert isinstance(payload, dict) and payload
        assert DQNAgent.load(str(path)).grad_steps == 6  # str paths work too
    assert (tmp_path / "model_True.pt").stat().st_size > (tmp_path / "model_False.pt").stat().st_size


def test_load_without_optimizer_state_starts_a_fresh_optimizer(tmp_path):
    agent, rng = trained_agent()
    path = agent.save(tmp_path / "weights_only.pt", include_optimizer=False)
    loaded = DQNAgent.load(path)
    assert same(weights(loaded.q_net), weights(agent.q_net))
    assert same(weights(loaded.target_net), weights(agent.target_net))
    assert loaded.grad_steps == 6 and loaded.cfg == AWKWARD_CFG

    assert isinstance(loaded.optimizer, torch.optim.Adam)
    assert len(loaded.optimizer.state_dict()["state"]) == 0  # no moments yet
    assert loaded.optimizer.param_groups[0]["lr"] == AWKWARD_CFG.lr
    assert len(agent.optimizer.state_dict()["state"]) > 0

    # It trains (on its own weights), but Adam's moments are gone: the next
    # step differs from the original agent's.
    batch = random_batch(rng)
    before = weights(loaded.q_net)
    loaded.update(batch)
    agent.update(batch)
    assert loaded.grad_steps == agent.grad_steps == 7
    assert not same(weights(loaded.q_net), before)
    assert not same(weights(loaded.q_net), weights(agent.q_net))


def test_load_device_argument_and_untrained_round_trip(tmp_path):
    agent = DQNAgent((OBS_DIM,), N_ACTIONS, cpu_cfg(hidden=()), seed=3)
    path = agent.save(tmp_path / "fresh.pt")
    loaded = DQNAgent.load(path, device="cpu")
    assert loaded.device == torch.device("cpu") and loaded.grad_steps == 0
    assert loaded.cfg.hidden == () and isinstance(loaded.cfg.hidden, tuple) and loaded.cfg == agent.cfg
    assert same(weights(loaded.q_net), weights(agent.q_net))
    obs = random_obs(np.random.default_rng(0), 20)
    assert np.array_equal(loaded.q_values(obs), agent.q_values(obs))
    assert np.array_equal(loaded.act_batch(obs), agent.act_batch(obs))


def test_saved_config_entry_spells_out_every_field_with_the_tuples_as_lists(tmp_path):
    agent, _ = trained_agent()
    payload = torch.load(agent.save(tmp_path / "model.pt"), map_location="cpu", weights_only=True)
    config = payload["config"]
    assert set(config) == {f.name for f in fields(DQNConfig)}
    assert config["hidden"] == [12, 8] and type(config["hidden"]) is list
    assert config["conv_channels"] == [5] and type(config["conv_channels"]) is list  # M6, also on observation A
    assert config["replay"] is False and config["double"] is True and config["grad_steps_per_vec_step"] == 1
    assert list(payload["obs_shape"]) == [OBS_DIM] and payload["n_actions"] == 4 and payload["grad_steps"] == 6


#: A grid agent with every field away from its default: 7 planes (no age plane) on 8x8.
GRID_SHAPE = (7, 8, 8)
AWKWARD_GRID_CFG = replace(AWKWARD_CFG, hidden=(12,), conv_channels=(3, 5))


def random_grid_batch(rng: np.random.Generator, n: int = 16, shape: tuple[int, int, int] = GRID_SHAPE) -> Batch:
    """Synthetic transitions on grid observations."""
    return Batch(
        obs=random_grid(rng, n, shape),
        actions=rng.integers(0, 4, size=n).astype(np.int64),
        rewards=rng.choice((-1.0, 0.0, 0.0, 1.0), size=n).astype(np.float32),
        next_obs=random_grid(rng, n, shape),
        terminated=rng.random(n) < 0.2,
    )


def trained_grid_agent(n_updates: int = 6) -> tuple[DQNAgent, np.random.Generator]:
    """Grid agent after ``n_updates`` updates (6: last sync at step 4, so target != online)."""
    agent = DQNAgent(GRID_SHAPE, 4, AWKWARD_GRID_CFG, seed=11)
    rng = np.random.default_rng(21)
    for _ in range(n_updates):
        agent.update(random_grid_batch(rng))
    return agent, rng


def test_grid_agent_save_load_round_trip_is_exact(tmp_path):
    agent, rng = trained_grid_agent()
    assert isinstance(agent.q_net, CNN)
    online, target = weights(agent.q_net), weights(agent.target_net)
    assert not same(online, target)
    path = agent.save(tmp_path / "grid" / "model.pt")
    assert path.is_file()
    payload = torch.load(path, map_location="cpu", weights_only=True)  # plain tensors and primitives
    assert payload["config"]["conv_channels"] == [3, 5] and type(payload["config"]["conv_channels"]) is list
    assert payload["config"]["hidden"] == [12] and payload["config"]["replay"] is False
    assert list(payload["obs_shape"]) == list(GRID_SHAPE) and payload["n_actions"] == 4
    assert same(weights(agent.q_net), online) and agent.grad_steps == 6  # saving changes nothing

    loaded = DQNAgent.load(path)
    assert isinstance(loaded, DQNAgent) and isinstance(loaded.q_net, CNN) and loaded is not agent
    assert same(weights(loaded.q_net), online) and same(weights(loaded.target_net), target)
    assert loaded.grad_steps == 6 and type(loaded.grad_steps) is int
    assert loaded.cfg == AWKWARD_GRID_CFG
    assert type(loaded.cfg.conv_channels) is tuple and type(loaded.cfg.hidden) is tuple
    assert tuple(loaded.obs_shape) == GRID_SHAPE and loaded.n_actions == 4
    assert [tuple(layer.weight.shape) for layer in conv_layers(loaded.q_net)] == [(3, 7, 3, 3), (5, 3, 3, 3)]
    assert loaded.q_net.n_flatten == 5 * 64

    # The optimiser state came along: both agents continue identically, through the next sync (step 8).
    for step in (7, 8, 9):
        batch = random_grid_batch(rng)
        assert loaded.update(batch) == agent.update(batch), f"step {step}"
        assert same(weights(loaded.q_net), weights(agent.q_net)), f"step {step}"
        assert same(weights(loaded.target_net), weights(agent.target_net)), f"step {step}"
    assert loaded.grad_steps == agent.grad_steps == 9

    # Weights only, as the checkpoints are written.
    light = agent.save(tmp_path / "light.pt", include_optimizer=False)
    assert "optimizer" not in torch.load(light, map_location="cpu", weights_only=True)
    again = DQNAgent.load(light)
    assert again.cfg == AWKWARD_GRID_CFG and same(weights(again.q_net), weights(agent.q_net))
    obs = random_grid(np.random.default_rng(0), 20, GRID_SHAPE)
    assert np.array_equal(again.q_values(obs), agent.q_values(obs))
    assert np.array_equal(again.act_batch(obs), agent.act_batch(obs))


def test_a_checkpoint_written_before_m6_loads_with_the_defaults_of_the_new_fields(tmp_path):
    """A pre-M6 file has no ``replay`` and no ``conv_channels`` in its config
    entry (built here by hand from a saved file): it loads with the defaults
    of those fields, everything else exact."""
    agent = DQNAgent((OBS_DIM,), 4, cpu_cfg(hidden=(8,), replay=False, conv_channels=(5,), lr=3e-3), seed=2)
    rng = np.random.default_rng(4)
    for _ in range(3):
        agent.update(random_batch(rng))
    payload = torch.load(agent.save(tmp_path / "m6.pt"), map_location="cpu", weights_only=True)
    assert payload["config"]["replay"] is False and payload["config"]["conv_channels"] == [5]
    old = dict(payload)
    old["config"] = {key: value for key, value in payload["config"].items() if key not in ("replay", "conv_channels")}
    assert set(old["config"]) == {f.name for f in fields(DQNConfig)} - {"replay", "conv_channels"}
    with open(tmp_path / "pre_m6.pt", "wb") as f:
        torch.save(old, f)

    loaded = DQNAgent.load(tmp_path / "pre_m6.pt")
    assert loaded.cfg == replace(agent.cfg, replay=True, conv_channels=(32, 64, 64))
    assert loaded.cfg.replay is True and loaded.cfg.conv_channels == (32, 64, 64)
    assert type(loaded.cfg.conv_channels) is tuple and loaded.cfg.lr == 3e-3 and loaded.cfg.hidden == (8,)
    assert same(weights(loaded.q_net), weights(agent.q_net)) and same(weights(loaded.target_net), weights(agent.target_net))
    assert loaded.grad_steps == 3 and tuple(loaded.obs_shape) == (OBS_DIM,) and loaded.n_actions == 4
    batch = random_batch(rng)
    assert loaded.update(batch) == agent.update(batch)  # the optimiser state too


# ------------------------------------------------------------- CUDA
@needs_cuda
def test_cuda_agent_trains_and_stays_finite():
    agent = DQNAgent((OBS_DIM,), N_ACTIONS, DQNConfig(hidden=(32, 32), lr=1e-3, target_update_every=5), seed=0)
    assert agent.cfg.device == "auto" and agent.device.type == "cuda"  # auto -> cuda when available
    for net in (agent.q_net, agent.target_net):
        assert all(p.device.type == "cuda" for p in net.parameters())
    # The initial weights are those of the CPU agent with the same seed.
    cpu = DQNAgent((OBS_DIM,), N_ACTIONS, cpu_cfg(hidden=(32, 32), lr=1e-3, target_update_every=5), seed=0)
    assert same(weights(agent.q_net), weights(cpu.q_net))

    rng = np.random.default_rng(0)
    for step in range(1, 51):
        out = agent.update(random_batch(rng, rewards=(-3.0, -1.0, 0.0, 1.0, 3.0)))
        assert set(out) == STAT_KEYS and all(type(v) is float and math.isfinite(v) for v in out.values()), step
    assert agent.grad_steps == 50
    assert same(weights(agent.q_net), weights(agent.target_net))  # 50 is a multiple of 5
    assert all(torch.isfinite(v).all() for v in weights(agent.q_net).values())

    obs = random_obs(rng, 64)
    q = agent.q_values(obs)
    assert isinstance(q, np.ndarray) and q.shape == (64, N_ACTIONS) and q.dtype == np.float32 and np.isfinite(q).all()
    batch = agent.act_batch(obs)
    assert batch.dtype == np.int64 and np.array_equal(batch, q.argmax(axis=1))
    assert type(agent.act(obs[0])) is int and agent.act(obs[0]) == batch[0]
    actions = agent.act_epsilon(obs, 0.3, np.random.default_rng(1))
    assert actions.shape == (64,) and actions.dtype == np.int64
    with pytest.raises(FloatingPointError):
        poisoned = random_batch(rng)
        poisoned.rewards[0] = np.inf
        agent.update(poisoned)


@needs_cuda
def test_cuda_checkpoint_round_trips_and_loads_on_cpu(tmp_path):
    agent = DQNAgent((OBS_DIM,), N_ACTIONS, DQNConfig(hidden=(16,), lr=1e-3, target_update_every=4, device="cuda"), seed=1)
    rng = np.random.default_rng(1)
    for _ in range(6):
        agent.update(random_batch(rng))
    path = agent.save(tmp_path / "cuda.pt")
    torch.load(path, map_location="cpu", weights_only=True)

    on_cpu = DQNAgent.load(path, device="cpu")
    assert on_cpu.device.type == "cpu" and all(p.device.type == "cpu" for p in on_cpu.q_net.parameters())
    assert same(weights(on_cpu.q_net), weights(agent.q_net))
    assert same(weights(on_cpu.target_net), weights(agent.target_net))
    assert on_cpu.grad_steps == 6
    assert math.isfinite(on_cpu.update(random_batch(rng))["loss"])

    on_cuda = DQNAgent.load(path)
    assert on_cuda.device.type == "cuda" and on_cuda.cfg == agent.cfg
    assert all(p.device.type == "cuda" for p in on_cuda.target_net.parameters())
    assert same(weights(on_cuda.q_net), weights(agent.q_net))
    assert math.isfinite(on_cuda.update(random_batch(rng))["loss"])


# ------------------------------------------------------------- learning sanity
#: 8x8, reward R0, observation A, relative actions. H = 30 instead of the
#: default 100 so that the short run meets starvation truncations.
LEARN_ENV = EnvConfig(W=8, starve_limit=30, reward=RewardConfig(variant="R0"))
LEARN_N_ENVS = 16
LEARN_VEC_STEPS = 500  # 8000 environment steps, 1748 gradient steps
LEARN_CFG = DQNConfig(
    gamma=0.9, lr=1e-3, batch_size=64, buffer_size=LEARN_N_ENVS * LEARN_VEC_STEPS, learning_starts=1024,
    target_update_every=250, hidden=(64, 64), device="cpu",
)
#: Run seed 0 in the convention of the training entry point: environment base
#: seed 100_000 * (run_seed + 1), exploration stream default_rng([run_seed, 1]),
#: replay sampling stream default_rng([run_seed, 2]).
LEARN_RUN_SEED = 0
#: Measured with a reference implementation of the spec: random 0.17, learned
#: 15.77 for this seed and 15.57 .. 17.11 over six run seeds (92 .. 101 times
#: random). The floor keeps 36 % slack under the worst of them and is still 59
#: times the random player.
LEARN_MIN_SCORE = 10.0
LEARN_MIN_RATIO = 40.0
#: CPU seconds for the 500 vec-steps of training (measured: about 2 s on one
#: thread). The budget only rejects an implementation that is an order of
#: magnitude too slow for the experiments.
LEARN_CPU_BUDGET_S = 30.0


def test_learns_to_beat_the_random_player_on_8x8():
    """The DQN training loop of the spec (D5), written with the agent API only."""
    cfg = LEARN_CFG
    total_env_steps = LEARN_N_ENVS * LEARN_VEC_STEPS
    env = VecSnakeEnv(LEARN_ENV, n_envs=LEARN_N_ENVS, seed=100_000 * (LEARN_RUN_SEED + 1))
    agent = DQNAgent(env.obs_shape, env.n_actions, cfg, seed=LEARN_RUN_SEED)
    buffer = ReplayBuffer(cfg.buffer_size, env.obs_shape)
    explore_rng = np.random.default_rng([LEARN_RUN_SEED, 1])
    sample_rng = np.random.default_rng([LEARN_RUN_SEED, 2])
    grad_steps_per_vec_step = max(1, LEARN_N_ENVS // 4)

    n_terminated = n_truncated = truncated_passed_as_terminated = 0
    stats: list[dict[str, float]] = []
    threads = torch.get_num_threads()
    torch.set_num_threads(1)  # tiny matrices: one thread is the fastest; restored below
    t0 = time.process_time()
    try:
        obs = env.reset()
        for k in range(LEARN_VEC_STEPS):
            epsilon = linear_schedule(
                k * LEARN_N_ENVS, cfg.eps_start, cfg.eps_end, cfg.eps_decay_fraction * total_env_steps)
            actions = agent.act_epsilon(obs, epsilon, explore_rng)
            next_obs, reward, terminated, truncated, info = env.step(actions)
            buffer.add(obs, actions, reward, info["final_obs"], terminated)  # true successor, terminated only
            obs = next_obs
            n_terminated += int(terminated.sum())
            n_truncated += int(truncated.sum())
            truncated_passed_as_terminated += int((truncated & terminated).sum())
            if (k + 1) * LEARN_N_ENVS >= cfg.learning_starts:
                for _ in range(grad_steps_per_vec_step):
                    stats.append(agent.update(buffer.sample(cfg.batch_size, sample_rng)))
        train_cpu_s = time.process_time() - t0

        # truncated != terminated, structurally: truncations did happen, none
        # was passed as terminal, and the buffer (large enough to keep every
        # transition) holds exactly the env's terminations as terminated.
        assert n_truncated >= 5 and n_terminated >= 80, (n_truncated, n_terminated)
        assert truncated_passed_as_terminated == 0
        assert len(buffer) == total_env_steps == buffer.capacity
        assert int(stored(buffer).terminated.sum()) == n_terminated

        # The updates: one per (vec-step from learning_starts on) x G, finite,
        # with action values inside the range of discounted returns of rewards in [-1, 1].
        first_learning_step = -(-cfg.learning_starts // LEARN_N_ENVS)  # ceil
        assert agent.grad_steps == len(stats) == (LEARN_VEC_STEPS - first_learning_step + 1) * grad_steps_per_vec_step
        assert all(math.isfinite(v) for s in stats for v in s.values())
        assert max(abs(s["mean_q"]) for s in stats) <= 1.0 / (1.0 - cfg.gamma)

        # Acceptance: 100 greedy episodes on the standard evaluation boards.
        learned = evaluate(agent, LEARN_ENV, n_episodes=100, seed=DEFAULT_EVAL_SEED)
    finally:
        torch.set_num_threads(threads)
    floor = evaluate(RandomPolicy(seed=DEFAULT_EVAL_SEED), LEARN_ENV, n_episodes=100, seed=DEFAULT_EVAL_SEED)
    assert learned.policy == "dqn" and len(learned.episodes) == 100
    assert 0.0 < floor.mean_score < 1.0, floor.mean_score
    assert learned.mean_score >= LEARN_MIN_SCORE, (learned.mean_score, floor.mean_score)
    assert learned.mean_score >= LEARN_MIN_RATIO * floor.mean_score, (learned.mean_score, floor.mean_score)
    assert learned.mean_score - learned.ci95 > floor.mean_score + floor.ci95

    assert train_cpu_s < LEARN_CPU_BUDGET_S, f"training took {train_cpu_s:.1f} CPU seconds"
