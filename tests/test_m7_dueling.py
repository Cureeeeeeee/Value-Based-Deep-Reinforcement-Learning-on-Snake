"""Dueling head (M7, spec D1 / D3): ``DuelingMLP``, ``CNN(..., dueling=True)``,
``build_q_network(..., dueling=...)``, the dueling ``DQNAgent`` (seeded initial
weights, untouched global RNG, the gradient step, save / load on observations
A and B) and the pre-M7 checkpoint rule.

Wang et al. 2016, the mean-subtracted aggregation (their eq. 9)::

    Q(s, a) = V(s) + A(s, a) - mean_a' A(s, a')

The reference forward pass is computed here from the raw weights of the
sub-modules with ``F.linear`` / ``torch.relu`` in float64, never through the
module's own ``forward``. The structure is inspected as the spec fixes it:
``shared`` (``Linear -> ReLU`` per width of ``hidden[:-1]``), ``value``
(``trunk -> hidden[-1] -> 1``) and ``advantage`` (``trunk -> hidden[-1] ->
n_actions``), created in that order so that the seeded initial weights are
the same function of the seed everywhere. Helpers, the hand-built
transitions and the awkward configs come from ``tests.test_dqn``.

This file is separate from ``tests/test_dqn.py`` so that the import of
``DuelingMLP`` cannot stop that file from collecting.
"""

from __future__ import annotations

import inspect
import math
import re
from dataclasses import fields, replace

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from snake_rl.agents.dqn.agent import DQNAgent
from snake_rl.agents.dqn.config import DQNConfig
from snake_rl.agents.dqn.networks import CNN, MLP, DuelingMLP, build_q_network
from snake_rl.agents.dqn.replay import Batch
from snake_rl.env.vec_env import VecSnakeEnv
from tests.helpers import EnvConfig, ObsConfig
from tests.test_dqn import (
    AWKWARD_CFG,
    AWKWARD_GRID_CFG,
    CNN_CASES,
    GRID_SHAPE,
    N_ACTIONS,
    OBS_DIM,
    STAT_KEYS,
    conv_layers,
    cpu_cfg,
    grid_agent,
    linear_layers,
    protocol_actions,
    random_batch,
    random_grid,
    random_grid_batch,
    random_obs,
    randomise,
    randomise_all,
    same,
    small_agent,
    weights,
)

M7_FIELDS = ("dueling", "per", "per_alpha", "per_beta_start", "per_eps")
M7_DEFAULTS = {"dueling": False, "per": False, "per_alpha": 0.6, "per_beta_start": 0.4, "per_eps": 1e-6}


# ------------------------------------------------------------- helpers
def linears(seq: torch.nn.Sequential) -> list[torch.nn.Linear]:
    """The ``Linear`` modules of one stream, in order."""
    return [m for m in seq if isinstance(m, torch.nn.Linear)]


def layout(seq: torch.nn.Sequential) -> list[str]:
    return [type(m).__name__ for m in seq]


def shapes(layers) -> list[tuple[int, ...]]:
    return [tuple(layer.weight.shape) for layer in layers]


def stream_forward(seq: torch.nn.Sequential, h: torch.Tensor) -> torch.Tensor:
    """``Linear -> ReLU`` ... ``Linear`` (no activation after the last) in float64 from the raw weights."""
    layers = linears(seq)
    for i, layer in enumerate(layers):
        h = F.linear(h, layer.weight.detach().double(), layer.bias.detach().double())
        if i < len(layers) - 1:
            h = torch.relu(h)
    return h


def dueling_reference(head: DuelingMLP, x) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spec D1 in float64: ``(Q, V, A)`` with ``Q = V + A - mean_a A`` from the raw weights."""
    h = x.double() if torch.is_tensor(x) else torch.from_numpy(np.asarray(x, dtype=np.float64))
    for layer in linears(head.shared):
        h = torch.relu(F.linear(h, layer.weight.detach().double(), layer.bias.detach().double()))
    v = stream_forward(head.value, h)
    a = stream_forward(head.advantage, h)
    return (v + a - a.mean(dim=1, keepdim=True)).numpy(), v.numpy(), a.numpy()


def conv_features(net: CNN, x: np.ndarray) -> torch.Tensor:
    """The convolutional part of a CNN in float64 (M6, D1): ``conv2d`` (stride 1, padding 1) -> ``relu``, flatten."""
    h = torch.from_numpy(np.asarray(x, dtype=np.float64))
    for conv in conv_layers(net):
        h = F.relu(F.conv2d(h, conv.weight.detach().double(), conv.bias.detach().double(), stride=1, padding=1))
    return h.flatten(1)


def formula_forward(head: DuelingMLP, params: dict[str, torch.Tensor], x: torch.Tensor) -> torch.Tensor:
    """``Q`` of the dueling formula from a dict of (differentiable) parameters named as in ``state_dict``."""

    def run(prefix: str, seq: torch.nn.Sequential, h: torch.Tensor, relu_last: bool) -> torch.Tensor:
        names = [name for name, m in seq.named_children() if isinstance(m, torch.nn.Linear)]
        for i, name in enumerate(names):
            h = F.linear(h, params[f"{prefix}.{name}.weight"], params[f"{prefix}.{name}.bias"])
            if relu_last or i < len(names) - 1:
                h = torch.relu(h)
        return h

    h = run("shared", head.shared, x, relu_last=True)
    v = run("value", head.value, h, relu_last=False)
    a = run("advantage", head.advantage, h, relu_last=False)
    return v + a - a.mean(dim=1, keepdim=True)


def dueling_agent(seed: int = 0, **kwargs) -> DQNAgent:
    """Dueling agent on observation A with a small network, on the CPU."""
    settings = dict(hidden=(16, 16), lr=1e-2, dueling=True)
    settings.update(kwargs)
    return DQNAgent((OBS_DIM,), N_ACTIONS, cpu_cfg(**settings), seed=seed)


def dueling_grid_agent(seed: int = 0, in_shape: tuple[int, int, int] = (8, 8, 8), **kwargs) -> DQNAgent:
    """Dueling agent on a grid observation with a small CNN, on the CPU."""
    settings = dict(conv_channels=(4, 4), hidden=(16,), lr=1e-2, dueling=True)
    settings.update(kwargs)
    return DQNAgent(in_shape, N_ACTIONS, cpu_cfg(**settings), seed=seed)


# ------------------------------------------------------------- D1: structure
#: hidden -> weight shapes of the three sub-modules (in_features OBS_DIM = 11, N_ACTIONS = 3).
DUELING_CASES = {
    (): dict(shared=[], value=[(1, OBS_DIM)], advantage=[(N_ACTIONS, OBS_DIM)]),
    (16,): dict(shared=[], value=[(16, OBS_DIM), (1, 16)], advantage=[(16, OBS_DIM), (N_ACTIONS, 16)]),
    (16, 8): dict(shared=[(16, OBS_DIM)], value=[(8, 16), (1, 8)], advantage=[(8, 16), (N_ACTIONS, 8)]),
    (16, 8, 4): dict(shared=[(16, OBS_DIM), (8, 16)], value=[(4, 8), (1, 4)], advantage=[(4, 8), (N_ACTIONS, 4)]),
}


@pytest.mark.parametrize("hidden", list(DUELING_CASES), ids=lambda h: f"hidden{h}")
def test_dueling_mlp_structure(hidden):
    want = DUELING_CASES[hidden]
    net = DuelingMLP(OBS_DIM, N_ACTIONS, hidden)
    assert isinstance(net, torch.nn.Module) and not isinstance(net, MLP)
    # Three Sequential sub-modules, created (registered) in the order shared, value, advantage.
    assert [name for name, _ in net.named_children()] == ["shared", "value", "advantage"]
    for name in ("shared", "value", "advantage"):
        assert isinstance(getattr(net, name), torch.nn.Sequential), name
    # shared: Linear -> ReLU for every width in hidden[:-1]; the empty Sequential (identity) otherwise.
    assert layout(net.shared) == ["Linear", "ReLU"] * len(want["shared"])
    assert shapes(linears(net.shared)) == want["shared"]
    # value and advantage: trunk -> hidden[-1] -> ReLU -> 1 / n_actions, or one Linear without hidden layers.
    for stream, stream_shapes in (("value", want["value"]), ("advantage", want["advantage"])):
        seq = getattr(net, stream)
        assert layout(seq) == (["Linear", "ReLU", "Linear"] if hidden else ["Linear"]), stream
        assert shapes(linears(seq)) == stream_shapes, stream
        assert all(layer.bias is not None for layer in linears(seq))
    assert shapes(linear_layers(net)) == want["shared"] + want["value"] + want["advantage"]  # registration order
    n_params = sum(o * i + o for (o, i) in want["shared"] + want["value"] + want["advantage"])
    assert sum(p.numel() for p in net.parameters()) == n_params
    assert all(key.startswith(("shared.", "value.", "advantage.")) for key in net.state_dict())
    x = torch.from_numpy(random_obs(np.random.default_rng(0), 5))
    out = net(x)
    assert out.shape == (5, N_ACTIONS) and out.dtype == torch.float32
    assert net(x[:1]).shape == (1, N_ACTIONS)


def test_dueling_mlp_accepts_any_sequence_of_widths_and_other_sizes():
    net = DuelingMLP(4, 2, [5])
    assert shapes(linears(net.shared)) == [] and shapes(linears(net.value)) == [(5, 4), (1, 5)]
    assert shapes(linears(net.advantage)) == [(5, 4), (2, 5)]
    big = DuelingMLP(OBS_DIM, 4, (256, 256))  # the spec's example: shared 11 -> 256, value 256 -> 256 -> 1, advantage 256 -> 256 -> 4
    assert shapes(linears(big.shared)) == [(256, OBS_DIM)]
    assert shapes(linears(big.value)) == [(256, 256), (1, 256)] and shapes(linears(big.advantage)) == [(256, 256), (4, 256)]
    wide = DuelingMLP(OBS_DIM, N_ACTIONS, (512,))  # no shared layer: value in -> 512 -> 1, advantage in -> 512 -> n
    assert len(list(wide.shared)) == 0
    assert shapes(linears(wide.value)) == [(512, OBS_DIM), (1, 512)] and shapes(linears(wide.advantage)) == [(512, OBS_DIM), (N_ACTIONS, 512)]


@pytest.mark.parametrize(
    "bad",
    [dict(in_features=0), dict(in_features=-1), dict(hidden=(0,)), dict(hidden=(16, -8)), dict(hidden=(16, 0, 8)),
     dict(n_actions=0), dict(n_actions=-3)],
    ids=lambda bad: ",".join(f"{k}={v}" for k, v in bad.items()),
)
def test_dueling_mlp_rejects_bad_sizes_and_names_them(bad):
    settings = dict(in_features=OBS_DIM, n_actions=N_ACTIONS, hidden=(16, 8))
    settings.update(bad)
    with pytest.raises(ValueError) as excinfo:
        DuelingMLP(settings["in_features"], settings["n_actions"], settings["hidden"])
    (value,) = bad.values()
    offending = [v for v in value if v < 1] if isinstance(value, tuple) else [value]
    assert all(str(v) in str(excinfo.value) for v in offending), str(excinfo.value)


# ------------------------------------------------------------- D1: forward
@pytest.mark.parametrize("hidden", list(DUELING_CASES), ids=lambda h: f"hidden{h}")
def test_dueling_forward_matches_the_functional_reference(hidden):
    """``forward`` against ``F.linear`` / ``relu`` and ``V + A - mean(A)`` on randomised weights, in float64."""
    net = DuelingMLP(OBS_DIM, 4, hidden)
    randomise(net, seed=len(hidden) + 1)  # every Linear layer: shared, value and advantage streams
    x = np.random.default_rng(len(hidden)).normal(size=(64, OBS_DIM)).astype(np.float32)
    with torch.no_grad():
        out = net(torch.from_numpy(x)).numpy()
    q, v, a = dueling_reference(net, x)
    assert out.shape == q.shape == (64, 4) and out.dtype == np.float32
    scale = np.abs(q).max()
    assert scale > 0.01
    assert np.allclose(out, q, rtol=1e-5, atol=1e-5 * scale)
    # The aggregation is identifiable: the mean over actions of Q is V (the advantages sum to zero after
    # the subtraction). Without the subtraction the mean would be V + mean(A); subtracting the mean of
    # V + A instead would give zero.
    assert np.allclose(out.mean(axis=1), v[:, 0], rtol=1e-5, atol=1e-5 * scale)
    assert np.abs(a.mean(axis=1)).max() > 1e-3 * scale  # the check is not vacuous: mean(A) is not zero
    assert (q < 0).any()  # action values are not rectified


def test_dueling_mlp_without_hidden_layers_is_a_linear_dueling_q_function():
    """``hidden = ()``: ``V(s) = w s + b``, ``A(s, .) = W s + c``, ``Q = V + A - mean(A)`` on exact numbers."""
    net = DuelingMLP(2, 3, ())
    assert len(list(net.shared)) == 0 and len(list(net.parameters())) == 4
    (v_layer,) = linears(net.value)
    (a_layer,) = linears(net.advantage)
    with torch.no_grad():
        v_layer.weight.copy_(torch.tensor([[1.0, 1.0]]))
        v_layer.bias.copy_(torch.tensor([0.5]))
        a_layer.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 1.0]]))
        a_layer.bias.copy_(torch.tensor([0.0, 0.5, 0.5]))
        out = net(torch.tensor([[1.0, 2.0], [0.0, 1.0], [4.0, -1.0], [-2.0, 2.0]]))
    # V = s0 + s1 + 0.5 = 3.5, 1.5, 3.5, 0.5; A = [s0, s1 + 0.5, s1 + 0.5] = [1, 2.5, 2.5], [0, 1.5, 1.5],
    # [4, -0.5, -0.5], [-2, 2.5, 2.5] with means 2, 1, 1, 1. V + A alone would give [4.5, 6, 6] on the first row.
    assert out.tolist() == [[2.5, 4.0, 4.0], [0.5, 2.0, 2.0], [6.5, 2.0, 2.0], [-2.5, 2.0, 2.0]]


def test_dueling_mlp_applies_relu_inside_both_streams_and_not_after_their_last_layers():
    net = DuelingMLP(2, 3, (2,))
    assert len(list(net.shared)) == 0
    v_first, v_last = linears(net.value)
    a_first, a_last = linears(net.advantage)
    with torch.no_grad():
        v_first.weight.copy_(torch.eye(2))
        v_first.bias.zero_()
        v_last.weight.copy_(torch.tensor([[1.0, -1.0]]))
        v_last.bias.zero_()
        a_first.weight.copy_(torch.tensor([[0.0, 1.0], [1.0, 0.0]]))  # swaps the two features
        a_first.bias.zero_()
        a_last.weight.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 2.0]]))
        a_last.bias.copy_(torch.tensor([0.0, 0.75, -0.75]))
        out = net(torch.tensor([[1.0, -2.0], [-1.0, 3.0], [0.0, 0.0]]))
    # s = (1, -2): V = relu(1) - relu(-2) = 1; A = [0, 1.75, 1.25] from relu([-2, 1]) = [0, 1] (mean 1)
    #              -> Q = 1 + [-1, 0.75, 0.25] = [0, 1.75, 1.25].
    # s = (-1, 3): V = 0 - 3 = -3; A = [3, 0.75, 5.25] from relu([3, -1]) = [3, 0] (mean 3) -> Q = [-3, -5.25, -0.75].
    # s = (0, 0): V = 0; A = [0, 0.75, -0.75] (mean 0) -> Q = [0, 0.75, -0.75]: negative outputs pass.
    # Without the ReLU in the value stream the first row would be [2, 3.75, 3.25]; without the one in the
    # advantage stream A would be [-2, 1.75, -0.75] and the row would not even be dyadic.
    assert out.tolist() == [[0.0, 1.75, 1.25], [-3.0, -5.25, -0.75], [0.0, 0.75, -0.75]]


def test_dueling_mlp_takes_the_default_initialisation_from_the_global_generator_in_module_order():
    """Torch's default ``Linear`` initialisation from the global CPU generator; the sub-modules are created
    shared, value, advantage, so the shared layer and the first value layer consume exactly the draws of
    the first two layers of an ``MLP`` of the same widths built from the same seed."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(3)
        a = DuelingMLP(OBS_DIM, N_ACTIONS, (16, 8))
        torch.manual_seed(3)
        b = DuelingMLP(OBS_DIM, N_ACTIONS, (16, 8))
        torch.manual_seed(3)
        mlp = MLP(OBS_DIM, N_ACTIONS, (16, 8))
        torch.manual_seed(4)
        c = DuelingMLP(OBS_DIM, N_ACTIONS, (16, 8))
    assert same(weights(a), weights(b)) and not same(weights(a), weights(c))
    (shared,) = linears(a.shared)
    value_first, _ = linears(a.value)
    advantage_first, _ = linears(a.advantage)
    mlp_layers = linear_layers(mlp)
    assert torch.equal(shared.weight, mlp_layers[0].weight) and torch.equal(shared.bias, mlp_layers[0].bias)
    assert torch.equal(value_first.weight, mlp_layers[1].weight) and torch.equal(value_first.bias, mlp_layers[1].bias)
    assert not torch.equal(advantage_first.weight, value_first.weight)  # fresh draws, not a copy
    for layer in linear_layers(a):
        assert layer.weight.std() > 0.01


# ------------------------------------------------------------- D1: build_q_network and the CNN
def test_build_q_network_dispatches_dueling_on_both_observation_shapes():
    params = inspect.signature(build_q_network).parameters
    assert list(params) == ["obs_shape", "n_actions", "hidden", "conv_channels", "dueling"]
    assert params["dueling"].default is False and tuple(params["conv_channels"].default) == (32, 64, 64)
    net = build_q_network((OBS_DIM,), N_ACTIONS, (32, 16), dueling=True)
    assert isinstance(net, DuelingMLP)
    assert shapes(linears(net.shared)) == [(32, OBS_DIM)]
    assert shapes(linears(net.value)) == [(16, 32), (1, 16)] and shapes(linears(net.advantage)) == [(16, 32), (N_ACTIONS, 16)]
    assert isinstance(build_q_network((OBS_DIM,), N_ACTIONS, (32, 16)), MLP)
    assert isinstance(build_q_network((OBS_DIM,), N_ACTIONS, (32, 16), (4, 4), False), MLP)
    linear = build_q_network((OBS_DIM,), 4, (), dueling=True)
    assert isinstance(linear, DuelingMLP) and shapes(linear_layers(linear)) == [(1, OBS_DIM), (4, OBS_DIM)]

    cnn = build_q_network((8, 8, 8), N_ACTIONS, (16,), (4, 6), dueling=True)
    assert isinstance(cnn, CNN) and isinstance(cnn.head, DuelingMLP) and cnn.n_flatten == 6 * 64
    assert shapes(conv_layers(cnn)) == [(4, 8, 3, 3), (6, 4, 3, 3)]
    assert shapes(linears(cnn.head.value)) == [(16, 6 * 64), (1, 16)]
    assert shapes(linears(cnn.head.advantage)) == [(16, 6 * 64), (N_ACTIONS, 16)]
    assert isinstance(build_q_network((8, 8, 8), N_ACTIONS, (16,), (4, 6)).head, MLP)
    assert isinstance(build_q_network((3, 10, 10), 4, (512,), dueling=True).head, DuelingMLP)
    flat = build_q_network((7, 8, 8), N_ACTIONS, (16,), (), dueling=True)  # the MLP-on-B ablation keeps the head
    assert isinstance(flat, CNN) and conv_layers(flat) == [] and isinstance(flat.head, DuelingMLP)
    assert shapes(linears(flat.head.value)) == [(16, 7 * 64), (1, 16)]
    # Any other shape is still a ValueError naming the shape.
    for shape in ((8, 8), (2, 3, 4, 5)):
        with pytest.raises(ValueError, match=re.escape(str(shape))):
            build_q_network(shape, N_ACTIONS, (16,), dueling=True)


@pytest.mark.parametrize("case", CNN_CASES, ids=lambda case: f"{case[0]}-conv{case[1]}-hidden{case[2]}")
def test_cnn_with_a_dueling_head_matches_the_functional_reference(case):
    in_shape, conv_channels, hidden = case
    assert inspect.signature(CNN.__init__).parameters["dueling"].default is False
    net = CNN(in_shape, 4, conv_channels, hidden, dueling=True)
    assert isinstance(net.head, DuelingMLP) and isinstance(CNN(in_shape, 4, conv_channels, hidden).head, MLP)
    assert isinstance(CNN(in_shape, 4, conv_channels, hidden, dueling=False).head, MLP)
    assert [name for name, _ in net.named_children()] == ["cnn", "head"]
    assert shapes(linear_layers(net.head))[0][1] == net.n_flatten  # the first layer of the head (shared, or a stream) reads the flattened grid
    trunk = hidden[-2] if len(hidden) >= 2 else net.n_flatten
    assert shapes(linears(net.head.value))[0][1] == trunk == shapes(linears(net.head.advantage))[0][1]
    randomise_all(net, seed=len(conv_channels) + 10 * len(hidden))
    x = random_grid(np.random.default_rng(len(conv_channels)), 32, in_shape)
    with torch.no_grad():
        out = net(torch.from_numpy(x)).numpy()
    q, v, _ = dueling_reference(net.head, conv_features(net, x))
    assert out.shape == q.shape == (32, 4) and out.dtype == np.float32
    scale = np.abs(q).max()
    assert scale > 0.01
    assert np.allclose(out, q, rtol=1e-5, atol=1e-5 * scale)
    assert np.allclose(out.mean(axis=1), v[:, 0], rtol=1e-5, atol=1e-5 * scale)
    assert all(key.startswith(("cnn.", "head.shared.", "head.value.", "head.advantage.")) for key in net.state_dict())


# ------------------------------------------------------------- D3: the dueling agent
def test_dueling_agent_networks_are_dueling_and_seeded():
    a = dueling_agent(seed=5)
    assert a.cfg.dueling is True
    assert isinstance(a.q_net, DuelingMLP) and isinstance(a.target_net, DuelingMLP) and a.target_net is not a.q_net
    assert same(weights(a.q_net), weights(a.target_net))
    assert shapes(linears(a.q_net.shared)) == [(16, OBS_DIM)]
    assert shapes(linears(a.q_net.value)) == [(16, 16), (1, 16)] and shapes(linears(a.q_net.advantage)) == [(16, 16), (N_ACTIONS, 16)]
    (group,) = a.optimizer.param_groups
    assert [id(p) for p in group["params"]] == [id(p) for p in a.q_net.parameters()]
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(999)
        torch.rand(17)  # wherever the global generator stands ...
        b = dueling_agent(seed=5)
    assert same(weights(a.q_net), weights(b.q_net))  # ... seed 5 gives these weights
    others = [weights(dueling_agent(seed=s).q_net) for s in (0, 6)]
    assert not any(same(weights(a.q_net), other) for other in others) and not same(others[0], others[1])
    # The same seeded draw as the plain agent: the shared layer and the first value layer of the dueling
    # network are the first two layers of the MLP of a plain agent with the same seed and widths.
    plain = small_agent(seed=5)  # hidden (16, 16)
    assert isinstance(plain.q_net, MLP)
    (shared,) = linears(a.q_net.shared)
    value_first = linears(a.q_net.value)[0]
    assert torch.equal(shared.weight, linear_layers(plain.q_net)[0].weight)
    assert torch.equal(value_first.weight, linear_layers(plain.q_net)[1].weight)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(5)
        reference = DuelingMLP(OBS_DIM, N_ACTIONS, (16, 16))
    assert same(weights(a.q_net), weights(reference))
    for layer in linear_layers(a.q_net):
        assert layer.weight.std() > 0.01


def test_dueling_grid_agent_networks_are_cnns_with_a_dueling_head_and_seeded():
    a = dueling_grid_agent(seed=5)
    assert isinstance(a.q_net, CNN) and isinstance(a.q_net.head, DuelingMLP) and isinstance(a.target_net.head, DuelingMLP)
    assert shapes(conv_layers(a.q_net)) == [(4, 8, 3, 3), (4, 4, 3, 3)]
    assert shapes(linears(a.q_net.head.value)) == [(16, 256), (1, 16)]
    assert shapes(linears(a.q_net.head.advantage)) == [(16, 256), (N_ACTIONS, 16)]
    assert same(weights(a.q_net), weights(a.target_net)) and a.target_net is not a.q_net
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(999)
        torch.rand(17)
        b = dueling_grid_agent(seed=5)
    assert same(weights(a.q_net), weights(b.q_net))
    assert not same(weights(a.q_net), weights(dueling_grid_agent(seed=6).q_net))
    # The convolutions are created before the head: they equal those of the plain grid agent of the same seed.
    plain = grid_agent(seed=5)
    for ours, theirs in zip(conv_layers(a.q_net), conv_layers(plain.q_net)):
        assert torch.equal(ours.weight, theirs.weight) and torch.equal(ours.bias, theirs.bias)
    assert torch.equal(linears(a.q_net.head.value)[0].weight, linear_layers(plain.q_net.head)[0].weight)


def test_constructing_a_dueling_agent_leaves_the_global_torch_rng_alone():
    before = torch.random.get_rng_state()
    dueling_agent(seed=3)
    DQNAgent((OBS_DIM,), 4, cpu_cfg(hidden=(), dueling=True), seed=8)
    dueling_grid_agent(seed=2)
    DQNAgent((3, 10, 10), 4, cpu_cfg(conv_channels=(), hidden=(), dueling=True), seed=8)
    assert torch.equal(torch.random.get_rng_state(), before)


def test_dueling_agent_acts_on_the_dueling_values():
    agent = dueling_agent(seed=3)
    randomise(agent.q_net, seed=1)
    randomise(agent.target_net, seed=2)  # q_values must not read the target network
    obs = random_obs(np.random.default_rng(0), 40)
    q = agent.q_values(obs)
    want, _, _ = dueling_reference(agent.q_net, obs)
    scale = np.abs(want).max()
    assert q.shape == (40, N_ACTIONS) and q.dtype == np.float32
    assert np.allclose(q, want, rtol=1e-5, atol=1e-5 * scale)
    assert np.abs(q - dueling_reference(agent.target_net, obs)[0]).max() > 0.1 * scale
    batch = agent.act_batch(obs)
    assert np.array_equal(batch, want.argmax(axis=1)) and len(set(batch.tolist())) > 1
    assert [agent.act(row) for row in obs] == batch.tolist()
    rng, twin = np.random.default_rng(99), np.random.default_rng(99)
    actions = agent.act_epsilon(obs, 0.3, rng)
    assert np.array_equal(actions, protocol_actions(want.argmax(axis=1), 0.3, N_ACTIONS, twin))
    assert rng.bit_generator.state == twin.bit_generator.state
    assert all(p.grad is None for p in agent.q_net.parameters()) and agent.grad_steps == 0


@pytest.mark.parametrize("double", [False, True], ids=["dqn", "double"])
def test_update_of_a_dueling_agent_matches_autograd_on_the_dueling_formula(double):
    """One gradient step: the target from the dueling formula on the target network (and the online
    network's argmax for Double DQN), the Huber loss, and the gradient the optimiser consumed equal to
    autograd on the formula written out from the raw parameters."""
    agent = DQNAgent((OBS_DIM,), N_ACTIONS, cpu_cfg(hidden=(6, 4), gamma=0.5, lr=1e-2, double=double, dueling=True), seed=3)
    randomise(agent.q_net, seed=1)
    randomise(agent.target_net, seed=2)
    batch = random_batch(np.random.default_rng(0), n=16)
    q_next_target, _, _ = dueling_reference(agent.target_net, batch.next_obs)
    if double:
        q_next_online, _, _ = dueling_reference(agent.q_net, batch.next_obs)
        bootstrap = q_next_target[np.arange(16), q_next_online.argmax(axis=1)]
    else:
        bootstrap = q_next_target.max(axis=1)
    targets = batch.rewards.astype(np.float64) + 0.5 * np.where(batch.terminated, 0.0, bootstrap)

    params = {name: p.detach().double().clone().requires_grad_(True) for name, p in agent.q_net.named_parameters()}
    q_sa = formula_forward(agent.q_net, params, torch.from_numpy(batch.obs).double())[torch.arange(16), torch.from_numpy(batch.actions)]
    td = q_sa - torch.from_numpy(targets)
    loss = torch.where(td.abs() < 1.0, 0.5 * td * td, td.abs() - 0.5).mean()
    grads = torch.autograd.grad(loss, list(params.values()))
    norm = math.sqrt(sum(float((g**2).sum()) for g in grads))
    scale = min(1.0, agent.cfg.grad_clip / norm)  # the consumed gradient is the clipped one

    out = agent.update(batch)
    assert set(out) == STAT_KEYS
    assert out["loss"] == pytest.approx(loss.item(), rel=1e-5)
    assert out["mean_target"] == pytest.approx(targets.mean(), rel=1e-5)
    assert out["mean_q"] == pytest.approx(q_sa.detach().mean().item(), rel=1e-5)
    assert out["grad_norm"] == pytest.approx(norm, rel=1e-4)
    beta1 = agent.optimizer.param_groups[0]["betas"][0]
    for (name, p), g in zip(agent.q_net.named_parameters(), grads):
        consumed = agent.optimizer.state[p]["exp_avg"].double() / (1.0 - beta1)
        assert torch.allclose(consumed, scale * g, rtol=1e-4, atol=1e-6), name
    assert agent.grad_steps == 1
    # The weighted update works on the dueling network too and reports the TD errors.
    out = agent.update(batch, np.full(16, 0.5, dtype=np.float32))
    assert set(out) == STAT_KEYS | {"td_abs"} and out["td_abs"].shape == (16,) and out["td_abs"].dtype == np.float32


def test_dueling_agent_trains_on_real_grid_transitions_and_syncs_its_target():
    env = VecSnakeEnv(EnvConfig(W=8, starve_limit=30, obs=ObsConfig(kind="B")), n_envs=16, seed=5)
    agent = DQNAgent(env.obs_shape, env.n_actions, cpu_cfg(conv_channels=(4,), hidden=(16,), lr=1e-3,
                                                              target_update_every=5, dueling=True), seed=1)
    assert isinstance(agent.q_net.head, DuelingMLP)
    rng = np.random.default_rng(0)
    obs = env.reset()
    for step in range(1, 11):
        actions = rng.integers(0, env.n_actions, size=16)
        next_obs, reward, terminated, truncated, info = env.step(actions)
        before, target_before = weights(agent.q_net), weights(agent.target_net)
        out = agent.update(Batch(obs=obs, actions=actions.astype(np.int64), rewards=reward.astype(np.float32),
                                 next_obs=info["final_obs"], terminated=terminated))
        assert all(type(v) is float and math.isfinite(v) for v in out.values()) and out["grad_norm"] > 0
        assert not same(weights(agent.q_net), before)
        if step % 5 == 0:
            assert same(weights(agent.target_net), weights(agent.q_net)), step
        else:
            assert same(weights(agent.target_net), target_before), step
        obs = next_obs
    assert agent.grad_steps == 10


# ------------------------------------------------------------- D3: save / load
DUELING_CFG = replace(AWKWARD_CFG, dueling=True)
DUELING_GRID_CFG = replace(AWKWARD_GRID_CFG, dueling=True)


def trained(cfg: DQNConfig, shape: tuple[int, ...], make_batch, n_updates: int = 6) -> tuple[DQNAgent, np.random.Generator]:
    """Agent after ``n_updates`` updates (6: last sync at step 4, so target != online)."""
    agent = DQNAgent(shape, 4, cfg, seed=11)
    rng = np.random.default_rng(21)
    for _ in range(n_updates):
        agent.update(make_batch(rng))
    return agent, rng


@pytest.mark.parametrize("which", ["A", "B"])
def test_dueling_agent_save_load_round_trip_is_exact(which, tmp_path):
    if which == "A":
        cfg, shape, make_batch = DUELING_CFG, (OBS_DIM,), random_batch
    else:
        cfg, shape, make_batch = DUELING_GRID_CFG, GRID_SHAPE, random_grid_batch
    agent, rng = trained(cfg, shape, make_batch)
    head = agent.q_net if which == "A" else agent.q_net.head
    assert isinstance(head, DuelingMLP)
    online, target = weights(agent.q_net), weights(agent.target_net)
    assert not same(online, target)
    path = agent.save(tmp_path / "dueling" / "model.pt")
    payload = torch.load(path, map_location="cpu", weights_only=True)  # plain tensors and primitives
    assert payload["config"]["dueling"] is True and set(payload["config"]) == {f.name for f in fields(DQNConfig)}
    prefixes = ("shared.", "value.", "advantage.") if which == "A" else ("cnn.", "head.shared.", "head.value.", "head.advantage.")
    assert all(key.startswith(prefixes) for key in payload["q_net"]) and payload["q_net"].keys() == payload["target_net"].keys()
    assert same(weights(agent.q_net), online) and agent.grad_steps == 6  # saving changes nothing

    loaded = DQNAgent.load(path)
    assert isinstance(loaded, DQNAgent) and loaded is not agent
    assert loaded.cfg == cfg and loaded.cfg.dueling is True
    loaded_head = loaded.q_net if which == "A" else loaded.q_net.head
    assert isinstance(loaded_head, DuelingMLP) and isinstance(loaded.target_net if which == "A" else loaded.target_net.head, DuelingMLP)
    assert same(weights(loaded.q_net), online) and same(weights(loaded.target_net), target)
    assert loaded.grad_steps == 6 and tuple(loaded.obs_shape) == shape and loaded.n_actions == 4
    # The optimiser state came along: both agents continue identically, through the next sync (step 8).
    for step in (7, 8, 9):
        batch = make_batch(rng)
        assert loaded.update(batch) == agent.update(batch), f"step {step}"
        assert same(weights(loaded.q_net), weights(agent.q_net)), f"step {step}"
        assert same(weights(loaded.target_net), weights(agent.target_net)), f"step {step}"
    # Weights only, as the checkpoints are written.
    light = agent.save(tmp_path / "light.pt", include_optimizer=False)
    again = DQNAgent.load(light, device="cpu")
    assert again.cfg == cfg and same(weights(again.q_net), weights(agent.q_net))
    obs = random_obs(np.random.default_rng(0), 20) if which == "A" else random_grid(np.random.default_rng(0), 20, GRID_SHAPE)
    assert np.array_equal(again.q_values(obs), agent.q_values(obs))
    assert np.array_equal(again.act_batch(obs), agent.act_batch(obs))


def test_a_checkpoint_written_before_m7_loads_with_the_defaults_of_the_new_fields(tmp_path):
    """A pre-M7 file has no ``dueling`` and no ``per*`` keys in its config entry (built here by hand from a
    saved file): it loads with the defaults of those fields, everything else exact."""
    agent = DQNAgent((OBS_DIM,), 4, cpu_cfg(hidden=(8,), lr=3e-3, double=True, target_update_every=2), seed=2)
    rng = np.random.default_rng(4)
    for _ in range(3):
        agent.update(random_batch(rng))
    payload = torch.load(agent.save(tmp_path / "m7.pt"), map_location="cpu", weights_only=True)
    assert all(key in payload["config"] for key in M7_FIELDS)
    old = dict(payload)
    old["config"] = {key: value for key, value in payload["config"].items() if key not in M7_FIELDS}
    assert set(old["config"]) == {f.name for f in fields(DQNConfig)} - set(M7_FIELDS)
    with open(tmp_path / "pre_m7.pt", "wb") as f:
        torch.save(old, f)

    loaded = DQNAgent.load(tmp_path / "pre_m7.pt")
    assert loaded.cfg == agent.cfg  # the agent was built with the defaults of the five fields
    assert {name: getattr(loaded.cfg, name) for name in M7_FIELDS} == M7_DEFAULTS
    assert isinstance(loaded.q_net, MLP)
    assert same(weights(loaded.q_net), weights(agent.q_net)) and same(weights(loaded.target_net), weights(agent.target_net))
    assert loaded.grad_steps == 3 and tuple(loaded.obs_shape) == (OBS_DIM,) and loaded.n_actions == 4
    batch = random_batch(rng)
    assert loaded.update(batch) == agent.update(batch)  # the optimiser state too

    # A pre-M6 file (no replay / conv_channels either) still loads, with the defaults of all seven fields.
    older = dict(old)
    older["config"] = {key: value for key, value in old["config"].items() if key not in ("replay", "conv_channels")}
    with open(tmp_path / "pre_m6.pt", "wb") as f:
        torch.save(older, f)
    assert DQNAgent.load(tmp_path / "pre_m6.pt").cfg == agent.cfg
