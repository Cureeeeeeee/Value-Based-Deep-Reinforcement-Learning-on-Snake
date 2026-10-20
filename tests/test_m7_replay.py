"""Prioritised replay (M7, spec D2): ``PrioritizedReplayBuffer`` and
``PrioritizedBatch`` of ``snake_rl.agents.dqn.replay``.

Schaul et al. 2016, proportional prioritisation: the priority of a transition
is ``(|td| + eps) ** alpha``, new transitions arrive with the largest priority
ever assigned, a batch is drawn by stratified sampling of the cumulative
priority mass, and the bias of the sampling is corrected by
importance-sampling weights ``(p_i / p_min) ** -beta`` (``(n P(i)) ** -beta``
normalised by its maximum over the buffer, so every weight is in ``(0, 1]``).

RNG protocol under test (fixed by the spec): exactly one call ``u =
rng.random(batch_size)``; with ``p`` the priorities of the ``n`` stored slots
in slot order and ``total = p.sum()``::

    mass_k = (k + u_k) / batch_size * total
    idx_k  = min(searchsorted(cumsum(p), mass_k, side="right"), n - 1)

The storage is read through this protocol with a generator whose ``random``
is scripted (the pattern of ``ScriptedGenerator`` in ``tests/test_dqn.py``):
with equal priorities sample ``k`` of a batch of ``n`` always lands in slot
``k``, and any single slot can be read with a batch of one whose ``u`` lies
inside the slot's mass interval. The tagged transitions, ``ids_of`` and the
base-class reader ``stored`` come from ``tests.test_dqn``.
"""

from __future__ import annotations

import dataclasses
import inspect
import subprocess
import sys

import numpy as np
import pytest

from snake_rl.agents.dqn.replay import Batch, PrioritizedBatch, PrioritizedReplayBuffer, ReplayBuffer
from tests.test_dqn import (
    BATCH_FIELDS,
    OBS_DIM,
    REPO_ROOT,
    TAG_SHAPE,
    arrays,
    ids_of,
    starvation_truncation,
    stored,
    tagged,
)

PER_FIELDS = (*BATCH_FIELDS, "indices", "weights")


# ------------------------------------------------------------- helpers
class ScriptedUniform(np.random.Generator):
    """A generator whose ``random(size)`` returns scripted numbers in [0, 1)
    and records its calls; ``integers`` must not be called by the
    prioritised sampler."""

    def __init__(self, u) -> None:
        super().__init__(np.random.PCG64(0))
        self.u = np.asarray(u, dtype=np.float64)
        self.calls: list[int | None] = []

    def random(self, size=None, dtype=np.float64, out=None):
        assert out is None
        self.calls.append(None if size is None else int(np.prod(size)))
        if size is None:
            return float(self.u[0])
        return self.u.reshape(size).astype(dtype)

    def integers(self, *args, **kwargs):
        raise AssertionError("prioritised sampling must not call rng.integers")


def stratified_indices(priorities, batch_size: int, u) -> np.ndarray:
    """Spec D2, written out: the slot of every sample of a batch."""
    p = np.asarray(priorities, dtype=np.float64)
    mass = (np.arange(batch_size) + np.asarray(u, dtype=np.float64)) / batch_size * p.sum()
    return np.minimum(np.searchsorted(np.cumsum(p), mass, side="right"), len(p) - 1).astype(np.int64)


def is_weights(priorities, idx, beta: float) -> np.ndarray:
    """Spec D2: ``w_i = (p_i / p_min) ** -beta`` in float32."""
    p = np.asarray(priorities, dtype=np.float64)
    return ((p[idx] / p.min()) ** (-beta)).astype(np.float32)


def stored_per(buffer: PrioritizedReplayBuffer) -> PrioritizedBatch:
    """Every stored transition in slot order, through the sampling protocol:
    with equal priorities sample ``k`` of a batch of ``n`` lands in slot ``k``
    whatever ``u`` is (the buffer's priorities must never have been updated)."""
    n = len(buffer)
    p = buffer.priorities
    assert p.shape == (n,) and np.all(p == p[0]), "stored_per needs equal priorities"
    rng = ScriptedUniform(np.full(n, 0.5))
    batch = buffer.sample(n, rng, 0.0)
    assert rng.calls == [n], rng.calls  # one draw: random(batch_size)
    assert batch.indices.tolist() == list(range(n))
    return batch


def read_slot(buffer: PrioritizedReplayBuffer, slot: int, beta: float = 0.0) -> PrioritizedBatch:
    """The transition of one slot: a batch of one whose ``u`` lies inside the slot's mass interval."""
    p = buffer.priorities
    c = np.concatenate([[0.0], np.cumsum(p)])
    u = (c[slot] + c[slot + 1]) / 2.0 / p.sum()
    batch = buffer.sample(1, ScriptedUniform([u]), beta)
    assert batch.indices.tolist() == [slot]
    return batch


def filled(capacity: int, ids, alpha: float = 0.6, eps: float = 1e-6) -> PrioritizedReplayBuffer:
    buffer = PrioritizedReplayBuffer(capacity, TAG_SHAPE, alpha, eps)
    buffer.add(*tagged(ids))
    return buffer


def with_priorities(td_abs, alpha: float = 1.0, eps: float = 0.5) -> PrioritizedReplayBuffer:
    """Ids ``0 .. n - 1`` in slots ``0 .. n - 1`` with priorities ``(|td| + eps) ** alpha``
    (exact numbers for the dyadic ``td_abs`` used here: alpha 1, eps 0.5)."""
    n = len(td_abs)
    buffer = filled(n, range(n), alpha, eps)
    buffer.update_priorities(np.arange(n), np.asarray(td_abs, dtype=np.float32))
    return buffer


# ------------------------------------------------------------- constructor, attributes, errors
def test_constructor_attributes_defaults_and_signatures():
    buffer = PrioritizedReplayBuffer(5, TAG_SHAPE)
    assert isinstance(buffer, ReplayBuffer) and issubclass(PrioritizedReplayBuffer, ReplayBuffer)
    assert buffer.capacity == 5 and len(buffer) == 0 and buffer.obs_shape == TAG_SHAPE
    assert buffer.alpha == 0.6 and buffer.eps == 1e-6 and type(buffer.alpha) is float and type(buffer.eps) is float
    assert buffer.max_priority == 1.0 and type(buffer.max_priority) is float
    p = buffer.priorities
    assert isinstance(p, np.ndarray) and p.shape == (0,) and p.dtype == np.float64
    other = PrioritizedReplayBuffer(5, TAG_SHAPE, 0.7, 1e-3)
    assert other.alpha == 0.7 and other.eps == 1e-3
    keyword = PrioritizedReplayBuffer(5, TAG_SHAPE, eps=1e-3, alpha=0.7)
    assert keyword.alpha == 0.7 and keyword.eps == 1e-3
    assert PrioritizedReplayBuffer(5, TAG_SHAPE, alpha=0.0).alpha == 0.0  # uniform sampling in effect
    assert PrioritizedReplayBuffer(5, TAG_SHAPE, alpha=1.0).alpha == 1.0

    params = inspect.signature(PrioritizedReplayBuffer.__init__).parameters
    assert list(params) == ["self", "capacity", "obs_shape", "alpha", "eps"]
    assert params["alpha"].default == 0.6 and params["eps"].default == 1e-6
    params = inspect.signature(PrioritizedReplayBuffer.sample).parameters
    assert list(params) == ["self", "batch_size", "rng", "beta"] and params["beta"].default is inspect.Parameter.empty
    assert list(inspect.signature(ReplayBuffer.sample).parameters) == ["self", "batch_size", "rng"]  # unchanged


@pytest.mark.parametrize("alpha", [-0.1, 1.1, -1.0, 2.0])
def test_alpha_outside_the_unit_interval_is_refused(alpha):
    with pytest.raises(ValueError):
        PrioritizedReplayBuffer(5, TAG_SHAPE, alpha=alpha)


@pytest.mark.parametrize("eps", [0.0, 0, -1e-6, -1.0])
def test_eps_must_be_positive(eps):
    with pytest.raises(ValueError):
        PrioritizedReplayBuffer(5, TAG_SHAPE, eps=eps)


def test_base_class_rules_still_apply():
    for capacity in (0, -1, -100):
        with pytest.raises(ValueError):
            PrioritizedReplayBuffer(capacity, TAG_SHAPE)
    buffer = PrioritizedReplayBuffer(8, (OBS_DIM,))
    with pytest.raises(ValueError):  # inconsistent shapes, as in the base class
        buffer.add(np.zeros((3, 10), np.float32), np.zeros(3, np.int64), np.zeros(3, np.float32),
                   np.zeros((3, 10), np.float32), np.zeros(3, bool))
    assert len(buffer) == 0
    with pytest.raises(ValueError):  # empty
        buffer.sample(4, np.random.default_rng(0), 0.5)


def test_sample_validates_beta_and_requires_it():
    buffer = filled(8, range(4))
    rng = np.random.default_rng(0)
    for beta in (-0.1, 1.5, -1.0, 2.0):
        with pytest.raises(ValueError):
            buffer.sample(2, rng, beta)
    for beta in (0.0, 1.0, 0.4, np.float64(0.5)):
        assert isinstance(buffer.sample(2, rng, beta), PrioritizedBatch)
    with pytest.raises(TypeError):
        buffer.sample(2, rng)  # beta is required


# ------------------------------------------------------------- add: priorities and ring semantics
def test_add_gives_new_slots_the_max_priority_and_keeps_the_ring_semantics():
    """The same stream of adds (batches below, at and above the capacity, wrapping inside one add) into
    a ``ReplayBuffer`` and a ``PrioritizedReplayBuffer``: the same slots, and every slot at priority 1."""
    capacity = 5
    base, per = ReplayBuffer(capacity, TAG_SHAPE), PrioritizedReplayBuffer(capacity, TAG_SHAPE)
    t = 0
    for n in (3, 4, 1, 5, 2, 5, 4, 1, 1, 3):
        base.add(*tagged(range(t, t + n)))
        per.add(*tagged(range(t, t + n)))
        t += n
        assert len(per) == len(base) == min(t, capacity)
        batch = stored_per(per)
        assert ids_of(batch) == ids_of(stored(base)), f"after {t} transitions"
        assert per.priorities.tolist() == [1.0] * len(per) and per.max_priority == 1.0
        assert batch.weights.tolist() == [1.0] * len(per)
    # More than the capacity at once: the last ``capacity`` transitions, all at the maximal priority.
    big = PrioritizedReplayBuffer(4, TAG_SHAPE)
    big.add(*tagged(range(10)))
    assert len(big) == 4 and sorted(ids_of(stored_per(big))) == [6, 7, 8, 9]
    assert big.priorities.tolist() == [1.0] * 4
    one = PrioritizedReplayBuffer(1, TAG_SHAPE)
    one.add(*tagged([4]))
    one.add(*tagged([5, 6]))
    assert ids_of(stored_per(one)) == [6]


def test_overwritten_slots_get_the_current_max_priority():
    # Ids 0 .. 3 with priorities 1, 2, 4, 3 (alpha 1, eps 0.5): max_priority 4.
    buffer = with_priorities([0.5, 1.5, 3.5, 2.5])
    assert buffer.priorities.tolist() == [1.0, 2.0, 4.0, 3.0] and buffer.max_priority == 4.0
    buffer.add(*tagged([4, 5]))  # into slots 0 and 1
    assert buffer.priorities.tolist() == [4.0, 4.0, 4.0, 3.0]
    assert [ids_of(read_slot(buffer, i)) for i in range(4)] == [[4], [5], [2], [3]]
    buffer.update_priorities(np.array([2]), np.array([0.0]))  # slot 2 down to 0.5; the maximum stays 4
    assert buffer.priorities.tolist() == [4.0, 4.0, 0.5, 3.0] and buffer.max_priority == 4.0
    buffer.add(*tagged([6]))  # slot 2 again
    assert buffer.priorities.tolist() == [4.0, 4.0, 4.0, 3.0]
    assert ids_of(read_slot(buffer, 2)) == [6]


def test_the_transition_arrays_are_those_of_the_base_class():
    """Same stream into both buffers, non-uniform priorities in the prioritised one: slot by slot the
    five arrays (and their dtypes) are the base class's."""
    base, per = ReplayBuffer(6, TAG_SHAPE), PrioritizedReplayBuffer(6, TAG_SHAPE, 1.0, 0.5)
    rng = np.random.default_rng(3)
    t = 0
    for _ in range(12):
        n = int(rng.integers(1, 9))
        base.add(*tagged(range(t, t + n)))
        per.add(*tagged(range(t, t + n)))
        t += n
    per.update_priorities(np.arange(6), rng.random(6).astype(np.float32) * 3)
    reference = stored(base)
    for slot in range(6):
        batch = read_slot(per, slot)
        for name in BATCH_FIELDS:
            got, want = getattr(batch, name), getattr(reference, name)[slot : slot + 1]
            assert got.dtype == want.dtype and np.array_equal(got, want), (slot, name)


def test_dtypes_do_not_depend_on_the_dtypes_added():
    buffer = PrioritizedReplayBuffer(4, TAG_SHAPE)
    obs, ids, rewards, next_obs, terminated = tagged(range(3))
    buffer.add(obs.astype(np.float64), ids.astype(np.int32), rewards.astype(np.float64), next_obs.astype(np.float64),
               terminated)
    batch = stored_per(buffer)
    assert ids_of(batch) == [0, 1, 2]
    assert [a.dtype for a in arrays(batch)] == [np.float32, np.int64, np.float32, np.float32, np.bool_]
    assert batch.indices.dtype == np.int64 and batch.weights.dtype == np.float32


def test_starvation_truncation_is_stored_as_in_the_base_class():
    obs, actions, reward, final_obs, terminated, reset_obs = starvation_truncation()
    buffer = PrioritizedReplayBuffer(4, (OBS_DIM,))
    buffer.add(obs, actions, reward, final_obs, terminated)
    batch = read_slot(buffer, 0)
    assert np.array_equal(batch.obs, obs) and np.array_equal(batch.next_obs, final_obs)
    assert not np.array_equal(batch.next_obs, reset_obs)
    assert batch.terminated.tolist() == [False] and batch.rewards.tolist() == [0.0]


# ------------------------------------------------------------- sample: the RNG protocol
@pytest.mark.parametrize("batch_size", [1, 7, 64, 500])
def test_sample_follows_the_rng_protocol_one_uniform_draw(batch_size):
    """``u = rng.random(batch_size)``, one call per ``sample``: the indices are the stratified formula on
    the twin generator's draw, the batch holds those slots, the weights follow, and both generators end in
    the same state."""
    # Priorities 1 2 4 3 1 8 1 (ids 0 .. 6 in slots 0 .. 6), total 20.
    buffer = with_priorities([0.5, 1.5, 3.5, 2.5, 0.5, 7.5, 0.5])
    p = buffer.priorities
    assert p.tolist() == [1, 2, 4, 3, 1, 8, 1]
    rng, twin = np.random.default_rng(5), np.random.default_rng(5)
    for beta in (0.5, 1.0, 0.0):
        batch = buffer.sample(batch_size, rng, beta)
        u = twin.random(batch_size)
        idx = stratified_indices(p, batch_size, u)
        assert batch.indices.tolist() == idx.tolist()
        assert ids_of(batch) == idx.tolist()  # the slot holds id = slot here
        assert np.allclose(batch.weights, is_weights(p, idx, beta), rtol=1e-6)
        assert rng.bit_generator.state == twin.bit_generator.state
    # The scripted generator: exactly one random(batch_size) call, no integers call.
    scripted = ScriptedUniform(np.linspace(0.0, 0.999, batch_size))
    batch = buffer.sample(batch_size, scripted, 0.5)
    assert scripted.calls == [batch_size]
    assert batch.indices.tolist() == stratified_indices(p, batch_size, scripted.u).tolist()

    # Full and wrapped with equal priorities: index = storage slot (ids 5 6 2 3 4 after 3 + 4 adds into 5 slots).
    ring = PrioritizedReplayBuffer(5, TAG_SHAPE)
    ring.add(*tagged(range(3)))
    ring.add(*tagged(range(3, 7)))
    slots = np.array([5, 6, 2, 3, 4])
    for _ in range(2):
        batch = ring.sample(batch_size, rng, 0.4)
        idx = stratified_indices(ring.priorities, batch_size, twin.random(batch_size))
        assert ids_of(batch) == slots[idx].tolist() and batch.indices.tolist() == idx.tolist()
        assert batch.weights.tolist() == [1.0] * batch_size  # equal priorities: every weight is 1
        assert rng.bit_generator.state == twin.bit_generator.state


def test_stratified_slot_formula_by_hand():
    """Priorities 1 2 4 3 (cumsum 1 3 7 10, total 10) and scripted u: the mass of sample k is
    (k + u_k) / B * 10 and its slot the first whose cumulative mass exceeds it."""
    buffer = with_priorities([0.5, 1.5, 3.5, 2.5])
    assert buffer.priorities.tolist() == [1.0, 2.0, 4.0, 3.0]
    # B = 5: mass [0, 2.5, 4, 7.8, 9.98] -> slots [0, 1, 2, 3, 3].
    batch = buffer.sample(5, ScriptedUniform([0.0, 0.25, 0.0, 0.9, 0.99]), 1.0)
    assert batch.indices.tolist() == [0, 1, 2, 3, 3] and ids_of(batch) == [0, 1, 2, 3, 3]
    assert batch.indices.dtype == np.int64 and batch.weights.dtype == np.float32
    # beta = 1: w = (p / p_min) ** -1 = [1, 1/2, 1/4, 1/3, 1/3].
    assert batch.weights.tolist() == pytest.approx([1.0, 0.5, 0.25, 1 / 3, 1 / 3], rel=1e-6)
    # B = 2: mass [2.5, 7.5] -> slots [1, 3]; B = 1 with u near 1: mass 9.99999 -> slot 3.
    assert buffer.sample(2, ScriptedUniform([0.5, 0.5]), 0.0).indices.tolist() == [1, 3]
    assert buffer.sample(1, ScriptedUniform([0.999999]), 0.0).indices.tolist() == [3]
    # A scripted u of exactly 1.0 (a real generator never returns it): the mass equals the total and
    # searchsorted gives n; the clip to n - 1 keeps the draw inside the storage.
    assert buffer.sample(1, ScriptedUniform([1.0]), 0.0).indices.tolist() == [3]
    assert buffer.sample(4, ScriptedUniform([1.0, 1.0, 1.0, 1.0]), 0.0).indices.tolist() == [1, 2, 3, 3]
    # Every sample k is drawn from the slots whose mass covers the k-th of B equal segments of [0, total).
    rng = np.random.default_rng(0)
    cumsum = np.concatenate([[0.0], np.cumsum(buffer.priorities)])
    for batch_size in (3, 10, 25):
        batch = buffer.sample(batch_size, rng, 0.0)
        for k, slot in enumerate(batch.indices.tolist()):
            lo, hi = k / batch_size * 10.0, (k + 1) / batch_size * 10.0
            assert cumsum[slot] < hi and cumsum[slot + 1] > lo, (batch_size, k, slot)


def test_slot_frequencies_are_proportional_to_the_priorities():
    buffer = with_priorities([0.5, 1.5, 3.5, 2.5])  # p = 1 2 4 3, P = 0.1 0.2 0.4 0.3
    want = np.array([0.1, 0.2, 0.4, 0.3])
    rng = np.random.default_rng(11)
    counts = np.zeros(4)
    for _ in range(400):
        batch = buffer.sample(50, rng, 0.4)
        counts += np.bincount(batch.indices, minlength=4)
        assert ids_of(batch) == batch.indices.tolist()
    assert np.abs(counts / 20_000 - want).max() < 0.01, counts / 20_000
    # One big stratified batch pins the frequencies to within 2 / B.
    big = buffer.sample(20_000, rng, 1.0)
    assert np.abs(np.bincount(big.indices, minlength=4) / 20_000 - want).max() < 0.001
    # With alpha = 0 every priority is 1 whatever the TD errors: uniform sampling.
    flat = filled(4, range(4), alpha=0.0)
    flat.update_priorities(np.arange(4), np.array([0.0, 1.0, 10.0, 100.0]))
    assert flat.priorities.tolist() == [1.0] * 4
    frequency = np.bincount(flat.sample(20_000, rng, 1.0).indices, minlength=4) / 20_000
    assert np.abs(frequency - 0.25).max() < 0.001


# ------------------------------------------------------------- weights
@pytest.mark.parametrize("beta", [0.0, 0.4, 0.5, 1.0])
def test_weights_formula(beta):
    buffer = with_priorities([0.5, 1.5, 3.5, 2.5])  # p = 1 2 4 3, p_min = 1
    p = buffer.priorities
    # B = 4 with u = 0.5: mass [1.25, 3.75, 6.25, 8.75] -> slots [1, 2, 2, 3].
    batch = buffer.sample(4, ScriptedUniform([0.5] * 4), beta)
    assert batch.indices.tolist() == [1, 2, 2, 3]
    want = np.array([2.0, 4.0, 4.0, 3.0]) ** (-beta)
    assert batch.weights.dtype == np.float32 and batch.weights.shape == (4,)
    assert np.allclose(batch.weights, want, rtol=1e-6) and np.allclose(batch.weights, is_weights(p, batch.indices, beta), rtol=1e-6)
    assert ((batch.weights > 0) & (batch.weights <= 1)).all()
    # The same thing as (n P(i)) ** -beta normalised by its maximum over the whole buffer.
    n, total = len(p), p.sum()
    unnormalised = (n * p / total) ** (-beta)
    assert np.allclose(batch.weights, (unnormalised / unnormalised.max())[batch.indices], rtol=1e-6)
    # The slot with the minimum priority has weight 1; at beta 0 every weight is 1.
    assert read_slot(buffer, 0, beta).weights.tolist() == [1.0]
    if beta == 0.0:
        assert batch.weights.tolist() == [1.0] * 4
    else:
        assert batch.weights.max() < 1.0  # slots 1 .. 3 are all above the minimum
    # Equal priorities: every weight is 1 for any beta.
    assert filled(4, range(4)).sample(6, np.random.default_rng(0), beta).weights.tolist() == [1.0] * 6


def test_weights_shrink_with_beta_for_a_high_priority_slot():
    buffer = with_priorities([0.5, 1.5, 3.5, 2.5])
    weights = [read_slot(buffer, 2, beta).weights[0] for beta in (0.0, 0.25, 0.5, 1.0)]
    assert weights == pytest.approx([1.0, 4.0**-0.25, 0.5, 0.25], rel=1e-6)


# ------------------------------------------------------------- update_priorities
def test_update_priorities_arithmetic_and_max_priority_growth():
    buffer = filled(8, range(6))  # alpha 0.6, eps 1e-6
    assert buffer.priorities.tolist() == [1.0] * 6 and buffer.max_priority == 1.0
    buffer.update_priorities(np.array([0, 2, 5]), np.array([0.0, 0.5, 2.0], dtype=np.float32))
    p = buffer.priorities
    want = (np.array([0.0, 0.5, 2.0]) + 1e-6) ** 0.6
    assert p[[0, 2, 5]].tolist() == pytest.approx(want.tolist(), rel=1e-9)
    assert p[[1, 3, 4]].tolist() == [1.0] * 3  # untouched slots
    assert p.dtype == np.float64 and p.shape == (6,)
    assert buffer.max_priority == pytest.approx((2.0 + 1e-6) ** 0.6, rel=1e-12) and type(buffer.max_priority) is float
    assert buffer.max_priority > 1.0
    # |td|: a negative value counts by its magnitude.
    buffer.update_priorities(np.array([1]), np.array([-3.0]))
    assert buffer.priorities[1] == pytest.approx((3.0 + 1e-6) ** 0.6, rel=1e-9)
    assert buffer.max_priority == pytest.approx((3.0 + 1e-6) ** 0.6, rel=1e-12)
    # The maximum is the largest priority ever assigned: lowering a slot does not lower it.
    buffer.update_priorities(np.array([1]), np.array([0.0]))
    assert buffer.priorities[1] == pytest.approx(1e-6**0.6, rel=1e-9)
    assert buffer.max_priority == pytest.approx((3.0 + 1e-6) ** 0.6, rel=1e-12)
    # A new transition arrives with that maximum.
    buffer.add(*tagged([6]))
    assert buffer.priorities[6] == buffer.max_priority
    # Lists and numpy integer dtypes are fine; the stored priorities are already raised to alpha.
    buffer.update_priorities(np.array([3], dtype=np.int32), np.array([7.0], dtype=np.float64))
    assert buffer.priorities[3] == pytest.approx((7.0 + 1e-6) ** 0.6, rel=1e-9)


def test_update_priorities_with_alpha_one_and_eps_half_is_exact():
    buffer = with_priorities([0.5, 1.5, 3.5, 2.5])
    assert buffer.priorities.tolist() == [1.0, 2.0, 4.0, 3.0] and buffer.max_priority == 4.0
    buffer.update_priorities(np.array([3, 1]), np.array([9.5, 0.0], dtype=np.float32))
    assert buffer.priorities.tolist() == [1.0, 0.5, 4.0, 10.0] and buffer.max_priority == 10.0


def test_update_priorities_duplicate_indices_last_value_wins():
    buffer = with_priorities([0.5, 1.5, 3.5, 2.5])
    buffer.update_priorities(np.array([2, 0, 2]), np.array([0.5, 1.0, 3.0]))
    assert buffer.priorities.tolist() == [1.5, 2.0, 3.5, 3.0]
    buffer.update_priorities(np.array([2, 2]), np.array([19.5, 0.5]))
    assert buffer.priorities[2] == 1.0  # the last value
    assert buffer.max_priority == 20.0  # max(max_priority, p.max()) over all values given


@pytest.mark.parametrize(
    "indices, td_abs",
    [
        (np.array([0, 1]), np.array([0.5])),  # shapes differ
        (np.array([[0, 1]]), np.array([[0.5, 0.5]])),  # not 1-D
        (np.array([0]), np.array([[0.5]])),
        (np.array([-1]), np.array([0.5])),  # outside 0 .. len - 1
        (np.array([4]), np.array([0.5])),  # == len(self) (the buffer holds 4 of 8 slots)
        (np.array([7]), np.array([0.5])),  # an allocated but unwritten slot
        (np.array([0]), np.array([np.nan])),  # not finite
        (np.array([0]), np.array([np.inf])),
        (np.array([0, 1]), np.array([0.5, -np.inf])),
    ],
    ids=["shape-mismatch", "2d", "2d-td", "negative-index", "index-len", "index-unwritten", "nan", "inf", "-inf"],
)
def test_update_priorities_errors(indices, td_abs):
    buffer = filled(8, range(4))
    with pytest.raises(ValueError):
        buffer.update_priorities(indices, td_abs)


def test_priorities_property_is_a_copy():
    buffer = with_priorities([0.5, 1.5, 3.5, 2.5])
    p = buffer.priorities
    p[:] = 99.0
    assert buffer.priorities.tolist() == [1.0, 2.0, 4.0, 3.0]
    assert buffer.sample(1, ScriptedUniform([0.05]), 0.0).indices.tolist() == [0]  # the storage was not touched


# ------------------------------------------------------------- PrioritizedBatch
def test_prioritized_batch_is_a_batch_with_indices_and_weights():
    assert dataclasses.is_dataclass(PrioritizedBatch) and issubclass(PrioritizedBatch, Batch)
    assert [f.name for f in dataclasses.fields(PrioritizedBatch)] == list(PER_FIELDS)
    obs = np.zeros((2, 3), np.float32)
    batch = PrioritizedBatch(obs=obs, actions=np.zeros(2, np.int64), rewards=np.zeros(2, np.float32), next_obs=obs,
                             terminated=np.zeros(2, bool), indices=np.array([0, 1]), weights=np.ones(2, np.float32))
    assert isinstance(batch, Batch) and batch.indices.tolist() == [0, 1]
    with pytest.raises(dataclasses.FrozenInstanceError):
        batch.weights = np.zeros(2)


@pytest.mark.parametrize("obs_shape", [(11,), (3, 8, 8)])
def test_sample_shapes_dtypes_and_fresh_copies(obs_shape):
    rng = np.random.default_rng(0)
    buffer = PrioritizedReplayBuffer(20, obs_shape)
    n = 6
    obs = rng.random((n, *obs_shape)).astype(np.float32)
    buffer.add(obs, rng.integers(0, 3, size=n), rng.normal(size=n).astype(np.float32), rng.random((n, *obs_shape)).astype(np.float32),
               rng.random(n) < 0.5)
    for batch_size in (1, 4, 50):  # more than the buffer holds: with replacement
        batch = buffer.sample(batch_size, rng, 0.5)
        assert isinstance(batch, PrioritizedBatch) and isinstance(batch, Batch)
        assert batch.obs.shape == (batch_size, *obs_shape) and batch.obs.dtype == np.float32
        assert batch.next_obs.shape == (batch_size, *obs_shape) and batch.next_obs.dtype == np.float32
        assert batch.actions.shape == (batch_size,) and batch.actions.dtype == np.int64
        assert batch.rewards.shape == (batch_size,) and batch.rewards.dtype == np.float32
        assert batch.terminated.shape == (batch_size,) and batch.terminated.dtype == np.bool_
        assert batch.indices.shape == (batch_size,) and batch.indices.dtype == np.int64
        assert batch.weights.shape == (batch_size,) and batch.weights.dtype == np.float32
        assert ((batch.indices >= 0) & (batch.indices < n)).all()
        assert np.array_equal(batch.obs, obs[batch.indices])
    first = buffer.sample(16, np.random.default_rng(1), 0.5)
    kept = [array.copy() for array in (*arrays(first), first.indices, first.weights)]
    second = buffer.sample(16, np.random.default_rng(2), 0.5)  # a later sample leaves the earlier batch alone
    for array, copy_ in zip((*arrays(first), first.indices, first.weights), kept):
        assert np.array_equal(array, copy_)
    for a in (*arrays(first), first.indices, first.weights):
        for b in (*arrays(second), second.indices, second.weights):
            assert not np.shares_memory(a, b)
    assert not np.shares_memory(first.obs, first.next_obs)
    for array in (*arrays(first), first.indices, first.weights):  # writing into a batch reaches nothing
        array[...] = 1
    again = buffer.sample(16, np.random.default_rng(1), 0.5)
    for array, copy_ in zip((*arrays(again), again.indices, again.weights), kept):
        assert np.array_equal(array, copy_)
    assert buffer.priorities.tolist() == [1.0] * n


def test_sample_draws_only_stored_transitions_while_filling():
    buffer = PrioritizedReplayBuffer(1000, TAG_SHAPE)
    buffer.add(*tagged(range(1, 8)))  # ids 1 .. 7: an unwritten slot (all zeros) is not a valid transition
    batch = buffer.sample(3000, np.random.default_rng(0), 0.5)
    assert set(ids_of(batch)) == set(range(1, 8)) and set(batch.indices.tolist()) == set(range(7))
    assert buffer.priorities.shape == (7,)


# ------------------------------------------------------------- the package
def test_package_exports_the_prioritised_buffer_and_stays_torch_free():
    code = (
        "import sys\n"
        "import snake_rl.agents.dqn\n"
        "from snake_rl.agents.dqn import Batch, PrioritizedBatch, PrioritizedReplayBuffer, ReplayBuffer\n"
        "import snake_rl.agents.dqn.replay as replay\n"
        "assert PrioritizedReplayBuffer is replay.PrioritizedReplayBuffer and PrioritizedBatch is replay.PrioritizedBatch\n"
        "assert {'PrioritizedBatch', 'PrioritizedReplayBuffer'} <= set(snake_rl.agents.dqn.__all__)\n"
        "b = PrioritizedReplayBuffer(4, (2,))\n"
        "import numpy as np\n"
        "b.add(np.zeros((2, 2), np.float32), np.zeros(2, np.int64), np.zeros(2, np.float32), np.zeros((2, 2), np.float32), np.zeros(2, bool))\n"
        "batch = b.sample(3, np.random.default_rng(0), 0.5)\n"
        "assert isinstance(batch, PrioritizedBatch) and batch.weights.tolist() == [1.0] * 3\n"
        "print('torch-loaded' if 'torch' in sys.modules else 'torch-free')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "torch-free"


def test_module_docstring_explains_the_choice_against_a_sum_tree():
    import snake_rl.agents.dqn.replay as replay

    doc = replay.__doc__.lower()
    assert "tree" in doc and "cumulative" in doc
