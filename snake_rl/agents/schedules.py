"""Hyper-parameter schedules shared by the learning agents.

Schedules are pure functions of the step counter: no state and no RNG, so a
training loop can recompute any past value and a test can replicate it.
"""

from __future__ import annotations

import numpy as np


def linear_schedule(step, start: float, end: float, decay_steps: float) -> float:
    """Linear interpolation from ``start`` (at step 0) to ``end`` (at ``decay_steps``).

    ``start + (end - start) * min(1.0, step / decay_steps)`` for
    ``decay_steps > 0``; constant afterwards. ``decay_steps <= 0`` means "no
    decay phase" and gives ``end`` for every step.

    ``step`` is a number (the result is a Python ``float``) or a numpy array
    of steps (the result is a float64 array of the same shape).

    The value on the plateau is the formula evaluated at fraction 1.0, i.e.
    ``start + (end - start)``, which can differ from ``end`` in the last bit
    (1.0 -> 0.05 gives 0.05000000000000004). It is constant, so nothing
    downstream depends on the difference.
    """
    if isinstance(step, np.ndarray):
        if decay_steps <= 0:
            return np.full(step.shape, end, dtype=np.float64)
        return start + (end - start) * np.minimum(1.0, step / decay_steps)
    if decay_steps <= 0:
        return float(end)
    return float(start + (end - start) * min(1.0, step / decay_steps))
