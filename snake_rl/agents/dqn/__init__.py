"""DQN (PLAN.md milestones M5-M7).

Modules::

    config.py    DQNConfig: the hyper-parameters (no torch)
    replay.py    ReplayBuffer, Batch: uniform replay on numpy arrays (no torch);
                 PrioritizedReplayBuffer, PrioritizedBatch: prioritised replay (M7)
    networks.py  MLP, CNN, DuelingMLP, build_q_network (torch)
    agent.py     DQNAgent: acting, the gradient step, checkpoints (torch)

Importing this package does not import torch, so configuration handling and
tabular runs stay light. The names below are the torch-free part; the agent is
imported from its module: ``from snake_rl.agents.dqn.agent import DQNAgent``.
"""

from snake_rl.agents.dqn.config import DQNConfig
from snake_rl.agents.dqn.replay import Batch, PrioritizedBatch, PrioritizedReplayBuffer, ReplayBuffer

__all__ = ["Batch", "DQNConfig", "PrioritizedBatch", "PrioritizedReplayBuffer", "ReplayBuffer"]
