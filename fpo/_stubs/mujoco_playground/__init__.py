"""
Minimal stub for mujoco_playground.

fpo.py and rollouts.py import this package only for type annotations
(MjxEnv, State).  Since we replace BatchedRolloutState and never call
the original rollouts.py rollout at runtime, no real implementation is
needed — the names just have to be importable.
"""


class MjxEnv:
    """Stub for mujoco_playground.MjxEnv (used as type annotation only)."""
    observation_size: int
    action_size: int


class State:
    """Stub for mujoco_playground.State (used as type annotation only)."""
    obs: object
    done: object
    reward: object
    data: object
