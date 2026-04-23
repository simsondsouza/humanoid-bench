"""
FPO on HumanoidBench Gymnasium environments.

Ports the FPO algorithm (fpo/playground/src/flow_policy/) to work with
standard Gymnasium environments from HumanoidBench, replacing the JAX-based
MuJoCo Playground interface with CPU Gymnasium.

Files left UNCHANGED:
  fpo/playground/src/flow_policy/fpo.py
  fpo/playground/src/flow_policy/rollouts.py
  fpo/playground/src/flow_policy/networks.py
  fpo/playground/src/flow_policy/math_utils.py

Only the environment creation and rollout/eval helpers are swapped out.
The training loop structure (rollout → training_step → wandb log) is
identical to fpo/playground/scripts/train_fpo.py.

Usage:
    python fpo/run_fpo_humanoidbench.py \\
        --env_name g1-walk-v0 \\
        --wandb_entity intro_to_robot_learning_cmu \\
        --seed 0
"""

import os
import site
import sys

# ---------------------------------------------------------------------------
# GPU setup: pip-installed nvidia packages put cuDNN/CUDA libs in site-packages
# but don't add them to LD_LIBRARY_PATH.  JAX's CUDA plugin can't find cuDNN
# without them.  We set LD_LIBRARY_PATH and re-exec once so the dynamic linker
# picks up the libs before any CUDA code is loaded.
# ---------------------------------------------------------------------------
_nvidia_lib_dirs = [
    os.path.join(sp, "nvidia", pkg, "lib")
    for sp in site.getsitepackages()
    for pkg in ("cudnn", "cublas", "cuda_runtime", "cufft", "curand", "cusolver", "cusparse")
    if os.path.isdir(os.path.join(sp, "nvidia", pkg, "lib"))
]
if _nvidia_lib_dirs:
    _ld = os.environ.get("LD_LIBRARY_PATH", "")
    _missing = [d for d in _nvidia_lib_dirs if d not in _ld]
    if _missing:
        os.environ["LD_LIBRARY_PATH"] = ":".join(_missing) + (":" + _ld if _ld else "")
        os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse
import datetime
import time
import warnings
from pathlib import Path

# Suppress XLA/CUDA C++ log noise (e.g. WSL2 driver version format warnings).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import gymnasium as gym
import jax
import jax_dataclasses as jdc
import numpy as onp
import wandb
from jax import numpy as jnp
from tqdm import tqdm

# Suppress GLFW headless-display warning from humanoid_bench rendering code.
warnings.filterwarnings("ignore", message=".*GLFW.*", category=UserWarning)
import humanoid_bench  # noqa: F401 — registers HumanoidBench gym environments

# Make `flow_policy` importable from fpo/playground/src/.
sys.path.insert(0, str(Path(__file__).parent / "playground" / "src"))
# Provide a minimal stub for mujoco_playground so that fpo.py / rollouts.py
# can be imported without the actual mujoco_playground package installed.
# The stub only defines the type names used in annotations; no real MJX code
# is ever called because we replace BatchedRolloutState and eval_policy.
sys.path.insert(0, str(Path(__file__).parent / "_stubs"))

from flow_policy import fpo, rollouts
from flow_policy.fpo import FpoConfig, FpoState


# ---------------------------------------------------------------------------
# 1.  Thin env adapter
#     FpoState.init() and sample_action() only read two attributes from the
#     env object: `observation_size` and `action_size`.  This adapter
#     provides exactly those, wrapping any gym.Env.
# ---------------------------------------------------------------------------

class GymEnvAdapter:
    """
    Exposes `observation_size` / `action_size` so FpoState (and the flow
    policy networks) can be initialised from a Gymnasium environment without
    any changes to fpo.py.

    FpoState stores this object in a `jdc.Static` field, which means JAX
    uses it as a compile-time cache key — hence __hash__ / __eq__.
    """

    def __init__(self, env: gym.Env) -> None:
        obs_shape = env.observation_space.shape
        act_shape = env.action_space.shape
        assert obs_shape is not None and len(obs_shape) == 1, (
            "Only flat (Box) observation spaces are supported."
        )
        assert act_shape is not None and len(act_shape) == 1, (
            "Only flat (Box) action spaces are supported."
        )
        self.observation_size: int = int(obs_shape[0])
        self.action_size: int = int(act_shape[0])

    def __hash__(self) -> int:
        return hash((self.observation_size, self.action_size))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GymEnvAdapter):
            return NotImplemented
        return (
            self.observation_size == other.observation_size
            and self.action_size == other.action_size
        )

    def __repr__(self) -> str:
        return (
            f"GymEnvAdapter(obs={self.observation_size}, act={self.action_size})"
        )


# ---------------------------------------------------------------------------
# 2.  Gymnasium-based rollout state
#     Drop-in replacement for rollouts.BatchedRolloutState.
#     Produces TransitionStruct tensors with the exact (T, B, ...) shape
#     convention that FpoState.training_step() expects — no changes to
#     fpo.py required.
# ---------------------------------------------------------------------------

class GymBatchedRolloutState:
    """
    CPU rollout using gymnasium.vector.SyncVectorEnv.

    The .rollout() method collects `iterations_per_env` steps from each of
    `num_envs` parallel environments and returns a TransitionStruct whose
    arrays have leading shape (iterations_per_env, num_envs, ...).

    BEFORE (mujoco_playground):
        rollout_state = rollouts.BatchedRolloutState.init(env, prng, num_envs)
    AFTER (gymnasium):
        rollout_state = GymBatchedRolloutState.init(env_fn, prng, num_envs)
    """

    def __init__(
        self,
        vec_env: gym.vector.VectorEnv,
        obs: onp.ndarray,    # (num_envs, obs_dim)  current observations
        steps: onp.ndarray,  # (num_envs,)          steps since last reset
        prng: jax.Array,
        num_envs: int,
    ) -> None:
        self.vec_env = vec_env
        self.obs = obs
        self.steps = steps
        self.prng = prng
        self.num_envs = num_envs

    @staticmethod
    def init(
        env_fn,         # callable () -> gym.Env
        prng: jax.Array,
        num_envs: int,
    ) -> "GymBatchedRolloutState":
        vec_env = gym.vector.SyncVectorEnv(
            [env_fn for _ in range(num_envs)]
        )
        obs, _ = vec_env.reset()
        return GymBatchedRolloutState(
            vec_env=vec_env,
            obs=onp.array(obs, dtype=onp.float32),
            steps=onp.zeros(num_envs, dtype=onp.int32),
            prng=prng,
            num_envs=num_envs,
        )

    def rollout(
        self,
        agent_state: FpoState,
        episode_length: int,
        iterations_per_env: int,
        auto_reset: bool = True,
        deterministic: bool = False,
    ) -> tuple["GymBatchedRolloutState", rollouts.TransitionStruct]:
        """
        Collect `iterations_per_env` steps across `num_envs` environments.

        Gymnasium auto-resets terminated/truncated environments, so we
        recover the true final observation from info["final_observation"]
        when present — this keeps the value-bootstrap target correct on
        truncated episodes (same logic as rollouts.py).

        The policy is called with a batched (B, obs_dim) JAX array; tanh
        squashing is applied before passing actions to the environment,
        consistent with rollouts.py line: `jnp.tanh(action)`.
        """
        T = iterations_per_env
        B = self.num_envs
        obs_dim = agent_state.env.observation_size
        act_dim = agent_state.env.action_size

        # Pre-allocate numpy buffers for speed.
        buf_obs        = onp.empty((T, B, obs_dim), dtype=onp.float32)
        buf_next_obs   = onp.empty((T, B, obs_dim), dtype=onp.float32)
        buf_actions    = onp.empty((T, B, act_dim), dtype=onp.float32)
        buf_rewards    = onp.empty((T, B),          dtype=onp.float32)
        buf_truncation = onp.empty((T, B),          dtype=onp.float32)
        buf_discount   = onp.empty((T, B),          dtype=onp.float32)
        action_infos   = []

        obs   = self.obs
        steps = self.steps.copy()
        prng  = self.prng

        for t in range(T):
            prng, prng_act = jax.random.split(prng)

            # Query policy with the full batch as a JAX array.
            jax_obs = jnp.array(obs)                               # (B, obs_dim)
            action, action_info = agent_state.sample_action(
                jax_obs, prng_act, deterministic=deterministic
            )

            # Apply tanh squashing before stepping (matches rollouts.py).
            np_action = onp.array(jnp.tanh(action), dtype=onp.float32)  # (B, act_dim)

            # Step all environments simultaneously.
            next_obs_raw, reward, terminated, truncated, info = (
                self.vec_env.step(np_action)
            )

            done_or_tr = terminated | truncated
            steps += 1

            # Gymnasium auto-resets: next_obs_raw for done envs is already the
            # first obs of the new episode.  Retrieve the true final obs (used
            # for value bootstrapping on truncated episodes).
            next_obs = next_obs_raw.astype(onp.float32).copy()
            if "final_observation" in info:
                for env_idx in range(B):
                    if done_or_tr[env_idx]:
                        final = info["final_observation"][env_idx]
                        if final is not None:
                            next_obs[env_idx] = onp.asarray(final, dtype=onp.float32)

            buf_obs[t]        = obs
            buf_next_obs[t]   = next_obs
            buf_actions[t]    = onp.array(action, dtype=onp.float32)
            buf_rewards[t]    = reward.astype(onp.float32)
            buf_truncation[t] = truncated.astype(onp.float32)
            # discount = 1 on ongoing/truncated steps, 0 on true termination
            buf_discount[t]   = (1.0 - terminated.astype(onp.float32))
            action_infos.append(action_info)

            obs   = next_obs_raw.astype(onp.float32)
            steps = onp.where(done_or_tr, 0, steps)

        # Stack the per-step action_infos (each has batch dim B) into
        # shape (T, B, ...) by stacking along a new leading axis.
        stacked_action_info = jax.tree.map(
            lambda *xs: jnp.stack(xs, axis=0), *action_infos
        )

        transitions = rollouts.TransitionStruct(
            obs=jnp.array(buf_obs),
            next_obs=jnp.array(buf_next_obs),
            action=jnp.array(buf_actions),
            action_info=stacked_action_info,
            reward=jnp.array(buf_rewards),
            truncation=jnp.array(buf_truncation),
            discount=jnp.array(buf_discount),
        )

        new_state = GymBatchedRolloutState(
            vec_env=self.vec_env,
            obs=obs,
            steps=steps,
            prng=prng,
            num_envs=self.num_envs,
        )
        return new_state, transitions


# ---------------------------------------------------------------------------
# 3.  Gymnasium-based policy evaluation
#     Replaces rollouts.eval_policy().  Returns a rollouts.EvalOutputs so
#     the printing and wandb logging in the training loop work unchanged.
# ---------------------------------------------------------------------------

def gym_eval_policy(
    agent_state: FpoState,
    env_fn,            # callable () -> gym.Env
    prng: jax.Array,
    num_episodes: int = 10,
    max_episode_length: int = 1000,
) -> rollouts.EvalOutputs:
    """
    Run the policy deterministically for `num_episodes` sequential episodes.

    Returns rollouts.EvalOutputs so the training loop's eval block (print
    statements + eval_outputs.log_to_wandb) works without modification.
    """
    env = env_fn()
    action_dim = agent_state.env.action_size

    episode_rewards: list[float] = []
    episode_lengths: list[int]   = []
    # Padded action buffer for EvalOutputs (shape matches rollouts.py convention)
    actions_padded = onp.zeros(
        (max_episode_length, num_episodes, action_dim), dtype=onp.float32
    )
    valid_mask = onp.zeros((max_episode_length, num_episodes), dtype=onp.float32)

    for ep in range(num_episodes):
        obs, _ = env.reset()
        ep_reward = 0.0
        ep_length = 0

        while ep_length < max_episode_length:
            prng, prng_act = jax.random.split(prng)
            # Add batch dim of 1 for sample_action's (*batch_dims, obs_dim) API.
            jax_obs = jnp.array(obs[None])              # (1, obs_dim)
            action, _ = agent_state.sample_action(
                jax_obs, prng_act, deterministic=True
            )
            np_action = onp.array(jnp.tanh(action[0]))  # (act_dim,)

            obs, reward, terminated, truncated, _ = env.step(np_action)
            ep_reward += float(reward)
            actions_padded[ep_length, ep] = np_action
            valid_mask[ep_length, ep]     = 1.0
            ep_length += 1

            if terminated or truncated:
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)

    env.close()

    rewards_arr = onp.array(episode_rewards, dtype=onp.float32)
    steps_arr   = onp.array(episode_lengths,  dtype=onp.float32)

    scalar_metrics = {
        "reward_mean": jnp.array(rewards_arr.mean()),
        "reward_min":  jnp.array(rewards_arr.min()),
        "reward_max":  jnp.array(rewards_arr.max()),
        "reward_std":  jnp.array(rewards_arr.std()),
        "steps_mean":  jnp.array(steps_arr.mean()),
        "steps_min":   jnp.array(steps_arr.min()),
        "steps_max":   jnp.array(steps_arr.max()),
        "steps_std":   jnp.array(steps_arr.std()),
    }
    histogram_metrics = {
        "reward": jnp.array(rewards_arr),
        "steps":  jnp.array(steps_arr),
    }

    return rollouts.EvalOutputs(
        scalar_metrics=scalar_metrics,
        histogram_metrics=histogram_metrics,
        actions=jnp.array(actions_padded),
        action_timestep_mask=jnp.array(valid_mask),
    )


# ---------------------------------------------------------------------------
# 4.  CLI & main
# ---------------------------------------------------------------------------

def make_env_fn(env_name: str):
    """Returns a zero-argument callable that creates a new gym.Env."""
    def _make() -> gym.Env:
        return gym.make(env_name)
    return _make


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train FPO on a HumanoidBench Gymnasium environment."
    )
    # Required
    p.add_argument("--env_name",      type=str, required=True,
                   help="Gymnasium env id, e.g. g1-walk-v0 or g1-stair-v0")
    p.add_argument("--wandb_entity",  type=str, required=True,
                   help="WandB entity (username or team name)")
    # Optional with sensible defaults
    p.add_argument("--wandb_project", type=str, default="humanoid-bench",
                   help="WandB project name")
    p.add_argument("--exp_name",      type=str, default="",
                   help="Optional experiment name suffix in WandB run name")
    p.add_argument("--seed",          type=int, default=0)
    p.add_argument("--max_steps",     type=int, default=20000000,
                   help="Total environment steps")
    # FpoConfig overrides — reduced from JAX defaults to suit CPU Gymnasium
    p.add_argument("--num_envs",        type=int, default=8,
                   help="Parallel CPU environments (default 8; JAX default is 2048)")
    p.add_argument("--num_minibatches", type=int, default=4,
                   help="Minibatches per update (default 4; JAX default is 32)")
    p.add_argument("--batch_size",      type=int, default=256,
                   help="Minibatch size (default 256; JAX default is 1024)")
    p.add_argument("--num_evals",       type=int, default=10,
                   help="Number of evaluation checkpoints")
    p.add_argument("--eval_episodes",   type=int, default=10,
                   help="Episodes per evaluation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # Build FpoConfig.
    # We keep all algorithm hyper-parameters at their paper defaults but
    # reduce num_envs / num_minibatches / batch_size so that
    # iterations_per_env = (num_minibatches * batch_size * unroll_length)
    #                      // num_envs
    # stays at a manageable size for a CPU Gymnasium loop.
    #
    # With the CLI defaults (num_envs=8, batch_size=256, num_minibatches=4):
    #   iterations_per_env = (4 * 256 * 30) // 8 = 3840
    #   outer_iters        = 10_000_000 // (3840 * 8) ≈ 325
    # ------------------------------------------------------------------
    config = FpoConfig(
        num_envs=args.num_envs,
        num_minibatches=args.num_minibatches,
        batch_size=args.batch_size,
        num_timesteps=args.max_steps,
        num_evals=args.num_evals,
    )

    # ------------------------------------------------------------------
    # Environment setup.
    #   BEFORE: env = registry.load(env_name, config=env_config)
    #   AFTER:  env = gym.make(env_name)   (via adapter + rollout wrapper)
    # ------------------------------------------------------------------
    probe_env = gym.make(args.env_name)
    gym_adapter = GymEnvAdapter(probe_env)
    probe_env.close()

    env_fn = make_env_fn(args.env_name)

    # ------------------------------------------------------------------
    # WandB — identical to train_fpo.py.
    # ------------------------------------------------------------------
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    wandb_run = wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=f"fpo_{args.env_name}_{args.exp_name}_{timestamp}",
        config={
            "env_name":          args.env_name,
            "fpo_params":        jdc.asdict(config),
            "learning_rate":     config.learning_rate,
            "clipping_epsilon":  config.clipping_epsilon,
            "seed":              args.seed,
        },
    )

    # ------------------------------------------------------------------
    # Initialise agent and rollout state.
    # ------------------------------------------------------------------
    agent_state = FpoState.init(
        prng=jax.random.key(args.seed),
        env=gym_adapter,      # adapter provides observation_size / action_size
        config=config,
    )
    rollout_state = GymBatchedRolloutState.init(
        env_fn=env_fn,
        prng=jax.random.key(args.seed),
        num_envs=config.num_envs,
    )

    # ------------------------------------------------------------------
    # Training loop — identical structure to train_fpo.py.
    # Only eval and rollout calls use Gymnasium-based helpers instead of
    # rollouts.eval_policy / rollouts.BatchedRolloutState.
    # ------------------------------------------------------------------
    outer_iters = max(1, config.num_timesteps // (config.iterations_per_env * config.num_envs))
    eval_iters  = set(onp.linspace(0, outer_iters - 1, config.num_evals, dtype=int))

    global_step = 0   # cumulative environment steps (for x-axis in plots)
    times = [time.time()]

    for i in tqdm(range(outer_iters)):

        # ---- Evaluation (same structure as train_fpo.py) ----
        if i in eval_iters:
            eval_outputs = gym_eval_policy(
                agent_state,
                env_fn=env_fn,
                prng=jax.random.fold_in(agent_state.prng, i),
                num_episodes=args.eval_episodes,
                max_episode_length=config.episode_length,
            )

            # Convert to numpy for printing — identical to train_fpo.py.
            s_np = {k: onp.array(v) for k, v in eval_outputs.scalar_metrics.items()}

            print(f"Eval metrics at step {i}:")
            print(
                f"  Reward: mean={s_np['reward_mean']:.2f},"
                f" min={s_np['reward_min']:.2f},"
                f" max={s_np['reward_max']:.2f},"
                f" std={s_np['reward_std']:.2f}"
            )
            print(
                f"  Steps:  mean={s_np['steps_mean']:.1f},"
                f" min={s_np['steps_min']:.1f},"
                f" max={s_np['steps_max']:.1f},"
                f" std={s_np['steps_std']:.1f}"
            )

            # Log to wandb — identical to train_fpo.py.
            eval_outputs.log_to_wandb(wandb_run, step=i)
            # Extra keys required for comparison plots.
            wandb_run.log(
                {
                    "results/return": float(s_np["reward_mean"]),
                    "global_step":    global_step,
                },
                step=i,
            )

        # ---- Training step — identical to train_fpo.py ----
        rollout_state, transitions = rollout_state.rollout(
            agent_state,
            episode_length=config.episode_length,
            iterations_per_env=config.iterations_per_env,
        )
        agent_state, metrics = agent_state.training_step(transitions)

        global_step += config.iterations_per_env * config.num_envs

        # ---- Train metric logging — identical to train_fpo.py ----
        wandb_run.log(
            {
                "train/mean_reward": onp.mean(transitions.reward),
                "train/mean_steps": (
                    # Approximate mean steps per episode (from train_fpo.py).
                    transitions.discount.size / jnp.sum(transitions.discount == 0.0)
                ),
                "train/reward_histogram": wandb.Histogram(
                    onp.array(transitions.reward.flatten()[::16])
                ),
                **{f"train/{k}": onp.mean(v) for k, v in metrics.items()},
                "global_step": global_step,
            },
            step=i,
        )

        times.append(time.time())

    if len(times) > 1:
        print("First train step time:", times[1] - times[0])
        print("~Train time:",           times[-1] - times[1])

    wandb_run.finish()


if __name__ == "__main__":
    main()
