# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train a PPO agent using JAX on the specified environment."""

from datetime import datetime
import functools
import json
import os
import time
import warnings

xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from etils import epath
from flax.training import orbax_utils
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco
import mujoco.viewer
from orbax import checkpoint as ocp
from tensorboardX import SummaryWriter
import wandb

import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Enable persistent compilation cache.
jax.config.update("jax_compilation_cache_dir", "./jax_cache")
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")


_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "LeapCubeReorient",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_VISION = flags.DEFINE_boolean("vision", False, "Use vision input")
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Path to load checkpoint from"
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train"
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    False,
    "Use Weights & Biases for logging (ignored in play-only mode)",
)
_USE_TB = flags.DEFINE_boolean(
    "use_tb", False, "Use TensorBoard for logging (ignored in play-only mode)"
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 1_000_000, "Number of timesteps"
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "Number of evaluations")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 0.1, "Reward scaling")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 10, "Unroll length")
_NUM_MINIBATCHES = flags.DEFINE_integer(
    "num_minibatches", 8, "Number of minibatches"
)
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    "num_updates_per_batch", 8, "Number of updates per batch"
)
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "Discounting")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 5e-4, "Learning rate")
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 5e-3, "Entropy cost")
_NUM_ENVS = flags.DEFINE_integer("num_envs", 1024, "Number of environments")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 128, "Number of evaluation environments"
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Batch size")
_MAX_GRAD_NORM = flags.DEFINE_float("max_grad_norm", 1.0, "Max grad norm")
_CLIPPING_EPSILON = flags.DEFINE_float(
    "clipping_epsilon", 0.2, "Clipping epsilon for PPO"
)
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes",
    [64, 64, 64],
    "Policy hidden layer sizes",
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes",
    [64, 64, 64],
    "Value hidden layer sizes",
)
_POLICY_OBS_KEY = flags.DEFINE_string(
    "policy_obs_key", "state", "Policy obs key"
)
_VALUE_OBS_KEY = flags.DEFINE_string("value_obs_key", "state", "Value obs key")


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
  if env_name in mujoco_playground.manipulation._envs:
    return manipulation_params.brax_ppo_config(env_name)
  elif env_name in mujoco_playground.locomotion._envs:
    return locomotion_params.brax_ppo_config(env_name)
  elif env_name in mujoco_playground.dm_control_suite._envs:
    return dm_control_suite_params.brax_ppo_config(env_name)

  raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


def main(argv):
  """Run training and evaluation for the specified environment."""

  del argv

  # Load environment configuration
  env_cfg = registry.get_default_config(_ENV_NAME.value)
  env_cfg.pert_config.enable = True
  env_cfg.pert_config.velocity_kick = [3.0, 6.0]
  env_cfg.pert_config.kick_wait_times = [5.0, 15.0]
  env_cfg.command_config.a = [1.5, 0.8, 2*jp.pi]

  velocity_kick_range = [0.0, 0.0]  # Disable velocity kick.
  kick_duration_range = [0.05, 0.2]

  ppo_params = get_rl_config(_ENV_NAME.value)

  if _NUM_TIMESTEPS.present:
    ppo_params.num_timesteps = _NUM_TIMESTEPS.value
  if _PLAY_ONLY.present:
    ppo_params.num_timesteps = 0
  if _NUM_EVALS.present:
    ppo_params.num_evals = _NUM_EVALS.value
  if _REWARD_SCALING.present:
    ppo_params.reward_scaling = _REWARD_SCALING.value
  if _EPISODE_LENGTH.present:
    ppo_params.episode_length = _EPISODE_LENGTH.value
  if _NORMALIZE_OBSERVATIONS.present:
    ppo_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
  if _ACTION_REPEAT.present:
    ppo_params.action_repeat = _ACTION_REPEAT.value
  if _UNROLL_LENGTH.present:
    ppo_params.unroll_length = _UNROLL_LENGTH.value
  if _NUM_MINIBATCHES.present:
    ppo_params.num_minibatches = _NUM_MINIBATCHES.value
  if _NUM_UPDATES_PER_BATCH.present:
    ppo_params.num_updates_per_batch = _NUM_UPDATES_PER_BATCH.value
  if _DISCOUNTING.present:
    ppo_params.discounting = _DISCOUNTING.value
  if _LEARNING_RATE.present:
    ppo_params.learning_rate = _LEARNING_RATE.value
  if _ENTROPY_COST.present:
    ppo_params.entropy_cost = _ENTROPY_COST.value
  if _NUM_ENVS.present:
    ppo_params.num_envs = _NUM_ENVS.value
  if _NUM_EVAL_ENVS.present:
    ppo_params.num_eval_envs = _NUM_EVAL_ENVS.value
  if _BATCH_SIZE.present:
    ppo_params.batch_size = _BATCH_SIZE.value
  if _MAX_GRAD_NORM.present:
    ppo_params.max_grad_norm = _MAX_GRAD_NORM.value
  if _CLIPPING_EPSILON.present:
    ppo_params.clipping_epsilon = _CLIPPING_EPSILON.value
  if _POLICY_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.policy_hidden_layer_sizes = list(
        map(int, _POLICY_HIDDEN_LAYER_SIZES.value)
    )
  if _VALUE_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.value_hidden_layer_sizes = list(
        map(int, _VALUE_HIDDEN_LAYER_SIZES.value)
    )
  if _POLICY_OBS_KEY.present:
    ppo_params.network_factory.policy_obs_key = _POLICY_OBS_KEY.value
  if _VALUE_OBS_KEY.present:
    ppo_params.network_factory.value_obs_key = _VALUE_OBS_KEY.value

  env = registry.load(_ENV_NAME.value, config=env_cfg)

  print(f"Environment Config:\n{env_cfg}")
  print(f"PPO Training Parameters:\n{ppo_params}")

  exp_name = _LOAD_CHECKPOINT_PATH.value

  # Set up logging directory
  logdir = epath.Path("logs").resolve() / exp_name
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"Logs are being stored in: {logdir}")

  # Handle checkpoint loading
  if _LOAD_CHECKPOINT_PATH.value is not None:
    # Convert to absolute path
    ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
    if ckpt_path.is_dir():
      latest_ckpts = list(ckpt_path.glob("*"))
      latest_ckpts = [ckpt for ckpt in latest_ckpts if ckpt.is_dir()]
      latest_ckpts.sort(key=lambda x: int(x.name))
      latest_ckpt = latest_ckpts[-1]
      restore_checkpoint_path = latest_ckpt
      print(f"Restoring from: {restore_checkpoint_path}")
    else:
      restore_checkpoint_path = ckpt_path
      print(f"Restoring from checkpoint: {restore_checkpoint_path}")
  else:
    print("No checkpoint path provided, not restoring from checkpoint")
    restore_checkpoint_path = None

  # Set up checkpoint directory
  ckpt_path = logdir / "checkpoints"
  ckpt_path.mkdir(parents=True, exist_ok=True)
  print(f"Checkpoint path: {ckpt_path}")

  # Save environment configuration
  with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)

  # Define policy parameters function for saving checkpoints
  def policy_params_fn(current_step, make_policy, params):  # pylint: disable=unused-argument
    orbax_checkpointer = ocp.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(params)
    path = ckpt_path / f"{current_step}"
    orbax_checkpointer.save(path, params, force=True, save_args=save_args)

  training_params = dict(ppo_params)
  if "network_factory" in training_params:
    del training_params["network_factory"]

  network_fn = ppo_networks.make_ppo_networks

  if hasattr(ppo_params, "network_factory"):
    network_factory = functools.partial(
        network_fn, **ppo_params.network_factory
    )
  else:
    network_factory = network_fn

  if _DOMAIN_RANDOMIZATION.value:
    training_params["randomization_fn"] = registry.get_domain_randomizer(
        _ENV_NAME.value
    )

  num_eval_envs = ppo_params.get("num_eval_envs", 128)

  if "num_eval_envs" in training_params:
    del training_params["num_eval_envs"]

  train_fn = functools.partial(
      ppo.train,
      **training_params,
      network_factory=network_factory,
      policy_params_fn=policy_params_fn,
      seed=_SEED.value,
      restore_checkpoint_path=restore_checkpoint_path,
      wrap_env_fn= wrapper.wrap_for_brax_training,
      num_eval_envs=num_eval_envs,
  )

  # Load evaluation environment
  eval_env = registry.load(_ENV_NAME.value, config=env_cfg)

  # Train or load the model
  make_inference_fn, params, _ = train_fn(  # pylint: disable=no-value-for-parameter
      environment=env,
      eval_env=eval_env,
  )

  print("Starting inference...")

  # Create inference function
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)

  # Prepare for evaluation
  num_envs = 1

  jit_reset = jax.jit(eval_env.reset)
  jit_step = jax.jit(eval_env.step)

  x_vel = 0.1 #@param {type: "number"}
  y_vel = 0.0  #@param {type: "number"}
  yaw_vel = 0.0  #@param {type: "number"}
  command = jp.array([x_vel, y_vel, yaw_vel])

  rng = jax.random.PRNGKey(123)
  rng, reset_rng = jax.random.split(rng)
  state = jit_reset(reset_rng)
  state.info["command"] = command

  data = mujoco.MjData(eval_env.mj_model)
  viewer = mujoco.viewer.launch_passive(eval_env.mj_model, data, show_left_ui=False, show_right_ui=False)

  # Viewer settings
  viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True
  viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
  viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False


  frame = 0
  viewer_start = time.time()
  realtime = True
  
  # Run evaluation rollout
  while True:
      # Sync the state with the viewer
    if realtime:
        current_ts = viewer_start + frame * eval_env.dt
        to_sleep = current_ts - time.time()
        if to_sleep > 0:
            time.sleep(to_sleep)
    
    viewer.sync()

    act_rng, rng = jax.random.split(rng)
    ctrl, _ = jit_inference_fn(state.obs, act_rng)
    state = jit_step(state, ctrl)
    state.info["command"] = command
    data.ctrl = ctrl
    mujoco.mj_step(eval_env.mj_model, data)
    frame += 1

    # if state.done:
    #   break

if __name__ == "__main__":
  app.run(main)
