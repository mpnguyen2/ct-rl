# evaluations/evaluation_helpers.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Type, Union
import os
from pathlib import Path

import numpy as np
import torch as th
from stable_baselines3 import SAC, TD3, PPO

try:
    from sb3_contrib import TRPO
except ImportError:
    TRPO = None

from environment.base import ContinuousEnv
from environment.dmc import DMCContinuousEnv
from environment.vec_env import VecContinuousEnv
from environment.trading_env import TradingContinuousEnv
from data.trading.config import EVAL_NPZ
from models.base import Model
from models.actor_q_critic import ActorQCriticModel
from models.actor_v_critic import ActorVCriticModel
from models.coupled_vq import CoupledVqModel
from common.utils import load_ct_hyperparams_from_table, get_device


# Global map from algorithm string to the corresponding model or algorithm class
ALGO_CLASS_MAP = {
    "ct_sac": ActorQCriticModel,
    "ct_td3": ActorQCriticModel,
    "ct_ddpg": ActorQCriticModel,
    "cpg": ActorVCriticModel,
    "cppo": ActorVCriticModel,
    "coupled_sarsa": CoupledVqModel,
    "sac": SAC,
    "td3": TD3,
    "ppo": PPO,
    "trpo": TRPO,
}


def _is_vec_env(env: Any) -> bool:
    return hasattr(env, "num_envs") and isinstance(getattr(env, "num_envs"), int)


# Step-level evaluation
def evaluate_policy_per_step(
    model: Optional[Model],
    env: ContinuousEnv,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    render_interval: int = 10,
) -> Dict[str, Any]:
    """
    Step-level eval for continuous-time model on a Single ContinuousEnv.

    Returns a dict:
      {
        "episode_step_rewards": List[List[float]],
        "episode_timestamps":   List[List[float]],
        "episode_lengths":      List[int],
        "episode_returns":      List[float],
        "episode_frames":       Optional[List[List[np.ndarray]]],
      }
    """
    episode_step_rewards: List[List[float]] = []
    episode_timestamps: List[List[float]] = []
    episode_lengths: List[int] = []
    episode_returns: List[float] = []
    episode_frames: Optional[List[List[np.ndarray]]] = [] if render else None

    device = model.device if model is not None else "auto"
    device = get_device(device)
    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        obs = np.asarray(obs, dtype=np.float32)

        done = False
        step_rewards: List[float] = []
        ts: List[float] = []
        frames: List[np.ndarray] = []
        step_idx = 0

        while not done:
            if model is None:
                action = env.action_space.sample()
            else:
                obs_tensor = th.as_tensor(
                    obs, dtype=th.float32, device=device
                ).unsqueeze(0)
                with th.no_grad():
                    act_tensor, _ = model.act(obs_tensor, deterministic=deterministic)
                    action = act_tensor.detach().cpu().numpy()[0]

            obs_t, t, _, reward, next_obs, next_t, terminated, truncated, info = (
                env.step_dt(action)
            )
            done = bool(terminated or truncated)

            step_rewards.append(float(reward))
            ts.append(float(t))
            obs = np.asarray(next_obs, dtype=np.float32)

            if render and (step_idx % render_interval == 0):
                frame = env.render(mode="rgb_array")
                if frame is not None:
                    frames.append(frame)

            step_idx += 1

        episode_step_rewards.append(step_rewards)
        episode_timestamps.append(ts)
        episode_lengths.append(len(step_rewards))
        episode_returns.append(float(np.sum(step_rewards)))
        if render and episode_frames is not None:
            episode_frames.append(frames)

    return {
        "episode_step_rewards": episode_step_rewards,
        "episode_timestamps": episode_timestamps,
        "episode_lengths": episode_lengths,
        "episode_returns": episode_returns,
        "episode_frames": episode_frames,
    }


# Episodic evaluation on either single env or vectorized(multiple)-env
def evaluate_policy_per_episode(
    model: Model,
    env: Union[ContinuousEnv, VecContinuousEnv],
    n_eval_episodes: int = 10,
    deterministic: bool = True,
) -> Tuple[List[float], List[int]]:
    """
    Episodic eval for continuous-time model.
    The implementation is aware of whether the env is a Monitor wrapper:
      - if info has "episode", use info["episode"]["r"] and ["l"]
      - otherwise use accumulated sums

    Returns:
      episode_returns: List[float] length n_eval_episodes
      episode_lengths: List[int]   length n_eval_episodes
    """
    is_vec_env = _is_vec_env(env)
    n_envs = int(getattr(env, "num_envs", 1)) if is_vec_env else 1

    obs, _ = env.reset()
    if not is_vec_env:
        obs = np.asarray(obs, dtype=np.float32)

    running_returns = np.zeros((n_envs,), dtype=np.float64)
    running_lengths = np.zeros((n_envs,), dtype=np.int64)

    episode_returns: List[float] = []
    episode_lengths: List[int] = []

    while len(episode_returns) < n_eval_episodes:
        obs_batch = obs if is_vec_env else obs[None, ...]
        obs_tensor = th.as_tensor(obs_batch, dtype=th.float32, device=model.device)

        with th.no_grad():
            act_tensor, _ = model.act(obs_tensor, deterministic=deterministic)
        actions = act_tensor.detach().cpu().numpy()
        action_for_env = actions if is_vec_env else actions[0]

        obs_t, t, _, reward, next_obs, next_t, terminated, truncated, infos = (
            env.step_dt(action_for_env)
        )
        done = np.logical_or(terminated, truncated)

        # normalize to arrays
        rew_arr = (
            np.asarray(reward, dtype=np.float64).reshape(-1)
            if is_vec_env
            else np.asarray([reward], dtype=np.float64)
        )
        done_arr = (
            np.asarray(done, dtype=bool).reshape(-1)
            if is_vec_env
            else np.asarray([bool(done)], dtype=bool)
        )

        running_returns += rew_arr
        running_lengths += 1

        if is_vec_env:
            info_list = infos if isinstance(infos, (list, tuple)) else [infos] * n_envs

            # Record stats at the end of episode of a subset of envs inside the vec_env
            done_indices = np.where(done_arr)[0]
            for i in done_indices:
                info_i = info_list[i]
                if isinstance(info_i, dict) and "episode" in info_i:
                    ep_r = float(info_i["episode"]["r"])
                    ep_l = int(info_i["episode"]["l"])
                else:
                    ep_r = float(running_returns[i])
                    ep_l = int(running_lengths[i])

                episode_returns.append(ep_r)
                episode_lengths.append(ep_l)

                running_returns[i] = 0.0
                running_lengths[i] = 0

                if len(episode_returns) >= n_eval_episodes:
                    break

            # VecContinuousEnv already auto-resets done envs internally;
            # next_obs is already the reset obs for those env slots.
            obs = next_obs

        else:
            if bool(done_arr[0]):
                # Record stats at the end of episode
                info = infos
                if isinstance(info, dict) and "episode" in info:
                    ep_r = float(info["episode"]["r"])
                    ep_l = int(info["episode"]["l"])
                else:
                    ep_r = float(running_returns[0])
                    ep_l = int(running_lengths[0])

                episode_returns.append(ep_r)
                episode_lengths.append(ep_l)

                running_returns[0] = 0.0
                running_lengths[0] = 0

                # single env must reset explicitly
                obs, _ = env.reset()
                obs = np.asarray(obs, dtype=np.float32)
            else:
                obs = np.asarray(next_obs, dtype=np.float32)

    return episode_returns, episode_lengths


def evaluate_sb3_policy_per_step(
    sb3_model: Any,
    env: ContinuousEnv,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    render_interval: int = 10,
) -> Dict[str, Any]:
    """
    Step-level eval for SB3 model, for visualization/benchmarking.
    """
    episode_step_rewards: List[List[float]] = []
    episode_timestamps: List[List[float]] = []
    episode_lengths: List[int] = []
    episode_returns: List[float] = []
    episode_frames: Optional[List[List[np.ndarray]]] = [] if render else None

    for _ in range(n_eval_episodes):
        obs, _ = env.reset()
        done = False

        step_rewards: List[float] = []
        ts: List[float] = []
        frames: List[np.ndarray] = []
        step_idx = 0

        while not done:
            action, _ = sb3_model.predict(obs, deterministic=deterministic)
            obs_t, t, _, reward, next_obs, next_t, terminated, truncated, info = (
                env.step_dt(action)
            )
            done = bool(terminated or truncated)

            step_rewards.append(float(reward))
            ts.append(float(t))
            obs = next_obs

            if render and (step_idx % render_interval == 0):
                frame = env.render(mode="rgb_array")
                if frame is not None:
                    frames.append(frame)

            step_idx += 1

        episode_step_rewards.append(step_rewards)
        episode_timestamps.append(ts)
        episode_lengths.append(len(step_rewards))
        episode_returns.append(float(np.sum(step_rewards)))
        if render and episode_frames is not None:
            episode_frames.append(frames)

    return {
        "episode_step_rewards": episode_step_rewards,
        "episode_timestamps": episode_timestamps,
        "episode_lengths": episode_lengths,
        "episode_returns": episode_returns,
        "episode_frames": episode_frames,
    }


def create_evaluation_env_and_model(
    env_id: str,
    model_class: Optional[Type[Model]],
    seed: int,
    algo: str,
    mode: Optional[str] = None,
    hyperparams_dir: Optional[str] = None,
    env_kwargs: Optional[Dict[str, Any]] = None,
    model_kwargs: Optional[Dict[str, Any]] = None,
    quarters: Optional[List[str]] = None,
) -> Tuple[ContinuousEnv, Optional[Model]]:
    # Load/get hyperparams
    if mode:
        if hyperparams_dir is None:
            hyperparams_dir = "benchmarks/hyperparams"
            if env_id.startswith("trading"):
                hyperparams_dir += "/trading"

        _, loaded_env_kwargs, loaded_model_kwargs, _, _ = (
            load_ct_hyperparams_from_table(
                algo=algo, env_id=env_id, mode=mode, hyperparams_dir=hyperparams_dir
            )
        )
        env_kwargs = (
            loaded_env_kwargs
            if env_kwargs is None
            else {**loaded_env_kwargs, **env_kwargs}
        )
        model_kwargs = (
            loaded_model_kwargs
            if model_kwargs is None
            else {**loaded_model_kwargs, **model_kwargs}
        )
    elif env_kwargs is None or model_kwargs is None:
        raise ValueError(
            "Either 'mode' or both 'env_kwargs' and 'model_kwargs' must be provided."
        )

    # Create the env
    if "n_envs" in env_kwargs:
        env_kwargs.pop("n_envs")  # Don't support visualize vectorized env yet.

    if env_id.startswith("trading") and quarters is not None:
        env_kwargs["eval_quarters"] = quarters
        env_kwargs["eval_cycle_tickers"] = True

    if env_id.startswith("trading"):
        env = TradingContinuousEnv(
            npz_path=EVAL_NPZ,
            seed=seed,
            **env_kwargs,
        )
    else:
        if "-" not in env_id:
            raise ValueError("env-id must be 'domain-task', e.g. 'cheetah-run'.")
        domain_name, task_name = env_id.split("-", 1)
        env = DMCContinuousEnv(
            domain_name=domain_name,
            task_name=task_name,
            seed=seed,
            **env_kwargs,
        )

    # Create the model.
    model_instance = None
    if model_class is not None:
        model_instance = model_class(
            observation_space=env.observation_space,
            action_space=env.action_space,
            **model_kwargs,
        )
    return env, model_instance


def get_latest_run_dir(base_dir: Union[str, Path]) -> str:
    path = Path(base_dir)
    if not path.exists():
        return str(base_dir)
    subdirs = [d for d in path.iterdir() if d.is_dir()]
    if not subdirs:
        return str(base_dir)

    latest_subdir = max(subdirs, key=lambda p: p.stat().st_mtime)
    return str(latest_subdir)
