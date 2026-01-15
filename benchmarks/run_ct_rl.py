# experiments/run_ct_rl.py

from __future__ import annotations

import os
from datetime import datetime
from functools import partial
import argparse
from pathlib import Path

import numpy as np

from environment.dmc import DMCContinuousEnv
from environment.monitor import Monitor
from environment.vec_env import VecContinuousEnv
from environment.trading_env import TradingContinuousEnv
from environment.base import ContinuousEnv
from algorithms.ct_sac import CTSAC
from algorithms.ct_td3 import CTTD3
from algorithms.ct_ddpg import CTDDPG
from algorithms.q_learning import qLearning
from algorithms.cpg import CPG
from algorithms.cppo import CPPO
from models import ActorQCriticModel, CoupledVqModel, ActorVCriticModel
from models.noise import (
    ActionNoise,
    GaussianActionNoise,
    OrnsteinUhlenbeckActionNoise,
)
from common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
)

from common.logger import configure
from common.utils import (
    load_ct_hyperparams_from_table,
    build_save_path,
    normalize_eval_range,
    get_eval_episode_count,
)
from data.trading.config import TRAIN_NPZ, EVAL_NPZ, GROUPS


def _create_action_noise_from_hyperparams(
    env: DMCContinuousEnv, algo_kwargs: dict
) -> ActionNoise | None:
    noise_type = algo_kwargs.pop("noise_type", None)
    noise_params_str = algo_kwargs.pop("noise_params", None)
    if not noise_type:
        return None

    params = {}
    if noise_params_str:
        for part in noise_params_str.split(";"):
            key, val = part.strip().split("=")
            params[key] = float(val)

    action_dim = int(np.prod(env.action_space.shape))
    mean = params.get("mean", 0.0) * np.ones(action_dim)

    if noise_type == "gaussian" or noise_type == "normal":
        return GaussianActionNoise(
            mean=mean, sigma=params.get("sigma", 0.1) * np.ones(action_dim)
        )
    elif noise_type == "ornstein":
        dt = params.get("dt", env.dt)
        return OrnsteinUhlenbeckActionNoise(
            mean=mean,
            sigma=params.get("sigma", 0.1) * np.ones(action_dim),
            theta=params.get("theta", 0.15),
            dt=dt,
        )
    raise ValueError(f"Unknown noise type: {noise_type}")


def make_ct_env(
    env_id: str,
    seed: int,
    env_kwargs: dict,
    log_dir: Path | str | None = None,
    npz_path: str | None = None,
) -> ContinuousEnv:
    """
    Build a single continuous-time environment instance.
    """
    if env_id.startswith("trading"):
        env = TradingContinuousEnv(
            npz_path=npz_path,
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

    # Continuous-time Monitor wrapper
    if log_dir:
        env = Monitor(env)
    return env


def run_algorithm(
    algo: str,
    env_id: str,
    mode: str,
    eval_mode: str | None,
    seed: int,
    hyperparams_dir: str,
    log_root_dir: str,
    save_root_dir: str,
    total_timesteps_override: int | None,
    desc: str,
    n_eval_episodes: int = 10,
    eval_range: str | None = None,
):
    """
    Runs a single RL algorithm experiment.
    """
    print(
        f"\n{'='*50}\nRunning: {algo} on {env_id} (mode: {mode}, eval_mode: {eval_mode or mode}, seed: {seed})\n{'='*50}"
    )

    (
        total_timesteps,
        env_kwargs,
        model_kwargs,
        algo_kwargs,
        log_kwargs,
    ) = load_ct_hyperparams_from_table(
        algo=algo,
        env_id=env_id,
        mode=mode,
        hyperparams_dir=hyperparams_dir,
    )

    if eval_mode:
        _, eval_env_kwargs, _, _, _ = load_ct_hyperparams_from_table(
            algo=algo,
            env_id=env_id,
            mode=eval_mode,
            hyperparams_dir=hyperparams_dir,
        )
    else:
        eval_env_kwargs = env_kwargs.copy()

    if env_id.startswith("trading") and eval_range is not None:
        eval_env_kwargs = eval_env_kwargs.copy()
        eval_range = normalize_eval_range(eval_range)
        eval_env_kwargs["eval_range"] = eval_range
        eval_env_kwargs["eval_cycle_tickers"] = True

        # Approximate number of episodes (2-weeks trading periods)
        n_time_windows = get_eval_episode_count(eval_range)
        n_ticker_cycles = max(len(v) for v in GROUPS.values())
        n_eval_episodes = n_time_windows * n_ticker_cycles

    if total_timesteps_override is not None:
        total_timesteps = total_timesteps_override

    # Build logs and saved_models save paths
    log_dir = build_save_path(log_root_dir, algo, env_id, mode, seed, env_kwargs, desc)
    save_dir = build_save_path(
        save_root_dir, algo, env_id, mode, seed, env_kwargs, desc
    )

    configure(
        folder=str(log_dir),
        output_formats=["csv", "json", "tensorboard", "log"],
    )

    # Create (vectorized) train environments
    n_envs = int(env_kwargs.pop("n_envs", 1))

    make_train_env_fn = partial(
        make_ct_env,
        env_id=env_id,
        env_kwargs=env_kwargs,
        log_dir=log_dir / "train",
        npz_path=TRAIN_NPZ,
    )
    train_env = (
        VecContinuousEnv(
            [lambda i=i: make_train_env_fn(seed=seed + i) for i in range(n_envs)]
        )
        if n_envs > 1
        else make_train_env_fn(seed=seed)
    )

    # Create (vectorized) evaluation environment
    eval_n_envs = int(eval_env_kwargs.pop("n_envs", 1))
    make_eval_env_fn = partial(
        make_ct_env,
        env_id=env_id,
        env_kwargs=eval_env_kwargs,
        npz_path=EVAL_NPZ,
    )
    eval_env = (
        VecContinuousEnv(
            [
                lambda i=i: make_eval_env_fn(seed=seed + 1000 + i)
                for i in range(eval_n_envs)
            ]
        )
        if eval_n_envs > 1
        else make_eval_env_fn(seed=seed + 1000)
    )

    # Create algorithm
    algo_map = {
        "ct_sac": CTSAC,
        "ct_td3": CTTD3,
        "ct_ddpg": CTDDPG,
        "q_learning": qLearning,
        "cpg": CPG,
        "cppo": CPPO,
    }
    if algo not in algo_map:
        raise ValueError(f"Unknown algorithm: {algo}")

    AlgoClass = algo_map[algo]
    if AlgoClass is None:
        print(f"Algorithm '{algo}' is not implemented. Skipping.")
        return

    # Handle model selection and action noise
    if algo == "q_learning":
        model_class = CoupledVqModel
    elif algo == "cppo" or algo == "cpg":
        #  CPG and CPPO is an on-policy algorithm
        model_class = ActorVCriticModel
    else:
        # CT-SAC, CT-TD3, and CT-DDPG
        model_class = ActorQCriticModel
        # For DDPG/TD3, action_noise is for exploration
        if algo in ["ct_ddpg", "ct_td3"]:
            algo_kwargs["action_noise"] = _create_action_noise_from_hyperparams(
                train_env, algo_kwargs
            )

    # model_kwargs from CSV: q_net_arch, pi_net_arch, n_critics, activation_fn, ...
    # algo_kwargs from CSV: learning_rate, buffer_size, batch_size, gamma, tau, ...
    algorithm = AlgoClass(
        env=train_env,
        model=model_class,
        model_kwargs=model_kwargs,
        seed=seed,
        **algo_kwargs,
    )

    # Setup callbacks
    save_freq = log_kwargs.get("save_freq", 100000)
    eval_freq = log_kwargs.get("eval_freq", 10000)
    log_interval = log_kwargs.get("interval", 1000)

    eval_callback = EvalCallback(
        eval_env=eval_env,
        eval_freq=max(eval_freq // n_envs, 1),
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
        best_model_save_path=str(save_dir / "best_model"),
        log_path=str(log_dir / "eval"),
        verbose=1,
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=save_freq,
        save_path=str(save_dir),
        name_prefix=f"{algo}_{env_id}_{mode}",
        verbose=1,
    )
    callback = CallbackList([checkpoint_callback, eval_callback])

    # Training
    env_kwargs["n_envs"] = n_envs  # Put back n_envs only for printing
    env_kwargs["eval_n_envs"] = eval_n_envs
    print(
        f"\n[{algo.upper()}] env={env_id} mode={mode}\n"
        f"total_timesteps={total_timesteps}\n\n"
        f"env_kwargs={env_kwargs}\n\n"
        f"model_kwargs={model_kwargs}\n\n"
        f"algo_kwargs={algo_kwargs}\n\n"
        f"log_kwargs={log_kwargs}\n\n"
    )
    algorithm.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=log_interval,
    )
    print("Training finished.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--algos",
        type=str,
        default="ct_sac",
        help="Comma-separated list of algorithms to run (e.g., 'ct_sac, cppo'). "
        "Choices: ct_sac, ct_td3, ct_ddpg, q_learning, cpg, cppo.",
    )
    parser.add_argument(
        "--env_id",
        type=str,
        default="cheetah-run",
        help="DMC env as 'domain-task', e.g. 'cheetah-run', 'walker-walk'.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="default",
        help="Mode key matching the CSV row",
    )
    parser.add_argument(
        "--eval_mode",
        type=str,
        default=None,
        help="Evaluation mode key for EvalCallback. Defaults to --mode if not set.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--hyperparams_dir",
        type=str,
        default="benchmarks/hyperparams",
        help="Directory containing per-algo hyperparam CSVs.",
    )
    parser.add_argument(
        "--log_root",
        type=str,
        default="logs",
        help="Log root relative to project dir.",
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default="saved_models",
        help="Model save root relative to project dir.",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=None,
        help="Override total_timesteps from CSV.",
    )
    parser.add_argument(
        "--desc",
        type=str,
        default="",
        help="Optional description to append to run directory name.",
    )
    parser.add_argument(
        "--n_eval_episodes",
        type=int,
        default=10,
        help="Number of episodes to evaluate during EvalCallback.",
    )
    parser.add_argument(
        "--eval_range",
        type=str,
        default="Q4_2025-Q4_2025",
        help="Evaluation quarters for the trading environment",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    algos_to_run = [algo.strip() for algo in args.algos.split(",") if algo.strip()]
    env_id = args.env_id
    if env_id.startswith("trading"):
        eval_range = args.eval_range
    else:
        eval_range = None

    for algo in algos_to_run:
        try:
            run_algorithm(
                algo=algo,
                env_id=env_id,
                mode=args.mode,
                eval_mode=args.eval_mode,
                seed=args.seed,
                hyperparams_dir=args.hyperparams_dir,
                log_root_dir=args.log_root,
                save_root_dir=args.save_root,
                total_timesteps_override=args.total_timesteps,
                desc=args.desc,
                n_eval_episodes=args.n_eval_episodes,
                eval_range=eval_range,
            )
        except (FileNotFoundError, KeyError) as e:
            print(f"\nCould not run {algo} due to a configuration error: {e}\n")


if __name__ == "__main__":
    main()
