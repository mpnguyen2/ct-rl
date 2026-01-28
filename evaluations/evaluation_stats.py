# evaluations/evaluation_stats.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Union

import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from environment.base import ContinuousEnv
from models.base import Model
from evaluations.evaluation_helpers import (
    ALGO_CLASS_MAP,
    create_evaluation_env_and_model,
    evaluate_policy_per_step,
    evaluate_sb3_policy_per_step,
)


def select_best_seed_from_eval_npz(
    *,
    logs_root: str | Path,
    algo: str,
    env_id: str,
    mode: str,
    metric: str = "mean_reward",
) -> int:
    """
    Picks the seed with the best final metric from eval npz files under:
      logs_root/algo/env_id/mode/seed_*/eval/*.npz
    """

    logs_root = Path(logs_root)
    base = logs_root / algo / env_id / mode

    best_seed = 0
    best_val = -np.inf

    for seed_dir in sorted(base.glob("seed_*")):
        try:
            seed = int(seed_dir.name.split("_")[-1])
        except Exception:
            continue

        candidates = [
            seed_dir / "eval" / "evaluation.npz",
            seed_dir / "eval" / "evaluations.npz",
            seed_dir / "evaluation.npz",
            seed_dir / "evaluations.npz",
        ]
        npz_path = next((p for p in candidates if p.exists()), None)
        if npz_path is None:
            continue

        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception:
            continue

        val = None

        # direct metric
        if metric in data:
            arr = data[metric]
            val = float(arr[-1]) if np.ndim(arr) > 0 else float(arr)

        # stable-baselines style: results
        elif "results" in data:
            results = data["results"]
            mean_curve = (
                results.mean(axis=1) if results.ndim == 2 else results.astype(float)
            )
            val = float(mean_curve[-1]) if metric != "max" else float(max(mean_curve))

        # fallbacks
        else:
            for k in [
                "episode_returns",
                "returns",
                "eval_returns",
                "mean_rewards",
                "mean_reward",
            ]:
                if k in data:
                    arr = data[k]
                    val = float(arr[-1]) if np.ndim(arr) > 0 else float(arr)
                    break

        if val is not None and val > best_val:
            best_val = val
            best_seed = seed

    return best_seed


def load_best_models_for_eval(
    *,
    algos,
    env_id,
    mode,
    seed,
    saved_models_root: Path,
    quarters=None,
) -> Dict[str, Union[Model, BaseAlgorithm]]:
    """
    Returns:
      algo_name -> loaded model instance (custom Model or SB3 BaseAlgorithm)
    """
    models: Dict[str, Union[Model, BaseAlgorithm]] = {}

    for algo in algos:
        algo_cls = ALGO_CLASS_MAP[algo]
        model_dir = (
            saved_models_root / algo / env_id / mode / f"seed_{seed}" / "best_model"
        )

        if issubclass(algo_cls, BaseAlgorithm):
            model = algo_cls.load(str(model_dir / "best_model.zip"), env=None)
        else:
            _, model = create_evaluation_env_and_model(
                env_id=env_id,
                model_class=algo_cls,
                seed=seed,
                algo=algo,
                mode=mode,
                quarters=quarters,
            )
            model.load_state(str(model_dir / "best_model.pth"))

        models[algo] = model

    return models


def evaluate_algorithms_sum(
    *,
    models: Dict[str, Union[Model, BaseAlgorithm]],
    env: ContinuousEnv,
    n_eval_episodes: int,
) -> Dict[str, float]:
    """
    Evaluate multiple algorithms on the SAME environment.

    Returns:
      algo -> SUM of episode returns over n_eval_episodes
    """
    results: Dict[str, float] = {}

    for algo_name, model in models.items():
        if isinstance(model, BaseAlgorithm):
            out = evaluate_sb3_policy_per_step(
                model,
                env,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,
            )
        else:
            out = evaluate_policy_per_step(
                model,
                env,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
                render=False,
            )

        returns = np.asarray(out["episode_returns"], dtype=float)
        results[algo_name] = float(returns.sum())

    return results
