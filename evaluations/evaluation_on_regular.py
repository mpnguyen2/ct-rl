# evaluations/evaluation_on_regular.py
from __future__ import annotations

import os
import warnings
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from evaluations.evaluation_helpers import (
    ALGO_CLASS_MAP,
    create_evaluation_env_and_model,
)
from evaluations.evaluation_stats import (
    load_best_models_for_eval,
    evaluate_algorithms_sum,
)


# -----------------------------
# Quiet logs (stdout only)
# -----------------------------
warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)
for name in ["gym", "gymnasium", "stable_baselines3", "matplotlib", "imageio", "PIL"]:
    logging.getLogger(name).setLevel(logging.ERROR)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"


# -----------------------------
# CONFIG
# -----------------------------
LOGS_ROOT = Path("logs")  # not used here, kept for consistency
SAVED_MODELS_ROOT = Path("saved_models")
OUT_DIR = Path("out/final_reports/metrics_under_regular")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Control tasks
ENV_IDS: List[str] = ["cheetah-run", "walker-run", "quadruped-run", "humanoid-walk"]
MODE = "regular"

# 8 algos
ALL_ALGOS: List[str] = [
    "ct_sac",
    "sac",
    "td3",
    "ppo",
    "trpo",
    "q_learning",
    "cppo",
    "ct_td3",
]

# Evaluate each (env, algo, training_seed) by rolling out N episodes
N_EVAL_EPISODES = 10

# Training seeds is fixed 0..11
SEEDS: List[int] = list(range(12))


def _print_header(msg: str) -> None:
    bar = "=" * max(10, len(msg) + 2)
    print(f"\n{bar}\n{msg}\n{bar}\n")


def _seed_model_dir_exists(*, algo: str, env_id: str, seed: int) -> bool:
    base = SAVED_MODELS_ROOT / algo / env_id / "top" / f"seed_{seed}" / "best_model"
    cls = ALGO_CLASS_MAP[algo]
    if "stable_baselines3" in str(cls) or hasattr(cls, "load"):
        # SB3
        return (base / "best_model.zip").exists() or (base / "best_model.pth").exists()
    # CT custom
    return (base / "best_model.pth").exists()


def main() -> None:
    # per-seed rows: (env, algo, seed) -> score
    rows = []

    for env_id in ENV_IDS:
        _print_header(f"[ENV] {env_id}  mode={MODE}")

        for seed in SEEDS:
            # Skip seed if none of the algos exist for that seed (common when partial sweeps)
            if not any(
                _seed_model_dir_exists(algo=a, env_id=env_id, seed=seed)
                for a in ALL_ALGOS
            ):
                print(f"[seed_{seed}] skip (no models found for any algo)")
                continue

            print(f"[seed_{seed}] load models + rollout")

            # Load models for this training seed (ONE call, uses your helper)
            # Note: quarters=None because this is control tasks only.
            models = load_best_models_for_eval(
                algos=ALL_ALGOS,
                env_id=env_id,
                mode="top",
                seed=seed,
                saved_models_root=SAVED_MODELS_ROOT,
                quarters=None,
            )

            # Create a "common" env instance for this (env, seed), regular timing
            # (env_kwargs come from hyperparams table via create_evaluation_env_and_model)
            env, _ = create_evaluation_env_and_model(
                env_id=env_id,
                model_class=ALGO_CLASS_MAP["ct_sac"],  # just to construct env correctly
                seed=seed,  # env randomness tied to training seed for reproducibility
                algo="ct_sac",
                mode=MODE,
                quarters=None,
            )

            try:
                # Evaluate all algos on the SAME env instance (sequentially).
                # Score per algo = mean return over episodes = sum / N_EVAL_EPISODES
                sums = evaluate_algorithms_sum(
                    models=models,
                    env=env,
                    n_eval_episodes=N_EVAL_EPISODES,
                )

                for algo, s in sums.items():
                    rows.append(
                        dict(
                            env_id=env_id,
                            algo=algo,
                            train_seed=seed,
                            n_eval_episodes=N_EVAL_EPISODES,
                            score=float(s) / float(N_EVAL_EPISODES),
                        )
                    )
                    print(f"  {algo:12s}: {float(s) / float(N_EVAL_EPISODES):.4f}")

            finally:
                try:
                    env.close()
                except Exception:
                    pass

    df = pd.DataFrame(rows)
    if df.empty:
        _print_header("[DONE] No results collected (check saved_models paths / seeds).")
        return

    # Summary: mean±std over training seeds for each (env, algo)
    summary = (
        df.groupby(["env_id", "algo"])["score"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(
            columns={
                "count": "n_seeds",
                "mean": "mean_over_seeds",
                "std": "std_over_seeds",
            }
        )
    )

    # Save CSVs
    per_seed_csv = OUT_DIR / "per_seed_scores.csv"
    summary_csv = OUT_DIR / "summary_mean_std_over_seeds.csv"
    df.to_csv(per_seed_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    print(f"\n[saved] {per_seed_csv}")
    print(f"[saved] {summary_csv}")

    # Print compact table: rows=env_id, cols=algo, values=mean_over_seeds
    pivot = summary.pivot(index="env_id", columns="algo", values="mean_over_seeds")
    pivot = pivot.reindex(index=ENV_IDS, columns=ALL_ALGOS)
    _print_header("[TABLE] mean_over_seeds (rows=env_id, cols=algo)")
    with pd.option_context("display.max_columns", 200, "display.width", 200):
        print(pivot.round(4).to_string())

    # Also print std table
    pivot_std = summary.pivot(index="env_id", columns="algo", values="std_over_seeds")
    pivot_std = pivot_std.reindex(index=ENV_IDS, columns=ALL_ALGOS)
    _print_header("[TABLE] std_over_seeds (rows=env_id, cols=algo)")
    with pd.option_context("display.max_columns", 200, "display.width", 200):
        print(pivot_std.round(4).to_string())


if __name__ == "__main__":
    main()
