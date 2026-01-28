# make_random_start.py
from __future__ import annotations

from pathlib import Path
import numpy as np

from .evaluation_helpers import (
    create_evaluation_env_and_model,
    evaluate_policy_per_step,
)


TASKS = ["cheetah-run", "walker-run", "humanoid-walk", "quadruped-run", "trading"]
SEEDS = list(range(12))


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--hyperparams_dir", type=str, default="benchmarks/all_hyperparams")
    ap.add_argument("--algo_for_env_kwargs", type=str, default="ct_sac")
    ap.add_argument("--mode_for_env_kwargs", type=str, default="top")
    ap.add_argument("--out", type=str, default="out/random_start.npz")
    args = ap.parse_args()

    hp_dir = Path(args.hyperparams_dir)
    out_path = Path(args.out)

    returns = np.zeros((len(TASKS), len(SEEDS)), dtype=float)

    for ti, task in enumerate(TASKS):
        n_eps = 1 if task == "trading" else 10

        for sj, seed in enumerate(SEEDS):
            env, _model = create_evaluation_env_and_model(
                env_id=task,
                model_class=None,  # random policy
                seed=seed,
                algo=args.algo_for_env_kwargs,
                mode=args.mode_for_env_kwargs,
                hyperparams_dir=hp_dir,
            )

            # model=None => random actions (see evaluation_helpers.py)
            out = evaluate_policy_per_step(
                model=None,
                env=env,
                n_eval_episodes=n_eps,
                deterministic=True,
            )

            ep_returns = out["episode_returns"]
            avg_ret = float(np.mean(ep_returns))
            returns[ti, sj] = avg_ret
            print(f"[{task}] seed={seed:02d}  avg_return={avg_ret:.3f}")

            env.close()

    np.savez(
        out_path,
        tasks=np.array(TASKS, dtype=object),
        seeds=np.array([str(s) for s in SEEDS], dtype=object),
        returns=returns,
    )
    print(f"\nSaved: {out_path}  shape={returns.shape}")


if __name__ == "__main__":
    main()
