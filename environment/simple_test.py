import os
import imageio
import argparse
import torch
import numpy as np

from .trading_env import TradingContinuousEnv
from evaluations.evaluation_helpers import get_latest_run_dir
from common.utils import load_ct_hyperparams_from_table

try:
    from models.actor_q_critic import ActorQCriticModel
except ImportError:
    ActorQCriticModel = None

os.makedirs("data/debug/videos", exist_ok=True)

env_id = "trading"
mode = "irregular_dt"
prefix = "saved_models/trading/"
algo = "ct_sac"
seed = 0
prefix_algo = prefix + algo + "/" + env_id + "/" + mode + "/seed_" + str(seed)
try:
    DEFAULT_MODEL_PATH = get_latest_run_dir(prefix_algo) + "/best_model/best_model.pth"
except Exception:
    DEFAULT_MODEL_PATH = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=DEFAULT_MODEL_PATH,
        help="Path to best_model.pth",
    )
    parser.add_argument(
        "--quarters", type=str, default="Q4_2025", help="Comma separated quarters"
    )
    parser.add_argument("--episodes", type=int, default=30, help="Number of episodes")
    args = parser.parse_args()

    quarters = args.quarters.split(",") if args.quarters else None

    # Load environment kwargs from hyperparams table
    _, env_kwargs, model_kwargs, _, _ = load_ct_hyperparams_from_table(
        algo=algo,
        env_id=env_id,
        mode=mode,
        hyperparams_dir="benchmarks/hyperparams/trading",
    )

    # Overwrite/Add specific evaluation settings
    env_kwargs.update(
        dict(
            npz_path="data/trading/processed_data/eval.npz",
            render_mode="rgb_array",
            eval_quarters=quarters,
            eval_cycle_tickers=True,
        )
    )
    env_kwargs.pop("n_envs", None)

    print(f"--- Pass 1: Running {args.episodes} episodes to find top 5 PnL ---")
    env = TradingContinuousEnv(**env_kwargs)

    model = None
    if args.model_path:
        if ActorQCriticModel is None:
            print("Warning: Could not import ActorQCriticModel. Running random policy.")
        else:
            print(f"Loading model from {args.model_path}")
            model = ActorQCriticModel(
                env.observation_space, env.action_space, **model_kwargs
            )
            model.load_state(args.model_path)

    seed = 17
    n_episodes = args.episodes
    fps = 10
    render_interval = 10

    # Seed ONCE at the start for a reproducible run
    obs, info = env.reset(seed=seed)

    episode_pnls = []  # List of (index, pnl, chosen)

    for ep in range(n_episodes):
        if ep > 0:
            obs, info = env.reset()

        chosen = info.get("chosen", [])
        # Minimal logging for Pass 1
        if ep % 5 == 0:
            print(f"  Processing Episode {ep}/{n_episodes}...")

        done = False
        ep_pnl = 0.0

        while not done:
            if model is not None:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    act, _ = model.act(obs_t, deterministic=True)
                    a = act.cpu().numpy()[0]
            else:
                a = env.action_space.sample()

            obs, t, action, reward, next_obs, next_t, terminated, truncated, info = (
                env.step_dt(a)
            )

            ep_pnl += reward

            done = terminated or truncated

        print(f"  Ep {ep}: PnL = {ep_pnl:.2f}")
        episode_pnls.append((ep, ep_pnl, chosen))

    # Identify top 5
    episode_pnls.sort(key=lambda x: x[1], reverse=True)
    top_5 = episode_pnls[:5]
    top_5_indices = set(x[0] for x in top_5)

    print("\n" + "=" * 30)
    print("Top 5 Episodes by PnL:")
    for ep, pnl, chosen in top_5:
        print(f"  Ep {ep}: PnL {pnl:.2f}")
        for x in chosen:
            print(
                f"    {x['group']:<10} {x['ticker']:<8} {x['start_date_ny']} -> {x['end_date_ny']}"
            )
    print("=" * 30)

    env.close()

    # --- Pass 2: Render Top 5 ---
    print("\n--- Pass 2: Rendering Top 5 ---")

    # Re-create env to reset deterministic pointers
    env = TradingContinuousEnv(**env_kwargs)
    obs, info = env.reset(seed=seed)

    for ep in range(n_episodes):
        if ep > 0:
            obs, info = env.reset()

        if ep not in top_5_indices:
            # Skip this episode (reset() already advanced the schedule)
            continue

        print(f"Rendering Ep {ep}...")
        frames = []
        done = False
        ep_pnl = 0.0
        step_cnt = 0

        while not done:
            if model is not None:
                with torch.no_grad():
                    obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                    act, _ = model.act(obs_t, deterministic=True)
                    a = act.cpu().numpy()[0]
            else:
                a = env.action_space.sample()

            obs, t, action, reward, next_obs, next_t, terminated, truncated, info = (
                env.step_dt(a)
            )
            ep_pnl += reward

            if step_cnt % render_interval == 0:
                img = env.render()
                if img is not None:
                    frames.append(img)

            step_cnt += 1

            done = terminated or truncated

        out_path = f"data/debug/videos/trading_test_ep{ep:02d}_pnl{int(ep_pnl)}.mp4"
        if frames:
            imageio.mimsave(out_path, frames, fps=fps)
            print(f"  Saved {out_path} (frames={len(frames)})")

    env.close()


if __name__ == "__main__":
    main()
