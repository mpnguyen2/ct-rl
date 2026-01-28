# evaluations/evaluation_visualize.py
from __future__ import annotations

import copy
import re
from bisect import bisect_right
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Type, Union

import cv2
import imageio
import numpy as np
from stable_baselines3.common.base_class import BaseAlgorithm

from environment.base import ContinuousEnv
from models.base import Model
from evaluations.evaluation_helpers import (
    evaluate_policy_per_step,
    evaluate_sb3_policy_per_step,
)


# -----------------------------
# Display name mapping
# -----------------------------
ALGO_DISPLAY = {
    "ct_sac": "CT-SAC",
    "sac": "SAC",
    "ct_td3": "CT-TD3",
    "td3": "TD3",
    "ppo": "PPO",
    "trpo": "TRPO",
    "q_learning": "q-Learning",
    "cppo": "CPPO",
}


def _display_title(raw: str) -> str:
    if raw in ALGO_DISPLAY:
        return ALGO_DISPLAY[raw]
    return raw.replace("_", "-").upper()


def _infer_frame_times(
    episode_timestamps: List[float],
    n_frames: int,
    render_interval: int,
) -> List[float]:
    """
    evaluate_policy_per_step returns episode_timestamps at every step,
    while episode_frames are collected every render_interval.
    Infer timestamps for frames by sampling episode_timestamps at those indices.
    """
    if not episode_timestamps or n_frames <= 0:
        return []

    idxs = list(range(0, len(episode_timestamps), max(1, render_interval)))
    times = [float(episode_timestamps[i]) for i in idxs]

    # Trim/pad to match frames length
    if len(times) > n_frames:
        times = times[:n_frames]
    elif len(times) < n_frames:
        times.extend([times[-1]] * (n_frames - len(times)))

    return times


def _pick_frame_at_time(
    frames: List[np.ndarray],
    times: Optional[List[float]],
    t: Union[int, float],
) -> Optional[np.ndarray]:
    if not frames:
        return None
    if not times:
        # fallback: treat t as index
        if isinstance(t, int):
            return frames[min(max(t, 0), len(frames) - 1)]
        return frames[-1]

    # rightmost time <= t
    i = bisect_right(times, float(t)) - 1
    i = max(0, min(i, len(frames) - 1))
    return frames[i]


def _add_title_bar(
    frame: np.ndarray,
    title: str,
    timestamp: Optional[float],
    *,
    bar_h: int = 80,
    time_unit="s",
) -> np.ndarray:
    """
    Add a top bar with centered, larger title.
    Also optionally show timestamp on second line (smaller).
    """
    h, w = frame.shape[:2]
    bar = np.zeros((bar_h, w, 3), dtype=np.uint8)

    title = _display_title(title)

    # Title (big)
    font = cv2.FONT_HERSHEY_SIMPLEX
    title_scale = 1.4
    title_th = 3
    (tw, th), _ = cv2.getTextSize(title, font, title_scale, title_th)
    tx = max(0, (w - tw) // 2)
    ty = int(bar_h * 0.55)
    cv2.putText(bar, title, (tx, ty), font, title_scale, (255, 255, 255), title_th)

    # Timestamp (small)
    if timestamp is not None:
        if time_unit != "s":
            ts_str = f"t = {int(timestamp)}" + time_unit
        else:
            ts_str = f"t = {timestamp:.3f}" + time_unit
        ts_scale = 0.8
        ts_th = 2
        (sw, sh), _ = cv2.getTextSize(ts_str, font, ts_scale, ts_th)
        sx = max(0, (w - sw) // 2)
        sy = int(bar_h * 0.90)
        cv2.putText(bar, ts_str, (sx, sy), font, ts_scale, (200, 200, 200), ts_th)

    return np.vstack([bar, frame])


def _choose_single_checkpoint(
    model_dir: Union[str, Path], *, ext: str
) -> Optional[Path]:
    """
    For comparison videos we want a single policy snapshot.
    Priority:
      1) best_model.<ext>
      2) highest *_steps.<ext>
      3) first lexicographic
    """
    p = Path(model_dir)
    best = p / f"best_model.{ext}"
    if best.exists():
        return best

    files = list(p.glob(f"*.{ext}"))
    if not files:
        return None

    step_files = [f for f in files if "_steps" in f.name]
    if step_files:

        def _steps(f: Path) -> int:
            m = re.search(r"_(\d+)_steps\." + re.escape(ext) + r"$", f.name)
            return int(m.group(1)) if m else -1

        return max(step_files, key=_steps)

    return sorted(files)[0]


def generate_progression_frames(
    model_obj: Union[Model, Type[BaseAlgorithm]],
    model_dir: str,
    env: ContinuousEnv,
    *,
    title: str,
    render_interval: int = 1,
) -> Tuple[List[np.ndarray], List[float]]:
    """
    Returns (frames, frame_times) for ONE checkpoint (best_model preferred).
    """
    # trading env is stateful (matplotlib), so clone per model
    model_env = copy.deepcopy(env)

    try:
        if isinstance(model_obj, Model):
            ckpt = _choose_single_checkpoint(model_dir, ext="pth")
            if ckpt is None:
                print(f"No .pth found in {model_dir}")
                return [], []
            model_obj.load_state(str(ckpt))

            out = evaluate_policy_per_step(
                model_obj,
                model_env,
                n_eval_episodes=1,
                deterministic=True,
                render=True,
            )

        elif isinstance(model_obj, type) and issubclass(model_obj, BaseAlgorithm):
            ckpt = _choose_single_checkpoint(model_dir, ext="zip")
            if ckpt is None:
                print(f"No .zip found in {model_dir}")
                return [], []
            sb3_model = model_obj.load(str(ckpt), env=model_env)

            out = evaluate_sb3_policy_per_step(
                sb3_model,
                model_env,
                n_eval_episodes=1,
                deterministic=True,
                render=True,
            )
        else:
            raise TypeError(f"Unsupported model type: {type(model_obj)}")

        episode_frames = out.get("episode_frames", [])
        episode_timestamps = out.get("episode_timestamps", [])

        if not episode_frames:
            return [], []

        frames = episode_frames[0]
        times = _infer_frame_times(episode_timestamps[0], len(frames), render_interval)
        return frames, times

    finally:
        try:
            model_env.close()
        except Exception:
            pass


def create_comparison_video(
    models_to_compare: Dict[Union[Model, Type[BaseAlgorithm]], Tuple[str, str]],
    env: ContinuousEnv,
    output_path: Optional[Union[str, Path]],
    *,
    render_interval: int = 1,
    fps: int = 30,
    return_frames: bool = False,
    output_frame_path: Optional[Union[str, Path]] = None,
):
    """
    Creates a side-by-side comparison video of multiple trained agents.

    models_to_compare maps:
      - custom Model instance OR SB3 algorithm class
        -> (model_dir, display_title)
    """
    all_models = []
    all_models_dir = []
    for model_obj, (model_dir, model_title) in models_to_compare.items():
        frames, times = generate_progression_frames(
            model_obj,
            model_dir,
            env,
            title=model_title,
            render_interval=render_interval,
        )
        all_models_dir.append(model_dir)
        all_models.append((frames, times, model_title))

    if not any(frames for frames, _, _ in all_models):
        print("No frames were generated for any model. Cannot create video.")
        return [] if return_frames else None

    # build global time grid = union of all frame times (sorted unique)
    time_grid = sorted({t for _, times, _ in all_models for t in (times or [])})
    if not time_grid:
        # fallback: align by length
        max_len = max(len(frames) for frames, _, _ in all_models)
        time_grid = list(range(max_len))

    combined_video_frames: List[np.ndarray] = []
    is_trading = "trading" in all_models_dir[0]
    time_unit = " mins" if is_trading else "s"
    for t in time_grid:
        cols = []
        for frames, times, title in all_models:
            frame = _pick_frame_at_time(frames, times, t)
            if frame is None:
                continue
            cols.append(
                _add_title_bar(
                    frame,
                    title,
                    float(t) if isinstance(t, (int, float)) else None,
                    time_unit=time_unit,
                )
            )
        if cols:
            # ensure same height by resizing to min height
            min_h = min(im.shape[0] for im in cols)
            cols = [
                (
                    cv2.resize(im, (im.shape[1], min_h), interpolation=cv2.INTER_AREA)
                    if im.shape[0] != min_h
                    else im
                )
                for im in cols
            ]
            combined_video_frames.append(np.hstack(cols))

    if output_path is not None and combined_video_frames:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimsave(str(out), combined_video_frames, fps=fps)

    # Save last frame image if requested
    if output_frame_path is not None and combined_video_frames:
        fp = Path(output_frame_path)
        fp.parent.mkdir(parents=True, exist_ok=True)
        last = combined_video_frames[-1]
        cv2.imwrite(str(fp), cv2.cvtColor(last, cv2.COLOR_RGB2BGR))

    if return_frames:
        return combined_video_frames
    return None


def create_comparison_video_and_last_frame(
    *,
    models_to_compare,
    env,
    output_video_path: Path,
    output_frame_path: Path,
    render_interval: int,
    fps: int,
):
    """
    Convenience wrapper: write mp4 + last png.
    """
    create_comparison_video(
        models_to_compare=models_to_compare,
        env=env,
        output_path=output_video_path,
        render_interval=render_interval,
        fps=fps,
        return_frames=False,
        output_frame_path=output_frame_path,
    )
