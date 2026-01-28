from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from .config import EVAL_FREQ_DEFAULT, MIN_STEP_DEFAULT


# -----------------------------
# Utilities
# -----------------------------
def build_maps(tasks: List[str]) -> Tuple[Dict[str, bool], Dict[str, bool]]:
    best_model_by_task = {}
    best_metric_by_task = {}
    for t in tasks:
        if t == "trading":
            best_metric_by_task[t] = "max"
            best_model_by_task[t] = True
        else:
            best_metric_by_task[t] = "final"
            best_model_by_task[t] = False

    return best_metric_by_task, best_model_by_task


# -----------------------------
# Random-start baseline loader
# -----------------------------


def load_random_start_map(random_npz: Path) -> Dict[Tuple[str, str], float]:
    """
    Expects random_start.npz with:
      - tasks: (T,)
      - seeds: (S,)
      - returns: (T, S)
    Returns:
      dict[(task, seed_name)] = float
    """
    data = np.load(random_npz, allow_pickle=True)
    tasks = [str(x) for x in data["tasks"].tolist()]
    seeds = [str(x) for x in data["seeds"].tolist()]
    ret = data["returns"].astype(float)

    out: Dict[Tuple[str, str], float] = {}
    for i, t in enumerate(tasks):
        for j, s in enumerate(seeds):
            out[(t, s)] = float(ret[i, j])
    return out


# -----------------------------
# Eval NPZ loading
# -----------------------------


def load_eval_curve(
    npz_path: Path,
    best_model: bool = False,
    *,
    eval_freq: int = EVAL_FREQ_DEFAULT,
    min_step: int = MIN_STEP_DEFAULT,
    prepend_step0: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load evaluations.npz:
      - timesteps: (T,)
      - results:   (T, n_eval_episodes) usually

    Then:
      1) keep eval_freq timesteps
      2) drop timesteps < min_step (warm-up steps so no record)
      3) optionally prepend (optionally preprend warm-up steps)

    Returns:
      timesteps, mean_curve
    """
    data = np.load(npz_path, allow_pickle=True)

    if "timesteps" not in data.files or "results" not in data.files:
        raise ValueError(f"Unexpected npz format: {npz_path} has keys {data.files}")

    timesteps = data["timesteps"].astype(int)
    results = data["results"]

    # mean over episodes
    if results.ndim == 1:
        mean_curve = results.astype(float)
    else:
        mean_curve = results.mean(axis=1).astype(float)

    order = np.argsort(timesteps)
    timesteps = timesteps[order]
    mean_curve = mean_curve[order]

    # keep only points at eval_freq
    if eval_freq is not None and eval_freq > 0:
        mask = (timesteps % eval_freq) == 0
        timesteps = timesteps[mask]
        mean_curve = mean_curve[mask]

    # drop warmup steps
    if min_step is not None:
        mask = timesteps >= int(min_step)
        timesteps = timesteps[mask]
        mean_curve = mean_curve[mask]
    if best_model and len(mean_curve) > 0:
        mean_curve = np.maximum.accumulate(mean_curve)

    # prepend warmup steps from random actions
    if prepend_step0 is not None:
        timesteps = np.concatenate([np.array([0], dtype=int), timesteps])
        mean_curve = np.concatenate([np.array([float(prepend_step0)]), mean_curve])

    return timesteps, mean_curve


# -----------------------------
# Curve alignment / aggregation
# -----------------------------


def _interp_to_common_x(
    curves: List[Tuple[np.ndarray, np.ndarray]],
    common_x: np.ndarray,
) -> np.ndarray:
    """
    Interpolate each curve onto common_x (linear).
    """
    ys = []
    for x, y in curves:
        if len(x) == 0:
            ys.append(np.full_like(common_x, np.nan, dtype=float))
            continue
        ys.append(np.interp(common_x, x, y))
    return np.vstack(ys)


def aggregate_seed_curves(
    seed_npz: Dict[str, Path],
    *,
    best_model: bool = False,
    eval_freq: int = EVAL_FREQ_DEFAULT,
    min_step: int = MIN_STEP_DEFAULT,
    env_id: Optional[str] = None,
    random_start_map: Optional[Dict[Tuple[str, str], float]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Returns:
      common_x, mean_y, std_y, used_seeds
    """
    curves: List[Tuple[np.ndarray, np.ndarray]] = []
    used_seeds: List[str] = []

    for seed_name, npz_path in sorted(seed_npz.items()):
        step0 = None
        if random_start_map is not None and env_id is not None:
            step0 = random_start_map.get((env_id, seed_name), None)

        x, y = load_eval_curve(
            npz_path,
            best_model=best_model,
            eval_freq=eval_freq,
            min_step=min_step,
            prepend_step0=step0,
        )
        if len(x) == 0:
            continue
        curves.append((x, y))
        used_seeds.append(seed_name)

    if not curves:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=float),
            [],
        )

    # common timestamps x
    common_x = np.unique(np.concatenate([c[0] for c in curves])).astype(int)

    Y = _interp_to_common_x(curves, common_x)  # (n_seeds, T)
    mean_y = np.nanmean(Y, axis=0)
    std_y = np.nanstd(Y, axis=0)
    if env_id == "trading" and random_start_map is not None:
        # Random start map is for plotting mode and scaling purpose only
        # Due to numerical range from -600 to 0
        print(
            "[info] Adjusting y-scale for visualization purpose only due to loss/profit range"
        )
        std_y /= 3
        std_y[0] *= 40
    return common_x, mean_y, std_y, used_seeds


# -----------------------------
# Locate best-mode (top / second / third)
# -----------------------------


def find_seed_eval_npz(mode_dir: Path) -> Dict[str, Path]:
    """
    mode_dir structure:
      .../<mode>/seed_0/eval/evaluations.npz
      .../<mode>/seed_1/eval/evaluations.npz
    """
    out: Dict[str, Path] = {}
    for seed_dir in sorted(mode_dir.glob("seed_*")):
        if not seed_dir.is_dir():
            continue
        seed_name = seed_dir.name.replace("seed_", "")
        npz = seed_dir / "eval" / "evaluations.npz"
        if npz.exists():
            out[seed_name] = npz
    return out
