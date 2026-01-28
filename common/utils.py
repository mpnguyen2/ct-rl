# common/utils.py

from __future__ import annotations

import csv
import re
from datetime import datetime
from pathlib import Path
import random
from typing import Dict, Any, Optional, Tuple, Union

import numpy as np
import torch as th
from gymnasium import spaces


def get_device(device: Union[str, th.device] = "auto") -> th.device:
    """
    Parse a torch.device from a string or device spec.

    - "auto": cuda if available, else cpu
    - "cpu" / "cuda" / "cuda:0", etc.
    - existing torch.device is passed through
    """
    if isinstance(device, th.device):
        return device
    if device == "auto":
        return th.device("cuda" if th.cuda.is_available() else "cpu")
    return th.device(device)


def set_seed(seed: int) -> None:
    """
    Set Python, NumPy, and Torch RNG seeds.
    """
    random.seed(seed)
    np.random.seed(seed)
    th.manual_seed(seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(seed)


def get_obs_shape(space: spaces.Space) -> tuple[int, ...]:
    """
    Basic observation shape helper supporting Box and Discrete.
    """
    if isinstance(space, spaces.Box):
        return tuple(space.shape)
    if isinstance(space, spaces.Discrete):
        return (1,)
    if isinstance(space, spaces.MultiBinary):
        return (int(space.n),)
    if isinstance(space, spaces.MultiDiscrete):
        return (int(space.nvec.shape[0]),)
    raise NotImplementedError(f"Unsupported observation space type: {type(space)}")


def get_flattened_obs_dim(space: spaces.Space) -> int:
    """
    Return the flattened dimension of an observation space.
    """
    return int(np.prod(get_obs_shape(space)))


def get_action_dim(space: spaces.Space) -> int:
    """
    Basic action dim helper supporting Box and Discrete.
    """
    if isinstance(space, spaces.Box):
        return int(np.prod(space.shape))
    if isinstance(space, spaces.Discrete):
        return 1
    if isinstance(space, spaces.MultiBinary):
        return int(space.n)
    if isinstance(space, spaces.MultiDiscrete):
        return int(space.nvec.shape[0])
    raise NotImplementedError(f"Unsupported action space type: {type(space)}")


_ACTIVATION_MAP = {
    "ReLU": th.nn.ReLU,
    "Tanh": th.nn.Tanh,
    "LeakyReLU": th.nn.LeakyReLU,
    "ELU": th.nn.ELU,
}


def _normalize_activation_fn(name: str):
    """Parsing helper for activation_fn in benchmarks table."""
    name = (name or "ReLU").strip()
    if name in _ACTIVATION_MAP:
        return _ACTIVATION_MAP[name]
    raise ValueError(f"Unknown activation_fn '{name}' in benchmarks table.")


def _parse_scalar(value: str):
    """
    Convert a CSV string cell into int/float/string/None.
    Handles e.g. '3e-4', '0.99', 'auto'.
    """
    if value is None:
        return None
    s = str(value).strip()
    if s == "":
        return None
    lower = s.lower()
    if lower in {"auto"}:
        return s
    if lower == "none":
        return None
    # try int
    try:
        return int(s)
    except ValueError:
        pass
    # try float (handles 3e-4)
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _parse_net_arch(arch_str: str):
    """
    Convert "256,256" -> [256, 256].
    """
    if not arch_str:
        return None
    parts = [p.strip() for p in arch_str.split(",") if p.strip() != ""]
    return [int(p) for p in parts]


def _parse_dict_from_string(value: str) -> Optional[Dict[str, Any]]:
    """
    Convert a string like "key1=val1;key2=val2" into a dictionary.
    Values are parsed into floats or strings.
    """
    if not value or not isinstance(value, str):
        return None
    result = {}
    parts = [p.strip() for p in value.split(";") if p.strip()]
    for part in parts:
        if "=" in part:
            key, val_str = part.split("=", 1)
            result[key.strip()] = _parse_scalar(val_str.strip())
    return result if result else None


def load_sb3_hyperparams_from_table(
    algo: str,
    env_id: str,
    mode: str,
    hyperparams_dir: Union[str, Path] = Path("benchmarks") / "hyperparams",
) -> Tuple[int, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Load SB3 hyperparameters for (algo, env_id) from CSV tables in benchmarks/hyperparams.

    CSV column convention:
      - algo, env_id
      - total_timesteps
      - env_*    -> environment metadata (returned as `env_meta`), e.g.
                    env_max_episode_steps, env_dt, ...
      - policy_* -> policy architecture (net_arch, activation_fn, ...)
      - algo_*   -> algorithm kwargs for SB3 constructor
                    (learning_rate, gamma, buffer_size, batch_size, tau, ...)
      - log_*    -> log / runner kwargs (e.g. log_interval, save_freq, eval_freq, ...)

    Assumes the hyperparameters files are located at hyperparams_dir/<algo>.csv
    """
    algo = algo.lower()
    filename_map = {
        "sac": "sac.csv",
        "ppo": "ppo.csv",
        "trpo": "trpo.csv",
        "td3": "td3.csv",
    }
    if algo not in filename_map:
        raise ValueError(f"Unsupported algo '{algo}' for SB3 benchmarks.")

    # Direct relative path from project root
    csv_path = Path(hyperparams_dir) / filename_map[algo]
    if not csv_path.exists():
        raise FileNotFoundError(f"Hyperparams table not found: {csv_path}")

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        row = None
        for r in reader:
            if r.get("env_id") == env_id and (r.get("mode") or "").strip() == mode:
                row = r
                break

    if row is None:
        raise KeyError(
            f"No row found in {csv_path} for algo='{algo}', env_id='{env_id}', mode='{mode}'."
        )

    # total timesteps
    total_timesteps = int(float(row.get("total_timesteps", "1000000")))

    # env meta (env_* columns)
    env_meta: Dict[str, Any] = {}
    for key, val in row.items():
        if key.startswith("env_"):
            env_key = key[len("env_") :]
            if env_key == "time_sampling_kwargs":
                env_meta[env_key] = _parse_dict_from_string(val)
            else:
                env_meta[env_key] = _parse_scalar(val)

    # policy kwargs
    policy_kwargs: Dict[str, Any] = {}
    arch_str = row.get("policy_net_arch", "")
    if arch_str:
        policy_kwargs["net_arch"] = _parse_net_arch(arch_str)

    act_name = row.get("policy_activation_fn", "ReLU")
    policy_kwargs["activation_fn"] = _normalize_activation_fn(act_name)

    # algo kwargs + log kwargs
    algo_kwargs: Dict[str, Any] = {}
    log_kwargs: Dict[str, Any] = {}

    skip_keys = {
        "algo",
        "env_id",
        "mode",
        "total_timesteps",
        "comment",
        "policy_net_arch",
        "policy_activation_fn",
    }

    for key, val in row.items():
        if key in skip_keys or key.startswith("env_") or key.startswith("policy_"):
            continue
        if val is None or str(val).strip() == "":
            continue

        if key.startswith("log_"):
            log_key = key[len("log_") :]
            log_kwargs[log_key] = _parse_scalar(val)
        else:
            algo_kwargs[key] = _parse_scalar(val)

    return total_timesteps, env_meta, policy_kwargs, algo_kwargs, log_kwargs


def load_ct_hyperparams_from_table(
    algo: str,
    env_id: str,
    mode: str,
    hyperparams_dir: Union[str, Path] = Path("benchmarks") / "hyperparams",
) -> Tuple[int, Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Load continuous-time hyperparameters for (algo, env_id, mode).

    CSV columns convention:
      - algo, env_id, mode
      - env_*    -> passed (with 'env_' stripped) as kwargs to ContinuousEnv
      - model_*  -> becomes model_kwargs for CT algo (ActorQCriticModel, etc.)
      - algo_*   -> becomes algo_kwargs for CT algo (learning_rate, buffer_size, ...)
      - log_*  -> becomes log_kwargs for runner (log_interval, save_freq, ...)
    Assumes the hyperparameters files are located at hyperparams_dir/<algo>.csv

    Returns:
        total_timesteps: int
        env_kwargs: dict for ContinuousEnv(...)
        model_kwargs: dict for CTSAC(..., model_kwargs=model_kwargs)
        algo_kwargs: dict for CTSAC(..., **algo_kwargs)
        log_kwargs: dict for runner script (e.g. for callbacks)
    """
    algo = algo.lower()
    filename_map = {
        "ct_sac": "ct_sac.csv",
        "ct_td3": "ct_td3.csv",
        "ct_ddpg": "ct_ddpg.csv",
        "q_learning": "q_learning.csv",
        "cpg": "cpg.csv",
        "cppo": "cppo.csv",
    }
    if algo not in filename_map:
        raise ValueError(f"Unsupported CT algo '{algo}'.")

    # Direct relative path from project root
    csv_path = Path(hyperparams_dir) / filename_map[algo]
    if not csv_path.exists():
        raise FileNotFoundError(f"CT hyperparams table not found: {csv_path}")

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        row = None
        for r in reader:
            # Match env_id and use `mode` to filter by `time_sampling` column.
            if r.get("env_id") == env_id and (r.get("mode") or "").strip() == mode:
                row = r
                break

    if row is None:
        raise KeyError(
            f"No row found in {csv_path} for algo='{algo}', env_id='{env_id}', mode='{mode}'."
        )

    total_timesteps = int(float(row.get("total_timesteps", "1000000")))

    env_kwargs: Dict[str, Any] = {}
    model_kwargs: Dict[str, Any] = {}
    algo_kwargs: Dict[str, Any] = {}
    log_kwargs: Dict[str, Any] = {}

    skip_keys = {"algo", "env_id", "mode", "total_timesteps", "comment"}

    for key, val in row.items():
        if key in skip_keys:
            continue
        if val is None or str(val).strip() == "":
            continue

        if key.startswith("env_"):
            env_key = key[len("env_") :]
            if env_key == "time_sampling_kwargs":
                env_kwargs[env_key] = _parse_dict_from_string(val)
            else:
                env_kwargs[env_key] = _parse_scalar(val)

        elif key.startswith("model_"):
            model_key = key[len("model_") :]
            if model_key.endswith("_net_arch"):
                # e.g. model_q_net_arch -> q_net_arch
                model_kwargs[model_key] = _parse_net_arch(val)
            elif model_key == "activation_fn":
                model_kwargs["activation_fn"] = _normalize_activation_fn(val)
            elif model_key in ["deterministic_policy", "use_actor_target"]:
                model_kwargs[model_key] = bool(val.lower() == "true")
            else:
                model_kwargs[model_key] = _parse_scalar(val)

        elif key.startswith("algo_"):
            algo_key = key[len("algo_") :]
            algo_kwargs[algo_key] = _parse_scalar(val)

        elif key.startswith("log_"):
            log_key = key[len("log_") :]
            log_kwargs[log_key] = _parse_scalar(val)

    return total_timesteps, env_kwargs, model_kwargs, algo_kwargs, log_kwargs


def build_save_path(
    root_dir: str,
    algo,
    env_id,
    mode: str,
    seed: int,
    env_kwargs: Dict[str, Any],
    desc: str,
):
    root = Path(root_dir) / algo / env_id / mode / f"seed_{seed}"

    # Build a descriptive run_name with env timing info and timestamps
    run_name_parts = []

    dt = env_kwargs.get("dt")
    if dt is not None:
        run_name_parts.append(f"dt_{str(dt).replace('.', '_')}")

    max_s = env_kwargs.get("max_steps")
    if max_s is not None:
        run_name_parts.append(f"maxs_{max_s}")

    if desc:
        safe_desc = "".join(c if (c.isalnum() or c in "-_") else "_" for c in str(desc))
        run_name_parts.append(safe_desc)

    run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name_parts.append(run_timestamp)
    run_name = "_".join(run_name_parts)

    dir = root / run_name
    dir.mkdir(parents=True, exist_ok=True)

    return dir


def normalize_eval_range(eval_range: str) -> str:
    """
    Ensures eval_range is in 'START-END' format.
    If it's a single token (e.g. 'Q4_2024'), converts to 'Q4_2024-Q4_2024'.
    """
    if "-" not in eval_range:
        return f"{eval_range}-{eval_range}"
    return eval_range


def get_eval_episode_count(eval_range: str) -> int:
    """
    Count number of evaluation episodes (time windows) based on range type.
    Multipliers: Year=12, Quarter=6, Month=2.
    """
    eval_range = normalize_eval_range(eval_range)
    start, end = eval_range.split("-")

    # Year: YYYY
    if re.match(r"^\d{4}$", start) and re.match(r"^\d{4}$", end):
        diff = int(end) - int(start) + 1
        return diff * 12

    # Quarter: Qx_YYYY
    m1 = re.match(r"^Q([1-4])_(\d{4})$", start)
    m2 = re.match(r"^Q([1-4])_(\d{4})$", end)
    if m1 and m2:
        quarters = (
            (int(m2.group(2)) - int(m1.group(2))) * 4
            + (int(m2.group(1)) - int(m1.group(1)))
            + 1
        )
        return quarters * 6

    # Month: MM_YYYY
    m1 = re.match(r"^(\d{1,2})_(\d{4})$", start)
    m2 = re.match(r"^(\d{1,2})_(\d{4})$", end)
    if m1 and m2:
        months = (
            (int(m2.group(2)) - int(m1.group(2))) * 12
            + (int(m2.group(1)) - int(m1.group(1)))
            + 1
        )
        return months * 2

    raise ValueError(f"Unknown eval_range format: {eval_range}")
