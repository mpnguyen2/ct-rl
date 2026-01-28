# helpers.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats
import pandas as pd


from .config import TASK_DISPLAY_DEFAULT, ALGO_DISPLAY_DEFAULT, RANK_MODES
from .config import EVAL_FREQ_DEFAULT, MIN_STEP_DEFAULT
from .load_helpers import load_eval_curve, find_seed_eval_npz, aggregate_seed_curves


# -----------------------------
# Hyperparam label helpers (optional, used for ablation table)
# -----------------------------


def build_mode_labels_from_hyperparams(hyperparams_csv: Path) -> Dict[str, str]:
    """
    Build a short description string from hyperparameter configuration.

    Output:
      labels[mode] = "lr=...;tau=...;..."
    """
    if not hyperparams_csv.exists():
        return {}

    df = pd.read_csv(hyperparams_csv)

    # drop garbage columns
    drop_cols = [c for c in df.columns if c.lower().startswith("unnamed")]
    if "comment" in df.columns:
        drop_cols.append("comment")
    df = df.drop(columns=drop_cols, errors="ignore")

    if "mode" not in df.columns:
        return {}

    labels: Dict[str, str] = {}
    for _, row in df.iterrows():
        mode = str(row["mode"])
        parts = []
        for c in df.columns:
            if c == "mode":
                continue
            v = row[c]
            if pd.isna(v):
                continue
            sv = str(v).strip()
            if sv == "" or sv.lower() == "none":
                continue
            parts.append(f"{c}={sv}")
        labels[mode] = ";".join(parts) if parts else "original"
    return labels


# -----------------------------
# Top/second/third ablation table
# -----------------------------
def make_ablation_grid_top3_with_diffs(
    logs_root: Path,
    tasks: List[str],
    algos: List[str],
    hyperparams_dir: Path,
    *,
    metric_by_task: Optional[Dict[str, str]] = None,
    best_model_by_task: Optional[Dict[str, bool]] = None,
    task_display: Optional[Dict[str, str]] = None,
    algo_display: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    Returns a grid table:
      rows = algorithms
      cols = tasks
      each cell = 3 lines (top/second/third):
        top: <mode> (<score>): <hyperparams>
    """
    metric_by_task = metric_by_task or {}
    best_model_by_task = best_model_by_task or {}
    task_display = task_display or TASK_DISPLAY_DEFAULT
    algo_display = algo_display or ALGO_DISPLAY_DEFAULT
    hyperparams_dir = Path(hyperparams_dir)

    def _drop_garbage_cols(df: pd.DataFrame) -> pd.DataFrame:
        drop_cols = [c for c in df.columns if str(c).lower().startswith("unnamed")]
        if "comment" in df.columns:
            drop_cols.append("comment")
        return df.drop(columns=drop_cols, errors="ignore")

    def _detect_task_col(df: pd.DataFrame) -> Optional[str]:
        for c in ["task", "env_id", "env", "environment"]:
            if c in df.columns:
                return c
        return None

    def _detect_mode_col(df: pd.DataFrame) -> Optional[str]:
        for c in ["mode", "dt_option", "option", "irregular_dt_option"]:
            if c in df.columns:
                return c
        # heuristic
        for c in df.columns:
            s = df[c].astype(str).str.lower()
            if s.str.contains("top|second|third|option_").any():
                return c
        return None

    def _fmt(v) -> str:
        if pd.isna(v):
            return ""
        s = str(v).strip()
        if s.lower() in {"none", "nan"}:
            return ""
        return s

    # cache hyperparam df per algo
    hp_cache: Dict[str, Optional[pd.DataFrame]] = {}
    for algo in algos:
        p = hyperparams_dir / f"{algo}.csv"
        if p.exists():
            df = pd.read_csv(p)
            df = _drop_garbage_cols(df)
            hp_cache[algo] = df
        else:
            hp_cache[algo] = None

    # build table content
    grid = pd.DataFrame(
        index=[algo_display.get(a, a) for a in algos],
        columns=[task_display.get(t, t) for t in tasks],
        data="",
    )

    for algo in algos:
        df_hp = hp_cache.get(algo, None)

        task_col = _detect_task_col(df_hp) if df_hp is not None else None
        mode_col = _detect_mode_col(df_hp) if df_hp is not None else None

        for env_id in tasks:
            metric = metric_by_task.get(env_id, "final")
            best_model = bool(best_model_by_task.get(env_id, False))

            modes = [m for m in RANK_MODES if (logs_root / algo / env_id / m).exists()]
            if not modes:
                grid.loc[
                    algo_display.get(algo, algo), task_display.get(env_id, env_id)
                ] = ""
                continue

            scores = {}
            for m in modes:
                mode_dir = logs_root / algo / env_id / m
                seed_npz = find_seed_eval_npz(mode_dir)
                if not seed_npz:
                    scores[m] = float("nan")
                    continue

                xs, mean_y, _, _ = aggregate_seed_curves(
                    seed_npz,
                    best_model=best_model,
                    env_id=env_id,
                    random_start_map=None,  # scoring is reporting only
                )
                if len(mean_y) == 0:
                    scores[m] = float("nan")
                else:
                    scores[m] = float(
                        np.nanmax(mean_y) if metric == "max" else mean_y[-1]
                    )

            # Report hyperparam configuration
            diffs_by_mode: Dict[str, str] = {m: "" for m in modes}

            if df_hp is not None and task_col is not None and mode_col is not None:
                df_task = df_hp[df_hp[task_col].astype(str) == str(env_id)].copy()
                df_sel = df_task[df_task[mode_col].astype(str).isin(modes)].copy()

                if not df_sel.empty:
                    df_sel["_rank"] = (
                        df_sel[mode_col]
                        .astype(str)
                        .apply(lambda x: modes.index(x) if x in modes else 999)
                    )
                    df_sel = df_sel.sort_values("_rank").drop(columns=["_rank"])

                    ignore = {task_col, mode_col}
                    param_cols = [c for c in df_sel.columns if c not in ignore]

                    # find varying cols among these 3 rows
                    varying_cols = []
                    for c in param_cols:
                        vals = [_fmt(v) for v in df_sel[c].tolist()]
                        vals_nonempty = [v for v in vals if v != ""]
                        if len(set(vals_nonempty)) > 1:
                            varying_cols.append(c)

                    # build per-mode diff string
                    for _, row in df_sel.iterrows():
                        m = str(row[mode_col])
                        if m not in diffs_by_mode:
                            continue
                        parts = []
                        for c in varying_cols:
                            val = _fmt(row[c])
                            if val != "":
                                parts.append(f"{c}={val}")
                        diffs_by_mode[m] = "; ".join(parts)

            # Cell formatting
            lines = []
            for m in modes:
                sc = scores.get(m, float("nan"))
                hp_diff = diffs_by_mode.get(m, "")
                hp_part = f": {hp_diff}" if hp_diff else ""
                lines.append(f"{m} ({sc:.2f}){hp_part}")

            cell = "\n".join(lines)

            grid.loc[algo_display.get(algo, algo), task_display.get(env_id, env_id)] = (
                cell
            )

    return grid


# -----------------------------
# Statistical testing
# -----------------------------


def _final_score_per_seed(
    logs_root: Path,
    algo: str,
    env_id: str,
    mode: str,
    *,
    metric: str,
    best_model: bool,
) -> Dict[str, float]:
    """
    Returns: {seed_name: score}
    score is either final value or max value for that seed curve.
    """
    mode_dir = logs_root / algo / env_id / mode
    seed_npz = find_seed_eval_npz(mode_dir)
    out = {}

    for seed_name, npz_path in seed_npz.items():
        x, y = load_eval_curve(
            npz_path,
            best_model=best_model,
            eval_freq=EVAL_FREQ_DEFAULT,
            min_step=MIN_STEP_DEFAULT,
            prepend_step0=None,  # not part of scoring
        )
        if len(y) == 0:
            continue
        out[seed_name] = float(np.nanmax(y) if metric == "max" else y[-1])

    return out


def run_ctsac_significance_tests(
    logs_root: Path,
    tasks: List[str],
    others: List[str],
    *,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Outputs a detailed DataFrame with significance flags.

    Columns include:
      task, other_algo,
      ct_mode, other_mode,
      mean_ct, mean_other, diff,
      p_paired, p_welch,
      ct_better,
      sig_paired, sig_welch,
      n_common_seeds
    """

    def paired_p(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        if len(a) < 2:
            return np.nan
        if stats is None:
            return np.nan
        return float(stats.ttest_rel(a, b).pvalue)

    def welch_p(a, b):
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        if len(a) < 2 or len(b) < 2:
            return np.nan
        if stats is None:
            return np.nan
        return float(stats.ttest_ind(a, b, equal_var=False).pvalue)

    rows = []

    for task in tasks:
        metric = "max" if task == "trading" else "final"
        best_model = bool(task == "trading")

        # No scoring: compare TOP vs TOP only
        ct_mode = "top"
        if not (logs_root / "ct_sac" / task / ct_mode).exists():
            continue

        omode = "top"

        ct_scores = _final_score_per_seed(
            logs_root, "ct_sac", task, ct_mode, metric=metric, best_model=best_model
        )

        for other in others:
            if not (logs_root / other / task / omode).exists():
                continue

            oscores = _final_score_per_seed(
                logs_root, other, task, omode, metric=metric, best_model=best_model
            )

            common = sorted(set(ct_scores.keys()) & set(oscores.keys()))
            a = np.array([ct_scores[s] for s in common], dtype=float)
            b = np.array([oscores[s] for s in common], dtype=float)

            mean_ct = float(np.mean(a)) if len(a) else np.nan
            mean_other = float(np.mean(b)) if len(b) else np.nan
            diff = float(mean_ct - mean_other)
            ct_better = diff > 0

            p_p = paired_p(a, b)
            p_w = welch_p(a, b)

            sig_paired = bool(ct_better and (not np.isnan(p_p)) and (p_p < alpha))
            sig_welch = bool(ct_better and (not np.isnan(p_w)) and (p_w < alpha))

            rows.append(
                dict(
                    task=task,
                    other_algo=other,
                    ct_mode=ct_mode,
                    other_mode=omode,
                    mean_ct=mean_ct,
                    mean_other=mean_other,
                    diff=diff,
                    ct_better=ct_better,
                    p_paired=p_p,
                    p_welch=p_w,
                    sig_paired=sig_paired,
                    sig_welch=sig_welch,
                    n_common_seeds=len(common),
                )
            )

    return pd.DataFrame(rows)


def print_only_not_significant(df_stats: pd.DataFrame) -> None:
    """
    Only prints algorithms that are NOT significant (lower) vs CT-SAC under each test.
    If none exist, print 'none'.
    """
    if df_stats is None or df_stats.empty:
        print("[stats] No results.")
        return

    for task in sorted(df_stats["task"].unique()):
        sub = df_stats[df_stats["task"] == task].copy()
        if sub.empty:
            continue

        not_sig_paired = sub[~sub["sig_paired"]]["other_algo"].tolist()
        not_sig_welch = sub[~sub["sig_welch"]]["other_algo"].tolist()

        print(f"\n=== {task} ===")
        if not_sig_paired:
            print(
                "Paired t-test NOT significant vs CT-SAC:",
                ", ".join(sorted(set(not_sig_paired))),
            )
        else:
            print("Paired t-test NOT significant vs CT-SAC: none")

        if not_sig_welch:
            print(
                "Welch t-test NOT significant vs CT-SAC:",
                ", ".join(sorted(set(not_sig_welch))),
            )
        else:
            print("Welch t-test NOT significant vs CT-SAC: none")


# -----------------------------
# Runtime (top folder only)
# -----------------------------


def _find_latest_progress_file(seed_dir: Path) -> Optional[Path]:
    """
    Try to locate progress.csv or progress.json under:
      .../top/seed_k/(progress.*)  OR  .../top/seed_k/**/progress.*
    Prefers progress.csv if both exist.
    """
    direct_csv = seed_dir / "progress.csv"
    direct_json = seed_dir / "progress.json"
    if direct_csv.exists():
        return direct_csv
    if direct_json.exists():
        return direct_json

    # search recursively
    candidates = list(seed_dir.glob("**/progress.csv")) + list(
        seed_dir.glob("**/progress.json")
    )
    if not candidates:
        return None

    # prefer CSV; otherwise newest mtime
    def _rank(p: Path):
        prefer_csv = 0 if p.suffix == ".csv" else 1
        return (prefer_csv, -p.stat().st_mtime)

    candidates.sort(key=_rank)
    return candidates[0]


def _find_col_any(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """
    Match columns
    """
    cols = list(df.columns)
    colset = set(cols)

    for c in candidates:
        if c in colset:
            return c

    # substring fallback
    for c in cols:
        for pat in candidates:
            if pat in str(c):
                return c
    return None


def _read_runtime_seconds_from_progress(progress_path: Path) -> Optional[float]:
    """
    Returns final wall-clock elapsed seconds from progress.(csv|json).
    """
    try:
        if progress_path.suffix == ".csv":
            df = pd.read_csv(progress_path)
        else:
            # progress.json can be dict / list[dict] / jsonlines
            txt = progress_path.read_text(encoding="utf-8").strip()
            if not txt:
                return None

            obj = None
            try:
                obj = json.loads(txt)
            except Exception:
                # jsonlines fallback
                rows = []
                for line in txt.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
                obj = rows

            if isinstance(obj, list):
                df = pd.DataFrame(obj)
            elif isinstance(obj, dict):
                # if dict-of-lists, pd.DataFrame works; if scalar dict, wrap
                if any(isinstance(v, list) for v in obj.values()):
                    df = pd.DataFrame(obj)
                else:
                    df = pd.DataFrame([obj])
            else:
                return None

        if df.empty:
            return None

        df = df.replace([np.inf, -np.inf], np.nan)

        # match the same style as estimate_runtime.py
        t_col = _find_col_any(
            df,
            [
                "time/time_elapsed",
                "time_elapsed",
                "time/elapsed",
                "elapsed",
                "wall_time",
                "walltime",
            ],
        )
        if t_col is None:
            return None

        s = pd.to_numeric(df[t_col], errors="coerce").dropna()
        if s.empty:
            return None

        return float(s.iloc[-1])
    except Exception:
        return None


def make_top_runtime_mean_std_table(
    logs_root: Path,
    tasks: List[str],
    algos: List[str],
    *,
    seeds: Optional[List[int]] = None,
    mode: str = "top",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Runtime table for *top folder only*:
      logs_root/algo/task/top/seed_k/**/progress.(csv|json)

    Returns:
      grid_df: index=algo, columns=task, values like "2.31 ± 0.08 h"
      details_df: long form [algo, task, seed, runtime_sec, runtime_hr]
    """
    if seeds is None:
        seeds = list(range(12))  # default 0..11

    rows = []
    for algo in algos:
        for task in tasks:
            base = logs_root / algo / task / mode
            if not base.exists():
                continue

            for seed in seeds:
                seed_dir = base / f"seed_{seed}"
                if not seed_dir.exists():
                    continue

                prog = _find_latest_progress_file(seed_dir)
                if prog is None:
                    continue

                rt_sec = _read_runtime_seconds_from_progress(prog)
                if rt_sec is None:
                    continue

                rows.append(
                    {
                        "algo": algo,
                        "task": task,
                        "seed": seed,
                        "runtime_sec": rt_sec,
                        "runtime_hr": rt_sec / 3600.0,
                        "progress_path": str(prog),
                    }
                )

    details_df = pd.DataFrame(rows)
    grid = pd.DataFrame(index=algos, columns=tasks, dtype=object)

    if details_df.empty:
        return grid, details_df

    g = details_df.groupby(["algo", "task"])["runtime_hr"]
    stats_df = g.agg(["mean", "std", "count"]).reset_index()

    for _, r in stats_df.iterrows():
        m = float(r["mean"])
        s = float(r["std"]) if not pd.isna(r["std"]) else 0.0
        grid.loc[r["algo"], r["task"]] = f"{m:.2f} ± {s:.2f} h (n={int(r['count'])})"

    return grid, details_df
