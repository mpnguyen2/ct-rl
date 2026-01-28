# helpers.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple
from itertools import zip_longest

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator

import seaborn as sns

from .config import TASK_DISPLAY_DEFAULT, ALGO_DISPLAY_DEFAULT
from .load_helpers import find_seed_eval_npz, aggregate_seed_curves


# Fixed per-algo colors
ALGO_COLORS: Dict[str, str] = {
    "ct_sac": "#1f77b4",  # blue
    "sac": "#d62728",  # red
    "td3": "#2ca02c",  # green
    "ppo": "#9467bd",  # purple
    "trpo": "#7f7f7f",  # gray
    "cppo": "#ff7f0e",  # orange
    "q_learning": "#e377c2",  # pink
    "ct_td3": "#17becf",  # cyan
    "sac_increment_modeling": "#ff7f0e",  # orange
}

# Trading plot config
TRADING_Y_BREAKS = [-800, 0.0, 10, 20, 30, 40, 45]
TRADING_Y_WEIGHTS = [2.0, 1.0, 1.0, 1.0, 1.0, 0.5]
TRADING_Y_TICKS = [-600, -400, -200, 0, 10, 20, 30, 40, 45]
TRADING_STD_REF_SEG = 1


def set_paper_style() -> None:
    """
    Dark inside axes, white outside
    """
    sns.set_theme(context="paper", style="darkgrid")
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.facecolor": "#EAEAF2",  # seaborn-ish gray
            "grid.color": "white",
            "grid.linewidth": 1.0,
            "axes.edgecolor": "#CCCCCC",
            "axes.linewidth": 1.0,
            "font.size": 11,
        }
    )


# -----------------------------
# Trading plot helpers to easier visualize returns
# -----------------------------


def apply_piecewise_yscale(
    ax: plt.Axes,
    y_breaks: List[float],
    weights: List[float],
    *,
    ticks: Optional[List[float]] = None,
) -> None:
    """
    General piecewise-linear function y-scale.

    Example (4 intervals):
      y_breaks = [y_min, y0, y1, y2, y_max]
      weights =  [a,    b,  c,  d]
    """
    yb = np.asarray(y_breaks, dtype=float)
    w = np.asarray(weights, dtype=float)

    if len(yb) < 2 or len(w) != (len(yb) - 1):
        return

    if np.any(np.diff(yb) <= 0):
        return

    s = float(np.sum(w))
    if s <= 0:
        return
    frac = w / s  # segment display fractions
    t_edges = np.concatenate([[0.0], np.cumsum(frac)])

    def fwd(y):
        y = np.asarray(y, dtype=float)
        t = np.empty_like(y, dtype=float)

        # segment index: i s.t. y in [yb[i], yb[i+1]]
        idx = np.searchsorted(yb[1:-1], y, side="right")
        idx = np.clip(idx, 0, len(frac) - 1)

        for i in range(len(frac)):
            mask = idx == i
            if not np.any(mask):
                continue
            y0, y1 = yb[i], yb[i + 1]
            if y1 == y0 or frac[i] == 0:
                t[mask] = t_edges[i]
            else:
                t[mask] = t_edges[i] + (y[mask] - y0) * (frac[i] / (y1 - y0))
        return t

    def inv(t):
        t = np.asarray(t, dtype=float)
        y = np.empty_like(t, dtype=float)

        idx = np.searchsorted(t_edges[1:-1], t, side="right")
        idx = np.clip(idx, 0, len(frac) - 1)

        for i in range(len(frac)):
            mask = idx == i
            if not np.any(mask):
                continue
            y0, y1 = yb[i], yb[i + 1]
            if frac[i] == 0:
                y[mask] = y0
            else:
                y[mask] = y0 + (t[mask] - t_edges[i]) * ((y1 - y0) / frac[i])
        return y

    try:
        ax.set_yscale("function", functions=(fwd, inv))
    except Exception:
        return

    ax.set_ylim(float(yb[0]), float(yb[-1]))

    if ticks is not None:
        ax.set_yticks(ticks)
        ax.set_yticklabels([str(t) for t in ticks])


def apply_trading_piecewise_yscale(ax: plt.Axes) -> None:
    """
    Y-scale trading view for easier visualization and profits of algorithms.
    PnL starting from -$60000 (loss) to $4000 (profit).
    Mismatch scale requires adjusting y-axis for visualization.
    """
    y_breaks = TRADING_Y_BREAKS
    weights = TRADING_Y_WEIGHTS
    ticks = TRADING_Y_TICKS
    apply_piecewise_yscale(ax, y_breaks, weights, ticks=ticks)


def apply_y_dollar_formatter(ax, scale: float = 100.0, decimals: int = 0) -> None:
    """
    Display dollar amounts
    """

    def _fmt(v: float) -> str:
        vv = v * scale
        sign = "-" if vv < 0 else ""
        vv = abs(vv)
        if decimals <= 0:
            return f"{sign}${vv:,.0f}"
        return f"{sign}${vv:,.{decimals}f}"

    ticks = ax.get_yticks()
    if ticks is not None and len(ticks) > 0:
        ax.set_yticklabels([_fmt(t) for t in ticks])
    else:
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, pos: _fmt(y)))


# -----------------------------
# Legend helpers
# -----------------------------


def _dedup_legend(handles, labels):
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    return list(seen.values()), list(seen.keys())


def reorder_legend_single_row(
    handles: List,
    labels: List[str],
    row: List[str],
) -> Tuple[List, List[str]]:
    """
    Force legend order in a single row, left-to-right:
      first = labels in `row` (in that order, if present)
      then  = any remaining labels in their original order
    """
    lab2handle = {l: h for h, l in zip(handles, labels)}

    first = [l for l in row if l in lab2handle]
    used = set(first)
    rest = [l for l in labels if l not in used]

    ordered_labels = first + rest
    ordered_handles = [lab2handle[l] for l in ordered_labels]
    return ordered_handles, ordered_labels


def reorder_legend_two_rows(
    handles: List,
    labels: List[str],
    top_row: List[str],
    bottom_row: List[str],
    *,
    keep_rest: bool = True,
) -> Tuple[List, List[str]]:
    """
    Force legend to display as:
      row1 = top_row (left->right)
      row2 = bottom_row (left->right)

    Matplotlib legend is column-major, so we INTERLEAVE:
      [top1, bot1, top2, bot2, ...]

    top_row/bottom_row should be the display labels (after algo_display mapping).
    """
    lab2handle = {l: h for h, l in zip(handles, labels)}

    top = [l for l in top_row if l in lab2handle]
    bot = [l for l in bottom_row if l in lab2handle]

    used = set(top + bot)

    if keep_rest:
        rest = [l for l in labels if l not in used]
        bot = bot + rest  # append extras to bottom row

    ordered_labels = []
    for t, b in zip_longest(top, bot, fillvalue=None):
        if t is not None:
            ordered_labels.append(t)
        if b is not None:
            ordered_labels.append(b)

    ordered_handles = [lab2handle[l] for l in ordered_labels]
    return ordered_handles, ordered_labels


def apply_legend_ordering(
    handles: List,
    labels: List[str],
    algo_display: Dict[str, str],
    *,
    legend_row1: Optional[List[str]] = None,
    legend_row2: Optional[List[str]] = None,
    default_ncol: int = 4,
    split_threshold: int = 5,
) -> Tuple[List, List[str], int]:
    """
    Returns (handles, labels, ncol) after applying optional legend ordering rules:

    1) If len(labels) > split_threshold AND legend_row1+legend_row2 provided:
       force 2-row layout (row1 then row2), ncol=len(legend_row1)

    2) If legend_row1 provided AND legend_row2 is None:
       force single-row ordering by legend_row1, ncol=len(labels)

    3) Else:
       default ncol = default_ncol if len(labels) > split_threshold else len(labels)
    """
    # 2-row forced ordering (only when legend would "split")
    if (
        len(labels) > split_threshold
        and legend_row1 is not None
        and legend_row2 is not None
    ):
        row1 = [algo_display.get(a, a) for a in legend_row1]
        row2 = [algo_display.get(a, a) for a in legend_row2]
        handles, labels = reorder_legend_two_rows(handles, labels, row1, row2)
        ncol = max(1, len(legend_row1))
        return handles, labels, ncol

    # 1-row forced ordering
    if legend_row1 is not None and legend_row2 is None:
        row = [algo_display.get(a, a) for a in legend_row1]
        handles, labels = reorder_legend_single_row(handles, labels, row)
        ncol = len(labels)
        return handles, labels, ncol

    # default
    ncol = default_ncol if len(labels) > split_threshold else len(labels)
    return handles, labels, ncol


# -----------------------------
# Plotting helpers
# -----------------------------
def plot_best_on_axis(
    ax: plt.Axes,
    logs_root: Path,
    env_id: str,
    algos: List[str],
    *,
    best_model: bool = False,
    show_band: bool = True,
    random_start_map: Optional[Dict[Tuple[str, str], float]] = None,
    task_display: Optional[Dict[str, str]] = None,
    algo_display: Optional[Dict[str, str]] = None,
    title_fontsize: int = 14,
) -> None:
    """
    Plot each algo's top mode on a single axis.
    """
    task_display = task_display or TASK_DISPLAY_DEFAULT
    algo_display = algo_display or ALGO_DISPLAY_DEFAULT

    if env_id == "trading":
        best_model = True
    for algo in algos:
        mode_dir = logs_root / algo / env_id / "top"
        if not mode_dir.exists():
            continue

        seed_npz = find_seed_eval_npz(mode_dir)
        if not seed_npz:
            continue

        xs, mean_y, std_y, _ = aggregate_seed_curves(
            seed_npz,
            best_model=best_model,
            env_id=env_id,
            random_start_map=random_start_map,
        )

        if len(xs) == 0:
            continue

        label = algo_display.get(algo, algo)
        color = ALGO_COLORS.get(algo, None)

        ax.plot(xs, mean_y, linewidth=2.2, label=label, color=color)
        if show_band:
            ax.fill_between(xs, mean_y - std_y, mean_y + std_y, alpha=0.18, color=color)

    # Title and x, y axis
    ax.set_title(
        task_display.get(env_id, env_id), fontsize=title_fontsize, fontweight="bold"
    )
    ax.set_xlabel("Step (millions)")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x/1e6:.1f}"))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.xaxis.get_offset_text().set_visible(False)
    if env_id == "trading":
        ax.set_ylabel("Evaluation Return ($)")
    else:
        ax.set_ylabel("Evaluation Return")


def plot_task_grid(
    logs_root: Path,
    tasks: List[str],
    algos: List[str],
    *,
    out_path: Path,
    best_model_by_task: Optional[Dict[str, bool]] = None,
    show_band: bool = True,
    random_start_map: Optional[Dict[Tuple[str, str], float]] = None,
    task_display: Optional[Dict[str, str]] = None,
    algo_display: Optional[Dict[str, str]] = None,
    legend_fontsize: int = 14,
    title_fontsize: int = 14,
    xlabel_fontsize: float = 12,
    xtick_fontsize: float = 11,
    ylabel_fontsize: float = 12,
    ytick_fontsize: float = 11,
    legend_row1: Optional[List[str]] = None,
    legend_row2: Optional[List[str]] = None,
) -> None:
    """
    Plot 1 x N grid with legend on top (outside axes) for N tasks.
    """
    set_paper_style()
    task_display = task_display or TASK_DISPLAY_DEFAULT
    algo_display = algo_display or ALGO_DISPLAY_DEFAULT

    best_model_by_task = best_model_by_task or {}

    n = len(tasks)
    fig, axes = plt.subplots(1, n, figsize=(3.6 * n, 3.2), sharey=False)

    if n == 1:
        axes = [axes]  # type: ignore

    for ax, env_id in zip(axes, tasks):
        best_model = bool(best_model_by_task.get(env_id, False))

        plot_best_on_axis(
            ax,
            logs_root,
            env_id,
            algos,
            best_model=best_model,
            show_band=show_band,
            random_start_map=random_start_map,
            task_display=task_display,
            algo_display=algo_display,
            title_fontsize=title_fontsize,
        )

        ax.xaxis.label.set_size(xlabel_fontsize)
        ax.tick_params(axis="x", labelsize=xtick_fontsize)
        ax.yaxis.label.set_size(ylabel_fontsize)
        ax.tick_params(axis="y", labelsize=ytick_fontsize)

        if env_id == "trading":
            apply_trading_piecewise_yscale(ax)
            apply_y_dollar_formatter(ax, scale=100.0, decimals=0)

        # Grid-only
        # ax.set_ylabel("")

        # remove per-axis legends
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    # Grid-only
    # fig.supylabel("Evaluation Return", fontsize=11.5, x=0.016)

    # collect handles/labels from last axis
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    handles, labels = _dedup_legend(handles, labels)
    handles, labels, ncol = apply_legend_ordering(
        handles,
        labels,
        algo_display,
        legend_row1=legend_row1,
        legend_row2=legend_row2,
        default_ncol=4,
        split_threshold=5,
    )

    # Legend
    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.15),
        ncol=ncol,
        frameon=False,
        fontsize=legend_fontsize,
        handlelength=2.8,
        handletextpad=0.6,
        columnspacing=1.2,
    )
    for lh in leg.legend_handles:
        try:
            lh.set_linewidth(3.5)
        except Exception:
            pass

    if len(labels) > 5:
        fig.tight_layout(rect=[0, 0, 1, 0.9])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.98])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_single_task(
    logs_root: Path,
    env_id: str,
    algos: List[str],
    *,
    out_path: Path,
    best_model: bool = False,
    show_band: bool = True,
    random_start_map: Optional[Dict[Tuple[str, str], float]] = None,
    task_display: Optional[Dict[str, str]] = TASK_DISPLAY_DEFAULT,
    algo_display: Optional[Dict[str, str]] = ALGO_DISPLAY_DEFAULT,
    legend_fontsize: int = 14,
    title_fontsize: int = 14,
    xlabel_fontsize: float = 12,
    xtick_fontsize: float = 11,
    ylabel_fontsize: float = 12,
    ytick_fontsize: float = 11,
    legend_row1: Optional[List[str]] = None,
    legend_row2: Optional[List[str]] = None,
) -> None:
    """
    Single plot for 1 task, legend on top (outside axes).
    """
    set_paper_style()

    if env_id != "trading":
        fig, ax = plt.subplots(1, 1, figsize=(3.6, 3.2))
    else:
        fig, ax = plt.subplots(1, 1, figsize=(6.4, 5.6))
    plot_best_on_axis(
        ax,
        logs_root,
        env_id,
        algos,
        best_model=best_model,
        show_band=show_band,
        random_start_map=random_start_map,
        task_display=task_display,
        algo_display=algo_display,
        title_fontsize=title_fontsize,
    )
    ax.xaxis.label.set_size(xlabel_fontsize)
    ax.tick_params(axis="x", labelsize=xtick_fontsize)
    ax.yaxis.label.set_size(ylabel_fontsize)
    ax.tick_params(axis="y", labelsize=ytick_fontsize)

    if env_id == "trading":
        apply_trading_piecewise_yscale(ax)
        apply_y_dollar_formatter(ax, scale=100.0, decimals=0)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend().remove()

    handles, labels = _dedup_legend(handles, labels)
    handles, labels, ncol = apply_legend_ordering(
        handles,
        labels,
        algo_display,
        legend_row1=legend_row1,
        legend_row2=legend_row2,
        default_ncol=4,
        split_threshold=5,
    )

    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.10),
        ncol=ncol,
        frameon=False,
        fontsize=legend_fontsize,
        handlelength=2.8,
        handletextpad=0.6,
        columnspacing=1.2,
    )
    for lh in leg.legend_handles:
        try:
            lh.set_linewidth(3.5)
        except Exception:
            pass

    if env_id != "trading":
        fig.tight_layout(rect=[0, 0, 1, 0.98])
    else:
        fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# -----------------------------
# Plot trading side-by-side
# -----------------------------


def plot_trading_side_by_side(
    logs_root: Path,
    out_path: Path,
    *,
    continuous_algos: Optional[List[str]] = None,
    discrete_algos: Optional[List[str]] = None,
    random_start_map: Optional[Dict[Tuple[str, str], float]] = None,
    task_display: Optional[Dict[str, str]] = None,
    algo_display: Optional[Dict[str, str]] = None,
    legend_fontsize: int = 14,
    title_fontsize: int = 14,
    xlabel_fontsize: float = 12,
    xtick_fontsize: float = 11,
    ylabel_fontsize: float = 12,
    ytick_fontsize: float = 11,
    legend_row1: Optional[List[str]] = None,
    legend_row2: Optional[List[str]] = None,
) -> None:
    """
    Trading plot side-by-side:
      left  = continuous group
      right = discrete group (must include ct_sac)

    Legend is shared on top.
    Supports optional legend ordering via legend_row1/legend_row2
      - if legend_row1 set and legend_row2 None: single-row forced order
      - if both set and many labels: two-row forced order
    """
    set_paper_style()

    task_display = task_display or {}
    algo_display = algo_display or {}

    if continuous_algos is None:
        continuous_algos = ["ct_sac", "ct_td3", "cppo", "q_learning"]

    if discrete_algos is None:
        discrete_algos = ["ct_sac", "sac", "td3", "ppo", "trpo"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 5.0), sharey=False)

    def _plot_panel(ax: plt.Axes, algos: List[str], title: str) -> None:
        plot_best_on_axis(
            ax=ax,
            logs_root=logs_root,
            env_id="trading",
            algos=algos,
            best_model=True,
            random_start_map=random_start_map,
            task_display=task_display,
            algo_display=algo_display,
            title_fontsize=title_fontsize,
        )
        apply_trading_piecewise_yscale(ax)
        apply_y_dollar_formatter(ax, scale=100.0, decimals=0)

        ax.set_title(title, fontsize=title_fontsize, fontweight="bold", pad=10)

        # axis label + tick fonts
        ax.xaxis.label.set_size(xlabel_fontsize)
        ax.tick_params(axis="x", labelsize=xtick_fontsize)
        ax.yaxis.label.set_size(ylabel_fontsize)
        ax.tick_params(axis="y", labelsize=ytick_fontsize)

        # ensure we don't show per-axis legends
        leg = ax.get_legend()
        if leg is not None:
            leg.remove()

    _plot_panel(axes[0], continuous_algos, "Trading (Continuous)")
    _plot_panel(axes[1], discrete_algos, "Trading (Discrete)")

    # cleaner: only left y-label
    axes[1].set_ylabel("")

    # shared legend (dedup across panels)
    handles, labels = [], []
    for ax in axes:
        h, l = ax.get_legend_handles_labels()
        handles.extend(h)
        labels.extend(l)

    handles, labels = _dedup_legend(handles, labels)

    # optional ordering + ncol decision
    handles, labels, ncol = apply_legend_ordering(
        handles,
        labels,
        algo_display,
        legend_row1=legend_row1,
        legend_row2=legend_row2,
        default_ncol=4,
        split_threshold=5,
    )

    leg = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.06),
        ncol=ncol,
        frameon=False,
        fontsize=legend_fontsize,
        handlelength=3.2,
        handletextpad=0.8,
        columnspacing=1.6,
    )
    for lh in leg.legend_handles:
        try:
            lh.set_linewidth(3.5)
        except Exception:
            pass

    # reserve space for legend
    fig.tight_layout(rect=[0, 0, 1, 0.88])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
