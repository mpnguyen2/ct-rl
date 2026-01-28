# evaluations/performance_report.py
from __future__ import annotations

import argparse
from pathlib import Path

from .config import (
    TASK_DISPLAY_DEFAULT,
    ALGO_DISPLAY_DEFAULT,
)


from .load_helpers import (
    load_random_start_map,
    build_maps,
)
from .plot_helpers import (
    plot_single_task,
    plot_task_grid,
    plot_trading_side_by_side,
)

from .report_helpers import (
    make_ablation_grid_top3_with_diffs,
    run_ctsac_significance_tests,
    print_only_not_significant,
    make_top_runtime_mean_std_table,
)

CONTROL_TASKS = ["cheetah-run", "walker-run", "humanoid-walk", "quadruped-run"]
ALL_TASKS = CONTROL_TASKS + ["trading"]

# Algorithm groups
ALGOS_CONTINUOUS = ["ct_sac", "ct_td3", "cppo", "q_learning"]
ALGOS_DISCRETE_VS = ["ct_sac", "sac", "td3", "ppo", "trpo"]
ALGOS_ALL = [
    "ct_sac",
    "ct_td3",
    "cppo",
    "q_learning",
    "sac",
    "td3",
    "ppo",
    "trpo",
]

# CT-SAC versus discrete and reward-shaping version
ALGOS_CTSAC_VS_INCREMENT_MODELING = [
    "ct_sac",
    "sac",
    "sac_increment_modeling",
]

LEGEND_FONTSIZE = 16
TITLE_FONTSIZE = 14


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--logs_root",
        type=str,
        default="logs",
        help="Parent log folder",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="out/final_reports",
        help="Where to write plots/tables/reports",
    )
    parser.add_argument("--random_start_npz", type=str, default="out/random_start.npz")
    parser.add_argument("--parts_to_run", type=str, default="A,B,C,D,E,F,G")
    args = parser.parse_args()

    logs_root = Path(args.logs_root)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    parts_to_run = [p.strip() for p in args.parts_to_run.split(",")]

    # Load warm-up random action file
    random_start_map = None
    rs_path = Path(args.random_start_npz)
    if rs_path.exists():
        random_start_map = load_random_start_map(rs_path)

    metric_by_task, best_model_by_task = build_maps(ALL_TASKS)
    _, best_4 = build_maps(CONTROL_TASKS)

    # -----------------------------
    # TASK A: Plot subgroups: ct_sac against continuous-time algos and ct_sac against discrete-time algos
    # -----------------------------
    if "A" in parts_to_run:
        plot_task_grid(
            logs_root,
            CONTROL_TASKS,
            ALGOS_CONTINUOUS,
            out_path=out_dir / "continuous_algorithms_control_tasks.png",
            best_model_by_task=best_4,
            show_band=True,
            random_start_map=random_start_map,
            task_display=TASK_DISPLAY_DEFAULT,
            algo_display=ALGO_DISPLAY_DEFAULT,
            legend_fontsize=LEGEND_FONTSIZE,
            title_fontsize=TITLE_FONTSIZE,
            xlabel_fontsize=11.5,
            xtick_fontsize=11.5,
            ylabel_fontsize=11.5,
            ytick_fontsize=11.5,
            legend_row1=["CT-SAC", "q-Learning", "CPPO", "CT-TD3"],
            legend_row2=None,
        )
        plot_task_grid(
            logs_root,
            CONTROL_TASKS,
            ALGOS_DISCRETE_VS,
            out_path=out_dir / "discrete_algorithms_control_tasks.png",
            best_model_by_task=best_4,
            show_band=True,
            random_start_map=random_start_map,
            task_display=TASK_DISPLAY_DEFAULT,
            algo_display=ALGO_DISPLAY_DEFAULT,
            legend_fontsize=LEGEND_FONTSIZE,
            title_fontsize=TITLE_FONTSIZE,
            xlabel_fontsize=11.5,
            xtick_fontsize=11.5,
            ylabel_fontsize=11.5,
            ytick_fontsize=11.5,
            legend_row1=["CT-SAC", "SAC", "TD3", "PPO", "TRPO"],
            legend_row2=None,
        )
        print(f"[info] Done plotting Continuous-time & Discrete-time subgroups.")

    # -----------------------------
    # TASK B: Plot trading tasks
    # -----------------------------
    if "B" in parts_to_run:
        plot_single_task(
            logs_root,
            "trading",
            ALGOS_ALL,
            out_path=out_dir / "trading_task_continuous_and_discrete.png",
            best_model=True,
            show_band=True,
            random_start_map=random_start_map,
            task_display=TASK_DISPLAY_DEFAULT,
            algo_display=ALGO_DISPLAY_DEFAULT,
            legend_fontsize=LEGEND_FONTSIZE,
            title_fontsize=18,
            xlabel_fontsize=12,
            xtick_fontsize=12,
            ylabel_fontsize=14,
            ytick_fontsize=12,
            legend_row1=["CT-SAC", "CPPO", "q-Learning", "CT-TD3"],
            legend_row2=["SAC", "TD3", "PPO", "TRPO"],
        )
        plot_trading_side_by_side(
            logs_root=logs_root,
            out_path=out_dir / "trading_side_by_side.png",
            random_start_map=random_start_map,
            legend_fontsize=LEGEND_FONTSIZE,
            title_fontsize=TITLE_FONTSIZE,
            xlabel_fontsize=12,
            xtick_fontsize=12,
            ylabel_fontsize=14,
            ytick_fontsize=12,
            legend_row1=["CT-SAC", "CPPO", "q-Learning", "CT-TD3"],
            legend_row2=["SAC", "TD3", "PPO", "TRPO"],
        )
        print(f"[info] Done plotting trading tasks.")

    # -----------------------------
    # TASK C: Plot all algorithms on control tasks
    # -----------------------------
    if "C" in parts_to_run:
        plot_task_grid(
            logs_root,
            CONTROL_TASKS,
            ALGOS_ALL,
            out_path=out_dir / "all_algorithms_control_tasks",
            best_model_by_task=best_4,
            show_band=False,
            random_start_map=random_start_map,
            task_display=TASK_DISPLAY_DEFAULT,
            algo_display=ALGO_DISPLAY_DEFAULT,
            legend_fontsize=LEGEND_FONTSIZE,
            title_fontsize=TITLE_FONTSIZE,
            xlabel_fontsize=11.5,
            xtick_fontsize=11.5,
            ylabel_fontsize=11.5,
            ytick_fontsize=11.5,
            legend_row1=["CT-SAC", "q-Learning", "CPPO", "CT-TD3"],
            legend_row2=["SAC", "TD3", "PPO", "TRPO"],
        )
        print(f"[info] Done plotting all algorithms on control tasks.")

    # -----------------------------
    # TASK D: Plot increment modeling for each task
    # -----------------------------
    if "D" in parts_to_run:
        best_inc = {t: False for t in CONTROL_TASKS}
        plot_task_grid(
            logs_root,
            CONTROL_TASKS,
            ALGOS_CTSAC_VS_INCREMENT_MODELING,
            out_path=out_dir / "increment_modeling_control_tasks",
            best_model_by_task=best_inc,
            show_band=True,
            random_start_map=random_start_map,
            task_display=TASK_DISPLAY_DEFAULT,
            algo_display=ALGO_DISPLAY_DEFAULT,
            legend_fontsize=LEGEND_FONTSIZE,
            title_fontsize=TITLE_FONTSIZE,
            xlabel_fontsize=11.5,
            xtick_fontsize=11.5,
            ylabel_fontsize=11.5,
            ytick_fontsize=11.5,
            legend_row1=["CT-SAC", "SAC", "SAC-increment modeling"],
            legend_row2=None,
        )

        plot_single_task(
            logs_root,
            "trading",
            ALGOS_CTSAC_VS_INCREMENT_MODELING,
            out_path=out_dir / "increment_modeling_trading.png",
            best_model=True,
            show_band=True,
            random_start_map=random_start_map,
            task_display=TASK_DISPLAY_DEFAULT,
            algo_display=ALGO_DISPLAY_DEFAULT,
            legend_fontsize=LEGEND_FONTSIZE,
            title_fontsize=18,
            xlabel_fontsize=12,
            xtick_fontsize=12,
            ylabel_fontsize=14,
            ytick_fontsize=12,
            legend_row1=["CT-SAC", "SAC", "SAC-increment modeling"],
            legend_row2=None,
        )

        print(f"[info] Done plotting increment modeling.")

    # -----------------------------
    # TASK E: Ablation study
    # -----------------------------
    if "E" in parts_to_run:
        ablation_grid = make_ablation_grid_top3_with_diffs(
            logs_root=logs_root,
            tasks=ALL_TASKS,
            algos=ALGOS_ALL,
            hyperparams_dir="benchmarks/hyperparams",
            metric_by_task=metric_by_task,
            best_model_by_task=best_model_by_task,
        )

        # Save CSV (multiline cells are fine)
        ablation_grid.to_csv(out_dir / "ablation.csv")

        # Save HTML with line breaks preserved
        html = ablation_grid.to_html(escape=True).replace("\\n", "<br>")
        (out_dir / "ablation.html").write_text(html, encoding="utf-8")
        print(f"[info] Done ablation report.")

    # -----------------------------
    # TASK F: Statistical analysis
    # -----------------------------
    if "F" in parts_to_run:
        others_for_tests = [
            "sac",
            "td3",
            "ppo",
            "trpo",
            "cppo",
            "q_learning",
            "ct_td3",
            "sac_increment_modeling",
        ]

        df_stats = run_ctsac_significance_tests(
            logs_root,
            ALL_TASKS,
            others_for_tests,
            alpha=0.05,
        )

        # Save ALL details to one CSV
        stats_csv = out_dir / "significance_testing.csv"
        df_stats.to_csv(stats_csv, index=False)

        # Print only NOT significant (per test)
        print_only_not_significant(df_stats)

        print(f"[info] Done performing statistical testing.")

    # -----------------------------
    # TASK G: Runtime report mean ± std over 12 seeds
    # -----------------------------
    if "G" in parts_to_run:
        rt_grid, rt_details = make_top_runtime_mean_std_table(
            logs_root=logs_root,
            tasks=ALL_TASKS,
            algos=ALGOS_ALL,
            seeds=list(range(12)),
            mode="top",
        )
        rt_grid.to_csv(out_dir / "runtime_mean_std.csv")
        print("[info] Done runtime report.")


if __name__ == "__main__":
    main()
