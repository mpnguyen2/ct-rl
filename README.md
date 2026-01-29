# Continuous-Time Reinforcement Learning (CT-RL)

This repository contains implementations of CT-SAC and CT-TD3 and benchmarks for other Continuous-Time Reinforcement Learning (CT-RL) algorithms together standard Discrete-Time RL baselines. It includes environments for control tasks (DeepMind Control Suite) and financial trading.

## Installation & Setup

It is recommended to use a Conda environment to manage dependencies.

1. **Create and activate a Conda environment:**
   ```bash
   conda create -n ct-rl python=3.9
   conda activate ct-rl
   ```

2. **Install dependencies:**
   You can install the required packages using the provided `requirements.txt`.
   ```bash
   pip install -r requirements.txt
   ```
   *Note: If you need to generate a new requirements file based on imports, you can use `pipreqs`.*

## Training

The repository provides two main entry points for training: one for Continuous-Time algorithms and one for Discrete-Time algorithms.

### Continuous-Time RL
To train a continuous-time algorithm (e.g., CT-SAC), use the `benchmarks.run_ct_rl` module:

```bash
python -m benchmarks.run_ct_rl --algo ct_sac --env_id cheetah-run --log_root log_test --save_root save_test --total_timesteps 80000
```

**Parameters:**
*   `--algo`: The algorithm to use. Options: `ct_sac`, `ct_td3`, `cppo`, `q_learning`.
*   `--env_id`: The environment ID. Options: `cheetah-run`, `walker-run`, `humanoid-walk`, `quadruped-run`, `trading`.
*   `--log_root`: Directory to store TensorBoard logs and CSV metrics.
*   `--save_root`: Directory to save trained models.
*   `--total_timesteps`: Total number of timesteps to train.

### Discrete-Time RL
To train a standard discrete-time algorithm (e.g., SAC) using Stable-Baselines3, use the `benchmarks.run_discrete_rl` module:

```bash
python -m benchmarks.run_discrete_rl --algo sac --env_id cheetah-run --log_root log_test --save_root save_test --total_timesteps 80000
```

**Parameters:**
*   `--algo`: The algorithm to use. Options: `sac`, `td3`, `ppo`, `trpo`.
*   `--env_id`: Same options as above.

## Reproducing Results & Reporting

This repository includes a comprehensive evaluation pipeline to generate plots, tables, and statistical tests.

### Large Assets & Pre-trained Models

The `logs`, `saved_models`, and `data/trading/processed_data` asset folders are too large to be included directly in the repository. They are hosted on SwissTransfer via 2 different links. Because these links expire every month due to SwissTransfer's policy, they will be updated here frequently for reproducibility:

Link1: [https://www.swisstransfer.com/d/702cbd8d-825c-4a69-bc67-79383c210482](https://www.swisstransfer.com/d/702cbd8d-825c-4a69-bc67-79383c210482)

Link2: [https://www.swisstransfer.com/d/172ac0ef-4423-4e28-80e0-a9e962709066](https://www.swisstransfer.com/d/172ac0ef-4423-4e28-80e0-a9e962709066)

Password: icml2026

Please download from these SwissTransfer links the following folders `logs`, `saved_models`, and `data/trading/processed_data` in order to get train logs, saved checkpoints, and trading processed features. After that, please proceed to the below steps for reproducing our reports.

### Performance Report
To generate the full performance report from trained logs:

```bash
python -m evaluations.performance_report
```

This script will read from your log directories and produce the following in `out/final_reports`:
*   **RL Evaluation Plots**: Learning curves comparing algorithms.
*   **Ablation Table**: Analysis of top performing hyperparameters.
*   **Significance Testing**: Statistical analysis using Welch t-test and paired t-test.
*   **Runtime Analysis**: Execution time statistics.

**Hyperparameters:**
Detailed hyperparameter tuning spaces for each of the 5 environments and 8 algorithms (one file per algorithm) can be found in:
`out/final_report/hyperparam_spaces`

### Evaluation on Regular Settings
To evaluate the performance of models trained under irregular time settings when deployed in a regular time setting:

```bash
python -m evaluations.evaluation_on_regular
```

## Trading Environment Data

If using the `trading` environment, you need to download and preprocess the data first.

1.  **Download Data**: Use `data/trading/download_data.py` to fetch data using the Alpaca API.
2.  **Process Data**: Use `data/trading/preprocess_data.py` to generate the feature sets required for the environment.

## Directory Structure

*   **`algorithms/`**: Implementations of all main continuous-time RL algorithms (CT-SAC, CT-TD3, CPPO, etc.).
*   **`benchmarks/`**: Contains the main training scripts (`run_ct_rl.py` and `run_discrete_rl.py`) and hyperparameter configurations.
*   **`common/`**: Shared utilities, including callbacks, replay buffers, and logger configurations.
*   **`data/`**: Logic for downloading financial data and processing it for the trading environment.
*   **`environment/`**: Implementations of the environments (DMC wrappers, Trading environment).
*   **`evaluations/`**: Scripts for generating reports, plots, and statistical analysis from saved logs and models.
*   **`models/`**: Definitions of neural networks (Stochastic policies, Value functions V, q, and Q=V+q) used by the algorithms.
*   **`tests/`**: Unit and integration tests.

## Media

Visualizations of agent performance (videos and images) are generated in `out/final_reports/media/images` or `out/final_reports/media/videos`.

### Sample Episodes

**Walker Sample Run**

The image below shows CT-SAC's performance over a single Walker episode, compared against continuous-time baselines

![Continuous-time RL Walker](sample_images/walker-run_ct_sac_vs_continuous_regular_last.png)

---

**Cheetah Sample Run**

The image below shows CT-SAC's performance over a single Cheetah episode, compared against discrete-time baselines

![Dicrete-time RL Cheetah](sample_images/cheetah-run_ct_sac_vs_discrete_regular_last.png)

---

**Trading Sample Run**

The image below shows CT-SAC's performance over a single Trading (2-weeks) episode, compared against SAC with $15,000 and $5,500 PnL respectively.

![Trading CT-SAC vs SAC](sample_images/trading_ct_sac_vs_sac.png)
