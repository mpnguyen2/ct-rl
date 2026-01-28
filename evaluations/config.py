from typing import Dict


# -----------------------------
# Global style / display config
# -----------------------------

TASK_DISPLAY_DEFAULT: Dict[str, str] = {
    "cheetah-run": "Cheetah",
    "walker-run": "Walker",
    "humanoid-walk": "Humanoid",
    "quadruped-run": "Quadruped",
    "trading": "Trading",
}

ALGO_DISPLAY_DEFAULT: Dict[str, str] = {
    "ct_sac": "CT-SAC",
    "sac": "SAC",
    "td3": "TD3",
    "ppo": "PPO",
    "trpo": "TRPO",
    "cppo": "CPPO",
    "q_learning": "q-Learning",
    "ct_td3": "CT-TD3",
    "sac_increment_modeling": "SAC-increment modeling",
}

# Evaluation and warm-up steps
EVAL_FREQ_DEFAULT = 40_000
MIN_STEP_DEFAULT = 40_000

# Top 3 hyperparams configs
RANK_MODES = ["top", "second", "third"]
