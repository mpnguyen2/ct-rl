import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from environment.trading_env import TradingContinuousEnv


class TestTradingEnv(unittest.TestCase):
    def setUp(self):
        # Patch GROUPS to ensure we have multiple tickers for testing cycling
        # We use 2 groups: Tech (2 tickers), Finance (1 ticker)
        self.mock_groups_dict = {
            "Tech": ["T1", "T2"],
            "Finance": ["F1"]
        }
        self.patcher = patch("environment.trading_env.GROUPS", self.mock_groups_dict)
        self.patcher.start()

        self.tickers = ["T1", "T2", "F1"]

        # Generate synthetic timestamps
        # The environment expects timestamps that align with NY trading sessions (09:30 start)
        # We generate 10 days of data to support a short episode test
        dates = pd.date_range(start="2023-01-01", periods=10, freq="B")
        timestamps_list = []
        for d in dates:
            # Create a session starting at 09:30 America/New_York
            ts_start = pd.Timestamp(d).tz_localize("America/New_York") + pd.Timedelta(
                hours=9, minutes=30
            )
            # 390 minutes per trading session
            day_mins = pd.date_range(start=ts_start, periods=390, freq="min")
            timestamps_list.append(day_mins)

        # Convert to UTC numpy array as expected by the env
        self.timestamps = np.concatenate(
            [ts.tz_convert("UTC").values for ts in timestamps_list]
        )

        T = len(self.timestamps)
        A = len(self.tickers)

        # Generate synthetic features [A, T, 23]
        # Feature 0 is price, others are lags. We make price positive.
        self.features = np.random.randn(A, T, 23).astype(np.float32)
        self.features[:, :, 0] = np.abs(self.features[:, :, 0]) + 100.0

        # Lags array
        self.lags = np.arange(23, dtype=np.int32)

        # Common kwargs for env initialization
        self.env_kwargs = {
            "timestamps": self.timestamps,
            "tickers": self.tickers,
            "features": self.features,
            "lags": self.lags,
            "episode_days": 5,  # Short episode for testing
            "init_capital": 10000.0,
            "render_mode": None,
            "dt": 1.0,
        }

    def tearDown(self):
        self.patcher.stop()

    def test_initialization(self):
        """Test that the environment initializes correctly with provided data."""
        env = TradingContinuousEnv(**self.env_kwargs)
        self.assertIsInstance(env, TradingContinuousEnv)
        self.assertEqual(env.action_space.shape, (2,))
        # Obs dim check: 2*23 (features) + 5*2 (past actions) + 5 (past rewards) + 2 (positions) + 1 (capital) = 46+10+5+2+1 = 64
        self.assertEqual(env.observation_space.shape, (64,))
        env.close()

    def test_reset(self):
        """Test reset functionality and initial observation."""
        env = TradingContinuousEnv(**self.env_kwargs)
        obs, info = env.reset(seed=42)
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertIn("capital", info)
        self.assertEqual(info["capital"], 10000.0)
        self.assertIn("chosen", info)
        self.assertEqual(len(info["chosen"]), 2)
        env.close()

    def test_step(self):
        """Test standard Gym step."""
        env = TradingContinuousEnv(**self.env_kwargs)
        env.reset(seed=42)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("PnL", info)
        env.close()

    def test_eval_mode(self):
        """Test deterministic evaluation with quarters and ticker cycling."""
        kwargs = self.env_kwargs.copy()
        kwargs["eval_range"] = "Q1_2023"
        kwargs["eval_cycle_tickers"] = True

        env = TradingContinuousEnv(**kwargs)

        # We expect 2 valid episodes in the 10-day data with 5-day stride.
        # Tickers: Tech=[T1, T2], Finance=[F1]. Cycle len = 2.
        # Sequence: (Ep0, T0), (Ep1, T0), (Ep0, T1), (Ep1, T1), (Ep0, T0)...

        # 1. Ep 0, Ticker 0
        _, info = env.reset()
        chosen = info["chosen"]
        self.assertEqual(chosen[0]["ticker"], "T1")
        self.assertEqual(chosen[1]["ticker"], "F1")
        start_date_0 = chosen[0]["start_date_ny"]

        # 2. Ep 1, Ticker 0
        _, info = env.reset()
        chosen = info["chosen"]
        self.assertEqual(chosen[0]["ticker"], "T1")
        self.assertEqual(chosen[1]["ticker"], "F1")
        start_date_1 = chosen[0]["start_date_ny"]
        self.assertNotEqual(start_date_0, start_date_1)

        # 3. Ep 0, Ticker 1
        _, info = env.reset()
        chosen = info["chosen"]
        self.assertEqual(chosen[0]["ticker"], "T2")
        self.assertEqual(chosen[1]["ticker"], "F1")
        self.assertEqual(chosen[0]["start_date_ny"], start_date_0)

        # 4. Ep 1, Ticker 1
        _, info = env.reset()
        chosen = info["chosen"]
        self.assertEqual(chosen[0]["ticker"], "T2")
        self.assertEqual(chosen[1]["ticker"], "F1")
        self.assertEqual(chosen[0]["start_date_ny"], start_date_1)

        # 5. Cycle back to Ep 0, Ticker 0
        _, info = env.reset()
        chosen = info["chosen"]
        self.assertEqual(chosen[0]["ticker"], "T1")
        self.assertEqual(chosen[1]["ticker"], "F1")
        self.assertEqual(chosen[0]["start_date_ny"], start_date_0)

        env.close()


if __name__ == "__main__":
    unittest.main()
