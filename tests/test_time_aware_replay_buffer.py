import math
import unittest

import numpy as np
from gymnasium import spaces

from algorithms.time_aware_replay_buffer import TimeAwareReplayBuffer


class TestTimeAwareReplayBuffer(unittest.TestCase):
    def test_samples_contain_elapsed_time_discounts(self):
        buffer = TimeAwareReplayBuffer(
            buffer_size=4,
            observation_space=spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32),
            action_space=spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            base_gamma=0.98,
            dt_default=0.01,
            n_envs=1,
        )
        buffer.add(
            np.zeros((1, 2), dtype=np.float32),
            np.ones((1, 2), dtype=np.float32),
            np.zeros((1, 1), dtype=np.float32),
            np.ones(1, dtype=np.float32),
            np.zeros(1, dtype=np.float32),
            [{"dt": 0.03}],
        )

        sample = buffer.sample(1)
        expected = math.exp(-(-math.log(0.98)) * 0.03 / 0.01)
        self.assertAlmostEqual(sample.discounts.item(), expected, places=6)

    def test_rejects_missing_dt(self):
        buffer = TimeAwareReplayBuffer(
            buffer_size=1,
            observation_space=spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            action_space=spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            base_gamma=0.99,
            dt_default=1.0,
            n_envs=1,
        )
        with self.assertRaisesRegex(KeyError, "contain.*'dt'"):
            buffer.add(
                np.zeros((1, 1), dtype=np.float32),
                np.zeros((1, 1), dtype=np.float32),
                np.zeros((1, 1), dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                np.zeros(1, dtype=np.float32),
                [{}],
            )


if __name__ == "__main__":
    unittest.main()
