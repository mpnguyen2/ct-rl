# tests/test_env_base.py
import unittest
import numpy as np
import gymnasium as gym

from environment.base import (
    ContinuousEnv,
    generate_uniform_time_grid,
    sample_dt_from_distribution,
    generate_irregular_time_grid,
)


class DummyLinearEnv(ContinuousEnv):
    """
    Minimal ContinuousEnv subclass for testing.

    State: x in R (1D)
    Dynamics: x_{k+1} = x_k + dt
    Reward:  r = -|x_{k+1}|
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self._state = 0.0

    def _reset_physics(self, *, seed=None, options=None):
        self._state = 0.0
        obs = np.array([self._state], dtype=np.float32)
        info = {}
        return obs, info

    def _step_physics(self, action, dt):
        # Ignore action; purely time-driven state for test
        self._state = self._state + float(dt)
        obs = np.array([self._state], dtype=np.float32)
        reward = -abs(self._state)
        terminated = False
        truncated = False
        info = {}
        # Pretend physics uses requested dt exactly
        return obs, reward, terminated, truncated, info, float(dt)


class TestTimeGrids(unittest.TestCase):
    def test_uniform_time_grid(self):
        T = 1.0
        num_steps = 4
        times = generate_uniform_time_grid(T, num_steps)

        self.assertEqual(times.shape, (num_steps + 1,))
        expected = np.linspace(0.0, T, num_steps + 1)
        np.testing.assert_allclose(times, expected)

    def test_irregular_time_grid_constraints(self):
        T = 1.0
        num_steps = 5
        min_dt = 0.05
        max_dt = 0.5
        rng = np.random.default_rng(123)

        times = generate_irregular_time_grid(
            T,
            num_steps,
            min_dt=min_dt,
            max_dt=max_dt,
            time_sampling_kwargs={"dist": "uniform"},
            rng=rng,
        )

        self.assertEqual(times.shape, (num_steps,))
        self.assertTrue(np.isclose(times[0], 0.0))
        self.assertTrue(np.isclose(times[-1], T))

        dts = np.diff(times)
        self.assertTrue(np.all(dts >= min_dt - 1e-9))
        self.assertTrue(np.all(dts <= max_dt + 1e-9))

    def test_sample_dt_from_distribution_with_kwargs(self):
        """Test that time_sampling_kwargs are correctly passed and used."""
        rng = np.random.default_rng(456)
        min_dt = 0.1
        max_dt = 0.2
        physics_dt = 0.01
        kwargs = {
            "dist": "two_tail_uniform",
            "tail_p": 0.9,
            "tail_split": 0.2,
        }
        dt = sample_dt_from_distribution(
            rng,
            min_dt=min_dt,
            max_dt=max_dt,
            physics_dt=physics_dt,
            time_sampling_kwargs=kwargs,
        )
        self.assertGreater(dt, 0)


class TestDummyLinearEnv(unittest.TestCase):
    def test_uniform_grid_and_step_dt(self):
        """Check uniform grid and that step_dt follows it."""
        T = 1.0
        num_steps = 4
        env = DummyLinearEnv(
            time_sampling="uniform",
            dt=T / num_steps,
            episode_duration=T,
        )

        obs0, info0 = env.reset(seed=0)  # noqa: F841

        times = env.time_points
        self.assertIsNotNone(times)
        self.assertEqual(times.shape, (num_steps + 1,))
        np.testing.assert_allclose(times, np.linspace(0.0, T, num_steps + 1))

        self.assertTrue(np.isclose(env.cur_t, 0.0))
        np.testing.assert_allclose(obs0, np.array([0.0], dtype=np.float32))

        # Step through entire grid
        for k in range(num_steps):
            (
                obs,
                t,
                action,
                reward,
                next_obs,
                next_t,
                terminated,
                truncated,
                info,
            ) = env.step_dt(np.array([0.0], dtype=np.float32))

            self.assertTrue(np.isclose(t, times[k]))
            self.assertTrue(np.isclose(next_t, times[k + 1]))
            self.assertTrue(np.isclose(env.cur_t, next_t))

            expected_state = times[k + 1]  # since x starts at 0.0
            np.testing.assert_allclose(
                next_obs, np.array([expected_state], dtype=np.float32)
            )

            self.assertFalse(terminated)
            if k < num_steps - 1:
                self.assertFalse(truncated)
            else:
                self.assertTrue(truncated)
                self.assertTrue(info.get("time_limit_reached", False))
            self.assertGreater(info["dt"], 0.0)

        # One more step: should truncate / not advance time
        (
            obs,
            t,
            action,
            reward,
            next_obs,
            next_t,
            terminated,
            truncated,
            info,
        ) = env.step_dt(np.array([0.0], dtype=np.float32))

        self.assertTrue(truncated)
        self.assertFalse(terminated)
        self.assertTrue(info.get("time_limit_reached", False))

        self.assertTrue(np.isclose(t, env.cur_t))
        self.assertTrue(np.isclose(next_t, env.cur_t))
        np.testing.assert_allclose(next_obs, obs)

    def test_step_wrapper_updates_time_and_returns_gym_tuple(self):
        env = DummyLinearEnv(
            time_sampling="uniform",
            dt=0.25,
            episode_duration=1.0,
        )
        env.reset(seed=123)

        prev_t = env.cur_t
        next_obs, reward, terminated, truncated, info = env.step(
            np.array([0.0], dtype=np.float32)
        )

        # Time should advance
        self.assertGreater(env.cur_t, prev_t)

        # Info may be empty for this dummy env; we only require a dict and correct obs shape.
        self.assertIsInstance(info, dict)
        self.assertEqual(next_obs.shape, env.observation_space.shape)


if __name__ == "__main__":
    unittest.main()
