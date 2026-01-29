# evaluations/test_evaluations.py
import unittest
from unittest.mock import MagicMock, patch

from models.base import Model
from evaluations.evaluation_helpers import (
    create_evaluation_env_and_model,
    ALGO_CLASS_MAP,
)
from environment import ContinuousEnv, DMCContinuousEnv
import torch as th


class TestEvaluationModules(unittest.TestCase):
    """
    Unit tests for the evaluation modules.
    """

    def setUp(self):
        """Set up common mocks and parameters for tests."""
        self.env_id = "cheetah-run"
        self.seed = 42
        self.algo = "ct_sac"
        self.mode = "irregular_time"  # Use a mode that has time_sampling_kwargs
        self.expected_env_kwargs = {
            "time_sampling": "irregular",
            "dt": 0.01,
            "physics_dt": 0.002,
            "min_dt": 0.002,
            "max_dt": 0.016,
            "max_steps": 1000,
            "episode_duration": 10.0,
            "time_sampling_kwargs": {"tail_p": 0.8, "tail_split": 0.8},
        }
        self.expected_model_kwargs = {
            "q_net_arch": [400, 300],
            "pi_net_arch": [400, 300],
            "activation_fn": th.nn.ReLU,
            "log_std_init": -3.0,
            "n_critics": 2,
            "deterministic_policy": False,
            "use_actor_target": False,
        }

    @patch("evaluations.evaluation_helpers.load_ct_hyperparams_from_table")
    @patch("evaluations.evaluation_helpers.DMCContinuousEnv", spec=DMCContinuousEnv)
    def test_create_evaluation_env_and_model_env_only_with_kwargs(
        self, MockDMCEnv, mock_load_params
    ):
        """
        Tests that create_evaluation_env_and_model correctly creates an environment
        when no model class is provided, and handles time_sampling_kwargs.
        """
        # Mock the hyperparameter loading to return some dummy env_kwargs
        mock_load_params.return_value = (1000, self.expected_env_kwargs, {}, {}, {})

        env, model = create_evaluation_env_and_model(
            env_id=self.env_id,
            model_class=None,  # We are only testing environment creation
            seed=self.seed,
            algo=self.algo,
            mode=self.mode,
        )

        # Assertions
        MockDMCEnv.assert_called_once_with(
            domain_name="cheetah",
            task_name="run",
            seed=self.seed,
            **self.expected_env_kwargs,
        )
        self.assertIsInstance(env, DMCContinuousEnv)
        self.assertIsNone(model)

    @patch("evaluations.evaluation_helpers.load_ct_hyperparams_from_table")
    @patch("evaluations.evaluation_helpers.DMCContinuousEnv", spec=DMCContinuousEnv)
    def test_create_evaluation_env_and_model_with_model(
        self, MockDMCEnv, mock_load_params
    ):
        """
        Tests the creation of an environment and a model instance, including time_sampling_kwargs.
        """
        # Mock the hyperparameter loading to return some dummy env_kwargs and model_kwargs
        mock_load_params.return_value = (
            1000,
            self.expected_env_kwargs,
            self.expected_model_kwargs,
            {},
            {},
        )

        # Mock the model class that would be returned by ALGO_CLASS_MAP
        MockModelClass = MagicMock(spec=ALGO_CLASS_MAP[self.algo])
        # Need to mock the observation and action spaces that the env would provide
        mock_env_instance = MockDMCEnv.return_value
        mock_env_instance.observation_space = MagicMock()
        mock_env_instance.action_space = MagicMock()

        env, model = create_evaluation_env_and_model(
            env_id=self.env_id,
            model_class=MockModelClass,
            seed=self.seed,
            algo=self.algo,
            mode=self.mode,
        )

        # Assertions for environment
        MockDMCEnv.assert_called_once_with(
            domain_name="cheetah",
            task_name="run",
            seed=self.seed,
            **self.expected_env_kwargs,
        )
        self.assertIsInstance(env, DMCContinuousEnv)

        # Assertions for model
        self.assertIsNotNone(model)
        MockModelClass.assert_called_once_with(
            observation_space=mock_env_instance.observation_space,
            action_space=mock_env_instance.action_space,
            **self.expected_model_kwargs,
        )


if __name__ == "__main__":
    unittest.main()
