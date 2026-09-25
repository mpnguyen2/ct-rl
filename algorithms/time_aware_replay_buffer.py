from __future__ import annotations

import math
from typing import Any, Optional

import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import ReplayBufferSamples
from stable_baselines3.common.vec_env import VecNormalize


class TimeAwareReplayBuffer(ReplayBuffer):
    """SB3 replay buffer with a discount for each transition's elapsed time."""

    def __init__(
        self,
        *args: Any,
        base_gamma: float,
        dt_default: float,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0.0 < base_gamma <= 1.0:
            raise ValueError(f"base_gamma must be in (0, 1], got {base_gamma}")
        if dt_default <= 0.0:
            raise ValueError(f"dt_default must be positive, got {dt_default}")

        # base_gamma is the discount over one native environment timestep.
        self.discount_rate = -math.log(float(base_gamma)) / float(dt_default)
        self.time_discounts = np.ones(
            (self.buffer_size, self.n_envs), dtype=np.float32
        )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        write_pos = self.pos
        elapsed = np.asarray(
            [self._elapsed_time(info) for info in infos], dtype=np.float64
        )
        if elapsed.shape != (self.n_envs,):
            raise ValueError(
                f"Expected dt for {self.n_envs} environments, got {elapsed.shape}"
            )

        self.time_discounts[write_pos] = np.exp(
            -self.discount_rate * elapsed
        ).astype(np.float32)
        super().add(obs, next_obs, action, reward, done, infos)

    @staticmethod
    def _elapsed_time(info: dict[str, Any]) -> float:
        if "dt" not in info:
            raise KeyError(
                "TimeAwareReplayBuffer requires each environment info dict to "
                "contain the transition duration under 'dt'."
            )
        dt = float(info["dt"])
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError(f"Transition dt must be finite and positive, got {dt}")
        return dt

    def _get_samples(
        self,
        batch_inds: np.ndarray,
        env: Optional[VecNormalize] = None,
    ) -> ReplayBufferSamples:
        env_indices = np.random.randint(0, high=self.n_envs, size=len(batch_inds))

        if self.optimize_memory_usage:
            next_obs = self._normalize_obs(
                self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :],
                env,
            )
        else:
            next_obs = self._normalize_obs(
                self.next_observations[batch_inds, env_indices, :], env
            )

        data = (
            self._normalize_obs(
                self.observations[batch_inds, env_indices, :], env
            ),
            self.actions[batch_inds, env_indices, :],
            next_obs,
            (
                self.dones[batch_inds, env_indices]
                * (1 - self.timeouts[batch_inds, env_indices])
            ).reshape(-1, 1),
            self._normalize_reward(
                self.rewards[batch_inds, env_indices].reshape(-1, 1), env
            ),
            self.time_discounts[batch_inds, env_indices].reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))
