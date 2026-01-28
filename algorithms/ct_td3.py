from typing import Union, Optional, Type
import numpy as np
import torch as th
import torch.nn.functional as F

from common.utils import get_action_dim
from environment.base import ContinuousEnv
from models.base import Model
from .off_policy import OffPolicyAlgorithm
from common.schedules import Schedule
from common.buffers import ReplayBatch
from models.noise import ActionNoise, GaussianActionNoise
from models.actor_q_critic import ActorQCriticModel


class CTTD3(OffPolicyAlgorithm):
    """
    Continuous-time Twin Delayed Deep Deterministic Policy Gradient (TD3) using our theoretical work.
    This is the continuous-time version of TD3, which improves on CT-DDPG by
    using twin critics, delayed policy updates, and target policy smoothing.
    """

    def __init__(
        self,
        env: ContinuousEnv,
        model: Union[
            ActorQCriticModel, str, Type[ActorQCriticModel]
        ] = "ActorQCriticModel",
        model_kwargs: Optional[dict] = None,
        device: Union[str, th.device] = "auto",
        seed: Optional[int] = None,
        gamma: float = 0.99,
        buffer_size: int = 1_000_000,
        learning_rate: Union[float, Schedule] = 3e-4,
        batch_size: int = 256,
        train_freq: int = 1,
        gradient_steps: int = 1,
        learning_starts: int = 100,
        action_noise: Optional[ActionNoise] = None,
        # Continuous-time TD3 specific hyperparameters
        tau: float = 0.005,  # Euler step
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,  # Used if target_action_noise is None
        target_noise_clip: float = 0.5,
        target_action_noise: Optional[ActionNoise] = None,
    ) -> None:
        super().__init__(
            env=env,
            model=model,
            model_kwargs=model_kwargs,
            device=device,
            seed=seed,
            gamma=gamma,
            buffer_size=buffer_size,
            learning_rate=learning_rate,
            batch_size=batch_size,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            learning_starts=learning_starts,
            action_noise=action_noise,
        )

        if not self.model.deterministic_policy or not self.model.use_actor_target:
            raise ValueError(
                "CTTD3 requires a deterministic policy and a target actor in ActorQCriticModel."
            )

        self.tau = float(tau)
        self.policy_delay = int(policy_delay)
        self.target_noise_clip = float(target_noise_clip)
        self._gradient_step_counter = 0

        if target_action_noise is None:
            # Default to Gaussian noise for target smoothing
            action_dim = get_action_dim(self.env.action_space)
            self.target_action_noise = GaussianActionNoise(
                mean=np.zeros(action_dim),
                sigma=target_policy_noise * np.ones(action_dim),
            )
        else:
            self.target_action_noise = target_action_noise

        self.actor_optimizer = th.optim.Adam(
            self.model.actor_parameters, lr=self.lr_schedule(1.0)
        )
        self.critic_optimizer = th.optim.Adam(
            self.model.critic_parameters, lr=self.lr_schedule(1.0)
        )
        self.optimizers = [self.actor_optimizer, self.critic_optimizer]

    def _policy_act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = th.as_tensor(obs, device=self.device).float()
        single = obs_t.ndim == 1
        if single:
            obs_t = obs_t.unsqueeze(0)  # (1, obs_dim)
        with th.no_grad():
            actions, _ = self.model.act(obs_t, deterministic=True)
        actions_np = actions.detach().cpu().numpy()

        return actions_np[0] if single else actions_np

    def train(self, gradient_steps: int, batch_size: int) -> None:
        for _ in range(gradient_steps):
            self._gradient_step_counter += 1
            batch: ReplayBatch = self.replay_buffer.sample(batch_size)

            obs = batch.observations
            actions = batch.actions
            next_obs = batch.next_observations
            rewards = batch.rewards
            dones = batch.dones
            dt = batch.dt

            ## Critic update
            with th.no_grad():
                # Compute Q_target(s, a) with s and a from batch
                q_target_current = self.model.target_min_q(obs, actions)

                # Generate next actions
                next_actions = self.model.act_target(next_obs)

                # Generate noise from the ActionNoise object
                noise = self.target_action_noise()
                noise = th.as_tensor(noise, dtype=th.float32, device=self.device)

                # Clip noise and add to next_actions
                noise = noise.clamp(-self.target_noise_clip, self.target_noise_clip)
                action_low = th.from_numpy(self.env.action_space.low).to(self.device)
                action_high = th.from_numpy(self.env.action_space.high).to(self.device)
                next_actions = (next_actions + noise).clamp(action_low, action_high)

                # Compute Q_target(s', a')
                q_target_next = self.model.target_min_q(next_obs, next_actions)

                # Construct Q_fast target
                dt *= self.time_rescale
                gamma_dt = th.exp(-self.beta * dt)
                fraction = (gamma_dt * q_target_next - q_target_current) / (dt + 1e-8)
                future_val = q_target_current + fraction
                q_fast_target = rewards + (1 - dones) * future_val

            # Calculate critic loss
            current_q_list = self.model.q_values(obs, actions)
            critic_loss = sum(F.mse_loss(q, q_fast_target) for q in current_q_list)

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            self.logger.record("train/fraction", th.max(th.abs(fraction)).item())
            self.logger.record("train/critic_loss", critic_loss.item())

            ## Delayed policy and target updates
            if self._gradient_step_counter % self.policy_delay == 0:
                # Actor update
                for p in self.model.critic_parameters:
                    p.requires_grad = False
                actions_pi, _ = self.model.act(obs)

                # Actor loss uses the first critic
                q_values_pi = self.model.q_values(obs, actions_pi)[0]
                actor_loss = -q_values_pi.mean()
                self.logger.record("train/actor_loss", actor_loss.item())

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                self.actor_optimizer.step()

                for p in self.model.critic_parameters:
                    p.requires_grad = True

                # Target update
                self.model.soft_update_targets(tau=self.tau, update_actor=True)
