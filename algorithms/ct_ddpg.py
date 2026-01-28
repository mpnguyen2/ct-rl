from typing import Union, Optional, Type
import numpy as np
import torch as th
import torch.nn.functional as F

from environment.base import ContinuousEnv
from models.base import Model
from .off_policy import OffPolicyAlgorithm
from common.schedules import Schedule
from common.buffers import ReplayBatch
from models.noise import ActionNoise
from models.actor_q_critic import ActorQCriticModel


class CTDDPG(OffPolicyAlgorithm):
    """
    Continuous-time Deep Deterministic Policy Gradient (DDPG) using our theoretical work.
    This is the continuous-time version of DDPG, which can be seen as a special
    case of CT-SAC with alpha=0 and a deterministic policy.
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
        # Euler step
        tau: float = 0.005,
        # Critic loss type
        critic_loss_type: str = "mse",  # "mse" or "smooth_l1"
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

        if not self.model.deterministic_policy:
            raise ValueError(
                "CTDDPG requires a deterministic policy in ActorQCriticModel."
            )

        self.tau = float(tau)
        self.critic_loss_type = critic_loss_type

        # The learning rate will be updated by the base algorithm class
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
        """
        Implement CT-DDPG core update.
        """
        for _ in range(gradient_steps):
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

                # Compute Q_target(s', a') with a' = π_target(s')
                next_actions = self.model.act_target(next_obs)
                q_target_next = self.model.target_min_q(next_obs, next_actions)

                # Construct Q_fast target
                # Q_fast = r + Q_target(s, a) + (e^(-β*dt) Q_target(s',π_t(s')) - Q_target(s, a)) / dt
                with th.no_grad():
                    dt *= self.time_rescale
                    gamma_dt = th.exp(-self.beta * dt)
                    fraction = (gamma_dt * q_target_next - q_target_current) / (
                        dt + 1e-8
                    )
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

            ## Actor update
            for p in self.model.critic_parameters:
                p.requires_grad = False

            actions_pi, _ = self.model.act(obs)
            q_values_pi = self.model.min_q(obs, actions_pi)

            actor_loss = -q_values_pi.mean()
            self.logger.record("train/actor_loss", actor_loss.item())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for p in self.model.critic_parameters:
                p.requires_grad = True

            # Target update
            self.model.soft_update_targets(tau=self.tau, update_actor=True)
