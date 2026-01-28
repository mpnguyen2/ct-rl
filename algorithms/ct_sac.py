from typing import Union, Optional, Dict, Any, Type
import numpy as np
import torch as th
import torch.nn.functional as F

from environment.base import ContinuousEnv
from models.base import Model
from .off_policy import OffPolicyAlgorithm
from common.schedules import Schedule
from common.buffers import ReplayBatch
from models.actor_q_critic import ActorQCriticModel


class CTSAC(OffPolicyAlgorithm):
    """
    Continuous-time SAC using ActorQCriticModel using our theoretical work.
    """

    def __init__(
        self,
        env: ContinuousEnv,
        model: Union[
            ActorQCriticModel, str, Type[ActorQCriticModel]
        ] = "ActorQCriticModel",
        model_kwargs: Optional[Dict[str, Any]] = None,
        device: Union[str, th.device] = "auto",
        seed: Optional[int] = None,
        gamma: float = 0.99,
        buffer_size: int = 1_000_000,
        learning_rate: Union[float, Schedule] = 3e-4,
        batch_size: int = 256,
        train_freq: int = 1,
        gradient_steps: int = 1,
        learning_starts: int = 100,
        # CT-SAC specific hyperparameters
        alpha: float = 0.2,  # Entropy temperature. Auto means optimize alpha as well
        tau: float = 0.005,  # Euler step
        num_expectation_samples: int = 1,  # Num samples for expectation approximation
        target_entropy: Union[
            float, str
        ] = "auto",  # Target entropy when learning alpha
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
        )
        # Target entropy: "auto" -> -|A|, else use float value
        self.target_entropy = target_entropy
        action_dim = int(np.prod(self.env.action_space.shape))

        if self.target_entropy == "auto":
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = float(self.target_entropy)

        # Learnable log-alpha or fixed entropy coefficient alpha
        self.log_alpha: Optional[th.Tensor] = None
        self.alpha_optimizer: Optional[th.optim.Adam] = None
        self.alpha_tensor: Optional[th.Tensor] = None  # for fixed-alpha case

        if isinstance(alpha, str) and alpha.startswith("auto"):
            # Default initial value when learning alpha
            init_value = 1.0
            if "_" in alpha:
                init_value = float(alpha.split("_")[1])
                assert init_value > 0.0, "Initial alpha must be > 0"

            # log alpha is the trainable parameter
            self.log_alpha = th.log(
                th.ones(1, device=self.device) * init_value
            ).requires_grad_(True)
            self.alpha_optimizer = th.optim.Adam(
                [self.log_alpha], lr=self.lr_schedule(1.0)
            )
            self.alpha = float(init_value)
        else:
            # Fixed alpha
            alpha_float = float(alpha)
            self.alpha_tensor = th.tensor(alpha_float, device=self.device)
            self.alpha = alpha_float

        self.tau = float(tau)
        self.num_expectation_samples = int(num_expectation_samples)

        # The learning rate will be updated by the base algorithm class
        self.actor_optimizer = th.optim.Adam(
            self.model.actor.parameters(), lr=self.lr_schedule(1.0)
        )
        self.critic_optimizer = th.optim.Adam(
            self.model.critic_parameters, lr=self.lr_schedule(1.0)
        )
        self.optimizers = [self.actor_optimizer, self.critic_optimizer]
        if self.alpha_optimizer is not None:
            self.optimizers.append(self.alpha_optimizer)

        # For logging how many gradient updates we’ve done
        self._n_updates = 0

    def _policy_act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = th.as_tensor(obs, device=self.device).float()
        single = obs_t.ndim == 1
        if single:
            obs_t = obs_t.unsqueeze(0)  # (1, obs_dim)
        with th.no_grad():
            actions, _ = self.model.act(obs_t, deterministic=deterministic)
        actions_np = actions.detach().cpu().numpy()

        return actions_np[0] if single else actions_np

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Implement CT-SAC core update:

          Q_fast(s,a) = r + E_a[ Q̃_k(s,a) ]
                        + (γ E_{a'}[Q̃_k(s',a')] - E_a[Q̃_k(s,a)]) / u

        The critic is trained to minimize MSE against Q_fast.
        The target networks are then updated using averaging.
        """
        for _ in range(gradient_steps):
            batch: ReplayBatch = self.replay_buffer.sample(batch_size)

            obs = batch.observations
            actions = batch.actions
            next_obs = batch.next_observations
            rewards = batch.rewards
            dones = batch.dones
            dt = batch.dt  # u in our notation

            ## Alpha (entropy temperature) update
            actions_pi, log_prob_pi = self.model.act(
                obs
            )  # Current policy on sampled states
            alpha_loss = None
            if self.alpha_optimizer is not None and self.log_alpha is not None:
                alpha_tensor = th.exp(self.log_alpha.detach())
                alpha_loss = -(
                    self.log_alpha * (log_prob_pi + self.target_entropy).detach()
                ).mean()
            else:
                # Fixed alpha
                alpha_tensor = self.alpha_tensor

            # Log alpha
            self.alpha = float(alpha_tensor.detach().item())
            self.logger.record("train/alpha", self.alpha)

            if alpha_loss is not None and self.alpha_optimizer is not None:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.logger.record("train/alpha_loss", alpha_loss.item())

            ## Critic update
            with th.no_grad():
                # Compute E[Q̃(s', a')]
                next_obs_repeat = next_obs.repeat_interleave(
                    self.num_expectation_samples, dim=0
                )
                next_actions, next_log_prob = self.model.act(next_obs_repeat)

                # Q̃(s',a') = Q_target(s',a') - alpha * log π(a'|s')
                target_q_min = self.model.target_min_q(next_obs, next_actions)
                q_tilde_next = target_q_min - alpha_tensor * next_log_prob
                q_tilde_next = q_tilde_next.view(
                    batch_size, self.num_expectation_samples, 1
                )
                expectation_q_tilde_next = q_tilde_next.mean(dim=1)

                # Compute E[Q̃(s, a)]
                obs_repeat = obs.repeat_interleave(self.num_expectation_samples, dim=0)
                sampled_actions, sampled_log_prob = self.model.act(obs)
                target_q_min_current = self.model.target_min_q(
                    obs_repeat, sampled_actions
                )
                q_tilde_current = target_q_min_current - alpha_tensor * sampled_log_prob
                q_tilde_current = q_tilde_current.view(
                    batch_size, self.num_expectation_samples, 1
                )
                expectation_q_tilde_current = q_tilde_current.mean(dim=1)

                # Construct Q_fast target
                # Q_fast = r + E[Q̃(s,a)] + (e^(-β*dt) E[Q̃(s',a')] - E[Q̃(s,a)]) / dt
                dt *= self.time_rescale
                gamma_dt = th.exp(-self.beta * dt)

                fraction = (
                    gamma_dt * expectation_q_tilde_next - expectation_q_tilde_current
                ) / (dt + 1e-8)
                future_val = expectation_q_tilde_current + fraction
                q_fast_target = rewards + (1 - dones) * future_val

            # Calculate critic loss
            current_q_list = self.model.q_values(obs, actions)
            critic_loss = sum(F.mse_loss(q, q_fast_target) for q in current_q_list)

            self.logger.record("train/fraction", th.max(th.abs(fraction)).item())
            self.logger.record("train/critic_loss", critic_loss.item())

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            ## Actor update
            for p in self.model.critic_parameters:
                p.requires_grad = False  # Freeze critic parameters for actor update

            # Re-use actions_pi, log_prob_pi from above
            q_values_pi = self.model.min_q(obs, actions_pi)
            actor_loss = (alpha_tensor * log_prob_pi - q_values_pi).mean()
            self.logger.record("train/actor_loss", actor_loss.item())

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Unfreeze critic parameters
            for p in self.model.critic_parameters:
                p.requires_grad = True

            # Target update
            self.model.soft_update_targets(tau=self.tau)

        # Track number of gradient updates
        # self._n_updates += gradient_steps
        # self.logger.record("train/n_updates", self._n_updates)
