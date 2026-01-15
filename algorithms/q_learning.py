# algorithms/coupled_sarsa.py
from __future__ import annotations

from typing import Optional, Union, Type, Iterable

import numpy as np
import torch as th

from environment.base import ContinuousEnv
from .off_policy import OffPolicyAlgorithm
from common.schedules import Schedule
from common.buffers import ReplayBatch
from models.coupled_vq import CoupledVqModel


class qLearning(OffPolicyAlgorithm):
    """
    (SMALL) q-Learning [Jia & Zhou, 2023] with (V, q, π) modeled by CoupledVqModel.

    [Jia & Zhou, 2023] Yanwei Jia and Xun Yu Zhou. q-learning in continuous time. Journal of Machine Learning Research, 24(161):1-61, 2023.
    """

    def __init__(
        self,
        env: ContinuousEnv,
        model: Union[CoupledVqModel, str, Type[CoupledVqModel]] = "CoupledVqModel",
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
        # q-Learning specific hyperparameters
        alpha: Union[
            float, str
        ] = 0.2,  # Entropy coefficient. Auto means optimize alpha as well
        target_entropy: Union[
            float, str
        ] = "auto",  # Target entropy when learning alpha
        max_grad_norm: float = 0.5,
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
        self.model: CoupledVqModel = self.model
        self.max_grad_norm = float(max_grad_norm)

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
            init_value = 1.0
            if "_" in alpha:
                init_value = float(alpha.split("_")[1])
                assert init_value > 0.0, "Initial alpha must be > 0"

            self.log_alpha = th.log(
                th.ones(1, device=self.device) * init_value
            ).requires_grad_(True)
            self.alpha_optimizer = th.optim.Adam(
                [self.log_alpha], lr=self.lr_schedule(1.0)
            )
            self.alpha = float(init_value)
        else:
            alpha_float = float(alpha)
            self.alpha_tensor = th.tensor(alpha_float, device=self.device)
            self.alpha = alpha_float

        self.v_optimizer = th.optim.Adam(
            self.model.value_parameters,
            lr=self.lr_schedule(1.0),
        )
        self.q_optimizer = th.optim.Adam(
            self.model.rate_parameters,
            lr=self.lr_schedule(1.0),
        )
        self.actor_optimizer = th.optim.Adam(
            self.model.actor_parameters,
            lr=self.lr_schedule(1.0),
        )
        self.optimizers = [
            self.v_optimizer,
            self.q_optimizer,
            self.actor_optimizer,
        ]
        if self.alpha_optimizer is not None:
            self.optimizers.append(self.alpha_optimizer)

    def _policy_act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        obs_t = th.as_tensor(obs, device=self.device).float()
        single = obs_t.ndim == 1
        if single:
            obs_t = obs_t.unsqueeze(0)  # (1, obs_dim)
        with th.no_grad():
            actions, _ = self.model.act(obs_t, deterministic=deterministic)
        actions_np = actions.detach().cpu().numpy()

        return actions_np[0] if single else actions_np

    def _optimize(
        self,
        optimizer: th.optim.Optimizer,
        parameters: Iterable[th.nn.Parameter],
        loss: th.Tensor,
    ) -> None:
        optimizer.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
        if th.isfinite(grad_norm):
            optimizer.step()
        else:
            optimizer.zero_grad()

    def train(self, gradient_steps: int, batch_size: int) -> None:
        """
        Implement q-learning update using continuous-time TD error δ:
            δ_rate = (V(x_{t+1}) - V(x_t)) + r_t * dt - q(x_t, a_t) * dt - β * V(x_t) * dt

        The parameters are updated using the gradients:
          ∇_θ V(x_t) * δ_rate
          ∇_φ q(x_t, a_t) * δ_rate

        The actor is updated to maximize L = q(s,a) - α log π(a|s).
        """
        for _ in range(gradient_steps):
            batch: ReplayBatch = self.replay_buffer.sample(batch_size)

            obs = batch.observations
            actions = batch.actions
            next_obs = batch.next_observations
            rewards = batch.rewards
            dones = batch.dones
            dt = batch.dt

            # Calculate δ_rate = [(V(x') - V(x))/dt + r(x, a) - q(x, a) - β*V(t)] * dt
            V_t = self.model.value(obs)
            with th.no_grad():
                dt *= self.time_rescale
                V_t_next = self.model.value(next_obs)
                q_t = self.model.rate(obs, actions)
                delta_rate = (V_t_next - V_t) / dt + rewards - q_t - self.beta * V_t
                delta_rate *= dt
                delta_rate *= 1 - dones  # No update for terminal states

            # Update V fct by ∇_θ V(x_t) * δ_rate. Hence, loss for gradient descent is -V * δ_rate.
            v_loss = (-V_t * delta_rate).mean()
            self._optimize(self.v_optimizer, self.model.value_parameters, v_loss)
            self.logger.record("train/critic_loss", v_loss.item())

            # Update q function by ∇_φ q(x_t, a_t) * δ_rate. Hence, loss for gradient descent is -q * δ_rate.
            q_t = self.model.rate(obs, actions)
            q_loss = (-q_t * delta_rate).mean()
            self._optimize(self.q_optimizer, self.model.rate_parameters, q_loss)
            self.logger.record("train/q_loss", q_loss.item())

            ## Optimize actor
            # Resample actions from the current policy to update the actor
            actions_pi, log_prob_pi = self.model.act(obs, deterministic=False)

            # Alpha update
            alpha_loss = None
            if self.alpha_optimizer is not None and self.log_alpha is not None:
                alpha_tensor = th.exp(self.log_alpha.detach())
                alpha_loss = -(
                    self.log_alpha * (log_prob_pi + self.target_entropy).detach()
                ).mean()
            else:
                alpha_tensor = self.alpha_tensor

            self.alpha = float(alpha_tensor.detach().item())
            self.logger.record("train/alpha", self.alpha)

            if alpha_loss is not None and self.alpha_optimizer is not None:
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()
                self.logger.record("train/alpha_loss", alpha_loss.item())

            q_pi = self.model.rate(obs, actions_pi)
            actor_loss = (-q_pi + alpha_tensor * log_prob_pi).mean()
            self._optimize(
                self.actor_optimizer, self.model.actor_parameters, actor_loss
            )
            self.logger.record("train/actor_loss", actor_loss.item())
