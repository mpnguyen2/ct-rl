# ctrllib/env/base.py
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


class ContinuousEnv(gym.Env, ABC):
    """Base class for continuous-time environments.

    Responsibilities
    ----------------
    - Track current time `cur_t`.
    - Provide a time-aware step method `step_dt(action)` that returns:
        (obs, t, action, reward, next_obs, next_t, terminated, truncated, info)
    - Provide a Gym-compatible `step(action)` that just returns the next state:
        (next_obs, reward, terminated, truncated, info)
    - Optionally pre-generate a time grid at reset:
        - uniform:    dt = T / num_steps
        - irregular:  random timestamps in [0, T]

    Subclasses must implement:
    - `_reset_physics(seed, options) -> (obs, info)`
    - `_step_physics(action, dt) -> (next_obs, reward, terminated, truncated, info, actual_dt)`
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        *,
        time_sampling: str = "uniform",  # "uniform" or "irregular"
        dt: float = 0.02,  # dt for uniform (also serves as mean_dt for irregular sampling)
        physics_dt: Optional[float] = None,  # override physics timestep
        min_dt: Optional[float] = None,  # optional for irregular sampling
        max_dt: Optional[float] = None,  # optional for irregular sampling
        max_steps: Optional[int] = None,  # hard cap on number of steps
        episode_duration: Optional[float] = None,  # total time horizon T
        time_sampling_kwargs: Optional[
            Dict[str, Any]
        ] = None,  # Keyword arguments for time sampling ("irregular")
        return_reward_increment: bool = False,  # Whether to return increment or the whole reward
    ) -> None:
        super().__init__()

        if time_sampling not in ("uniform", "irregular"):
            raise ValueError(
                f"time_sampling must be 'uniform' or 'irregular', got {time_sampling!r}"
            )

        self.time_sampling = time_sampling
        self.dt = float(dt)
        self.min_dt = float(min_dt) if min_dt is not None else None
        self.max_dt = float(max_dt) if max_dt is not None else None
        self.episode_duration = (
            float(episode_duration) if episode_duration is not None else None
        )
        self.max_steps = int(max_steps) if max_steps is not None else None

        # Physics solver timestep (optional). Only used by extreme dt distributions.
        if physics_dt is not None:
            physics_dt = float(physics_dt)
            if physics_dt <= 0.0:
                raise ValueError(f"physics_dt must be > 0, got {physics_dt}")
        self._physics_dt: Optional[float] = (
            float(physics_dt) if physics_dt is not None else None
        )
        self.dt_default: float = self.dt

        # Default dist for irregular sampling is two_tail_uniform.
        self.time_sampling_kwargs: Dict[str, Any] = dict(time_sampling_kwargs or {})
        if self.time_sampling == "irregular":
            self.time_sampling_kwargs.setdefault("dist", "two_tail_uniform")
        else:
            self.time_sampling_kwargs.setdefault("dist", "uniform")
        self.time_sampling_kwargs.setdefault("tail_p", 0.8)
        self.time_sampling_kwargs.setdefault("tail_split", 0.5)
        self.time_sampling_kwargs.setdefault("beta_alpha", 0.5)

        # Current time and indexing into the episode
        self.cur_t: float = 0.0
        self._step_index: int = 0  # number of transitions taken in current episode

        # Optional pre-generated time grid of shape (num_steps + 1,)
        self._time_points: Optional[np.ndarray] = None

        # Last observation (needed to expose (s, t, a, r, s', t') in step_dt)
        self._last_obs: Optional[np.ndarray] = None

        # RNG for dt sampling / irregular grids
        self._np_random: Optional[np.random.Generator] = None

        # Whether to return reward increment
        self.return_reward_increment = return_reward_increment

    @property
    def physics_dt(self) -> Optional[float]:
        """Physics solver timestep (seconds), if known."""
        return self._physics_dt

    # -------------------------------- Abstract methods --------------------------------

    @abstractmethod
    def _reset_physics(
        self,
        *,
        seed: Optional[int],
        options: Optional[Dict[str, Any]],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the underlying simulator and return (obs, info)."""
        raise NotImplementedError

    @abstractmethod
    def _step_physics(
        self,
        action: np.ndarray,
        dt: float,
    ) -> Tuple[
        np.ndarray,  # next_obs
        float,  # reward
        bool,  # terminated
        bool,  # truncated
        Dict[str, Any],  # info
        float,  # actual_dt_used
    ]:
        """Advance the underlying simulator by `dt` (seconds).

        Returns (next_obs, reward, terminated, truncated, info, actual_dt_used).
        `actual_dt_used` is the *real* time advanced by the physics engine (may
        differ slightly from requested dt due to substep quantization).
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Time grid and dt sampling
    # ------------------------------------------------------------------

    def _build_time_grid(self) -> None:
        """Build per-episode time grid if episode_duration is set."""
        if self.episode_duration is not None:
            if self.time_sampling == "uniform":
                num_steps = int(round(self.episode_duration / self.dt))
                self._time_points = generate_uniform_time_grid(
                    self.episode_duration,
                    num_steps,
                )
            else:
                # irregular
                if self.max_steps is None:
                    self.max_steps = int(round(self.episode_duration / self.dt))
                self._time_points = generate_irregular_time_grid(
                    self.episode_duration,
                    self.max_steps,
                    min_dt=self.min_dt,
                    max_dt=self.max_dt,
                    mean_dt=self.dt,
                    rng=self._np_random,
                    physics_dt=self.physics_dt,
                    time_sampling_kwargs=self.time_sampling_kwargs,
                )
        else:
            self._time_points = None

    def _sample_dt_online(self) -> float:
        """Sample dt on-the-fly when no pre-generated time grid is used."""
        if self.time_sampling == "uniform":
            return self.dt

        # Irregular online sampling
        if self._np_random is None:
            # Fallback: create RNG if reset() hasn't been called properly
            self._np_random, _ = gym.utils.seeding.np_random(None)
        low = self.min_dt if self.min_dt is not None else 0.5 * self.dt
        high = self.max_dt if self.max_dt is not None else 1.5 * self.dt
        if high <= low:
            high = low + 1e-8
        return sample_dt_from_distribution(
            self._np_random,
            dist=self.irregular_dist,
            min_dt=float(low),
            max_dt=float(high),
            mean_dt=self.dt,
            physics_dt=self.physics_dt,
            **self.time_sampling_kwargs,
        )

    def _next_dt_request(self) -> float:
        """Return the requested Δt for the next step (may be 0 if episode is over)."""
        if self._time_points is not None:
            # Using a fixed time grid: times shape is (num_steps + 1,)
            if self._step_index >= len(self._time_points) - 1:
                # No more intervals left; consumer should respect termination.
                return 0.0
            t0 = float(self._time_points[self._step_index])
            t1 = float(self._time_points[self._step_index + 1])
            return max(0.0, t1 - t0)
        else:
            # Sample dt on the fly
            return self._sample_dt_online()

    # ----------------------------------- Gymnasium API -----------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset env:

        - initializes RNG
        - resets current time to 0
        - (optionally) builds a fresh time grid for this episode
        - calls `_reset_physics` from the subclass
        """
        # Manage the (numpy) random generator
        if seed is not None:
            self._np_random, seed = gym.utils.seeding.np_random(seed)
        elif self._np_random is None:
            self._np_random, seed = gym.utils.seeding.np_random(None)
        self.cur_t = 0.0
        self._step_index = 0

        # Build per-episode time grid if configured
        self._build_time_grid()

        obs, info = self._reset_physics(seed=seed, options=options)
        self._last_obs = np.asarray(obs, dtype=np.float32)

        return obs, info

    def step(self, action: np.ndarray):
        """Gym-compatible step: returns only (next_obs, reward, terminated, truncated, info).

        The continuous-time info (t, dt, ...) is stored in the `info` dict.
        """
        _, _, _, reward, next_obs, _, terminated, truncated, info = self.step_dt(action)
        return next_obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Continuous-time API
    # ------------------------------------------------------------------

    def step_dt(
        self,
        action: np.ndarray,
    ) -> Tuple[
        np.ndarray,  # obs
        float,  # t
        np.ndarray,  # action
        float,  # reward
        np.ndarray,  # next_obs
        float,  # next_t
        bool,  # terminated
        bool,  # truncated
        Dict[str, Any],  # info
    ]:
        """Continuous-time step with timestamps.

        Returns:
          (obs, t, action, reward, next_obs, next_t, terminated, truncated, info)
        """
        if self._last_obs is None:
            raise RuntimeError("Call reset() before step().")

        obs = self._last_obs
        t = float(self.cur_t)
        action = np.asarray(action, dtype=np.float32)

        dt_req = self._next_dt_request()
        if dt_req <= 0.0 and self._time_points is not None:
            # Time grid exhausted; treat as time-limit truncation (no further stepping).
            truncated = True
            terminated = False
            info: Dict[str, Any] = {"time_limit_reached": True}
            return obs, t, action, 0.0, obs, t, terminated, truncated, info

        next_obs, reward, terminated, truncated, info, dt_used = self._step_physics(
            action,
            dt_req,
        )

        dt_used = float(dt_used)
        if dt_used <= 0.0:
            # Fallback to requested dt if physics didn't report a useful dt
            dt_used = float(dt_req)

        # Adjust reward if we only return increment part
        reward = float(reward)
        reward = (
            reward * (dt_used / self.dt_default)
            if self.return_reward_increment
            else reward
        )

        # Update time / counters
        self.cur_t = float(t + dt_used)
        self._step_index += 1
        self._last_obs = np.asarray(next_obs, dtype=np.float32)

        # Enforce additional time-limits if configured
        time_limit_reached = False

        if (
            self.episode_duration is not None
            and self.cur_t >= self.episode_duration - 1e-9
        ):
            time_limit_reached = True
        if self.max_steps is not None and self._step_index >= self.max_steps:
            time_limit_reached = True

        if time_limit_reached and not terminated:
            truncated = True
            info = dict(info)  # copy to avoid mutating downstream references
            info.setdefault("time_limit_reached", True)

        return (
            obs,
            t,
            action,
            reward,
            next_obs,
            self.cur_t,
            terminated,
            truncated,
            info,
        )

    @property
    def time_points(self) -> Optional[np.ndarray]:
        """Return the last built time grid (shape (num_steps + 1,)) or None."""
        return self._time_points


# ----------------------------------- Time grid helpers -----------------------------------


def generate_uniform_time_grid(
    episode_duration: float,
    num_steps: int,
) -> np.ndarray:
    """Generate uniform time grid: t_0 = 0, t_{num_steps} = T.

    This yields num_steps transitions, each with Δt = T / num_steps.
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be >= 1 for a non-empty episode.")
    return np.linspace(0.0, episode_duration, num_steps + 1, dtype=np.float64)


def sample_dt_from_distribution(
    rng: np.random.Generator,
    *,
    min_dt: float,
    max_dt: float,
    mean_dt: Optional[float] = None,  # used by triangular; uniform ignores it
    physics_dt: Optional[float] = None,  # used ONLY by two_tail_*
    time_sampling_kwargs: Optional[Dict[str, Any]] = None,
) -> float:
    ts = dict(time_sampling_kwargs or {})
    dist = ts.get("dist", "uniform")

    lo, hi = float(min_dt), float(max_dt)
    if hi <= lo:
        return lo

    # Continuous (no quantization)
    if dist == "uniform":
        return float(rng.uniform(lo, hi))

    if dist == "triangular":
        if mean_dt is None:
            raise ValueError("triangular dist requires mean_dt (env dt).")
        mode = 3.0 * float(mean_dt) - lo - hi
        mode = float(np.clip(mode, lo, hi))
        return float(rng.triangular(lo, mode, hi))

    # Quantized only for extreme/two-tail dists
    if dist in ("two_tail_uniform", "two_tail_beta"):
        if physics_dt is None or float(physics_dt) <= 0:
            raise ValueError("two_tail_* requires physics_dt > 0.")
        pd = float(physics_dt)

        n_lo = int(round(lo / pd))
        n_hi = int(round(hi / pd))
        if n_hi < n_lo:
            raise ValueError(
                f"No feasible dt multiple in [{lo}, {hi}] for physics_dt={pd}."
            )

        tail_p = float(ts.get("tail_p", 0.8))
        tail_split = float(ts.get("tail_split", 0.5))
        beta_alpha = float(ts.get("beta_alpha", 0.5))

        if not (0.0 <= tail_p <= 1.0):
            raise ValueError("tail_p must be in [0, 1].")
        if not (0.0 <= tail_split <= 1.0):
            raise ValueError("tail_split must be in [0, 1].")
        if beta_alpha <= 0.0:
            raise ValueError("beta_alpha must be > 0.")

        i_lo, i_hi = n_lo + 1, n_hi - 1
        has_interior = i_lo <= i_hi

        # Put >= tail_p mass on endpoints
        if (rng.random() < tail_p) or (not has_interior):
            choose_min = rng.random() < tail_split
            n = n_lo if choose_min else n_hi
            return float(n * pd)

        # Interior sampling
        if dist == "two_tail_uniform":
            n = int(rng.integers(i_lo, i_hi + 1))
            return float(n * pd)

        # dist == "two_tail_beta": symmetric U-shape via Beta(alpha, alpha)
        u = float(rng.beta(beta_alpha, beta_alpha))
        x = i_lo + u * (i_hi - i_lo)
        n = int(np.clip(int(np.round(x)), i_lo, i_hi))
        return float(n * pd)

    raise ValueError(
        f"Unsupported dist={dist!r}. Use 'uniform', 'triangular', 'two_tail_uniform', or 'two_tail_beta'."
    )


def generate_irregular_time_grid(
    episode_duration: float,
    num_steps: int,
    *,
    min_dt: Optional[float] = None,
    max_dt: Optional[float] = None,
    mean_dt: Optional[float] = None,
    rng: Optional[np.random.Generator] = None,
    physics_dt: Optional[float] = None,
    time_sampling_kwargs: Optional[Dict[str, Any]] = None,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Renewal-style irregular grid.
    Returns times [t0=0, ..., tK] with K <= num_steps and each dt in [min_dt, max_dt].

    - dist is read from `time_sampling_kwargs["dist"]`
    - default dist (if absent) should be set by the caller (e.g., base env init)
    - uniform/triangular are continuous (no quantization)
    - two_tail_* are quantized to multiples of physics_dt
    """
    if num_steps <= 0:
        raise ValueError("num_steps must be >= 1.")
    T = float(episode_duration)
    if T <= 0:
        raise ValueError("episode_duration must be > 0.")
    if rng is None:
        rng = np.random.default_rng()

    if mean_dt is None:
        mean_dt = T / float(num_steps)

    lo = float(min_dt) if min_dt is not None else 0.5 * float(mean_dt)
    hi = float(max_dt) if max_dt is not None else 1.5 * float(mean_dt)
    if lo <= 0:
        raise ValueError("min_dt must be > 0 (or mean_dt must be > 0).")
    if hi < lo:
        raise ValueError(f"Need max_dt >= min_dt, got min_dt={lo}, max_dt={hi}.")

    ts = dict(time_sampling_kwargs or {})
    dist = ts.get("dist", "uniform")

    times = [0.0]
    t = 0.0

    for k in range(int(num_steps)):
        remaining = T - t

        # Can't take another bounded step
        if remaining < lo - eps:
            break

        # Can finish now
        if remaining <= hi + eps:
            if dist in ("two_tail_uniform", "two_tail_beta"):
                if physics_dt is None or float(physics_dt) <= 0:
                    break
                pd = float(physics_dt)
                n = int(np.floor(remaining / pd + 1e-12))
                dt = float(n * pd)
                if dt < lo - eps:
                    break
                dt = float(np.clip(dt, lo, hi))
            else:
                dt = float(np.clip(remaining, lo, hi))

            t += dt
            times.append(t)
            break

        # Need another dt; leave room for a final min_dt if possible
        cap = hi
        if k < int(num_steps) - 1:
            cap = min(hi, remaining - lo)
            if cap < lo + eps:
                break

        dt = sample_dt_from_distribution(
            rng,
            min_dt=lo,
            max_dt=cap,
            mean_dt=mean_dt,
            physics_dt=physics_dt,
            time_sampling_kwargs=ts,
        )
        dt = float(np.clip(dt, lo, cap))
        t += dt
        times.append(t)

    return np.asarray(times, dtype=np.float64)
