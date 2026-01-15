from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import gymnasium as gym
from gymnasium import spaces

from .base import ContinuousEnv
from data.trading.config import (
    GROUPS,
    NY_TZ,
    PRICE_SCALE,
    SESSION_MIN_PER_DAY,
    SESSION_START_HHMM,
    ticker_label,
)


MIN_FRAC_PLOT = 0.06


def _parse_eval_range_ny(range_str: str) -> tuple[str, str]:
    def parse_comp(comp: str, is_end: bool) -> pd.Timestamp:
        # YYYY
        if re.match(r"^\d{4}$", comp):
            y = int(comp)
            if is_end:
                return pd.Timestamp(year=y + 1, month=1, day=1, tz=NY_TZ)
            return pd.Timestamp(year=y, month=1, day=1, tz=NY_TZ)
        
        # Qx_YYYY
        m_q = re.match(r"^Q([1-4])_(\d{4})$", comp)
        if m_q:
            q, y = int(m_q.group(1)), int(m_q.group(2))
            start_month = 3 * (q - 1) + 1
            if is_end:
                if start_month == 10:
                    return pd.Timestamp(year=y + 1, month=1, day=1, tz=NY_TZ)
                else:
                    return pd.Timestamp(year=y, month=start_month + 3, day=1, tz=NY_TZ)
            else:
                return pd.Timestamp(year=y, month=start_month, day=1, tz=NY_TZ)

        # MM_YYYY
        m_m = re.match(r"^(\d{1,2})_(\d{4})$", comp)
        if m_m:
            month, y = int(m_m.group(1)), int(m_m.group(2))
            if is_end:
                if month == 12:
                    return pd.Timestamp(year=y + 1, month=1, day=1, tz=NY_TZ)
                else:
                    return pd.Timestamp(year=y, month=month + 1, day=1, tz=NY_TZ)
            else:
                return pd.Timestamp(year=y, month=month, day=1, tz=NY_TZ)
        
        raise ValueError(f"Unknown date format component: {comp}")

    if "-" in range_str:
        parts = range_str.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid range: {range_str}")
        start = parse_comp(parts[0], is_end=False)
        end = parse_comp(parts[1], is_end=True)
    else:
        start = parse_comp(range_str, is_end=False)
        end = parse_comp(range_str, is_end=True)

    return start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")


@dataclass
class EpisodeSpec:
    start_idx: int  # index into global time axis
    end_idx_exclusive: int  # exclusive
    start_date_ny: str
    end_date_ny: str


def _load_npz(npz_path: str):
    npz = np.load(npz_path, allow_pickle=True)
    timestamps = npz["timestamps"]  # datetime64[ns], UTC-naive but meant as UTC
    tickers = [str(x) for x in npz["tickers"].tolist()]
    features = npz["features"].astype(np.float32, copy=False)  # [A, T, 23]
    lags = npz["lags"].astype(np.int32, copy=False)
    return timestamps, tickers, features, lags


def _timestamps_to_utc_index(timestamps: np.ndarray):
    # timestamps are datetime64[ns] (tz-naive). Treat them as UTC and localize.
    ts = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if ts.tz is not None:
        return ts.tz_convert("UTC")
    return ts.tz_localize("UTC")


def _compute_day_start_indices(ts_utc) -> np.ndarray:
    # Day start = 09:30 NY time
    ts_ny = ts_utc.tz_convert(NY_TZ)
    hh, mm = SESSION_START_HHMM
    is_start = (ts_ny.hour == hh) & (ts_ny.minute == mm)
    return np.where(is_start)[0]


def _build_episode_specs(ts_utc, episode_days: int) -> Tuple[List[EpisodeSpec], int]:
    """
    Build non-overlapping "2-week" episodes aligned to NY day starts.

    episode_days=10 means 10 trading days (390 min each) => episode_points=3900.
    Episodes are sampled by choosing a day-start index s, then taking [s, s+episode_points).
    """
    episode_points = int(episode_days) * int(SESSION_MIN_PER_DAY)

    day_starts = _compute_day_start_indices(ts_utc)
    if day_starts.size == 0:
        raise RuntimeError(
            "No NY 09:30 day-start timestamps found in timestamps array."
        )

    T = len(ts_utc)
    valid_starts = day_starts[day_starts + episode_points <= T]
    if valid_starts.size == 0:
        raise RuntimeError(
            f"No valid episode windows: need episode_points={episode_points} but T={T}."
        )

    ts_ny = ts_utc.tz_convert(NY_TZ)
    specs: List[EpisodeSpec] = []
    for s in valid_starts.tolist():
        e = s + episode_points
        start_date = str(pd.Timestamp(ts_ny[s]).date())
        end_date = str(pd.Timestamp(ts_ny[e - 1]).date())
        specs.append(EpisodeSpec(s, e, start_date, end_date))

    return specs, episode_points


class TradingContinuousEnv(ContinuousEnv):
    """
    Continuous-time trading environment based on preprocessed minute features.
    - Consider G industry groups, where we trade stocks for each group
    - G action dims (one per group): buy/sell intensity in [-1, 1]
    - observation = price features (G*23) + past 5 actions (5*G) + past 5 rewards (5)
                    + positions (G) + capital (1)
      where positions and capital are normalized by init_capital for scale stability.

    Notes on time:
    - We interpret dt (from ContinuousEnv) in minutes.
    - Data is 1-minute (physic_dt=1) grid => we quantize dt to integer minute steps.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        *,
        seed: Optional[int] = None,
        npz_path: Optional[str] = None,
        timestamps: Optional[np.ndarray] = None,
        tickers: Optional[List[str]] = None,
        features: Optional[np.ndarray] = None,
        lags: Optional[np.ndarray] = None,
        # Episode structure
        episode_days: int = 10,  # "2 weeks" = 10 trading days
        # Trading / reward
        init_capital: float = 1000.0,
        max_trade_fraction: float = 0.25,  # max fraction of capital traded per step per asset (via action)
        transaction_cost: float = 1e-3,  # fraction of traded notional
        position_limit_fraction: float = 1.0,  # clamp |position_value| <= position_limit_fraction * init_capital
        # Render
        render_mode: Optional[str] = None,
        # ContinuousEnv time config (units = minutes)
        time_sampling: str = "uniform",  # "uniform" or "irregular"
        dt: float = 1.0,  # mean control dt in minutes
        physics_dt: float = 1.0,
        min_dt: Optional[float] = None,
        max_dt: Optional[float] = None,
        max_steps: Optional[int] = None,
        episode_duration: Optional[float] = None,
        time_sampling_kwargs: Optional[Dict[str, Any]] = None,
        return_reward_increment: bool = False,
        # Deterministic evaluation (optional)
        eval_range: Optional[str] = None,
        eval_cycle_tickers: bool = True,
    ):
        if npz_path is not None:
            timestamps0, tickers0, features0, lags0 = _load_npz(npz_path)
            timestamps = timestamps0
            tickers = tickers0
            features = features0
            lags = lags0

        if timestamps is None or tickers is None or features is None:
            raise ValueError(
                "Provide either npz_path or (timestamps, tickers, features)."
            )
        print("\nDone loading price data.\n")

        self.timestamps = timestamps
        self.tickers = list(tickers)
        self.features_all = np.asarray(features, dtype=np.float32)  # [A,T,23]
        self.lags = np.asarray(lags, dtype=np.int32) if lags is not None else None

        if self.features_all.ndim != 3 or self.features_all.shape[2] != 23:
            raise ValueError(
                f"features must be [A,T,23], got {self.features_all.shape}"
            )

        # Groups of industry to invest in (i.e. Tech, Finance, Energy, or Healthcare)
        self.groups = list(GROUPS.keys())
        self.n_groups = len(self.groups)
        if self.n_groups < 1:
            raise ValueError("GROUPS must contain at least 1 group.")

        # Map ticker -> row in features_all
        self._ticker_to_array_idx = {t: i for i, t in enumerate(self.tickers)}

        # Tickers available per group (keep order)
        self._group_tickers: List[List[str]] = []
        for g in self.groups:
            ts = [t for t in GROUPS[g] if t in self._ticker_to_array_idx]
            if len(ts) == 0:
                raise RuntimeError(f"No tickers for group={g} found in npz tickers.")
            self._group_tickers.append(ts)

        # Episode specs (aligned to day-start; fixed length in points)
        self._ts_utc = _timestamps_to_utc_index(self.timestamps)
        self._episode_specs, self.episode_points = _build_episode_specs(
            self._ts_utc, episode_days
        )

        # Deterministic evaluation plan
        self.eval_range = eval_range
        self.eval_cycle_tickers = bool(eval_cycle_tickers)
        self._eval_spec_indices: Optional[List[int]] = None
        self._eval_spec_ptr: int = 0
        self._eval_ticker_ptr: int = 0
        self._eval_ticker_cycle_len: int = max(len(ts) for ts in self._group_tickers)
        if self.eval_range is not None:
            self._eval_spec_indices = self._build_eval_spec_indices_from_range(
                range_str=self.eval_range,
                stride_days=episode_days,
            )

        # Internal per-episode selection
        self._chosen_spec: EpisodeSpec = self._episode_specs[
            0
        ]  # shared episode window across all groups
        self._chosen_tickers: List[str] = [""] * self.n_groups
        self._chosen_specs: List[EpisodeSpec] = [self._episode_specs[0]] * self.n_groups

        # Trading state.
        self.init_capital = init_capital
        self.max_trade_fraction = max_trade_fraction
        self.transaction_cost = transaction_cost
        self.position_limit_fraction = position_limit_fraction
        self.capital = float(self.init_capital)
        self.positions_shares = np.zeros(self.n_groups, dtype=np.float32)

        self.past_actions = np.zeros((5, self.n_groups), dtype=np.float32)
        self.past_rewards = np.zeros((5,), dtype=np.float32)

        # Price/feature tensor for the chosen tickers in the episode: [G, Te, 23]
        self.X = np.zeros((self.n_groups, self.episode_points, 23), dtype=np.float32)

        # Episode timestamps (same slice for all groups)
        self._ts_ep_utc: List[np.ndarray] = [None] * self.n_groups

        # Trade log (per group) for rendering: (t_idx, shares_delta, price)
        self._trade_log: List[List[Tuple[int, float, float]]] = [
            [] for _ in range(self.n_groups)
        ]

        # Rendering
        self.render_mode = render_mode
        self._fig = None
        self._axs = None
        self._price_lines = None
        self._cursor_lines = None
        self._buy_scatters = None
        self._sell_scatters = None
        self._info_titles = None

        # ContinuousEnv config (in minutes)
        # Episode has episode_points observations => episode_points-1 transitions.
        if max_steps is None:
            max_steps = int(self.episode_points - 1)

        adjusted_episode_duration = float(self.episode_points - 1)
        if episode_duration is not None:
            adjusted_episode_duration = min(
                adjusted_episode_duration, float(episode_duration)
            )

        super().__init__(
            time_sampling=time_sampling,
            dt=float(dt),
            physics_dt=float(physics_dt),
            min_dt=min_dt,
            max_dt=max_dt,
            max_steps=max_steps,
            episode_duration=adjusted_episode_duration,  # minutes
            time_sampling_kwargs=time_sampling_kwargs,
            return_reward_increment=return_reward_increment,
        )

        if seed is not None:
            self._np_random, _ = gym.utils.seeding.np_random(seed)

        # Interpret "dt_default" as 10 minutes so 1 minutes is like dt ~ 0.1
        self.dt_default = 10.0

        # Action = buy/sell 1 ticker per group
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_groups,), dtype=np.float32
        )

        # obs = G*23 (price features) + 5*G (past 5 actions) + 5 (past 5 rewards)
        # + G (position value fraction) + 1 (capital fraction)
        obs_dim = self.n_groups * 23 + 5 * self.n_groups + 5 + self.n_groups + 1

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

    # --------------------------- Evaluation specification if quarters specified ---------------------------

    def _build_eval_spec_indices_from_range(
        self,
        range_str: str,
        stride_days: int,
    ) -> List[int]:
        """Build a deterministic list of episode spec indices."""
        if stride_days <= 0:
            raise ValueError(f"stride_days must be > 0, got {stride_days}")

        out: List[int] = []
        stride_points = stride_days * SESSION_MIN_PER_DAY

        q_start, q_end = _parse_eval_range_ny(range_str)

        # Candidate specs fully inside range:
        candidates: List[int] = []
        for i, spec in enumerate(self._episode_specs):
            # spec.start_date_ny and spec.end_date_ny are 'YYYY-MM-DD' strings
            if spec.start_date_ny >= q_start and spec.end_date_ny < q_end:
                candidates.append(i)

        # Greedy pick to enforce stride in index-space (minute index)
        last_kept_start = None
        for i in candidates:
            s = self._episode_specs[i].start_idx
            if last_kept_start is None or (s - last_kept_start) >= stride_points:
                out.append(i)
                last_kept_start = s

        return out

    # --------------------------- ContinuousEnv abstract method implementations ---------------------------

    def _reset_physics(self, *, seed=None, options=None):
        rng = (
            self._np_random
            if self._np_random is not None
            else np.random.default_rng(seed)
        )

        self._t_idx = 0
        self.capital = float(self.init_capital)
        self.positions_shares[:] = 0.0
        self.past_actions[:] = 0.0
        self.past_rewards[:] = 0.0
        self._trade_log = [[], [], [], []]

        # Choose a shared episode window for all groups
        force_episode_index = None
        if options is not None:
            force_episode_index = options.get("force_episode_index", None)

        if force_episode_index is not None:
            ep_i = int(force_episode_index) % len(self._episode_specs)
        elif self._eval_spec_indices is not None:
            ep_i = self._eval_spec_indices[self._eval_spec_ptr]
        else:
            ep_i = int(self._np_random.integers(0, len(self._episode_specs)))
        spec = self._episode_specs[ep_i]

        self._chosen_spec = spec

        # Precompute shared timestamp slice once
        ts_ep = self._ts_utc[spec.start_idx : spec.end_idx_exclusive].to_numpy()

        # Reset episode state/logs
        self.capital = float(self.init_capital)
        self.positions_shares[:] = 0.0
        self.past_actions[:] = 0.0
        self.past_rewards[:] = 0.0
        self._trade_log = [[] for _ in range(self.n_groups)]

        # For each group, pick a ticker, but reuse the same spec window
        for group_idx, group_name in enumerate(self.groups):
            tickers_in_group = GROUPS[group_name]
            if self._eval_spec_indices is not None and self.eval_cycle_tickers:
                # Cycle tickers deterministically
                ticker_k = self._eval_ticker_ptr % len(tickers_in_group)
                ticker = tickers_in_group[ticker_k]
            else:
                # Random ticker in normal training
                ticker = str(self._np_random.choice(tickers_in_group))

            self._chosen_tickers[group_idx] = ticker
            self._chosen_specs[group_idx] = spec

            # Find ticker index in the preprocessed array and slice the same window
            array_idx = int(self._ticker_to_array_idx[ticker])
            self.X[group_idx, :, :] = self.features_all[
                array_idx, spec.start_idx : spec.end_idx_exclusive, :
            ].astype(np.float32)

            # All groups share the same episode dates
            self._ts_ep_utc[group_idx] = ts_ep

        # Advance deterministic schedule
        if self._eval_spec_indices is not None:
            # Move to next spec for the current ticker
            self._eval_spec_ptr += 1
            if self._eval_spec_ptr >= len(self._eval_spec_indices):
                # If all specs for this ticker finished, move to next ticker
                self._eval_spec_ptr = 0
                if self.eval_cycle_tickers:
                    self._eval_ticker_ptr += 1
                    if self._eval_ticker_ptr >= self._eval_ticker_cycle_len:
                        self._eval_ticker_ptr = 0

        obs = self._make_obs(self._t_idx)
        info = {
            "capital": self.capital,
            "chosen": [
                {
                    "group": self.groups[i],
                    "ticker": self._chosen_tickers[i],
                    "start_date_ny": self._chosen_specs[i].start_date_ny,
                    "end_date_ny": self._chosen_specs[i].end_date_ny,
                }
                for i in range(self.n_groups)
            ],
        }

        # Reset rendering if needed
        if self._fig is not None and (self.render_mode in ("human", "rgb_array")):
            self._reset_render_episode()

        return obs, info

    def _step_physics(self, action, dt):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)

        if action.shape != (self.n_groups,):
            raise ValueError(
                f"Action must have shape ({self.n_groups},), got {action.shape}"
            )

        # Quantize dt (minutes) to integer minute steps
        dt_steps_req = int(max(1, int(np.round(float(dt)))))

        # Already at end
        if self._t_idx >= self.episode_points - 1:
            next_obs = self._make_obs(self._t_idx)
            return next_obs, 0.0, True, False, {"at_end": True}, 0.0

        next_idx = min(self._t_idx + dt_steps_req, self.episode_points - 1)
        dt_used = float(next_idx - self._t_idx)
        if dt_used <= 0:
            next_obs = self._make_obs(self._t_idx)
            return next_obs, 0.0, True, False, {"at_end": True}, 0.0

        # Prices (PRICE_SCALE units) at current and next time step
        current_price = self.X[:, self._t_idx, 0].astype(np.float32)
        next_price = self.X[:, next_idx, 0].astype(np.float32)
        price_delta = (next_price - current_price).astype(np.float32)

        # Convert action to # of shares to be traded using fraction-of-capital rule
        # trade_notional_i = action_i * max_trade_fraction * capital
        capital = float(self.capital)
        trade_notional = (action * (self.max_trade_fraction * capital)).astype(
            np.float32
        )  # PRICE SCALE units
        safe_current_price = np.maximum(current_price, 1e-3)
        shares_delta = (trade_notional / safe_current_price).astype(np.float32)

        # Apply position with limit: |position_value| <= position_limit_fraction * init_capital
        new_position = (self.positions_shares + shares_delta).astype(np.float32)
        position_value = new_position * safe_current_price
        limit_value = float(self.position_limit_fraction * self.init_capital)
        position_value = np.clip(position_value, -limit_value, limit_value)
        new_position = (position_value / safe_current_price).astype(np.float32)

        # Actual executed shares delta (after limit)
        actual_shares_delta = (new_position - self.positions_shares).astype(np.float32)
        self.positions_shares = new_position

        # Transaction costs proportional to traded notional
        traded_notional_abs = np.abs(actual_shares_delta) * safe_current_price
        cost = float(self.transaction_cost) * float(np.sum(traded_notional_abs))

        # PnL: hold new positions over the dt interval
        PnL = float(np.sum(self.positions_shares * price_delta)) - cost
        self.capital = float(self.capital + PnL)

        # Reward is set to increment PnL
        reward = float(PnL)

        # Update history (store raw actions, normalized rewards)
        self.past_actions = np.roll(self.past_actions, shift=-1, axis=0)
        self.past_actions[-1] = action
        self.past_rewards = np.roll(self.past_rewards, shift=-1)
        self.past_rewards[-1] = reward

        # Trade log (per group) for rendering
        for i in range(self.n_groups):
            if actual_shares_delta[i] != 0.0:
                self._trade_log[i].append(
                    (
                        self._t_idx,
                        float(actual_shares_delta[i]),
                        float(current_price[i]),
                    )
                )

        # Advance index
        self._t_idx = int(next_idx)

        terminated = (self._t_idx >= self.episode_points - 1) or (self.capital <= 0.0)
        truncated = False

        next_obs = self._make_obs(self._t_idx)
        info = {
            "capital": float(self.capital),
            "PnL": float(PnL),
            "cost": float(cost),
            "dt_minutes_used": float(dt_used),
            "t_index": int(self._t_idx),
            "tickers": list(self._chosen_tickers),
            "current_price": current_price.astype(np.float32),
            "next_price": next_price.astype(np.float32),
        }

        return next_obs, reward, terminated, truncated, info, float(dt_used)

    def _make_obs(self, t_idx: int) -> np.ndarray:
        # 23*G
        price_features = self.X[:, t_idx, :].reshape(-1).astype(np.float32)

        # G + 1
        current_price = self.X[:, t_idx, 0].astype(np.float32)
        position_value_fraction = (self.positions_shares * current_price) / float(
            self.init_capital
        )
        capital_fraction = np.array(
            [float(self.capital / self.init_capital)], dtype=np.float32
        )

        # 5*G + 5
        past_actions = self.past_actions.reshape(-1).astype(np.float32)
        past_rewards = self.past_rewards.astype(np.float32)

        # Total dimension: 29 * G + 6
        obs = np.concatenate(
            [
                price_features,
                past_actions,
                past_rewards,
                position_value_fraction.astype(np.float32),
                capital_fraction,
            ],
            dtype=np.float32,
        )
        return obs

    def render(self, mode: Optional[str] = None):
        mode = mode or self.render_mode or "human"
        if mode not in ("human", "rgb_array"):
            return None

        if self._fig is None:
            self._init_render()

        # Update cursor + trade sticks + text
        last_trade_summaries = []
        for i in range(self.n_groups):
            ax = self._axs[i]
            cur_x = int(self._t_idx)

            # Cursor line
            self._cursor_lines[i].set_xdata([cur_x, cur_x])

            # Build vertical stick for buy/sell
            trades = self._trade_log[i]
            if len(trades) == 0:
                self._buy_sticks[i].set_segments([])
                self._sell_sticks[i].set_segments([])
                last_trade = None
            else:
                ymin, ymax = self._y_lims[i]
                y_range = max(1e-6, float(ymax - ymin))
                stick_max_len = 0.25 * y_range  # 25% of the range
                max_shares_plot = float(self._max_shares_plot[i])

                buy_segments = []
                sell_segments = []
                last_trade = trades[-1]

                for t_idx, shares_delta, price in trades:
                    t_idx = int(t_idx)
                    shares_delta = float(shares_delta)
                    price_dollars = float(price) * float(PRICE_SCALE)

                    # Calculate stick length
                    frac = min(1.0, abs(shares_delta) / max(1e-6, max_shares_plot))
                    L = stick_max_len * max(frac, MIN_FRAC_PLOT)

                    # Anchor stick at trade price, extend up for buy and down for sell
                    if shares_delta > 0:
                        buy_segments.append(
                            [(t_idx, price_dollars), (t_idx, price_dollars + L)]
                        )
                    elif shares_delta < 0:
                        sell_segments.append(
                            [(t_idx, price_dollars), (t_idx, price_dollars - L)]
                        )

                self._buy_sticks[i].set_segments(buy_segments)
                self._sell_sticks[i].set_segments(sell_segments)

            # Per-panel text (position + last trade)
            current_price = float(self.X[i, self._t_idx, 0] * float(PRICE_SCALE))
            num_shares = float(self.positions_shares[i])

            if last_trade is None:
                last_str = "___"
                last_summary = f"{self.groups[i]}: ___"
            else:
                last_t, last_shares_delta, last_price = last_trade
                last_shares_delta = float(last_shares_delta)
                last_price = float(last_price) * PRICE_SCALE
                side = "BUY" if last_shares_delta > 0 else "SELL"
                last_shares_quantity = abs(last_shares_delta)
                last_str = (
                    f"{side} {last_shares_quantity:.2f} share at ${last_price:.2f}"
                )
                last_summary = f"{self.groups[i]}: {side} {last_shares_quantity:.2f} shares at ${last_price:.2f}"
            txt = (
                f"shares={num_shares:.2f}; price=${current_price:.2f}\n{last_str}"
            ).replace("$", r"\$")
            self._info_titles[i].set_text(txt)
            last_trade_summaries.append(last_summary.replace("$", r"\$"))

            # Keep view stable and clean
            ax.set_xlim(*self._x_lim)
            ax.set_ylim(*self._y_lims[i])

        # Global title + info strip
        PnL_total = float(self.capital - self.init_capital)
        PnL_total_dollar = PnL_total * float(PRICE_SCALE)
        capital_dollar = self.capital * float(PRICE_SCALE)

        suptitle_txt = (
            f"Capital=${capital_dollar:,.0f}    |   PnL=${PnL_total_dollar:,.0f}"
            + f"    |   t={self._t_idx}/{self.episode_points-1}"
        ).replace("$", r"\$")
        self._fig.suptitle(suptitle_txt, fontsize=12)

        self._global_info_text.set_text(" | ".join(last_trade_summaries))

        self._fig.canvas.draw()

        if mode == "rgb_array":
            buf = np.asarray(self._fig.canvas.buffer_rgba())
            img = np.asarray(buf[:, :, :3], dtype=np.uint8).copy()
            return img

        plt.pause(0.01)
        return None

    def _panel_title_left(self, i: int) -> str:
        ticker = self._chosen_tickers[i] if self._chosen_tickers[i] else "?"
        group = self.groups[i]
        spec = self._chosen_specs[i]
        return f"{group}: {ticker_label(ticker)}\n{spec.start_date_ny} → {spec.end_date_ny}"

    def _episode_price_xy(self, i: int):
        x = np.arange(self.episode_points, dtype=np.int32)
        price = (self.X[i, :, 0] * float(PRICE_SCALE)).astype(np.float32)
        return x, price

    def _compute_y_lims(self, price: np.ndarray):
        pmin = float(np.nanmin(price))
        pmax = float(np.nanmax(price))
        margin = 0.06 * max(1e-6, (pmax - pmin))
        return pmin - margin, pmax + margin

    def _apply_episode_to_axes(
        self,
        i: int,
        *,
        create_artists: bool,
        LineCollection,
    ):
        """
        Common logic used by both _init_render and _reset_render_episode.
        - create_artists=True: create lines/sticks/titles and append to lists
        - create_artists=False: update existing artists in-place
        """
        ax = self._axs[i]
        self._x_lim = (0, int(self.episode_points - 1))

        x, price = self._episode_price_xy(i)
        ymin, ymax = self._compute_y_lims(price)

        if create_artists:
            # Price line
            (ln,) = ax.plot(x, price, linewidth=1.5, alpha=0.9)
            self._price_lines.append(ln)

            # Cursor
            cur_ln = ax.axvline(self._t_idx, linestyle="--", linewidth=1.0, alpha=0.8)
            self._cursor_lines.append(cur_ln)

            # Stick collections
            buy_lc = LineCollection([], linewidths=1.2, alpha=0.95)
            buy_lc.set_color("#2ca02c")
            buy_lc.set_zorder(5)
            ax.add_collection(buy_lc)
            self._buy_sticks.append(buy_lc)

            sell_lc = LineCollection([], linewidths=1.2, alpha=0.95)
            sell_lc.set_color("#d62728")
            sell_lc.set_zorder(5)
            ax.add_collection(sell_lc)
            self._sell_sticks.append(sell_lc)

            # Titles (left = static, right = live info)
            ax.set_title(self._panel_title_left(i), loc="left", pad=10)
            info_txt = ax.set_title("", loc="right", pad=10)
            self._info_titles.append(info_txt)

            # Labels / cosmetics
            ax.set_xlabel("minutes")
            ax.set_ylabel("price ($)")
            ax.grid(True, alpha=0.25)
        else:
            # Update existing artists
            self._price_lines[i].set_data(x, price)
            self._cursor_lines[i].set_xdata([0, 0])
            self._buy_sticks[i].set_segments([])
            self._sell_sticks[i].set_segments([])
            if self._info_titles[i] is not None:
                self._info_titles[i].set_text("")
            ax.set_title(self._panel_title_left(i), loc="left", pad=10)

        # Apply stable limits
        if create_artists:
            self._y_lims.append((ymin, ymax))
        else:
            self._y_lims[i] = (ymin, ymax)

        ax.set_xlim(*self._x_lim)
        ax.set_ylim(ymin, ymax)

        # For trade stick plotting, also calculate max shares purchasable
        max_notional_plot = float(self.max_trade_fraction * self.init_capital)
        min_price_units = float(np.nanmin(self.X[i, :, 0]))
        max_shares_plot = max_notional_plot / max(1e-6, min_price_units)

        if create_artists:
            if not hasattr(self, "_max_shares_plot") or self._max_shares_plot is None:
                self._max_shares_plot = []
            self._max_shares_plot.append(max_shares_plot)
        else:
            self._max_shares_plot[i] = max_shares_plot

    def _init_render(self):
        plt.style.use("seaborn-v0_8-darkgrid")

        # Create panels
        ncols = int(math.ceil(math.sqrt(self.n_groups)))
        nrows = int(math.ceil(self.n_groups / ncols))
        self._fig, axs = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows))
        axs = np.array(axs).reshape(-1)
        self._axs = axs.tolist()

        # Hide unused axes
        for j in range(self.n_groups, len(self._axs)):
            self._axs[j].set_visible(False)

        self._price_lines = []
        self._cursor_lines = []
        self._buy_sticks = []
        self._sell_sticks = []
        self._info_titles = []

        self._x_lim = (0, int(self.episode_points - 1))
        self._y_lims = []

        # Legend handles
        buy_handle = Line2D([0], [0], color="#2ca02c", lw=2.0, label="Buy (up stick)")
        sell_handle = Line2D(
            [0], [0], color="#d62728", lw=2.0, label="Sell (down stick)"
        )
        price_handle = Line2D([0], [0], color="black", lw=1.5, alpha=0.9, label="Price")

        for i in range(self.n_groups):
            self._apply_episode_to_axes(
                i, create_artists=True, LineCollection=LineCollection
            )

        self._legend = self._fig.legend(
            handles=[price_handle, buy_handle, sell_handle],
            loc="upper center",
            ncol=3,
            framealpha=0.2,
            fontsize=9,
            bbox_to_anchor=(0.5, 0.94),
        )

        self._global_info_text = self._fig.text(
            0.5, 0.02, "", ha="center", va="bottom", fontsize=10, alpha=0.9
        )

        self._fig.tight_layout(rect=[0, 0.06, 1, 0.86])

    def _reset_render_episode(self):
        """
        Refresh all render artists to reflect the newly sampled episode (self.X, tickers, specs).
        Only call if self._fig is not None.
        """

        self._x_lim = (0, int(self.episode_points - 1))

        for i in range(self.n_groups):
            self._apply_episode_to_axes(
                i, create_artists=False, LineCollection=LineCollection
            )

        if self._global_info_text is not None:
            self._global_info_text.set_text("")

        try:
            self._fig.tight_layout(rect=[0, 0.06, 1, 0.86])
        except Exception:
            pass

        self._fig.canvas.draw_idle()

    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._axs = None
            self._price_lines = None
            self._cursor_lines = None
            self._buy_sticks = None
            self._sell_sticks = None
            self._info_titles = None
            self._global_info_text = None
            self._x_lim = None
            self._y_lims = None

        super().close()
