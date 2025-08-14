"""
| Symbol  | ADTV (shares) | Bucket Count (n) | Bucket Size (V = ADTV/n) |
| ------- | ------------- | ---------------- | ------------------------ |
| **SPY** | 72 M          | 25               | 2,880,000 shares         |
|         |               | 50 (default)     | 1,440,000 shares         |
|         |               | 100              | 720,000 shares           |
|         |               | 200              | 360,000 shares           |
| **GME** | 14 M          | 10               | 1,400,000 shares         |
|         |               | 25               | 560,000 shares           |
|         |               | 50               | 280,000 shares           |
|         |               | 100              | 140,000 shares           |







Minimal, production-style VPIN implementation (recursive, EWMA volatility for BVC) — NumPy/SciPy/Pandas edition.

Files in this single-module package-style script:
- vpin.py (this file): core classes + batch helper

Dependencies: numpy (required), pandas (optional for batch helper), scipy (for Norm CDF)

Usage (streaming):
    from vpin import VpinConfig, VpinOnline
    cfg = VpinConfig()
    vpin = VpinOnline(cfg)
    for t, p, v in stream:  # time, price, volume
        value = vpin.update(t, p, v)  # returns current VPIN (float) or None before first price

Usage (batch with pandas DataFrame having columns ['time','price','volume']):
    from vpin import VpinConfig, compute_vpin_series
    cfg = VpinConfig(bucket_max_volume=100_000, n_buckets=20, vol_decay=0.8)
    s = compute_vpin_series(df, cfg)  # pandas.Series aligned to df.index
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
import pandas as pd  # pandas now required for batch helper; remove if undesired
from scipy.stats import norm


# -----------------------------
# Configuration
# -----------------------------

@dataclass(frozen=True)
class VpinConfig:
    """Configuration for recursive VPIN.

    Attributes
    ----------
    bucket_max_volume : int
        Target volume per bucket V (equal-volume buckets).
    n_buckets : int
        Number of buckets kept in the circular window for VPIN aggregation.
    vol_decay : float
        EWMA decay lambda for variance update in [0, 1). Higher → smoother.
    return_mode : str
        One of {"pnl", "simple", "log"}. Controls how r_t is computed for EWMA.
    eps_vol : float
        Small floor to avoid division by zero in normalization.
    clip_z : Optional[float]
        If set, clip z-score to [-clip_z, clip_z] to avoid extreme CDF saturation.
    time_col, price_col, volume_col : str
        Column names for batch helper.
    """
    bucket_max_volume: int = 100_000
    n_buckets: int = 20
    vol_decay: float = 0.8  # lambda in EWMA
    return_mode: str = "pnl"
    eps_vol: float = 1e-8
    clip_z: Optional[float] = 8.0
    time_col: str = "time"
    price_col: str = "price"
    volume_col: str = "volume"


# -----------------------------
# Core online VPIN engine
# -----------------------------

class VpinOnline:
    """Online VPIN calculator with EWMA volatility for bulk volume classification.

    The algorithm maintains:
      - last price and EWMA variance (sigma^2)
      - circular buffers for bucket buy, sell, and volume (length n_buckets)
      - current bucket index and fill state

    Each update(time, price, volume) does:
      1) Update EWMA variance from chosen return definition.
      2) Compute buy_frac = Phi( (p_t - p_{t-1}) / sigma_t ).
      3) Pour 'volume' into equal-volume buckets with spillover, splitting by buy_frac.
      4) Compute VPIN = sum(|B - S|) / sum(V) over non-empty buckets.
    """

    __slots__ = (
        "cfg",
        "_last_price",
        "_sigma2",
        "_bucket_buy",
        "_bucket_sell",
        "_bucket_vol",
        "_idx",
    )

    def __init__(self, cfg: VpinConfig) -> None:
        self.cfg = cfg
        self._last_price: Optional[float] = None
        self._sigma2: Optional[float] = None
        self._bucket_buy = np.zeros(cfg.n_buckets, dtype=np.float64)
        self._bucket_sell = np.zeros(cfg.n_buckets, dtype=np.float64)
        self._bucket_vol = np.zeros(cfg.n_buckets, dtype=np.float64)
        self._idx: int = 0  # current bucket index

    # -------- EWMA variance update --------
    def _ret(self, p_t: float, p_tm1: float) -> float:
        mode = self.cfg.return_mode
        if mode == "pnl":
            return p_t - p_tm1
        if mode == "simple":
            return 0.0 if p_tm1 == 0 else (p_t / p_tm1) - 1.0
        if mode == "log":
            # Guard non-positive prices for log returns
            return 0.0 if (p_t <= 0 or p_tm1 <= 0) else float(np.log(p_t / p_tm1))
        raise ValueError(f"Unknown return_mode: {mode}")

    def _update_sigma2(self, p_tm1: float, p_t: float) -> float:
        """Update and return the EWMA variance (sigma^2) using return_mode.

        Returns the new variance estimate. On the first call, seeds sigma^2 with eps_vol^2.
        """
        if self._sigma2 is None:
            # Bootstrap variance with a small positive value
            self._sigma2 = max(self.cfg.eps_vol ** 2, 1e-12)
        r_t = self._ret(p_t, p_tm1)
        lam = float(self.cfg.vol_decay)
        self._sigma2 = (1.0 - lam) * (r_t * r_t) + lam * self._sigma2
        return self._sigma2

    # -------- Pour volume into buckets --------
    def _advance_bucket(self) -> None:
        self._idx = (self._idx + 1) % self.cfg.n_buckets
        self._bucket_buy[self._idx] = 0.0
        self._bucket_sell[self._idx] = 0.0
        self._bucket_vol[self._idx] = 0.0

    def _pour(self, volume: float, buy_frac: float) -> None:
        Vmax = float(self.cfg.bucket_max_volume)
        remaining = float(max(0.0, volume))
        buy_frac = float(np.clip(buy_frac, 0.0, 1.0))
        while remaining > 0.0:
            cap = Vmax - self._bucket_vol[self._idx]
            pour = remaining if remaining < cap else cap
            if pour <= 0.0:
                break
            buy = buy_frac * pour
            sell = pour - buy
            self._bucket_buy[self._idx] += buy
            self._bucket_sell[self._idx] += sell
            self._bucket_vol[self._idx] += pour
            remaining -= pour
            # Close bucket if (numerically) full
            if self._bucket_vol[self._idx] >= Vmax - 1e-12:
                self._advance_bucket()

    # -------- Public API --------
    def update(self, time: object, price: float, volume: float) -> Optional[float]:
        """Ingest one bar and return the current VPIN (or None on the very first bar).

        Parameters
        ----------
        time : any
            Carried through for alignment; unused by the engine.
        price : float
        volume : float
        """
        p_t = float(price)
        v_t = float(max(0.0, volume))

        if self._last_price is None:
            # Need two prices to start classification
            self._last_price = p_t
            return None

        p_tm1 = float(self._last_price)
        sigma2 = self._update_sigma2(p_tm1, p_t)
        sigma = float(np.sqrt(max(sigma2, self.cfg.eps_vol ** 2)))

        # Bulk classification always uses PnL delta (price change), not return_mode
        delta = p_t - p_tm1
        z = delta / sigma if sigma > 0 else (np.inf if delta > 0 else (-np.inf if delta < 0 else 0.0))
        if self.cfg.clip_z is not None:
            z = float(np.clip(z, -float(self.cfg.clip_z), float(self.cfg.clip_z)))
        buy_frac = float(norm.cdf(z))

        if v_t > 0.0:
            self._pour(v_t, buy_frac)

        total_vol = float(self._bucket_vol.sum())
        if total_vol <= 0.0:
            vpin = 0.0
        else:
            imbalance = float(np.abs(self._bucket_buy - self._bucket_sell).sum())
            vpin = imbalance / total_vol

        # advance state
        self._last_price = p_t
        return vpin

    # Convenience accessors
    @property
    def buckets_snapshot(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
        """Return copies of (buy, sell, vol, current_index) for inspection/testing."""
        return (
            self._bucket_buy.copy(),
            self._bucket_sell.copy(),
            self._bucket_vol.copy(),
            int(self._idx),
        )


# -----------------------------
# Batch helper (pandas)
# -----------------------------

def compute_vpin_series(df: pd.DataFrame, cfg: VpinConfig) -> pd.Series:
    """Compute VPIN for a whole DataFrame with columns [time, price, volume].

    The first element will be NaN because we need two prices to start classification.
    """
    time_col, price_col, volume_col = cfg.time_col, cfg.price_col, cfg.volume_col
    missing = [c for c in (time_col, price_col, volume_col) if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    engine = VpinOnline(cfg)
    out = np.full(len(df), np.nan, dtype=float)

    it = df.itertuples(index=False)
    for i, row in enumerate(it):
        t = getattr(row, time_col)
        p = float(getattr(row, price_col))
        v = float(getattr(row, volume_col))
        val = engine.update(t, p, v)
        if val is not None:
            out[i] = val

    return pd.Series(out, index=df.index, name="vpin")
