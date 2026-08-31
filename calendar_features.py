
from __future__ import annotations

import numpy as np
import pandas as pd

N_CALENDAR_FEATURES = 4                                        


def calendar_features(timestamps_utc: pd.DatetimeIndex | np.ndarray, tz: str) -> np.ndarray:
    
    idx = pd.DatetimeIndex(timestamps_utc)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    local = idx.tz_convert(tz)
    hour = local.hour.to_numpy(dtype=np.float32)
    dow = local.dayofweek.to_numpy(dtype=np.float32)            
    sin_hr = np.sin(2 * np.pi * hour / 24.0).astype(np.float32)
    cos_hr = np.cos(2 * np.pi * hour / 24.0).astype(np.float32)
    sin_dow = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
    cos_dow = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)
    return np.stack([sin_hr, cos_hr, sin_dow, cos_dow], axis=0)


N_CALENDAR_FEATURES_EXTENDED = 9                                                                              

_US_HOLIDAYS = None


def _us_holidays():
    global _US_HOLIDAYS
    if _US_HOLIDAYS is None:
        import holidays as _holidays_pkg
        _US_HOLIDAYS = _holidays_pkg.US()
    return _US_HOLIDAYS


def calendar_features_extended(timestamps_utc: pd.DatetimeIndex | np.ndarray, tz: str) -> np.ndarray:
    
    base = calendar_features(timestamps_utc, tz)            
    idx = pd.DatetimeIndex(timestamps_utc)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    local = idx.tz_convert(tz)
    us_hols = _us_holidays()
    is_holiday = np.array([1.0 if d in us_hols else 0.0 for d in local.date], dtype=np.float32)
    doy_frac = ((local.dayofyear.to_numpy() - 1) / 365.25).astype(np.float32)
    two_pi_doy = 2 * np.pi * doy_frac
    sin_doy1 = np.sin(two_pi_doy).astype(np.float32)
    cos_doy1 = np.cos(two_pi_doy).astype(np.float32)
    sin_doy2 = np.sin(2 * two_pi_doy).astype(np.float32)
    cos_doy2 = np.cos(2 * two_pi_doy).astype(np.float32)
    return np.concatenate([base, is_holiday[None, :], sin_doy1[None, :], cos_doy1[None, :],
                            sin_doy2[None, :], cos_doy2[None, :]], axis=0)
