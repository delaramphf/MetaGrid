






from __future__ import annotations

import pandas as pd

from src.config import IsoConfig


def da_gate_cutoff_utc(interval_start_utc: pd.Series, cfg: IsoConfig) -> pd.Series:
    










    ts = pd.to_datetime(interval_start_utc, utc=True)
    local_naive = ts.dt.tz_convert(cfg.tz).dt.tz_localize(None)
    gate_naive = (local_naive.dt.normalize()
                  - pd.Timedelta(days=1)
                  + pd.Timedelta(hours=float(cfg.gate_local_hour))).dt.round("s")
    gate_local = gate_naive.dt.tz_localize(cfg.tz, ambiguous=True, nonexistent="shift_forward")
    return gate_local.dt.tz_convert("UTC")
