
from __future__ import annotations

import logging

import pandas as pd

from src.config import IsoConfig, ProjectConfig
from src.loaders import load_gas_price, load_lmp, load_load_forecast, load_miso_zonal

log = logging.getLogger(__name__)

PANEL_COLUMNS = [
    "iso", "location", "interval_start_utc",
    "lmp", "energy", "congestion", "loss",
    "load_forecast", "load_forecast_available",
]


def build_iso_panel(cfg: IsoConfig) -> pd.DataFrame:
    lmp = load_lmp(cfg)
    lf = load_load_forecast(cfg)

    if lf is None:
        df = lmp.copy()
        df["load_forecast"] = float("nan")
        df["load_forecast_available"] = False
    else:
                                                                       
        df = lmp.merge(lf[["interval_start_utc", "load_forecast"]], on="interval_start_utc", how="left")
        df["load_forecast_available"] = df["load_forecast"].notna()

    df["iso"] = cfg.name
    df["lmp"] = df["lmp"].astype("float32")
    for c in ("energy", "congestion", "loss", "load_forecast"):
        df[c] = df[c].astype("float32")
    return df[PANEL_COLUMNS]


def check_hourly_index(df: pd.DataFrame) -> pd.DataFrame:
    


    rows = []
    for (iso, loc), g in df.groupby(["iso", "location"]):
        t = g["interval_start_utc"].sort_values()
        n_dup = t.duplicated().sum()
        t_unique = t.drop_duplicates()
        if len(t_unique) > 1:
            full_range = pd.date_range(t_unique.min(), t_unique.max(), freq="h", tz="UTC")
            n_missing = len(full_range) - len(t_unique)
        else:
            n_missing = 0
        is_increasing = t.is_monotonic_increasing
        if n_dup or n_missing or not is_increasing:
            rows.append({"iso": iso, "location": loc, "n_rows": len(t),
                         "n_duplicate_ts": int(n_dup), "n_missing_hours": int(n_missing),
                         "strictly_increasing_after_sort": bool(is_increasing)})
    return pd.DataFrame(rows)


def build_panel(cfg: ProjectConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    
    frames = []
    for iso_cfg in cfg.isos:
        log.info("building panel rows for %s", iso_cfg.name)
        frames.append(build_iso_panel(iso_cfg))
    panel = pd.concat(frames, ignore_index=True)
    panel["iso"] = panel["iso"].astype("category")
    panel["location"] = panel["location"].astype("category")

    violations = check_hourly_index(panel)
    if len(violations):
        log.warning("hourly-index violations found for %d (iso, location) pairs -- see report, "
                     "NOT auto-fixed", len(violations))

    miso_cfg = cfg.iso("MISO")
    miso_zonal = load_miso_zonal(miso_cfg)

    return panel, miso_zonal, violations


def attach_gas_price(panel: pd.DataFrame, cfg: ProjectConfig) -> pd.DataFrame:
    




    if cfg.gas_price.mode == "none":
        raise ValueError("gas_price.mode is 'none' in config -- nothing to attach")
    gas = load_gas_price(lag_days=cfg.gas_price.lag_days)
    out = panel.merge(gas, on="interval_start_utc", how="left")
    out["gas_price"] = out["gas_price"].astype("float32")
    out["gas_price_available"] = out["gas_price_available"].infer_objects(copy=False).fillna(False).astype(bool)
    return out
