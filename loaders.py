
from __future__ import annotations

import glob
import logging
import os

import pandas as pd

from src.config import IsoConfig
from src.leakage import da_gate_cutoff_utc

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
log = logging.getLogger(__name__)


def _files(iso: str, kind: str) -> list[str]:
    return sorted(glob.glob(os.path.join(DATA_DIR, iso, kind, "*.parquet")))


def _read_all(iso: str, kind: str, columns: list[str] | None = None) -> pd.DataFrame:
    fs = _files(iso, kind)
    if not fs:
        raise FileNotFoundError(f"no {kind} files for {iso} under {DATA_DIR}/{iso}/{kind}")
    return pd.concat([pd.read_parquet(f, columns=columns) for f in fs], ignore_index=True)


def load_lmp(cfg: IsoConfig) -> pd.DataFrame:
    
    cols = ["Interval Start", "Location", cfg.lmp_column]
    comp = cfg.lmp_components
    for c in (comp.energy, comp.congestion, comp.loss):
        if c is not None:
            cols.append(c)
    df = _read_all(cfg.name, "lmp", columns=list(dict.fromkeys(cols)))

    n_before = len(df)
    df = df[df["Location"].astype(str).isin(cfg.locations)].reset_index(drop=True)

                                                                            
    out = pd.DataFrame({
        "location": df["Location"].astype(str),
        "interval_start_utc": pd.to_datetime(df["Interval Start"], utc=True),
        "lmp": df[cfg.lmp_column].astype("float32"),
    })
    import numpy as np
    out["energy"] = df[comp.energy].astype("float32") if comp.energy else np.float32(np.nan)
    out["congestion"] = df[comp.congestion].astype("float32") if comp.congestion else np.float32(np.nan)
    out["loss"] = df[comp.loss].astype("float32") if comp.loss else np.float32(np.nan)

    dup = out.duplicated(subset=["location", "interval_start_utc"]).sum()
    if dup:
        log.warning("%s lmp: dropping %d duplicate (location, interval_start_utc) rows", cfg.name, dup)
        out = out.drop_duplicates(subset=["location", "interval_start_utc"], keep="last")

    log.info("%s lmp: %d rows -> %d after location filter to %d locations (dropped %d dup)",
              cfg.name, n_before, len(out), len(cfg.locations), dup)
    return out.sort_values(["location", "interval_start_utc"]).reset_index(drop=True)


def load_load_forecast(cfg: IsoConfig) -> pd.DataFrame | None:
    
    lf = cfg.load_forecast
    if lf.mode == "none":
        return None

                                                                            
    sample_cols = pd.read_parquet(_files(cfg.name, "load_forecast")[0]).columns
    publish_col = "Publish Time" if "Publish Time" in sample_cols else "Forecast Time"
    cols = ["Interval Start", publish_col, lf.column]
    df = _read_all(cfg.name, "load_forecast", columns=cols)

    out = pd.DataFrame({
        "interval_start_utc": pd.to_datetime(df["Interval Start"], utc=True),
        "load_forecast": df[lf.column].astype("float32"),
        "publish_time_utc": pd.to_datetime(df[publish_col], utc=True),
    })

    n_before = len(out)
    dup = out.duplicated().sum()
    if dup:
        log.warning("%s load_forecast: dropping %d fully-duplicate rows", cfg.name, dup)
        out = out.drop_duplicates()

                                                                                   
    cutoff = da_gate_cutoff_utc(out["interval_start_utc"], cfg)
    n_after_gate_only = (out["publish_time_utc"] <= cutoff).sum()
    out = out.loc[out["publish_time_utc"] <= cutoff]
    dup_ts = out.duplicated(subset=["interval_start_utc"]).sum()
    if dup_ts:
        out = out.sort_values(["interval_start_utc", "publish_time_utc"])
        out = out.drop_duplicates(subset=["interval_start_utc"], keep="last")

    log.info("%s load_forecast: %d rows -> %d exact-dup-free -> %d leak-free "
              "(gate=%.1fh local D-1, %d had a usable pre-gate issue, %d re-forecast dup collapsed)",
              cfg.name, n_before, n_before - dup, n_after_gate_only, cfg.gate_local_hour,
              n_after_gate_only, dup_ts)
    return out.sort_values("interval_start_utc").reset_index(drop=True)


GAS_CSV = os.path.join(os.path.dirname(DATA_DIR), "..", "gas_henry_hub_daily.csv")
GAS_CSV = os.path.normpath(GAS_CSV)


def load_gas_price(lag_days: int = 2, csv_path: str = GAS_CSV) -> pd.DataFrame:
    




    raw = pd.read_csv(csv_path, parse_dates=["date"])
    raw = raw.sort_values("date").drop_duplicates(subset=["date"])

    full_days = pd.date_range(raw["date"].min(), raw["date"].max(), freq="D")
    daily = raw.set_index("date")["price"].reindex(full_days).ffill()
    daily.index.name = "date"

    lagged = daily.shift(lag_days)

    hourly_index = pd.date_range(lagged.index.min(), lagged.index.max() + pd.Timedelta(hours=23),
                                  freq="h", tz="UTC")
    hourly_day = hourly_index.floor("D").tz_localize(None)
    gas_price = lagged.reindex(hourly_day).to_numpy(dtype="float32")

    out = pd.DataFrame({
        "interval_start_utc": hourly_index,
        "gas_price": gas_price,
    })
    out["gas_price_available"] = out["gas_price"].notna()
    log.info("gas_price: %d daily rows (%s -> %s) -> %d hourly rows, %d-day lag, %d available",
              len(raw), raw["date"].min().date(), raw["date"].max().date(), len(out), lag_days,
              int(out["gas_price_available"].sum()))
    return out


def load_miso_zonal(cfg: IsoConfig) -> pd.DataFrame:
    
    zonal_cols = list(cfg.load_forecast.zonal_columns)
    cols = ["Interval Start", "Publish Time"] + zonal_cols
    df = _read_all(cfg.name, "load_forecast", columns=cols)
    out = df.rename(columns={"Interval Start": "interval_start_local", "Publish Time": "publish_time_local"})
    out["interval_start_utc"] = pd.to_datetime(out["interval_start_local"], utc=True)
    for c in zonal_cols:
        out[c] = out[c].astype("float32")
    return out.drop_duplicates(subset=["interval_start_utc"]).sort_values("interval_start_utc").reset_index(drop=True)
