
from __future__ import annotations

import os

import pandas as pd

from src.config import IsoConfig, ProjectConfig, load_config
from src.leakage import da_gate_cutoff_utc

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_PATH = os.path.join(PROJECT_DIR, "data", "processed", "panel.parquet")

CONTEXT_HOURS = 336           

                                                                                           
                                                                       
MIN_HISTORY_DATE = "2021-01-01"

                                                                                                 
                                                                                                
                                                                            
TEST_YEAR = 2025


def load_iso_panel(iso: str, panel_path: str | None = None) -> pd.DataFrame:
    
    panel = pd.read_parquet(panel_path or PANEL_PATH)
    sub = panel[panel["iso"] == iso].copy()
    sub["location"] = sub["location"].astype(str)
    return sub.sort_values(["location", "interval_start_utc"]).reset_index(drop=True)


def zone_series(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    

    out = {}
    for z, g in panel.groupby("location"):
        g = g.set_index("interval_start_utc").sort_index()
        out[z] = g[["lmp", "load_forecast", "load_forecast_available"]]
    return out


def delivery_days(panel: pd.DataFrame, split: str, iso_cfg: IsoConfig,
                   cfg: ProjectConfig | None = None) -> list[pd.Timestamp]:
    



    cfg = cfg or load_config()
    local_dates = panel["interval_start_utc"].dt.tz_convert(iso_cfg.tz).dt.normalize()
    train_end = pd.Timestamp(cfg.splits.train_end, tz=iso_cfg.tz)
    val_end = pd.Timestamp(cfg.splits.val_end, tz=iso_cfg.tz)
    if split == "train":
        mask = local_dates <= train_end
    elif split == "val":
        mask = (local_dates > train_end) & (local_dates <= val_end)
    elif split == "test":
        mask = local_dates > val_end
    else:
        raise ValueError(f"split must be train/val/test, got {split!r}")
    dates = sorted(local_dates[mask].unique())
    return list(pd.to_datetime(dates))


def origins_for_days(days: list[pd.Timestamp], iso_cfg: IsoConfig
                      ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    
    out = []
    for d in days:
        day_start_utc = d.tz_convert("UTC")
        day_end_utc = (d + pd.DateOffset(days=1)).tz_convert("UTC")
        gate = da_gate_cutoff_utc(pd.Series([day_start_utc]), iso_cfg).iloc[0]
        out.append((gate, day_start_utc, day_end_utc))
    return out


def origins_for_split(panel: pd.DataFrame, split: str, iso_cfg: IsoConfig,
                       cfg: ProjectConfig | None = None, year: int | None = None
                       ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    
    days = delivery_days(panel, split, iso_cfg, cfg=cfg)
    if year is not None:
        days = [d for d in days if d.year == year]
    return origins_for_days(days, iso_cfg)


def days_for_data_budget(panel: pd.DataFrame, iso_cfg: IsoConfig, n_days: int | None,
                         cfg: ProjectConfig | None = None) -> list[pd.Timestamp]:
    







    cfg = cfg or load_config()
    end = pd.Timestamp(cfg.splits.val_end, tz=iso_cfg.tz)
    min_date = pd.Timestamp(MIN_HISTORY_DATE, tz=iso_cfg.tz)
    local_dates = panel["interval_start_utc"].dt.tz_convert(iso_cfg.tz).dt.normalize()
    history = sorted(local_dates[(local_dates <= end) & (local_dates >= min_date)].unique())
    history = list(pd.to_datetime(history))
    return history if n_days is None else history[-n_days:]


def origins_for_budget(panel: pd.DataFrame, iso_cfg: IsoConfig, budget: str,
                        cfg: ProjectConfig | None = None
                        ) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    





    cfg = cfg or load_config()
    end = pd.Timestamp(cfg.splits.train_end if budget == "train" else cfg.splits.val_end,
                       tz=iso_cfg.tz)
    min_date = pd.Timestamp(MIN_HISTORY_DATE, tz=iso_cfg.tz)
    local_dates = panel["interval_start_utc"].dt.tz_convert(iso_cfg.tz).dt.normalize()
    history = sorted(local_dates[(local_dates <= end) & (local_dates >= min_date)].unique())
    history = list(pd.to_datetime(history))
    if budget == "0":
        return []
    if budget in ("full", "train"):
        days = history
    else:
        n_days = {"24h": 1, "48h": 2, "1wk": 7, "1mo": 30, "6mo": 182, "2yr": 730}[budget]
        days = history[-n_days:]
    return origins_for_days(days, iso_cfg)
