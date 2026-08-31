
from __future__ import annotations

import pandas as pd

from src.config import SplitsConfig


def split_name(interval_start_utc: pd.Series, splits: SplitsConfig) -> pd.Series:
    
    train_end = pd.Timestamp(splits.train_end, tz="UTC") + pd.Timedelta(days=1)                               
    val_end = pd.Timestamp(splits.val_end, tz="UTC") + pd.Timedelta(days=1)
    out = pd.Series("test", index=interval_start_utc.index, dtype="object")
    out[interval_start_utc < val_end] = "val"
    out[interval_start_utc < train_end] = "train"
    return out.astype("category")


def filter_split(df: pd.DataFrame, split: str, splits: SplitsConfig, time_col: str = "interval_start_utc") -> pd.DataFrame:
    if split not in ("train", "val", "test"):
        raise ValueError(f"split must be train/val/test, got {split!r}")
    return df[split_name(df[time_col], splits) == split]
