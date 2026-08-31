






from __future__ import annotations

import os

import numpy as np
import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_PATH = os.path.join(PROJECT_DIR, "data", "processed", "panel.parquet")
OUT_PATH = os.path.join(PROJECT_DIR, "data", "processed", "panel_clipped.parquet")

ROLLING_HOURS = 240           
MIN_PERIODS = 24
N_STD = 3.0


def clip_group(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("interval_start_utc").reset_index(drop=True)
    lmp = g["lmp"]
    prior = lmp.shift(1)                                          
    roll_mean = prior.rolling(ROLLING_HOURS, min_periods=MIN_PERIODS).mean()
    roll_std = prior.rolling(ROLLING_HOURS, min_periods=MIN_PERIODS).std()
    lo = roll_mean - N_STD * roll_std
    hi = roll_mean + N_STD * roll_std
    clipped = lmp.clip(lower=lo, upper=hi)
                                                           
    clipped = clipped.where(roll_std.notna(), lmp)
    g["lmp_was_clipped"] = (clipped != lmp) & roll_std.notna()
    g["lmp"] = clipped.astype(np.float32)
    return g


def main():
    panel = pd.read_parquet(PANEL_PATH)
    out_groups = []
    for (iso, loc), g in panel.groupby(["iso", "location"], observed=True):
        out_groups.append(clip_group(g))
    out = pd.concat(out_groups, ignore_index=True)

    print("=== fraction of rows clipped, per ISO ===")
    per_iso = out.groupby("iso", observed=True)["lmp_was_clipped"].mean().sort_values(ascending=False)
    for iso, frac in per_iso.items():
        print(f"  {iso}: {frac * 100:.3f}%")
    print(f"  OVERALL: {out['lmp_was_clipped'].mean() * 100:.3f}%")

    out = out.drop(columns=["lmp_was_clipped"])
    out = out.sort_values(["iso", "location", "interval_start_utc"]).reset_index(drop=True)
    out.to_parquet(OUT_PATH, index=False)
    print(f"wrote {OUT_PATH} ({len(out)} rows)")


if __name__ == "__main__":
    main()
