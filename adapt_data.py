
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.calendar_features import (N_CALENDAR_FEATURES, N_CALENDAR_FEATURES_EXTENDED,
                                    calendar_features, calendar_features_extended)

CONTEXT_HOURS = 336
PATCH_SIZE = 16


class AdaptDataset:
    def __init__(self, zone_series: dict[str, pd.DataFrame], budget_days: list[pd.Timestamp],
                 seed: int = 0, tz: str | None = None, context_hours: int = CONTEXT_HOURS,
                 calendar_mode: str = "basic", patch_size: int = PATCH_SIZE,
                 context_in_budget: bool = False):
        









        self.rng = np.random.RandomState(seed)
        self.zone_names = list(zone_series.keys())
        self.N = len(self.zone_names)
        self.tz = tz
        self.context_hours = context_hours
        self.calendar_mode = calendar_mode
        self.patch_size = patch_size
        self.context_in_budget = context_in_budget
        self.n_calendar_features = N_CALENDAR_FEATURES if calendar_mode == "basic" else N_CALENDAR_FEATURES_EXTENDED

        common_index = None
        for s in zone_series.values():
            idx = s.index
            common_index = idx if common_index is None else common_index.intersection(idx)
        self.common_index = common_index if common_index is not None else pd.DatetimeIndex([])
        cal_fn = calendar_features if calendar_mode == "basic" else calendar_features_extended
        self.calendar_arr = cal_fn(self.common_index, tz) if (tz and len(self.common_index)) else None

        self.arrays = {}
        for z in self.zone_names:
            s = zone_series[z].reindex(self.common_index)
            self.arrays[z] = {
                "lmp": s["lmp"].to_numpy(dtype=np.float32),
                "load_forecast": np.nan_to_num(s["load_forecast"].to_numpy(dtype=np.float32), nan=0.0),
                "load_forecast_available": s["load_forecast_available"].fillna(False).to_numpy(dtype=bool),
            }

        self.valid_starts = np.array([])
        if budget_days and len(self.common_index) >= self.context_hours + self.patch_size:
            lo = budget_days[0].tz_convert("UTC")
            hi = (budget_days[-1] + pd.DateOffset(days=1)).tz_convert("UTC")
            candidates = self.common_index[(self.common_index >= lo) & (self.common_index < hi)]
            positions = self.common_index.searchsorted(candidates)
            floor = int(self.common_index.searchsorted(lo)) if self.context_in_budget else 0
            starts = []
            for pos in positions:
                if pos - self.context_hours < floor or pos + self.patch_size > len(self.common_index):
                    continue
                span_hours = (self.common_index[pos - 1] - self.common_index[pos - self.context_hours]).total_seconds() / 3600
                if span_hours == self.context_hours - 1:
                    starts.append(pos)
            self.valid_starts = np.array(starts)

    def is_empty(self) -> bool:
        return len(self.valid_starts) == 0

    def sample_batch(self, batch_size: int, use_covariate: bool,
                     include_calendar: bool = False) -> dict[str, torch.Tensor]:
        target = np.empty((batch_size, self.N, self.context_hours), dtype=np.float32)
        target_next = np.empty((batch_size, self.N, self.patch_size), dtype=np.float32)
        cov = np.empty((batch_size, self.N, self.context_hours), dtype=np.float32)
        cov_mask = np.empty((batch_size, self.N, self.context_hours), dtype=bool)
        cal = None
        if include_calendar:
            assert self.calendar_arr is not None, "include_calendar=True requires tz= at construction"
            cal = np.empty((batch_size, self.N, self.n_calendar_features, self.context_hours), dtype=np.float32)
            cal_next = np.empty((batch_size, self.N, self.n_calendar_features, self.patch_size), dtype=np.float32)

        for b in range(batch_size):
            pos = self.valid_starts[self.rng.randint(len(self.valid_starts))]
            for n, z in enumerate(self.zone_names):
                a = self.arrays[z]
                target[b, n] = a["lmp"][pos - self.context_hours:pos]
                target_next[b, n] = a["lmp"][pos:pos + self.patch_size]
                if use_covariate:
                    cov[b, n] = a["load_forecast"][pos - self.context_hours:pos]
                    cov_mask[b, n] = a["load_forecast_available"][pos - self.context_hours:pos]
            if include_calendar:
                cal[b, :, :, :] = self.calendar_arr[:, pos - self.context_hours:pos]
                cal_next[b, :, :, :] = self.calendar_arr[:, pos:pos + self.patch_size]
        out = {"target": torch.from_numpy(target), "target_next": torch.from_numpy(target_next)}
        if use_covariate:
            out["covariate"] = torch.from_numpy(cov)
            out["cov_mask"] = torch.from_numpy(cov_mask)
        if include_calendar:
            out["calendar"] = torch.from_numpy(cal)
            out["calendar_next"] = torch.from_numpy(cal_next)
        return out
