





from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src.calendar_features import (N_CALENDAR_FEATURES, N_CALENDAR_FEATURES_EXTENDED,
                                    calendar_features, calendar_features_extended)
from src.config import ProjectConfig, load_config
from src.dayahead_data import MIN_HISTORY_DATE
from src.splits import filter_split

                                                                                    
                                                                                             
                                                                                                 
                                                                                             
                                                    
                                                                                              
                                                        
EXCLUDED_ISOS: set[str] = {"ISONE"}


class HeldOutPretrainDataset:
    def __init__(self, panel: pd.DataFrame, held_out_iso: str, context_len: int, patch_size: int,
                 cfg: ProjectConfig | None = None, seed: int = 0,
                 excluded_isos: frozenset[str] | None = None,
                 include_target_train: bool = False, include_calendar: bool = False,
                 calendar_mode: str = "basic"):
        



        self.cfg = cfg or load_config()
        self.context_len = context_len
        self.patch_size = patch_size
        self.rng = np.random.RandomState(seed)
        self.include_calendar = include_calendar
        self.calendar_mode = calendar_mode
        self.n_calendar_features = N_CALENDAR_FEATURES if calendar_mode == "basic" else N_CALENDAR_FEATURES_EXTENDED
        self.iso_calendar: dict[str, np.ndarray] = {}

        base_excluded = EXCLUDED_ISOS if excluded_isos is None else excluded_isos
        excluded = base_excluded if include_target_train else (base_excluded | {held_out_iso})
        train = filter_split(panel[~panel["iso"].isin(excluded)], "train", self.cfg.splits)
        train = train[train["interval_start_utc"] >= pd.Timestamp(MIN_HISTORY_DATE, tz="UTC")]

        self.iso_zones: dict[str, dict[str, dict]] = {}
        self.iso_valid_starts: dict[str, np.ndarray] = {}
        self.iso_n_future_patches: dict[str, int] = {}
        for iso, g_iso in train.groupby("iso", observed=True):
            iso_cfg = self.cfg.iso(iso)
            zones = {}
            common_index = None
            for loc, g in g_iso.groupby("location", observed=True):
                g = g.sort_values("interval_start_utc")
                idx = g["interval_start_utc"].to_numpy()
                zones[loc] = {
                    "lmp": g["lmp"].to_numpy(dtype=np.float32),
                    "load_forecast": g["load_forecast"].to_numpy(dtype=np.float32),
                    "load_forecast_available": g["load_forecast_available"].to_numpy(dtype=bool),
                    "index": idx,
                }
                common_index = idx if common_index is None else np.intersect1d(common_index, idx)
            if common_index is None:
                continue

                                                                               
            total_future_hours = int(np.ceil(48 - iso_cfg.gate_local_hour))
            n_future_patches = -(-total_future_hours // patch_size)
            future_hours = n_future_patches * patch_size
            window_len = context_len + future_hours
            if len(common_index) < window_len:
                continue

            for loc, z in zones.items():
                pos = np.searchsorted(z["index"], common_index)
                zones[loc] = {k: (v[pos] if k != "index" else common_index) for k, v in z.items()}

                                                                                      
            local_hour = pd.DatetimeIndex(common_index).tz_convert(iso_cfg.tz).hour
            gate_hour_int = int(round(iso_cfg.gate_local_hour)) % 24
            candidates = np.where(local_hour == gate_hour_int)[0]
            candidates = candidates[(candidates >= context_len) & (candidates <= len(common_index) - future_hours)]
            if len(candidates) == 0:
                continue

            self.iso_zones[iso] = zones
            self.iso_valid_starts[iso] = candidates
            self.iso_n_future_patches[iso] = n_future_patches
            if self.include_calendar:
                                                                                          
                cal_fn = calendar_features if self.calendar_mode == "basic" else calendar_features_extended
                self.iso_calendar[iso] = cal_fn(common_index, iso_cfg.tz)

        self.isos = list(self.iso_zones.keys())
        if not self.isos:
            raise ValueError(f"no ISO with a usable gate-aligned window after holding out {held_out_iso}")
        if not include_target_train and held_out_iso in self.isos:
            raise ValueError(f"held-out ISO {held_out_iso} leaked into the pretraining pool: {self.isos}")

    def sample_batch(self, batch_size: int) -> dict[str, torch.Tensor | str | int]:
        
        iso = self.isos[self.rng.randint(len(self.isos))]
        return self.sample_batch_from(iso, self.iso_valid_starts[iso], batch_size)

    def sample_batch_from(self, iso: str, starts, batch_size: int) -> dict[str, torch.Tensor | str | int]:
        





        zones = self.iso_zones[iso]
        zone_names = list(zones.keys())
        N = len(zone_names)
        n_future_patches = self.iso_n_future_patches[iso]
        future_hours = n_future_patches * self.patch_size

        full_len = self.context_len + future_hours
        target = np.empty((batch_size, N, self.context_len), dtype=np.float32)
        future = np.empty((batch_size, N, n_future_patches, self.patch_size), dtype=np.float32)
        cov_full = np.empty((batch_size, N, full_len), dtype=np.float32)
        cov_mask_full = np.empty((batch_size, N, full_len), dtype=bool)
        cal_full = None
        if self.include_calendar:
            cal_full = np.empty((batch_size, N, self.n_calendar_features, full_len), dtype=np.float32)
        iso_cal = self.iso_calendar.get(iso) if self.include_calendar else None

        for b in range(batch_size):
            gate = starts[self.rng.randint(len(starts))]
            lo, hi = gate - self.context_len, gate + future_hours
            for n, z in enumerate(zone_names):
                s = zones[z]
                target[b, n] = s["lmp"][lo:gate]
                future[b, n] = s["lmp"][gate:hi].reshape(n_future_patches, self.patch_size)
                cov_full[b, n] = np.nan_to_num(s["load_forecast"][lo:hi], nan=0.0)
                cov_mask_full[b, n] = s["load_forecast_available"][lo:hi]
            if self.include_calendar:
                cal_full[b, :, :, :] = iso_cal[:, lo:hi]

        out = {
            "iso": iso,
            "zones": zone_names,
            "n_future_patches": n_future_patches,
            "target": torch.from_numpy(target),
            "future": torch.from_numpy(future),
            "covariate_full": torch.from_numpy(cov_full),
            "cov_mask_full": torch.from_numpy(cov_mask_full),
        }
        if self.include_calendar:
            out["calendar_full"] = torch.from_numpy(cal_full)
        return out
