







from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import torch

from src.calendar_features import calendar_features

CONTEXT_HOURS = 336


def _anchor_position(index: pd.DatetimeIndex, gate: pd.Timestamp) -> int:
    





    anchor = gate.floor("h")
    pos = int(index.searchsorted(anchor, side="right")) - 1
    if pos < 0 or index[pos] != anchor:
        raise KeyError(f"gate hour {anchor} is not in the zone index")
    return pos


@torch.no_grad()
def forecast_day_batch(model, zone_series: dict[str, pd.DataFrame], gate: pd.Timestamp,
                        day_start: pd.Timestamp, day_end: pd.Timestamp, device: str,
                        use_covariate: bool, graph_bias: torch.Tensor | None = None,
                        calendar_tz: str | None = None,
                        context_hours: int = CONTEXT_HOURS) -> pd.DataFrame:
    




    zone_names = list(zone_series.keys())
    if not zone_names:
        return pd.DataFrame()

    common = None
    for s in zone_series.values():
        common = s.index if common is None else common.intersection(s.index)
    if common is None or len(common) == 0:
        return pd.DataFrame()
    common = common.sort_values()

    try:
        pos = _anchor_position(common, gate)
    except KeyError:
        return pd.DataFrame()
    if pos < context_hours:
        return pd.DataFrame()

                                                                                             
    span = (common[pos - 1] - common[pos - context_hours]).total_seconds() / 3600
    if span != context_hours - 1:
        return pd.DataFrame()

    anchor_ts = common[pos]
    horizon = int((day_end - anchor_ts) / pd.Timedelta(hours=1))
    if horizon <= 0:
        return pd.DataFrame()
    n_patches = -(-horizon // model.patch_size)
    total = n_patches * model.patch_size

    future_index = pd.date_range(anchor_ts, periods=total, freq="h", tz="UTC")
    full_index = common[pos - context_hours:pos].append(future_index)

    N = len(zone_names)
    series = np.empty((1, N, context_hours + total), dtype=np.float32)
    cov = np.zeros((1, N, context_hours + total), dtype=np.float32)
    cov_mask = np.zeros((1, N, context_hours + total), dtype=bool)

    for n, z in enumerate(zone_names):
        s = zone_series[z].reindex(full_index)
        lmp = s["lmp"].to_numpy(dtype=np.float32)
        if not np.isfinite(lmp[:context_hours]).all():
            return pd.DataFrame()
                                                                                                 
                                                                                              
                                                                                                
                                                                                                 
                                                                                             
        series[0, n] = lmp
        series[0, n, context_hours:] = lmp[context_hours - 1]
        if use_covariate:
                                                                                              
                                                                                   
            cov[0, n] = np.nan_to_num(s["load_forecast"].to_numpy(dtype=np.float32), nan=0.0)
            cov_mask[0, n] = s["load_forecast_available"].fillna(False).to_numpy(dtype=bool)

    cal = None
    if model.n_calendar_features:
        if calendar_tz is None:
            raise ValueError("model was built with calendar features but calendar_tz is None")
        cal_arr = calendar_features(full_index, calendar_tz)          
        cal = np.broadcast_to(cal_arr[None, None], (1, N) + cal_arr.shape).copy()

    series_t = torch.from_numpy(series).to(device)
    cov_t = torch.from_numpy(cov).to(device) if use_covariate else None
    mask_t = torch.from_numpy(cov_mask).to(device) if use_covariate else None
    cal_t = torch.from_numpy(cal).to(device) if cal is not None else None

                                                                                        
                                                                                            
    wants_future_cal = 'calendar_future' in inspect.signature(model.forward).parameters

    model.eval()
    q_out = []
    for p in range(n_patches):
        lo = p * model.patch_size
        ctx = series_t[:, :, lo:lo + context_hours]
        cov_p = cov_t[:, :, lo:lo + context_hours] if use_covariate else None
        mask_p = mask_t[:, :, lo:lo + context_hours] if use_covariate else None
        cal_p = cal_t[:, :, :, lo:lo + context_hours] if cal_t is not None else None
        extra = {}
        if wants_future_cal and cal_t is not None:
                                                                                               
            extra['calendar_future'] = cal_t[:, :, :,
                                             lo + context_hours:lo + context_hours + model.patch_size]
        pred = model(ctx, cov_p, mask_p, graph_bias=graph_bias, calendar=cal_p, **extra)
        q_out.append(pred[0])                                
                                                                                                
        hi = context_hours + lo
        series_t[:, :, hi:hi + model.patch_size] = model.median(pred)

    q = torch.cat(q_out, dim=1).cpu().numpy()                           

    keep = (future_index >= day_start) & (future_index < day_end)
    if not keep.any():
        return pd.DataFrame()
    q = q[:, keep, :]
    kept_index = future_index[keep]

    q_cols = [f"q{lvl:.1f}" for lvl in model.quantiles]
    frames = []
    for n, z in enumerate(zone_names):
        d = pd.DataFrame(q[n], columns=q_cols)
        d.insert(0, "interval_start_utc", kept_index)
        d.insert(0, "location", z)
        d.insert(2, "y_pred", q[n, :, model.median_idx])
        frames.append(d)
    return pd.concat(frames, ignore_index=True)
