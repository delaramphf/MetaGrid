

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch
import yaml

CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "configs", "zone_adjacency.yaml")


def learned_adjacency(panel: pd.DataFrame, iso: str, locations: list[str],
                       bias_scale: float = 2.0, train_only: bool = True) -> torch.Tensor:
    




    sub = panel[panel["iso"] == iso]
    if train_only:
        from src.config import load_config
        from src.splits import filter_split
        sub = filter_split(sub, "train", load_config().splits)
    wide = sub.pivot(index="interval_start_utc", columns="location", values="lmp")
    wide = wide.reindex(columns=locations)
    corr = wide.corr().fillna(0.0).values
    np.fill_diagonal(corr, 1.0)
    return torch.tensor(corr * bias_scale, dtype=torch.float32)


def congestion_adjacency(panel: pd.DataFrame, iso: str, locations: list[str],
                          bias_scale: float = 2.0, train_only: bool = True) -> torch.Tensor:
    



    if "congestion" not in panel.columns or panel.loc[panel["iso"] == iso, "congestion"].isna().all():
        raise NotImplementedError(f"no congestion data for {iso} -- can't build a congestion-based bias")
    sub = panel[panel["iso"] == iso]
    if train_only:
        from src.config import load_config
        from src.splits import filter_split
        sub = filter_split(sub, "train", load_config().splits)
    wide = sub.pivot(index="interval_start_utc", columns="location", values="congestion")
    wide = wide.reindex(columns=locations)
    corr = wide.corr().fillna(0.0).values
    np.fill_diagonal(corr, 1.0)
    return torch.tensor(corr * bias_scale, dtype=torch.float32)


def _neighbor_mask(iso: str, locations: list[str]) -> torch.Tensor:
    

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    if iso not in cfg or not cfg[iso]:
        raise NotImplementedError(
            f"no published zone adjacency on file for {iso} in {CONFIG_PATH} -- "
            f"this graph-bias mode needs this filled in, not guessed at"
        )
    adj = cfg[iso]
    N = len(locations)
    mask = torch.zeros((N, N), dtype=torch.bool)
    idx = {loc: i for i, loc in enumerate(locations)}
    for i in range(N):
        mask[i, i] = True
    for a, neighbors in adj.items():
        if a not in idx:
            continue
        for b in neighbors:
            if b in idx:
                mask[idx[a], idx[b]] = True
                mask[idx[b], idx[a]] = True
    return mask


def scaled_laplacian(iso: str, locations: list[str]) -> torch.Tensor:
    









    mask = _neighbor_mask(iso, locations)
    A = mask.to(torch.float32)
    d_inv_sqrt = A.sum(dim=1).clamp_min(1e-12).pow(-0.5)
    N = len(locations)
    eye = torch.eye(N, dtype=torch.float32)
    L = eye - d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :]
    lam_max = float(torch.linalg.eigvalsh(L).max().clamp_min(1e-6))
    return (2.0 / lam_max) * L - eye


def true_adjacency(iso: str, locations: list[str]) -> torch.Tensor:
    
    mask = _neighbor_mask(iso, locations)
    bias = torch.where(mask, torch.zeros_like(mask, dtype=torch.float32), torch.full_like(mask, -5.0, dtype=torch.float32))
    return bias


def hybrid_adjacency(panel: pd.DataFrame, iso: str, locations: list[str],
                      bias_scale: float = 2.0, non_neighbor_bias: float = -5.0,
                      train_only: bool = True) -> torch.Tensor:
    




    mask = _neighbor_mask(iso, locations)
    corr = learned_adjacency(panel, iso, locations, bias_scale=bias_scale, train_only=train_only)
    return torch.where(mask, corr, torch.full_like(corr, non_neighbor_bias))
