
from __future__ import annotations

import numpy as np
import pandas as pd
import torch


def pinball_loss(pred_quantiles: torch.Tensor, target: torch.Tensor,
                  quantiles: tuple[float, ...]) -> torch.Tensor:
    
    q = torch.tensor(quantiles, device=pred_quantiles.device, dtype=pred_quantiles.dtype)
    target = target.unsqueeze(-1)
    diff = target - pred_quantiles
    loss = torch.maximum(q * diff, (q - 1) * diff)
    return loss.mean()


def crps_from_quantiles(y_true: np.ndarray, quantile_preds: np.ndarray, quantiles: tuple[float, ...]) -> float:
    
    q = np.asarray(quantiles)
    diff = y_true[:, None] - quantile_preds
    loss = np.maximum(q * diff, (q - 1) * diff)
    return float(2 * loss.mean())


def score_with_crps(pred_df: pd.DataFrame, quantile_cols: list[str], quantiles: tuple[float, ...]) -> pd.DataFrame:
    
    rows = []
    for loc, g in pred_df.groupby("location"):
        mae = float(np.mean(np.abs(g["y_true"] - g["y_pred"])))
        rmse = float(np.sqrt(np.mean((g["y_true"] - g["y_pred"]) ** 2)))
        crps = crps_from_quantiles(g["y_true"].to_numpy(), g[quantile_cols].to_numpy(), quantiles)
        rows.append({"location": loc, "mae": mae, "rmse": rmse, "crps": crps, "n": len(g)})
    tab = pd.DataFrame(rows).set_index("location")
    avg = pd.DataFrame({"mae": [tab["mae"].mean()], "rmse": [tab["rmse"].mean()],
                        "crps": [tab["crps"].mean()], "n": [tab["n"].sum()]}, index=["AVERAGE"])
    return pd.concat([tab, avg])
