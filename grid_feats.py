





from __future__ import annotations

import torch
import torch.nn as nn


class CalendarEmbed(nn.Module):
    






    def __init__(self, d_model: int, n_hours: int = 24, n_dows: int = 7, n_months: int = 12):
        super().__init__()
        self.hour = nn.Embedding(n_hours, d_model)
        self.dow = nn.Embedding(n_dows, d_model)
        self.month = nn.Embedding(n_months, d_model)
        for emb in (self.hour, self.dow, self.month):
            nn.init.normal_(emb.weight, std=0.02)

    def forward(self, hour: torch.Tensor, dow: torch.Tensor,
                month: torch.Tensor | None = None) -> torch.Tensor:
        out = self.hour(hour) + self.dow(dow)
        if month is not None:
            out = out + self.month(month)
        return out
