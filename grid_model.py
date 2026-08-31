
from __future__ import annotations

import math

import torch
import torch.nn as nn


class RevIN(nn.Module):
    

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                     
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(self.eps)
        return (x - mean) / std, mean, std

    @staticmethod
    def denorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean


class SharedScaleNorm(nn.Module):
    






    def __init__(self, per_zone_loc: bool = False, eps: float = 1e-5):
        super().__init__()
        self.per_zone_loc, self.eps = per_zone_loc, eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        loc_dims = (-1,) if self.per_zone_loc else (1, 2)
        loc = x.mean(dim=loc_dims, keepdim=True)
        scale = x.std(dim=(1, 2), keepdim=True).clamp_min(self.eps)
        return (x - loc) / scale, loc, scale

    @staticmethod
    def denorm(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return x * std + mean


class AsinhRevIN(nn.Module):
    

    def __init__(self, eps: float = 1e-5):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True).clamp_min(self.eps)
        return torch.asinh((x - mean) / std), mean, std

    @staticmethod
    def denorm(y: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return torch.sinh(y) * std + mean


class AsinhRobustNorm(nn.Module):
    

    def __init__(self, c: float = 40.0, eps: float = 1e-5):
        super().__init__()
        self.c = c
        self.eps = eps

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                                                              
        z = torch.asinh(x / self.c)
        median = z.median(dim=-1, keepdim=True).values
        q75 = z.quantile(0.75, dim=-1, keepdim=True)
        q25 = z.quantile(0.25, dim=-1, keepdim=True)
        iqr = (q75 - q25).clamp_min(self.eps)
        y = (z - median) / iqr
        return y, median, iqr

    def denorm(self, y: torch.Tensor, median: torch.Tensor, iqr: torch.Tensor) -> torch.Tensor:
        z = y * iqr + median
        return torch.sinh(z) * self.c


class PatchEmbed(nn.Module):
    






    CONV_CHANNELS = 32
    CONV_KERNEL = 5

    def __init__(self, patch_size: int, d_model: int, use_covariate: bool, tokenizer_mode: str = "linear",
                 n_calendar_features: int = 0, patch_stride: int | None = None):
        super().__init__()
        self.patch_stride = patch_stride if patch_stride is not None else patch_size
        assert tokenizer_mode in ("linear", "mlp", "conv")
        assert n_calendar_features == 0 or tokenizer_mode in ("linear", "mlp"), \
            "calendar features not implemented for tokenizer_mode='conv'"
        self.patch_size = patch_size
        self.use_covariate = use_covariate
        self.tokenizer_mode = tokenizer_mode
        self.n_calendar_features = n_calendar_features
        in_dim = patch_size * (2 if use_covariate else 1) + patch_size * n_calendar_features
        if tokenizer_mode == "mlp":
            self.shortcut = nn.Linear(in_dim, d_model)
            self.mlp_fc1 = nn.Linear(d_model, d_model)
            self.mlp_act = nn.GELU()
            self.mlp_fc2 = nn.Linear(d_model, d_model)
        elif tokenizer_mode == "conv":
            n_channels = 2 if use_covariate else 1
                                                                                                   
            self.conv = nn.Conv1d(n_channels, self.CONV_CHANNELS * n_channels, kernel_size=self.CONV_KERNEL,
                                   groups=n_channels, padding=0)
            self.conv_act = nn.GELU()
            conv_in_dim = patch_size * self.CONV_CHANNELS * n_channels
            self.proj = nn.Linear(conv_in_dim, d_model)
        else:
            self.proj = nn.Linear(in_dim, d_model)
        if use_covariate:
            self.mask_token = nn.Parameter(torch.zeros(patch_size))
            nn.init.normal_(self.mask_token, std=0.02)
        self.pos_embed_scale = nn.Parameter(torch.ones(1))

    def _to_patches(self, x: torch.Tensor) -> torch.Tensor:
                                                         
        B, N, L = x.shape
        if self.patch_stride == self.patch_size:
            assert L % self.patch_size == 0, f"context length {L} must be a multiple of patch_size {self.patch_size}"
            return x.view(B, N, L // self.patch_size, self.patch_size)
        return x.unfold(-1, self.patch_size, self.patch_stride)                                 

    def _raw_channels(self, target: torch.Tensor, covariate: torch.Tensor | None,
                       cov_mask: torch.Tensor | None) -> torch.Tensor:
        
        if not self.use_covariate:
            return target.unsqueeze(2)
        if covariate is None or cov_mask is None:
            cov_full = self.mask_token.repeat(target.shape[-1] // self.patch_size)
            cov_full = cov_full.expand_as(target)
        else:
            mask_full = cov_mask.bool()
            cov_full = torch.where(mask_full, covariate, self.mask_token.repeat(target.shape[-1] // self.patch_size))
        return torch.stack([target, cov_full], dim=2)

    def forward(self, target: torch.Tensor, covariate: torch.Tensor | None = None,
                cov_mask: torch.Tensor | None = None, calendar: torch.Tensor | None = None) -> torch.Tensor:
        
        if self.tokenizer_mode == "conv":
            B, N, L = target.shape
            n_channels = 2 if self.use_covariate else 1
            raw = self._raw_channels(target, covariate, cov_mask)                      
            raw = raw.reshape(B * N, n_channels, L)
                                                                                        
            padded = nn.functional.pad(raw, (self.CONV_KERNEL - 1, 0))
            feat = self.conv_act(self.conv(padded))                                      
            feat = feat.view(B, N, self.CONV_CHANNELS * n_channels, L)
            feat = feat.permute(0, 1, 3, 2)                                    
            num_patches = L // self.patch_size
            feat = feat.reshape(B, N, num_patches, self.patch_size * self.CONV_CHANNELS * n_channels)
            return self.proj(feat)

        tp = self._to_patches(target)
        if self.use_covariate:
            if covariate is None or cov_mask is None:
                cp = self.mask_token.expand_as(tp)
            else:
                cp = self._to_patches(covariate)
                cmask = self._to_patches(cov_mask.float()).mean(dim=-1, keepdim=True) > 0.5
                cp = torch.where(cmask, cp, self.mask_token.expand_as(cp))
            patches = torch.cat([tp, cp], dim=-1)
        else:
            patches = tp
        if self.n_calendar_features > 0:
            assert calendar is not None and calendar.shape[2] == self.n_calendar_features
            cal_patches = torch.cat([self._to_patches(calendar[:, :, c, :]) for c in range(self.n_calendar_features)],
                                     dim=-1)
            patches = torch.cat([patches, cal_patches], dim=-1)
        if self.tokenizer_mode == "mlp":
            shortcut = self.shortcut(patches)
            return shortcut + self.mlp_fc2(self.mlp_act(self.mlp_fc1(shortcut)))
        return self.proj(patches)


class CrossSeriesAttention(nn.Module):
    

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, bias: torch.Tensor | None) -> torch.Tensor:
                                                                        
        B, N, T, D = x.shape
        xt = x.permute(0, 2, 1, 3).reshape(B * T, N, D)
        attn_mask = None
        if bias is not None:
                                                                         
            attn_mask = bias.unsqueeze(0).expand(B * T * self.n_heads, -1, -1)
        out, _ = self.attn(xt, xt, xt, attn_mask=attn_mask, need_weights=False)
        out = self.norm(out + xt)
        return out.view(B, T, N, D).permute(0, 2, 1, 3)


class ChebCrossZone(nn.Module):
    












    def __init__(self, d_model: int, K: int = 3, dropout: float = 0.1):
        super().__init__()
        self.K = K
        self.theta = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(K + 1)])
        for lin in self.theta:
            nn.init.normal_(lin.weight, std=0.02)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, l_tilde: torch.Tensor | None) -> torch.Tensor:
                                                                                    
        assert l_tilde is not None, "graph_bias_mode='cheb' requires a scaled Laplacian"
        xt = x.permute(0, 2, 1, 3)                
        lap = l_tilde.to(xt.dtype)
        t_prev = xt                
        out = self.theta[0](t_prev)
        if self.K >= 1:
            t_cur = lap @ xt                  
            out = out + self.theta[1](t_cur)
            for k in range(2, self.K + 1):
                t_next = 2.0 * (lap @ t_cur) - t_prev
                out = out + self.theta[k](t_next)
                t_prev, t_cur = t_cur, t_next
        out = self.norm(self.dropout(out) + xt)
        return out.permute(0, 2, 1, 3)

    @torch.no_grad()
    def lipschitz_constant(self, n_grid: int = 257) -> float:
        





        if self.K < 1:
            return 0.0
        w = torch.stack([lin.weight for lin in self.theta])               
        lam = torch.linspace(-1.0, 1.0, n_grid, device=w.device, dtype=w.dtype)
        us = [torch.ones_like(lam)]       
        if self.K >= 2:
            us.append(2.0 * lam)       
            for _ in range(2, self.K):
                us.append(2.0 * lam * us[-1] - us[-2])
        best = 0.0
        for i in range(n_grid):
            m = torch.zeros_like(w[0])
            for k in range(1, self.K + 1):
                m = m + k * us[k - 1][i] * w[k]
            best = max(best, float(torch.linalg.matrix_norm(m, ord=2)))
        return best


DEFAULT_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


class GatedResidualHead(nn.Module):
    













    def __init__(self, d_model: int, patch_size: int, n_quantiles: int, hidden: int = 256):
        super().__init__()
        self.patch, self.n_q = patch_size, n_quantiles
        self.base = nn.Linear(d_model, patch_size * n_quantiles)
        self.common = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                    nn.Linear(hidden, patch_size))
        self.residual = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                      nn.Linear(hidden, patch_size))
        self.g = nn.Parameter(torch.zeros(()))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        
        B, N, D = z.shape
        base = self.base(z).view(B, N, self.patch, self.n_q)
        m = self.common(z.mean(dim=1)).unsqueeze(1)                                    
        r_t = self.residual(z)
        r = r_t - r_t.mean(dim=1, keepdim=True)                                       
        out = base + (self.g * (m + r)).unsqueeze(-1)                                       
        out, _ = torch.sort(out, dim=-1)
        return out, r


class GridTransferModel(nn.Module):
    






    def __init__(self, patch_size: int = 16, d_model: int = 384, n_heads: int = 6,
                 num_layers: int = 8, ff_dim: int = 1536, use_covariate: bool = True,
                 graph_bias_mode: str = "none", dropout: float = 0.1,
                 quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
                 tokenizer_mode: str = "linear", norm_mode: str = "revin",
                 film_mode: bool = False, zone_vocab_size: int | None = None,
                 n_calendar_features: int = 0, use_learned_calendar: bool = False,
                 asinh_robust_c: float = 40.0, patch_stride: int | None = None,
                 head_mode: str = "linear", cross_mode: str = "masked",
                 graph_init_logit: float = -4.0,
                 use_adjacency: bool = True, use_correlation: bool = True,
                 corr_shrink: float = 0.1, gamma: float = 2.0,
                 corr_center_zones: bool = False):
        super().__init__()
        assert graph_bias_mode in ("none", "learned", "true", "hybrid")
        assert norm_mode in ("revin", "asinh_revin", "asinh_robust", "market", "zoneloc")
        assert head_mode in ("linear", "structured", "v4")
        assert cross_mode in ("masked", "graphaware")
        self.cross_mode = cross_mode
        self.corr_shrink = corr_shrink
        self.corr_center_zones = corr_center_zones
        self.head_mode = head_mode
        assert not film_mode or zone_vocab_size is not None, "film_mode requires zone_vocab_size"
        self.patch_size = patch_size
        self.graph_bias_mode = graph_bias_mode
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)
        self.median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
        self.norm_mode = norm_mode
        self.n_calendar_features = n_calendar_features
        if norm_mode in ("market", "zoneloc"):
            self.revin = SharedScaleNorm(per_zone_loc=(norm_mode == "zoneloc"))
        elif norm_mode == "asinh_revin":
            self.revin = AsinhRevIN()
        elif norm_mode == "asinh_robust":
            self.revin = AsinhRobustNorm(c=asinh_robust_c)
        else:
            self.revin = RevIN()
        self.patch_stride = patch_stride if patch_stride is not None else patch_size
        self.embed = PatchEmbed(patch_size, d_model, use_covariate, tokenizer_mode=tokenizer_mode,
                                 n_calendar_features=n_calendar_features, patch_stride=patch_stride)

        self.use_learned_calendar = use_learned_calendar
        if use_learned_calendar:
            from src.grid_feats import CalendarEmbed
            self.calendar_embed = CalendarEmbed(d_model)

        self.film_mode = film_mode
        if film_mode:
            self.zone_embed = nn.Embedding(zone_vocab_size, d_model)
            self.film_proj = nn.Linear(d_model, 2 * d_model)
                                                             
            nn.init.zeros_(self.film_proj.weight)
            nn.init.zeros_(self.film_proj.bias)

        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout=dropout,
                                            batch_first=True, activation="gelu")
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

        if graph_bias_mode == "none":
            self.cross_series = None
        elif cross_mode == "graphaware":
                                                                                            
                                                                                       
            from src.gridmeta_v3 import GraphAwareCrossZone
            self.cross_series = GraphAwareCrossZone(d_model, n_heads, dropout, gamma=gamma,
                                                   init_logit=graph_init_logit,
                                                   use_adjacency=use_adjacency,
                                                   use_correlation=use_correlation)
        else:
            self.cross_series = CrossSeriesAttention(d_model, n_heads)

        if head_mode == "v4":
            self.head = GatedResidualHead(d_model, patch_size, self.n_quantiles)
        elif head_mode == "structured":
                                                                                          
            from src.gridmeta_v3 import StructuredHead
            self.head = StructuredHead(d_model, patch_size, self.n_quantiles)
        else:
                                                                                 
            self.head = nn.Linear(d_model, patch_size * self.n_quantiles)
        self._last_residual = None

    def forward(self, target: torch.Tensor, covariate: torch.Tensor | None = None,
                cov_mask: torch.Tensor | None = None, graph_bias: torch.Tensor | None = None,
                zone_ids: torch.Tensor | None = None, calendar: torch.Tensor | None = None,
                learned_calendar_idx: dict[str, torch.Tensor] | None = None,
                target_calendar_idx: dict[str, torch.Tensor] | None = None) -> torch.Tensor:
        

        B, N, L = target.shape
        norm_target, mean, std = self.revin(target)
        norm_cov = None
        if covariate is not None:
            cov_mean = covariate.mean(dim=-1, keepdim=True)
            cov_std = covariate.std(dim=-1, keepdim=True).clamp_min(1e-5)
            if self.norm_mode == "asinh_revin":
                norm_cov = torch.asinh((covariate - cov_mean) / cov_std)
            elif self.norm_mode == "asinh_robust":
                norm_cov, _, _ = self.revin(covariate)
            else:
                norm_cov = (covariate - cov_mean) / cov_std

        patches = self.embed(norm_target, norm_cov, cov_mask, calendar)                
        T, D = patches.shape[2], patches.shape[3]

        if self.use_learned_calendar:
            assert learned_calendar_idx is not None, "use_learned_calendar=True requires learned_calendar_idx"
            cal_emb = self.calendar_embed(**learned_calendar_idx)
            patches = patches + cal_emb

        if self.film_mode:
            assert zone_ids is not None and zone_ids.shape[0] == N, \
                "film_mode=True requires zone_ids of shape [N]"
            gamma_beta = self.film_proj(self.zone_embed(zone_ids))           
            gamma, beta = gamma_beta.chunk(2, dim=-1)
            patches = patches * (1.0 + gamma.view(1, N, 1, D)) + beta.view(1, N, 1, D)

        x = patches.reshape(B * N, T, D)
        x = self.temporal_encoder(x)
        x = x.reshape(B, N, T, D)

        if self.cross_series is not None:
            bias = graph_bias if self.graph_bias_mode in ("learned", "true", "hybrid") else None
            if self.cross_mode == "graphaware":
                from src.gridmeta_v3 import context_correlation
                x = self.cross_series(x, bias, context_correlation(
                    norm_target, self.corr_shrink, self.corr_center_zones))
            else:
                x = self.cross_series(x, bias)

        last = x[:, :, -1, :]             
        if self.use_learned_calendar and target_calendar_idx is not None:
            last = last + self.calendar_embed(**target_calendar_idx)
        if self.head_mode in ("structured", "v4"):
            pred_norm, self._last_residual = self.head(last)
        else:
            pred_norm = self.head(last).view(B, N, self.patch_size, self.n_quantiles)
            pred_norm, _ = torch.sort(pred_norm, dim=-1)                              
        if self.norm_mode == "asinh_robust":
            return self.revin.denorm(pred_norm, mean.unsqueeze(-1), std.unsqueeze(-1))
        denorm_fn = AsinhRevIN.denorm if self.norm_mode == "asinh_revin" else RevIN.denorm
                                                                                              
        return denorm_fn(pred_norm, mean.unsqueeze(-1), std.unsqueeze(-1))

    def median(self, quantile_forecast: torch.Tensor) -> torch.Tensor:
        
        return quantile_forecast[..., self.median_idx]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
