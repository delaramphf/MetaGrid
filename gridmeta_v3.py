

















from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_QUANTILES = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def context_correlation(x: torch.Tensor, shrink: float = 0.1,
                        center_zones: bool = False) -> torch.Tensor:
    











    if center_zones:
        x = x - x.mean(dim=1, keepdim=True)                            
    xc = x - x.mean(dim=-1, keepdim=True)
    xn = xc / xc.std(dim=-1, keepdim=True).clamp_min(1e-5)
    c = torch.einsum('bnl,bml->bnm', xn, xn) / x.shape[-1]
    eye = torch.eye(x.shape[1], device=x.device, dtype=x.dtype).unsqueeze(0)
    return (1.0 - shrink) * c.clamp(-1.0, 1.0) + shrink * eye


class GraphAwareCrossZone(nn.Module):
    

    def __init__(self, d_model: int, n_heads: int = 6, dropout: float = 0.1,
                 gamma: float = 2.0, init_logit: float = -4.0,
                 use_adjacency: bool = True, use_correlation: bool = True):
        super().__init__()
        assert d_model % n_heads == 0
        self.h, self.dk, self.gamma = n_heads, d_model // n_heads, gamma
        self.qp = nn.Linear(d_model, d_model)
        self.kp = nn.Linear(d_model, d_model)
        self.vp = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
                                                                                                 
                                                                          
        self.a_logit = nn.Parameter(torch.tensor(init_logit))
        self.b_logit = nn.Parameter(torch.tensor(init_logit))
                                                                                        
                                                                                          
        self.use_adjacency = use_adjacency
        self.use_correlation = use_correlation
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

    def trust(self):
        a = self.gamma * torch.sigmoid(self.a_logit) if self.use_adjacency else self.a_logit.new_zeros(())
        b = self.gamma * torch.sigmoid(self.b_logit) if self.use_correlation else self.b_logit.new_zeros(())
        return a, b

    def forward(self, x, adj, corr):
        
        B, N, P, D = x.shape
        xt = x.permute(0, 2, 1, 3).reshape(B * P, N, D)
        q = self.qp(xt).view(B * P, N, self.h, self.dk).transpose(1, 2)
        k = self.kp(xt).view(B * P, N, self.h, self.dk).transpose(1, 2)
        v = self.vp(xt).view(B * P, N, self.h, self.dk).transpose(1, 2)
        s = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)

        alpha, beta = self.trust()
        if adj is not None and self.use_adjacency:
            s = s + alpha * adj.to(s.dtype).view(1, 1, N, N)
        if corr is not None and self.use_correlation:
            s = s + beta * corr.to(s.dtype).repeat_interleave(P, dim=0).unsqueeze(1)

        o = (self.drop(torch.softmax(s, dim=-1)) @ v).transpose(1, 2).reshape(B * P, N, D)
        o = self.norm(self.drop(self.out(o)) + xt)
        return o.view(B, P, N, D).permute(0, 2, 1, 3)


class StructuredHead(nn.Module):
    





    def __init__(self, d_model: int, patch_size: int, n_quantiles: int, hidden: int = 256,
                 n_cal: int = 0):
        super().__init__()
        self.patch, self.n_q = patch_size, n_quantiles
        self.median_idx = n_quantiles // 2
        self.n_cal = n_cal
        if n_cal:
                                                                                              
                                                                                              
                                                             
            self.cal_proj = nn.Linear(n_cal * patch_size, d_model)
            nn.init.zeros_(self.cal_proj.weight)
            nn.init.zeros_(self.cal_proj.bias)
        self.common = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, patch_size))
        self.residual = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, patch_size))
        self.spread = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(),
                                    nn.Linear(hidden, patch_size * (n_quantiles - 1)))

    def forward(self, z, e=None):
        
        if self.n_cal and e is not None:
            z = z + self.cal_proj(e.flatten(2))
        B, N, D = z.shape
        m = self.common(z.mean(dim=1)).unsqueeze(1)                                           
        r_t = self.residual(z)                                                   
        r = r_t - r_t.mean(dim=1, keepdim=True)                                              
        median = m + r
        d = F.softplus(self.spread(z)).view(B, N, self.patch, self.n_q - 1) + 1e-4
        lo = median.unsqueeze(-1) - torch.flip(
            torch.cumsum(torch.flip(d[..., :self.median_idx], dims=[-1]), dim=-1), dims=[-1])
        hi = median.unsqueeze(-1) + torch.cumsum(d[..., self.median_idx:], dim=-1)
        return torch.cat([lo, median.unsqueeze(-1), hi], dim=-1), r



class LoRA(nn.Module):
    






    def __init__(self, d_in: int, d_out: int, rank: int):
        super().__init__()
        self.A = nn.Linear(d_in, rank, bias=False)
        self.B = nn.Linear(rank, d_out, bias=False)
        nn.init.normal_(self.A.weight, std=0.02)
        nn.init.zeros_(self.B.weight)

    def forward(self, x):
        return self.B(self.A(x))


class GridMetaV3(nn.Module):
    

    def __init__(self, patch_size: int = 16, d_model: int = 384, n_heads: int = 6,
                 num_layers: int = 4, ff_dim: int = 1536, use_covariate: bool = False,
                 dropout: float = 0.1, quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
                 n_calendar_features: int = 0, gamma: float = 2.0, corr_shrink: float = 0.1,
                 lora_rank: int = 0, norm_mode: str = 'market'):
        super().__init__()
        assert norm_mode in ('market', 'revin', 'zoneloc')
        self.norm_mode = norm_mode
        self.patch_size = patch_size
        self.quantiles = quantiles
        self.n_quantiles = len(quantiles)
        self.median_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
        self.use_covariate = use_covariate
        self.n_calendar_features = n_calendar_features
        self.corr_shrink = corr_shrink

        in_dim = patch_size * (2 if use_covariate else 1) + patch_size * n_calendar_features
        self.proj = nn.Linear(in_dim, d_model)
        self.mask_token = nn.Parameter(torch.zeros(patch_size))
        nn.init.normal_(self.mask_token, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, ff_dim, dropout=dropout,
                                           batch_first=True, activation='gelu')
        self.temporal_encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.cross_series = GraphAwareCrossZone(d_model, n_heads, dropout, gamma=gamma)
        self.head = StructuredHead(d_model, patch_size, self.n_quantiles,
                                   n_cal=n_calendar_features)
        self._last_residual = None
        self.lora_rank = lora_rank
        self.lora = nn.ModuleDict()
        if lora_rank > 0:
            self._attach_lora(lora_rank)

    def _attach_lora(self, rank: int):
        
        targets = []
        for i, layer in enumerate(self.temporal_encoder.layers):
            targets += [(f'enc{i}_attnout', layer.self_attn.out_proj),
                        (f'enc{i}_fc1', layer.linear1),
                        (f'enc{i}_fc2', layer.linear2)]
        targets += [('cz_out', self.cross_series.out)]
        for name, mod in targets:
            lo = LoRA(mod.in_features, mod.out_features, rank)
            self.lora[name] = lo
            mod.register_forward_hook(
                lambda m, inp, out, _lo=lo: out + _lo(inp[0]))

    def market_norm(self, x):
        










        if self.norm_mode == 'revin':
            loc = x.mean(dim=-1, keepdim=True)
            scale = x.std(dim=-1, keepdim=True).clamp_min(1e-5)
        elif self.norm_mode == 'zoneloc':
            loc = x.mean(dim=-1, keepdim=True)                                       
            scale = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)                  
        else:
            loc = x.mean(dim=(1, 2), keepdim=True)
            scale = x.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
        return (x - loc) / scale, loc, scale

    def _patches(self, x):
        B, N, L = x.shape
        return x.view(B, N, L // self.patch_size, self.patch_size)

    def forward(self, target, covariate=None, cov_mask=None, graph_bias=None, calendar=None,
                calendar_future=None, **kw):
        
        xn, loc, scale = self.market_norm(target)
        corr = context_correlation(xn, self.corr_shrink)

        tp = self._patches(xn)
        feats = [tp]
        if self.use_covariate:
            if covariate is None:
                feats.append(self.mask_token.expand_as(tp))
            else:
                cn = (covariate - covariate.mean(-1, keepdim=True)) / covariate.std(
                    -1, keepdim=True).clamp_min(1e-5)
                cp = self._patches(cn)
                if cov_mask is not None:
                    ok = self._patches(cov_mask.float()).mean(-1, keepdim=True) > 0.5
                    cp = torch.where(ok, cp, self.mask_token.expand_as(cp))
                feats.append(cp)
        if self.n_calendar_features:
            feats += [self._patches(calendar[:, :, c, :]) for c in range(self.n_calendar_features)]
        tok = self.proj(torch.cat(feats, dim=-1))

        B, N, P, D = tok.shape
        h = self.temporal_encoder(tok.reshape(B * N, P, D)).view(B, N, P, D)
        h = self.cross_series(h, graph_bias, corr)
        z = h[:, :, -1, :]                                                          

        q, r = self.head(z, calendar_future)
        self._last_residual = r
        return q * scale.unsqueeze(-1) + loc.unsqueeze(-1)

    def median(self, q):
        return q[..., self.median_idx]

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters())


def smoothness(r: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
    
    diff = r.unsqueeze(2) - r.unsqueeze(1)                            
    w = adj.to(diff.dtype).unsqueeze(0).unsqueeze(-1)
    wi = adj.sum().clamp_min(1e-6)
    return (w * diff.pow(2)).sum(dim=(1, 2)).mean() / wi


                                                                                               
                                                                                    
ADAPTER_SUFFIXES = ('norm1.weight', 'norm1.bias', 'norm2.weight', 'norm2.bias',
                    'cross_series.norm.weight', 'cross_series.norm.bias',
                    'a_logit', 'b_logit')
ADAPTER_PREFIXES = ('head.', 'lora.')


def adapter_parameters(model):
    names, params = [], []
    for n, p in model.named_parameters():
        if n.endswith(ADAPTER_SUFFIXES) or n.startswith(ADAPTER_PREFIXES):
            names.append(n); params.append(p)
    if not params:
        raise RuntimeError('no adapter parameters matched')
    return names, params
