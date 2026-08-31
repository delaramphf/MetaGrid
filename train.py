



from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.calendar_features import N_CALENDAR_FEATURES
from src.grid_adjacency import learned_adjacency, true_adjacency
from src.grid_model import GridTransferModel
from src.pretrain_data import HeldOutPretrainDataset
from src.quantile_losses import pinball_loss

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(PROJECT_DIR, 'results', 'checkpoints_grid3_calendar_heldout_4lay')
LOG_DIR = os.path.join(PROJECT_DIR, 'logs')

CONTEXT_LEN = 336
PATCH_SIZE = 16
BATCH_SIZE = 64


def ckpt_path(held_out, ckpt_dir=None, stem=None):
    d = ckpt_dir or CKPT_DIR
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f'{stem or "grid3_calendar_heldout_4lay"}_{held_out}.pt')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--held-out', required=True)
                                                                                                
                                                                                            
                                                     
    ap.add_argument('--ckpt-dir', default=CKPT_DIR)
    ap.add_argument('--steps', type=int, default=2000)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--log-every', type=int, default=50)
    ap.add_argument('--use-covariate', action='store_true', default=False)
                                                                                      
                                                                                              
    ap.add_argument('--cross-mode', default='masked', choices=['masked', 'graphaware'])
    ap.add_argument('--head-mode', default='linear', choices=['linear', 'structured', 'v4'])
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--stem', default=None,
                    help='checkpoint stem; defaults to the historical name')
    args = ap.parse_args()

    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    log_tag = f'{args.held_out}_grid3_calendar_heldout_4lay'
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s',
                        handlers=[logging.FileHandler(os.path.join(LOG_DIR, f'pretrain_{log_tag}.log'), mode='a'),
                                  logging.StreamHandler()])
    log = logging.getLogger('pretrain_grid3_calendar_heldout')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    log.info('held_out=%s device=%s include_target_train=False include_calendar=True', args.held_out, device)

    import pandas as pd
    from src.config import load_config
    cfg = load_config()
    panel = pd.read_parquet(os.path.join(PROJECT_DIR, 'data', 'processed', 'panel_clipped.parquet'))
    ds = HeldOutPretrainDataset(panel, args.held_out, CONTEXT_LEN, PATCH_SIZE, cfg=cfg,
                                include_target_train=False, include_calendar=True)
    log.info('source ISOs (excl. %s train): %s', args.held_out, ds.isos)

    bias_cache = {}
    for iso in ds.isos:
        zones = list(ds.iso_zones[iso].keys())
        try:
            bias_cache[iso] = true_adjacency(iso, zones).to(device)
        except NotImplementedError as e:
            log.warning('no true adjacency for %s (%s) -- falling back to learned', iso, e)
            bias_cache[iso] = learned_adjacency(panel, iso, zones).to(device)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = GridTransferModel(patch_size=PATCH_SIZE, num_layers=4, graph_bias_mode='true',
                              use_covariate=args.use_covariate,
                              cross_mode=args.cross_mode, head_mode=args.head_mode,
                              n_calendar_features=N_CALENDAR_FEATURES).to(device)
    n_params = model.num_parameters()
    log.info('model parameters: %d (%.1fM)', n_params, n_params / 1e6)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    warmup_steps = max(1, int(0.05 * args.steps))

    def lr_lambda(step):
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, args.steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    start_step = 0
    curve = []
    cpath = ckpt_path(args.held_out, args.ckpt_dir, args.stem)
    if os.path.exists(cpath):
        ck = torch.load(cpath, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        opt.load_state_dict(ck['opt'])
        start_step = ck['step']
        curve = ck.get('curve', [])
        for _ in range(start_step):
            sched.step()
        log.info('resumed from step %d', start_step)

    model.train()
    t0 = time.time()
    running = 0.0
    for step in range(start_step, args.steps):
        batch = ds.sample_batch(BATCH_SIZE)
        target = batch['target'].to(device)
        future = batch['future'].to(device)
        bias = bias_cache.get(batch['iso'])
        n_future_patches = batch['n_future_patches']
        context_len = target.shape[-1]
        if args.use_covariate:
            cov_full = batch['covariate_full'].to(device)
            mask_full = batch['cov_mask_full'].to(device)
        cal_full = batch['calendar_full'].to(device)

        full_target = torch.cat([target, future.reshape(future.shape[0], future.shape[1], -1)], dim=-1)
        step_losses = []
        for p in range(n_future_patches):
            lo = p * model.patch_size
            ctx_p = full_target[:, :, lo:lo + context_len]
            cov_p = mask_p = None
            if args.use_covariate:
                cov_p = cov_full[:, :, lo:lo + context_len]
                mask_p = mask_full[:, :, lo:lo + context_len]
            cal_p = cal_full[:, :, :, lo:lo + context_len]
            pred = model(ctx_p, cov_p, mask_p, graph_bias=bias, calendar=cal_p)
            step_losses.append(pinball_loss(pred, future[:, :, p, :], model.quantiles))
        loss = torch.stack(step_losses).mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        running += loss.item()

        if (step + 1) % args.log_every == 0:
            avg = running / args.log_every
            elapsed = time.time() - t0
            log.info('step %d/%d loss=%.4f lr=%.2e (%.3fs/step)', step + 1, args.steps, avg,
                     sched.get_last_lr()[0], elapsed / (step + 1 - start_step))
            curve.append({'step': step + 1, 'loss': avg})
            running = 0.0
            torch.save({'model': model.state_dict(), 'opt': opt.state_dict(),
                        'step': step + 1, 'curve': curve,
                        'graph_bias_mode': 'true', 'use_covariate': args.use_covariate,
                        'cross_mode': args.cross_mode, 'head_mode': args.head_mode,
                        'seed': args.seed,
                        'n_calendar_features': N_CALENDAR_FEATURES,
                        'include_target_train': False}, cpath)

    total_time = time.time() - t0
    per_step = total_time / max(1, args.steps - start_step)
    log.info('done: %d steps in %.1fs (%.3fs/step)', args.steps - start_step, total_time, per_step)

    summary = {'held_out': args.held_out, 'n_params': n_params, 'steps': args.steps,
               'seconds': total_time, 'include_target_train': False, 'include_calendar': True,
               'use_covariate': args.use_covariate, 'source_isos': ds.isos}
    with open(cpath.replace('.pt', '_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
