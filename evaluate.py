














from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.adapt_data import AdaptDataset
from src.calendar_features import N_CALENDAR_FEATURES
from src.config import load_config
from src.dayahead_data import TEST_YEAR, load_iso_panel, origins_for_budget, origins_for_split, zone_series
from src.grid_adjacency import _neighbor_mask, learned_adjacency, true_adjacency
from src.grid_model import GridTransferModel
from src.grid_rollout import forecast_day_batch
from src.quantile_losses import pinball_loss, score_with_crps

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(PROJECT_DIR, 'results', 'checkpoints_grid3_calendar_heldout_4lay')
PRED_DIR = os.path.join(PROJECT_DIR, 'results', 'predictions', 'phase4_grid3_calendar_heldout_4lay')
CLIPPED_PANEL_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'panel_clipped.parquet')
RAW_PANEL_PATH = os.path.join(PROJECT_DIR, 'data', 'processed', 'panel.parquet')

ISOS = ['ISONE', 'NYISO', 'CAISO', 'MISO', 'ERCOT']
                                                                                           
                                                                                         
NO_COVARIATE_ISOS = {'ERCOT', 'ISONE'}
ADAPT_BATCH = 32
ADAPT_LR = 3e-4
CHECKPOINTS = [300, 600, 1000, 1400, 2000]

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s',
                    handlers=[logging.FileHandler(os.path.join(PROJECT_DIR, 'logs',
                               'eval_grid3_calendar_heldout_4lay_adaptcurve.log'), mode='a'),
                              logging.StreamHandler()])
log = logging.getLogger('adaptcurve')


def bias_for(panel, iso, zones, device):
    try:
        return true_adjacency(iso, zones).to(device)
    except NotImplementedError:
        return learned_adjacency(panel, iso, zones).to(device)


def evaluate(model, zs, panel, iso_cfg, with_covariates, device, graph_bias, split='test',
             truth_panel=None):
    

                                                                           
    origins = origins_for_split(panel, split, iso_cfg, year=TEST_YEAR if split == 'test' else None)
    frames = []
    model.eval()
    for gate, day_start, day_end in origins:
        pred = forecast_day_batch(model, zs, gate, day_start, day_end, device, with_covariates,
                                  graph_bias=graph_bias, calendar_tz=iso_cfg.tz)
        if not pred.empty:
            frames.append(pred)
    pred_all = pd.concat(frames, ignore_index=True)
    truth_panel = panel if truth_panel is None else truth_panel
    truth = truth_panel[['location', 'interval_start_utc', 'lmp']].rename(columns={'lmp': 'y_true'})
    pred_all = pred_all.merge(truth, on=['location', 'interval_start_utc'], how='left')
    q_cols = [c for c in pred_all.columns if c.startswith('q')]
    return pred_all[['location', 'interval_start_utc', 'y_true', 'y_pred'] + q_cols]


def score(pred):
    mae = float((pred['y_true'] - pred['y_pred']).abs().mean())
    q_cols = [c for c in pred.columns if c.startswith('q') and len(c) <= 4]
    if q_cols:
        s = score_with_crps(pred, q_cols, [float(c[1:]) for c in q_cols])
        aql = float(s.loc['AVERAGE', 'crps']) / 2
    else:
        aql = float('nan')
    return mae, aql


def scored(model, zs, panel, iso_cfg, with_covariates, device, graph_bias, split, out_path,
           truth_panel=None):
    
    if os.path.exists(out_path):
        pred = pd.read_parquet(out_path)
    else:
        pred = evaluate(model, zs, panel, iso_cfg, with_covariates, device, graph_bias, split=split,
                        truth_panel=truth_panel)
        pred.to_parquet(out_path, index=False)
    mae, aql = score(pred)
    return pred, mae, aql


def main():
    ap = argparse.ArgumentParser()
                                                                                                 
                                                                                             
                                        
    ap.add_argument('--isos', default=None, help='comma-separated subset of ISOS (default: all)')
    ap.add_argument('--tag', default=None,
                    help='suffix for the per-run curve/summary CSVs, so parallel runs do not clobber')
                                                                                                 
                                                                                              
                                          
    ap.add_argument('--checkpoints', default=None,
                    help='comma-separated adaptation-step ladder (default: 300,600,1000,1400,2000)')
                                                                                               
                                                                           
    ap.add_argument('--ckpt-dir', default=CKPT_DIR)
    ap.add_argument('--ckpt-prefix', default='grid3_calendar_heldout_4lay_')
    ap.add_argument('--ckpt-suffix', default='',
                    help="appended after the ISO, before .pt -- e.g. '_s1' for a seeded pretrain")
    ap.add_argument('--seed', type=int, default=0,
                    help='seeds evaluation-time adaptation; pair it with the pretrain seed so a '
                         'seed repeat varies both stages, not just pretraining')
    ap.add_argument('--stem', default='grid3cal_heldout_4lay',
                    help='middle of the prediction filenames, keeps arms from clobbering')
    args = ap.parse_args()
    isos = [x.strip() for x in args.isos.split(',')] if args.isos else ISOS
    if args.checkpoints:
        globals()['CHECKPOINTS'] = [int(x) for x in args.checkpoints.split(',')]

    os.makedirs(PRED_DIR, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg = load_config()
    curve_rows = []
    selected_rows = []

    for iso in isos:
        ckpt = os.path.join(args.ckpt_dir, f'{args.ckpt_prefix}{iso}{args.ckpt_suffix}.pt')
        if not os.path.exists(ckpt):
            log.warning('%s: no checkpoint, skipping', iso)
            continue

        iso_cfg = cfg.iso(iso)
        panel = load_iso_panel(iso, panel_path=CLIPPED_PANEL_PATH)
        truth_panel = load_iso_panel(iso, panel_path=RAW_PANEL_PATH)
        zs = zone_series(panel)
        graph_bias = bias_for(panel, iso, list(zs.keys()), device)

        state = torch.load(ckpt, map_location=device, weights_only=False)
                                                                                                  
                                                                                              
                                                                                                  
                                                                                           
        if state.get('cross_mode', 'masked') == 'graphaware':
                                                                                               
            graph_bias = _neighbor_mask(iso, list(zs.keys())).to(torch.float32)
            shuf = state.get('shuffle_adjacency', 0)
            if shuf:
                from scripts.pretrain_meta_grid3 import _permute_adjacency
                graph_bias = _permute_adjacency(graph_bias, shuf, iso)
            graph_bias = graph_bias.to(device)
        use_covariate = state.get('use_covariate', False)
        with_covariates = use_covariate and iso not in NO_COVARIATE_ISOS
        cov_tag = 'cov' if with_covariates else 'nocov'
                                                                                        
                                                                                            
                                                                                     
        model = GridTransferModel(patch_size=16, num_layers=4, use_covariate=use_covariate,
                                  graph_bias_mode=state.get('graph_bias_mode', 'true'),
                                  norm_mode=state.get('norm_mode', 'revin'),
                                  head_mode=state.get('head_mode', 'linear'),
                                  cross_mode=state.get('cross_mode', 'masked'),
                                  graph_init_logit=state.get('graph_init_logit', -4.0),
                                  use_adjacency=state.get('use_adjacency', True),
                                  use_correlation=state.get('use_correlation', True),
                                  corr_center_zones=state.get('corr_center_zones', False),
                                  n_calendar_features=N_CALENDAR_FEATURES).to(device)
        model.load_state_dict(state['model'])

        stem = f'{iso}_{args.stem}'
        _, mae0, aql0 = scored(model, zs, panel, iso_cfg, with_covariates, device, graph_bias,
                               'test', os.path.join(PRED_DIR, f'{stem}_b0_{cov_tag}_s0.parquet'),
                               truth_panel)
        _, vmae0, vaql0 = scored(model, zs, panel, iso_cfg, with_covariates, device, graph_bias,
                                 'val', os.path.join(PRED_DIR, f'{stem}_b0_{cov_tag}_s0_val.parquet'),
                                 truth_panel)
        log.info('%s step=0: val MAE=%.3f | test MAE=%.3f AQL=%.3f', iso, vmae0, mae0, aql0)
        curve_rows.append({'iso': iso, 'steps': 0, 'val_mae': vmae0, 'mae': mae0, 'aql': aql0})

                                                                                  
        origins_budget = origins_for_budget(panel, iso_cfg, 'train')
        budget_days = sorted({o[1].tz_convert(iso_cfg.tz).normalize() for o in origins_budget})
        ds = AdaptDataset(zs, budget_days, seed=0, tz=iso_cfg.tz)

        torch.manual_seed(args.seed)
        opt = torch.optim.AdamW(model.parameters(), lr=ADAPT_LR)
        max_steps = CHECKPOINTS[-1]
        warmup_steps = max(1, int(0.05 * max_steps))

        def lr_lambda(step):
            if step < warmup_steps:
                return (step + 1) / warmup_steps
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            return 0.5 * (1 + np.cos(np.pi * min(progress, 1.0)))

        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
        prev_ckpt = 0

        for ckpt_step in CHECKPOINTS:
            out_path = os.path.join(PRED_DIR, f'{stem}_adapt{ckpt_step}_{cov_tag}_s0.parquet')
            val_path = os.path.join(PRED_DIR, f'{stem}_adapt{ckpt_step}_{cov_tag}_s0_val.parquet')
            if os.path.exists(out_path) and os.path.exists(val_path):
                mae, aql = score(pd.read_parquet(out_path))
                vmae, _ = score(pd.read_parquet(val_path))
                for _ in range(prev_ckpt, ckpt_step):
                    sched.step()
                prev_ckpt = ckpt_step
                log.info('%s step=%d: val MAE=%.3f | test MAE=%.3f AQL=%.3f (cached)',
                         iso, ckpt_step, vmae, mae, aql)
                curve_rows.append({'iso': iso, 'steps': ckpt_step, 'val_mae': vmae,
                                   'mae': mae, 'aql': aql})
                continue

            model.train()
            for step in range(prev_ckpt, ckpt_step):
                batch = ds.sample_batch(ADAPT_BATCH, with_covariates, include_calendar=True)
                target = batch['target'].to(device)
                target_next = batch['target_next'].to(device)
                cov_t = batch.get('covariate')
                cov_t = cov_t.to(device) if cov_t is not None else None
                mask = batch.get('cov_mask')
                mask = mask.to(device) if mask is not None else None
                cal = batch['calendar'].to(device)
                pred_t = model(target, cov_t, mask, graph_bias=graph_bias, calendar=cal)
                loss = pinball_loss(pred_t, target_next, model.quantiles)
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step()
            prev_ckpt = ckpt_step

            _, vmae, _ = scored(model, zs, panel, iso_cfg, with_covariates, device, graph_bias,
                                'val', val_path, truth_panel)
            _, mae, aql = scored(model, zs, panel, iso_cfg, with_covariates, device, graph_bias,
                                 'test', out_path, truth_panel)
            log.info('%s step=%d: val MAE=%.3f | test MAE=%.3f AQL=%.3f', iso, ckpt_step, vmae, mae, aql)
            curve_rows.append({'iso': iso, 'steps': ckpt_step, 'val_mae': vmae, 'mae': mae, 'aql': aql})

                                                                            
        candidates = curve_rows[-(len(CHECKPOINTS) + 1):]
        best = min(candidates, key=lambda r: r['val_mae'])
        suffix = 'b0' if best['steps'] == 0 else f'adapt{best["steps"]}'
        best_path = os.path.join(PRED_DIR, f'{stem}_{suffix}_{cov_tag}_s0.parquet')
        import shutil
        shutil.copy(best_path, os.path.join(PRED_DIR, f'{stem}_bfull_{cov_tag}_s0.parquet'))
        log.info('%s: selected step=%d on val (val MAE=%.3f) -> test MAE=%.3f, copied to bfull',
                 iso, best['steps'], best['val_mae'], best['mae'])
        selected_rows.append({'iso': iso, 'selected_steps': best['steps'],
                              'val_mae': best['val_mae'], 'test_mae': best['mae'],
                              'test_aql': best['aql']})

    df = pd.DataFrame(curve_rows)
    for label, col in (('val MAE', 'val_mae'), ('test MAE', 'mae'), ('test AQL', 'aql')):
        print(f'\n=== Adaptation curve ({label}) ===')
        print(df.pivot_table(index='steps', columns='iso', values=col).round(3).to_string())

    sel = pd.DataFrame(selected_rows)
    print('\n=== Selected checkpoints (chosen on val, reported on test) ===')
    print(sel.round(3).to_string(index=False))
    print(f"\n{len(sel)}-ISO average test MAE: {sel['test_mae'].mean():.3f}  "
          f"AQL: {sel['test_aql'].mean():.3f}")

    table_dir = os.path.join(PROJECT_DIR, 'results', 'tables')
    os.makedirs(table_dir, exist_ok=True)
    tag = f'_{args.tag}' if args.tag else ''
    df.to_csv(os.path.join(table_dir, f'adaptcurve{tag}.csv'), index=False)
    sel.to_csv(os.path.join(table_dir, f'adaptcurve_selected{tag}.csv'), index=False)
    log.info('wrote %s', os.path.join(table_dir, f'adaptcurve_selected{tag}.csv'))


if __name__ == '__main__':
    main()
