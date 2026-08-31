



from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import load_config
from src.panel import build_panel

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PANEL_PATH = os.path.join(PROJECT_DIR, "data", "processed", "panel.parquet")
MISO_ZONAL_PATH = os.path.join(PROJECT_DIR, "data", "processed", "miso_zonal_load.parquet")
QA_REPORT_PATH = os.path.join(PROJECT_DIR, "reports", "data_qa.md")
LOG_PATH = os.path.join(PROJECT_DIR, "logs", "build_panel.log")


def setup_logging():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(LOG_PATH, mode="w"), logging.StreamHandler()],
    )


def price_stats(s: pd.Series) -> dict:
    s = s.dropna()
    pcts = s.quantile([0.01, 0.05, 0.50, 0.95, 0.99])
    return {
        "n": len(s), "mean": s.mean(), "std": s.std(), "min": s.min(), "max": s.max(),
        "p01": pcts[0.01], "p05": pcts[0.05], "p50": pcts[0.50], "p95": pcts[0.95], "p99": pcts[0.99],
        "negative_share": (s < 0).mean(),
    }


def write_qa_report(panel: pd.DataFrame, miso_zonal: pd.DataFrame, violations: pd.DataFrame) -> str:
    lines = ["# Data QA report\n"]

    lines.append("## Rows per ISO per year\n")
    yr = panel["interval_start_utc"].dt.year
    tab = panel.groupby(["iso", yr.rename("year")], observed=True).size().unstack(fill_value=0)
    lines.append(tab.to_string())

    lines.append("\n## Location counts and names\n")
    for iso, g in panel.groupby("iso", observed=True):
        locs = sorted(g["location"].unique().astype(str))
        lines.append(f"- **{iso}** ({len(locs)}): {locs}")

    lines.append("\n## Missing-hour / duplicate-timestamp violations (per (iso, location))\n")
    if violations.empty:
        lines.append("None -- every (iso, location) has a strictly increasing, dup-free, gap-free hourly UTC index.")
    else:
        lines.append(violations.to_string(index=False))

    lines.append("\n## Null share per column, per ISO\n")
    null_cols = ["lmp", "energy", "congestion", "loss", "load_forecast"]
    null_tab = panel.groupby("iso", observed=True)[null_cols].apply(lambda g: g.isna().mean())
    lines.append(null_tab.to_string())

    lines.append("\n## Price statistics per ISO ($/MWh, `lmp`)\n")
    rows = []
    for iso, g in panel.groupby("iso", observed=True):
        st = price_stats(g["lmp"])
        st["iso"] = iso
        rows.append(st)
    price_tab = pd.DataFrame(rows).set_index("iso")[
        ["n", "mean", "std", "min", "p01", "p05", "p50", "p95", "p99", "max", "negative_share"]]
    lines.append(price_tab.to_string())

    lines.append("\n## PJM |congestion| distribution\n")
    pjm_cong = panel.loc[panel["iso"] == "PJM", "congestion"].dropna().abs()
    if len(pjm_cong):
        lines.append(f"- n = {len(pjm_cong):,}")
        for thresh in (1, 5, 10):
            share = (pjm_cong > thresh).mean()
            lines.append(f"- share of hours with |congestion| > ${thresh}/MWh: {share:.4f}")
    else:
        lines.append("No PJM congestion data.")

    lines.append("\n## MISO zonal load (preserved separately, not in main panel)\n")
    lines.append(f"- {len(miso_zonal):,} rows, columns: {list(miso_zonal.columns)}")
    lines.append(f"- range: {miso_zonal['interval_start_utc'].min()} -> {miso_zonal['interval_start_utc'].max()}")

    return "\n".join(str(l) for l in lines)


def main():
    ap = argparse.ArgumentParser()
                                                                                           
                                                                                        
                               
    ap.add_argument("--isos", default=None,
                    help="comma-separated subset of configs/isos.yaml to build (default: all)")
    args = ap.parse_args()

    setup_logging()
    log = logging.getLogger("build_panel")

    cfg = load_config()
    if args.isos:
        want = [s.strip() for s in args.isos.split(",") if s.strip()]
        unknown = [w for w in want if w not in cfg.iso_names]
        if unknown:
            raise SystemExit(f"--isos names {unknown}, which are not in configs/isos.yaml "
                             f"({list(cfg.iso_names)})")
        cfg = dataclasses.replace(cfg, isos=tuple(cfg.iso(w) for w in want))
    log.info("config-driven ISO list: %s", cfg.iso_names)

    panel, miso_zonal, violations = build_panel(cfg)

    os.makedirs(os.path.dirname(PANEL_PATH), exist_ok=True)
    panel.to_parquet(PANEL_PATH, index=False)
    miso_zonal.to_parquet(MISO_ZONAL_PATH, index=False)
    log.info("wrote %s (%d rows), %s (%d rows)", PANEL_PATH, len(panel), MISO_ZONAL_PATH, len(miso_zonal))

    report = write_qa_report(panel, miso_zonal, violations)
    os.makedirs(os.path.dirname(QA_REPORT_PATH), exist_ok=True)
    with open(QA_REPORT_PATH, "w") as f:
        f.write(report)
    log.info("wrote %s", QA_REPORT_PATH)
    print(report)


if __name__ == "__main__":
    main()
