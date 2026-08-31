# MetaGrid

Code for topology-aware transfer learning in day-ahead electricity price forecasting.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Place the public ISO files under `data/raw/<ISO>/`, then build the hourly panel:

```bash
python scripts/build_panel.py
python scripts/clip_panel.py
```

The configuration in `configs/isos.yaml` defines the input layout, market time zones, forecast gates, and chronological splits.

## Training

Pretrain on the source markets while holding out one target:

```bash
python scripts/train.py --heldout ERCOT
```

Evaluate zero-shot and after target-market adaptation:

```bash
python scripts/evaluate.py --iso ERCOT
```

Outputs are written to `results/`. Repeat the two commands with each target ISO used in the study.

## Layout

- `src/`: model and data pipeline
- `scripts/`: panel construction, training, and evaluation
- `configs/`: market metadata and graph structure

Data, checkpoints, and generated results are not included.
