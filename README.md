# Cross-Domain DR Grading with Few-Shot Calibration & MC-Dropout

Few-shot domain adaptation for **diabetic retinopathy (DR) 5-class grading**: train on **DDR**, adapt with **50 labeled APTOS** images (10/class), evaluate on a **3,612-sample blind test** with **MC-Dropout** selective classification.

## Requirements

- Python 3.8+
- CUDA GPU recommended
- PyTorch, see `requirements.txt`

```bash
pip install -r requirements.txt
```

## Data preparation

Follow [data/README.md](data/README.md). Summary:

| Variable       | Default path                  |
| -------------- | ----------------------------- |
| `DDR_ROOT`     | `data/ddr`                    |
| `APTOS_IMAGES` | `data/aptos/train_images`     |
| `APTOS_LABELS` | `data/labels/aptos_train.csv` |

Override with environment variables (see `src/config.py`).

```bash
export DDR_ROOT=/path/to/DDR_Final_Ready_256_del_1
export APTOS_IMAGES=/path/to/aptos/train_images
```

## Reproduce experiments

```bash
# Full pipeline: source train → baseline → linear probe → ours
python scripts/02_run_pipeline.py

# Source training only
python scripts/01_train_source.py --epochs 15

# Ablations (T values, unfreezing depth)
python scripts/03_run_ablations.py
```

Outputs are written to `outputs/` (git-ignored):

| File                           | Description                  |
| ------------------------------ | ---------------------------- |
| `resnet50_ddr_baseline.pth`    | DDR source weights           |
| `resnet50_linear_probe.pth`    | FC-only calibration          |
| `resnet50_deep_calibrated.pth` | Layer4+FC calibration (Ours) |
| `aptos_*.csv`                  | Per-method prediction CSVs   |
| `ablations/`                   | T / unfreeze ablation tables |

## Method overview

1. **Source training** — ResNet-50 on DDR `train/`, 15 epochs, Dropout 0.5  
2. **ROI crop** — remove fundus black borders (`crop_image_from_gray`)  
3. **Few-shot calibration** — Layer4+FC finetune, 12 epochs, 50 APTOS labels  
4. **Inference** — MC-Dropout T=30, predictive entropy for selective rejection  

## Project layout

```
├── src/           # Core library (config, data, models, inference)
├── scripts/       # Runnable experiment entry points
├── tools/figures/ # Optional: paper figures & tables (needs outputs/)
├── configs/       # Default hyper-parameters
├── data/          # Labels + README (images not included)
└── outputs/       # Weights & CSV results (generated)
```
