#!/usr/bin/env python3
"""
Full reproduction pipeline (DDR -> APTOS cross-domain DR grading):
  1. Source training on DDR
  2. Direct cross-domain baseline (full APTOS)
  3. Linear probe (FC-only, 50-shot calib)
  4. Ours (Layer4+FC calib + MC-Dropout blind test)
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from config import (
    OUTPUT_DIR, APTOS_LABELS, APTOS_IMG_DIR, WEIGHT_BASELINE,
    WEIGHT_LINEAR, WEIGHT_OURS, SOURCE_EPOCHS, CALIB_EPOCHS, BATCH_SIZE,
)
from common import (
    split_aptos_calib_test, APTOSDataset, get_calib_transform, get_test_transform,
    build_ddr_train_dataset, get_ddr_train_transform,
)
from models import build_resnet50, set_trainable, optimizer_for_mode
from inference import mc_dropout_predict, metrics


def finetune(base_weights, mode, out_path, epochs=CALIB_EPOCHS):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df_calib, _ = split_aptos_calib_test()
    loader = DataLoader(
        APTOSDataset(df_calib, APTOS_IMG_DIR, get_calib_transform()),
        batch_size=10, shuffle=True, num_workers=2,
    )
    model = build_resnet50(pretrained=False)
    model.load_state_dict(torch.load(base_weights, map_location=device))
    model = model.to(device)
    set_trainable(model, mode)
    opt = optimizer_for_mode(model, mode)
    crit = nn.CrossEntropyLoss()
    model.train()
    print(f'[*] Calibrate mode={mode}, n={len(df_calib)}, epochs={epochs}')
    for _ in range(epochs):
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            crit(model(imgs), labels).backward()
            opt.step()
    torch.save(model.state_dict(), out_path)
    print(f'[+] Saved {out_path}')
    return out_path


def eval_weights(weights, csv_name, test_df=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if test_df is None:
        _, test_df = split_aptos_calib_test()
    loader = DataLoader(
        APTOSDataset(test_df, APTOS_IMG_DIR, get_test_transform()),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=4,
    )
    model = build_resnet50(pretrained=False)
    model.load_state_dict(torch.load(weights, map_location=device))
    model = model.to(device)
    df = mc_dropout_predict(model, loader, device)
    path = OUTPUT_DIR / csv_name
    df.to_csv(path, index=False)
    acc, qwk, f1, n = metrics(df, 0.0)
    acc30, qwk30, _, _ = metrics(df, 0.3)
    print(f'  {csv_name}: n={n} Acc={acc:.2f}% QWK={qwk:.4f} | @reject30% Acc={acc30:.2f}%')
    return df


def train_source_if_missing(epochs):
    if WEIGHT_BASELINE.exists():
        print(f'[*] Using existing {WEIGHT_BASELINE}')
        return
    import subprocess
    subprocess.check_call([
        sys.executable, str(ROOT / 'scripts' / '01_train_source.py'),
        '--epochs', str(epochs),
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-train', action='store_true', help='Require existing source weights')
    parser.add_argument('--source-epochs', type=int, default=SOURCE_EPOCHS)
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[*] Device: {device}')

    if args.skip_train:
        if not WEIGHT_BASELINE.exists():
            raise FileNotFoundError(f'Missing {WEIGHT_BASELINE}')
    else:
        train_source_if_missing(args.source_epochs)

    # Baseline: full APTOS, no adaptation
    df_all = pd.read_csv(APTOS_LABELS)
    print('[*] Direct cross-domain baseline (full APTOS)')
    eval_weights(WEIGHT_BASELINE, 'aptos_cross_domain_baseline.csv', test_df=df_all)

    # Linear probe
    finetune(WEIGHT_BASELINE, 'fc_only', WEIGHT_LINEAR)
    print('[*] Linear probe blind test')
    eval_weights(WEIGHT_LINEAR, 'aptos_linear_probe.csv')

    # Ours
    finetune(WEIGHT_BASELINE, 'layer4_fc', WEIGHT_OURS)
    print('[*] Ours blind test')
    eval_weights(WEIGHT_OURS, 'aptos_ours_blind.csv')

    print('\n[+] Pipeline complete. Results in outputs/')


if __name__ == '__main__':
    main()
