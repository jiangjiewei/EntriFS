#!/usr/bin/env python3
"""Ablation: MC-Dropout T values and unfreezing depth."""
import json
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from config import OUTPUT_DIR, APTOS_IMG_DIR, WEIGHT_BASELINE, WEIGHT_OURS, CALIB_EPOCHS
from common import split_aptos_calib_test, APTOSDataset, get_calib_transform, get_test_transform
from models import build_resnet50, set_trainable, optimizer_for_mode
from inference import mc_dropout_predict, metrics


def finetune(mode, epochs=CALIB_EPOCHS):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df_calib, _ = split_aptos_calib_test()
    loader = DataLoader(
        APTOSDataset(df_calib, APTOS_IMG_DIR, get_calib_transform()),
        batch_size=10, shuffle=True, num_workers=2,
    )
    base = WEIGHT_OURS if WEIGHT_OURS.exists() else WEIGHT_BASELINE
    model = build_resnet50(pretrained=False)
    model.load_state_dict(torch.load(base, map_location=device))
    model = model.to(device)
    set_trainable(model, mode)
    opt = optimizer_for_mode(model, mode)
    crit = nn.CrossEntropyLoss()
    model.train()
    for _ in range(epochs):
        for imgs, labels, _ in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            opt.zero_grad()
            crit(model(imgs), labels).backward()
            opt.step()
    path = OUTPUT_DIR / 'ablations' / f'weights_{mode}.pth'
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def evaluate(weight_path, t_samples):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, df_test = split_aptos_calib_test()
    loader = DataLoader(
        APTOSDataset(df_test, APTOS_IMG_DIR, get_test_transform()),
        batch_size=32, shuffle=False, num_workers=4,
    )
    model = build_resnet50(pretrained=False)
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model = model.to(device)
    df = mc_dropout_predict(model, loader, device, t_samples=t_samples)
    acc, qwk, _, _ = metrics(df)
    return acc, qwk


def main():
    out = OUTPUT_DIR / 'ablations'
    out.mkdir(parents=True, exist_ok=True)

    weight = WEIGHT_OURS if WEIGHT_OURS.exists() else WEIGHT_BASELINE
    if not weight.exists():
        print('Run scripts/02_run_pipeline.py first.')
        sys.exit(1)

    t_rows = []
    for t in [5, 10, 20, 30, 50]:
        acc, qwk = evaluate(weight, t)
        t_rows.append({'T': t, 'Accuracy (%)': round(acc, 2), 'QWK': round(qwk, 4)})
        print(f'  T={t:2d} -> Acc={acc:.2f}%  QWK={qwk:.4f}')
    pd.DataFrame(t_rows).to_csv(out / 'ablation_t_values.csv', index=False)

    l_rows = []
    for mode, label in [
        ('fc_only', 'FC only'),
        ('layer4_fc', 'Layer4 + FC'),
        ('layer34_fc', 'Layer3-4 + FC'),
        ('all', 'All layers'),
    ]:
        print(f'[*] Finetuning: {label}')
        w = finetune(mode)
        acc, qwk = evaluate(w, 30)
        l_rows.append({
            'Unfrozen Layers': label, 'Accuracy (%)': round(acc, 2),
            'QWK': round(qwk, 4), 'Calib Samples': 50,
        })
        print(f'  {label} -> Acc={acc:.2f}%  QWK={qwk:.4f}')
    pd.DataFrame(l_rows).to_csv(out / 'ablation_unfreeze_layers.csv', index=False)

    with open(out / 'ablation_summary.json', 'w') as f:
        json.dump({'t_ablation': t_rows, 'layer_ablation': l_rows}, f, indent=2)
    print(f'[+] Saved to {out}')


if __name__ == '__main__':
    main()
