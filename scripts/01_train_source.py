#!/usr/bin/env python3
"""Stage 1: Train ResNet-50 on DDR source domain (train split, paper protocol)."""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from config import OUTPUT_DIR, WEIGHT_BASELINE, SOURCE_EPOCHS, BATCH_SIZE
from common import build_ddr_train_dataset, get_ddr_train_transform
from models import build_resnet50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=SOURCE_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dataset = build_ddr_train_dataset(get_ddr_train_transform())
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = build_resnet50(pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)

    print(f'[*] DDR source training: {len(dataset)} images, {args.epochs} epochs')
    model.train()
    for epoch in range(args.epochs):
        loss_sum, correct, total = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
            correct += out.argmax(1).eq(y).sum().item()
            total += y.size(0)
        print(f'  epoch {epoch+1}/{args.epochs}  loss={loss_sum/len(loader):.4f}  acc={100*correct/total:.2f}%')

    torch.save(model.state_dict(), WEIGHT_BASELINE)
    print(f'[+] Saved {WEIGHT_BASELINE}')


if __name__ == '__main__':
    main()
