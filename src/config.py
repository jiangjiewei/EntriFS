"""Project paths and hyper-parameters (override via environment variables)."""
import os
from pathlib import Path

ROOT = Path(os.environ.get('DR_PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()

# Dataset roots (download & place data as described in data/README.md)
DDR_ROOT = Path(os.environ.get('DDR_ROOT', ROOT / 'data' / 'ddr')).resolve()
APTOS_LABELS = Path(os.environ.get('APTOS_LABELS', ROOT / 'data' / 'labels' / 'aptos_train.csv')).resolve()
APTOS_IMG_DIR = Path(os.environ.get('APTOS_IMAGES', ROOT / 'data' / 'aptos' / 'train_images')).resolve()

OUTPUT_DIR = Path(os.environ.get('OUTPUT_DIR', ROOT / 'outputs')).resolve()

# Protocol (paper)
CALIB_PER_CLASS = int(os.environ.get('CALIB_PER_CLASS', '10'))
RANDOM_SEED = int(os.environ.get('RANDOM_SEED', '42'))
T_SAMPLES = int(os.environ.get('T_SAMPLES', '30'))

SOURCE_EPOCHS = int(os.environ.get('SOURCE_EPOCHS', '15'))
CALIB_EPOCHS = int(os.environ.get('CALIB_EPOCHS', '12'))
BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '32'))

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

WEIGHT_BASELINE = OUTPUT_DIR / 'resnet50_ddr_baseline.pth'
WEIGHT_LINEAR = OUTPUT_DIR / 'resnet50_linear_probe.pth'
WEIGHT_OURS = OUTPUT_DIR / 'resnet50_deep_calibrated.pth'
