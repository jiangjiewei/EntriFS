"""Shared data loading, ROI crop, and training utilities."""
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, ConcatDataset
from torchvision import datasets, transforms

from config import (
    DDR_ROOT, APTOS_LABELS, APTOS_IMG_DIR, CALIB_PER_CLASS, RANDOM_SEED,
    IMAGENET_MEAN, IMAGENET_STD,
)


def crop_image_from_gray(img_pil, tol=7):
    """ROI crop: remove black borders from fundus images."""
    img = np.array(img_pil)
    if img.ndim == 2:
        mask = img > tol
        if not np.any(mask):
            return img_pil
        return Image.fromarray(img[np.ix_(mask.any(1), mask.any(0))])
    if img.ndim == 3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img > tol
        if not np.any(mask):
            return img_pil
        cropped = img[np.ix_(mask.any(1), mask.any(0))]
        if cropped.shape[0] == 0 or cropped.shape[1] == 0:
            return img_pil
        return Image.fromarray(cropped)
    return img_pil


def get_ddr_train_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_calib_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def get_test_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_ddr_train_dataset(transform):
    """Paper protocol: DDR train split only."""
    return datasets.ImageFolder(str(DDR_ROOT / 'train'), transform=transform)


def build_ddr_dataset(transform):
    """Optional: train + valid for extended experiments."""
    train_ds = datasets.ImageFolder(str(DDR_ROOT / 'train'), transform=transform)
    valid_ds = datasets.ImageFolder(str(DDR_ROOT / 'valid'), transform=transform)
    return ConcatDataset([train_ds, valid_ds]), train_ds.class_to_idx


def split_aptos_calib_test(seed=RANDOM_SEED):
    """First CALIB_PER_CLASS per grade for calibration; remainder for blind test."""
    df_all = pd.read_csv(APTOS_LABELS)
    calib_parts, test_parts = [], []
    for c in range(5):
        c_df = df_all[df_all['diagnosis'] == c]
        calib_parts.append(c_df.head(CALIB_PER_CLASS))
        test_parts.append(c_df.tail(len(c_df) - CALIB_PER_CLASS))
    return pd.concat(calib_parts).reset_index(drop=True), pd.concat(test_parts).reset_index(drop=True)


class APTOSDataset(Dataset):
    def __init__(self, dataframe, img_dir, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.img_dir = str(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = f"{row['id_code']}.png"
        raw = Image.open(f"{self.img_dir}/{img_name}").convert('RGB')
        cropped = crop_image_from_gray(raw, tol=7)
        label = int(row['diagnosis'])
        if self.transform:
            cropped = self.transform(cropped)
        return cropped, label, row['id_code']


def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
