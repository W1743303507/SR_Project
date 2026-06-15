from pathlib import Path
import csv
import random

import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image


class SRDataset(Dataset):
    """
    Super-resolution dataset.

    It reads LR-HR image pairs from a CSV file.
    For training, it randomly crops LR patches and corresponding HR patches.
    For validation/testing, it returns full images.
    """

    def __init__(
        self,
        project_root,
        csv_path,
        scale=4,
        training=True,
        lr_patch_size=48,
        augment=True,
    ):
        self.project_root = Path(project_root)
        self.csv_path = self.project_root / csv_path
        self.scale = scale
        self.training = training
        self.lr_patch_size = lr_patch_size
        self.hr_patch_size = lr_patch_size * scale
        self.augment = augment

        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")

        self.samples = []
        with open(self.csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.samples.append(row)

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples found in CSV: {self.csv_path}")

    def __len__(self):
        return len(self.samples)

    def _load_image(self, relative_path):
        img_path = self.project_root / relative_path
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")
        return Image.open(img_path).convert("RGB")

    def _random_crop(self, lr_img, hr_img):
        lr_w, lr_h = lr_img.size
        hr_w, hr_h = hr_img.size

        expected_hr_w = lr_w * self.scale
        expected_hr_h = lr_h * self.scale

        if hr_w != expected_hr_w or hr_h != expected_hr_h:
            raise ValueError(
                f"LR-HR size mismatch: LR={lr_img.size}, HR={hr_img.size}, scale={self.scale}"
            )

        if lr_w < self.lr_patch_size or lr_h < self.lr_patch_size:
            raise ValueError(
                f"LR image is smaller than patch size. LR={lr_img.size}, patch={self.lr_patch_size}"
            )

        x_lr = random.randint(0, lr_w - self.lr_patch_size)
        y_lr = random.randint(0, lr_h - self.lr_patch_size)

        x_hr = x_lr * self.scale
        y_hr = y_lr * self.scale

        lr_patch = lr_img.crop(
            (
                x_lr,
                y_lr,
                x_lr + self.lr_patch_size,
                y_lr + self.lr_patch_size,
            )
        )

        hr_patch = hr_img.crop(
            (
                x_hr,
                y_hr,
                x_hr + self.hr_patch_size,
                y_hr + self.hr_patch_size,
            )
        )

        return lr_patch, hr_patch

    def _augment(self, lr_img, hr_img):
        if random.random() < 0.5:
            lr_img = lr_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            hr_img = hr_img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        if random.random() < 0.5:
            lr_img = lr_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            hr_img = hr_img.transpose(Image.Transpose.FLIP_TOP_BOTTOM)

        if random.random() < 0.5:
            lr_img = lr_img.transpose(Image.Transpose.ROTATE_90)
            hr_img = hr_img.transpose(Image.Transpose.ROTATE_90)

        return lr_img, hr_img

    def _to_tensor(self, img):
        arr = np.array(img).astype(np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)

    def __getitem__(self, index):
        sample = self.samples[index]

        lr_img = self._load_image(sample["lr_path"])
        hr_img = self._load_image(sample["hr_path"])

        if self.training:
            lr_img, hr_img = self._random_crop(lr_img, hr_img)

            if self.augment:
                lr_img, hr_img = self._augment(lr_img, hr_img)

        lr_tensor = self._to_tensor(lr_img)
        hr_tensor = self._to_tensor(hr_img)

        return {
            "lr": lr_tensor,
            "hr": hr_tensor,
            "lr_path": sample["lr_path"],
            "hr_path": sample["hr_path"],
            "dataset": sample["dataset"],
        }