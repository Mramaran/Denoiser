"""Datasets for end-to-end (NoisyLR -> GT) restoration training.

The challenge ships 3200 real degraded/clean pairs. Those are the actual test
distribution, so they are the primary training signal; the calibrated synthetic
degradation in augment_pipeline.py supplements them.
"""

import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from utils.utils_image import augment_img


class PairedRealDataset(Dataset):
    """The real (NoisyLR, GT) pairs shipped with the challenge.

    Crops are chosen in LR coordinates and mirrored into GT coordinates at
    `scale`x, so input and target stay pixel-aligned. The same geometric
    augmentation mode is applied to both halves of the pair.
    """

    def __init__(self, noisy_dir, gt_dir, file_list, lr_patch=64, scale=2, augment=True):
        super().__init__()
        self.noisy_dir = noisy_dir
        self.gt_dir = gt_dir
        self.files = [os.path.basename(f) for f in file_list]
        self.lr_patch = lr_patch
        self.scale = scale
        self.augment = augment

        if not self.files:
            raise RuntimeError("PairedRealDataset received an empty file list")

        print(f"[PairedRealDataset] {len(self.files)} pairs | "
              f"lr_patch={lr_patch} | augment={augment}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]
        y = np.load(os.path.join(self.noisy_dir, name)).astype(np.float32)
        x = np.load(os.path.join(self.gt_dir, name)).astype(np.float32)

        p, s = self.lr_patch, self.scale
        h, w = y.shape

        if self.augment and h > p and w > p:
            i = random.randint(0, h - p)
            j = random.randint(0, w - p)
        else:
            i, j = max(0, (h - p) // 2), max(0, (w - p) // 2)

        y = y[i:i + p, j:j + p]
        x = x[i * s:(i + p) * s, j * s:(j + p) * s]

        if self.augment:
            mode = random.randint(0, 7)
            y, x = augment_img(y, mode), augment_img(x, mode)

        return (torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0),
                torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0))
