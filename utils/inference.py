"""Inference helpers for the end-to-end restoration model."""

import numpy as np
import torch

from utils.utils_image import augment_img

# augment_img mode -> the mode that undoes it. Verified exhaustively:
# 3 (rot90) and 5 (rot270) are each other's inverse; every other mode is an
# involution. tests/test_inference.py re-checks this.
INVERSE_MODE = {0: 0, 1: 1, 2: 2, 3: 5, 4: 4, 5: 3, 6: 6, 7: 7}


@torch.no_grad()
def restore_e2e(img, model, device, self_ensemble=True):
    """Restore one LR image with the end-to-end model.

    With self_ensemble=True the image is restored under all 8 flip/rotation
    variants and the inverse-transformed outputs are averaged. This is the
    geometric self-ensemble introduced by EDSR (Lim et al., CVPRW 2017), which
    reports +0.10-0.25 dB for it at no training cost. It is valid only for
    symmetric downsampling; ours is 2x average pooling, which qualifies.

    Args:
        img: LR image as a numpy array (H, W), float32
        model: a DRUNet built with sf=2, already on `device` and in eval mode
        device: torch device
        self_ensemble: average over 8 geometric transforms
    Returns:
        restored image (2H, 2W), float32, clipped to [0, 1]
    """
    return restore_e2e_batch([img], model, device, self_ensemble)[0]


@torch.no_grad()
def restore_e2e_batch(imgs, model, device, self_ensemble=True):
    """Restore a list of LR images in as few forward passes as possible.

    The 8 self-ensemble variants of every image are collected into one batch
    rather than run as 8 separate batch-1 passes, which also collapses 8 device
    round-trips into one. Arithmetic is unchanged, but floating-point reduction
    order is not, so results match the per-image path to ~1e-6, not bit-exactly.

    Args:
        imgs: list of LR images, each (H, W) float32
        model: a DRUNet built with sf=2, already on `device` and in eval mode
        device: torch device
        self_ensemble: average over 8 geometric transforms
    Returns:
        list of restored images (2H, 2W), float32, clipped to [0, 1]
    """
    modes = list(range(8)) if self_ensemble else [0]

    # rot90/rot270 transpose a non-square image, so variants do not all share a
    # shape and cannot go into one tensor. Bucket by shape and run one batch per
    # bucket. For the square inputs this challenge ships there is exactly one.
    variants = [(i, mode, augment_img(img, mode))
                for i, img in enumerate(imgs) for mode in modes]

    buckets = {}
    for idx, (i, mode, arr) in enumerate(variants):
        buckets.setdefault(arr.shape, []).append(idx)

    outputs = [None] * len(variants)
    for shape, indices in buckets.items():
        batch = np.stack([np.ascontiguousarray(variants[k][2]) for k in indices])
        x = torch.from_numpy(batch).float().unsqueeze(1).to(device)
        out = model(x)[:, 0].float().cpu().numpy()
        for slot, k in enumerate(indices):
            outputs[k] = out[slot]

    restored = []
    for i in range(len(imgs)):
        accumulator = None
        for k, (img_idx, mode, _) in enumerate(variants):
            if img_idx != i:
                continue
            undone = augment_img(outputs[k], INVERSE_MODE[mode])
            accumulator = undone if accumulator is None else accumulator + undone
        restored.append(np.clip(accumulator / len(modes), 0.0, 1.0).astype(np.float32))

    return restored
