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
    standard geometric self-ensemble used by every top NTIRE entry and is
    worth +0.1-0.4 dB at no training cost.

    Args:
        img: LR image as a numpy array (H, W), float32
        model: a DRUNet built with sf=2, already on `device` and in eval mode
        device: torch device
        self_ensemble: average over 8 geometric transforms
    Returns:
        restored image (2H, 2W), float32, clipped to [0, 1]
    """
    modes = list(range(8)) if self_ensemble else [0]
    accumulator = None

    for mode in modes:
        transformed = augment_img(img, mode)
        x = torch.from_numpy(np.ascontiguousarray(transformed)).float()
        x = x.unsqueeze(0).unsqueeze(0).to(device)

        out = model(x)[0, 0].float().cpu().numpy()
        out = augment_img(out, INVERSE_MODE[mode])

        accumulator = out if accumulator is None else accumulator + out

    restored = accumulator / len(modes)
    return np.clip(restored, 0.0, 1.0).astype(np.float32)
