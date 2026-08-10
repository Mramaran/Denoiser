"""Compute PSNR and SSIM metrics between restored outputs and ground truth.

Usage:
    python compute_metrics.py --restored_dir ./restored_outputs --gt_dir ../Dataset/train/train/GT

This script is for validation only — the test set has no ground truth.
"""

import os
import argparse
import glob
import numpy as np
from utils.utils_image import load_npy, calculate_psnr, calculate_ssim


def main():
    parser = argparse.ArgumentParser(description="Compute PSNR/SSIM metrics")
    parser.add_argument("--restored_dir", required=True,
                        help="Path to directory with restored .npy images")
    parser.add_argument("--gt_dir", required=True,
                        help="Path to directory with ground-truth .npy images")
    parser.add_argument("--border", type=int, default=0,
                        help="Border pixels to exclude from metrics (default: 0)")
    args = parser.parse_args()

    restored_files = sorted(glob.glob(os.path.join(args.restored_dir, "*.npy")))
    if len(restored_files) == 0:
        print(f"ERROR: No .npy files in {args.restored_dir}")
        return

    total_psnr = 0.0
    total_ssim = 0.0
    count = 0

    for rpath in restored_files:
        fname = os.path.basename(rpath)
        gpath = os.path.join(args.gt_dir, fname)

        if not os.path.isfile(gpath):
            continue

        restored = load_npy(rpath).clip(0, 1)
        gt = load_npy(gpath)

        # Ensure same shape
        if restored.shape != gt.shape:
            print(f"  SKIP {fname}: shape mismatch {restored.shape} vs {gt.shape}")
            continue

        psnr = calculate_psnr(restored, gt, border=args.border)
        ssim = calculate_ssim(restored, gt, border=args.border)

        total_psnr += psnr
        total_ssim += ssim
        count += 1

    if count == 0:
        print("ERROR: No matching file pairs found")
        return

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count

    print(f"\n{'='*50}")
    print(f"  Results ({count} images)")
    print(f"{'='*50}")
    print(f"  Average PSNR: {avg_psnr:.4f} dB")
    print(f"  Average SSIM: {avg_ssim:.6f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
