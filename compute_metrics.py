"""Score restored outputs against ground truth.

Usage:
    python compute_metrics.py --restored_dir ./restored_val --gt_dir ../Dataset/train/train/GT \
        --file_list eval_split.txt --baseline_dir ../Dataset/train/train/NoisyLR --lpips

--file_list restricts scoring to the frozen held-out split, which is the only
honest way to score a trained model. --baseline_dir prints the bicubic-upsample
reference alongside, so every number has its control.
"""

import argparse
import glob
import os

import numpy as np

from utils.utils_image import load_npy, calculate_psnr, calculate_ssim


def _bicubic_upsample(img, scale=2):
    from scipy.ndimage import zoom
    return zoom(img, scale, order=3)


def _make_lpips_fn(device="cpu"):
    """Return an lpips scorer, or None if the package is unavailable."""
    try:
        import lpips
        import torch
    except ImportError:
        print("[Metrics] lpips not installed - skipping LPIPS")
        return None

    net = lpips.LPIPS(net="alex").to(device).eval()

    def score(pred, gt):
        import torch
        with torch.no_grad():
            p = torch.from_numpy(pred).float()[None, None].repeat(1, 3, 1, 1) * 2.0 - 1.0
            g = torch.from_numpy(gt).float()[None, None].repeat(1, 3, 1, 1) * 2.0 - 1.0
            return float(net(p.to(device), g.to(device)).item())

    return score


def main():
    parser = argparse.ArgumentParser(description="Compute PSNR/SSIM/LPIPS metrics")
    parser.add_argument("--restored_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--file_list", default=None,
                        help="Text file of filenames to score (e.g. eval_split.txt)")
    parser.add_argument("--baseline_dir", default=None,
                        help="Directory of LR inputs; prints a bicubic-upsample control")
    parser.add_argument("--lpips", action="store_true", help="Also compute LPIPS")
    parser.add_argument("--border", type=int, default=0)
    args = parser.parse_args()

    if args.file_list:
        with open(args.file_list) as f:
            names = [line.strip() for line in f if line.strip()]
    else:
        names = sorted(os.path.basename(p)
                       for p in glob.glob(os.path.join(args.restored_dir, "*.npy")))

    lpips_fn = _make_lpips_fn() if args.lpips else None

    psnrs, ssims, lps = [], [], []
    base_psnrs, base_ssims = [], []
    skipped = 0

    for name in names:
        rpath = os.path.join(args.restored_dir, name)
        gpath = os.path.join(args.gt_dir, name)
        if not (os.path.isfile(rpath) and os.path.isfile(gpath)):
            skipped += 1
            continue

        restored = load_npy(rpath).clip(0, 1)
        gt = load_npy(gpath)
        if restored.shape != gt.shape:
            print(f"  SKIP {name}: shape {restored.shape} vs GT {gt.shape}")
            skipped += 1
            continue

        psnrs.append(calculate_psnr(restored, gt, border=args.border))
        ssims.append(calculate_ssim(restored, gt, border=args.border))
        if lpips_fn is not None:
            lps.append(lpips_fn(restored, gt))

        if args.baseline_dir:
            bpath = os.path.join(args.baseline_dir, name)
            if os.path.isfile(bpath):
                up = _bicubic_upsample(load_npy(bpath)).clip(0, 1)
                base_psnrs.append(calculate_psnr(up, gt, border=args.border))
                base_ssims.append(calculate_ssim(up, gt, border=args.border))

    if not psnrs:
        print("ERROR: no matching file pairs found")
        return

    print(f"\n{'=' * 56}")
    print(f"  Scored {len(psnrs)} images ({skipped} skipped)")
    print(f"{'=' * 56}")
    print(f"  PSNR : {np.mean(psnrs):8.4f} dB")
    print(f"  SSIM : {np.mean(ssims):8.6f}")
    if lps:
        print(f"  LPIPS: {np.mean(lps):8.6f}   (lower is better)")
    if base_psnrs:
        print(f"{'-' * 56}")
        print(f"  bicubic baseline PSNR : {np.mean(base_psnrs):8.4f} dB")
        print(f"  bicubic baseline SSIM : {np.mean(base_ssims):8.6f}")
        print(f"  gain over bicubic     : {np.mean(psnrs) - np.mean(base_psnrs):+8.4f} dB")
    print(f"{'=' * 56}")


if __name__ == "__main__":
    main()
