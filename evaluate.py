"""Evaluation script for the PnP-HQS image restoration pipeline.

This script loads the trained DRUNet denoiser and runs the full PnP-HQS
restoration pipeline on all .npy images in the input directory.

Usage:
    python evaluate.py --input_dir ./test_images --output_dir ./restored_outputs

The script will:
1. Load the trained DRUNet denoiser from model_weights/
2. For each .npy image in --input_dir:
   a. Run the PnP-HQS restoration (denoise + despeckle + 2x super-resolve)
   b. Save the restored 256x256 image to --output_dir
3. Report timing statistics

Requirements:
    - PyTorch >= 2.0
    - NumPy
    - Trained model weights in model_weights/drunet_denoiser_best.pth

Reference:
    Zhang et al., "Plug-and-Play Image Restoration with Deep Denoiser Prior",
    TPAMI 2021. https://arxiv.org/abs/2008.13751
"""

import argparse
import os
import sys
import time
import glob

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.drunet import DRUNet
from utils.utils_pnp import pnp_hqs_restore
from utils.utils_image import load_npy, save_npy


def find_model_weights(weight_dir="model_weights"):
    """Find the best available model weights file.

    Search priority:
    1. drunet_denoiser_best.pth (best validation PSNR)
    2. drunet_denoiser_final.pth (final epoch)
    3. Any .pth file in the directory

    Args:
        weight_dir: directory containing model weights

    Returns:
        path to the model weights file, or None if not found
    """
    candidates = [
        os.path.join(weight_dir, "drunet_denoiser_best.pth"),
        os.path.join(weight_dir, "drunet_denoiser_final.pth"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path

    # Fallback: find any .pth file
    pth_files = glob.glob(os.path.join(weight_dir, "*.pth"))
    if pth_files:
        return sorted(pth_files)[-1]  # Pick the latest

    return None


def main():
    parser = argparse.ArgumentParser(
        description="PnP-HQS Image Restoration — Evaluation Script"
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Path to directory containing degraded test images (.npy files)"
    )
    parser.add_argument(
        "--output_dir", required=True,
        help="Path to directory for writing restored output images"
    )
    parser.add_argument(
        "--weights", type=str, default=None,
        help="Path to model weights file (default: auto-detect in model_weights/)"
    )
    parser.add_argument(
        "--num_iterations", type=int, default=15,
        help="Number of PnP-HQS iterations (default: 15)"
    )
    parser.add_argument(
        "--sigma_max", type=float, default=49.0 / 255.0,
        help="Initial noise level for denoiser schedule (default: 49/255)"
    )
    parser.add_argument(
        "--sigma_min", type=float, default=1.0 / 255.0,
        help="Final noise level for denoiser schedule (default: 1/255)"
    )
    parser.add_argument(
        "--scale_factor", type=int, default=2,
        help="Super-resolution scale factor (default: 2)"
    )
    parser.add_argument(
        "--no_log_domain", action="store_true",
        help="Disable log-domain speckle transform"
    )
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Validate inputs
    # -----------------------------------------------------------------------
    if not os.path.isdir(args.input_dir):
        print(f"ERROR: Input directory not found: {args.input_dir}")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Find input files
    input_files = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))
    if len(input_files) == 0:
        print(f"ERROR: No .npy files found in {args.input_dir}")
        sys.exit(1)
    print(f"[Evaluate] Found {len(input_files)} input images")

    # -----------------------------------------------------------------------
    # Load model
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Evaluate] Device: {device}")

    # Find weights
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if args.weights:
        weights_path = args.weights
    else:
        weights_path = find_model_weights(
            os.path.join(script_dir, "model_weights")
        )

    if weights_path is None or not os.path.isfile(weights_path):
        print(f"ERROR: Model weights not found. Expected in model_weights/")
        print(f"       Run train.py first, or provide --weights path")
        sys.exit(1)

    print(f"[Evaluate] Loading weights: {weights_path}")

    # Initialize model
    model = DRUNet(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)

    # Handle checkpoint format (may have 'model_state_dict' key or be raw state_dict)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
        epoch_info = checkpoint.get("epoch", "unknown")
        psnr_info = checkpoint.get("val_psnr", "unknown")
        print(f"[Evaluate] Model from epoch {epoch_info}, val PSNR: {psnr_info}")
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Evaluate] Model parameters: {num_params:,}")

    # -----------------------------------------------------------------------
    # Run restoration
    # -----------------------------------------------------------------------
    print(f"\n[Evaluate] PnP-HQS Settings:")
    print(f"  Iterations: {args.num_iterations}")
    print(f"  Sigma: {args.sigma_max:.4f} -> {args.sigma_min:.4f}")
    print(f"  Scale factor: {args.scale_factor}x")
    print(f"  Log-domain speckle: {not args.no_log_domain}")
    print(f"  Output dir: {args.output_dir}")
    print("=" * 60)

    total_time = 0.0

    for i, input_path in enumerate(input_files):
        filename = os.path.basename(input_path)
        t0 = time.time()

        # Load input image
        img = load_npy(input_path)

        # Run PnP-HQS restoration
        restored = pnp_hqs_restore(
            y=img,
            denoiser_model=model,
            scale_factor=args.scale_factor,
            num_iterations=args.num_iterations,
            sigma_max=args.sigma_max,
            sigma_min=args.sigma_min,
            use_log_domain=not args.no_log_domain,
            device=device,
        )

        # Save restored image
        output_path = os.path.join(args.output_dir, filename)
        save_npy(restored, output_path)

        elapsed = time.time() - t0
        total_time += elapsed

        if (i + 1) % 50 == 0 or (i + 1) == len(input_files):
            print(f"  [{i + 1:4d}/{len(input_files)}] {filename} | "
                  f"Input: {img.shape} -> Output: {restored.shape} | "
                  f"Time: {elapsed:.3f}s")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    avg_time = total_time / len(input_files)
    print("=" * 60)
    print(f"[Evaluate] Restoration complete!")
    print(f"  Total images: {len(input_files)}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Average time per image: {avg_time:.3f}s")
    print(f"  Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
