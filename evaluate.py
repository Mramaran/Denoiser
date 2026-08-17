"""Evaluation script for the KLA image restoration pipeline.

Two restoration modes are available, selected with --mode:
  e2e (default): single-pass end-to-end DRUNet-SR model with an optional 8x
                 geometric self-ensemble. See utils/inference.py.
  pnp:           iterative PnP-HQS restoration (denoise + despeckle + 2x
                 super-resolve) used as the ablation baseline.

Usage:
    python evaluate.py --input_dir ./test_images --output_dir ./restored_outputs

The script will:
1. Auto-detect trained weights for the selected mode in model_weights/
   (or load the path given by --weights)
2. For each .npy image in --input_dir, restore it and save the 256x256
   result to --output_dir
3. Report timing statistics

Requirements:
    - PyTorch >= 2.0
    - NumPy
    - Trained model weights in model_weights/ (drunet_sr_inference.pth for
      e2e mode, drunet_denoiser_best.pth for pnp mode)

Reference:
    Zhang et al., "Plug-and-Play Image Restoration with Deep Denoiser Prior",
    TPAMI 2021. https://arxiv.org/abs/2008.13751 -- the pnp mode above.
    The default e2e mode is a single-pass DRUNet-SR model (see models/drunet.py,
    utils/inference.py) evaluated with an 8x geometric flip/rotation
    self-ensemble, a standard technique in top NTIRE super-resolution entries.
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
from utils.inference import restore_e2e
from utils.utils_pnp import pnp_hqs_restore
from utils.utils_image import load_npy, save_npy


def find_model_weights(weight_dir="model_weights", mode="e2e"):
    """Find the best available model weights file for the requested mode."""
    if mode == "e2e":
        # Prefer the shipped fp16 inference checkpoint; fall back to the
        # local full checkpoint when training on this machine.
        candidates = [
            os.path.join(weight_dir, "drunet_sr_inference.pth"),
            os.path.join(weight_dir, "drunet_sr_best.pth"),
        ]
    else:
        candidates = [
            os.path.join(weight_dir, "drunet_denoiser_best.pth"),
            os.path.join(weight_dir, "drunet_denoiser_final.pth"),
        ]
    for path in candidates:
        if os.path.isfile(path):
            return path

    pth_files = glob.glob(os.path.join(weight_dir, "*.pth"))
    return sorted(pth_files)[-1] if pth_files else None


def main():
    parser = argparse.ArgumentParser(
        description="KLA Image Restoration — Evaluation Script (e2e / pnp)"
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
        "--mode", choices=["e2e", "pnp"], default="e2e",
        help="e2e: single-pass end-to-end model (default). pnp: iterative PnP-HQS."
    )
    parser.add_argument(
        "--no_self_ensemble", action="store_true",
        help="Disable 8x geometric self-ensemble in e2e mode"
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
        "--scale_factor", type=int, default=None,
        help="Super-resolution scale factor (default: 2). In e2e mode this is "
             "a property of the checkpoint (its saved 'sf' value) and cannot "
             "be overridden; an explicit value that disagrees only triggers "
             "a warning. In pnp mode it is used as passed."
    )
    parser.add_argument(
        "--no_log_domain", action="store_true",
        help="Disable log-domain speckle transform"
    )
    args = parser.parse_args()

    # --scale_factor defaults to 2, but we need to know whether the user
    # explicitly passed it (as opposed to relying on the default) so that
    # e2e mode can warn only on a genuine conflict with the checkpoint,
    # never on the plain `--input_dir X --output_dir Y` invocation.
    scale_factor_explicit = args.scale_factor is not None
    if args.scale_factor is None:
        args.scale_factor = 2

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
            os.path.join(script_dir, "model_weights"), mode=args.mode
        )

    if weights_path is None or not os.path.isfile(weights_path):
        print(f"ERROR: Model weights not found. Expected in model_weights/")
        print(f"       Run train.py first, or provide --weights path")
        sys.exit(1)

    print(f"[Evaluate] Loading weights: {weights_path}")

    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)

    if args.mode == "e2e":
        # The scale factor is a property of the trained weights, not a
        # runtime choice: --scale_factor must not silently override it.
        checkpoint_sf = checkpoint.get("sf", 2)
        if scale_factor_explicit and args.scale_factor != checkpoint_sf:
            print(f"[Evaluate] WARNING: --scale_factor={args.scale_factor} was given, but "
                  f"this checkpoint was trained with sf={checkpoint_sf}. In e2e mode the "
                  f"scale factor comes from the checkpoint, not the CLI; using "
                  f"sf={checkpoint_sf}.")
        args.scale_factor = checkpoint_sf

        model = DRUNet(in_nc=checkpoint.get("in_nc", 1), out_nc=1,
                       nc=[64, 128, 256, 512], nb=4, sf=checkpoint_sf)
        # Prefer the EMA weights: they are what training scored. Fall back to
        # model_state_dict, and finally to the checkpoint itself being a bare
        # state_dict (mirrors the pnp branch below) so a checkpoint saved as
        # plain torch.save(model.state_dict(), path) still loads cleanly
        # instead of raising a raw KeyError.
        if "ema_state_dict" in checkpoint:
            state = checkpoint["ema_state_dict"]
        elif "model_state_dict" in checkpoint:
            state = checkpoint["model_state_dict"]
        else:
            state = checkpoint
        # The shipped inference checkpoint stores fp16 to stay under 100 MB.
        if checkpoint.get("fp16"):
            state = {k: (v.float() if v.is_floating_point() else v)
                     for k, v in state.items()}
        model.load_state_dict(state)
    else:
        model = DRUNet(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4)
        state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
        model.load_state_dict(state)

    if "epoch" in checkpoint:
        print(f"[Evaluate] Model from epoch {checkpoint['epoch']}, "
              f"val PSNR: {checkpoint.get('val_psnr', 'unknown')}")

    model = model.to(device)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"[Evaluate] Mode: {args.mode} | parameters: {num_params:,}")

    # -----------------------------------------------------------------------
    # Run restoration
    # -----------------------------------------------------------------------
    print(f"\n[Evaluate] Settings:")
    if args.mode == "e2e":
        print(f"  Self-ensemble: {not args.no_self_ensemble}")
    else:
        print(f"  Iterations: {args.num_iterations}")
        print(f"  Sigma: {args.sigma_max:.4f} -> {args.sigma_min:.4f}")
        print(f"  Log-domain speckle: {not args.no_log_domain}")
    print(f"  Scale factor: {args.scale_factor}x")
    print(f"  Output dir: {args.output_dir}")
    print("=" * 60)

    total_time = 0.0

    for i, input_path in enumerate(input_files):
        filename = os.path.basename(input_path)
        t0 = time.time()

        img = load_npy(input_path)

        if args.mode == "e2e":
            restored = restore_e2e(
                img, model, device,
                self_ensemble=not args.no_self_ensemble,
            )
        else:
            restored = pnp_hqs_restore(
                y=img, denoiser_model=model,
                scale_factor=args.scale_factor,
                num_iterations=args.num_iterations,
                sigma_max=args.sigma_max, sigma_min=args.sigma_min,
                use_log_domain=not args.no_log_domain, device=device,
            )

        save_npy(restored, os.path.join(args.output_dir, filename))

        elapsed = time.time() - t0
        total_time += elapsed

        if (i + 1) % 50 == 0 or (i + 1) == len(input_files):
            print(f"  [{i + 1:4d}/{len(input_files)}] {filename} | "
                  f"{img.shape} -> {restored.shape} | {elapsed:.3f}s")

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
