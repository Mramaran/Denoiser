#!/usr/bin/env python
"""KLA submission entry point.

    python run.py <input-dir> <output-dir>

Reads every .npy in <input-dir>, restores it, and writes one .npy of the same
name to <output-dir>. The output directory is created if it does not exist.

Guarantees enforced below, one per submission-checklist item:
  - one output per input, identical filename
  - grayscale, shape (H, W)
  - float32, finite, inside [0, 1]
  - 2x the input resolution

Runs on an NVIDIA GPU when one is visible and falls back to CPU otherwise.
Needs no internet access, no API keys, and no downloads: the weights ship in
this repository, and the inference import chain never touches `lpips` (that is
a training-only dependency, imported solely by models/losses.py).

--weights is optional and exists only for local experiments; the default
auto-detection is what a fresh clone uses.
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.drunet import DRUNet
from utils.inference import restore_e2e_batch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Searched in order. models/ is where the shipped checkpoint lives and where
# training writes, matching the folder layout the submission brief specifies.
# model_weights/ is retained as a fallback so a clone using the older layout
# still runs without an edit.
WEIGHT_CANDIDATES = [
    os.path.join("models", "drunet_sr_inference.pth"),
    os.path.join("model_weights", "drunet_sr_inference.pth"),
    os.path.join("models", "drunet_sr_best.pth"),
    os.path.join("model_weights", "drunet_sr_best.pth"),
]


def find_weights(explicit=None):
    if explicit:
        if not os.path.isfile(explicit):
            sys.exit(f"ERROR: weights not found at {explicit}")
        return explicit
    for rel in WEIGHT_CANDIDATES:
        path = os.path.join(SCRIPT_DIR, rel)
        if os.path.isfile(path):
            return path
    for folder in ("models", "model_weights"):
        found = sorted(glob.glob(os.path.join(SCRIPT_DIR, folder, "*.pth")))
        if found:
            return found[0]
    sys.exit("ERROR: no .pth checkpoint found in models/ or model_weights/")


def load_model(weights_path, device):
    ckpt = torch.load(weights_path, map_location=device, weights_only=True)

    sf = ckpt.get("sf", 2)
    in_nc = ckpt.get("in_nc", 1)
    model = DRUNet(in_nc=in_nc, out_nc=1, nc=[64, 128, 256, 512], nb=4, sf=sf)

    # EMA weights are what training scored, so prefer them. Fall back to the
    # live weights, then to the checkpoint being a bare state_dict.
    if "ema_state_dict" in ckpt:
        state = ckpt["ema_state_dict"]
    elif "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt

    # The shipped checkpoint stores fp16 to stay under GitHub's 100 MB limit.
    if ckpt.get("fp16"):
        state = {k: (v.float() if v.is_floating_point() else v)
                 for k, v in state.items()}

    model.load_state_dict(state)
    return model.to(device).eval(), sf, ckpt


def as_2d(arr, name):
    """Accept (H, W), (H, W, 1) or (1, H, W); always hand the model (H, W)."""
    a = np.asarray(arr)
    if a.ndim == 3:
        if a.shape[-1] == 1:
            a = a[:, :, 0]
        elif a.shape[0] == 1:
            a = a[0]
        else:
            sys.exit(f"ERROR: {name} has shape {a.shape}; expected grayscale")
    elif a.ndim != 2:
        sys.exit(f"ERROR: {name} has {a.ndim} dimensions; expected 2")
    return a.astype(np.float32)


def sanitise(out, name):
    """Enforce the output contract. Returns (array, n_nonfinite_repaired)."""
    out = np.asarray(out, dtype=np.float32)
    bad = int((~np.isfinite(out)).sum())
    if bad:
        # Should never fire. If it does, a clean file beats a NaN-poisoned one,
        # and the count is reported at the end rather than swallowed.
        out = np.nan_to_num(out, nan=0.0, posinf=1.0, neginf=0.0)
    return np.clip(out, 0.0, 1.0).astype(np.float32), bad


def main():
    ap = argparse.ArgumentParser(
        description="KLA image restoration. Usage: python run.py <input-dir> <output-dir>")
    ap.add_argument("input_dir", help="directory containing degraded .npy images")
    ap.add_argument("output_dir", help="directory to write restored .npy images")
    ap.add_argument("--weights", default=None,
                    help="optional explicit checkpoint path (default: auto-detect)")
    ap.add_argument("--no_self_ensemble", action="store_true",
                    help="disable the 8x geometric self-ensemble (faster, ~2.8 dB worse)")
    args = ap.parse_args()

    if not os.path.isdir(args.input_dir):
        sys.exit(f"ERROR: input directory not found: {args.input_dir}")

    os.makedirs(args.output_dir, exist_ok=True)

    inputs = sorted(glob.glob(os.path.join(args.input_dir, "*.npy")))
    if not inputs:
        sys.exit(f"ERROR: no .npy files found in {args.input_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_path = find_weights(args.weights)
    model, sf, ckpt = load_model(weights_path, device)

    print(f"[run] device        : {device}")
    if device.type == "cuda":
        print(f"[run] gpu           : {torch.cuda.get_device_name(0)}")
    print(f"[run] weights       : {os.path.relpath(weights_path, SCRIPT_DIR)}")
    print(f"[run] scale factor  : {sf}x")
    print(f"[run] self-ensemble : {not args.no_self_ensemble}")
    print(f"[run] input images  : {len(inputs)}")

    t0 = time.time()
    repaired = 0

    for i, path in enumerate(inputs, 1):
        name = os.path.basename(path)
        img = as_2d(np.load(path), name)

        restored = restore_e2e_batch(
            [img], model, device, self_ensemble=not args.no_self_ensemble)[0]

        restored, bad = sanitise(restored, name)
        repaired += bad

        expected = (img.shape[0] * sf, img.shape[1] * sf)
        if restored.shape != expected:
            sys.exit(f"ERROR: {name} produced {restored.shape}, expected {expected}")

        np.save(os.path.join(args.output_dir, name), restored)

        if i % 50 == 0 or i == len(inputs):
            print(f"  [{i:4d}/{len(inputs)}] {name}  {img.shape} -> {restored.shape}")

    elapsed = time.time() - t0
    written = len(glob.glob(os.path.join(args.output_dir, "*.npy")))

    print(f"[run] wrote         : {written} files to {args.output_dir}")
    print(f"[run] time          : {elapsed:.1f}s total, {elapsed/len(inputs):.3f}s per image")
    if repaired:
        print(f"[run] WARNING       : repaired {repaired} non-finite values")

    if written < len(inputs):
        sys.exit(f"ERROR: {len(inputs)} inputs but only {written} outputs")


if __name__ == "__main__":
    main()
