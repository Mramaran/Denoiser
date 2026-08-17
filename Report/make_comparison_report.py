"""Build a visual comparison PDF: GT / NoisyLR / bicubic / restored, one page per image.

Images are drawn from the frozen held-out split (`eval_split.txt`), so nothing
shown here was seen during training. Each page carries per-image PSNR and SSIM
for both the bicubic control and the model, and the first page summarises the
aggregate over every image in the report.

Usage:
    python Report/make_comparison_report.py                      # 100 held-out images
    python Report/make_comparison_report.py --num_images 25      # quicker
    python Report/make_comparison_report.py --no_self_ensemble   # ablation view

Run from the repository root.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import scipy.ndimage as ndi

from models.drunet import DRUNet
from utils.inference import restore_e2e
from utils.utils_image import calculate_psnr, calculate_ssim


def load_model(weights, device):
    ckpt = torch.load(weights, map_location=device, weights_only=True)
    state = ckpt.get("ema_state_dict") or ckpt.get("model_state_dict") or ckpt
    if ckpt.get("fp16"):
        state = {k: (v.float() if v.is_floating_point() else v) for k, v in state.items()}
    model = DRUNet(in_nc=ckpt.get("in_nc", 1), out_nc=1,
                   nc=[64, 128, 256, 512], nb=4, sf=ckpt.get("sf", 2)).to(device).eval()
    model.load_state_dict(state)
    return model, ckpt


def add_summary_page(pdf, rows, ckpt, self_ensemble):
    """First page: what this report is, and the aggregate numbers."""
    bic = np.array([[r["bic_psnr"], r["bic_ssim"]] for r in rows]).mean(axis=0)
    mdl = np.array([[r["psnr"], r["ssim"]] for r in rows]).mean(axis=0)
    wins = sum(r["psnr"] > r["bic_psnr"] for r in rows)

    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.94, "Restoration Comparison — Held-Out Images",
             ha="center", fontsize=20, weight="bold")
    fig.text(0.5, 0.895, "AI-Based Restoration of Degraded Semiconductor Inspection Images",
             ha="center", fontsize=12, style="italic")

    body = [
        f"Images shown           : {len(rows)}, drawn from eval_split.txt",
        f"Checkpoint             : epoch {ckpt.get('epoch', '?')}",
        f"Self-ensemble          : {'8x geometric' if self_ensemble else 'disabled'}",
        "",
        "These images sit inside the training data directory but were excluded",
        "from training by a seeded 10% holdout, so every number below is measured",
        "on data the model has never seen.",
        "",
        f"{'':26}{'PSNR (dB)':>12}{'SSIM':>10}",
        f"{'-'*48}",
        f"{'Bicubic upsample':26}{bic[0]:12.3f}{bic[1]:10.4f}",
        f"{'Restored (this model)':26}{mdl[0]:12.3f}{mdl[1]:10.4f}",
        f"{'Gain':26}{mdl[0]-bic[0]:+12.3f}{mdl[1]-bic[1]:+10.4f}",
        "",
        f"Model beats bicubic on {wins} of {len(rows)} images.",
        "",
        "Each following page shows one image as four panels:",
        "",
        "   Ground Truth  |  Noisy LR (input)  |  Bicubic  |  Restored",
        "",
        "Bicubic is the control — what plain interpolation gives with no learning,",
        "and the bar the model has to clear. The Noisy LR panel is the 128x128",
        "input; its raw values fall outside [0,1], and the panel title reports the",
        "true range while the display is clipped for visibility.",
    ]
    fig.text(0.07, 0.80, "\n".join(body), fontsize=10.5, family="monospace",
             va="top", linespacing=1.55)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_image_page(pdf, name, gt, lr, bicubic, restored, r, dpi):
    fig, ax = plt.subplots(1, 4, figsize=(18, 5))
    panels = [
        (gt, "Ground Truth\n256x256"),
        (np.clip(lr, 0, 1),
         f"Noisy LR (input)\n128x128   raw range [{lr.min():.2f}, {lr.max():.2f}]"),
        (bicubic, f"Bicubic control\n{r['bic_psnr']:.2f} dB   SSIM {r['bic_ssim']:.3f}"),
        (restored, f"Restored\n{r['psnr']:.2f} dB   SSIM {r['ssim']:.3f}"),
    ]
    for a, (img, title) in zip(ax, panels):
        a.imshow(img, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        a.set_title(title, fontsize=11)
        a.axis("off")
    fig.suptitle(f"{name}    model vs bicubic: {r['psnr'] - r['bic_psnr']:+.2f} dB",
                 fontsize=13, weight="bold")
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches="tight", dpi=dpi)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description="Visual comparison PDF on held-out images")
    p.add_argument("--gt_dir", default="../Dataset/train/train/GT")
    p.add_argument("--noisy_dir", default="../Dataset/train/train/NoisyLR")
    p.add_argument("--file_list", default="eval_split.txt")
    p.add_argument("--weights", default="models/drunet_sr_inference.pth")
    p.add_argument("--num_images", type=int, default=100)
    p.add_argument("--no_self_ensemble", action="store_true")
    p.add_argument("--out", default="Report/restoration_comparison.pdf")
    p.add_argument("--dpi", type=int, default=90,
                   help="Lower keeps the PDF small; 90 stays legible at 4 panels wide")
    args = p.parse_args()

    for path in (args.gt_dir, args.noisy_dir, args.file_list, args.weights):
        if not os.path.exists(path):
            sys.exit(f"ERROR: not found: {path}\n"
                     f"       run this from the repository root")

    names = [l.strip() for l in open(args.file_list) if l.strip()][:args.num_images]
    if not names:
        sys.exit(f"ERROR: no filenames in {args.file_list}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.weights, device)
    self_ensemble = not args.no_self_ensemble
    print(f"[Report] {len(names)} held-out images | device {device} | "
          f"self-ensemble {self_ensemble}")

    # One pass: restore everything, keep it in memory (100 x 256x256 float32 is ~26 MB),
    # so the summary page can be written first without restoring twice.
    items = []
    for i, name in enumerate(names):
        gt = np.load(os.path.join(args.gt_dir, name)).astype(np.float32)
        lr = np.load(os.path.join(args.noisy_dir, name)).astype(np.float32)
        bicubic = np.clip(ndi.zoom(lr, 2, order=3), 0, 1).astype(np.float32)
        restored = restore_e2e(lr, model, device, self_ensemble=self_ensemble)
        r = dict(psnr=calculate_psnr(restored, gt), ssim=calculate_ssim(restored, gt),
                 bic_psnr=calculate_psnr(bicubic, gt), bic_ssim=calculate_ssim(bicubic, gt))
        items.append((name, gt, lr, bicubic, restored, r))
        if (i + 1) % 25 == 0 or (i + 1) == len(names):
            print(f"  restored [{i+1:3d}/{len(names)}] {name}  {r['psnr']:.2f} dB "
                  f"({r['psnr'] - r['bic_psnr']:+.2f} vs bicubic)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    rows = [it[5] | {"name": it[0]} for it in items]

    with PdfPages(args.out) as pdf:
        add_summary_page(pdf, rows, ckpt, self_ensemble)
        for name, gt, lr, bicubic, restored, r in items:
            add_image_page(pdf, name, gt, lr, bicubic, restored, r, args.dpi)

    bic = np.array([[r["bic_psnr"], r["bic_ssim"]] for r in rows]).mean(axis=0)
    mdl = np.array([[r["psnr"], r["ssim"]] for r in rows]).mean(axis=0)
    print(f"\n[Report] wrote {args.out}  "
          f"({os.path.getsize(args.out)/1e6:.1f} MB, {len(names)+1} pages)")
    print(f"  bicubic  {bic[0]:7.3f} dB   SSIM {bic[1]:.4f}")
    print(f"  restored {mdl[0]:7.3f} dB   SSIM {mdl[1]:.4f}")
    print(f"  gain     {mdl[0]-bic[0]:+7.3f} dB   SSIM {mdl[1]-bic[1]:+.4f}")
    print(f"  model wins on {sum(r['psnr'] > r['bic_psnr'] for r in rows)}/{len(rows)} images")


if __name__ == "__main__":
    main()
