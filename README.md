# AI-Based Restoration of Degraded Semiconductor Inspection Images

> **SEMICON India Hackathon 2026 — Track 1 (KLA)**

A single end-to-end **DRUNet-SR** network restores degraded 128×128 inspection images to clean 256×256 — removing **Gaussian noise**, **speckle noise**, and performing **2× super-resolution** in one forward pass.

## Architecture

### Shipped pipeline: End-to-End DRUNet-SR 

```
Input: Degraded LR image (128×128)
    │
    ▼
8× Geometric Self-Ensemble (flips + rotations, one batched pass)
    │
    ▼
┌── DRUNet-SR: single forward pass (33.7M params) ───────┐
│                                                        │
│   Encoder (features stay at LR resolution)             │
│   ├── 3×3 Conv → 64 channels                           │
│   ├── 4× ResBlock @ 64   ──┐ skip                      │
│   ├── 4× ResBlock @ 128  ──┤ skip   (stride-2 ↓)       │
│   └── 4× ResBlock @ 256  ──┤ skip   (stride-2 ↓)       │
│                            │                           │
│   Bottleneck               │                           │
│   └── 4× ResBlock @ 512    │                           │
│                            │                           │
│   Decoder                  │                           │
│   ├── 4× ResBlock @ 256  ◄─┤ concat + 1×1 conv (↑)     │
│   ├── 4× ResBlock @ 128  ◄─┤ concat + 1×1 conv (↑)     │
│   └── 4× ResBlock @ 64   ◄─┘ concat + 1×1 conv (↑)     │
│                                                        │
│   SR Head                                              │
│   ├── 3×3 Conv → PixelShuffle ×2 (no checkerboard)     │
│   └── + Bicubic×2(input)  ← global residual            │
│                                                        │
└────────────────────────────────────────────────────────┘
    │
    ▼
Inverse transforms → average the 8 outputs
    │
    ▼
Output: Restored image (256×256, float32, [0, 1])
```

## Results

Held out on 320 never-seen images (seeded 10% split, `eval_split.txt`):

| Configuration | PSNR (dB) | SSIM | LPIPS ↓ |
|---|---|---|---|
| Bicubic upsample (control) | 22.780 | 0.5202 | 0.4645 |
| DRUNet-SR, single pass | 23.884 | 0.5537 | 0.3405 |
| **DRUNet-SR + 8× self-ensemble (shipped)** | **26.706** | **0.7133** | **0.3019** |

**+3.93 dB PSNR over bicubic**, at **0.163 s/image** on an RTX 4060 Laptop.
Trained from scratch in **~3 hours** on a single laptop GPU.

## Run It 

```bash
git clone https://github.com/Mramaran/Denoiser.git
cd Denoiser
pip install -r requirements.txt
python run.py <input-dir> <output-dir>
```

## How It Works

- **End-to-end learning on real pairs** — trained directly on (NoisyLR, GT) pairs with a composite Charbonnier + FFT + SSIM + Sobel + LPIPS loss, so the network learns the actual degradation instead of an assumed one.
- **Calibrated degradation model** — `estimate_parameters.py` reverse-engineers the noise/blur parameters from the real data to drive synthetic augmentation.
- **8× geometric self-ensemble** — flips/rotations averaged at test time, batched into a single forward pass (+2.8 dB for 1.48× the runtime, not 8×).
- **Physics-aware PnP-HQS ablation** — an interpretable Plug-and-Play baseline (log-domain speckle transform + iterative SR data-fidelity inside HQS) is retained in the repo (`--mode pnp` in `evaluate.py`) and benchmarked in `Report/ablation.md`.

## Repository Map

```
run.py                      # SUBMISSION ENTRY POINT: python run.py <in> <out>
train.py                    # Reproduce training (~3 h on one laptop GPU)
compute_metrics.py          # PSNR / SSIM / LPIPS scoring
models/                     # DRUNet-SR (33.7M params) + shipped weights
utils/inference.py          # Batched self-ensemble inference
tests/                      # 69 pytest tests, no GPU/network needed
Report/                     # Ablations + 101-page visual comparison PDF
restored_outputs/           # All 400 restored test-set images, precomputed
```

## Reproduce

```bash
# Training
python train.py --data_dir <train_dir> --epochs 150 --batch_size 16

# Scoring on the held-out split
python compute_metrics.py --restored_dir ./restored_val --gt_dir <GT> \
    --file_list eval_split.txt --baseline_dir <NoisyLR> --lpips

# Tests (69 pass, ~100 s, CPU-only)
python -m pytest tests/ -v
```

## Engineering Honesty

The composite loss is mis-scaled (FFT term takes 66.6% of gradient signal vs 8.8% for the pixel term), so the model peaked at epoch 28 — the shipped checkpoint is that epoch's EMA. The diagnosed fix is documented in `Report/report.md` and is the highest-value next step. We ship what we measured, not what we hoped.

## References

1. Zhang et al., "Plug-and-Play Image Restoration with Deep Denoiser Prior", TPAMI 2021 ([DPIR](https://github.com/cszn/DPIR))
2. Zhang et al., "Beyond a Gaussian Denoiser", TIP 2017
3. Zhu et al., "Denoising Diffusion Models for Plug-and-Play Image Restoration", CVPR 2023
4. Zhang, Q., "SAR Image Despeckling Based on Convolutional Denoising Autoencoder"