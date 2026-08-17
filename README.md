# AI-Based Restoration of Degraded Semiconductor Inspection Images

> **SEMICON India Hackathon 2026 — Track 1 (KLA)**
> End-to-end DRUNet-SR image restoration, with a Physics-Aware Plug-and-Play (PnP-HQS) path retained as an ablation baseline

## Overview

This repository restores degraded semiconductor inspection images, handling three degradations at once:

1. **Gaussian noise** — additive sensor/electronic noise
2. **Speckle noise** — multiplicative coherent-imaging noise
3. **2× super-resolution** — restoring 128×128 → 256×256

**What ships:** `evaluate.py` defaults (`--mode e2e`) to an **end-to-end
DRUNet-SR** model — a single trained network, with an optional PixelShuffle
SR head, that maps the degraded 128×128 input directly to a restored 256×256
output. It is trained directly on real (NoisyLR, GT) pairs and evaluated
with an 8× geometric self-ensemble. This is the primary method, and the
numbers in [Results](#results) below are measured on it.

A second path, **Physics-Aware Plug-and-Play (PnP-HQS)** (`--mode pnp`), is
retained in the repository as an ablation baseline. Instead of learning the
degraded-to-clean mapping end-to-end, it explicitly models the degradation
physics and uses the same DRUNet architecture — here trained as a denoiser
rather than an SR network — as a learned image prior inside an iterative
Half-Quadratic Splitting (HQS) optimization loop. See
[Architecture](#architecture) and [PnP-HQS Algorithm](#pnp-hqs-algorithm)
for how it works, and `Report/ablation.md` for its ablation results.

### Key Innovation

No existing method combines Gaussian noise + speckle noise + super-resolution in a single PnP loop. The **unified data-fidelity term** in the PnP-HQS ablation path handles all three degradations in sequence via:
- An iterative gradient-descent SR update, following the HQS scheme in [DPIR](https://github.com/cszn/DPIR)
- Log-domain transform for speckle noise (converting multiplicative → additive)

## Architecture

*The diagram and algorithm below describe the PnP-HQS ablation path
(`--mode pnp`). The shipped default (`--mode e2e`) is a single forward pass
through the DRUNet backbone with a PixelShuffle SR head — see
[DRUNet Architecture](#drunet-architecture) and `utils/inference.py`.*

```
Input: Degraded LR image (128×128)
    │
    ▼
Bicubic Upsample → 256×256 (initialization)
    │
    ▼
┌── PnP-HQS Loop (K=15 iterations) ──────────────────────┐
│                                                        │
│   Step A: Data Fidelity (iterative, stable step size)  │
│   ├── Log-domain speckle transform                     │
│   ├── SR back-projection (gradient descent)            │
│   └── 30-step GD solve, step size 1/L                  │
│                                                        │
│   Step B: DRUNet Denoiser Prior                        │
│   ├── U-Net with residual blocks                       │
│   ├── Noise level map input (σ_k)                      │
│   └── σ decays geometrically per iter                  │
│                                                        │
└────────────────────────────────────────────────────────┘
    │
    ▼
Output: Restored image (256×256)
```

### DRUNet Architecture

| Component | Details |
|---|---|
| Input | 1 channel (shipped e2e model). The PnP denoiser variant takes 2: image + noise level map |
| Output | 1 channel (grayscale) |
| Encoder | 3 levels: 64 → 128 → 256 → 512 channels |
| Blocks | 4 ResBlocks per stage, 7 stages, 28 total |
| Downsampling | Strided convolution (2×2, stride 2) |
| Upsampling | Nearest-neighbour ×2 followed by a 3×3 convolution. The helper is named `upsample_convtranspose` for historical reasons but builds no transposed convolution; the model contains none |
| Skip connections | Concatenation + 1×1 conv |
| Parameters | 33,707,652 (33.7M) |
| Checkpoint (fp16, inference) | ~67 MB |

## Repository Structure

```
├── run.py                        # SUBMISSION ENTRY POINT: python run.py <in> <out>
├── README.md                     # This file
├── requirements.txt              # Python dependencies, all version-pinned
├── train.py                      # End-to-end training
├── evaluate.py                   # Older flag-based CLI, same outputs as run.py
├── compute_metrics.py            # PSNR / SSIM / LPIPS scoring
├── datasets.py                   # Real (NoisyLR, GT) pair dataset
├── augment_pipeline.py           # Calibrated synthetic degradation
├── estimate_parameters.py        # Reverse-engineers the degradation from real pairs
├── degradation_params.json       # Calibration output
├── make_eval_split.py            # Freezes the held-out split
├── eval_split.txt                # 320 held-out filenames
├── analysis.md                   # Working notes on the degradation analysis
├── Colab_Training_Runner.ipynb   # Notebook that reproduces training on Colab
├── models/
│   ├── drunet_sr_inference.pth   # Shipped weights: fp16 EMA, 67.5 MB
│   ├── basicblock.py             # ResBlock (LayerNorm + GELU), up/downsample
│   ├── drunet.py                 # DRUNet, optional PixelShuffle SR head
│   └── losses.py                 # Charbonnier + FFT + SSIM + Sobel + LPIPS
├── utils/
│   ├── utils_image.py            # I/O, PSNR, windowed SSIM, augmentation
│   ├── utils_pnp.py              # PnP-HQS loop (ablation path)
│   └── inference.py              # End-to-end inference + self-ensemble
├── tests/                        # pytest suite
├── restored_outputs/             # Restored test-set outputs (400 images)
└── Report/
    ├── ablation.md               # Held-out results + known limitations
    ├── make_comparison_report.py # Builds restoration_comparison.pdf
    ├── restoration_comparison.pdf# 101-page GT / input / bicubic / restored panels
    ├── report.py                 # Builds denoising_comparison.pdf
    ├── denoising_comparison.pdf  # Earlier denoising-only comparison
    ├── GT/                       # 101 ground-truth samples used by the reports
    ├── NoisyLR/                  # matching 101 degraded inputs
    └── restored_training_outputs/# matching 101 restored outputs
```

### Model weights

`run.py` and `evaluate.py` both load `models/drunet_sr_inference.pth`
automatically — the fp16 exponential-moving-average weights, 67.5 MB. No
arguments needed.

The weights sit in `models/` alongside the architecture code, matching the
folder layout the submission brief specifies. Both scripts fall back to
`model_weights/` if a clone still uses the older layout, so neither needs an
edit either way. `train.py --save_dir` also defaults to `models/`.

Round-tripping the weights through fp16 costs 94 dB PSNR against the fp32
originals, which is roughly 64 dB below the signal the model is scored on, so
the reduction is lossless in practice.

### Optional: the resumable training checkpoint

**Nothing needs to be downloaded to run this repository.** Inference uses the
67.5 MB checkpoint already committed at `models/drunet_sr_inference.pth`.

Training also writes `drunet_sr_best.pth`, a 539.7 MB checkpoint carrying the
live model, the EMA copy and both Adam moments. It exceeds GitHub's 100 MB
per-file limit, so it is gitignored and hosted externally instead:

> https://drive.google.com/drive/folders/1TuBgy8EQDuTciaLWzz8kP7zx2mfRHCQG?usp=sharing

That file is needed **only** for `python train.py --resume`, to continue an
interrupted training run. It is not used by `run.py`, by `evaluate.py`, or by
the test suite. If you only want to reproduce our results, ignore it entirely.

To use it, place it at `models/drunet_sr_best.pth`.

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

Verified from scratch in a clean virtualenv: the pinned set installs and runs
inference on all 400 test images with no manual steps.

`torch` is pinned without a CUDA local-version tag so the install stays
portable, which means PyPI serves the **CPU** wheel by default. Inference is
correct either way — `evaluate.py` selects CUDA when it is available and falls
back to CPU otherwise — but for GPU speed install the CUDA build first:

```bash
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

CPU and CUDA outputs were compared directly and agree to `1.6e-04` max absolute
difference, which is ordinary backend nondeterminism. Measured cost for the full
400-image test set, including the 8× self-ensemble:

| Backend | Per image | 400 images |
|---|---|---|
| CUDA (RTX 4060 Laptop) | 0.163 s | 1.1 min (measured) |
| CPU | 3.99 s | 26.6 min (extrapolated from 10) |

The 8 self-ensemble variants share a single forward pass rather than running as
8 batch-1 passes. Isolated on 24 images that is 0.2053 → 0.1386 s/img, a 1.48×
speedup for 0.52 GB peak memory. It is a GPU-side win only: the CPU path is
compute-bound and unchanged (4.03 → 3.99 s/img, within noise).

`torch.backends.cudnn.benchmark` is deliberately **not** enabled. Measured, it
returns ~2% while raising peak memory from 0.52 GB to 5.33 GB — a poor trade on
an 8.6 GB card, and it yields nothing once the variants are batched.

### 2. Run inference (this is what KLA's benchmark runs)

```bash
python run.py <input-dir> <output-dir>
```

`run.py` is the submission entry point and takes two positional arguments. It
reads every `.npy` in the input directory, creates the output directory if it
does not exist (including nested paths), and writes one restored `.npy` per
input under the same filename.

Each output is guaranteed to be grayscale `(H, W)`, `float32`, finite, inside
`[0, 1]`, and exactly 2× the input resolution. `run.py` verifies each of these
before writing and exits non-zero if any file or count is missing.

No internet access, API key, download, or manual configuration is required.
The weights ship in this repository, and the inference import chain never
touches `lpips`, which is a training-only dependency imported solely by
`models/losses.py`. Verified: importing the inference path pulls in no
network-capable module.

`evaluate.py` remains available for local experiments and takes the older
`--input_dir` / `--output_dir` flags plus `--mode pnp` for the Plug-and-Play
ablation path. Both scripts produce identical output.

### 3. Reproduce training

```bash
python make_eval_split.py --gt_dir <path_to_GT>
python train.py --data_dir <path_to_train_dir> --epochs 150 --batch_size 16
```

The training directory must contain `GT/` and `NoisyLR/` subdirectories.

### 4. Score against ground truth

```bash
python compute_metrics.py --restored_dir ./restored_val --gt_dir <path_to_GT> \
    --file_list eval_split.txt --baseline_dir <path_to_NoisyLR> --lpips
```

### 5. Run the test suite

```bash
python -m pytest tests/ -v
```

69 tests, ~100 s. No GPU or network required — the suite uses a bicubic test
double rather than the trained weights, and avoids constructing `HybridLoss`,
which would download AlexNet weights for its LPIPS term.

## Results

Held out on **320 images** (`eval_split.txt` — a seeded 10% shuffle of the
3200 real training pairs, never seen during training; see `Report/ablation.md`
for the full ablation table, including the PnP-HQS ablation row and error
budget):

| Configuration | PSNR (dB) | SSIM | LPIPS ↓ |
|---|---|---|---|
| Bicubic upsample of NoisyLR (control) | 22.780 | 0.5202 | 0.4645 |
| End-to-end DRUNet-SR, single pass | 23.884 | 0.5537 | 0.3405 |
| **End-to-end DRUNet-SR + 8× self-ensemble (shipped default)** | **26.706** | **0.7133** | **0.3019** |

**+3.926 dB PSNR over the bicubic control.** This is what
`python evaluate.py --input_dir <X> --output_dir <Y>` produces with no extra
flags — see step 2 of Quick Start.

Reproduce on the held-out split with:

```bash
python evaluate.py --input_dir ../Dataset/train/train/NoisyLR --output_dir ./restored_val
python compute_metrics.py --restored_dir ./restored_val --gt_dir ../Dataset/train/train/GT \
    --file_list eval_split.txt --baseline_dir ../Dataset/train/train/NoisyLR --lpips
```

### Training & inference cost

| | |
|---|---|
| Hardware | RTX 4060 Laptop, 8.6 GB, bf16 autocast |
| Training | 150 epochs, ~71 s/epoch, ~3.0 h total |
| Best checkpoint | **epoch 28** (selected by held-out PSNR) |
| Inference | 0.163 s/image with 8× self-ensemble (batched) |
| Test set | 400 images, 128×128 in → 256×256 out |

## Known limitations

The composite training loss is mis-scaled: its FFT (frequency-domain) term
takes **66.6%** of the weighted gradient signal, against **8.8%** for the
Charbonnier pixel term that actually drives PSNR. As a direct result, the
model peaked at **epoch 28** of 150 and declined afterward, so the shipped
checkpoint is that epoch-28 EMA rather than the final one. Correcting the
loss scaling and retraining is the single highest-value next step; see
`Report/ablation.md` for the full per-term breakdown and a
verified-but-deliberately-unapplied fix.

## Technical Details

### Degradation Model

```
y = D( η ⊙ (x + n_gaussian) )
```

- `x` = clean ground-truth image (256×256)
- `n_gaussian` = additive Gaussian noise
- `η` = multiplicative speckle noise (Gamma-distributed)
- `D(·)` = 2× average-pool downsampling
- `y` = observed degraded image (128×128)

### PnP-HQS Algorithm

For `k = 0, 1, ..., K-1`:

**Step A** (Data Fidelity — iterative, stable step size):
```
x_{k+1} = argmin_x  ||y - Dx||^2 / (2 mu_k)  +  ||x - z_k||^2 / 2
          solved by gradient descent with step 1/L, L = ||D||^2/mu_k + 1
```

**Step B** (Denoiser Prior):
```
z_{k+1} = DRUNet(x_{k+1}, σ_k)
```

With geometric schedules:
- `σ_k`: 49/255 → 1/255 (denoiser strength decreases)
- `μ_k`: 0.01 → 0.5 (data fidelity weight decreases)

### Speckle Handling

Log-domain transform converts multiplicative speckle to additive noise:
```
log(y) = log(x · η) = log(x) + log(η)
```
This allows the standard DPIR data-fidelity update to handle speckle noise.

## References

1. Zhang et al., "Plug-and-Play Image Restoration with Deep Denoiser Prior", TPAMI 2021 ([paper](https://arxiv.org/abs/2008.13751), [code](https://github.com/cszn/DPIR))
2. Zhang et al., "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN", TIP 2017 ([paper](https://arxiv.org/abs/1608.03981))
3. Zhu et al., "Denoising Diffusion Models for Plug-and-Play Image Restoration", CVPR 2023 ([paper](https://arxiv.org/abs/2305.08995))
4. Zhang, Q., "SAR Image Despeckling Based on Convolutional Denoising Autoencoder" ([paper](https://arxiv.org/pdf/2011.14627))

## Hardware

- **Training:** RTX 4060 Laptop (8.6 GB), bf16 autocast — 150 epochs in ~3.0 h. See [Results](#results) for the full breakdown.
- **Inference:** 0.163 s/image on GPU with the default 8× self-ensemble.

## License

This project is for the SEMICON India Hackathon 2026 evaluation.
