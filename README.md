# AI-Based Restoration of Degraded Semiconductor Inspection Images

> **SEMICON India Hackathon 2026 — Track 1 (KLA)**
> Physics-Aware Plug-and-Play (PnP-HQS) Image Restoration

## Overview

This repository implements a **Physics-Aware Plug-and-Play (PnP)** image restoration pipeline that simultaneously handles:

1. **Gaussian noise** — additive sensor/electronic noise
2. **Speckle noise** — multiplicative coherent-imaging noise
3. **2× super-resolution** — restoring 128×128 → 256×256

Unlike standard end-to-end deep learning approaches, we explicitly model the degradation physics and use a trained **DRUNet** denoiser only as a learned image prior within an iterative **Half-Quadratic Splitting (HQS)** optimization loop.

### Key Innovation

No existing method combines Gaussian noise + speckle noise + super-resolution in a single PnP loop. Our **unified data-fidelity term** handles all three degradations in sequence via:
- FFT-based closed-form SR update (following [DPIR](https://github.com/cszn/DPIR))
- Log-domain transform for speckle noise (converting multiplicative → additive)

## Architecture

```
Input: Degraded LR image (128×128)
    │
    ▼
Bicubic Upsample → 256×256 (initialization)
    │
    ▼
┌─── PnP-HQS Loop (K=15 iterations) ───┐
│                                         │
│   Step A: Data Fidelity (FFT)          │
│   ├── Log-domain speckle transform     │
│   ├── SR back-projection (frequency)   │
│   └── Closed-form solution             │
│                                         │
│   Step B: DRUNet Denoiser Prior        │
│   ├── U-Net with residual blocks       │
│   ├── Noise level map input (σ_k)      │
│   └── σ decays geometrically per iter  │
│                                         │
└─────────────────────────────────────────┘
    │
    ▼
Output: Restored image (256×256)
```

### DRUNet Architecture

| Component | Details |
|---|---|
| Input | 2 channels (image + noise level map) |
| Output | 1 channel (grayscale) |
| Encoder | 3 levels: 64 → 128 → 256 → 512 channels |
| Blocks | 4 ResBlocks per level (32 total) |
| Downsampling | Strided convolution (stride 2) |
| Upsampling | Transposed convolution (stride 2) |
| Skip connections | Concatenation + 1×1 conv |
| Parameters | ~5M |

## Repository Structure

```
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── train.py                   # DRUNet denoiser training script
├── evaluate.py                # Inference script (KLA CLI interface)
├── compute_metrics.py         # PSNR/SSIM evaluation
├── models/
│   ├── __init__.py
│   ├── basicblock.py          # ResBlock, conv, up/downsample blocks
│   └── drunet.py              # DRUNet architecture
├── utils/
│   ├── __init__.py
│   ├── utils_image.py         # Image I/O, PSNR, SSIM, augmentation
│   └── utils_pnp.py           # PnP-HQS loop, FFT data fidelity
├── model_weights/
│   └── drunet_denoiser_best.pth  # Trained model weights
└── restored_outputs/          # Test set restoration results
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch ≥ 2.0 is required. Install with CUDA support for GPU acceleration:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
> ```

### 2. Train the Denoiser

```bash
python train.py --data_dir <path_to_train_dir> --epochs 200 --batch_size 16
```

The training directory should contain a `GT/` subdirectory with ground-truth `.npy` images.

### 3. Run Inference

```bash
python evaluate.py --input_dir <path_to_test_npy> --output_dir ./restored_outputs
```

This loads the best trained model from `model_weights/` and restores all input images.

### 4. Evaluate Metrics (if GT available)

```bash
python compute_metrics.py --restored_dir ./restored_outputs --gt_dir <path_to_GT>
```

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

**Step A** (Data Fidelity — closed-form FFT):
```
x_{k+1} = F^{-1}( (μ_k · F(z_k) + F(D^T y)) / (μ_k + F(D^T D)) )
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

- **Training:** GPU recommended (2-4 hours on single GPU, ~8 hours on CPU)
- **Inference:** ~0.5s per image on GPU, ~3s on CPU

## License

This project is for the SEMICON India Hackathon 2026 evaluation.
