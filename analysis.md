# Evaluation Report: DRUNet Image Restoration Pipeline

This report summarizes the architectural, logic, and loss bottlenecks identified in the KLA image restoration codebase, and proposes specific improvements to achieve a higher PSNR/SSIM/LPIPS metric on the H100 GPU platform.

---

## 1. Model Architecture & Bottlenecks

### Summary of Current Architecture
The model defined in [`models/drunet.py`](file:///c:/Users/Ashwin/Documents/Github/Semicon_Hack/KLA/models/drunet.py) is a Deep Residual U-Net (DRUNet) based on Zhang et al. (2021):
*   **Receptive Field / Resolution Hierarchy:** 4 resolution levels with channel dimensions `nc = [64, 128, 256, 512]`. Downsampling is performed using stride-2 convolutions (`downsample_strideconv` with a $2\times2$ kernel and no padding in [`models/basicblock.py:31`](file:///c:/Users/Ashwin/Documents/Github/Semicon_Hack/KLA/models/basicblock.py#L31)). Upsampling uses transpose convolutions (`upsample_convtranspose` with a $2\times2$ kernel in [`models/basicblock.py:34`](file:///c:/Users/Ashwin/Documents/Github/Semicon_Hack/KLA/models/basicblock.py#L34)).
*   **Skip Connections:** Symmetrical skip connections concatenate encoder feature maps with decoder feature maps (e.g., [`models/drunet.py:67`](file:///c:/Users/Ashwin/Documents/Github/Semicon_Hack/KLA/models/drunet.py#L67)), followed by a $1\times1$ convolution to project channels back (e.g., `nc[2]*2 -> nc[2]`).
*   **Residual Blocks:** Each scale contains `nb = 4` sequential ResBlocks. The block design in [`models/basicblock.py:21`](file:///c:/Users/Ashwin/Documents/Github/Semicon_Hack/KLA/models/basicblock.py#L21) uses:
    $$\text{ResBlock}(x) = x + \text{Conv2d}(\text{ReLU}(\text{Conv2d}(x)))$$
*   **Noise-Level Conditioning:** The model accepts a noise level $\sigma$ and concatenates a spatially tiled noise map of shape $(B, 1, H, W)$ to the input image channel dimension inside `forward()` ([`models/drunet.py:51`](file:///c:/Users/Ashwin/Documents/Github/Semicon_Hack/KLA/models/drunet.py#L51)), resulting in a 2-channel input.

### Architectural Bottlenecks & Deviations from Reference Architectures
1.  **Lack of Normalization Layers:** There are no normalization layers (e.g., BatchNorm, LayerNorm, or InstanceNorm) anywhere in the model body or ResBlocks. Standard networks like NAFNet or Restormer rely heavily on normalization (LayerNorm) to stabilize training, especially when handling out-of-distribution values (exceeding $[0, 1]$).
2.  **No Attention Modules:** Modern restoration backbones (Restormer, NAFNet, SwinIR) use self-attention or channel/spatial attention. DRUNet contains no attention modules, reducing its capacity to selectively weigh signals under heavy speckle (multiplicative) degradation.
3.  **Kernel Artifacts from Down/Upsampling:** Downsampling with a $2\times2$ stride-2 convolution and upsampling with a $2\times2$ transpose convolution (both with 0 padding) are prone to grid/block boundary artifacts. Standard reference architectures use $3\times3$ or $4\times4$ convolutions with padding for smooth transitions.
4.  **No Dynamic Range Adaptation:** Since pixel values can exceed $[0, 1]$, the model's reliance on standard `ReLU` (which clamps negative values to 0 but is unbounded on the positive end) without any scaling layer makes it sensitive to out-of-distribution high values. Replacing `ReLU` with `LeakyReLU` or `GELU` would improve gradient flow at negative tails.

---

## 2. Weight Update Logic & Validation Loop

### Weight Update Logic Trace
Located in [`train.py:114`](file:///c:/Users/Ashwin/Documents/Github/Semicon_Hack/KLA/train.py#L114):
*   **Optimizer:** `optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))` ([`train.py:263`](file:///c:/Users/Ashwin/Documents/Github/Semicon_Hack/KLA/train.py#L263)). Initial learning rate is `1e-4`.
*   **LR Scheduler:** `CosineAnnealingLR` with `T_max = args.epochs` and `eta_min = 1e-6` ([`train.py:264`](file:///c:/Users/Ashwin/Documents/Github/Semicon_Hack/KLA/train.py#L264)). Steps at the end of each epoch ([`train.py:295`](file:///c:/Users/Ashwin/Documents/Github/Semicon_Hack/KLA/train.py#L295)).
*   **Mixed Precision / Accumulation / EMA:** Currently, no Mixed Precision (AMP), gradient accumulation, or Exponential Moving Average (EMA) of weights is implemented.

### Validation & Correctness Bugs Found
1.  **Randomized/Augmented Validation Split (Critical Bug):** 
    The validation dataset is created by calling `random_split` on `full_dataset` ([`train.py:234`](file:///c:/Users/Ashwin/Documents/Github/Semicon_Hack/KLA/train.py#L234)), where `full_dataset` has `augment=True`. As a result, the validation dataloader applies **random crop, random flips/rotations, and random noise levels on-the-fly** to the validation images every epoch. 
    *   *Consequence:* The validation PSNR and loss fluctuate randomly every epoch. Selecting the "best" model checkpoint based on this noisy metric is unreliable (e.g., a checkpoint might be saved just because it randomly received a clean or easy-to-denoise crop).
2.  **Clipped PSNR Calculation vs. Unclipped GT:**
    In [`train.py:163-165`](file:///c:/Users/Ashwin/Documents/Github/Semicon_Hack/KLA/train.py#L163), the validation PSNR clips the model prediction to $[0, 1]$:
    ```python
    pred = output[i, 0].cpu().numpy().clip(0, 1)
    target = gt[i, 0].cpu().numpy()
    ```
    However, if GT contains values exceeding $[0, 1]$, this clipping introduces a metric mismatch. (Note: Our dataset inspection showed GT max is exactly `1.0` and min is `0.0`, but NoisyLR has values up to `1.32` and below `0.0`).
3.  **H100 Throughput Underutilization:**
    `batch_size = 16` and `patch_size = 128` with `num_workers = 0` on an H100 GPU is highly sub-optimal. Training can be significantly accelerated by increasing `num_workers` (e.g., 4 or 8) and `batch_size` (e.g., 32 or 64), combined with Mixed Precision (AMP) to save memory and maximize tensor core utilization.

---

## 3. Loss Function Analysis

### Current Loss Function
The model is trained using plain L1 loss: `criterion = nn.L1Loss()` ([`train.py:262`](file:///c:/Users/Ashwin/Documents/Github/Semicon_Hack/KLA/train.py#L262)).

### Why It Caps PSNR at 33.5 dB
1.  **L1 Under-optimizes SSIM & LPIPS:** Minimizing L1 loss (pixel-level differences) produces structurally blurred results under heavy noise, since L1 treats all pixels independently. The evaluation metric is a weighted combination of **PSNR, SSIM, and LPIPS**. A model optimized solely on L1 will perform poorly on SSIM and LPIPS.
2.  **No Speckle/Degradation Adaptation:** The training dataset `DenoisingDataset` in `train.py` ONLY trains the denoiser on clean images with additive Gaussian noise. It is completely blind to multiplicative speckle noise, downsampling operators, and clipping distortions. Although the PnP physics loop tries to handle these at inference, the denoiser is out-of-distribution because it has never seen log-domain artifacts or structured residuals.

---

## 4. Hybrid Loss Function Feasibility

A hybrid loss combining pixel, structural (SSIM), perceptual (LPIPS), and frequency losses will directly optimize for the evaluation metric.

### LPIPS Feasibility on H100
*   *Compute Cost:* On H100, a VGG/AlexNet forward pass for LPIPS is extremely fast. However, to maintain high throughput, we can evaluate LPIPS every step (which is fine on H100) or calculate it using downscaled feature maps to reduce memory usage.
*   *Implementation:* We can use the official `lpips` library.
*   *Gradients:* All terms are summed into a single loss scalar, allowing a single `loss.backward()` call for correct gradient flow.

### Loss Code Sketch
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips  # pip install lpips

class CharbonnierLoss(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps2 = eps ** 2
        
    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.eps2))

class FrequencyLoss(nn.Module):
    def forward(self, pred, target):
        # Compute 2D FFT
        pred_fft = torch.fft.rfft2(pred, norm="backward")
        target_fft = torch.fft.rfft2(target, norm="backward")
        return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))

class HybridLoss(nn.Module):
    def __init__(self, device='cuda', w_pixel=1.0, w_ssim=0.2, w_lpips=0.05, w_freq=0.05, w_grad=0.1):
        super().__init__()
        self.pixel_loss = CharbonnierLoss()
        self.freq_loss = FrequencyLoss()
        self.lpips_loss = lpips.LPIPS(net='alex').to(device).eval()  # 'alex' is fast and lightweight
        
        self.w_pixel = w_pixel
        self.w_ssim = w_ssim
        self.w_lpips = w_lpips
        self.w_freq = w_freq
        self.w_grad = w_grad
        
    def forward(self, pred, target, epoch=None, warm_up_epochs=10):
        # Pixel loss (Charbonnier)
        loss_pix = self.pixel_loss(pred, target)
        
        # Frequency domain loss
        loss_freq = self.freq_loss(pred, target)
        
        # Perceptual / SSIM Warm-up Strategy:
        # Avoid noisy LPIPS gradients early in training by scaling weight
        lpips_factor = min(1.0, epoch / warm_up_epochs)
        
        # LPIPS expects 3-channel input scaled to [-1, 1]
        pred_rgb = pred.repeat(1, 3, 1, 1) * 2.0 - 1.0
        target_rgb = target.repeat(1, 3, 1, 1) * 2.0 - 1.0
        loss_lpips = torch.mean(self.lpips_loss(pred_rgb, target_rgb))
        
        # Total loss
        total_loss = (self.w_pixel * loss_pix + 
                      self.w_freq * loss_freq + 
                      self.w_lpips * lpips_factor * loss_lpips)
        return total_loss
```

---

## 5. Data Augmentation & Degradation Logic

Currently, training is blind to the real degradation pipeline. We need to wire the calibrated degradation pipeline from [`augment_pipeline.py`](file:///c:/Users/Ashwin/Documents/Github/Semicon_Hack/augment_pipeline.py) (which uses parameters calibrated in `degradation_params.json` via statistical analysis) into the active training code.

### Augmentation Sketch (Integrating with train.py)
We can replace `DenoisingDataset` in `train.py` with `CalibratedDegradationDataset` (or import/subclass it) so the model trains on exact replicas of the real degradation:

```python
# Integration sketch inside train.py:
from augment_pipeline import CalibratedDegradationDataset

# Replace the training dataset:
train_dataset = CalibratedDegradationDataset(
    gt_dir=gt_dir,
    params_path='../degradation_params.json',  # point to calibrated parameters
    patch_size=args.patch_size,
    is_train=True
)

# Since CalibratedDegradationDataset returns (noisy, lr_ref, gt),
# we train the model to map noisy -> lr_ref (or directly map noisy -> gt for end-to-end learning)
```

---

## Prioritized Action Plan (Top 5 Changes)

| Priority | Change Proposed | Expected Gain (PSNR/SSIM) | Expected Effort | Description |
| :--- | :--- | :---: | :---: | :--- |
| **1** | **Fix Validation Loop & Split** | Stabilization / Metrics | **S** (Low) | Separate a fixed, unaugmented subset of validation images. Remove random noise, flips, and crops from validation batches to make checkpoint selection reliable. |
| **2** | **Integrate Calibrated Degradation** | $+1.5\text{ to }2.0\text{ dB}$ | **M** (Med) | Replace the pure Gaussian noise loader with `CalibratedDegradationDataset` from `augment_pipeline.py` to train the denoiser on matched distributions. |
| **3** | **Implement Hybrid Loss Module** | Significant SSIM/LPIPS boost | **M** (Med) | Deploy `Charbonnier + FFT + LPIPS` loss with a warm-up scheduler to optimize directly for the target metrics. |
| **4** | **Upgrade Normalization & Activations** | $+0.4\text{ to }0.8\text{ dB}$ | **S** (Low) | Add LayerNorm or InstanceNorm to blocks and swap `ReLU` for `LeakyReLU` / `GELU` to handle out-of-distribution values ($> 1.0$) smoothly. |
| **5** | **Optimize Throughput (AMP + Batch)** | Speed / Efficiency | **S** (Low) | Enable PyTorch AMP (`torch.cuda.amp.autocast`), increase batch size to 32/64, and set `num_workers > 0` to exploit H100 capabilities. |
