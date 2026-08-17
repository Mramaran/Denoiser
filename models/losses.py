import torch
import torch.nn as nn
import torch.nn.functional as F
import lpips
from pytorch_msssim import ssim

# Reference:
# Zhang et al., "Plug-and-Play Image Restoration with Deep Denoiser Prior", TPAMI 2021.
# Charbonnier / frequency / LPIPS losses standard implementations.

class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 variant for robust image restoration)"""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        return torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))

class FrequencyLoss(nn.Module):
    """FFT-based frequency domain reconstruction loss.

    `norm` selects the FFT normalisation. Per the torch.fft docs, "backward"
    applies no normalisation while "ortho" divides by sqrt(H*W), so at 256x256
    the "backward" magnitudes are 256x larger. That factor is why the original
    weighting handed this term 66.6% of the weighted gradient against 8.8% for
    the Charbonnier term that actually drives PSNR -- see Report/ablation.md.
    "ortho" is the default; "backward" reproduces the original objective.
    """
    def __init__(self, norm="ortho"):
        super(FrequencyLoss, self).__init__()
        if norm not in ("ortho", "backward"):
            raise ValueError(f"norm must be 'ortho' or 'backward', got {norm!r}")
        self.norm = norm

    def forward(self, pred, target):
        # Compute 2D Real FFT in float32 to prevent precision issues under AMP autocast
        with torch.amp.autocast(device_type="cuda", enabled=False):
            pred_f32 = pred.float()
            target_f32 = target.float()
            pred_fft = torch.fft.rfft2(pred_f32, norm=self.norm)
            target_fft = torch.fft.rfft2(target_f32, norm=self.norm)
            return F.l1_loss(torch.abs(pred_fft), torch.abs(target_fft))

class GradientLoss(nn.Module):
    """Sobel-gradient L1 loss.

    Penalises edge and structure mismatch directly, which is what SSIM
    rewards. Used by the NTIRE 2026 IK-LAB restoration entry alongside
    Charbonnier and FFT losses.
    """
    def __init__(self):
        super(GradientLoss, self).__init__()
        kx = torch.tensor([[-1., 0., 1.],
                           [-2., 0., 2.],
                           [-1., 0., 1.]]).view(1, 1, 3, 3)
        self.register_buffer("kx", kx)
        self.register_buffer("ky", kx.transpose(2, 3).contiguous())

    def forward(self, pred, target):
        kx, ky = self.kx.to(pred.dtype), self.ky.to(pred.dtype)
        gx = F.l1_loss(F.conv2d(pred, kx, padding=1), F.conv2d(target, kx, padding=1))
        gy = F.l1_loss(F.conv2d(pred, ky, padding=1), F.conv2d(target, ky, padding=1))
        return gx + gy

def _warmup_factor(epoch=None, warm_up_epochs=10):
    """Perceptual-term warm-up weight.

    epoch=None means "fully warmed up" and is what validation passes, so that
    validation scores the same objective training optimises. An explicit
    integer gives the linear ramp used during training. Kept as a module-level
    pure function so it is unit-testable without constructing HybridLoss,
    which would download AlexNet weights.
    """
    if epoch is None:
        return 1.0
    return min(1.0, epoch / max(1, warm_up_epochs))

# Term weights, paired with the FFT normalisation they were tuned for.
#
# "backward" is the original objective: it produced the shipped epoch-28
# checkpoint, and `train.py --loss_norm backward` still reproduces it exactly.
# Its measured split was 8.8% pixel / 66.6% freq / 13.7% SSIM / 7.8% grad /
# 3.1% LPIPS -- frequency-dominated, and the model peaked at epoch 28 of 150.
#
# "ortho" is the correction. Dividing the frequency magnitudes by sqrt(H*W)
# brings that term into the same range as Charbonnier, and these weights then
# give a pixel-dominant 58/5/23/10/4 split. Numbers from Report/ablation.md.
LOSS_PRESETS = {
    "ortho":    dict(w_pixel=1.0, w_freq=0.15, w_ssim=0.05, w_grad=0.02, w_lpips=0.01),
    "backward": dict(w_pixel=1.0, w_freq=0.05, w_ssim=0.20, w_grad=0.10, w_lpips=0.05),
}


class HybridLoss(nn.Module):
    """Charbonnier + FFT + SSIM + Sobel-gradient + LPIPS, with a warm-up on the
    perceptual terms.

    `freq_norm` picks both the FFT normalisation and the matching weight preset
    from LOSS_PRESETS. Individual weights may still be overridden explicitly.
    """
    def __init__(self, device='cuda', freq_norm='ortho', w_pixel=None,
                 w_ssim=None, w_lpips=None, w_freq=None, w_grad=None):
        super(HybridLoss, self).__init__()
        if freq_norm not in LOSS_PRESETS:
            raise ValueError(f"freq_norm must be one of {sorted(LOSS_PRESETS)}, "
                             f"got {freq_norm!r}")
        preset = LOSS_PRESETS[freq_norm]

        self.pixel_loss = CharbonnierLoss(eps=1e-3)
        self.freq_loss = FrequencyLoss(norm=freq_norm)
        self.grad_loss = GradientLoss().to(device)

        # 'alex' is faster, more stable, and recommended for restoration backprop.
        # LPIPS is an AlexNet trained on natural RGB photos, so on grayscale
        # inspection images it is noisy signal - keep its weight low.
        self.lpips_loss = lpips.LPIPS(net='alex').to(device).eval()
        self.lpips_loss.requires_grad_(False)

        # An explicit argument overrides the preset; None falls back to it.
        self.freq_norm = freq_norm
        self.w_pixel = preset["w_pixel"] if w_pixel is None else w_pixel
        self.w_ssim = preset["w_ssim"] if w_ssim is None else w_ssim
        self.w_lpips = preset["w_lpips"] if w_lpips is None else w_lpips
        self.w_freq = preset["w_freq"] if w_freq is None else w_freq
        self.w_grad = preset["w_grad"] if w_grad is None else w_grad

    def forward(self, pred, target, epoch=None, warm_up_epochs=10):
        loss_pix = self.pixel_loss(pred, target)
        loss_freq = self.freq_loss(pred, target)
        loss_grad = self.grad_loss(pred, target)
        loss_ssim = 1.0 - ssim(pred.float(), target.float(),
                               data_range=1.0, size_average=True)

        warmup_factor = _warmup_factor(epoch, warm_up_epochs)

        # LPIPS expects 3 channels in [-1, 1]; run it in float32 under autocast.
        with torch.amp.autocast(device_type="cuda", enabled=False):
            pred_rgb = pred.float().repeat(1, 3, 1, 1) * 2.0 - 1.0
            target_rgb = target.float().repeat(1, 3, 1, 1) * 2.0 - 1.0
            loss_lpips = torch.mean(self.lpips_loss(pred_rgb, target_rgb))

        return (self.w_pixel * loss_pix +
                self.w_freq * loss_freq +
                self.w_grad * loss_grad +
                self.w_ssim * warmup_factor * loss_ssim +
                self.w_lpips * warmup_factor * loss_lpips)
