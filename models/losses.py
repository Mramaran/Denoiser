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
    """FFT-based frequency domain reconstruction loss"""
    def forward(self, pred, target):
        # Compute 2D Real FFT in float32 to prevent precision issues under AMP autocast
        with torch.cuda.amp.autocast(enabled=False):
            pred_f32 = pred.float()
            target_f32 = target.float()
            pred_fft = torch.fft.rfft2(pred_f32, norm="backward")
            target_fft = torch.fft.rfft2(target_f32, norm="backward")
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

class HybridLoss(nn.Module):
    """Charbonnier + FFT + SSIM + Sobel-gradient + LPIPS, with a warm-up on the
    perceptual terms."""
    def __init__(self, device='cuda', w_pixel=1.0, w_ssim=0.2, w_lpips=0.05,
                 w_freq=0.05, w_grad=0.1):
        super(HybridLoss, self).__init__()
        self.pixel_loss = CharbonnierLoss(eps=1e-3)
        self.freq_loss = FrequencyLoss()
        self.grad_loss = GradientLoss().to(device)

        # 'alex' is faster, more stable, and recommended for restoration backprop.
        # LPIPS is an AlexNet trained on natural RGB photos, so on grayscale
        # inspection images it is noisy signal - keep its weight low.
        self.lpips_loss = lpips.LPIPS(net='alex').to(device).eval()
        self.lpips_loss.requires_grad_(False)

        self.w_pixel = w_pixel
        self.w_ssim = w_ssim
        self.w_lpips = w_lpips
        self.w_freq = w_freq
        self.w_grad = w_grad

    def forward(self, pred, target, epoch=None, warm_up_epochs=10):
        loss_pix = self.pixel_loss(pred, target)
        loss_freq = self.freq_loss(pred, target)
        loss_grad = self.grad_loss(pred, target)
        loss_ssim = 1.0 - ssim(pred.float(), target.float(),
                               data_range=1.0, size_average=True)

        warmup_factor = _warmup_factor(epoch, warm_up_epochs)

        # LPIPS expects 3 channels in [-1, 1]; run it in float32 under autocast.
        with torch.cuda.amp.autocast(enabled=False):
            pred_rgb = pred.float().repeat(1, 3, 1, 1) * 2.0 - 1.0
            target_rgb = target.float().repeat(1, 3, 1, 1) * 2.0 - 1.0
            loss_lpips = torch.mean(self.lpips_loss(pred_rgb, target_rgb))

        return (self.w_pixel * loss_pix +
                self.w_freq * loss_freq +
                self.w_grad * loss_grad +
                self.w_ssim * warmup_factor * loss_ssim +
                self.w_lpips * warmup_factor * loss_lpips)
