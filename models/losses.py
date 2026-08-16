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

class HybridLoss(nn.Module):
    """Hybrid loss combining Pixel, Frequency, SSIM, and LPIPS losses with warmup"""
    def __init__(self, device='cuda', w_pixel=1.0, w_ssim=0.2, w_lpips=0.1, w_freq=0.05):
        super(HybridLoss, self).__init__()
        self.pixel_loss = CharbonnierLoss(eps=1e-3)
        self.freq_loss = FrequencyLoss()
        
        # 'alex' is faster, more stable, and recommended for general restoration backprop
        # weights are frozen during init, and eval mode is set to disable dropout/batchnorm updates
        self.lpips_loss = lpips.LPIPS(net='alex').to(device).eval()
        self.lpips_loss.requires_grad_(False)
        
        self.w_pixel = w_pixel
        self.w_ssim = w_ssim
        self.w_lpips = w_lpips
        self.w_freq = w_freq

    def forward(self, pred, target, epoch=0, warm_up_epochs=10):
        # 1. Pixel loss (Charbonnier L1)
        loss_pix = self.pixel_loss(pred, target)

        # 2. Frequency loss (FFT L1)
        loss_freq = self.freq_loss(pred, target)

        # 3. SSIM loss (1 - SSIM)
        # ssim expects data range [0, 1] or similar. We use data_range=1.0.
        # Ensure it is run in float32.
        loss_ssim = 1.0 - ssim(pred, target, data_range=1.0, size_average=True)

        # 4. LPIPS perceptual loss (AlexNet)
        # Scaled weight warm-up scheduler: scales linearly from 0 to 1 over first 10 epochs
        warmup_factor = min(1.0, epoch / max(1, warm_up_epochs))

        # LPIPS expects 3 channels and [-1, 1] range.
        # We repeat the grayscale channel 3 times.
        # Run LPIPS in float32 to prevent precision/underflow issues under AMP autocast
        with torch.cuda.amp.autocast(enabled=False):
            pred_rgb = pred.float().repeat(1, 3, 1, 1) * 2.0 - 1.0
            target_rgb = target.float().repeat(1, 3, 1, 1) * 2.0 - 1.0
            loss_lpips = torch.mean(self.lpips_loss(pred_rgb, target_rgb))

        # Combine all loss terms
        total_loss = (self.w_pixel * loss_pix + 
                      self.w_freq * loss_freq + 
                      self.w_ssim * warmup_factor * loss_ssim + 
                      self.w_lpips * warmup_factor * loss_lpips)
        
        return total_loss
