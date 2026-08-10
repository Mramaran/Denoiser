"""
PnP-HQS Restoration Pipeline

Implements Physics-Aware Plug-and-Play image restoration for:
- Gaussian noise removal
- Speckle noise removal (via log-domain transform)
- 2x Super-Resolution

Based on DPIR (Zhang et al., TPAMI 2021) with novel speckle handling.
"""

import torch
import torch.nn.functional as F
import numpy as np
import math


def bicubic_upsample(img_tensor, scale_factor=2):
    """Upsample a tensor by scale_factor using bicubic interpolation.
    Args:
        img_tensor: (B, C, H, W) tensor
        scale_factor: int, upsampling factor
    Returns:
        (B, C, H*sf, W*sf) tensor
    """
    return F.interpolate(img_tensor, scale_factor=scale_factor, 
                         mode='bicubic', align_corners=False)


def downsample(img_tensor, scale_factor=2):
    """Downsample a tensor by scale_factor using strided averaging.
    This is the forward degradation operator D.
    Args:
        img_tensor: (B, C, H, W) tensor  
        scale_factor: int
    Returns:
        (B, C, H//sf, W//sf) tensor
    """
    return F.avg_pool2d(img_tensor, kernel_size=scale_factor, stride=scale_factor)


def upsample_transpose(img_tensor, scale_factor=2, target_size=None):
    """Transpose of the downsample operator (D^T).
    Zero-fill upsampling followed by scaling.
    Args:
        img_tensor: (B, C, H, W) tensor
        scale_factor: int
        target_size: (H_target, W_target) tuple
    Returns:
        (B, C, H*sf, W*sf) tensor
    """
    B, C, H, W = img_tensor.shape
    if target_size is None:
        target_size = (H * scale_factor, W * scale_factor)
    # D^T for avg_pool is: place values in top-left of each sf x sf block, multiply by sf^2
    # Equivalent to nearest-neighbor upsampling (since avg_pool averages sf^2 pixels)
    out = torch.zeros(B, C, target_size[0], target_size[1], 
                      device=img_tensor.device, dtype=img_tensor.dtype)
    out[:, :, ::scale_factor, ::scale_factor] = img_tensor
    return out


def data_fidelity_sr(y, z_k, mu, scale_factor=2):
    """FFT-based closed-form data fidelity step for super-resolution.
    
    Solves: x_{k+1} = argmin_x (1/(2*mu)) * ||y - Dx||^2 + (1/2) * ||x - z_k||^2
    
    Following DPIR (Zhang et al., 2021), this has a closed-form solution
    in the frequency domain.
    
    For the simple average-pool downsampling case:
    x = F^{-1}( (mu * F(z_k) + F(D^T y)) / (mu + F(D^T D)) )
    
    Args:
        y: observed LR image (B, C, H_lr, W_lr)
        z_k: current estimate from denoiser (B, C, H_hr, W_hr)  
        mu: penalty parameter (float)
        scale_factor: downsampling factor
    Returns:
        x_{k+1}: updated estimate (B, C, H_hr, W_hr)
    """
    sf = scale_factor
    B, C, H, W = z_k.shape
    
    # D^T y: transpose of downsample applied to y
    # For avg_pool with sf, D^T places each pixel into a sf x sf block
    DTy = torch.zeros_like(z_k)
    DTy[:, :, ::sf, ::sf] = y  # Place LR pixels at grid positions
    
    # D^T D: this is a diagonal operator in spatial domain
    # For avg_pool: D^T D has value 1/(sf^2) at grid positions, 0 elsewhere
    # In frequency domain, we can compute this via FFT
    # Create the D^T D kernel
    DTD = torch.zeros_like(z_k)
    DTD[:, :, ::sf, ::sf] = 1.0 / (sf * sf)
    
    # Solve in frequency domain
    z_fft = torch.fft.fft2(z_k)
    DTy_fft = torch.fft.fft2(DTy)
    DTD_fft = torch.fft.fft2(DTD)
    
    # Closed-form solution
    numerator = mu * z_fft + DTy_fft
    denominator = mu + DTD_fft
    
    x_fft = numerator / denominator
    x = torch.fft.ifft2(x_fft).real
    
    return x


def log_domain_transform(y, eps=1e-3):
    """Transform image to log domain to convert multiplicative speckle noise
    to additive noise.
    
    Based on SAR despeckling literature (Zhang, 2020; arxiv:2011.14627).
    log(y) = log(x * eta) = log(x) + log(eta)
    where eta is multiplicative speckle noise (Gamma-distributed).
    
    Args:
        y: input image tensor (may contain values <= 0 due to Gaussian noise)
        eps: small constant for numerical stability
    Returns:
        log-transformed image
    """
    return torch.log(torch.clamp(y, min=eps))


def exp_domain_transform(y_log, eps=1e-3):
    """Inverse of log_domain_transform.
    Args:
        y_log: log-domain image
        eps: same eps used in forward transform
    Returns:
        spatial-domain image
    """
    return torch.exp(y_log)


def noise_level_schedule(iteration, total_iterations, sigma_max=49.0/255.0, sigma_min=1.0/255.0):
    """Geometric noise level schedule for the PnP loop.
    
    Following DPIR: sigma decreases from sigma_max to sigma_min
    across iterations using geometric decay.
    
    Args:
        iteration: current iteration index (0-based)
        total_iterations: total number of iterations K
        sigma_max: initial (high) noise level
        sigma_min: final (low) noise level
    Returns:
        sigma_k: noise level for this iteration
    """
    if total_iterations <= 1:
        return sigma_min
    ratio = iteration / (total_iterations - 1)
    sigma_k = sigma_max * (sigma_min / sigma_max) ** ratio
    return sigma_k


def mu_schedule(iteration, total_iterations, mu_min=0.01, mu_max=0.5):
    """Penalty parameter schedule. mu increases across iterations.
    As mu increases, we trust the denoiser more and the data term less.
    
    Args:
        iteration: current iteration index
        total_iterations: total K
        mu_min: initial mu (low = strong data fidelity)
        mu_max: final mu (high = trust denoiser more)
    Returns:
        mu_k: penalty parameter for this iteration
    """
    if total_iterations <= 1:
        return mu_max
    ratio = iteration / (total_iterations - 1)
    mu_k = mu_min * (mu_max / mu_min) ** ratio
    return mu_k


@torch.no_grad()
def pnp_hqs_restore(y, denoiser_model, scale_factor=2, num_iterations=15,
                    sigma_max=49.0/255.0, sigma_min=1.0/255.0,
                    mu_min=0.01, mu_max=0.5, use_log_domain=True,
                    eps=1e-3, device='cpu'):
    """Full PnP-HQS restoration pipeline.
    
    Restores a degraded image suffering from:
    1. Gaussian noise
    2. Speckle noise (multiplicative)
    3. 2x downsampling
    
    Args:
        y: input degraded LR image as numpy array (H_lr, W_lr), float32
        denoiser_model: trained DRUNet model
        scale_factor: SR scale factor (default 2)
        num_iterations: number of HQS iterations K
        sigma_max: initial noise level for denoiser
        sigma_min: final noise level for denoiser
        mu_min: initial penalty parameter
        mu_max: final penalty parameter
        use_log_domain: whether to apply log-domain speckle handling
        eps: stability constant for log transform
        device: 'cuda' or 'cpu'
    Returns:
        restored: numpy array (H_hr, W_hr), float32, clipped to [0, 1]
    """
    denoiser_model.eval()
    
    # Convert to tensor: (1, 1, H, W)
    y_tensor = torch.from_numpy(y).unsqueeze(0).unsqueeze(0).float().to(device)
    
    # Optional log-domain transform for speckle noise
    if use_log_domain:
        y_proc = log_domain_transform(y_tensor, eps=eps)
    else:
        y_proc = y_tensor.clone()
    
    # Initialize x_0 via bicubic upsampling
    x = bicubic_upsample(y_proc, scale_factor=scale_factor)
    
    H_hr, W_hr = x.shape[2], x.shape[3]
    
    # PnP-HQS iterative loop
    for k in range(num_iterations):
        # Get schedules for this iteration
        sigma_k = noise_level_schedule(k, num_iterations, sigma_max, sigma_min)
        mu_k = mu_schedule(k, num_iterations, mu_min, mu_max)
        
        # Step A: Data Fidelity (closed-form FFT solution)
        x = data_fidelity_sr(y_proc, x, mu_k, scale_factor)
        
        # Step B: Denoiser Prior
        x = denoiser_model(x, sigma_k)
    
    # Inverse log-domain transform
    if use_log_domain:
        x = exp_domain_transform(x, eps=eps)
    
    # Convert back to numpy, clip to [0, 1]
    restored = x.squeeze().cpu().numpy()
    restored = np.clip(restored, 0.0, 1.0)
    
    return restored
