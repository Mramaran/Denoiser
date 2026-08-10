import numpy as np
import os
import math

def load_npy(path):
    """Load a .npy image file and return as float32 numpy array."""
    img = np.load(path).astype(np.float32)
    return img

def save_npy(img, path):
    """Save numpy array as .npy file."""
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    np.save(path, img.astype(np.float32))

def calculate_psnr(img1, img2, border=0):
    """Calculate PSNR between two images (numpy arrays).
    Assumes images are in [0, 1] range."""
    if border > 0:
        img1 = img1[border:-border, border:-border]
        img2 = img2[border:-border, border:-border]
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return float('inf')
    return 10.0 * math.log10(1.0 / mse)

def calculate_ssim(img1, img2, border=0):
    """Calculate SSIM between two grayscale images.
    Implements the standard SSIM formula without requiring scikit-image."""
    # Implement SSIM manually to avoid external dependency issues
    # Use the standard Wang et al. 2004 formulation
    # K1=0.01, K2=0.03, L=1.0 (for [0,1] range)
    # Window: 11x11 uniform (simplified from Gaussian for robustness)
    if border > 0:
        img1 = img1[border:-border, border:-border]
        img2 = img2[border:-border, border:-border]
    
    C1 = (0.01) ** 2
    C2 = (0.03) ** 2
    
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1**2 + mu2**2 + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim

def augment_img(img, mode=0):
    """Data augmentation: 8 possible flips/rotations for a 2D image."""
    if mode == 0: return img
    elif mode == 1: return np.flipud(img).copy()
    elif mode == 2: return np.fliplr(img).copy()
    elif mode == 3: return np.rot90(img, k=1).copy()
    elif mode == 4: return np.rot90(img, k=2).copy()
    elif mode == 5: return np.rot90(img, k=3).copy()
    elif mode == 6: return np.flipud(np.rot90(img, k=1)).copy()
    elif mode == 7: return np.fliplr(np.rot90(img, k=1)).copy()
    else: return img
