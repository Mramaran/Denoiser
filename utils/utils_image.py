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

def calculate_ssim(img1, img2, border=0, data_range=1.0):
    """Windowed SSIM (Wang et al., 2004): 11x11 Gaussian window, sigma=1.5.

    The previous implementation used a single global mean and variance, which
    scores a pixel-shuffled image near 1.0 against its own source.
    """
    from scipy.ndimage import gaussian_filter

    if border > 0:
        img1 = img1[border:-border, border:-border]
        img2 = img2[border:-border, border:-border]

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    C1 = (0.01 * data_range) ** 2
    C2 = (0.03 * data_range) ** 2

    # sigma=1.5 with truncate=3.5 gives radius 5, i.e. the standard 11x11 window
    def filt(x):
        return gaussian_filter(x, sigma=1.5, truncate=3.5, mode="nearest")

    mu1, mu2 = filt(img1), filt(img2)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    sigma1_sq = filt(img1 * img1) - mu1_sq
    sigma2_sq = filt(img2 * img2) - mu2_sq
    sigma12 = filt(img1 * img2) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean())

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
