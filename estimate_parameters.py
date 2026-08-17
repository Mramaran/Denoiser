import os
import argparse
import numpy as np
import scipy.ndimage as ndimage
import scipy.signal as signal
from scipy.optimize import curve_fit
import json

def estimate_downsample_kernel(gt_img, lr_img, scale=2):
    # Try different standard kernels to downsample GT and check spectral match
    h, w = lr_img.shape
    candidates = {}
    
    # Candidate 1: Bilinear (using zoom)
    candidates['bilinear'] = ndimage.zoom(gt_img, 1.0/scale, order=1)
    
    # Candidate 2: Bicubic (using zoom)
    candidates['bicubic'] = ndimage.zoom(gt_img, 1.0/scale, order=3)
    
    # Candidate 3: Gaussian blur + subsample
    blurred = ndimage.gaussian_filter(gt_img, sigma=1.0)
    candidates['gaussian_sub'] = blurred[::scale, ::scale]
    
    # Candidate 4: Box filter (average pooling)
    box = np.zeros((h, w))
    for i in range(scale):
        for j in range(scale):
            box += gt_img[i::scale, j::scale]
    candidates['box'] = box / (scale * scale)
    
    # Find spectral match (correlation in frequency domain)
    lr_fft = np.abs(np.fft.fftshift(np.fft.fft2(lr_img - np.mean(lr_img))))
    best_corr = -1
    best_kernel = 'bicubic'
    
    for name, cand in candidates.items():
        cand_fft = np.abs(np.fft.fftshift(np.fft.fft2(cand - np.mean(cand))))
        # Compare high frequencies below noise floor (e.g. outer 50% band)
        mask = np.ones_like(lr_fft)
        cy, cx = h // 2, w // 2
        y, x = np.ogrid[-cy:h-cy, -cx:w-cx]
        r = np.sqrt(x**2 + y**2)
        mask[r > min(cy, cx) * 0.8] = 0  # avoid extreme high freq noise
        mask[r < min(cy, cx) * 0.2] = 0  # avoid low freq signal
        
        corr = np.corrcoef(lr_fft[mask == 1], cand_fft[mask == 1])[0, 1]
        if corr > best_corr:
            best_corr = corr
            best_kernel = name
            
    return best_kernel

def fit_noise_params(gt_lr, lr):
    # residual = lr - D(gt)
    residual = lr - gt_lr
    
    # Local mean and local variance
    local_mean = ndimage.uniform_filter(gt_lr, size=7)
    local_sq_mean = ndimage.uniform_filter(gt_lr**2, size=7)
    local_var = local_sq_mean - local_mean**2
    local_var[local_var < 0] = 0
    
    # Also get residual local variance
    res_mean = ndimage.uniform_filter(residual, size=7)
    res_sq_mean = ndimage.uniform_filter(residual**2, size=7)
    res_var = res_sq_mean - res_mean**2
    res_var[res_var < 0] = 0
    
    # Bin by local intensity
    bins = np.linspace(0.0, 1.0, 20)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    var_profile = []
    
    for i in range(len(bins)-1):
        mask = (gt_lr >= bins[i]) & (gt_lr < bins[i+1])
        if np.sum(mask) > 100:
            var_profile.append(np.mean(res_var[mask]))
        else:
            var_profile.append(0.0)
            
    # Fit: var(I) = a + b * I + c * I^2
    def quadratic_fit(I, a, b, c):
        return a + b * I + c * I**2
        
    try:
        popt, _ = curve_fit(quadratic_fit, bin_centers, var_profile, p0=[0.01, 0.01, 0.0], bounds=(0, np.inf))
        a, b, c = popt
    except:
        a, b, c = 0.01, 0.01, 0.0
        
    return float(a), float(b), float(c), residual

def infer_order(residual, scale=2):
    # Analyze spatial autocorrelation of residual
    # Flat autocorrelation -> noise added after downsampling (still white)
    # Smoothed / correlated autocorrelation -> noise added before downsampling (correlated by downsampling kernel)
    h, w = residual.shape
    res_norm = (residual - np.mean(residual)) / (np.std(residual) + 1e-8)
    
    # Compute 2D Autocorrelation
    fft = np.fft.fft2(res_norm)
    autocorr = np.real(np.fft.ifft2(fft * np.conjugate(fft))) / (h * w)
    autocorr = np.fft.fftshift(autocorr)
    
    cy, cx = h // 2, w // 2
    # Compare center pixel (peak=1.0) with immediate neighbors
    neighbor_val = (autocorr[cy+1, cx] + autocorr[cy-1, cx] + autocorr[cy, cx+1] + autocorr[cy, cx-1]) / 4.0
    
    # If neighbor correlation is high, noise was downsampled (noise-before-downsampling)
    # If neighbor correlation is near 0, noise is white (noise-after-downsampling)
    if neighbor_val > 0.15:
        return 'noise_before_downsampling'
    else:
        return 'noise_after_downsampling'

def main():
    parser = argparse.ArgumentParser(
        description="Estimate the degradation parameters from real (GT, NoisyLR) pairs"
    )
    parser.add_argument("--gt_dir", default="../Dataset/train/train/GT")
    parser.add_argument("--noisy_dir", default="../Dataset/train/train/NoisyLR")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="How many pairs to estimate from (default: 100)")
    parser.add_argument("--out", default="degradation_params.json")
    args = parser.parse_args()

    gt_dir, noisy_dir = args.gt_dir, args.noisy_dir
    for d in (gt_dir, noisy_dir):
        if not os.path.exists(d):
            print(f"Dataset path not found: {d}")
            return

    filenames = sorted(os.listdir(gt_dir))[:args.num_samples]
    
    kernels = []
    gauss_vars = []
    speckle_coeffs = []
    orders = []
    min_vals = []
    max_vals = []
    
    print(f"Estimating parameters from {len(filenames)} pairs...")
    for fn in filenames:
        gt = np.load(os.path.join(gt_dir, fn))
        noisy = np.load(os.path.join(noisy_dir, fn))
        
        # Min/max ranges
        min_vals.append(float(np.min(noisy)))
        max_vals.append(float(np.max(noisy)))
        
        # Kernel
        kernel = estimate_downsample_kernel(gt, noisy)
        kernels.append(kernel)
        
        # Get degraded GT under estimated kernel to compute residuals
        if kernel == 'bilinear':
            gt_degraded = ndimage.zoom(gt, 0.5, order=1)
        elif kernel == 'bicubic':
            gt_degraded = ndimage.zoom(gt, 0.5, order=3)
        elif kernel == 'box':
            gt_degraded = (gt[::2, ::2] + gt[1::2, ::2] + gt[::2, 1::2] + gt[1::2, 1::2]) / 4.0
        else:
            blurred = ndimage.gaussian_filter(gt, sigma=1.0)
            gt_degraded = blurred[::2, ::2]
            
        # Fit noise
        gauss, speckle_b, speckle_c, residual = fit_noise_params(gt_degraded, noisy)
        gauss_vars.append(gauss)
        speckle_coeffs.append((speckle_b, speckle_c))
        
        # Order
        order = infer_order(residual)
        orders.append(order)
        
    # Compile statistics
    unique_kernels, counts_kernels = np.unique(kernels, return_counts=True)
    kernel_dist = {k: float(c)/len(kernels) for k, c in zip(unique_kernels, counts_kernels)}
    
    unique_orders, counts_orders = np.unique(orders, return_counts=True)
    order_dist = {o: float(c)/len(orders) for o, c in zip(unique_orders, counts_orders)}
    
    params = {
        'scale_factor': 2,
        'kernel_distribution': kernel_dist,
        'order_distribution': order_dist,
        'mean_gaussian_variance': float(np.mean(gauss_vars)),
        'std_gaussian_variance': float(np.std(gauss_vars)),
        'mean_speckle_b': float(np.mean([sc[0] for sc in speckle_coeffs])),
        'mean_speckle_c': float(np.mean([sc[1] for sc in speckle_coeffs])),
        'min_val_range': [float(np.percentile(min_vals, 5)), float(np.percentile(min_vals, 95))],
        'max_val_range': [float(np.percentile(max_vals, 5)), float(np.percentile(max_vals, 95))]
    }
    
    print("\n--- Estimated Parameters ---")
    print(json.dumps(params, indent=2))
    
    with open(args.out, 'w') as f:
        json.dump(params, f, indent=2)
    print(f"\nSaved parameters to {args.out}")

if __name__ == '__main__':
    main()
