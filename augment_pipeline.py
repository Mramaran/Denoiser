import os
import json
import numpy as np
import scipy.ndimage as ndimage
import torch
from torch.utils.data import Dataset, DataLoader

class CalibratedDegradationDataset(Dataset):
    def __init__(self, gt_dir, params_path='degradation_params.json', patch_size=128, is_train=True, file_list=None):
        self.gt_dir = gt_dir
        self.patch_size = patch_size
        self.is_train = is_train
        
        if file_list is not None:
            self.filenames = [os.path.basename(f) for f in file_list]
        else:
            self.filenames = sorted(os.listdir(gt_dir))
        
        # Load params or use default fallbacks
        if os.path.exists(params_path):
            with open(params_path, 'r') as f:
                self.params = json.load(f)
        else:
            self.params = {
                'scale_factor': 2,
                'kernel_distribution': {'gaussian_sub': 0.5, 'bicubic': 0.5},
                'order_distribution': {'noise_before_downsampling': 0.5, 'noise_after_downsampling': 0.5},
                'mean_gaussian_variance': 0.005,
                'std_gaussian_variance': 0.001,
                'mean_speckle_b': 0.02,
                'mean_speckle_c': 0.001,
                'min_val_range': [-0.005, 0.005],
                'max_val_range': [0.995, 1.3]
            }

    def __len__(self):
        return len(self.filenames)

    def apply_noise(self, img, gauss_var, speckle_b, speckle_c):
        # Multiplicative speckle noise (Gamma-distributed modeled as multiplicative)
        # We model local noise variance: sigma^2 = gauss_var + speckle_b * img + speckle_c * img^2
        # Speckle noise = eta * img, where eta is zero-mean multiplicative noise
        # Joint noise: total_noise = additive_gaussian + multiplicative_speckle
        
        # Generate independent standard normals
        n_g = np.random.normal(0, np.sqrt(gauss_var), img.shape)
        
        # Multiplicative component variance depends on intensity
        multiplicative_std = np.sqrt(np.maximum(0, speckle_b * img + speckle_c * (img ** 2)))
        n_s = np.random.normal(0, 1.0, img.shape) * multiplicative_std
        
        noisy = img + n_g + n_s
        return noisy

    def apply_downsample(self, img, kernel_type, scale=2):
        if kernel_type == 'bilinear':
            return ndimage.zoom(img, 1.0/scale, order=1)
        elif kernel_type == 'bicubic':
            return ndimage.zoom(img, 1.0/scale, order=3)
        elif kernel_type == 'box':
            box = np.zeros((img.shape[0]//scale, img.shape[1]//scale))
            for i in range(scale):
                for j in range(scale):
                    box += img[i::scale, j::scale]
            return box / (scale * scale)
        else: # gaussian_sub
            blurred = ndimage.gaussian_filter(img, sigma=1.0)
            return blurred[::scale, ::scale]

    def __getitem__(self, idx):
        gt_path = os.path.join(self.gt_dir, self.filenames[idx])
        x = np.load(gt_path).astype(np.float32)
        scale = self.params['scale_factor']

        # patch_size is the LR patch size, so the GT crop is scale x larger.
        # Cropping is unconditional on is_train: training draws a random
        # offset, eval takes a deterministic centre crop (mirroring
        # PairedRealDataset), so both paths yield the same output shape and
        # stay ConcatDataset-compatible.
        if x.shape[0] >= self.patch_size * scale:
            h, w = x.shape
            th = tw = self.patch_size * scale
            if self.is_train:
                i = np.random.randint(0, h - th + 1)
                j = np.random.randint(0, w - tw + 1)
            else:
                i, j = (h - th) // 2, (w - tw) // 2
            x = x[i:i + th, j:j + tw]

        kernels = list(self.params['kernel_distribution'].keys())
        kernel_probs = list(self.params['kernel_distribution'].values())
        kernel = np.random.choice(kernels, p=kernel_probs)

        orders = list(self.params['order_distribution'].keys())
        order_probs = list(self.params['order_distribution'].values())
        order = np.random.choice(orders, p=order_probs)

        gauss_var = max(1e-6, np.random.normal(self.params['mean_gaussian_variance'],
                                               self.params['std_gaussian_variance']))
        speckle_b = max(0, self.params['mean_speckle_b'])
        speckle_c = max(0, self.params['mean_speckle_c'])

        if order == 'noise_before_downsampling':
            x_noisy_hr = self.apply_noise(x, gauss_var, speckle_b, speckle_c)
            y = self.apply_downsample(x_noisy_hr, kernel, scale=scale)
        else:
            x_lr = self.apply_downsample(x, kernel, scale=scale)
            y = self.apply_noise(x_lr, gauss_var, speckle_b, speckle_c)

        min_clip = np.random.uniform(self.params['min_val_range'][0],
                                     self.params['min_val_range'][1])
        max_clip = np.random.uniform(self.params['max_val_range'][0],
                                     self.params['max_val_range'][1])
        y = np.clip(y, min_clip, max_clip).astype(np.float32)

        # Target is the full-resolution GT crop: this dataset now trains the
        # end-to-end LR -> HR mapping, matching PairedRealDataset.
        return (torch.from_numpy(np.ascontiguousarray(y)).unsqueeze(0).float(),
                torch.from_numpy(np.ascontiguousarray(x)).unsqueeze(0).float())

if __name__ == '__main__':
    print("Testing CalibratedDegradationDataset...")
    dummy_gt = np.random.rand(256, 256).astype(np.float32)
    os.makedirs('dummy_train/GT', exist_ok=True)
    np.save('dummy_train/GT/000000.npy', dummy_gt)

    dataset = CalibratedDegradationDataset(gt_dir='dummy_train/GT', patch_size=64)
    y, x = dataset[0]
    print(f"y shape: {y.shape}, x shape: {x.shape}")
    print(f"y min: {y.min().item():.4f}, y max: {y.max().item():.4f}")

    import shutil
    shutil.rmtree('dummy_train')
