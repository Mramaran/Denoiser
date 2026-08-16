"""
Training script for DRUNet denoiser — KLA SEMICON India Hackathon 2026

Trains a DRUNet (Deep Residual U-Net) to perform blind Gaussian denoising
across a range of noise levels. The trained denoiser is used as the learned
prior in the PnP-HQS restoration pipeline.

The denoiser is NOT trained end-to-end on (NoisyLR, GT) pairs. It only learns
to remove Gaussian noise from clean-resolution images. The PnP loop handles
super-resolution and speckle noise at inference time via the physics model.

Usage:
    python train.py --data_dir ../Dataset/train/train --epochs 200 --batch_size 16

Reference:
    Zhang et al., "Plug-and-Play Image Restoration with Deep Denoiser Prior",
    TPAMI 2021. https://arxiv.org/abs/2008.13751
"""

import os
import argparse
import math
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from models.drunet import DRUNet
from utils.utils_image import calculate_psnr, augment_img
from augment_pipeline import CalibratedDegradationDataset
from models.losses import HybridLoss, CharbonnierLoss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class DenoisingDataset(Dataset):
    """Dataset for training the DRUNet denoiser.

    Loads ground-truth images and adds synthetic Gaussian noise on-the-fly.
    Random 128x128 crops and augmentations are applied for training.
    For validation, deterministic center cropping, fixed noise level,
    and deterministic seed are used.
    """

    def __init__(self, gt_dir=None, file_list=None, patch_size=128, sigma_min=0.0, sigma_max=50.0 / 255.0,
                 augment=True, is_val=False):
        """
        Args:
            gt_dir: path to directory containing GT .npy files
            file_list: optional list of pre-split .npy file paths
            patch_size: size of crops (default 128)
            sigma_min: minimum noise level (default 0)
            sigma_max: maximum noise level (default 50/255)
            augment: whether to apply random flips/rotations (training only)
            is_val: whether this is a validation dataset
        """
        super().__init__()
        self.patch_size = patch_size
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.augment = augment and not is_val
        self.is_val = is_val

        if file_list is not None:
            self.file_list = file_list
        elif gt_dir is not None:
            self.file_list = sorted([
                os.path.join(gt_dir, f)
                for f in os.listdir(gt_dir)
                if f.endswith('.npy')
            ])
        else:
            raise ValueError("Either gt_dir or file_list must be provided")

        if len(self.file_list) == 0:
            raise RuntimeError("No .npy files found in dataset")
        print(f"[Dataset] Found {len(self.file_list)} images. is_val={self.is_val}")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # Load GT image (H, W), float32, [0, 1]
        gt = np.load(self.file_list[idx]).astype(np.float32)

        H, W = gt.shape
        ps = self.patch_size

        if self.is_val:
            # Deterministic center crop
            if H >= ps and W >= ps:
                top = (H - ps) // 2
                left = (W - ps) // 2
                gt = gt[top:top + ps, left:left + ps]

            # Deterministic noise level (middle of the range)
            sigma = (self.sigma_min + self.sigma_max) / 2.0

            # Deterministic noise via fixed seed based on index
            rng = np.random.default_rng(idx)
            noise = rng.standard_normal(gt.shape, dtype=np.float32) * sigma
            noisy = gt + noise
        else:
            # Random crop
            if H >= ps and W >= ps:
                top = random.randint(0, H - ps)
                left = random.randint(0, W - ps)
                gt = gt[top:top + ps, left:left + ps]

            # Random augmentation (flip/rotate)
            if self.augment:
                mode = random.randint(0, 7)
                gt = augment_img(gt, mode)

            # Random noise level
            sigma = random.uniform(self.sigma_min, self.sigma_max)

            # Generate noisy image
            noise = np.random.randn(*gt.shape).astype(np.float32) * sigma
            noisy = gt + noise

        # Convert to tensors: (1, H, W)
        gt_tensor = torch.from_numpy(gt).unsqueeze(0)
        noisy_tensor = torch.from_numpy(noisy).unsqueeze(0)
        sigma_tensor = torch.tensor(sigma, dtype=torch.float32)

        return noisy_tensor, gt_tensor, sigma_tensor


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch=0, scaler=None):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    count = 0

    for noisy, gt, sigma in dataloader:
        noisy = noisy.to(device)
        gt = gt.to(device)
        sigma = sigma.to(device)

        # Forward pass under mixed precision autocast
        enable_amp = (scaler is not None) and (device.type == "cuda")
        with torch.cuda.amp.autocast(enabled=enable_amp):
            output = model(noisy, sigma)
            if isinstance(criterion, HybridLoss):
                loss = criterion(output, gt, epoch=epoch)
            else:
                loss = criterion(output, gt)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * noisy.size(0)
        count += noisy.size(0)

    return total_loss / count


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate model. Returns average loss and PSNR."""
    model.eval()
    total_loss = 0.0
    total_psnr = 0.0
    count = 0

    for noisy, gt, sigma in dataloader:
        noisy = noisy.to(device)
        gt = gt.to(device)
        sigma = sigma.to(device)

        output = model(noisy, sigma)
        loss = criterion(output, gt)

        total_loss += loss.item() * noisy.size(0)

        # Calculate PSNR per sample
        for i in range(output.size(0)):
            pred = output[i, 0].cpu().numpy().clip(0, 1)
            target = gt[i, 0].cpu().numpy()
            total_psnr += calculate_psnr(pred, target)
            count += 1

    avg_loss = total_loss / count
    avg_psnr = total_psnr / count
    return avg_loss, avg_psnr


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Train DRUNet denoiser for PnP-HQS restoration"
    )
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to training data directory containing GT/ and NoisyLR/ subdirs")
    parser.add_argument("--epochs", type=int, default=200,
                        help="Number of training epochs (default: 200)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--patch_size", type=int, default=128,
                        help="Training patch size (default: 128)")
    parser.add_argument("--sigma_max", type=float, default=50.0 / 255.0,
                        help="Max noise level for training (default: 50/255)")
    parser.add_argument("--val_split", type=float, default=0.1,
                        help="Validation split ratio (default: 0.1)")
    parser.add_argument("--save_dir", type=str, default="model_weights",
                        help="Directory to save model weights (default: model_weights)")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint to resume training from")
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Device: {device}")
    if device.type == "cuda":
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.save_dir, exist_ok=True)

    # GT directory
    gt_dir = os.path.join(args.data_dir, "GT")
    if not os.path.isdir(gt_dir):
        # Maybe data_dir itself contains the GT files
        gt_dir = args.data_dir
    print(f"[Train] GT directory: {gt_dir}")

    # -----------------------------------------------------------------------
    # Dataset & DataLoader
    # -----------------------------------------------------------------------
    # Collect files
    file_list = sorted([
        os.path.join(gt_dir, f)
        for f in os.listdir(gt_dir)
        if f.endswith('.npy')
    ])
    if len(file_list) == 0:
        raise RuntimeError(f"No .npy files found in {gt_dir}")

    # Deterministic split using a fixed seed
    rng = random.Random(42)
    rng.shuffle(file_list)

    val_size = max(1, int(len(file_list) * args.val_split))
    train_size = len(file_list) - val_size

    val_files = file_list[:val_size]
    train_files = file_list[val_size:]

    train_dataset = CalibratedDegradationDataset(
        gt_dir=gt_dir,
        params_path='degradation_params.json',
        patch_size=args.patch_size,
        is_train=True,
        file_list=train_files
    )
    val_dataset = DenoisingDataset(
        file_list=val_files,
        patch_size=args.patch_size,
        sigma_max=args.sigma_max,
        augment=False,
        is_val=True
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda")
    )

    print(f"[Train] Train files: {train_size} | Val files: {val_size}")
    print(f"[Train] Batch size: {args.batch_size} | Patch size: {args.patch_size}")

    # -----------------------------------------------------------------------
    # Model, Optimizer, Loss
    # -----------------------------------------------------------------------
    model = DRUNet(in_nc=2, out_nc=1, nc=[64, 128, 256, 512], nb=4)
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[Train] DRUNet parameters: {num_params:,}")

    criterion = HybridLoss(device=device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    start_epoch = 0

    # Resume from checkpoint
    if args.resume and os.path.isfile(args.resume):
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"[Train] Resumed from epoch {start_epoch}")

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    # GradScaler for Mixed Precision training
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    best_psnr = 0.0
    print(f"\n[Train] Starting training for {args.epochs} epochs...")
    print("=" * 70)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, epoch=epoch, scaler=scaler)

        # Validate
        val_loss, val_psnr = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]["lr"]

        elapsed = time.time() - t0

        # Print progress
        print(f"Epoch [{epoch + 1:3d}/{args.epochs}] "
              f"Train Loss: {train_loss:.6f} | "
              f"Val Loss: {val_loss:.6f} | "
              f"Val PSNR: {val_psnr:.2f} dB | "
              f"LR: {current_lr:.2e} | "
              f"Time: {elapsed:.1f}s")

        # Save best model
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_path = os.path.join(args.save_dir, "drunet_denoiser_best.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_psnr": val_psnr,
                "val_loss": val_loss,
            }, save_path)
            print(f"  -> Saved best model (PSNR: {best_psnr:.2f} dB)")

        # Save periodic checkpoint every 50 epochs
        if (epoch + 1) % 50 == 0:
            ckpt_path = os.path.join(args.save_dir, f"drunet_epoch_{epoch + 1}.pth")
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_psnr": val_psnr,
                "val_loss": val_loss,
            }, ckpt_path)

    # Save final model
    final_path = os.path.join(args.save_dir, "drunet_denoiser_final.pth")
    torch.save({
        "epoch": args.epochs - 1,
        "model_state_dict": model.state_dict(),
        "val_psnr": val_psnr,
    }, final_path)

    print("=" * 70)
    print(f"[Train] Training complete. Best Val PSNR: {best_psnr:.2f} dB")
    print(f"[Train] Best model saved to: {os.path.join(args.save_dir, 'drunet_denoiser_best.pth')}")
    print(f"[Train] Final model saved to: {final_path}")


if __name__ == "__main__":
    main()
