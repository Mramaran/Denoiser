"""End-to-end training for the KLA restoration model.

Trains DRUNet with a PixelShuffle SR head to map a degraded 128x128 NoisyLR
image directly to its 256x256 ground truth. The global bicubic skip inside the
model means the network only learns the residual over interpolation, so it
starts from the ~22.8 dB bicubic baseline rather than from scratch.

Training data is the 3200 real (NoisyLR, GT) pairs, optionally concatenated
with the calibrated synthetic degradation from augment_pipeline.py. Validation
uses the frozen held-out split in eval_split.txt and is never augmented.

Usage:
    python train.py --data_dir ../Dataset/train/train --epochs 150 --batch_size 16
"""

import argparse
import copy
import os
import time
import warnings

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm

from augment_pipeline import CalibratedDegradationDataset
from datasets import PairedRealDataset
from models.drunet import DRUNet
from models.losses import HybridLoss, CharbonnierLoss
from utils.utils_image import calculate_psnr


class ModelEMA:
    """Exponential moving average of weights.

    Worth a consistent +0.1-0.2 dB on restoration tasks and costs one extra
    copy of the model in memory.
    """

    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        ema_state = self.ema.state_dict()
        for k, m in model.state_dict().items():
            e = ema_state[k]
            if e.dtype.is_floating_point:
                e.mul_(self.decay).add_(m.detach(), alpha=1.0 - self.decay)
            else:
                e.copy_(m)


def split_files(all_files, eval_split_path):
    """Split filenames into (train, val) using the frozen held-out list."""
    with open(eval_split_path) as f:
        held_out = {line.strip() for line in f if line.strip()}
    val_files = [f for f in all_files if f in held_out]
    train_files = [f for f in all_files if f not in held_out]
    if not val_files:
        raise RuntimeError(f"No files from {eval_split_path} found in the dataset")
    return train_files, val_files


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch, ema, use_amp):
    model.train()
    total_loss, count = 0.0, 0

    for noisy, gt in tqdm(dataloader, desc=f"epoch {epoch + 1}", leave=False):
        noisy = noisy.to(device, non_blocking=True)
        gt = gt.to(device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp):
            output = model(noisy)
            if isinstance(criterion, HybridLoss):
                loss = criterion(output, gt, epoch=epoch)
            else:
                loss = criterion(output, gt)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        ema.update(model)

        total_loss += loss.item() * noisy.size(0)
        count += noisy.size(0)

    return total_loss / max(1, count)


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Score on the held-out split. PSNR is computed on the clipped output,
    which is exactly what evaluate.py writes to disk."""
    model.eval()
    total_loss, total_psnr, count = 0.0, 0.0, 0

    for noisy, gt in dataloader:
        noisy = noisy.to(device)
        gt = gt.to(device)

        output = model(noisy)
        total_loss += criterion(output, gt).item() * noisy.size(0)

        for i in range(output.size(0)):
            pred = output[i, 0].float().cpu().numpy().clip(0, 1)
            target = gt[i, 0].float().cpu().numpy()
            total_psnr += calculate_psnr(pred, target)
            count += 1

    return total_loss / max(1, count), total_psnr / max(1, count)


def main():
    parser = argparse.ArgumentParser(description="Train end-to-end DRUNet-SR restorer")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing GT/ and NoisyLR/ subdirectories")
    parser.add_argument("--eval_split", type=str, default="eval_split.txt",
                        help="Frozen held-out filename list (default: eval_split.txt)")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lr_patch", type=int, default=64,
                        help="LR crop size; the GT crop is 2x this (default: 64)")
    parser.add_argument("--loss", choices=["hybrid", "charbonnier"], default="hybrid")
    parser.add_argument("--loss_norm", choices=["ortho", "backward"], default="ortho",
                        help="FFT normalisation and matching weight preset for the "
                             "hybrid loss. 'ortho' (default) is the corrected, "
                             "pixel-dominant objective. 'backward' reproduces the "
                             "original frequency-dominated objective that produced "
                             "the shipped epoch-28 checkpoint. See Report/ablation.md.")
    parser.add_argument("--no_synthetic", action="store_true",
                        help="Train on real pairs only, skipping the calibrated augmenter")
    parser.add_argument("--save_dir", type=str, default="models",
                        help="Where checkpoints are written. Defaults to models/, "
                             "which is where run.py and evaluate.py look first.")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and torch.cuda.is_bf16_supported()
    print(f"[Train] Device: {device} | bf16 autocast: {use_amp}")
    if device.type == "cuda":
        print(f"[Train] GPU: {torch.cuda.get_device_name(0)}")

    os.makedirs(args.save_dir, exist_ok=True)

    gt_dir = os.path.join(args.data_dir, "GT")
    noisy_dir = os.path.join(args.data_dir, "NoisyLR")
    for d in (gt_dir, noisy_dir):
        if not os.path.isdir(d):
            raise RuntimeError(f"Expected directory not found: {d}")

    all_files = sorted(f for f in os.listdir(gt_dir) if f.endswith(".npy"))
    train_files, val_files = split_files(all_files, args.eval_split)
    print(f"[Train] {len(train_files)} train / {len(val_files)} held-out val")

    real_train = PairedRealDataset(noisy_dir, gt_dir, train_files,
                                   lr_patch=args.lr_patch, augment=True)
    if args.no_synthetic:
        train_dataset = real_train
    else:
        synthetic = CalibratedDegradationDataset(
            gt_dir=gt_dir, params_path="degradation_params.json",
            patch_size=args.lr_patch, is_train=True, file_list=train_files,
        )
        train_dataset = ConcatDataset([real_train, synthetic])
        print(f"[Train] Mixed dataset: {len(real_train)} real + {len(synthetic)} synthetic")

    val_dataset = PairedRealDataset(noisy_dir, gt_dir, val_files,
                                    lr_patch=args.lr_patch, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers,
                              pin_memory=(device.type == "cuda"), drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers,
                            pin_memory=(device.type == "cuda"))

    model = DRUNet(in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, sf=2).to(device)
    print(f"[Train] Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.loss == "hybrid":
        criterion = HybridLoss(device=device, freq_norm=args.loss_norm)
        print(f"[Train] HybridLoss freq_norm={args.loss_norm} | weights: "
              f"pixel {criterion.w_pixel} freq {criterion.w_freq} "
              f"ssim {criterion.w_ssim} grad {criterion.w_grad} "
              f"lpips {criterion.w_lpips}")
    else:
        criterion = CharbonnierLoss().to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs,
                                                     eta_min=1e-6)
    ema = ModelEMA(model, decay=args.ema_decay)

    start_epoch = 0
    best_psnr = 0.0
    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        ema.ema.load_state_dict(ckpt["ema_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_psnr = ckpt.get("best_psnr", ckpt.get("val_psnr", 0.0))

        # The scheduler was just constructed fresh (last_epoch=0); fast-forward
        # it to where the interrupted run left off so the cosine curve resumes
        # from the correct point instead of restarting at the initial LR.
        # This reliably fires PyTorch's "Detected call of `lr_scheduler.step()`
        # before `optimizer.step()`" UserWarning, which is a false positive
        # here: that warning is about a *real* training step skipping its LR
        # value, but this loop isn't a training step at all -- it's deliberately
        # replaying `start_epoch` already-elapsed epochs so the scheduler lands
        # where the interrupted run left off. The resulting LR has been
        # verified (standalone scheduler math) to exactly match what a never-
        # interrupted run would have at the same epoch, so suppress only this
        # exact message -- do not blanket-suppress UserWarning, and do not
        # remove this suppression without re-verifying the LR is still correct.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Detected call of `lr_scheduler\.step\(\)` before `optimizer\.step\(\)`",
                category=UserWarning,
            )
            for _ in range(start_epoch):
                scheduler.step()

        resumed_lr = optimizer.param_groups[0]["lr"]
        print(f"[Train] Fast-forwarded LR schedule to epoch {start_epoch} (lr {resumed_lr:.2e})")
        print(f"[Train] Resumed from epoch {start_epoch} | "
              f"best PSNR so far {best_psnr:.2f} dB | "
              f"lr {resumed_lr:.2e}")

    print(f"\n[Train] Training for {args.epochs} epochs")
    print("=" * 78)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion,
                                     device, epoch, ema, use_amp)

        # Score the EMA weights - they are what evaluate.py will use.
        val_loss, val_psnr = validate(ema.ema, val_loader, criterion, device)
        scheduler.step()

        print(f"Epoch [{epoch + 1:3d}/{args.epochs}] "
              f"train {train_loss:.5f} | val {val_loss:.5f} | "
              f"val PSNR {val_psnr:6.2f} dB | "
              f"lr {optimizer.param_groups[0]['lr']:.2e} | "
              f"{time.time() - t0:5.1f}s")

        if val_psnr > best_psnr:
            best_psnr = val_psnr

            # Full checkpoint: resumable, stays local. Measured at ~540 MB
            # (33.7M params x model + EMA + two Adam moments), far over
            # GitHub's 100 MB per-file limit, so this one is gitignored.
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "ema_state_dict": ema.ema.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_psnr": val_psnr,
                "val_loss": val_loss,
                "best_psnr": best_psnr,
                "sf": 2,
                "in_nc": 1,
            }, os.path.join(args.save_dir, "drunet_sr_best.pth"))

            # Inference-only checkpoint: EMA weights alone, in fp16. Measured
            # at 67 MB, which fits under the limit — this is the file that
            # ships in the repo and that evaluate.py loads by default.
            torch.save({
                "ema_state_dict": {
                    k: (v.half() if v.is_floating_point() else v)
                    for k, v in ema.ema.state_dict().items()
                },
                "epoch": epoch,
                "val_psnr": val_psnr,
                "sf": 2,
                "in_nc": 1,
                "fp16": True,
            }, os.path.join(args.save_dir, "drunet_sr_inference.pth"))

            print(f"  -> saved best ({best_psnr:.2f} dB)")

    print("=" * 78)
    print(f"[Train] Done. Best held-out PSNR: {best_psnr:.2f} dB")
    print(f"[Train] Weights: {os.path.join(args.save_dir, 'drunet_sr_best.pth')}")


if __name__ == "__main__":
    main()
