import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.ndimage as ndimage
import torch

REPO = Path(__file__).resolve().parent.parent


def _tiny_dataset(root, n=8):
    gt_dir, noisy_dir = root / "GT", root / "NoisyLR"
    gt_dir.mkdir(parents=True)
    noisy_dir.mkdir(parents=True)
    names = []
    for i in range(n):
        rng = np.random.default_rng(i)
        gt = ndimage.gaussian_filter(rng.random((64, 64)).astype(np.float32), 2.0)
        gt = ((gt - gt.min()) / (gt.max() - gt.min())).astype(np.float32)
        noisy = gt[::2, ::2] + rng.standard_normal((32, 32)).astype(np.float32) * 0.05
        name = f"{i:06d}.npy"
        np.save(gt_dir / name, gt)
        np.save(noisy_dir / name, noisy.astype(np.float32))
        names.append(name)
    return names


def test_training_runs_and_writes_a_loadable_checkpoint(tmp_path):
    names = _tiny_dataset(tmp_path / "data")
    split = tmp_path / "eval_split.txt"
    split.write_text("\n".join(names[:2]) + "\n")
    save_dir = tmp_path / "weights"

    result = subprocess.run(
        [sys.executable, "train.py",
         "--data_dir", str(tmp_path / "data"),
         "--eval_split", str(split),
         "--save_dir", str(save_dir),
         "--epochs", "2",
         "--batch_size", "2",
         "--lr_patch", "16",
         "--loss", "charbonnier",
         "--num_workers", "0",
         "--no_synthetic"],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"train.py failed:\n{result.stdout}\n{result.stderr}"

    ckpt_path = save_dir / "drunet_sr_best.pth"
    assert ckpt_path.is_file(), f"no checkpoint written. stdout:\n{result.stdout}"

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    for key in ("model_state_dict", "ema_state_dict", "val_psnr", "sf", "in_nc"):
        assert key in ckpt, f"checkpoint missing '{key}'"
    assert ckpt["sf"] == 2 and ckpt["in_nc"] == 1
    assert np.isfinite(ckpt["val_psnr"])


def test_validation_never_sees_training_files(tmp_path):
    """The held-out split must be excluded from the training file list."""
    sys.path.insert(0, str(REPO))
    from train import split_files

    names = [f"{i:06d}.npy" for i in range(10)]
    split = tmp_path / "eval_split.txt"
    split.write_text("\n".join(names[:3]) + "\n")

    train_files, val_files = split_files(names, str(split))
    assert set(val_files) == set(names[:3])
    assert set(train_files) == set(names[3:])
    assert not (set(train_files) & set(val_files))


def test_resume_does_not_regress_the_best_checkpoint(tmp_path):
    """Resuming must restore best_psnr from the checkpoint, not reset it to 0.

    Regression test for: resuming used to set best_psnr = 0.0 unconditionally,
    so the first epoch of any resumed run -- however bad -- beat that "best"
    and silently overwrote drunet_sr_best.pth / drunet_sr_inference.pth with a
    worse model.

    The actual guarantee that the resumed epoch cannot "win" comes from
    --ema_decay 1.0, not from --lr 0. validate() scores ema.ema, and
    ema_decay=1.0 makes ModelEMA.update() a no-op (e = 1.0*e + 0.0*m = e), so
    the checkpointed/scored weights are frozen for the resumed epoch no matter
    what the raw model does.

    --lr 0 is belt-and-braces here, not the mechanism -- it does NOT produce a
    literal zero learning rate on the resumed epoch. CosineAnnealingLR.step()
    uses a recursive formula seeded from the optimizer's *current* lr (which
    optimizer.load_state_dict() has just restored from the checkpoint), not
    the fresh base_lrs captured at this run's scheduler construction. With
    this test's exact parameters that recursion lands on eta_min (1e-6, not
    0.0 -- verified empirically). A bare `--lr 0` resume (without also
    freezing ema_decay) is therefore not by itself reliable: with the default
    ema_decay, ema.update() keeps pulling the EMA weights toward the
    already-better-trained raw model every step regardless of how small the
    raw model's own LR is, so in practice a resumed epoch with only `--lr 0`
    *improves* val_psnr almost every time (verified empirically across
    multiple trials), which would pass a naive "did not regress" check
    whether or not best_psnr was restored correctly.

    With ema.ema genuinely frozen, the resumed epoch's val_psnr ties the
    original exactly and deterministically. A guaranteed tie alone still
    can't distinguish correct from buggy behaviour, though (a tied-but-
    positive val_psnr would re-save an identical number either way). What
    actually distinguishes them is whether a save happens at all: with
    best_psnr wrongly reset to 0.0, the tied (positive) val_psnr still beats 0
    and triggers a spurious save, which also bumps the checkpoint's "epoch"
    field even though nothing improved. So this test checks both: the value
    must not regress, and the checkpoint must not be needlessly rewritten
    (epoch field unchanged).
    """
    names = _tiny_dataset(tmp_path / "data")
    split = tmp_path / "eval_split.txt"
    split.write_text("\n".join(names[:2]) + "\n")
    save_dir = tmp_path / "weights"

    base_args = [
        sys.executable, "train.py",
        "--data_dir", str(tmp_path / "data"),
        "--eval_split", str(split),
        "--save_dir", str(save_dir),
        "--batch_size", "2",
        "--lr_patch", "16",
        "--loss", "charbonnier",
        "--num_workers", "0",
        "--no_synthetic",
    ]

    result = subprocess.run(
        base_args + ["--epochs", "2"],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, f"initial run failed:\n{result.stdout}\n{result.stderr}"

    ckpt_path = save_dir / "drunet_sr_best.pth"
    original = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert "best_psnr" in original, "checkpoint missing 'best_psnr'"
    original_val_psnr = original["val_psnr"]
    original_epoch = original["epoch"]

    # Resume for one more epoch with --ema_decay 1.0, which freezes ema.ema --
    # the only thing validate()/checkpointing scores -- regardless of what the
    # raw model does. --lr 0 is extra insurance on the raw model but is not
    # itself what guarantees the resumed epoch can't beat the original; see
    # the docstring above for why (--lr 0 does not mean a literal zero LR).
    resume_result = subprocess.run(
        base_args + ["--epochs", "3", "--resume", str(ckpt_path),
                     "--lr", "0", "--ema_decay", "1.0"],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    assert resume_result.returncode == 0, (
        f"resume run failed:\n{resume_result.stdout}\n{resume_result.stderr}"
    )

    resumed = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    assert resumed["val_psnr"] >= original_val_psnr, (
        f"resume regressed the checkpoint: "
        f"{original_val_psnr} -> {resumed['val_psnr']}"
    )
    assert resumed["epoch"] == original_epoch, (
        "checkpoint was overwritten by a resumed epoch that could not have "
        f"beaten the original best (best_psnr was not restored on resume): "
        f"epoch {original_epoch} -> {resumed['epoch']}"
    )
