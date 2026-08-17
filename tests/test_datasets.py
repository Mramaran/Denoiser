import numpy as np
import pytest
import scipy.ndimage as ndimage
import torch

from datasets import PairedRealDataset


def _make_pair(tmp_path, n=1):
    """GT is smooth and structured; NoisyLR is its exact 2x subsample, so
    alignment can be checked precisely."""
    gt_dir, noisy_dir = tmp_path / "GT", tmp_path / "NoisyLR"
    gt_dir.mkdir()
    noisy_dir.mkdir()
    names = []
    for i in range(n):
        rng = np.random.default_rng(i)
        gt = ndimage.gaussian_filter(rng.random((256, 256)).astype(np.float32), 3.0)
        gt = ((gt - gt.min()) / (gt.max() - gt.min())).astype(np.float32)
        name = f"{i:06d}.npy"
        np.save(gt_dir / name, gt)
        np.save(noisy_dir / name, gt[::2, ::2].copy())
        names.append(name)
    return str(noisy_dir), str(gt_dir), names


def test_returns_lr_input_and_hr_target(tmp_path):
    nd, gd, names = _make_pair(tmp_path)
    ds = PairedRealDataset(nd, gd, names, lr_patch=64, augment=False)
    y, x = ds[0]
    assert y.shape == (1, 64, 64)
    assert x.shape == (1, 128, 128)
    assert y.dtype == torch.float32 and x.dtype == torch.float32


def test_center_crop_is_exactly_aligned(tmp_path):
    nd, gd, names = _make_pair(tmp_path)
    ds = PairedRealDataset(nd, gd, names, lr_patch=64, augment=False)
    y, x = ds[0]
    assert np.allclose(y[0].numpy(), x[0].numpy()[::2, ::2])


def test_random_crops_stay_aligned(tmp_path):
    """A wrong crop offset would drop the correlation to near zero."""
    nd, gd, names = _make_pair(tmp_path)
    ds = PairedRealDataset(nd, gd, names * 20, lr_patch=64, augment=True)
    for i in range(20):
        y, x = ds[i]
        corr = np.corrcoef(y[0].numpy().ravel(),
                           x[0].numpy()[::2, ::2].ravel())[0, 1]
        assert corr > 0.9, f"sample {i} misaligned: corr={corr:.3f}"


def test_augmentation_produces_variety(tmp_path):
    nd, gd, names = _make_pair(tmp_path)
    ds = PairedRealDataset(nd, gd, names * 30, lr_patch=64, augment=True)
    crops = {ds[i][0].numpy().tobytes() for i in range(30)}
    assert len(crops) > 15


def test_empty_file_list_is_rejected(tmp_path):
    nd, gd, _ = _make_pair(tmp_path)
    try:
        PairedRealDataset(nd, gd, [], lr_patch=64)
    except RuntimeError:
        return
    raise AssertionError("expected RuntimeError for an empty file list")


def test_calibrated_dataset_returns_an_hr_target(tmp_path):
    from augment_pipeline import CalibratedDegradationDataset
    _, gd, names = _make_pair(tmp_path)
    ds = CalibratedDegradationDataset(gt_dir=gd, params_path="degradation_params.json",
                                      patch_size=64, is_train=True,
                                      file_list=[f"{gd}/{n}" for n in names])
    y, x = ds[0]
    assert y.shape == (1, 64, 64)
    assert x.shape == (1, 128, 128)


def test_calibrated_dataset_non_train_mode_crops_deterministically(tmp_path):
    """is_train=False must still crop to patch_size (not fall through to the
    full GT, which would break the 2-tuple shape contract with
    PairedRealDataset), and must pick the same centre crop on every read.

    The synthetic noise applied to y is still randomised per call, so
    determinism is checked on the HR target x, which is nothing but the
    (now unconditional) crop of the loaded GT array.
    """
    from augment_pipeline import CalibratedDegradationDataset
    _, gd, names = _make_pair(tmp_path)
    ds = CalibratedDegradationDataset(gt_dir=gd, params_path="degradation_params.json",
                                      patch_size=64, is_train=False,
                                      file_list=[f"{gd}/{n}" for n in names])
    y1, x1 = ds[0]
    assert y1.shape == (1, 64, 64)
    assert x1.shape == (1, 128, 128)

    y2, x2 = ds[0]
    assert y2.shape == (1, 64, 64)
    assert x2.shape == (1, 128, 128)
    assert torch.equal(x1, x2), "HR target crop must be deterministic when is_train=False"


@pytest.mark.parametrize("kernel", ["bicubic", "bilinear", "gaussian_sub"])
def test_calibrated_dataset_hr_target_shape_per_kernel(tmp_path, kernel):
    """Force each downsampling kernel explicitly rather than letting
    np.random.choice sample one at random, so a kernel-specific regression
    (e.g. in the bilinear or gaussian_sub branch of apply_downsample) can't
    pass the suite just because a different kernel happened to be drawn."""
    from augment_pipeline import CalibratedDegradationDataset
    _, gd, names = _make_pair(tmp_path)
    ds = CalibratedDegradationDataset(gt_dir=gd, params_path="degradation_params.json",
                                      patch_size=64, is_train=True,
                                      file_list=[f"{gd}/{n}" for n in names])
    ds.params['kernel_distribution'] = {kernel: 1.0}
    y, x = ds[0]
    assert y.shape == (1, 64, 64), f"kernel={kernel} produced y.shape={tuple(y.shape)}"
    assert x.shape == (1, 128, 128), f"kernel={kernel} produced x.shape={tuple(x.shape)}"
