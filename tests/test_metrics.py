import numpy as np
import pytest

from utils.utils_image import calculate_psnr, calculate_ssim


def _smooth_image(seed=0, size=128):
    """A smooth, structured image — SSIM is meaningless on pure white noise."""
    from scipy.ndimage import gaussian_filter
    rng = np.random.default_rng(seed)
    img = gaussian_filter(rng.random((size, size)).astype(np.float32), sigma=3.0)
    return ((img - img.min()) / (img.max() - img.min())).astype(np.float32)


def test_ssim_of_identical_images_is_one():
    img = _smooth_image()
    assert calculate_ssim(img, img) == pytest.approx(1.0, abs=1e-6)


def test_ssim_is_windowed_not_global():
    """A pixel-shuffled image has identical global mean and variance, so a
    global-statistics SSIM scores it near 1.0. A windowed SSIM must not."""
    img = _smooth_image()
    rng = np.random.default_rng(1)
    shuffled = img.ravel().copy()
    rng.shuffle(shuffled)
    shuffled = shuffled.reshape(img.shape)
    assert calculate_ssim(img, shuffled) < 0.1


def test_ssim_degrades_monotonically_with_noise():
    img = _smooth_image()
    rng = np.random.default_rng(2)
    light = img + rng.standard_normal(img.shape).astype(np.float32) * 0.02
    heavy = img + rng.standard_normal(img.shape).astype(np.float32) * 0.20
    assert calculate_ssim(img, light) > calculate_ssim(img, heavy)


def test_ssim_matches_reference_implementation():
    """calculate_ssim must agree with the trusted skimage reference (windowed,
    Gaussian-weighted SSIM) — not just satisfy an indirect proxy property.

    On this image the two agree to within ~0.005. The pre-fix global-statistics
    formula does not: it scores this same pair 0.77 against a true ~0.38,
    because it only ever compares one whole-image mean/variance/covariance
    triple and never compares same-position local neighborhoods. abs=0.02 is
    ~4x the observed skimage agreement gap, so it is tight enough to catch a
    regression to global statistics (which misses by ~0.39) while absorbing
    the two implementations' differing border handling (skimage crops the
    filter border by default; ours uses mode='nearest' on the full image).
    """
    from skimage.metrics import structural_similarity as skimage_ssim

    img = _smooth_image()
    rng = np.random.default_rng(3)
    noisy = img + rng.standard_normal(img.shape).astype(np.float32) * 0.10

    ours = calculate_ssim(img, noisy)
    reference = skimage_ssim(img, noisy.astype(np.float64), data_range=1.0,
                              gaussian_weights=True, sigma=1.5,
                              use_sample_covariance=False)
    assert ours == pytest.approx(reference, abs=0.02)


def test_ssim_detects_localized_distortion_with_unchanged_global_mean():
    """A block distortion — the top half brightened by 0.3, the bottom half
    darkened by 0.3 (clipped to [0, 1]) — is built to fool global statistics:
    the whole-image mean barely moves, so a global-mean/variance SSIM is
    largely blind to it. It scores this pair ~0.19 ("very different"). The
    true windowed SSIM is ~0.73 ("same structure, shifted brightness"), since
    each local neighborhood still matches its source, just offset. That is a
    >0.5 gap, so > 0.5 fails against global statistics and passes against a
    real windowed computation.
    """
    img = _smooth_image()
    patch = img.copy()
    patch[:64, :] = np.clip(img[:64, :] + 0.3, 0, 1)
    patch[64:, :] = np.clip(img[64:, :] - 0.3, 0, 1)
    assert abs(patch.mean() - img.mean()) < 0.01  # distortion is ~zero-mean globally

    assert calculate_ssim(img, patch) > 0.5


def test_psnr_of_identical_images_is_infinite():
    img = _smooth_image()
    assert calculate_psnr(img, img) == float("inf")


def test_psnr_matches_known_mse():
    """A constant offset of 0.1 gives MSE = 0.01, so PSNR = 20 dB exactly.

    Tolerance is 1e-4, not 1e-6: img + 0.1 is computed in float32, and 0.6 has
    no exact float32 representation, so the actual MSE is ~1.0000005e-2 rather
    than 1e-2. That is a deterministic ~2e-6 dB quantization artifact (same on
    any IEEE-754 machine), not a property of calculate_psnr, so 1e-4 still
    catches any real regression while absorbing the float32 rounding noise.
    """
    img = np.full((32, 32), 0.5, dtype=np.float32)
    assert calculate_psnr(img, img + 0.1) == pytest.approx(20.0, abs=1e-4)
