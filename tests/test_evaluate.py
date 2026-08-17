"""CLI-contract tests for evaluate.py.

These invoke evaluate.py as a subprocess (matching the convention in
test_train_smoke.py) because it is an argparse script that the competition
organisers run as-is: testing through the real CLI is the only way to be
sure the behaviour they get matches what we verify here.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from models.drunet import DRUNet

REPO = Path(__file__).resolve().parent.parent


def _make_inference_checkpoint(path, sf=2, in_nc=1, val_psnr=25.0, epoch=3):
    """Build a synthetic checkpoint in the exact format train.py writes for
    drunet_sr_inference.pth: fp16 EMA weights of the real 33.7M-parameter
    architecture (evaluate.py hardcodes nc=[64,128,256,512], nb=4 for e2e
    mode regardless of checkpoint contents, so load_state_dict only succeeds
    if the checkpoint was built with that same shape), plus the metadata
    evaluate.py reads (sf, in_nc, fp16, epoch, val_psnr).
    """
    torch.manual_seed(0)
    model = DRUNet(in_nc=in_nc, out_nc=1, nc=[64, 128, 256, 512], nb=4, sf=sf).eval()
    state = {k: (v.half() if v.is_floating_point() else v)
             for k, v in model.state_dict().items()}
    torch.save({
        "ema_state_dict": state,
        "epoch": epoch,
        "val_psnr": val_psnr,
        "sf": sf,
        "in_nc": in_nc,
        "fp16": True,
    }, path)


def _tiny_input(dir_path, n=1, size=32):
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    names = []
    for i in range(n):
        img = rng.random((size, size)).astype(np.float32)
        name = f"{i:06d}.npy"
        np.save(dir_path / name, img)
        names.append(name)
    return names


def test_e2e_scale_factor_conflict_warns_and_checkpoint_value_wins(tmp_path):
    """--scale_factor is a runtime knob in pnp mode, but in e2e mode the
    scale factor is a property of the trained weights (checkpoint['sf']).
    An explicit --scale_factor that disagrees must not silently override
    it: evaluate.py should warn and proceed with the checkpoint's value.
    """
    weights = tmp_path / "weights" / "drunet_sr_inference.pth"
    weights.parent.mkdir(parents=True)
    _make_inference_checkpoint(weights, sf=2)

    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    names = _tiny_input(in_dir)

    result = subprocess.run(
        [sys.executable, "evaluate.py",
         "--input_dir", str(in_dir), "--output_dir", str(out_dir),
         "--weights", str(weights), "--scale_factor", "4"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"evaluate.py failed:\n{result.stdout}\n{result.stderr}"
    assert "WARNING" in result.stdout and "sf=2" in result.stdout, (
        f"expected a checkpoint-sf-wins warning in stdout:\n{result.stdout}"
    )
    # The printed settings banner must reflect the checkpoint's sf, not the
    # (rejected) CLI value -- otherwise the log would misreport what ran.
    assert "Scale factor: 2x" in result.stdout

    # And the actual restoration must have used sf=2 (32x32 -> 64x64), not
    # the sf=4 that --scale_factor asked for.
    out = np.load(out_dir / names[0])
    assert out.shape == (64, 64), (
        f"checkpoint sf=2 should win over --scale_factor 4, got {out.shape}"
    )


def test_e2e_scale_factor_matching_checkpoint_is_silent(tmp_path):
    """No warning -- and no behaviour change -- when the explicit
    --scale_factor already agrees with the checkpoint's sf."""
    weights = tmp_path / "weights" / "drunet_sr_inference.pth"
    weights.parent.mkdir(parents=True)
    _make_inference_checkpoint(weights, sf=2)

    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _tiny_input(in_dir)

    result = subprocess.run(
        [sys.executable, "evaluate.py",
         "--input_dir", str(in_dir), "--output_dir", str(out_dir),
         "--weights", str(weights), "--scale_factor", "2"],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"evaluate.py failed:\n{result.stdout}\n{result.stderr}"
    assert "WARNING" not in result.stdout


def test_e2e_mode_accepts_a_bare_state_dict_checkpoint(tmp_path):
    """A checkpoint saved as plain torch.save(model.state_dict(), path) has
    neither an 'ema_state_dict' nor a 'model_state_dict' wrapper key.

    Regression test for: the e2e branch used to do
    `checkpoint.get("ema_state_dict") or checkpoint["model_state_dict"]`,
    which raised an unhandled KeyError (raw traceback, not the script's
    clean "ERROR:" message) whenever neither key was present -- even though
    the pnp branch immediately below already guarded against exactly this
    with `checkpoint["model_state_dict"] if "model_state_dict" in checkpoint
    else checkpoint`. This cannot arise from this repo's own train.py, which
    always wraps its weights, but evaluate.py is run as-is by the
    organisers, so a malformed/hand-built checkpoint must not crash it.

    The fixed branch mirrors the pnp branch's fallback: if neither wrapper
    key is present, the loaded object is treated as the state dict itself.
    Loading successfully is the better behaviour (the checkpoint is in fact
    perfectly valid weights, just unwrapped), so that is what this test
    requires -- not merely "fails cleanly instead of a traceback".
    """
    torch.manual_seed(0)
    model = DRUNet(in_nc=1, out_nc=1, nc=[64, 128, 256, 512], nb=4, sf=2).eval()
    weights = tmp_path / "weights" / "bare_state_dict.pth"
    weights.parent.mkdir(parents=True)
    torch.save(model.state_dict(), weights)  # no wrapper dict at all

    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    names = _tiny_input(in_dir)

    result = subprocess.run(
        [sys.executable, "evaluate.py",
         "--input_dir", str(in_dir), "--output_dir", str(out_dir),
         "--weights", str(weights)],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert "KeyError" not in result.stderr and "Traceback" not in result.stderr, (
        f"a bare state_dict checkpoint must never raise a raw traceback:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert result.returncode == 0, (
        f"loading a bare state_dict should succeed (it is valid weights, "
        f"just unwrapped), not merely fail cleanly:\n{result.stdout}\n{result.stderr}"
    )
    out = np.load(out_dir / names[0])
    assert out.shape == (64, 64) and out.dtype == np.float32


def test_missing_weights_fails_cleanly(tmp_path):
    """A missing weights file must produce a helpful message and exit 1 --
    never a traceback or a hang.

    This points --weights at a path that does not exist, rather than relying on
    the weights directory being empty. The earlier version of this test asserted on
    the bare invocation and silently inverted its meaning the moment real
    weights were trained: the repo state it depended on was not its own to
    control. test_bare_invocation_succeeds_with_real_weights below now covers
    the bare-invocation contract.
    """
    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    _tiny_input(in_dir)

    result = subprocess.run(
        [sys.executable, "evaluate.py",
         "--input_dir", str(in_dir), "--output_dir", str(out_dir),
         "--weights", str(tmp_path / "does_not_exist.pth")],
        cwd=REPO, capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 1, (
        f"expected a clean exit(1), got {result.returncode}:\n"
        f"{result.stdout}\n{result.stderr}"
    )
    assert "Model weights not found" in result.stdout
    assert result.stderr == "" or "Traceback" not in result.stderr


def test_bare_invocation_succeeds_with_real_weights(tmp_path):
    """`evaluate.py --input_dir X --output_dir Y` with no other arguments must
    work unattended. (run.py is the submission entry point; this covers the
    older flag-based CLI.) With the shipped weights in models/ it must succeed
    and write correctly-shaped outputs.

    Skipped when the shipped checkpoint is absent, so the suite still passes on
    a fresh clone before training has been run.
    """
    import pytest

    shipped = REPO / "models" / "drunet_sr_inference.pth"
    if not shipped.is_file():
        # Fall back to the older layout so an unmigrated clone still exercises
        # this test rather than silently skipping it.
        shipped = REPO / "model_weights" / "drunet_sr_inference.pth"
    if not shipped.is_file():
        pytest.skip("drunet_sr_inference.pth not present in models/ or model_weights/")

    in_dir, out_dir = tmp_path / "in", tmp_path / "out"
    names = _tiny_input(in_dir)

    result = subprocess.run(
        [sys.executable, "evaluate.py",
         "--input_dir", str(in_dir), "--output_dir", str(out_dir)],
        cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    assert result.returncode == 0, (
        f"the organisers' command failed with {result.returncode}:\n"
        f"{result.stdout}\n{result.stderr}"
    )

    written = sorted(p.name for p in out_dir.glob("*.npy"))
    assert written == sorted(names), f"expected {sorted(names)}, got {written}"
    for name in written:
        src = np.load(in_dir / name)
        arr = np.load(out_dir / name)
        assert arr.dtype == np.float32, f"{name}: dtype {arr.dtype}, expected float32"
        # The shipped checkpoint has sf=2, so a 32x32 input must give 64x64.
        assert arr.shape == (src.shape[0] * 2, src.shape[1] * 2), (
            f"{name}: {src.shape} -> {arr.shape}, expected 2x upscale"
        )
        assert 0.0 <= float(arr.min()) and float(arr.max()) <= 1.0
