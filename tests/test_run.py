"""Contract tests for run.py, the submission entry point.

The submission checklist makes specific promises about the output files. These
tests pin the two helpers that enforce them so a refactor cannot quietly break
the contract. They are pure-function tests: no GPU, no weights, no network.
"""

import os
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from run import as_2d, sanitise, find_weights


# --- as_2d: every accepted input layout becomes (H, W) -------------------

def test_as_2d_passes_through_plain_2d():
    a = np.random.rand(8, 6).astype(np.float32)
    assert as_2d(a, "x").shape == (8, 6)


def test_as_2d_squeezes_trailing_channel():
    a = np.random.rand(8, 6, 1).astype(np.float32)
    assert as_2d(a, "x").shape == (8, 6)


def test_as_2d_squeezes_leading_channel():
    a = np.random.rand(1, 8, 6).astype(np.float32)
    assert as_2d(a, "x").shape == (8, 6)


def test_as_2d_always_returns_float32():
    assert as_2d(np.ones((4, 4), dtype=np.float64), "x").dtype == np.float32


def test_as_2d_rejects_multi_channel_colour():
    with pytest.raises(SystemExit):
        as_2d(np.random.rand(8, 6, 3), "colour.npy")


# --- sanitise: finite, float32, inside [0, 1] ----------------------------

def test_sanitise_clips_out_of_range_values():
    out, bad = sanitise(np.array([[-0.5, 0.5, 1.7]]), "x")
    assert bad == 0
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out[0, 1] == pytest.approx(0.5)


def test_sanitise_repairs_nan_and_inf_and_counts_them():
    out, bad = sanitise(np.array([[np.nan, np.inf, -np.inf, 0.25]]), "x")
    assert bad == 3
    assert np.isfinite(out).all()
    assert out[0, 3] == pytest.approx(0.25)


def test_sanitise_reports_zero_for_clean_input():
    _, bad = sanitise(np.full((4, 4), 0.5), "x")
    assert bad == 0


def test_sanitise_returns_float32():
    out, _ = sanitise(np.zeros((4, 4), dtype=np.float64), "x")
    assert out.dtype == np.float32


# --- weight discovery and CLI shape -------------------------------------

def test_find_weights_locates_the_shipped_checkpoint():
    path = find_weights()
    assert os.path.isfile(path) and path.endswith(".pth")


def test_find_weights_exits_on_a_bad_explicit_path():
    with pytest.raises(SystemExit):
        find_weights("no_such_checkpoint.pth")


def test_run_py_takes_two_positional_arguments():
    """The brief specifies `python run.py <input-dir> <output-dir>`."""
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, os.path.join(root, "run.py"), "--help"],
                       capture_output=True, text=True)
    assert r.returncode == 0
    assert "input_dir" in r.stdout and "output_dir" in r.stdout


def test_run_py_exits_non_zero_on_a_missing_input_dir():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r = subprocess.run([sys.executable, os.path.join(root, "run.py"),
                        "definitely_not_a_dir", "out"],
                       capture_output=True, text=True)
    assert r.returncode != 0
