import pytest
import torch
import torch.nn.functional as F

from models.losses import CharbonnierLoss, FrequencyLoss, GradientLoss
from models.losses import _warmup_factor


def test_gradient_loss_is_zero_for_identical_images():
    loss = GradientLoss()
    x = torch.rand(2, 1, 32, 32)
    assert loss(x, x).item() == pytest.approx(0.0, abs=1e-6)


def test_gradient_loss_penalises_a_blurred_edge():
    """A blurred step edge has the same pixel mean as a sharp one but a very
    different gradient field. This is the signal L1 alone cannot see."""
    loss = GradientLoss()
    sharp = torch.zeros(1, 1, 32, 32)
    sharp[:, :, :, 16:] = 1.0
    blurred = F.avg_pool2d(F.pad(sharp, (2, 2, 2, 2), mode="replicate"),
                           kernel_size=5, stride=1)
    assert loss(blurred, sharp).item() > 0.01


def test_gradient_loss_is_differentiable():
    loss = GradientLoss()
    x = torch.rand(1, 1, 16, 16, requires_grad=True)
    loss(x, torch.rand(1, 1, 16, 16)).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_charbonnier_is_zero_for_identical_images():
    x = torch.rand(2, 1, 16, 16)
    assert CharbonnierLoss()(x, x).item() == pytest.approx(1e-3, abs=1e-4)


def test_frequency_loss_is_zero_for_identical_images():
    x = torch.rand(2, 1, 16, 16)
    assert FrequencyLoss()(x, x).item() == pytest.approx(0.0, abs=1e-5)


# --- _warmup_factor -----------------------------------------------------
#
# HybridLoss.forward is not exercised directly in this suite (constructing
# HybridLoss downloads AlexNet weights), so the warm-up branch - the entire
# point of this task's bug fix - would otherwise have zero coverage. These
# tests import the extracted pure function directly instead.

def test_warmup_factor_epoch_none_is_fully_warmed_up():
    assert _warmup_factor(epoch=None, warm_up_epochs=10) == 1.0


def test_warmup_factor_epoch_zero_is_zero():
    assert _warmup_factor(epoch=0, warm_up_epochs=10) == 0.0


def test_warmup_factor_ramp_is_strictly_monotonic():
    values = [_warmup_factor(epoch=e, warm_up_epochs=10) for e in (0, 3, 5, 9)]
    assert values[0] < values[1] < values[2] < values[3]


def test_warmup_factor_clamps_to_one_at_and_beyond_warm_up_epochs():
    assert _warmup_factor(epoch=10, warm_up_epochs=10) == 1.0
    assert _warmup_factor(epoch=50, warm_up_epochs=10) == 1.0


def test_warmup_factor_zero_warm_up_epochs_does_not_raise():
    """warm_up_epochs=0 would divide by zero without the max(1, ...) guard."""
    assert _warmup_factor(epoch=0, warm_up_epochs=0) == 0.0
    assert _warmup_factor(epoch=1, warm_up_epochs=0) == 1.0
