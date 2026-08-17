import torch
import torch.nn.functional as F

from utils.utils_pnp import data_fidelity_sr, mu_schedule


def _fixture():
    torch.manual_seed(0)
    gt = torch.rand(1, 1, 256, 256)
    y = F.interpolate(gt, scale_factor=0.5, mode="bicubic",
                      align_corners=False, antialias=True)
    z = F.interpolate(y, scale_factor=2, mode="bicubic", align_corners=False)
    return y, z


def test_solver_stays_bounded_across_the_whole_mu_schedule():
    """The mu schedule starts at 0.01. A fixed lr=0.2 blows up to 1e4 there."""
    y, z = _fixture()
    for k in range(15):
        mu = mu_schedule(k, 15)
        x = data_fidelity_sr(y, z, mu, 2)
        assert torch.isfinite(x).all(), f"k={k} mu={mu:.4f} produced non-finite values"
        assert x.abs().max() < 2.0, \
            f"k={k} mu={mu:.4f} diverged: max|x|={x.abs().max().item():.3e}"


def test_solver_decreases_the_hqs_objective_at_the_hardest_mu():
    """A data-fidelity step must reduce the energy it is minimising."""
    y, z = _fixture()
    mu = 0.01

    def energy(x):
        Dx = F.interpolate(x, scale_factor=0.5, mode="bicubic",
                           align_corners=False, antialias=True)
        return (torch.sum((y - Dx) ** 2) / (2 * mu) + torch.sum((x - z) ** 2) / 2).item()

    assert energy(data_fidelity_sr(y, z, mu, 2)) < energy(z)


def test_solver_moves_the_estimate_toward_consistency_with_y():
    """After the step, re-degrading x must match y better than z did."""
    y, z = _fixture()
    x = data_fidelity_sr(y, z, 0.01, 2)

    def residual(v):
        Dv = F.interpolate(v, scale_factor=0.5, mode="bicubic",
                           align_corners=False, antialias=True)
        return torch.mean((y - Dv) ** 2).item()

    assert residual(x) < residual(z)
