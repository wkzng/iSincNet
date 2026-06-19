import torch

from training.utils.losses import MultiResolutionSTFTLoss


def test_mrstft_zero_for_identical_signals():
    loss = MultiResolutionSTFTLoss(fft_sizes=(256, 512), hop_sizes=(64, 128), win_lengths=(256, 512))
    x = torch.randn(2, 16000)
    assert loss(x, x).item() < 1e-5


def test_mrstft_decreases_as_estimate_approaches_target():
    loss = MultiResolutionSTFTLoss(fft_sizes=(256, 512), hop_sizes=(64, 128), win_lengths=(256, 512))
    torch.manual_seed(0)
    y = torch.randn(2, 16000)
    far = loss(torch.randn(2, 16000), y)          # unrelated
    near = loss(y + 0.01 * torch.randn(2, 16000), y)  # close
    assert near < far and near.item() >= 0.0


def test_mrstft_accepts_b1l_shape():
    loss = MultiResolutionSTFTLoss(fft_sizes=(256,), hop_sizes=(64,), win_lengths=(256,))
    y = torch.randn(2, 8000)
    assert torch.isfinite(loss(y.unsqueeze(1), y))   # (B,1,L) vs (B,L)
