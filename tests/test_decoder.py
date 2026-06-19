import torch

from sincnet import SincNet, frame_inverse
from sincnet.model import FastAnalyticDecoder1d, AnalyticDecoder1d, Decoder1d, LearnedEqualizerDecoder1d


def _snr(a, b):
    n = min(a.shape[-1], b.shape[-1])
    a, b = a[..., :n], b[..., :n]
    return 10 * torch.log10(a.pow(2).sum() / (a - b).pow(2).sum()).item()


def _lin(**kw):
    return SincNet(fs=16000, fps=128, scale="lin", component="complex", n_bins=128, causal=False, **kw)


# ---- decoder selection ----

def test_default_decoder_is_fast():
    assert isinstance(_lin().decoder, FastAnalyticDecoder1d)


def test_decoder_type_selects_the_module():
    assert isinstance(_lin(decoder_type="fast").decoder, FastAnalyticDecoder1d)
    assert isinstance(_lin(decoder_type="exact").decoder, AnalyticDecoder1d)
    assert isinstance(_lin(decoder_type="learnt").decoder, Decoder1d)
    assert isinstance(_lin(decoder_type="semi_learnt").decoder, LearnedEqualizerDecoder1d)


def test_only_learnt_has_trainable_weights():
    assert sum(p.numel() for p in _lin(decoder_type="fast").decoder.parameters()) == 0
    assert sum(p.numel() for p in _lin(decoder_type="exact").decoder.parameters()) == 0
    assert any(p.requires_grad for p in _lin(decoder_type="learnt").decoder.parameters())
    assert any(p.requires_grad for p in _lin(decoder_type="semi_learnt").decoder.parameters())


def test_exact_decoder_not_double_registered():
    """AnalyticDecoder1d holds the encoder by reference -> no decoder.* keys in the state_dict."""
    keys = _lin(decoder_type="exact").state_dict().keys()
    assert not any(k.startswith("decoder.") for k in keys)


# ---- reconstruction via the shared decode() API ----

def test_fast_decode_is_decent_and_length_safe():
    torch.manual_seed(0)
    m = _lin(decoder_type="fast")
    x = torch.randn(1, 4000)
    xhat = m.decode(m.encode(x), length=4000)
    assert xhat.shape == (1, 4000)
    assert _snr(x, xhat) > 15            # single-pass equalizer ~ tens of dB


def test_exact_decode_is_exact_and_length_safe():
    torch.manual_seed(0)
    m = _lin(decoder_type="exact")
    x = torch.randn(1, 4000)
    xhat = m.decode(m.encode(x), length=4000)
    assert xhat.shape == (1, 4000)
    assert _snr(x, xhat) > 100


def test_forward_preserves_input_length_exactly():
    torch.manual_seed(0)
    for dt in ("fast", "exact"):
        x = torch.randn(2, 4123)         # not a multiple of hop
        assert _lin(decoder_type=dt)(x).shape == (2, 4123)


def test_learnt_decode_runs_and_emits_frames_times_hop():
    m = _lin(decoder_type="learnt")
    spec = m.encode(torch.randn(1, 4000))
    assert m.decode(spec).shape[-1] == spec.shape[-1] * m.config.hop_length


def test_semi_learnt_decode_is_decent_and_length_safe():
    torch.manual_seed(0)
    m = _lin(decoder_type="semi_learnt")
    x = torch.randn(1, 4000)
    xhat = m.decode(m.encode(x), length=4000)
    assert xhat.shape == (1, 4000)
    # n_bins=128 is under-conditioned (nullspace near-singularities); FIR init gives ~12 dB.
    # Well-conditioned configs (n_bins >= 256) start at ~31 dB. Both converge toward CG quality
    # after pretrain_on_noise().
    assert _snr(x, xhat) > 8


def test_semi_learnt_has_fir_taps_equal_to_kernel_size():
    m = _lin(decoder_type="semi_learnt")
    assert m.decoder.fir.shape == (1, 1, m.config.kernel_size)


def test_semi_learnt_is_differentiable():
    torch.manual_seed(0)
    m = _lin(decoder_type="semi_learnt")
    s = m.encode(torch.randn(1, 4000)).requires_grad_(True)
    m.decode(s, length=4000).pow(2).sum().backward()
    assert s.grad is not None and float(s.grad.abs().sum()) > 0


# ---- frame_inverse core ----

def test_frame_inverse_standalone():
    torch.manual_seed(0)
    m = _lin()
    x = torch.randn(1, 4000)
    assert _snr(x, frame_inverse(m.encoder, m.encode(x), length=4000)) > 100


def test_exact_decode_batched():
    torch.manual_seed(0)
    m = _lin(decoder_type="exact")
    x = torch.randn(3, 4000)
    xhat = m.decode(m.encode(x), length=4000)
    assert xhat.shape == (3, 4000) and _snr(x, xhat) > 100


def test_exact_decoder_is_differentiable_with_exact_gradient():
    """exact decode is differentiable (for training); implicit backward is the true adjoint:
    for the linear map x = M s, <M s, y> == <s, Mᵀ y>  (Mᵀ y is what backward returns)."""
    torch.manual_seed(0)
    m = _lin(decoder_type="exact")
    s = m.encode(torch.randn(1, 4000)).requires_grad_(True)
    x = m.decode(s, length=4000)
    y = torch.randn_like(x)
    (x * y).sum().backward()
    assert s.grad is not None and torch.isfinite(s.grad).all()
    lhs = (x * y).sum().item()
    rhs = (s * s.grad).sum().item()
    assert abs(lhs - rhs) / (abs(lhs) + 1e-9) < 1e-4        # adjoint identity -> gradient is exact


def test_fast_decoder_is_also_differentiable():
    torch.manual_seed(0)
    m = _lin(decoder_type="fast")
    s = m.encode(torch.randn(1, 4000)).requires_grad_(True)
    m.decode(s, length=4000).pow(2).sum().backward()
    assert s.grad is not None and float(s.grad.abs().sum()) > 0
