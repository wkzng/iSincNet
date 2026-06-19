import torch

from sincnet import SincNet, frame_inverse
from sincnet.model import FastAnalyticDecoder1d, AnalyticDecoder1d, Decoder1d


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


def test_only_learnt_has_trainable_weights():
    assert sum(p.numel() for p in _lin(decoder_type="fast").decoder.parameters()) == 0
    assert sum(p.numel() for p in _lin(decoder_type="exact").decoder.parameters()) == 0
    assert any(p.requires_grad for p in _lin(decoder_type="learnt").decoder.parameters())


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
