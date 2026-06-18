import torch
import pytest

from sincnet.model import SincNet, ISTFTDecoder, Decoder1d


def _sinc(decoder="istft", n_fft=None):
    return SincNet(fs=16000, fps=128, n_bins=128, scale="mel", component="complex",
                   causal=False, decoder=decoder, n_fft=n_fft).eval()


def test_default_decoder_is_conv():
    """Backward compatible: the default decoder is still the conv overlap decoder."""
    assert isinstance(SincNet(scale="mel", component="complex", causal=False).decoder, Decoder1d)


def test_decoder_type_in_model_id():
    """model_id distinguishes the decoders; the default conv keeps the legacy (suffix-free) name."""
    base = dict(fs=16000, fps=128, n_bins=128, scale="mel", component="complex", causal=False)
    assert SincNet(**base, decoder="conv").name == "16000fs_128fps_128bins_mel_complex_ncausal"
    assert SincNet(**base, decoder="gnconv").name.endswith("_gnconv")
    assert SincNet(**base, decoder="istft").name.endswith("_istft")


def test_gnconv_has_groupnorm_and_round_trips():
    m = SincNet(fs=16000, fps=128, n_bins=128, scale="mel", component="complex", causal=False, decoder="gnconv")
    assert isinstance(m.decoder, Decoder1d) and isinstance(m.decoder.norm, torch.nn.GroupNorm)
    y = m.decode(m.encode(torch.randn(2, 16000)))
    assert y.shape[0] == 2 and y.ndim == 2


@pytest.mark.parametrize("shape", [(1, 16000), (2, 16000), (2, 1, 16000)])
def test_istft_decoder_round_trip_shape(shape):
    m = _sinc()
    assert isinstance(m.decoder, ISTFTDecoder)
    x = torch.randn(*shape)
    spec = m.encode(x)
    y = m.decode(spec, length=16000)
    assert y.shape == (shape[0], 16000)


@pytest.mark.parametrize("n_fft", [256, 512, 1024])
def test_istft_decoder_n_fft_configurable(n_fft):
    m = _sinc(n_fft=n_fft)
    assert m.decoder.freq_bins == n_fft // 2 + 1
    y = m.decode(m.encode(torch.randn(1, 16000)), length=16000)
    assert y.shape == (1, 16000)


def test_istft_decoder_is_differentiable():
    m = _sinc()
    loss = m.forward(torch.randn(2, 16000)).pow(2).mean()
    loss.backward()
    g = m.decoder.conv1d.weight.grad
    assert g is not None and torch.isfinite(g).all() and g.abs().sum() > 0


def test_istft_decoder_params_not_inflated():
    """The translator head stays in the same ballpark as the conv decoder (<3x)."""
    np = lambda mod: sum(p.numel() for p in mod.parameters())
    conv = _sinc(decoder="conv").decoder
    istft = _sinc(decoder="istft").decoder  # n_fft=256
    assert np(istft) < 3 * np(conv)


def test_forward_threads_length():
    """SincNet.forward should hand the istft decoder the input length for exact sizing."""
    m = _sinc()
    for L in (16000, 12345):
        y = m.forward(torch.randn(1, L))
        assert y.shape == (1, L)
