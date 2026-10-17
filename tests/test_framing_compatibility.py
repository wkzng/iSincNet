"""Consumer regression: SincNet's framing predates the STFT factory."""
import pytest
import torch
import torch.nn.functional as F

from sincnet import SincNet, STFT
from sincnet.mulaw import PolarMuLawQuant


@pytest.mark.parametrize("hops", [256, 320])
@pytest.mark.parametrize("delta", [-1, 0, 1])
@pytest.mark.parametrize("component", ["real", "complex"])
def test_original_sincnet_frames_and_coefficients(hops, delta, component):
    model = SincNet(component=component)
    length = hops * 125 + delta
    waveform = torch.randn(1, length)
    encoder = model.encoder
    padded = F.pad(waveform.unsqueeze(1), (encoder.padding, encoder.padding), mode="reflect")
    real = F.conv1d(padded, encoder.filters.real, stride=125)
    if component == "complex":
        imag = F.conv1d(padded, encoder.filters.imag, stride=125)
        legacy = torch.stack([real, imag], dim=1)
    else:
        legacy = real.unsqueeze(1)
    actual = model.encode(waveform)
    assert actual.shape[-1] == (length + 124) // 125
    torch.testing.assert_close(actual, legacy, rtol=0, atol=0)
    if delta == 0:
        assert actual.shape[-1] == hops
        # Exercise skip concatenation after five temporal down/up stages.
        skips = []
        value = actual
        for _ in range(5):
            skips.append(value)
            value = F.avg_pool2d(value, (1, 2))
        for skip in reversed(skips):
            value = F.interpolate(value, scale_factor=(1, 2), mode="nearest")
            value = torch.cat([value, skip], dim=1)
        assert value.shape[-1] == hops


@pytest.mark.parametrize("decoder", ["fast", "exact", "learnt"])
def test_original_default_decoder_length(decoder):
    model = SincNet(fs=1024, fps=32, n_bins=64, component="complex",
                    causal=False, decoder_type=decoder)
    waveform = torch.randn(1, 1024)
    spec = model.encode(waveform)
    assert spec.shape[-1] == 32
    assert model.decode(spec).shape == waveform.shape
    if decoder != "learnt":
        nonaligned = torch.randn(1, 1023)
        assert model.decode(model.encode(nonaligned), length=1023).shape == nonaligned.shape


@pytest.mark.parametrize("fs", [16000, 16384])  # odd and even hops
@pytest.mark.parametrize("n_bins,overlap", [(128, 2), (256, 4)])
@pytest.mark.parametrize("hops,delta", [(256, -1), (256, 0), (256, 1),
                                        (320, -1), (320, 0), (320, 1)])
def test_half_hop_stft_matches_original_sincnet_frame_count(fs, n_bins, overlap, hops, delta):
    model = STFT(fs=fs, fps=128, n_bins=n_bins, overlap=overlap)
    hop = fs // 128
    length = hops * hop + delta
    spec = model.encode(torch.randn(1, length))
    frames = (length + hop - 1) // hop
    assert spec.shape == (1, 2, n_bins, frames)
    assert model.frame_offset == hop // 2
    assert model.decode(spec).shape == (1, frames * hop)
    assert model.decode(spec, length=length).shape == (1, length)
    assert model.encode(model.decode(spec)).shape == spec.shape


@pytest.mark.parametrize("overlap,n_bins", [(2, 128), (4, 256)])
@pytest.mark.parametrize("mode", ["phase", "polar6", "polar8"])
def test_half_hop_modified_spectra_have_no_boundary_amplification(overlap, n_bins, mode):
    torch.manual_seed(123)
    model = STFT(n_bins=n_bins, overlap=overlap)
    t = torch.arange(32000) / 16000
    x = .2 * torch.sin(2 * torch.pi * 440 * t).unsqueeze(0)
    spec = model.encode(x)
    if mode == "phase":
        magnitude = model._to_complex(spec).abs()
        spec = torch.polar(magnitude, 2 * torch.pi * torch.rand_like(magnitude))
    else:
        q = PolarMuLawQuant(q_bits=int(mode[5:]), pre_scaling=True)
        tokens, scale = q.quantize(spec)
        spec = q.dequantize(tokens, scale)
    y = model.decode(spec)
    assert torch.isfinite(y).all()
    edges = torch.cat([y[:, :125], y[:, -125:]], dim=-1)
    assert edges.abs().max() / y.square().mean().sqrt() < 10
    assert y.abs().max() < 1


@pytest.mark.parametrize("layout", ["channel", "complex"])
def test_half_hop_encode_decode_adjoint_and_old_window_state(layout):
    model = STFT(fs=4096, fps=128, n_bins=64, layout=layout).double()
    model.load_state_dict({"window": model.window.clone()})
    x = torch.randn(1, 1000, dtype=torch.float64, requires_grad=True)
    y = model.decode(model.encode(x), length=1000)
    direction = torch.randn_like(y)
    (y * direction).sum().backward()
    assert torch.isfinite(x.grad).all()
    torch.testing.assert_close((y * direction).sum(), (x * x.grad).sum(), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("length", [0, -1, 1001])
def test_half_hop_decode_rejects_uncovered_lengths(length):
    model = STFT()
    spec = model.encode(torch.randn(1, 1000))  # eight hops = 1000 samples
    with pytest.raises(ValueError, match="length"):
        model.decode(spec, length=length)
