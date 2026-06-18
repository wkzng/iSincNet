import numpy as np
import torch
import pytest

from sincnet import STFT, stft_params


# ---- parametrization ----

def test_stft_params_basic():
    p = stft_params(fs=16000, fps=128, n_bins=128, overlap=2)
    assert p == {"n_fft": 256, "win_length": 250, "hop_length": 125}


def test_stft_params_fps_must_divide_fs():
    with pytest.raises(ValueError):
        stft_params(fs=16000, fps=300, n_bins=128)  # 16000 % 300 != 0


def test_stft_params_coupling_constraint():
    # need n_bins >= overlap*fs/(2*fps) = 2*16000/(2*128) = 125 -> 128 ok, 64 fails
    stft_params(fs=16000, fps=128, n_bins=128, overlap=2)
    with pytest.raises(ValueError):
        stft_params(fs=16000, fps=128, n_bins=64, overlap=2)


def test_n_bins_must_be_power_of_two():
    with pytest.raises(AssertionError):
        STFT(fs=16000, fps=128, n_bins=129)


# ---- shapes / API / layouts ----

@pytest.mark.parametrize("shape", [(16000,), (3, 16000), (3, 1, 16000)])
def test_channel_layout_shapes(shape):
    m = STFT(fs=16000, fps=128, n_bins=128, layout="channel")
    spec = m.encode(torch.randn(*shape))
    B = 1 if len(shape) == 1 else shape[0]
    assert spec.shape[:2] == (B, 2) and spec.shape[2] == 128 and not spec.is_complex()
    assert m.decode(spec, length=16000).shape == (B, 16000)


def test_complex_layout_shapes():
    m = STFT(fs=16000, fps=128, n_bins=128, layout="complex")
    spec = m.encode(torch.randn(3, 16000))
    assert spec.shape[0] == 3 and spec.shape[1] == 128 and spec.is_complex()
    assert m.decode(spec, length=16000).shape == (3, 16000)


def test_decode_accepts_either_layout():
    """decode auto-detects complex vs (B,2,F,T) so layouts are interchangeable."""
    x = torch.randn(2, 16000)
    chan = STFT(fs=16000, fps=128, n_bins=128, layout="channel")
    cplx = STFT(fs=16000, fps=128, n_bins=128, layout="complex")
    y_chan = chan.decode(chan.encode(x), length=16000)
    y_cplx = cplx.decode(cplx.encode(x), length=16000)
    assert torch.allclose(y_chan, y_cplx, atol=1e-5)


def test_fps_matches_hop():
    fps, fs = 128, 16000
    m = STFT(fs=fs, fps=fps, n_bins=128)
    assert m.hop_length == fs // fps
    for seconds in (1, 2, 3):
        L = fs * seconds
        spec = m.encode(torch.randn(L))
        # exactly fps frames per second: T == L // hop == fps * seconds (no boundary +1 frame)
        assert spec.shape[-1] == L // m.hop_length == fps * seconds


# ---- invertibility ----

@pytest.mark.parametrize("fps,n_bins,overlap", [(128, 128, 2), (128, 256, 2), (64, 256, 2), (128, 256, 4)])
def test_round_trip_is_accurate(fps, n_bins, overlap):
    fs = 16000
    m = STFT(fs=fs, fps=fps, n_bins=n_bins, overlap=overlap)
    # band-limited multitone (representative of audio; not edge-heavy like a full-band sweep)
    t = torch.linspace(0, 2, fs * 2)
    x = sum(0.2 * torch.sin(2 * torch.pi * f * t) for f in (110, 220, 440, 880)).unsqueeze(0)
    y = m.forward(x)
    snr = 10 * torch.log10((x ** 2).sum() / ((x - y) ** 2).sum() + 1e-12)
    assert snr.item() > 40.0, f"SNR {snr.item():.1f} dB too low"


def test_batch_matches_single():
    m = STFT(fs=16000, fps=128, n_bins=128)
    torch.manual_seed(0)
    xs = torch.randn(4, 16000)
    batch = m.forward(xs)
    for i in range(4):
        assert torch.allclose(batch[i], m.forward(xs[i]).squeeze(0), atol=1e-5)


# ---- SincNet-parity perks: magnitude / griffin-lim / mulaw ----

def test_magnitude_shape_both_layouts():
    x = torch.randn(2, 16000)
    for layout in ("channel", "complex"):
        m = STFT(fs=16000, fps=128, n_bins=128, layout=layout)
        mag = m.magnitude(m.encode(x))
        assert mag.shape == (2, 1, 128, m.encode(x).shape[-1]) and not mag.is_complex()


def test_griffin_lim_improves_consistency():
    """GLA always re-imposes the target magnitude; what must improve with iterations is
    spectrogram *consistency*: how well |STFT(iSTFT(spec))| matches the target magnitude."""
    fs = 16000
    m = STFT(fs=fs, fps=128, n_bins=128)
    t = torch.linspace(0, 1, fs)
    x = 0.5 * torch.sin(2 * torch.pi * 220 * t).unsqueeze(0)
    target_mag = m.magnitude(m.encode(x))
    length = target_mag.shape[-1] * m.hop_length   # T*hop -> re-encoding yields exactly T frames

    def consistency(spec):
        return (m.magnitude(m.encode(m.decode(spec, length=length))) - target_mag).norm()

    err0 = consistency(m.griffin_lim(target_mag, n_iters=0))    # random phase
    err1 = consistency(m.griffin_lim(target_mag, n_iters=60))   # refined
    assert err1 < 0.5 * err0, f"GLA did not improve consistency: {err0:.1f} -> {err1:.1f}"


def test_refine_spectrogram_phase_runs():
    m = STFT(fs=16000, fps=128, n_bins=128)
    spec = m.encode(torch.randn(1, 8000))
    refined = m.refine_spectrogram_phase(spec, n_iters=5)
    assert refined.shape == spec.shape


def test_mulaw_quantize_roundtrip_on_channel_layout():
    """The held MuLawQuant operates on the (B,2,F,T) channel layout, like SincNet."""
    m = STFT(fs=16000, fps=128, n_bins=128, q_bits=8, layout="channel")
    spec = m.encode(torch.randn(1, 8000))
    q, scale = m.mulaw.quantize(spec)
    assert q.dtype == torch.long and q.min() >= 0 and q.max() <= 255
    deq = m.mulaw.dequantize(q, scale)
    assert deq.shape == spec.shape
