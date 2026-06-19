import numpy as np
import torch
import pytest

from sincnet.model import (
    SincNet,
    lin_freqs, mel_freqs, bark_freqs, erb_freqs,
)


SCALES = {"lin": lin_freqs, "mel": mel_freqs, "bark": bark_freqs, "erb": erb_freqs}


@pytest.mark.parametrize("name,freqs", SCALES.items())
def test_freqs_cover_the_spectrum(name, freqs):
    fs, n_bins = 16000, 128
    centers, bands = freqs(fs=fs, n_bins=n_bins)

    assert centers.shape == (n_bins,)
    assert bands.shape == (n_bins,)
    assert np.isclose(centers[0], 0, atol=1e-6), f"{name} centers must start at 0 Hz"
    assert np.isclose(centers[-1], fs / 2, rtol=1e-6), f"{name} centers must end at Nyquist"
    assert np.all(np.diff(centers) > 0), f"{name} centers must be strictly increasing"
    assert np.all(bands > 0), f"{name} bandwidths must be positive"


@pytest.mark.parametrize("name", ["bark", "erb"])
def test_warped_scales_are_finer_at_low_frequencies(name):
    centers, bands = SCALES[name](fs=16000, n_bins=128)
    assert np.diff(centers)[0] < np.diff(centers)[-1]
    assert bands[0] < bands[-1]


@pytest.mark.parametrize("scale", ["bark", "erb"])
def test_sincnet_builds_and_round_trips(scale):
    torch.manual_seed(0)
    model = SincNet(fs=16000, fps=100, scale=scale, component="complex", n_bins=32, causal=False)
    wav = torch.randn(1, 8000)
    spec = model.encode(wav)
    assert spec.shape[:3] == (1, 2, 32)
    rec = model.decode(spec)
    assert rec.ndim == 2 and rec.size(0) == 1
    assert "bins_" + scale in model.name
