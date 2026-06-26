import torch

from sincnet.model import SincNet
from sincnet.mulaw import DemodulatedPolarQuant, PredictivePolarQuant


def test_predictive_polar_quant_defaults_to_shared_q_bits():
    q = PredictivePolarQuant(q_bits=6)
    assert q.q_mag == 6
    assert q.q_phi == 6
    assert q.mag_vocab_size == 64
    assert q.phase_vocab_size == 64


def test_predictive_polar_quant_allows_phase_override():
    q = PredictivePolarQuant(q_bits=6, q_phi=4)
    assert q.q_mag == 6
    assert q.q_phi == 4
    assert q.mag_vocab_size == 64
    assert q.phase_vocab_size == 16


def test_predictive_polar_quantize_shapes_and_ranges():
    q = PredictivePolarQuant(q_bits=6, q_phi=4, mag_silence_threshold=0)
    x = torch.randn(2, 2, 8, 5)

    tokens, scale = q.quantize(x)
    mag_tokens = tokens[:, 0]
    phi_tokens = tokens[:, 1]

    assert tokens.shape == x.shape
    assert tokens.dtype == torch.long
    assert scale.shape == (2, 1, 1, 1)
    assert mag_tokens.min() >= 0
    assert mag_tokens.max() < q.mag_vocab_size
    assert phi_tokens.min() >= 1
    assert phi_tokens.max() < q.phase_vocab_size


def test_predictive_polar_silence_gate_reserves_phase_zero():
    q = PredictivePolarQuant(q_bits=6, q_phi=4, mag_silence_threshold=4)
    x = torch.zeros(2, 2, 8, 5)

    tokens, _ = q.quantize(x)
    mag_tokens = tokens[:, 0]
    phi_tokens = tokens[:, 1]

    assert torch.all(mag_tokens < q.mag_silence_threshold)
    assert torch.all(phi_tokens == q.SILENCE_TOKEN)


def test_predictive_polar_dequantize_and_forward_roundtrip():
    q = PredictivePolarQuant(q_bits=6, q_phi=4)
    x = torch.randn(2, 2, 8, 5)

    tokens, scale = q.quantize(x)
    y = q.dequantize(tokens, scale)
    z = q.forward(x)

    assert y.shape == x.shape
    assert torch.isfinite(y).all()
    assert torch.allclose(z, y)


def test_sincnet_initialise_mulaw_predictive():
    model = SincNet(
        fs=16000,
        fps=128,
        scale="lin",
        component="complex",
        n_bins=128,
        causal=False,
    )

    quantizer = model.initialise_mulaw("predictive", q_bits=6)

    assert isinstance(quantizer, PredictivePolarQuant)
    assert model.mulaw is quantizer
    assert quantizer.q_mag == 6
    assert quantizer.q_phi == 6


def test_demodulated_polar_quantize_dequantize_shapes():
    q = DemodulatedPolarQuant(
        center_frequencies_hz=torch.linspace(0, 4000, 8),
        frame_rate=128,
        q_bits=6,
        q_phi=4,
    )
    x = torch.randn(2, 2, 8, 5)

    tokens, scale = q.quantize(x)
    y = q.dequantize(tokens, scale)

    assert tokens.shape == x.shape
    assert tokens.dtype == torch.long
    assert scale.shape == (2, 1, 1, 1)
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_demodulated_polar_removes_bin_carrier_phase():
    centers = torch.tensor([32.0])
    frame_rate = 128
    q = DemodulatedPolarQuant(
        center_frequencies_hz=centers,
        frame_rate=frame_rate,
        q_bits=6,
    )
    t = torch.arange(8).float()
    phase = 2 * torch.pi * centers[0] * t / frame_rate
    x = torch.stack([torch.cos(phase), torch.sin(phase)], dim=0).reshape(1, 2, 1, -1)

    polar, _ = q.compand(x)
    demodulated_phase = polar[:, 1] * (2 * torch.pi) - torch.pi

    assert torch.allclose(demodulated_phase, torch.zeros_like(demodulated_phase), atol=1e-5)


def test_sincnet_initialise_mulaw_demodulated():
    model = SincNet(
        fs=16000,
        fps=128,
        scale="lin",
        component="complex",
        n_bins=128,
        causal=False,
    )

    quantizer = model.initialise_mulaw("demodulated", q_bits=6)

    assert isinstance(quantizer, DemodulatedPolarQuant)
    assert model.mulaw is quantizer
    assert quantizer.center_frequencies_hz.numel() == model.config.n_bins
    assert quantizer.frame_rate == model.config.fps
