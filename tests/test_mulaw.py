import torch
import pytest

from sincnet.model import SincNet
from sincnet.mulaw import DemodulatedPolarQuant, PolarMuLawQuant, PredictivePolarQuant, TrigMuLawQuant


def test_polar_quant_default_keeps_full_phase_vocabulary():
    q = PolarMuLawQuant(q_bits=4)
    tokens = torch.zeros(1, 2, 1, 1, dtype=torch.long)
    tokens[:, 0] = q.mag_vocab_size - 1

    y = q.dequantize(tokens, scale=1.0)

    assert q.mag_silence_threshold == 0
    assert q.phase_levels == q.phase_vocab_size
    assert y[:, 0].item() < 0
    assert torch.allclose(y[:, 1], torch.zeros_like(y[:, 1]), atol=1e-6)


def test_polar_quant_silence_gate_reserves_phase_zero():
    q = PolarMuLawQuant(q_bits=6, q_phi=4, mag_silence_threshold=4)
    x = torch.zeros(2, 2, 8, 5)

    tokens, _ = q.quantize(x)
    mag_tokens = tokens[:, 0]
    phi_tokens = tokens[:, 1]

    assert q.phase_levels == q.phase_vocab_size - 1
    assert torch.all(mag_tokens < q.mag_silence_threshold)
    assert torch.all(phi_tokens == q.SILENCE_TOKEN)


def test_polar_quant_non_silent_phase_tokens_are_shifted():
    q = PolarMuLawQuant(q_bits=6, q_phi=4, mag_silence_threshold=4)
    x = torch.randn(2, 2, 8, 5)

    tokens, _ = q.quantize(x)
    mag_tokens = tokens[:, 0]
    phi_tokens = tokens[:, 1]
    non_silent = mag_tokens >= q.mag_silence_threshold

    assert torch.all(phi_tokens[non_silent] >= 1)
    assert torch.all(phi_tokens[non_silent] < q.phase_vocab_size)


def test_polar_quant_silent_phase_dequantizes_to_zero_angle():
    q = PolarMuLawQuant(q_bits=6, q_phi=4, mag_silence_threshold=4)
    tokens = torch.zeros(1, 2, 1, 1, dtype=torch.long)
    tokens[:, 0] = q.mag_vocab_size - 1

    y = q.dequantize(tokens, scale=1.0)

    assert y[:, 0].item() > 0
    assert torch.allclose(y[:, 1], torch.zeros_like(y[:, 1]), atol=1e-6)


def test_polar_quant_rejects_threshold_equal_to_vocab_size():
    with pytest.raises(ValueError, match="mag_silence_threshold"):
        PolarMuLawQuant(q_bits=4, mag_silence_threshold=16)


def test_trig_mulaw_quantize_dequantize_shapes_and_ranges():
    q = TrigMuLawQuant(q_mag=6, q_trig=4)
    x = torch.randn(2, 2, 8, 5)

    tokens, scale = q.quantize(x)
    y = q.dequantize(tokens, scale)

    assert tokens.shape == (2, 3, 8, 5)
    assert tokens.dtype == torch.long
    assert scale.shape == (2, 1, 1, 1)
    assert tokens[:, 0].min() >= 0
    assert tokens[:, 0].max() < q.mag_vocab_size
    assert tokens[:, 1].min() >= 0
    assert tokens[:, 1].max() < q.trig_vocab_size
    assert tokens[:, 2].min() >= 0
    assert tokens[:, 2].max() < q.trig_vocab_size
    assert y.shape == x.shape
    assert torch.isfinite(y).all()


def test_trig_mulaw_zero_input_roundtrips_to_zero():
    q = TrigMuLawQuant(q_bits=6)
    x = torch.zeros(2, 2, 8, 5)

    tokens, scale = q.quantize(x)
    y = q.dequantize(tokens, scale)

    assert torch.allclose(y, torch.zeros_like(y), atol=1e-7)


def test_trig_mulaw_dequantize_renormalizes_direction():
    q = TrigMuLawQuant(q_mag=8, q_trig=2)
    tokens = torch.zeros(1, 3, 1, 1, dtype=torch.long)
    tokens[:, 0] = q.mag_vocab_size - 1
    tokens[:, 1] = q.trig_vocab_size - 1
    tokens[:, 2] = q.trig_vocab_size - 1

    y = q.dequantize(tokens, scale=1.0)
    magnitude = torch.sqrt((y**2).sum(dim=1))

    assert torch.allclose(magnitude, torch.ones_like(magnitude), atol=1e-6)


def test_sincnet_initialise_mulaw_trig():
    model = SincNet(
        fs=16000,
        fps=128,
        scale="lin",
        component="complex",
        n_bins=128,
        causal=False,
    )

    quantizer = model.initialise_mulaw("trig", q_bits=6)

    assert isinstance(quantizer, TrigMuLawQuant)
    assert model.mulaw is quantizer
    assert quantizer.q_mag == 6
    assert quantizer.q_trig == 6


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


def test_predictive_polar_rejects_threshold_equal_to_vocab_size():
    with pytest.raises(ValueError, match="mag_silence_threshold"):
        PredictivePolarQuant(q_bits=4, mag_silence_threshold=16)


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


def test_predictive_polar_silent_delta_drops_phase_increment():
    q = PredictivePolarQuant(q_mag=2, q_phi=3, mag_silence_threshold=1)
    tokens = torch.full((1, 2, 1, 3), q.mag_vocab_size - 1, dtype=torch.long)
    tokens[:, 1, :, 0] = 4
    tokens[:, 1, :, 1] = q.SILENCE_TOKEN
    tokens[:, 1, :, 2] = 5

    y = q.dequantize(tokens, scale=1.0)
    phase = torch.atan2(y[:, 1], y[:, 0])

    assert torch.allclose(phase[..., 1], phase[..., 0], atol=1e-5)
    assert phase[..., 2].item() > phase[..., 1].item()


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
