import pytest
import torch

from sincnet import SincNet, STFT, create_spectrogram_transform
from sincnet.mulaw import (MuLawQuant, PolarMuLawQuant, PredictivePolarQuant,
                           DemodulatedPolarQuant, TrigMuLawQuant)


@pytest.mark.parametrize("length", [1000, 1023, 1024])
@pytest.mark.parametrize("mono_channel", [False, True])
@pytest.mark.parametrize("overlap", [2, 4])
def test_factory_backend_shape_parity(length, mono_channel, overlap):
    params = dict(fs=1024, fps=32, n_bins=64, overlap=overlap,
                  component="complex", layout="channel", scale="lin",
                  causal=False, apply_sinc_envelope=False, q_bits=8)
    original = params.copy()
    wav = torch.randn(2, 1, length) if mono_channel else torch.randn(2, length)
    expected = (2, 2, 64, 1 + length // 32)
    for spec_type, cls in [("stft", STFT), ("sincnet", SincNet)]:
        config = dict(params, spec_type=spec_type)
        model = create_spectrogram_transform(config)
        assert isinstance(model, cls)
        spec = model.encode(wav)
        assert spec.shape == expected
        assert not spec.is_complex()
        assert model.decode(spec, length=length).shape == (2, length)
        assert config == dict(original, spec_type=spec_type)
    assert params == original


def test_factory_defaults_have_two_channels():
    for spec_type in ("stft", "sincnet"):
        model = create_spectrogram_transform({"spec_type": spec_type})
        assert model.encode(torch.randn(2, 1000)).shape == (2, 2, 128, 9)


@pytest.mark.parametrize("spec_type,decoder", [
    ("stft", "fast"), ("sincnet", "fast"), ("sincnet", "exact"),
])
def test_factory_default_decode_length_and_gradient(spec_type, decoder):
    model = create_spectrogram_transform(dict(spec_type=spec_type, fs=1024, fps=32,
                                    n_bins=64, decoder_type=decoder, causal=False))
    spec = model.encode(torch.randn(2, 1024)).detach().requires_grad_(True)
    wav = model.decode(spec)
    assert wav.shape == (2, 1024)
    wav.square().mean().backward()
    assert spec.grad is not None
    assert torch.isfinite(spec.grad).all()
    assert spec.grad.abs().sum() > 0


@pytest.mark.parametrize("params, error", [
    ({}, ValueError),
    ({"spec_type": "other"}, ValueError),
    ({"spec_type": "stft", "n_bnis": 128}, TypeError),
    ({"spec_type": "sincnet", "component": "real"}, ValueError),
    ({"spec_type": "stft", "layout": "complex"}, ValueError),
    ({"spec_type": "sincnet", "fps": 300}, ValueError),
    ({"spec_type": "sincnet", "fps": 0}, ValueError),
    ({"spec_type": "stft", "overlap": 4, "n_bins": 128}, ValueError),
])
def test_factory_rejects_invalid_config(params, error):
    with pytest.raises(error):
        create_spectrogram_transform(params)


@pytest.mark.parametrize("coordinate_system,cls,extra,channels", [
    ("cartesian", MuLawQuant, {}, 2),
    ("polar", PolarMuLawQuant, {"q_mag": 7, "q_phi": 5}, 2),
    ("predictive", PredictivePolarQuant, {"q_mag": 7, "q_phi": 5,
                                         "mag_silence_threshold": 2}, 2),
    ("demodulated", DemodulatedPolarQuant, {"q_mag": 7, "q_phi": 5}, 2),
    ("trig", TrigMuLawQuant, {"q_mag": 7, "q_trig": 5}, 3),
])
@pytest.mark.parametrize("spec_type", ["stft", "sincnet"])
def test_factory_mulaw_configuration_and_roundtrip(spec_type, coordinate_system, cls, extra, channels):
    settings = dict(extra, coordinate_system=coordinate_system, pre_scaling=True, eps=1e-6, dither=False)
    model = create_spectrogram_transform(dict(
        spec_type=spec_type, fs=1024, fps=32, n_bins=64, q_bits=8,
        mulaw_kwargs=settings,
    ))
    quantizer = model.mulaw
    assert type(quantizer) is cls
    assert quantizer.q_bits == 8
    assert quantizer.pre_scaling is True
    assert quantizer.eps == 1e-6
    for key, value in extra.items():
        assert getattr(quantizer, key) == value
    spec = model.encode(torch.randn(2, 1024))
    tokens, scale = quantizer.quantize(spec)
    assert tokens.shape == (2, channels, 64, 33)
    reconstructed = quantizer.dequantize(tokens, scale)
    assert reconstructed.shape == spec.shape
    assert torch.isfinite(reconstructed).all()
    assert model.decode(reconstructed, length=1024).shape == (2, 1024)
    assert settings == dict(extra, coordinate_system=coordinate_system, pre_scaling=True, eps=1e-6, dither=False)
    if coordinate_system == "demodulated":
        expected = (torch.arange(64) * 8 if spec_type == "stft"
                    else model.encoder.get_frequencies())
        torch.testing.assert_close(quantizer.center_frequencies_hz, expected.float())
        assert quantizer.frame_rate == 32


def test_factory_mulaw_defaults_match():
    for spec_type in ("stft", "sincnet"):
        quantizer = create_spectrogram_transform({"spec_type": spec_type}).mulaw
        assert type(quantizer) is PolarMuLawQuant
        assert quantizer.q_bits == 6
        assert quantizer.pre_scaling is False


@pytest.mark.parametrize("spec_type", ["stft", "sincnet"])
def test_factory_nested_mulaw_options_override_defaults_without_mutation(spec_type):
    settings = {"q_bits": 7, "pre_scaling": True}
    model = create_spectrogram_transform(dict(spec_type=spec_type, q_bits=8,
                                    mulaw_kwargs=settings))
    assert type(model.mulaw) is PolarMuLawQuant
    assert model.mulaw.q_bits == 7
    assert model.mulaw.pre_scaling is True
    assert settings == {"q_bits": 7, "pre_scaling": True}


@pytest.mark.parametrize("cls", [STFT, SincNet])
def test_direct_constructor_mulaw_and_reinitialization(cls):
    model = cls(q_bits=7,
                mulaw_kwargs={"coordinate_system": "predictive", "pre_scaling": True,
                              "q_phi": 4, "dither": True})
    assert isinstance(model.mulaw, PredictivePolarQuant)
    assert model.mulaw.q_phi == 4
    assert model.mulaw.dither is True
    assert model.mulaw.pre_scaling is True
    quantizer = model.initialise_mulaw("demodulated", q_bits=8, pre_scaling=True, q_phi=5)
    assert model.mulaw is quantizer
    assert isinstance(quantizer, DemodulatedPolarQuant)
    assert quantizer.q_phi == 5


@pytest.mark.parametrize("spec_type", ["stft", "sincnet"])
@pytest.mark.parametrize("settings,error", [
    ({"mulaw_kwargs": {"coordinate_system": "unknown"}}, ValueError),
    ({"mulaw_kwargs": {"coordinate_system": "cartesian", "q_phi": 4}}, TypeError),
    ({"mulaw_kwargs": {"typo": 1}}, TypeError),
])
def test_factory_rejects_invalid_mulaw_settings(spec_type, settings, error):
    with pytest.raises(error):
        create_spectrogram_transform(dict(settings, spec_type=spec_type))
