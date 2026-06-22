import torch
import pytest

from sincnet import SincNet, frame_pseudo_inverse
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


def test_exact_decoder_iteration_configuration():
    assert _lin(decoder_type="exact").decoder.n_iter == 128
    assert _lin(decoder_type="exact", cg_iters=16).decoder.n_iter == 16


@pytest.mark.parametrize("cg_iters", [0, -1])
def test_cg_iters_must_be_positive(cg_iters):
    with pytest.raises(ValueError, match="cg_iters must be positive"):
        _lin(cg_iters=cg_iters)


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


def test_fast_decoder_reuses_equalizer_cache():
    model = _lin()
    spec = model.encode(torch.randn(1, 4000))
    model.decode(spec, length=4000)
    assert len(model.decoder._eq_cache) == 1
    cached = next(iter(model.decoder._eq_cache.values()))
    model.decode(spec, length=4000)
    assert next(iter(model.decoder._eq_cache.values())) is cached


def test_fast_decoder_cache_is_keyed_by_device():
    model = _lin()
    spec = model.encode(torch.randn(1, 4000))
    model.decode(spec, length=4000)
    cache_key = next(iter(model.decoder._eq_cache))
    assert cache_key[1] == spec.device


def test_fast_decoder_fused_complex_synthesis_matches_split_calls():
    torch.manual_seed(0)
    model = _lin()
    spec = model.encode(torch.randn(2, 4000))
    decoder = model.decoder

    a = decoder.filters.real.to(spec.dtype)
    b = decoder.filters.imag.to(spec.dtype)
    split = (
        torch.nn.functional.conv_transpose1d(spec[:, 0], a, stride=decoder.stride)
        + torch.nn.functional.conv_transpose1d(spec[:, 1], b, stride=decoder.stride)
    ).squeeze(1)
    split = decoder._equalize(split)
    split = split[..., decoder.padding : decoder.padding + 4000]

    fused = decoder(spec, length=4000)
    assert torch.allclose(fused, split, atol=1e-6, rtol=1e-5)


def test_fast_decoder_serializes_fused_filters():
    model = _lin()
    keys = model.state_dict().keys()
    assert "encoder.filters_cat" in keys
    assert "decoder.filters" in keys
    assert "decoder.filters_cat" in keys


def test_encoder_fused_complex_analysis_matches_split_calls():
    torch.manual_seed(0)
    model = _lin()
    waveform = torch.randn(2, 4000)
    encoder = model.encoder

    padded = torch.nn.functional.pad(
        waveform.unsqueeze(1), (encoder.padding, encoder.padding), mode="reflect"
    )
    real = torch.nn.functional.conv1d(padded, encoder.filters.real, stride=encoder.stride)
    imag = torch.nn.functional.conv1d(padded, encoder.filters.imag, stride=encoder.stride)
    split = torch.stack([real, imag], dim=1)

    assert torch.equal(encoder(waveform), split)


def test_fast_decoder_accepts_half_precision_coefficients():
    model = _lin()
    spec = model.encode(torch.randn(1, 4000)).half()
    output = model.decode(spec, length=4000)
    assert output.dtype == spec.dtype
    assert torch.isfinite(output).all()


def test_exact_decode_is_exact_and_length_safe():
    torch.manual_seed(0)
    m = _lin(decoder_type="exact")
    x = torch.randn(1, 4000)
    xhat = m.decode(m.encode(x), length=4000)
    assert xhat.shape == (1, 4000)
    assert _snr(x, xhat) > 100



def test_exact_decoder_rejects_ill_conditioned_configuration():
    with pytest.raises(ValueError, match="use n_bins >= 256"):
        SincNet(fs=16000, fps=128, n_bins=128, scale="mel", component="complex",
                causal=False, decoder_type="exact")


def test_exact_decoder_rejects_real_component():
    with pytest.raises(ValueError, match="requires component='complex'"):
        SincNet(component="real", decoder_type="exact")


def test_exact_decoder_recommended_mel_configuration_is_exact():
    torch.manual_seed(0)
    model = SincNet(fs=16000, fps=128, n_bins=256, scale="mel", component="complex",
                    causal=False, decoder_type="exact")
    waveform = torch.randn(1, 4000)
    reconstructed = model.decode(model.encode(waveform), length=4000)
    assert _snr(waveform, reconstructed) > 100


def test_forward_preserves_input_length_exactly():
    torch.manual_seed(0)
    for dt in ("fast", "exact"):
        x = torch.randn(2, 4123)         # not a multiple of hop
        assert _lin(decoder_type=dt)(x).shape == (2, 4123)


def test_learnt_decode_runs_and_emits_frames_times_hop():
    m = _lin(decoder_type="learnt")
    spec = m.encode(torch.randn(1, 4000))
    assert m.decode(spec).shape[-1] == spec.shape[-1] * m.config.hop_length


def test_functional_and_module_exact_inverse_are_identical():
    torch.manual_seed(0)
    model = _lin()
    waveform = torch.randn(1, 4000)
    spec = model.encode(waveform)
    decoder = AnalyticDecoder1d(model.config, model.encoder, n_iter=16)
    functional = frame_pseudo_inverse(
        spec,
        model.encoder,
        length=4000,
        n_iter=16,
    )
    assert torch.equal(functional, decoder(spec, length=4000))


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
