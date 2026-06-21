import torch

import pytest

from sincnet import SincNet, frame_pseudo_inverse
from sincnet.model import FastAnalyticDecoder1d, AnalyticDecoder1d


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


def test_exact_decoder_iteration_configuration():
    assert _lin(decoder_type="exact").decoder.n_iter == 64
    assert _lin(decoder_type="exact", cg_iters=16).decoder.n_iter == 16


@pytest.mark.parametrize("cg_iters", [0, -1])
def test_cg_iters_must_be_positive(cg_iters):
    with pytest.raises(ValueError, match="cg_iters must be positive"):
        _lin(cg_iters=cg_iters)


def test_unknown_decoder_type_raises():
    with pytest.raises(AssertionError):
        _lin(decoder_type="learnt")


def test_decoders_have_no_trainable_weights():
    assert sum(p.numel() for p in _lin(decoder_type="fast").decoder.parameters()) == 0
    assert sum(p.numel() for p in _lin(decoder_type="exact").decoder.parameters()) == 0


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
    assert "decoder.filters" in keys
    assert "decoder.filters_cat" in keys


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


def test_forward_preserves_input_length_exactly():
    torch.manual_seed(0)
    for dt in ("fast", "exact"):
        x = torch.randn(2, 4123)         # not a multiple of hop
        assert _lin(decoder_type=dt)(x).shape == (2, 4123)


# ---- functional exact inverse ----

@pytest.mark.parametrize("n_iter", [16, 64])
def test_functional_and_module_exact_inverse_are_identical(n_iter):
    torch.manual_seed(0)
    model = _lin()
    waveform = torch.randn(1, 4000)
    spec = model.encode(waveform)
    decoder = AnalyticDecoder1d(model.config, model.encoder, n_iter=n_iter)
    functional = frame_pseudo_inverse(
        spec,
        model.encoder,
        length=4000,
        n_iter=n_iter,
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


def test_sinc_envelope_defaults_on_and_can_be_disabled():
    default = _lin()
    disabled = _lin(apply_sinc_envelope=False)
    assert default.config.apply_sinc_envelope is True
    assert disabled.config.apply_sinc_envelope is False
    assert not torch.equal(default.encoder.filters, disabled.encoder.filters)
