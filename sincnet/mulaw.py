import math

import torch
import torch.nn as nn


def mu_law_compand(x: torch.Tensor, q_bits: int) -> torch.Tensor:
    """μ-law companding (doc: https://en.wikipedia.org/wiki/%CE%9C-law_algorithm)
    Supports:
        C=1 → real
        C=2 → complex (real, imag)
    Edge case for complex input:
        scale(x) = log(1 + μ * |x|) / (log(1 + μ) * |x|)
        scale(0) = μ / log(1 + μ) (limit value as x → 0)
    """
    mu = 2**q_bits - 1
    log_mu = math.log1p(mu)
    _, channels, _, _ = x.shape

    if channels == 1:
        return torch.sign(x) * torch.log1p(mu * torch.abs(x)) / log_mu
    elif channels == 2:
        real = x[:, 0]
        imag = x[:, 1]
        mag = torch.sqrt(real**2 + imag**2)
        scale = torch.zeros_like(mag)

        # separate zero and non-zero magnitudes to avoid division by zero
        zero_mask = mag == 0
        nonzero_mask = ~zero_mask

        # case A > 0
        mag_nz = mag[nonzero_mask]
        scale[nonzero_mask] = torch.log1p(mu * mag_nz) / (mag_nz * log_mu)

        # case A == 0 (limit value)
        scale[zero_mask] = mu / log_mu

        # apply scaling
        real_out = scale * real
        imag_out = scale * imag
        return torch.stack([real_out, imag_out], dim=1)
    else:
        raise ValueError(f"Unsupported number of channels: {channels}")


def mu_law_expand(x: torch.Tensor, q_bits: int) -> torch.Tensor:
    """Inverse μ-law companding (doc: https://en.wikipedia.org/wiki/%CE%9C-law_algorithm)
    Supports:
        C=1 → real
        C=2 → complex (real, imag)
    Edge case for complex input:
        scale(x) = (exp(log(1 + μ) * |x|) - 1) / (μ * |x|)
        scale(0) = log(1 + μ) / μ (limit value as x → 0)
    """
    mu = 2**q_bits - 1
    log_mu = math.log1p(mu)
    _, channels, _, _ = x.shape

    if channels == 1:
        return torch.sign(x) * (1.0 / mu) * torch.expm1(torch.abs(x) * log_mu)
    elif channels == 2:
        real = x[:, 0]
        imag = x[:, 1]
        mag = torch.sqrt(real**2 + imag**2)
        scale = torch.zeros_like(mag)

        # separate zero and non-zero magnitudes
        zero_mask = mag == 0
        nonzero_mask = ~zero_mask

        # case A > 0
        mag_nz = mag[nonzero_mask]
        scale[nonzero_mask] = torch.expm1(mag_nz * log_mu) / (mu * mag_nz)

        # case A == 0 (limit value)
        scale[zero_mask] = log_mu / mu

        # apply scaling
        real_out = scale * real
        imag_out = scale * imag
        return torch.stack([real_out, imag_out], dim=1)
    else:
        raise ValueError(f"Unsupported number of channels: {channels}")


def quantize_unit(x: torch.Tensor, vocab_size: int, add_noise: bool = False) -> torch.Tensor:
    """Map values in [0, 1] to integer bins in {0, ..., vocab_size - 1}."""
    y = x * (vocab_size - 1)
    if add_noise:
        y = y + torch.rand_like(y) - 0.5
    return torch.clamp(torch.round(y).long(), 0, vocab_size - 1)


def dequantize_unit(x: torch.Tensor, vocab_size: int) -> torch.Tensor:
    """Map integer bins in {0, ..., vocab_size - 1} back to [0, 1]."""
    if vocab_size == 1:
        return torch.zeros_like(x, dtype=torch.float)
    return x.float() / (vocab_size - 1)


def get_scale(magnitude: torch.Tensor, eps: float = 1e-8, pre_scaling: bool = True) -> torch.Tensor:
    """Get per-sample scale from a magnitude tensor shaped (B, F, T)."""
    if pre_scaling:
        scale = eps + torch.amax(magnitude, dim=[-2, -1], keepdim=True).unsqueeze(1)
    else:
        scale = 1.0
    return scale


class MuLawQuant(nn.Module):
    def __init__(
        self,
        q_bits: int = 8,
        eps: float = 1e-8,
        dither: bool = False,
        pre_scaling: bool = True,
    ):
        """Quantizer module that applies μ-law companding and quantization to the input tensor.
        Args:
            q_bits: Number of bits for quantization (e.g., 8 for 256 levels)
            eps: Small constant to avoid division by zero during scaling
            dither: Whether to add uniform noise in [-0.5, 0.5] before quantization to reduce bias
            pre_scaling: Whether to apply dynamic range scaling before companding based on the max magnitude in the input
        """
        super().__init__()
        self.q_bits = q_bits
        self.vocab_size = 2 ** q_bits
        self.eps = eps
        self.dither = dither
        self.pre_scaling = pre_scaling

    def mu_law_quantize(self, x: torch.Tensor, add_noise: bool = False) -> torch.Tensor:
        """transform tensor with values in [-1,1] into a tensor with values in [0, 2^q_bits-1]"""
        return quantize_unit((x + 1) / 2.0, self.vocab_size, add_noise=add_noise)

    def mu_law_dequantize(self, x: torch.Tensor) -> torch.Tensor:
        """transform tensor with values in [0, 2^q_bits-1] back into [-1,1]"""
        return 2 * dequantize_unit(x, self.vocab_size) - 1.0

    def compand(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ x ~ (B, C, F, T)"""
        magnitude = (x**2).sum(dim=1).sqrt()
        scale = get_scale(magnitude, eps=self.eps, pre_scaling=self.pre_scaling)
        x = mu_law_compand(x / scale, q_bits=self.q_bits)
        return x, scale

    def expand(self, x: torch.Tensor, scale: torch.Tensor | int = 1) -> torch.Tensor:
        return mu_law_expand(x, q_bits=self.q_bits) * scale

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ x ~ (B, C, F, T)"""
        x, scale = self.compand(x)
        x = self.mu_law_quantize(x, add_noise=self.dither)
        return x, scale

    def dequantize(self, x: torch.Tensor, scale: torch.Tensor | int = 1) -> torch.Tensor:
        """ x ~ (B, C, F, T)"""
        x = self.mu_law_dequantize(x)
        x = self.expand(x, scale=scale)
        return x


class PolarMuLawQuant(nn.Module):
    """
    Polar quantizer for complex spectrograms stored as (B, 2, F, T) float tensors
    where channel 0 = real, channel 1 = imaginary.

    Magnitude: mu-law companded, quantized to q_mag bits.
    Phase: uniform, quantized to q_phi bits.
    q_bits sets both defaults when q_mag/q_phi are not specified.

    compand() returns (polar, scale)
        polar ~ (B, 2, F, T) float, with channel 0 = magnitude and channel 1 = phase
        scale ~ (B, 1, 1, 1) float

    quantize() returns (tokens, scale)
        tokens ~ (B, 2, F, T) int64, with channel 0 = magnitude and channel 1 = phase
        scale  ~ (B, 1, 1, 1) float

    expand() returns x_hat ~ (B, 2, F, T) float
    dequantize() returns x_hat ~ (B, 2, F, T) float
    """

    def __init__(
        self,
        q_mag: int | None = None,
        q_phi: int | None = None,
        q_bits: int = 8,
        eps: float = 1e-8,
        dither: bool = False,
        pre_scaling: bool = True,
    ):
        super().__init__()
        q_mag = q_bits if q_mag is None else q_mag
        q_phi = q_bits if q_phi is None else q_phi

        self.q_bits = q_bits
        self.q_mag = q_mag
        self.q_phi = q_phi
        self.mag_vocab_size = 2 ** q_mag
        self.phase_vocab_size = 2 ** q_phi
        self.mu = float(self.mag_vocab_size - 1)
        self.log_mu = math.log1p(self.mu)
        self.eps = eps
        self.dither = dither
        self.pre_scaling = pre_scaling

        # Backwards-compatible aliases used by older experiments.
        self.n_mag = self.mag_vocab_size
        self.n_phi = self.phase_vocab_size

    def compand_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Map magnitudes in [0, 1] to mu-law companded magnitudes in [0, 1]."""
        return torch.log1p(self.mu * x) / self.log_mu

    def expand_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        """Invert mu-law companded magnitudes back to [0, 1]."""
        return torch.expm1(x * self.log_mu) / self.mu

    @staticmethod
    def _validate_complex_input(x: torch.Tensor) -> None:
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError(f"Expected (B, 2, F, T), got {tuple(x.shape)}")

    @staticmethod
    def _split_channels(x: torch.Tensor, name: str) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4 or x.shape[1] != 2:
            raise ValueError(f"Expected {name} shaped (B, 2, F, T), got {tuple(x.shape)}")
        return x[:, 0], x[:, 1]

    @staticmethod
    def _scale_for_magnitude(scale: torch.Tensor | float) -> torch.Tensor | float:
        if isinstance(scale, torch.Tensor) and scale.ndim == 4:
            return scale.squeeze(1)
        return scale

    def compand(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a complex spectrogram to continuous polar mu-law values."""
        self._validate_complex_input(x)
        real = x[:, 0]
        imag = x[:, 1]

        #compand the spectrogram magnitude into log-range
        magnitude = torch.sqrt(real**2 + imag**2)
        scale = get_scale(magnitude, eps=self.eps, pre_scaling=self.pre_scaling)
        magnitude = self.compand_magnitude(magnitude / self._scale_for_magnitude(scale))

        #rescale the phase range
        phase = torch.atan2(imag, real)
        phase = (phase + torch.pi) / (2 * torch.pi)

        return torch.stack([magnitude, phase], dim=1), scale

    def expand(self, x: torch.Tensor, scale: torch.Tensor | float = 1.0) -> torch.Tensor:
        """Reconstruct a complex spectrogram from continuous polar mu-law values."""
        magnitude, phase = self._split_channels(x, "polar tensor")

        magnitude = self.expand_magnitude(magnitude) * self._scale_for_magnitude(scale)
        phase = phase * (2 * torch.pi) - torch.pi

        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        return torch.stack([real, imag], dim=1)

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize a complex spectrogram shaped (B, 2, F, T)."""
        x, scale = self.compand(x)
        magnitude, phase = self._split_channels(x, "polar tensor")
        mag_tokens = quantize_unit(magnitude, self.mag_vocab_size, add_noise=self.dither)
        phi_tokens = quantize_unit(phase, self.phase_vocab_size, add_noise=self.dither)
        return torch.stack([mag_tokens, phi_tokens], dim=1), scale

    def dequantize(self, x: torch.Tensor, scale: torch.Tensor | float = 1.0) -> torch.Tensor:
        """Reconstruct a complex spectrogram from stacked magnitude/phase tokens."""
        mag_tokens, phi_tokens = self._split_channels(x, "token tensor")
        magnitude = dequantize_unit(mag_tokens, self.mag_vocab_size)
        phase = dequantize_unit(phi_tokens, self.phase_vocab_size)
        return self.expand(torch.stack([magnitude, phase], dim=1), scale=scale)


class PredictivePolarQuant(PolarMuLawQuant):
    """
    Polar quantizer with predictive phase tokens.

    Magnitude follows PolarMuLawQuant. Phase stores wrapped phase increments over
    time and reserves token 0 as SILENCE for low-magnitude bins.
    """

    SILENCE_TOKEN = 0

    def __init__(
        self,
        q_mag: int | None = None,
        q_phi: int | None = None,
        q_bits: int = 6,
        eps: float = 1e-8,
        dither: bool = False,
        pre_scaling: bool = True,
        mag_silence_threshold: int = 4,
    ):
        super().__init__(
            q_mag=q_mag,
            q_phi=q_phi,
            q_bits=q_bits,
            eps=eps,
            dither=dither,
            pre_scaling=pre_scaling,
        )
        if self.phase_vocab_size <= 1:
            raise ValueError("PredictivePolarQuant requires at least two phase tokens")
        if not 0 <= mag_silence_threshold <= self.mag_vocab_size:
            raise ValueError("mag_silence_threshold must be in [0, mag_vocab_size]")
        self.mag_silence_threshold = mag_silence_threshold
        self.phase_levels = self.phase_vocab_size - 1

    @staticmethod
    def _wrap_phase(x: torch.Tensor) -> torch.Tensor:
        return torch.remainder(x + torch.pi, 2 * torch.pi) - torch.pi

    def _phase_to_delta(self, phase: torch.Tensor) -> torch.Tensor:
        delta = torch.empty_like(phase)
        delta[..., 0] = phase[..., 0]
        delta[..., 1:] = phase[..., 1:] - phase[..., :-1]
        return self._wrap_phase(delta)

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Quantize complex spectrograms using magnitude and predictive phase tokens."""
        self._validate_complex_input(x)
        real = x[:, 0]
        imag = x[:, 1]

        magnitude = torch.sqrt(real**2 + imag**2)
        scale = get_scale(magnitude, eps=self.eps, pre_scaling=self.pre_scaling)
        magnitude = self.compand_magnitude(magnitude / self._scale_for_magnitude(scale))
        mag_tokens = quantize_unit(magnitude, self.mag_vocab_size, add_noise=self.dither)

        phase = torch.atan2(imag, real)
        delta_phase = self._phase_to_delta(phase)
        phase_unit = (delta_phase + torch.pi) / (2 * torch.pi)
        phi_tokens_raw = quantize_unit(
            phase_unit,
            self.phase_levels,
            add_noise=self.dither,
        ) + 1
        phi_tokens = torch.where(
            mag_tokens < self.mag_silence_threshold,
            torch.zeros_like(phi_tokens_raw),
            phi_tokens_raw,
        )

        return torch.stack([mag_tokens, phi_tokens], dim=1), scale

    def dequantize(self, x: torch.Tensor, scale: torch.Tensor | float = 1.0) -> torch.Tensor:
        """Reconstruct complex spectrograms from magnitude and predictive phase tokens."""
        mag_tokens, phi_tokens = self._split_channels(x, "token tensor")

        magnitude = dequantize_unit(mag_tokens, self.mag_vocab_size)
        magnitude = self.expand_magnitude(magnitude) * self._scale_for_magnitude(scale)

        silent = phi_tokens == self.SILENCE_TOKEN
        phi_tokens_unshifted = torch.clamp(phi_tokens - 1, min=0)
        phase_unit = dequantize_unit(phi_tokens_unshifted, self.phase_levels)
        delta_phase = phase_unit * (2 * torch.pi) - torch.pi
        delta_phase = torch.where(silent, torch.zeros_like(delta_phase), delta_phase)
        phase = torch.cumsum(delta_phase, dim=-1)

        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        return torch.stack([real, imag], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, scale = self.quantize(x)
        return self.dequantize(tokens, scale)


class DemodulatedPolarQuant(PolarMuLawQuant):
    """
    Polar quantizer that stores mu-law companded demodulated phase.

    Phase is represented as the wrapped residual after removing the deterministic
    carrier rotation for each frequency bin. Unlike PredictivePolarQuant this is
    absolute per frame and does not accumulate reconstruction error over time.
    """

    def __init__(
        self,
        center_frequencies_hz: torch.Tensor,
        frame_rate: float,
        q_mag: int | None = None,
        q_phi: int | None = None,
        q_bits: int = 6,
        eps: float = 1e-8,
        dither: bool = False,
        pre_scaling: bool = True,
    ):
        super().__init__(
            q_mag=q_mag,
            q_phi=q_phi,
            q_bits=q_bits,
            eps=eps,
            dither=dither,
            pre_scaling=pre_scaling,
        )
        if self.phase_vocab_size <= 1:
            raise ValueError("DemodulatedPolarQuant requires at least two phase tokens")
        center_frequencies_hz = torch.as_tensor(center_frequencies_hz, dtype=torch.float)
        if center_frequencies_hz.ndim != 1:
            raise ValueError("center_frequencies_hz must be a 1D tensor or sequence")
        if frame_rate <= 0:
            raise ValueError("frame_rate must be positive")
        self.register_buffer("center_frequencies_hz", center_frequencies_hz)
        self.frame_rate = float(frame_rate)
        self.phase_mu = float(self.phase_vocab_size - 1)
        self.log_phase_mu = math.log1p(self.phase_mu)

    @staticmethod
    def _wrap_phase(x: torch.Tensor) -> torch.Tensor:
        return torch.remainder(x + torch.pi, 2 * torch.pi) - torch.pi

    def _carrier_phase(self, n_frames: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        centers = self.center_frequencies_hz.to(device=device, dtype=dtype)
        frame_idx = torch.arange(n_frames, device=device, dtype=dtype)
        carrier = (2 * torch.pi * centers[:, None] / self.frame_rate) * frame_idx[None]
        return carrier.unsqueeze(0)

    def compand_phase(self, x: torch.Tensor) -> torch.Tensor:
        """Apply mu-law companding to phase residuals in [-pi, pi]."""
        phase_unit = x / torch.pi
        phase_unit = torch.sign(phase_unit) * torch.log1p(self.phase_mu * torch.abs(phase_unit)) / self.log_phase_mu
        return phase_unit * torch.pi

    def expand_phase(self, x: torch.Tensor) -> torch.Tensor:
        """Invert mu-law phase companding back to residuals in [-pi, pi]."""
        phase_unit = x / torch.pi
        phase_unit = torch.sign(phase_unit) * torch.expm1(torch.abs(phase_unit) * self.log_phase_mu) / self.phase_mu
        return phase_unit * torch.pi

    def _demodulate_phase(self, phase: torch.Tensor) -> torch.Tensor:
        if phase.shape[-2] != self.center_frequencies_hz.numel():
            raise ValueError(
                f"Expected {self.center_frequencies_hz.numel()} frequency bins, got {phase.shape[-2]}"
            )
        carrier = self._carrier_phase(phase.shape[-1], dtype=phase.dtype, device=phase.device)
        return self._wrap_phase(phase - carrier)

    def _remodulate_phase(self, residual: torch.Tensor) -> torch.Tensor:
        if residual.shape[-2] != self.center_frequencies_hz.numel():
            raise ValueError(
                f"Expected {self.center_frequencies_hz.numel()} frequency bins, got {residual.shape[-2]}"
            )
        carrier = self._carrier_phase(residual.shape[-1], dtype=residual.dtype, device=residual.device)
        return residual + carrier

    def compand(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a complex spectrogram to magnitude and demodulated phase values."""
        self._validate_complex_input(x)
        real = x[:, 0]
        imag = x[:, 1]

        magnitude = torch.sqrt(real**2 + imag**2)
        scale = get_scale(magnitude, eps=self.eps, pre_scaling=self.pre_scaling)
        magnitude = self.compand_magnitude(magnitude / self._scale_for_magnitude(scale))

        phase = torch.atan2(imag, real)
        phase = self._demodulate_phase(phase)
        phase = self.compand_phase(phase)
        phase = (phase + torch.pi) / (2 * torch.pi)

        return torch.stack([magnitude, phase], dim=1), scale

    def expand(self, x: torch.Tensor, scale: torch.Tensor | float = 1.0) -> torch.Tensor:
        """Reconstruct a complex spectrogram from demodulated polar values."""
        magnitude, phase = self._split_channels(x, "demodulated polar tensor")

        magnitude = self.expand_magnitude(magnitude) * self._scale_for_magnitude(scale)
        phase = phase * (2 * torch.pi) - torch.pi
        phase = self.expand_phase(phase)
        phase = self._remodulate_phase(phase)

        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        return torch.stack([real, imag], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, scale = self.quantize(x)
        return self.dequantize(tokens, scale)
