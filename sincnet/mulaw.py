
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def mu_law_compand( x: torch.Tensor, q_bits: int) -> torch.Tensor:
    """ μ-law companding (doc: https://en.wikipedia.org/wiki/%CE%9C-law_algorithm)
        Supports:
            C=1 → real
            C=2 → complex (real, imag)
        Edge case for complex input:
            scale(x) = log(1 + μ * |x|) / (log(1 + μ) * |x|)
            scale(0) = μ / log(1 + μ) (limit value as x → 0)
    """
    mu = 2 ** q_bits - 1
    log_mu = np.log(1 + mu)
    B, C, F, T = x.shape

    if C == 1:
        return torch.sign(x) * torch.log1p(mu * torch.abs(x)) / log_mu
    elif C == 2:
        real = x[:, 0]
        imag = x[:, 1]
        mag = torch.sqrt(real**2 + imag**2)
        scale = torch.zeros_like(mag)

        # separate zero and non-zero magnitudes to avoid division by zero
        zero_mask = (mag == 0)
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
        raise ValueError(f"Unsupported number of channels: {C}")


def mu_law_expand(x: torch.Tensor, q_bits: int) -> torch.Tensor:
    """ Inverse μ-law companding (doc: https://en.wikipedia.org/wiki/%CE%9C-law_algorithm)
        Supports:
            C=1 → real
            C=2 → complex (real, imag)
        Edge case for complex input:
            scale(x) = (exp(log(1 + μ) * |x|) - 1) / (μ * |x|)
            scale(0) = log(1 + μ) / μ (limit value as x → 0)
    """
    mu = 2 ** q_bits - 1
    log_mu = np.log(1 + mu)
    B, C, F, T = x.shape

    if C == 1:
        return torch.sign(x) * (1.0 / mu) * torch.expm1(torch.abs(x) * log_mu)
    elif C == 2:
        real = x[:, 0]
        imag = x[:, 1]
        mag = torch.sqrt(real**2 + imag**2)
        scale = torch.zeros_like(mag)

        # separate zero and non-zero magnitudes
        zero_mask = (mag == 0)
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
        raise ValueError(f"Unsupported number of channels: {C}")


def mu_law_quantize(x:torch.Tensor, q_bits:int, add_noise:bool=False) -> torch.Tensor:
    """transform tensor with values in [-1,1] into a tensor with values in [0, 2^q_bits-1]"""
    mu = 2**q_bits - 1
    y = (x + 1) / 2.0
    y = mu * y

    # Optionaly add uniform dither in [-0.5, 0.5] to reduce quantization bias
    if add_noise:
        noise = torch.rand_like(y) - 0.5
        y = y + noise

    y = torch.round(y).to(torch.long)
    y = torch.clamp(y, 0, mu)
    return y


def mu_law_dequantize(x:torch.Tensor, q_bits:int) -> torch.Tensor:
    """transform tensor with values in [0, 2^q_bits-1] back into [-1,1]"""
    mu = 2**q_bits - 1
    y = x.float() / mu
    y = 2 * y - 1.0
    return y


class MuLawQuant(nn.Module):
    def __init__(self, q_bits:int, eps:float=1e-8, dither:bool=False, pre_scaling:bool=True):
        """Quantizer module that applies μ-law companding and quantization to the input tensor.
        Args:
            q_bits: Number of bits for quantization (e.g., 8 for 256 levels)
            eps: Small constant to avoid division by zero during scaling
            dither: Whether to add uniform noise in [-0.5, 0.5] before quantization to reduce bias
            pre_scaling: Whether to apply dynamic range scaling before companding based on the max magnitude in the input
        """
        super().__init__()
        self.q_bits = q_bits
        self.vocab_size = 2**q_bits
        self.eps = eps
        self.dither = dither
        self.pre_scaling = pre_scaling

    def quantize(self, x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ x ~ (B, C, F, T)"""
        if self.pre_scaling:
            magnitude = (x**2).sum(dim=1).sqrt()
            scale = self.eps + torch.amax(magnitude, dim=[-2, -1], keepdim=True).unsqueeze(1)
            #scale = self.eps + torch.mean(magnitude, dim=1, keepdim=True).unsqueeze(1)
        else:
            scale = 1.0
        x = mu_law_compand(x / scale, q_bits=self.q_bits)
        x = mu_law_quantize(x, q_bits=self.q_bits, add_noise=self.dither)
        return x, scale
        
    def dequantize(self, x:torch.Tensor, scale:torch.Tensor|int=1) -> torch.Tensor:
        """ x ~ (B, C, F, T)"""
        x = mu_law_dequantize(x, q_bits=self.q_bits)
        x = mu_law_expand(x, q_bits=self.q_bits)
        return x * scale