"""Differentiable conjugate-gradient inversion for SincNet analysis frames.

This module owns the single exact inversion function used by the decoder
module defined in ``sincnet.model``.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn


class FrameInverseCGFunction(torch.autograd.Function):
    """Implicitly differentiable CG pseudo-inverse of a linear encoder.

    Solves ``(A.T A + reg I)x = A.T spec``. The CG loop is not recorded;
    backward solves the same symmetric normal equations. The exact encoder
    adjoint is built once per solve with ``torch.func.vjp`` and reused across
    all iterations.
    """

    @staticmethod
    def _cg_solve(
        normal_op: Callable[[torch.Tensor], torch.Tensor],
        b: torch.Tensor,
        n_iter: int = 64,
        tol: float = 1e-9,
        eps: float = 1e-30,
    ) -> torch.Tensor:
        """Solve the batched SPD system ``normal_op(x) = b`` from zero."""

        def bdot(a: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
            return (a * c).flatten(1).sum(dim=1, keepdim=True)

        x = torch.zeros_like(b)
        residual = b.clone()
        direction = residual.clone()
        residual_norm = bdot(residual, residual)
        for _ in range(n_iter):
            normal_direction = normal_op(direction)
            alpha = residual_norm / bdot(direction, normal_direction).clamp_min(eps)
            x = x + alpha * direction
            residual = residual - alpha * normal_direction
            next_norm = bdot(residual, residual)
            if float(next_norm.sqrt().max()) < tol:
                break
            direction = residual + (
                next_norm / residual_norm.clamp_min(eps)
            ) * direction
            residual_norm = next_norm
        return x

    @staticmethod
    def _adjoint_op(
        encoder: nn.Module,
        batch_size: int,
        length: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        """Build the exact encoder adjoint once and return its reusable VJP."""
        x0 = torch.zeros(batch_size, length, device=device, dtype=dtype)
        with torch.enable_grad():
            _, vjp_fn = torch.func.vjp(encoder, x0)

        def adjoint(residual: torch.Tensor) -> torch.Tensor:
            with torch.enable_grad():
                gradient, = vjp_fn(residual)
            return gradient.detach()

        return adjoint

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        spec: torch.Tensor,
        encoder: nn.Module,
        length: int,
        n_iter: int,
        tol: float,
        reg: float,
    ) -> torch.Tensor:
        adjoint = FrameInverseCGFunction._adjoint_op(
            encoder,
            spec.shape[0],
            length,
            spec.device,
            spec.dtype,
        )
        normal_op: Callable[[torch.Tensor], torch.Tensor] = (
            (lambda value: adjoint(encoder(value)) + reg * value)
            if reg
            else (lambda value: adjoint(encoder(value)))
        )
        ctx.encoder = encoder
        ctx.length = length
        ctx.n_iter = n_iter
        ctx.tol = tol
        ctx.reg = reg
        with torch.no_grad():
            return FrameInverseCGFunction._cg_solve(
                normal_op,
                adjoint(spec),
                n_iter=n_iter,
                tol=tol,
            )

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_x: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None, None]:
        encoder: nn.Module = ctx.encoder
        length: int = ctx.length
        reg: float = ctx.reg
        adjoint = FrameInverseCGFunction._adjoint_op(
            encoder,
            grad_x.shape[0],
            length,
            grad_x.device,
            grad_x.dtype,
        )
        normal_op: Callable[[torch.Tensor], torch.Tensor] = (
            (lambda value: adjoint(encoder(value)) + reg * value)
            if reg
            else (lambda value: adjoint(encoder(value)))
        )
        with torch.no_grad():
            solution = FrameInverseCGFunction._cg_solve(
                normal_op,
                grad_x.contiguous(),
                n_iter=ctx.n_iter,
                tol=ctx.tol,
            )
            grad_spec = encoder(solution)
        return grad_spec, None, None, None, None, None


def frame_pseudo_inverse(
    spec: torch.Tensor,
    encoder: nn.Module,
    length: int,
    n_iter: int = 64,
    tol: float = 1e-9,
    reg: float = 0.0,
) -> torch.Tensor:
    """Recover exactly ``length`` waveform samples from analysis coefficients.

    ``AnalyticDecoder1d`` delegates to this function, so functional and module
    calls share the same custom-autograd inverter and zero-initialized solve.
    """
    return FrameInverseCGFunction.apply(
        spec,
        encoder,
        length,
        n_iter,
        tol,
        reg,
    )
