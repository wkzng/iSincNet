import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiResolutionSTFTLoss(nn.Module):
    """ Multi-resolution STFT loss (Parallel WaveGAN / iSTFTNet style).

        Sum over several (n_fft, hop, win) resolutions of:
            - spectral convergence:  ||S - S_hat||_F / ||S||_F
            - log-magnitude L1:      || log S - log S_hat ||_1
        Both terms are band-normalised, so high frequencies are weighted as much as low ones —
        unlike an energy-weighted waveform/STFT L1 loss, which lets the model leave the
        high-frequency band smeared (the source of the frame-rate horizontal stripes).

        References: Parallel WaveGAN [Arxiv](https://arxiv.org/abs/1910.11480),
        iSTFTNet [Arxiv](https://arxiv.org/abs/2203.02395)
    """
    def __init__(
            self, 
            fft_sizes:tuple=(512, 1024, 2048), 
            hop_sizes:tuple=(128, 256, 512),
            win_lengths:tuple=(512, 1024, 2048), 
            eps:float=1e-7
        ):
        super().__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        self.resolutions = list(zip(fft_sizes, hop_sizes, win_lengths))
        self.eps = eps
        for i, (_, _, win) in enumerate(self.resolutions):
            self.register_buffer(f"window_{i}", torch.hann_window(win))

    def _magnitude(self, x:torch.Tensor, n_fft:int, hop:int, win:int, window:torch.Tensor) -> torch.Tensor:
        spectrum = torch.stft(x, n_fft, hop, win, window=window, return_complex=True, center=True)
        return spectrum.abs().clamp_min(self.eps)

    def forward(self, y_hat:torch.Tensor, y:torch.Tensor) -> torch.Tensor:
        """ y_hat, y ~ (B, L) or (B, 1, L). Returns the averaged multi-resolution STFT loss."""
        y_hat = y_hat.reshape(y_hat.shape[0], -1)
        y = y.reshape(y.shape[0], -1)
        n = min(y_hat.shape[-1], y.shape[-1])
        y_hat, y = y_hat[..., :n], y[..., :n]

        total = 0.0
        for i, (n_fft, hop, win) in enumerate(self.resolutions):
            window = getattr(self, f"window_{i}")
            S = self._magnitude(y, n_fft, hop, win, window)
            S_hat = self._magnitude(y_hat, n_fft, hop, win, window)
            spectral_convergence = torch.linalg.norm(S - S_hat) / (torch.linalg.norm(S) + self.eps)
            log_magnitude = F.l1_loss(torch.log(S_hat), torch.log(S))
            total = total + spectral_convergence + log_magnitude
        return total / len(self.resolutions)
