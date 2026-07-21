import torch
import torch.nn as nn
import numpy as np

from .mulaw import MuLawQuant



def stft_params(fs:int, fps:int, n_bins:int, overlap:int=2) -> dict:
    """ Resolve the desired (fps, n_bins) into the three STFT knobs (hop, n_fft, win)
            hop_length = fs / fps           -> fps must divide fs for an integer hop
            n_fft      = 2 * n_bins         -> a power of two when n_bins is
            win_length = overlap * hop      -> overlap in {2,4} for Hann perfect reconstruction
        The two axes couple only through the window fitting inside the fft:
            n_bins >= overlap * fs / (2 * fps)
    """
    if fs % fps != 0:
        raise ValueError(f"fps={fps} must divide fs={fs} for an integer hop")
    hop = fs // fps
    n_fft = 2 * n_bins
    win = overlap * hop
    if win > n_fft:
        need = overlap * fs / (2 * fps)
        raise ValueError(f"n_bins too small: need n_bins >= overlap*fs/(2*fps) = {need:.0f}, got {n_bins}")
    return {"n_fft": n_fft, "win_length": win, "hop_length": hop}




class STFT(nn.Module):
    """Exactly-invertible STFT frontend sharing the SincNet encode/decode API"""
    def __init__(self, fs:int=16000, fps:int=128, n_bins:int=128, overlap:int=2,
                 q_bits:int=8, layout:str="channel"):
        """ STFT transform parametrised by the two axes you actually control
            fs: int : sample rate of the input signal
            fps: int : frames per second of the spectrogram (fps = fs / hop)
            n_bins: int : number of frequency bins (the highest/Nyquist bin is dropped
                          so the bin axis is exactly n_bins, ideally a power of two)
            overlap: int : window overlap factor (2 -> 50%, 4 -> 75%) controlling invertibility
            q_bits: int : number of bits used by the spectrogram quantizer
            layout: str : channel -> (B,2,F,T) real (real,imag) | complex -> (B,F,T) complex

            Unlike the decimated SincNet filterbank the STFT overlap-adds with exact COLA
            normalisation: decode(encode(x)) ~= x with no aliasing and no horizontal stripes.
        """
        super().__init__()
        assert layout in ("channel", "complex")
        #NOTE: check that the number of bins is a power of 2
        assert n_bins > 0 and (n_bins & (n_bins - 1)) == 0

        params = stft_params(fs=fs, fps=fps, n_bins=n_bins, overlap=overlap)
        self.fs = fs
        self.fps = fps
        self.n_bins = n_bins
        self.overlap = overlap
        self.layout = layout
        self.n_fft = params["n_fft"]
        self.win_length = params["win_length"]
        self.hop_length = params["hop_length"]
        self.name = f"{fs}fs_{fps}fps_{n_bins}bins_stft"

        self.register_buffer("window", torch.hann_window(self.win_length))
        self.mulaw = MuLawQuant(q_bits=q_bits)

    def _as_waveform(self, wav:torch.Tensor) -> torch.Tensor:
        """(L,) or (B,L) or (B,1,L) -> (B,L)"""
        if wav.ndim == 1:
            return wav.unsqueeze(0)
        if wav.ndim == 3 and wav.size(1) == 1:
            return wav.squeeze(1)
        if wav.ndim == 2:
            return wav
        raise ValueError("Expected mono waveform (L,), (B,L) or (B,1,L)")

    def _analysis(self, wav:torch.Tensor) -> torch.Tensor:
        """(B,L) -> complex (B,F,T), dropping only the highest (Nyquist) bin.

        torch.stft(center=True) yields 1 + L//hop frames for hop-aligned signals. The final
        frame is centered at sample L and mostly comes from reflection padding, but keeping it
        is important when decoding modified spectra: otherwise arbitrary phase near the right
        boundary can be amplified by the small inverse-STFT window envelope.
        """
        x = self._as_waveform(wav).to(self.window.dtype)
        spectrum = torch.stft(
            x, self.n_fft, self.hop_length, self.win_length,
            window=self.window, return_complex=True, center=True
        )
        return spectrum[:, :self.n_bins].contiguous()

    def _synthesis(self, spectrum:torch.Tensor, length:int|None=None) -> torch.Tensor:
        """complex (B,F,T) -> (B,L), restoring the dropped bin as zeros.

        With center=True, T frames naturally cover (T - 1) hops; default to that when no explicit
        length is given."""
        B, _, T = spectrum.shape
        if length is None:
            length = (T - 1) * self.hop_length
        full = spectrum.new_zeros((B, self.n_fft // 2 + 1, T))
        full[:, :self.n_bins] = spectrum
        return torch.istft(
            full, self.n_fft, self.hop_length, self.win_length,
            window=self.window, center=True, length=length
        )

    def _to_complex(self, spectrogram:torch.Tensor) -> torch.Tensor:
        """(B,2,F,T) real or (B,F,T) complex -> (B,F,T) complex"""
        if spectrogram.is_complex():
            return spectrogram
        return torch.complex(spectrogram[:, 0], spectrogram[:, 1])

    def _to_layout(self, spectrum:torch.Tensor) -> torch.Tensor:
        """complex (B,F,T) -> the configured layout (B,2,F,T) real or (B,F,T) complex"""
        if self.layout == "complex":
            return spectrum
        return torch.stack([spectrum.real, spectrum.imag], dim=1)

    @torch.no_grad()
    def magnitude(self, spectrogram:torch.Tensor) -> torch.Tensor:
        """Compute the magnitude spectrogram ~ (B,1,F,T) on the input signal"""
        if spectrogram.is_complex():
            return spectrogram.abs().unsqueeze(1)
        real, imag = spectrogram.chunk(2, dim=1)
        return torch.sqrt(real**2 + imag**2)

    @torch.no_grad()
    def griffin_lim(self, magnitude:torch.Tensor, initial_angle:torch.Tensor|None=None, n_iters:int=50) -> torch.Tensor:
        """ Reconstruct audio from magnitude spectrogram using the Griffin-Lim algorithm
            magnitude ~ (B,1,F,T) or (B,F,T) | initial_angle ~ (B,1,F,T)
            returns the refined spectrogram in the module layout ~ (B,2,F,T) or (B,F,T)
        """
        if magnitude.ndim == 3:
            magnitude = magnitude.unsqueeze(1)
        if initial_angle is None:
            angle = torch.rand_like(magnitude) * 2 * np.pi
        else:
            angle = initial_angle.unsqueeze(1) if initial_angle.ndim == 3 else initial_angle

        length = (magnitude.shape[-1] - 1) * self.hop_length
        for _ in range(n_iters+1):
            spectrum = (magnitude * torch.complex(torch.cos(angle), torch.sin(angle))).squeeze(1)
            forward = self._analysis(self._synthesis(spectrum, length=length))
            angle = torch.atan2(forward.imag, forward.real).unsqueeze(1)
        return self._to_layout(spectrum)

    @torch.no_grad()
    def refine_spectrogram_phase(self, spectrogram:torch.Tensor, n_iters:int=50) -> torch.Tensor:
        """Refine the phase of the input spectrogram ~(B,2,F,T) or (B,F,T) using the Griffin-Lim algorithm"""
        magnitude = self.magnitude(spectrogram)
        spectrum = self._to_complex(spectrogram)
        initial_angle = torch.atan2(spectrum.imag, spectrum.real).unsqueeze(1)
        return self.griffin_lim(magnitude, initial_angle=initial_angle, n_iters=n_iters)

    def encode(self, x:torch.Tensor) -> torch.Tensor:
        """Compute the STFT spectrogram ~ (B,C,F,T) (C=2) or (B,F,T) for the complex layout"""
        return self._to_layout(self._analysis(x))

    def decode(self, x:torch.Tensor, length:int|None=None) -> torch.Tensor:
        """Reconstruct audio from the STFT spectrogram ~ (B,L)"""
        return self._synthesis(self._to_complex(x), length=length)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        length = self._as_waveform(x).shape[-1]
        x = self.encode(x)
        x = self.decode(x, length=length)
        return x
