import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .mulaw import create_mulaw



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
    """Half-hop STFT frontend sharing SincNet framing; Nyquist is omitted."""
    def __init__(
        self, fs:int=16000, fps:int=128, n_bins:int=128, overlap:int=2,
        q_bits:int=8, layout:str="channel", mulaw_kwargs:dict|None=None
    ):
        """ STFT transform parametrised by the two axes you actually control
            fs: int : sample rate of the input signal
            fps: int : frames per second of the spectrogram (fps = fs / hop)
            n_bins: int : number of frequency bins (the highest/Nyquist bin is dropped
                          so the bin axis is exactly n_bins, ideally a power of two)
            overlap: int : window overlap factor (2 -> 50%, 4 -> 75%). Changes window
                           duration/time-frequency tradeoff, not F or T; overlap*hop <= n_fft.
            q_bits: int : number of bits used by the spectrogram quantizer
            layout: str : channel -> (B,2,F,T) real (real,imag) | complex -> (B,F,T) complex
            mulaw_kwargs: dict : quantizer options including coordinate_system (default cartesian),
                pre_scaling (default True), q_mag, q_phi/q_trig, eps, dither, etc.; overrides q_bits
                Quantizers operate on the (B,2,F,T) real/imag channel representation.

            Frame centers are floor(hop/2) + t*hop, giving ceil(L/hop) frames.
            Normalized overlap-add keeps shifted boundary windows stable. Dropping
            Nyquist makes reconstruction approximate; default decode length is T*hop.
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
        self.frame_offset = self.hop_length // 2
        self.name = f"{fs}fs_{fps}fps_{n_bins}bins_stft"

        self.register_buffer("window", torch.hann_window(self.win_length))
        quantizer_options = dict(coordinate_system="cartesian", q_bits=q_bits, pre_scaling=True)
        quantizer_options.update(mulaw_kwargs or {})
        self.initialise_mulaw(**quantizer_options)

    def initialise_mulaw(self, coordinate_system: str = "polar", q_bits: int = 6, pre_scaling: bool = False, **kwargs) -> nn.Module:
        """ Configure the quantizer using the retained FFT bins' frequencies.
            Extra options go to the selected quantizer. For complex layout, convert
            to real/imag channels before passing a spectrogram to the quantizer.
        """
        centers = torch.arange(self.n_bins, device=self.window.device) * (self.fs / self.n_fft)
        self.mulaw = create_mulaw(
            coordinate_system, q_bits, pre_scaling,
            center_frequencies_hz=centers, frame_rate=self.fps, **kwargs,
        ).to(self.window.device)
        return self.mulaw

    def load_pretrained_weights(self, weights_folder:str|None=None, freeze:bool=True,
                                device:str="cpu", strict:bool=False, verbose:bool=False) -> "STFT":
        """Return self for SincNet API compatibility; STFT needs no pretrained weights."""
        return self

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
        """(B,L) -> complex (B,F,ceil(L/hop)), omitting Nyquist.

        Centers are floor(hop/2) + t*hop. Explicit reflection padding gives the
        original SincNet frame count while keeping both boundaries covered.
        """
        x = self._as_waveform(wav).to(self.window.dtype)
        left = self.n_fft // 2 - self.frame_offset
        right = self.n_fft // 2 + self.frame_offset - 1
        x = F.pad(x, (left, right), mode="reflect")
        spectrum = torch.stft(
            x, self.n_fft, self.hop_length, self.win_length,
            window=self.window, return_complex=True, center=False
        )
        return spectrum[:, :self.n_bins].contiguous()

    def _synthesis(self, spectrum:torch.Tensor, length:int|None=None) -> torch.Tensor:
        """complex (B,F,T) -> (B,L), using normalized overlap-add.

        Default L is T*hop; pass the original length to remove a partial final
        hop. Manual overlap-add permits the shifted crop without inverse-STFT
        checks on the unused zero-window endpoints of the padded signal.
        """
        T = spectrum.shape[-1]
        if length is None:
            length = T * self.hop_length
        if T < 1 or not 0 < length <= T * self.hop_length:
            raise ValueError("length must be positive and no greater than frames * hop_length")
        full = F.pad(spectrum, (0, 0, 0, 1))
        gap = self.n_fft - self.win_length
        window = F.pad(self.window, (gap // 2, gap - gap // 2)).to(spectrum.real)
        frames = torch.fft.irfft(full, n=self.n_fft, dim=1) * window[None, :, None]
        size = (T - 1) * self.hop_length + self.n_fft
        waveform = F.fold(frames, (1, size), (1, self.n_fft),
                          stride=(1, self.hop_length)).flatten(1)
        envelope = F.fold(window.square()[None, :, None].expand(1, self.n_fft, T),
                          (1, size), (1, self.n_fft),
                          stride=(1, self.hop_length)).flatten(1)
        left = self.n_fft // 2 - self.frame_offset
        waveform = waveform[:, left:left + length]
        envelope = envelope[:, left:left + length]
        if envelope.shape[-1] != length or torch.any(envelope <= 0):
            raise ValueError("Synthesis windows do not cover the requested waveform")
        return waveform / envelope

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

        length = magnitude.shape[-1] * self.hop_length
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check STFT frame-rate, bin-count and length contracts")
    parser.add_argument("--fs", type=int, default=16000)
    parser.add_argument("--fps", type=int, nargs="+", default=[64, 128, 250])
    parser.add_argument("--n-bins", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--overlap", type=int, nargs="+", choices=[2, 4], default=[2, 4])
    parser.add_argument("--seconds", type=int, default=1)
    args = parser.parse_args()
    if min(args.fs, args.seconds, *args.fps, *args.n_bins) <= 0:
        parser.error("fs, seconds, fps and n-bins must be positive")

    def check(condition:bool, message:str) -> None:
        if not condition:
            raise RuntimeError(message)

    torch.manual_seed(0)
    torch.set_num_threads(min(torch.get_num_threads(), 2))
    passed = skipped = cases = 0
    print(f"fs={args.fs} Hz; main input={args.seconds}s; B=2")
    print("Contract: H=fs/fps, F=n_bins, T=ceil(L/H), default decode length=T*H")
    print("Frame centers: floor(H/2) + t*H samples; layouts: channel and complex")
    print(f"{'fps':>5} {'bins':>5} {'overlap':>7} {'hop':>5} {'window':>6} {'(B,2,F,T)':>22}")
    with torch.no_grad():
        for fps in args.fps:
            for bins in args.n_bins:
                for overlap in args.overlap:
                    try:
                        model = STFT(fs=args.fs, fps=fps, n_bins=bins, overlap=overlap)
                    except (ValueError, AssertionError) as exc:
                        print(f"SKIP fps={fps}, bins={bins}, overlap={overlap}: {exc or 'n_bins must be a power of two'}")
                        skipped += 1
                        continue
                    hop = model.hop_length
                    check(hop * fps == args.fs, "Actual frame rate differs from requested fps")
                    main_length = args.fs * args.seconds
                    # Whole seconds, partial hops and the consumer's U-Net crop sizes.
                    lengths = sorted({main_length, main_length - 1, main_length + 1,
                                      *(hops * hop + delta for hops in (256, 320) for delta in (-1, 0, 1))})
                    main_shape = None
                    for length in lengths:
                        wav = torch.randn(2, length)
                        frames = (length + hop - 1) // hop
                        model.layout = "channel"
                        spec = model.encode(wav)
                        check(spec.shape == (2, 2, bins, frames), f"Unexpected shape: {spec.shape}")
                        check(torch.isfinite(spec).all().item(), "Non-finite spectrogram")
                        decoded = model.decode(spec)
                        check(decoded.shape == (2, frames * hop), "Wrong default decode length")
                        check(model.decode(spec, length=length).shape == wav.shape, "Explicit length was lost")
                        check(torch.isfinite(decoded).all().item(), "Non-finite reconstruction")
                        model.layout = "complex"
                        complex_spec = model.encode(wav)
                        check(complex_spec.shape == (2, bins, frames), "Wrong complex layout shape")
                        torch.testing.assert_close(complex_spec, torch.complex(spec[:, 0], spec[:, 1]))
                        torch.testing.assert_close(model.decode(complex_spec), decoded)
                        if length == main_length:
                            check(frames == fps * args.seconds, "Whole-second frame count differs from fps*seconds")
                            main_shape = tuple(spec.shape)
                        if length in (256 * hop, 320 * hop):
                            check(frames % 32 == 0, "Consumer frame count is not divisible by 32")
                        cases += 1
                    print(f"{fps:5d} {bins:5d} {overlap:7d} {hop:5d} {model.win_length:6d} {str(main_shape):>22}")
                    passed += 1
    if not passed:
        parser.error("No valid configurations; fps must divide fs and overlap*hop must fit in 2*n_bins")
    print(f"PASS: {passed} configurations, {cases} lengths in both layouts; {skipped} invalid combinations skipped.")
