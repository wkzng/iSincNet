import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import librosa
import warnings
import numpy as np
from dataclasses import dataclass, asdict
from .cgdecoder import frame_pseudo_inverse
from .mulaw import MuLawQuant, PolarMuLawQuant



@dataclass
class ModelArgs:
    """ Geometry & invertibility (filterbank framing -- NOT an STFT, so there is no n_fft and no
        `n_bins = n_fft/2 + 1` coupling: the number of filters N and the kernel length L are
        INDEPENDENT design choices).

            Frame rate fixes the hop:            fs = FPS * H               (H = hop_length)
            Kernel spans `coverage` hops:        L  = coverage * H + 1       (kernel_size; coverage=4)
            Each complex bin = 2 real channels   (cos + sin); component="real"/"imag" -> factor 1.

        Per frame the analysis maps an L-sample window -> factor*N real numbers. The count
        ``factor*N >= H`` is necessary for global invertibility, but it is not sufficient: stable
        inversion is controlled by the filterbank frame bounds. Strongly warped scales can leave
        near-null waveform directions even when their coefficient count exceeds the hop.

        Practically pick FPS dividing fs:
            fs = 16000 = 2^7 * 5^3        -> {40, 50, 64, 80, 100, 125, 128, 160}
            fs = 22050 = 2 * (3*5*7)^2    -> {45, 49, 50, 63, 70, 75, 90, 98, 105, 126, 147, 150}
            fs = 44100 = (2*3*5*7)^2      -> same as 22050 and their doubles
    """
    component: str
    causal: bool
    scale: str
    n_bins: int
    fps: int
    fs: int
    apply_sinc_envelope: bool = False

    @property
    def args(self) -> dict:
        return asdict(self)

    @property
    def hop_length(self) -> int:
        return self.fs // self.fps

    @property
    def kernel_size(self) -> int:
        return 4 * self.hop_length + 1

    @property
    def factor(self) -> int:
        """real coefficients per bin: 2 for complex (cos + sin), 1 for real/imag (quadrature factor)"""
        return 2 if self.component == "complex" else 1

    @property
    def coeffs_per_frame(self) -> int:
        """real numbers produced per time frame = factor * n_bins"""
        return self.factor * self.n_bins

    @property
    def recommended_stable_bins(self) -> int:
        """Conservative power-of-two bin count for stable float32 inversion.

        A maximum centre-frequency gap below 80% of the frame-rate alias spacing
        matched every configuration that reconstructed above 100 dB with 128 CG
        iterations in the analytical-decoder sweep.
        """
        n_bins = self.n_bins
        while n_bins < 8192:
            centers, _ = scale_freqs(self.fs, n_bins, self.scale)
            if np.diff(centers).max() <= 0.8 * self.fps:
                return n_bins
            n_bins *= 2
        return n_bins

    @property
    def model_id(self) -> str:
        causal = "causal" if self.causal else "ncausal"
        base = f"{self.fs}fs_{self.fps}fps_{self.n_bins}bins_{self.scale}_{self.component}_{causal}"
        return f"{base}_sinc" if self.apply_sinc_envelope else base

    def check_invertibility(self) -> str | None:
        """Warn when coefficient count or frequency coverage makes inversion unsafe."""
        coeffs, H = self.coeffs_per_frame, self.hop_length
        if coeffs < H:                       # redundancy < 1 -> information is genuinely lost
            need = -(-H // self.factor)      # ceil(H / factor)
            return warnings.warn(
                f"{self.model_id}: redundancy < 1 (factor*n_bins={coeffs} < hop={H}) -> the "
                f"transform is information-lossy. Raise n_bins to >= {need}.",
                stacklevel=2
            )
        stable_bins = self.recommended_stable_bins
        if stable_bins > self.n_bins:
            return warnings.warn(
                f"{self.model_id}: frequency coverage is numerically ill-conditioned for float32 "
                f"inversion. Coefficient count alone does not guarantee a stable frame; use "
                f"n_bins >= {stable_bins} for the exact decoder.",
                stacklevel=2
            )




def lin_freqs(fs:int, n_bins:int) -> np.ndarray:
    """The transform function is identity"""
    fmin = 0
    fmax = fs // 2
    centers = np.linspace(fmin, fmax, n_bins)

    fstep = centers[1] - centers[0]
    edges = np.append(centers, fmax + fstep)

    bands = np.diff(edges)
    return centers, bands



def mel_freqs(fs:int, n_bins:int) -> np.ndarray:
    """The transform function is MEL"""
    fmin = 0
    fmax = fs // 2

    fmin = librosa.hz_to_mel(fmin)
    fmax = librosa.hz_to_mel(fmax)
    centers_mel = np.linspace(fmin, fmax, n_bins)

    mel_step  = centers_mel[1] - centers_mel[0]
    edges_mel = np.linspace(centers_mel[0] - mel_step / 2, centers_mel[-1] + mel_step / 2, n_bins + 1)

    centers = librosa.mel_to_hz(centers_mel, htk=False)
    edges = librosa.mel_to_hz(edges_mel, htk=False)

    bands = np.diff(edges)
    return centers, bands



def bark_freqs(fs:int, n_bins:int) -> np.ndarray:
    """The transform function is BARK (critical-band rate, Traunmüller 1990)"""

    def hz_to_bark(f:np.ndarray) -> np.ndarray:
        """Hz -> Bark using the Traunmüller (1990) formula z = 26.81*f/(1960+f) - 0.53"""
        return 26.81 * f / (1960.0 + f) - 0.53

    def bark_to_hz(z:np.ndarray) -> np.ndarray:
        """Bark -> Hz, inverse of the Traunmüller (1990) formula f = 1960*(z+0.53)/(26.28-z)"""
        return 1960.0 * (z + 0.53) / (26.28 - z)

    fmin = 0
    fmax = fs // 2

    fmin = hz_to_bark(fmin)
    fmax = hz_to_bark(fmax)
    centers_bark = np.linspace(fmin, fmax, n_bins)

    bark_step  = centers_bark[1] - centers_bark[0]
    edges_bark = np.linspace(centers_bark[0] - bark_step / 2, centers_bark[-1] + bark_step / 2, n_bins + 1)

    centers = bark_to_hz(centers_bark)
    edges = bark_to_hz(edges_bark)

    bands = np.diff(edges)
    return centers, bands



def erb_freqs(fs:int, n_bins:int) -> np.ndarray:
    """The transform function is ERB (equivalent rectangular bandwidth rate, Glasberg & Moore 1990)"""
    
    def hz_to_erb(f:np.ndarray) -> np.ndarray:
        """Hz -> ERB-rate (Glasberg & Moore 1990) E = 21.4*log10(1 + 0.00437*f)"""
        return 21.4 * np.log10(1.0 + 0.00437 * f)

    def erb_to_hz(e:np.ndarray) -> np.ndarray:
        """ERB-rate -> Hz, inverse of the Glasberg & Moore (1990) formula f = (10^(E/21.4) - 1)/0.00437"""
        return (np.power(10.0, e / 21.4) - 1.0) / 0.00437
    
    fmin = 0
    fmax = fs // 2

    fmin = hz_to_erb(fmin)
    fmax = hz_to_erb(fmax)
    centers_erb = np.linspace(fmin, fmax, n_bins)

    erb_step  = centers_erb[1] - centers_erb[0]
    edges_erb = np.linspace(centers_erb[0] - erb_step / 2, centers_erb[-1] + erb_step / 2, n_bins + 1)

    centers = erb_to_hz(centers_erb)
    edges = erb_to_hz(edges_erb)

    bands = np.diff(edges)
    return centers, bands


def scale_freqs(fs:int, n_bins:int, scale:str) -> tuple[np.ndarray, np.ndarray]:
    """Return centre frequencies and bandwidths for a supported scale."""
    functions = {"lin": lin_freqs, "mel": mel_freqs, "bark": bark_freqs, "erb": erb_freqs}
    if scale not in functions:
        raise ValueError("Only lin, mel, bark, erb scales are supported for the SincNet Kernel")
    return functions[scale](fs, n_bins)



def compute_complex_kernel(kernel_size:int, fs:int, n_bins:int, scale:str, causal:bool, apply_sinc_envelope:bool=False) -> torch.Tensor:
    """ Compute real and imaginary part of sinc kernels
            r(x) = 2a*sinc(ax) - 2b*sinc(bx)  with x=2πt

        can be rewriten (using trig identities) as r(x) = cos(Fx) * w(x)
            with F = (a+b)/2 and B = (a-b)
            w(x) = 2B * sinc(Bx/2) 

        So the complex kernels can be written as 
            k(x) = exp(1j*Fx) * w(x)

        Reference: Section 2.1 of  FILTERBANK DESIGN FOR END-TO-END SPEECH SEPARATION [Arxiv](https://arxiv.org/pdf/1910.10400)
    """
    #compute oscillatory frequencies (the zeroth-will be removed later)
    freq_hz, band_hz = scale_freqs(fs, n_bins, scale)
    
    #compute time intervals
    t = torch.linspace(-1/2, 1/2, steps=kernel_size).view(1,-1) * kernel_size / fs
    x = 2 * torch.pi * t
    
    #compute oscillatory mode exp(i*Fx)
    F = torch.from_numpy(freq_hz).float().view(-1, 1)
    Fx = torch.matmul(F, x)
    vibrations = torch.exp(1j * Fx)

    envelope = 1
    if apply_sinc_envelope:
        #compte w(x) = 2B * sinc(Bx/2)
        #Note: the implementation torch.sinc = np.sinc corresponds to the normalised sinc defined as sinc(x)=sinc_π(x/π) 
        #Therefore w(x) = 2B * sinc_π(Bx/2π)
        B = torch.from_numpy(band_hz).float().view(-1, 1)
        Bx = torch.matmul(B, x)
        envelope = torch.sinc((Bx/2) / torch.pi) 

    #compute locality window
    window = torch.from_numpy(np.hanning(kernel_size)).float().view(1, -1)
    if causal:
        window[0, kernel_size//2+1:] = 0

    #normalise the kernel
    weights = vibrations * envelope * window
    weights = weights / torch.sum(weights.abs(), dim=1).max().item()
    return weights



class Encoder1d(nn.Module):
    def __init__(self, config:ModelArgs):
        super().__init__()
        self.stride = config.hop_length
        self.padding = config.kernel_size // 2
        self.component = config.component
        filters = compute_complex_kernel(
            kernel_size=config.kernel_size,
            fs=config.fs,
            n_bins=config.n_bins,
            scale=config.scale,
            causal=config.causal,
            apply_sinc_envelope=config.apply_sinc_envelope
        )
        filters = self.preprocess_filters(filters)
        self.register_buffer("filters", filters.unsqueeze(1))
        if self.component == "complex":
            self.register_buffer("filters_cat", torch.cat([self.filters.real, self.filters.imag], dim=0))


    def preprocess_filters(self, filters:torch.Tensor) -> torch.Tensor: 
        """ Pre-normalise the filters so that max spectrogram value <= 1"""
        assert self.component in ("real", "imag", "complex")
        if self.component == "real":
            weights = filters.real
        elif self.component == "imag":
            weights = filters.imag
        else:
            weights = filters

        norm = weights.abs().sum(dim=-1, keepdim=True)
        return filters / norm


    def forward(self, wav:torch.Tensor) -> torch.Tensor: 
        """(B,L) or (B,1,L) → (B,C,F,T) with C=1 for real/imag and C=2 for complex"""
        if len(wav.shape) < 3:
            wav = wav.unsqueeze(1)
        elif wav.size(1) != 1:
            raise ValueError("Expected mono waveform (B,1,L)")
        
        wav = F.pad(wav, (self.padding, self.padding), mode="reflect")
        
        if self.component == "complex":
            spectrogram = F.conv1d(wav, self.filters_cat, stride=self.stride)
            spectrogram = spectrogram.reshape(wav.shape[0], 2, self.filters.shape[0], -1)
        elif self.component == "real":
            spectrogram = F.conv1d(wav, weight=self.filters.real, bias=None, stride=self.stride, padding=0).unsqueeze(1)
        else:
            spectrogram = F.conv1d(wav, weight=self.filters.imag, bias=None, stride=self.stride, padding=0).unsqueeze(1)
        return spectrogram
    


class Decoder1d(nn.Module):
    def __init__(self, config:ModelArgs):
        super().__init__()
        self.config = config
        self.conv1d = nn.Conv1d(
            config.factor * config.n_bins, config.hop_length,
            kernel_size=3, padding=1, bias=False
        )
        self.conv1d.weight.data = torch.ones_like(self.conv1d.weight.data)

    def auto_resize(self, x:torch.Tensor) -> torch.Tensor:
        """Automatically pad or cut the frequency-axis to meet the dimensions of the inverter"""
        #resize frequency axis
        _, _, n_bins, _ = x.shape
        target_bins = self.config.n_bins
        if n_bins > target_bins:
            x = x[:,:,:target_bins]
        elif n_bins < target_bins:
            pad = target_bins - n_bins
            #pad from (N,C,F,T) to (N,C,F+pad,T)
            x = F.pad(x, (0,0,0,pad), mode="constant", value=0)
        return x.flatten(1,2)

    def forward(self, x:torch.Tensor, length:int|None=None) -> torch.Tensor:
        """(B,C,F,T) -> (B, L). `length` is accepted for parity with AnalyticDecoder1d but the
        learned decoder always emits n_frames * hop_length samples (it is ignored)."""
        x = self.auto_resize(x)
        x = self.conv1d(x).transpose(1,2)
        x = x.flatten(1)
        return x


class AnalyticDecoder1d(nn.Module):
    """Training-free conjugate-gradient pseudo-inverse of the encoder.

    The encoder is held by reference in a tuple so it is not registered twice
    in the module hierarchy or state dict. Exact reconstruction requires an
    injective analysis operator and enough iterations; ``reg`` optionally adds
    Tikhonov regularization for nearly singular systems. Numerical inversion is
    delegated to :func:`sincnet.cgdecoder.frame_pseudo_inverse`.
    """

    def __init__(self, config:ModelArgs, encoder:nn.Module, n_iter:int=64, tol:float=1e-9, reg:float=0.0) -> None:
        super().__init__()
        self.config = config
        self.n_iter = n_iter
        self.tol = tol
        self.reg = reg
        self._encoder = (encoder,)

    def forward(self, spec:torch.Tensor, length:int|None=None) -> torch.Tensor:
        """Map ``(B,C,F,T)`` coefficients to an exact-length waveform."""
        if length is None:
            length = spec.shape[-1] * self.config.hop_length
        return frame_pseudo_inverse(spec, self._encoder[0], length, self.n_iter, self.tol, self.reg)



class FastAnalyticDecoder1d(nn.Module):
    """ Fast differentiable decoder for Encoder1d.

        It applies the adjoint filterbank using conv_transpose1d, then approximately
        inverts the frame operator with an FFT-domain equalizer.

        This is not the exact finite-length pseudo-inverse, but it is cheap, stable,
        fully differentiable, and suitable for waveform-domain training losses.
    """

    def __init__(self, config, encoder:nn.Module, eq_eps:float=1e-2):
        super().__init__()
        self.stride = config.hop_length
        self.padding = config.kernel_size // 2
        self.component = config.component
        self.n_bins = config.n_bins
        self.kernel_size = config.kernel_size
        self.eq_eps = eq_eps

        # Reuse analysis filters.
        # Shape: (F, 1, K), complex-valued buffer.
        self.register_buffer("filters", encoder.filters.detach().clone())
        if self.component == "complex":
            self.register_buffer("filters_cat", torch.cat([self.filters.real, self.filters.imag], dim=0))
        self._eq_cache: dict = {}  # {(L_fft, device): eq_tensor}

    def _equalize(self, x_hat: torch.Tensor) -> torch.Tensor:
        """ Approximate dual-frame correction: divides by the FFT-domain frame-sum power.
            Uses the next power-of-2 FFT length so cuFFT always takes the radix-2 path
            regardless of signal length (e.g. L=16500 -> L_fft=32768 is ~4x faster than
            Bluestein on an arbitrary length). Also promotes to float32 before rfft so
            this works under AMP (half precision).
        """
        L = x_hat.shape[-1]
        L_fft = 1 << (L - 1).bit_length()  # next power of 2 >= L

        cache_key = (L_fft, x_hat.device)
        if cache_key not in self._eq_cache:
            # always float32: filter FFTs are constants, computed once and cached
            a = self.filters.real.squeeze(1).float()  # (F, K)
            b = self.filters.imag.squeeze(1).float()
            if self.component == "complex":
                G = (torch.fft.rfft(a, n=L_fft).abs() ** 2
                     + torch.fft.rfft(b, n=L_fft).abs() ** 2).sum(dim=0)
            elif self.component == "real":
                G = (torch.fft.rfft(a, n=L_fft).abs() ** 2).sum(dim=0)
            else:
                G = (torch.fft.rfft(b, n=L_fft).abs() ** 2).sum(dim=0)
            floor = self.eq_eps * G.max().clamp_min(1e-12)
            self._eq_cache[cache_key] = self.stride / torch.clamp(G, min=floor)

        eq = self._eq_cache[cache_key]
        dt = x_hat.dtype
        X = torch.fft.rfft(x_hat.float(), n=L_fft)
        y = torch.fft.irfft(X * eq, n=L_fft)
        return y[..., :L].to(dt)

    def forward(self, spec: torch.Tensor, length: int | None = None) -> torch.Tensor:
        """
            spec:
                complex mode: (B, 2, F, T)
                real/imag mode: (B, 1, F, T)

            returns:
                wav: (B, L)
        """
        # groups=1 with weight (F, 1, K): cuDNN sums over all F input channels and
        # emits (B, 1, L_ola), never materialising the (B, F, L_ola) intermediate
        # that groups=F_bins would create (which blows up memory for large F or L).
        if self.component == "complex":
            x_hat = F.conv_transpose1d(spec.flatten(1, 2), self.filters_cat.to(spec.dtype), stride=self.stride).squeeze(1)
        elif self.component == "real":
            a = self.filters.real.to(spec.dtype)
            x_hat = F.conv_transpose1d(spec[:, 0], a, stride=self.stride).squeeze(1)
        else:
            b = self.filters.imag.to(spec.dtype)
            x_hat = F.conv_transpose1d(spec[:, 0], b, stride=self.stride).squeeze(1)

        x_hat = self._equalize(x_hat)

        # The encoder reflect-pads the input by `padding` before the strided conv, so the valid
        # reconstruction sits at [padding : padding+length] of the (padding-free) transpose output.
        if length is None:
            length = spec.shape[-1] * self.stride
        x_hat = x_hat[..., self.padding : self.padding + length]
        return x_hat



class SincNet(nn.Module):
    """Custom mixed time and frequency trasnform """
    def __init__(self, fs:int=16000, fps:int=128, scale:str="lin", component:str="real", n_bins:int=128,
                 q_bits:int=6, causal:bool=True, apply_sinc_envelope:bool=True,
                 decoder_type:str="fast", cg_iters:int=128):
        """ STFT-like transform using the SincNet framework with added flexibility
            fs: int : sample rate of the input signal
            fps: int: number of frequency bins in the final 2D spectrogram
            scale: str : lin/mel/bark/erb determine the freauency spacing
            component:str : real/complex with real producing a the cos transform while complex produce the cos ans sin transforms
            n_bins: int : number of freauency bins to generate
            q_bits: int : number of bits used by the spectrogram quantizer
            causal: bool : enforce or not causality on filters
            apply_sinc_envelope: bool : whether to apply the sinc envelope to the kernels or not (see section 2.1 of https://arxiv.org/pdf/1910.10400.pdf for more details)
            decoder_type: str : reconstruction decoder (all share the same encode/decode API, all length-exact):
                "fast"   -> FastAnalyticDecoder1d: single-pass conv_transpose + 1/G equalizer, ~37 dB, differentiable, no weights (default)
                "exact"  -> AnalyticDecoder1d: conjugate-gradient pseudo-inverse, ~120 dB, no weights, slower,
                            DIFFERENTIABLE via implicit backward (usable in training, e.g. source separation)
                "learnt" -> Decoder1d: small trained overlap-add conv (requires a checkpoint)
            cg_iters: int : maximum CG iterations used by the exact decoder
        """
        super().__init__()
        assert component in ("real", "complex")
        assert decoder_type in ("fast", "exact", "learnt")
        if cg_iters <= 0:
            raise ValueError("cg_iters must be positive")
        #NOTE: check that the number of bins is a power of 2
        assert n_bins > 0 and (n_bins & (n_bins - 1)) == 0
        #NOTE: real component is only compatible with causal kernels
        causal:bool = True if component == "real" else causal

        self.config = ModelArgs(
            component=component, scale=scale, causal=causal, fps=fps, fs=fs,
            n_bins=n_bins, apply_sinc_envelope=apply_sinc_envelope
        )
        self.decoder_type = decoder_type
        self.cg_iters = cg_iters
        self.config.check_invertibility()
        self.name = self.config.model_id
        self.encoder = Encoder1d(self.config)
        if decoder_type == "fast":
            self.decoder = FastAnalyticDecoder1d(self.config, self.encoder)
        elif decoder_type == "exact":
            self.decoder = AnalyticDecoder1d(self.config, self.encoder, n_iter=cg_iters)
        else:
            self.decoder = Decoder1d(self.config)
        self.initialise_mulaw(coordinate_system="polar", q_bits=q_bits)

    def initialise_mulaw(self, coordinate_system: str = "polar", q_bits: int = 6) -> nn.Module:
        """Initialise the spectrogram quantizer."""
        coordinate_system = coordinate_system.lower()
        if coordinate_system == "polar":
            self.mulaw = PolarMuLawQuant(q_bits=q_bits)
        elif coordinate_system in ("cartesian", "cartesien"):
            self.mulaw = MuLawQuant(q_bits=q_bits)
        else:
            raise ValueError("coordinate_system must be 'polar' or 'cartesian'")
        return self.mulaw

    def load_pretrained_weights(self, weights_folder:str|None=None, freeze:bool=True, device:str="cpu", strict:bool=False, verbose:bool=False) -> None:
        """ Load pretrained weights for sincnet (only the "learnt" decoder has weights to load) """
        if self.decoder_type in ("fast", "exact"):
            return self
        weights_path = os.path.join(weights_folder, f"{self.name}.ckpt")
        checkpoint = torch.load(weights_path, map_location=torch.device(device))
        if verbose:
            print(f"Loading SincNet:{weights_path}...")
            print("EPOCH", checkpoint["epoch"], "// NSTEP", checkpoint["n_steps"])
        self.load_state_dict(checkpoint["state_dict"], strict=strict)
        for p in self.parameters():
            p.requires_grad = not freeze
        return self

    def plot_kernels(self, save_path:str=None) -> None:
        """Plot the sinc kernels in the time domain"""
        if save_path is not None:
            os.makedirs(save_path, exist_ok=True)

        filters = self.encoder.filters.cpu().numpy().squeeze(1)
        is_complex = self.config.component == "complex"
  
        for i in range(filters.shape[0]):
            fig, ax = plt.subplots(figsize=(10, 2))

            ax.plot(filters[i].real, color="blue", label="real")
            if is_complex:
                ax.plot(filters[i].imag, color="red", label="imag")

            ax.set_title(f"Kernel {i}")
            ax.legend()
            fig.tight_layout()
            if save_path is not None:
                fig.savefig(os.path.join(save_path, f"kernel_{i}.png"))

    @torch.no_grad()
    def magnitude(self, spectrogram:torch.Tensor) -> torch.Tensor:
        """Compute the magnitude spectrogram ~ (B,1,F,T) on the input signal"""
        if self.config.component == "complex":
            real, imag = spectrogram.chunk(2, dim=1)
            magnitude = torch.sqrt(real**2 + imag**2)
        else:
            magnitude = spectrogram.abs()
        return magnitude
    
    @torch.no_grad()
    def griffin_lim(self, magnitude:torch.Tensor, initial_angle:torch.Tensor|None=None, n_iters:int=50) -> torch.Tensor:
        """ Reconstruct audio from magnitude spectrogram using the Griffin-Lim algorithm
            magnitude ~ (B,1,F,T) or (B,F,T) | initial_angle ~ (B,1,F,T)
            returns complex spectrogram ~ (B,2,F,T)
        """
        assert self.config.component == "complex", "GLA requires complex spectrogram"

        if magnitude.ndim == 3:
            magnitude = magnitude.unsqueeze(1)
        if initial_angle is None:
            angle = torch.rand_like(magnitude) * 2 * np.pi
        else:
            angle = initial_angle.unsqueeze(1) if initial_angle.ndim == 3 else initial_angle

        for _ in range(n_iters+1):
            spectrogram = magnitude * torch.cat([torch.cos(angle), torch.sin(angle)], dim=1)
            forward = self.encode(self.decode(spectrogram))
            real, imag = forward.chunk(2, dim=1)
            angle = torch.atan2(imag, real)
        return spectrogram

    @torch.no_grad()
    def refine_spectrogram_phase(self, spectrogram:torch.Tensor, n_iters:int=50) -> torch.Tensor:
        """Refine the phase of the input spectrogram ~(B,2,F,T) using the Griffin-Lim algorithm """
        magnitude = self.magnitude(spectrogram)
        real, imag = spectrogram.chunk(2, dim=1)
        initial_angle = torch.atan2(imag, real)
        return self.griffin_lim(magnitude, initial_angle=initial_angle, n_iters=n_iters)

    def encode(self, x:torch.Tensor) -> torch.Tensor:
        """Compute the sincNet spectrogram ~ (B,C,F,T)"""
        return self.encoder(x)

    def decode(self, x:torch.Tensor, length:int|None=None) -> torch.Tensor:
        """ Reconstruct audio from the sincNet spectrogram ~ (B,L) with the configured decoder.
            `length` (target #samples) is honoured by the analytical decoder and ignored by the
            learned one (which always emits n_frames * hop_length). """
        return self.decoder(x, length)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        return self.decode(self.encode(x), length=length)



if __name__ == '__main__':
    from torchinfo import summary
    import matplotlib.pyplot as plt
    import os, librosa


    print("Loading audio file....")
    audio_file_path = "audio/invertibility/15033000.mp3"

    sr = 16000
    x, sr = librosa.load(audio_file_path, sr=sr, offset=0, duration=1)
    x = torch.tensor(x).unsqueeze(0)
    print("Audio file tensor shape", x.shape)
    #x = torch.rand(1, 16000)
    
    component = "complex"
    for scale in ["lin", "mel", "bark", "erb"]:
        sinc = SincNet(fs=sr, fps=128, component=component, scale=scale, causal=False, n_bins=128, apply_sinc_envelope=False)
        #sinc.plot_kernels(save_path="kernels/")

        scalogram = sinc.encode(x.unsqueeze(0))
        print(sinc.decode(scalogram).shape)
        print(scalogram.shape, scalogram.min(), scalogram.max())

        plt.imshow(scalogram.flatten(1,2)[0].detach().numpy())
        plt.savefig(f"spectral_representation_{sinc.config.scale}.png")

    
    summary(sinc, input_data=x)
