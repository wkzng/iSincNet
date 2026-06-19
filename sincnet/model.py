import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import librosa
import warnings
import numpy as np
from dataclasses import dataclass, asdict
from .mulaw import MuLawQuant



@dataclass
class ModelArgs:
    """ Geometry & invertibility (filterbank framing -- NOT an STFT, so there is no n_fft and no
        `n_bins = n_fft/2 + 1` coupling: the number of filters N and the kernel length L are
        INDEPENDENT design choices).

            Frame rate fixes the hop:            fs = FPS * H               (H = hop_length)
            Kernel spans `coverage` hops:        L  = coverage * H + 1       (kernel_size; coverage=4)
            Each complex bin = 2 real channels   (cos + sin); component="real"/"imag" -> factor 1.

        Per frame the analysis maps an L-sample window -> factor*N real numbers (factor=2 complex).
        There are TWO distinct thresholds (see verify in .work/feasibility_128.py):

            - GLOBAL invertibility (information preserved; may need a wide overlap-add decoder):
                  factor * N >= H   (redundancy >= 1)

            - PER-FRAME invertibility (a near-per-frame decoder suffices; the exact analytic inverse
              AND exact cross-scale projection both exist):
                  factor * N >= L = coverage * H
              => with factor=2:   N * FPS >= (coverage/2) * fs       (here  N * FPS >= 2 * fs)

        The "2" is quadrature (real+imag). Below the per-frame line the transform is still GLOBALLY
        invertible (factor*N >= H), but the per-frame guarantees are lost: there is no exact
        per-frame inverse and no exact projection onto another scale (both need factor*N >= L).
        Cross the line by raising N or shortening the kernel coverage.

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

    def check_invertibility(self) -> str | None:
        """ Return a heads-up string if (n_bins, fps, kernel) sit below an invertibility threshold,
            else None. `factor*n_bins >= hop` is global invertibility; `>= kernel_size` is per-frame.
        """
        coeffs, H, L = self.coeffs_per_frame, self.hop_length, self.kernel_size
        if coeffs < H:                       # redundancy < 1 -> information is genuinely lost
            need = -(-H // self.factor)      # ceil(H / factor)
            return warnings.warn(
                f"{self.model_id}: redundancy < 1 (factor*n_bins={coeffs} < hop={H}) -> the "
                f"transform is information-lossy. Raise n_bins to >= {need}.",
                stacklevel=2
            )
        if coeffs < L:                       # per-frame non-invertible (per-frame guarantees lost)
            need = -(-L // self.factor)      # ceil(L / factor)
            return warnings.warn(
                f"{self.model_id}: per-frame non-invertible (factor*n_bins={coeffs} < "
                f"kernel_size={L}). Still globally invertible, but there is no exact per-frame "
                f"inverse and no exact projection onto another scale below this line. Set "
                f"n_bins >= {need} (factor*n_bins >= kernel_size) or reduce the kernel coverage "
                f"to cross it.",
                stacklevel=2
            )

    @property
    def model_id(self) -> str:
        causal = "causal" if self.causal else "ncausal"
        base = f"{self.fs}fs_{self.fps}fps_{self.n_bins}bins_{self.scale}_{self.component}_{causal}"
        return f"{base}_sinc" if self.apply_sinc_envelope else base




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
    if scale == "lin":
        freq_hz, band_hz = lin_freqs(fs=fs, n_bins=n_bins)
    elif scale == "mel":
        freq_hz, band_hz = mel_freqs(fs=fs, n_bins=n_bins)
    elif scale == "bark":
        freq_hz, band_hz = bark_freqs(fs=fs, n_bins=n_bins)
    elif scale == "erb":
        freq_hz, band_hz = erb_freqs(fs=fs, n_bins=n_bins)
    else:
        raise ValueError("Only lin, mel, bark, erb scales are supported for the SincNet Kernel")
    
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
            real = F.conv1d(wav, weight=self.filters.real, bias=None, stride=self.stride, padding=0)
            imag = F.conv1d(wav, weight=self.filters.imag, bias=None, stride=self.stride, padding=0)
            spectrogram = torch.stack([real, imag], dim=1)
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

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """(B,C,F,T) -> (B, L)"""
        x = self.auto_resize(x)
        x = self.conv1d(x).transpose(1,2)
        x = x.flatten(1)
        return x



class SincNet(nn.Module):
    """Custom mixed time and frequency trasnform """
    def __init__(self, fs:int=16000, fps:int=128, scale:str="lin", component:str="real", n_bins:int=128,
                 q_bits:int=8, causal:bool=True, apply_sinc_envelope:bool=False):
        """ STFT-like transform using the SincNet framework with added flexibility
            fs: int : sample rate of the input signal
            fps: int: number of frequency bins in the final 2D spectrogram
            scale: str : lin/mel/bark/erb determine the freauency spacing
            component:str : real/complex with real producing a the cos transform while complex produce the cos ans sin transforms
            n_bins: int : number of freauency bins to generate
            q_bits: int : number of bits used by the spectrogram quantizer
            causal: bool : enforce or not causality on filters
            apply_sinc_envelope: bool : whether to apply the sinc envelope to the kernels or not (see section 2.1 of https://arxiv.org/pdf/1910.10400.pdf for more details)
        """
        super().__init__()
        assert component in ("real", "complex")
        #NOTE: check that the number of bins is a power of 2
        assert n_bins > 0 and (n_bins & (n_bins - 1)) == 0
        #NOTE: real component is only compatible with causal kernels
        causal:bool = True if component == "real" else causal

        self.config = ModelArgs(
            component=component, scale=scale, causal=causal, fps=fps, fs=fs,
            n_bins=n_bins, apply_sinc_envelope=apply_sinc_envelope
        )
        self.config.check_invertibility()
        self.name = self.config.model_id
        self.encoder = Encoder1d(self.config)
        self.decoder = Decoder1d(self.config)
        self.mulaw = MuLawQuant(q_bits=q_bits)

    def load_pretrained_weights(self, weights_folder:str, freeze:bool=True, device:str="cpu", verbose:bool=False) -> None:
        """ Load pretrained weights for sincnet """
        weights_path = os.path.join(weights_folder, f"{self.name}.ckpt")
        checkpoint = torch.load(weights_path, map_location=torch.device(device))
        if verbose:
            print(f"Loading SincNet:{weights_path}...")
            print("EPOCH", checkpoint["epoch"], "// NSTEP", checkpoint["n_steps"]) 
        self.load_state_dict(checkpoint["state_dict"], strict=True)
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

    def freeze_autoencoder(self) -> None:
        """Freeze the linear filterbank autoencoder"""
        for module in[self.encoder, self.decoder]:
            for p in module.parameters():
                p.requires_grad = False
        return self
    
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

    def decode(self, x:torch.Tensor,) -> torch.Tensor:
        """Reconstruct audio from the sincNet spectrogram ~ (B,L)."""
        return self.decoder(x)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = self.encode(x)
        x = self.decode(x)
        return x



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
