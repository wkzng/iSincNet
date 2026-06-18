import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import librosa
import numpy as np
from dataclasses import dataclass, asdict
from .mulaw import MuLawQuant



@dataclass
class ModelArgs:
    """ Theoreticall framework:
            STFT imposes:  
                n_bins = n_fft/2 + 1   -> n_fft = 2 * (n_bins - 1)
                window_length <= n_fft -> n_fft = coverage * window_length  with coverage>=1
                hop_lenth <= window_length -> window_length = overlap * hop_length  (often for anti-aliasing overlap>=4)
            Geometrically: 
                fs = FPS * hop_lenth
            Consequence:
                coverage * overlap * fs = 2 * FPS * (n_bins - 1)

        Practically: we want to control the FPS reasonably well so we need FPS to be a divisor of fs
            fs = 16000 = 2^7 * 5^3          so interesting candidates for FPS are {40, 50, 64, 80, 100, 125, 128, 160}
            fs = 22050 = 2^1 * (3*5*7)^2    so interesting candidates for FPS are {45 49 50 63 70 75 90 98 105 126 147 150}
            fs = 44100 = (2*3*5*7)^2        so interesting candidates for FPS are same as for 22050 and their doubles
    """
    component: str
    causal: bool
    scale: str
    n_bins: int
    fps: int
    fs: int
    apply_sinc_envelope: bool = False
    decoder: str = "conv"

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
    def model_id(self) -> str:
        causal = "causal" if self.causal else "ncausal"
        base = f"{self.fs}fs_{self.fps}fps_{self.n_bins}bins_{self.scale}_{self.component}_{causal}"
        if self.apply_sinc_envelope:
            base = f"{base}_sinc"
        #NOTE: the default conv decoder keeps the legacy name (backward-compatible with pretrained ckpts)
        if self.decoder != "conv":
            base = f"{base}_{self.decoder}"
        return base




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


def hz_to_bark(f:np.ndarray) -> np.ndarray:
    """Hz -> Bark using the Traunmüller (1990) formula z = 26.81*f/(1960+f) - 0.53"""
    return 26.81 * f / (1960.0 + f) - 0.53


def bark_to_hz(z:np.ndarray) -> np.ndarray:
    """Bark -> Hz, inverse of the Traunmüller (1990) formula f = 1960*(z+0.53)/(26.28-z)"""
    return 1960.0 * (z + 0.53) / (26.28 - z)


def bark_freqs(fs:int, n_bins:int) -> np.ndarray:
    """The transform function is BARK (critical-band rate, Traunmüller 1990)"""
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


def hz_to_erb(f:np.ndarray) -> np.ndarray:
    """Hz -> ERB-rate (Glasberg & Moore 1990) E = 21.4*log10(1 + 0.00437*f)"""
    return 21.4 * np.log10(1.0 + 0.00437 * f)


def erb_to_hz(e:np.ndarray) -> np.ndarray:
    """ERB-rate -> Hz, inverse of the Glasberg & Moore (1990) formula f = (10^(E/21.4) - 1)/0.00437"""
    return (np.power(10.0, e / 21.4) - 1.0) / 0.00437


def erb_freqs(fs:int, n_bins:int) -> np.ndarray:
    """The transform function is ERB (equivalent rectangular bandwidth rate, Glasberg & Moore 1990)"""
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
    def __init__(self, config:ModelArgs, normalize:bool=False):
        super().__init__()
        self.config = config
        self.factor = 2 if config.component == "complex" else 1
        in_channels = self.factor * config.n_bins
        # Optional freq-axis GroupNorm (2 groups = real/imag blocks). The warped sinc spectrogram
        # has tiny ~1e-3 magnitudes; normalising the input lets the conv train far faster/higher.
        self.norm = nn.GroupNorm(num_groups=2, num_channels=in_channels) if normalize else nn.Identity()
        self.conv1d = nn.Conv1d(
            in_channels,
            config.hop_length,
            kernel_size=3,
            padding=1,
            bias=False
        )
        # ones-init suits the raw-spectrogram conv, but is pathological after GroupNorm
        # (zero-mean input -> a ones-sum starts at ~0); keep default init when normalising.
        if not normalize:
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

    def forward(self, x:torch.Tensor, eps:float=1e-5) -> torch.Tensor:
        """(B,C,F,T) -> (B, L)"""
        x = self.auto_resize(x)
        x = self.norm(x)
        x = self.conv1d(x).transpose(1,2)
        x = x.flatten(1)
        return x



class ISTFTDecoder(nn.Module):
    """ iSTFTNet-style translator decoder: sinc-spectrogram -> linear STFT -> exact iSTFT -> waveform

        Instead of regressing the waveform directly (like Decoder1d), a thin conv re-grids the
        (warped) sinc bins into a *linear* STFT (real/imag) per frame, and torch.istft performs the
        phase-coherent overlap-add exactly. The learned part only has to learn the spectral
        re-gridding; the hard inversion is offloaded to an exact operator.
        Reference: iSTFTNet [Arxiv](https://arxiv.org/abs/2203.02395)
    """
    def __init__(self, config:ModelArgs, n_fft:int|None=None, win_length:int|None=None,
                 normalize_input:bool=True):
        super().__init__()
        self.config = config
        self.factor = 2 if config.component == "complex" else 1
        self.hop_length = config.hop_length
        self.n_fft = n_fft if n_fft is not None else 2 * config.n_bins
        self.win_length = win_length if win_length is not None else self.n_fft
        self.freq_bins = self.n_fft // 2 + 1
        in_channels = self.factor * config.n_bins
        # The (warped, L1-normalised) sinc spectrogram has tiny ~1e-3 magnitudes; without input
        # normalisation the conv gets starved gradients and converges very slowly. A freq-axis
        # GroupNorm (2 groups = real/imag) fixes the scale and lets the decoder train fast.
        self.norm = nn.GroupNorm(num_groups=2, num_channels=in_channels) if normalize_input else nn.Identity()
        # change the role of the conv: predict STFT real|imag instead of raw samples
        self.conv1d = nn.Conv1d(in_channels, 2 * self.freq_bins, kernel_size=3, padding=1)
        self.register_buffer("window", torch.hann_window(self.win_length))

    def auto_resize(self, x:torch.Tensor) -> torch.Tensor:
        """Pad or cut the frequency axis to n_bins, then flatten (B,C,F,T) -> (B, C*n_bins, T)"""
        _, _, n_bins, _ = x.shape
        target_bins = self.config.n_bins
        if n_bins > target_bins:
            x = x[:, :, :target_bins]
        elif n_bins < target_bins:
            x = F.pad(x, (0, 0, 0, target_bins - n_bins), mode="constant", value=0)
        return x.flatten(1, 2)

    def forward(self, x:torch.Tensor, length:int|None=None) -> torch.Tensor:
        """(B,C,F,T) -> (B, L) via a predicted linear STFT and an exact iSTFT"""
        x = self.auto_resize(x)                       # (B, C*n_bins, T)
        x = self.norm(x)                              # normalise the tiny sinc-spec scale
        x = self.conv1d(x)                            # (B, 2*freq_bins, T)
        real, imag = x.chunk(2, dim=1)                # (B, freq_bins, T) each
        spectrum = torch.complex(real, imag)          # (B, freq_bins, T)
        return torch.istft(
            spectrum, self.n_fft, self.hop_length, self.win_length,
            window=self.window, center=True, length=length
        )



class SincNet(nn.Module):
    """Custom mixed time and frequency trasnform """
    def __init__(self, fs:int=16000, fps:int=128, scale:str="lin", component:str="real", n_bins:int=128,
                 q_bits:int=8, causal:bool=True, apply_sinc_envelope:bool=False,
                 decoder:str="gnconv", n_fft:int|None=None):
        """ STFT-like transform using the SincNet framework with added flexibility
            fs: int : sample rate of the input signal
            fps: int: number of frequency bins in the final 2D spectrogram
            scale: str : lin/mel/bark/erb determine the freauency spacing
            component:str : real/complex with real producing a the cos transform while complex produce the cos ans sin transforms
            n_bins: int : number of freauency bins to generate
            q_bits: int : number of bits used by the spectrogram quantizer
            causal: bool : enforce or not causality on filters
            apply_sinc_envelope: bool : whether to apply the sinc envelope to the kernels or not (see section 2.1 of https://arxiv.org/pdf/1910.10400.pdf for more details)
            decoder: str : conv -> Decoder1d (learned overlap) | gnconv -> Decoder1d + freq GroupNorm | istft -> ISTFTDecoder (predict linear STFT, exact iSTFT synthesis)
            n_fft: int : synthesis fft size for the istft decoder (default 2*n_bins); ignored by the conv decoder
        """
        super().__init__()
        assert component in ("real", "complex")
        assert decoder in ("conv", "gnconv", "istft")
        #NOTE: check that the number of bins is a power of 2
        assert n_bins > 0 and (n_bins & (n_bins - 1)) == 0
        #NOTE: real component is only compatible with causal kernels
        causal:bool = True if component == "real" else causal

        self.config = ModelArgs(
            component=component, scale=scale, causal=causal, fps=fps, fs=fs,
            n_bins=n_bins, apply_sinc_envelope=apply_sinc_envelope, decoder=decoder
        )
        self.name = self.config.model_id
        self.encoder = Encoder1d(self.config)
        if decoder == "istft":
            self.decoder = ISTFTDecoder(self.config, n_fft=n_fft)        # predict linear STFT -> exact iSTFT
        elif decoder == "gnconv":
            self.decoder = Decoder1d(self.config, normalize=True)        # conv flow + freq GroupNorm
        else:
            self.decoder = Decoder1d(self.config)                        # conv flow (legacy)
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

    def decode(self, x:torch.Tensor, length:int|None=None) -> torch.Tensor:
        """Reconstruct audio from the sincNet spectrogram ~ (B,L).
        `length` is the target waveform length (used by the istft decoder for exact sizing)."""
        if isinstance(self.decoder, ISTFTDecoder):
            return self.decoder(x, length=length)
        return self.decoder(x)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        x = self.encode(x)
        x = self.decode(x, length=length)
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

    
    component = "complex"
    for scale in ["lin", "mel", "bark", "erb"]:
        sinc = SincNet(fs=sr, fps=128, component=component, scale=scale, causal=False, n_bins=256, apply_sinc_envelope=False)
        #sinc.plot_kernels(save_path="kernels/")

        scalogram = sinc.encode(x.unsqueeze(0))
        print(sinc.decode(scalogram).shape)
        print(scalogram.shape, scalogram.min(), scalogram.max())

        plt.imshow(scalogram.flatten(1,2)[0].detach().numpy())
        plt.savefig(f"spectral_representation_{sinc.config.scale}.png")

    
    summary(sinc, input_data=x)
