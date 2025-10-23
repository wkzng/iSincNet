import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import librosa
import numpy as np
from dataclasses import dataclass, asdict




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
    fps: int
    fs: int

    @property
    def args(self) -> dict:
        return asdict(self)

    @property
    def n_bins(self) -> int:
        ideal = self.fs // self.fps
        next_power_of_two = 1 << (ideal - 1).bit_length()
        next_power_of_two *= 1 if self.scale =="lin" else 2
        return next_power_of_two
    
    @property
    def hop_length(self) -> int:
        return self.fs // self.fps

    @property
    def kernel_size(self) -> int:
        return 4 * self.hop_length + 1

    @property
    def model_id(self) -> str:
        causal = "causal" if self.causal else "ncausal"
        return f"{self.fs}fs_{self.fps}fps_{self.n_bins}bins_{self.scale}_{self.component}_{causal}"


def compute_forward_mu_law_companding(x:torch.Tensor, q_bits:int) -> torch.Tensor:
    """ Compute the forward mu-law-companding of a scalogram
        doc: https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
    """
    mu = 2**q_bits - 1
    return torch.sign(x) * torch.log(1 + mu * torch.abs(x)) / np.log(1 + mu)


def compute_backward_mu_law_companding(x:torch.Tensor, q_bits:int) -> torch.Tensor:
    """ Compute the inverse mu-law-companding of a scalogram
        doc: https://en.wikipedia.org/wiki/%CE%9C-law_algorithm
    """
    mu = 2**q_bits - 1
    return torch.sign(x) * (1.0 / mu) * ((1.0 + mu)**torch.abs(x) - 1.0)


def compute_forward_mu_law_quantize(x:torch.Tensor, q_bits:int) -> torch.Tensor:
    """transform tensor with values in [-1,1] into a tensor with values in [0, 2^q_bits-1]"""
    mu = 2**q_bits - 1
    y = (x + 1) / 2.0
    y = mu * y
    y = torch.trunc(y).to(torch.long)
    return y


def compute_backward_mu_law_quantize(x:torch.Tensor, q_bits:int) -> torch.Tensor:
    """transform tensor with values in [0, 2^q_bits-1] back into [-1,1]"""
    mu = 2**q_bits - 1
    y = x.float() / mu
    y = 2 * y - 1.0
    return y


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


def compute_complex_kernel(kernel_size:int, fs:int, n_bins:int, scale:str, causal:bool, apply_sinc:bool=False) -> torch.Tensor:
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
    else:
        raise ValueError("Only lin, mel scales are supported for the SincNet Kernel")
    
    #compute time intervals
    t = torch.linspace(-1/2, 1/2, steps=kernel_size).view(1,-1) * kernel_size / fs
    x = 2 * torch.pi * t
    
    #compute oscillatory mode exp(i*Fx)
    F = torch.from_numpy(freq_hz).float().view(-1, 1)
    Fx = torch.matmul(F, x)
    vibrations = torch.exp(1j * Fx)

    envelope = 1
    if apply_sinc:
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
    def __init__(self, config:ModelArgs, scale:str):
        super().__init__()
        self.stride = config.hop_length
        self.padding = config.kernel_size // 2
        self.component = config.component
        filters = compute_complex_kernel(
            kernel_size=config.kernel_size,
            fs=config.fs,
            n_bins=config.n_bins,
            scale=scale,
            causal=config.causal
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
        """(B,L) or (B,1,L) → (B,F,T) or (B,F,T) complex"""
        if len(wav.shape) < 3:
            wav = wav.unsqueeze(1)
        elif wav.size(1) != 1:
            raise ValueError("Expected mono waveform (B,1,L)")
        
        wav = F.pad(wav, (self.padding, self.padding), mode="reflect")
        
        if self.component == "complex":
            real = F.conv1d(wav, weight=self.filters.real, bias=None, stride=self.stride, padding=0)
            imag = F.conv1d(wav, weight=self.filters.imag, bias=None, stride=self.stride, padding=0)
            spectrogram = torch.cat([real, imag], dim=1)
        elif self.component == "real":
            spectrogram = F.conv1d(wav, weight=self.filters.real, bias=None, stride=self.stride, padding=0)
        else:
            spectrogram = F.conv1d(wav, weight=self.filters.imag, bias=None, stride=self.stride, padding=0)
        return spectrogram
    


class Decoder1d(nn.Module):
    def __init__(self, config:ModelArgs):
        super().__init__()
        self.config = config
        self.factor = 2 if config.component == "complex" else 1
        self.conv1d = nn.Conv1d(
            self.factor * config.n_bins, 
            config.hop_length, 
            kernel_size=3,
            padding=1, 
            bias=False
        )
        self.conv1d.weight.data = torch.ones_like(self.conv1d.weight.data)
    
    def forward(self, x:torch.Tensor, eps:float=1e-5) -> torch.Tensor:
        """(B,F,T) -> (B, 1, L)"""
        x = self.conv1d(x).transpose(1,2)
        x = x.flatten(1)
        return x



class Quantizer(nn.Module):
    def __init__(self, q_bits:int):
        super().__init__()
        self.q_bits = q_bits
        self.vocab_size = 2**q_bits

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        x = compute_forward_mu_law_companding(x, q_bits=self.q_bits)
        x = compute_forward_mu_law_quantize(x, q_bits=self.q_bits)
        return x
        
    def inverse(self, x:torch.Tensor) -> torch.Tensor:
        x = compute_backward_mu_law_quantize(x, q_bits=self.q_bits)
        x = compute_backward_mu_law_companding(x, q_bits=self.q_bits)
        return x



class SincNet(nn.Module):
    """Custom mixed time and frequency trasnform """
    def __init__(self, fs:int=16000, fps:int=128, scale:str="lin", component:str="real"):
        """ STFT-like transform using the SincNet framework with added flexibility
            fs: int : sample rate of the input signal
            fps: int: number of frequency bins in the final 2D spectrogram
            scale: str : mel/lin determine the freauency spacing
            component:str : real/complex with real producing a the cos transform while complex produce the cos ans sin transforms
        """
        super().__init__()
        assert component in ("real", "complex")
        #NOTE: real component is only compatible with causal kernels
        causal:bool = True if component == "real" else False

        self.config = ModelArgs(component=component, scale=scale, causal=causal, fps=fps, fs=fs)
        self.complex_output = self.config.component == "complex"
        self.name = self.config.model_id
        self.encoder = Encoder1d(self.config, scale=scale)
        self.decoder = Decoder1d(self.config)

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
    
    def freeze_autoencoder(self) -> None:
        """Freeze the linear filterbank autoencoder"""
        for module in[self.encoder, self.decoder]:
            for p in module.parameters():
                p.requires_grad = False
        return self
    
    def auto_resize(self, x:torch.Tensor) -> torch.Tensor:
        """Automatically pad or cut the frequency-axis to meet the dimensions of the inverter"""
        _, n_bins, _ = x.shape
        target_bins = self.config.n_bins
        if n_bins > target_bins:
            x = x[:,:target_bins]
        elif n_bins < target_bins:
            pad = target_bins - n_bins
            x = F.pad(x, (0, 0, 0, pad))
        return x

    def split_components_or_combine_complex(self, x:torch.Tensor) -> torch.Tensor:
        """ reshape a spectrogram [real/img] to complex or vise versa"""
        if self.complex_output:
            if torch.is_complex(x):
                x = torch.cat([x.real, x.imag], dim=1)
            else:
                real, imag = torch.split(x, self.config.n_bins, dim=1)
                x = torch.complex(real, imag)
        return x

    def encode(self, x:torch.Tensor) -> torch.Tensor:
        """Compute the sincNet spectrogram"""
        return self.encoder(x)

    def decode(self, x:torch.Tensor) -> torch.Tensor:
        """Reconstruct audio from linear sincNet spectrogram"""
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

    sr = 44100
    x, sr = librosa.load(audio_file_path, sr=sr, offset=0, duration=1)
    x = torch.tensor(x).unsqueeze(0)
    print("Audio file tensor shape", x.shape)

    sinc = SincNet(fs=sr, fps=420)

    scalogram = sinc.encode(x.unsqueeze(0))
    print(sinc.decode(scalogram).shape)
    print(scalogram.shape, scalogram.min(), scalogram.max())


    plt.imshow(scalogram[0].detach().numpy())
    plt.savefig(f"spectral_representation.png")

    summary(sinc, input_data=x)
