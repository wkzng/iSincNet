import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import librosa
import numpy as np
from dataclasses import dataclass, asdict



def next_power_of_2(x: int) -> int:
    """ Calculates the smallest power of 2 that is greater than or equal to x.
        Example:
            >>> next_power_of_2(100): 128
            >>> next_power_of_2(64): 64
    """
    if x < 0:
        raise ValueError("Input must be a non-negative integer.")
    if x == 0:
        return 1
    # For a number x, the next power of 2 is 2 raised to the number of bits
    # in the binary representation of x-1.
    return 1 << (x - 1).bit_length()



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
            fs = 22050 = 2 * (3*5*7)^2  so interesting candidates for FPS are {45 49 50 63 70 75 90 98 105 126 147 150}
            fs = 16000 = 4^2 * (2*5)^3  so interesting candidates for FPS are {40, 50, 64, 80, 100, 125, 128, 160}
    """
    component: str 
    causal: bool
    fps: int
    fs:int = 16000
    q_bits : int = 8
    neighborhood_radius : int = 4

    @property
    def args(self) -> dict:
        return asdict(self)

    @property
    def n_bins(self) -> int:
        return next_power_of_2(self.fs // self.fps)

    @property
    def n_fft(self) -> int:
        return 2 * (self.n_bins - 1)
    
    @property
    def hop_length(self) -> int:
        return self.fs // self.fps

    @property
    def kernel_size(self) -> int:
        return 4 * self.hop_length + 1

    @property
    def model_id(self) -> str:
        causal = "causal" if self.causal else "ncausal"
        return f"{self.fs}fs_{self.fps}fps_{self.n_bins}bins_{self.component}_{causal}"
    
    @property
    def neighborhood(self) -> list[int]:
        radius = self.neighborhood_radius
        return range(-radius//2, radius//2 + 1)



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


def compute_complex_sinc_kernel(kernel_size:int, fs:int, n_bins:int, scale:str, causal:bool) -> torch.Tensor:
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
    
    #compute parametric centers and width tensors
    F = torch.from_numpy(freq_hz).float().view(-1, 1)
    B = torch.from_numpy(band_hz).float().view(-1, 1)

    #compute time intervals
    t = torch.linspace(-1/2, 1/2, steps=kernel_size).view(1,-1) * kernel_size / fs
    x = 2 * torch.pi * t
    Fx = torch.matmul(F, x)
    Bx = torch.matmul(B, x)
    
    #compute oscillatory mode exp(i*Fx)
    vibrations = torch.exp(1j * Fx)

    #compte w(x) = 2B * sinc(Bx/2)
    #Note: the implementation torch.sinc = np.sinc corresponds to the normalised sinc defined as sinc(x)=sinc_π(x/π) 
    #Therefore w(x) = 2B * sinc_π(Bx/2π)
    envelope = torch.sinc((Bx/2) / torch.pi) 

    #compute locality window
    window = torch.from_numpy(np.hanning(kernel_size)).float().view(1, -1)
    if causal:
        window[0, kernel_size//2+1:] = 0

    #normalise the kernel
    weights = vibrations * envelope * window
    weights = weights / torch.sum(weights.abs(), dim=1).max().item()
    return weights


def roll_and_mask(x:torch.Tensor, k:int) -> torch.Tensor:
    """ Roll and mask values of a (B,T,C) tensor and mask with zeros 
        k > 0: past is pushed to the present (conserves causality)
            torch.roll([1,2,3,4,5], shifts=2) --> [4,5,1,2,3] --> [0,0,1,2,3]
        k < 0: the future is pushed the the present (breach of causality)
            torch.roll([1,2,3,4,5], shifts=-2) --> [3,4,5,1,2] --> [3,4,5,0,0]
    """
    if k == 0:
        return x
    x = torch.roll(x, shifts=k, dims=1)
    if k < 0:
        x[:, k:, :] = 0
    else:
        x[:, :k, :] = 0
    return x



class Encoder1d(nn.Module):
    def __init__(self, config:ModelArgs, scale:str):
        super().__init__()
        self.stride = config.hop_length
        self.padding = config.kernel_size // 2
        self.component = config.component
        filters = compute_complex_sinc_kernel(
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
        self.shifts = config.neighborhood
        self.size = len(self.shifts)
        self.factor = 2 if config.component == "complex" else 1
        self.Winv = nn.Parameter(torch.ones(self.size * config.n_bins * self.factor, config.hop_length))
    
    def forward(self, x:torch.Tensor, eps:float=1e-5) -> torch.Tensor:
        """(B,F,T) -> (B, 1, L)"""
        #compensate for the missing component (real or imag) with
        #forward and backward lagged frames projected into the temporal space
        #NOTE: This manual operation of roll_and_mask + linear layer is strictly
        #equivalent to a 1d convolution with appropriate kernel size but for unknown
        #reasons, using linear layer produced more stable results
        x = x.transpose(1,2) #(B,T,F)
        x = [roll_and_mask(x=x, k=k) for k in self.shifts]
        x = torch.cat(x, dim=-1)
        x = x @ self.Winv
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


class FilterBankMorpher(nn.Module):
    def __init__(self, config:ModelArgs, scale:str):
        super().__init__()
        assert scale != "lin"
        self.factor = 2 if config.component == "complex" else 1
        n_bins = config.n_bins * self.factor
        self.encoder = Encoder1d(config, scale=scale)
        self.morpher = nn.Linear(n_bins, n_bins, bias=False)
        self.inverser = nn.Linear(n_bins, n_bins, bias=False)
        #NOTE: For unkown reasons, using Conv1d instead of Linear layer converged slower
        #and toward more slightly hissing white noise in reconstructed waveform.

    def encode(self, x):
        return self.encoder(x)

    def morphe(self, x):
        """(B,F,T) -> (B,F,T)"""
        x = self.morpher(x.transpose(1,2)).transpose(1,2)
        return x
    
    def inverse(self, x):
        """(B,F,T) -> (B,F,T)"""
        x = self.inverser(x.transpose(1,2)).transpose(1,2)
        return x 


class SincNet(nn.Module):
    """Custom mixed time and frequency trasnform """
    def __init__(self, component:str="real", causal:bool=True, fps:int=128):
        super().__init__()
        self.config = ModelArgs(component=component, causal=causal, fps=fps)
        self.complex_output = self.config.component == "complex"
        self.name = self.config.model_id
        self.encoder = Encoder1d(self.config, scale="lin")
        self.decoder = Decoder1d(self.config)
        self.morphers = nn.ModuleDict({
            "mel": FilterBankMorpher(self.config, scale="mel")
        })

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

    def encode(self, x:torch.Tensor, scale:str="lin") -> torch.Tensor:
        """Compute the sincNet spectrogram"""
        x = self.encoder(x)
        if scale != "lin":
            x = self.morphers[scale].morphe(x)
        return x

    def decode(self, x:torch.Tensor, scale:str="lin") -> torch.Tensor:
        """Reconstruct audio from linear sincNet spectrogram"""
        if scale != "lin":
            x = self.morphers[scale].inverse(x)
        return self.decoder(x)
    
    def forward(self, x:torch.Tensor, ret_multifb_loss:bool=True) -> torch.Tensor:
        x_lin = self.encode(x)

        scale_info = {}
        if ret_multifb_loss:
            loss_morph = 0
            loss_temp_cst = 0
            for _, filterbank in self.morphers.items():
                filterbank: FilterBankMorpher = filterbank
                x_scale = filterbank.encode(x).detach()

                # morphing consistancy (lin -> scale)
                x_scale_hat = filterbank.morphe(x_lin)
                loss_morph += F.l1_loss(x_scale_hat, x_scale)

                # roundtrip (lin -> scale -> lin)
                x_lin_hat = filterbank.inverse(x_scale_hat)
                loss_morph += F.l1_loss(x_lin_hat, x_lin)

            scale_info["morph"] = loss_morph
            scale_info["tcst"] = loss_temp_cst

        x = self.decode(x_lin)
        return x, scale_info





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

    sinc = SincNet()

    scalogram = sinc.encode(x.unsqueeze(0))
    print(sinc.decode(scalogram).shape)
    print(scalogram.shape, scalogram.min(), scalogram.max())


    plt.imshow(scalogram[0].detach().numpy())
    plt.savefig(f"spectral_representation.png")

    summary(sinc, input_data=x)
