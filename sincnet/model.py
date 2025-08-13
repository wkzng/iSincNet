import torch
import torch.nn as nn
import torch.nn.functional as F

import os
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
            fs = 22050 = 2 * (3*5*7)^2  so interesting candidates for FPS are {45 49 50 63 70 75 90 98 105 126 147 150}
            fs = 16000 = 4^2 * (2*5)^3  so interesting candidates for FPS are {40, 50, 64, 80, 100, 125, 128, 160}
    """
    scale : str
    fs:int = 16000
    fps: int = 128
    n_bins: int = 128
    q_bits : int = 8
    component : str = "real"
    causal: bool = True
    neighborhood_radius : int = 4

    @property
    def args(self) -> dict:
        return asdict(self)

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
        return f"{self.fs}_{self.fps}_{self.n_bins}_{self.component}_{self.scale}_{causal}"
    
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
    y = (x + 1)/2
    z = mu * y + 0.5
    z = torch.trunc(z).to(torch.long)
    return z


def compute_backward_mu_law_quantize(x:torch.Tensor, q_bits:int) -> torch.Tensor:
    """transform tensor with values in [0, 2^q_bits-1] back into [-1,1]"""
    mu = 2**q_bits - 1
    z = x.float()
    y = (z - 0.5) / mu
    x = 2 * y - 1
    return x


def lin_freqs(fs:int, n_bins:int) -> np.ndarray:
    """The transform function is identity"""
    return np.linspace(0, fs//2, n_bins)


def mel_freqs(fs:int, n_bins:int) -> np.ndarray:
    """The transform function is MEL"""
    a = 2595 / np.log(10)
    b = 700
    forward = lambda f : a * np.log(1 + f/b)
    backward = lambda z : b * (np.exp(z/a) - 1)
    fmin = 0
    fmax = fs//2
    return backward(np.linspace(forward(fmin), forward(fmax), n_bins))
    

def get_zero_overlap_band(freq_hz:np.ndarray, fs:float) -> np.ndarray:
    """ Adaptive computation of bands to remove any whole"""
    n = len(freq_hz)
    edges = []
    for k in range(1, n):
        right = (freq_hz[k] + freq_hz[k-1]) / 2
        edges.append(right)
    
    edges = [0] + edges + [freq_hz[-1]]
    edges = np.asarray(edges)
    bands = edges[1:] - edges[:-1]
    assert np.isclose(bands.sum(), fs/2.0, rtol=1e-5)
    return bands[1:]


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
        freq_hz = lin_freqs(fs=fs, n_bins=n_bins + 1)
    elif scale == "mel":
        freq_hz = mel_freqs(fs=fs, n_bins=n_bins + 1)
    else:
        raise ValueError("Only lin, mel scales are supported for the SincNet Kernel")
    
    #compute bandwidths with sanity check
    #band_hz = get_zero_overlap_band(freq_hz, fs)
    band_hz = np.diff(freq_hz)

    #compute parametric centers and width tensors
    F = torch.from_numpy(freq_hz[1:]).float().view(-1, 1)
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
    envelope = (2*B) * torch.sinc((Bx/2) / torch.pi)

    #compute locality window
    window = torch.from_numpy(np.hanning(kernel_size)).float().view(1, -1)
    if causal:
        window[0, kernel_size//2+1:] = 0

    #normalise the kernel
    weights = vibrations * envelope * window
    weights = weights / torch.sum(weights.abs(), dim=1).view(-1, 1)
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


def rescale_spectrogram(spectrogram:torch.Tensor, eps:float=1e-12) -> torch.Tensor:
    """ Rescale a spectrogram to use the fully range [-1, 1] for the values
        This operation is required to stabilise computations in mu-law companding
        and on the decoder (prevent algebra with too small values)

        Caveats: this operation breaks the linearity of the spectrogram w.r.t components
        If a signal is decomposed: w = w1 + w2
        Then:
            s1 = sincnet(w1)
            s2 = sincnet(w2)
            s  = sincnet(w1 + w2) = s1 + s2

        However linearity is generaly lost when using rescaling.
            rescaled(s) = s / s_max
                        = (s1 + s2) / s_max
                        = ( rescaled(s1) * s1_max + rescaled(s2) * s2_max ) / s_max
                        = (s1_max/s_max) * rescaled(s1) + (s2_max/s_max) * rescaled(s2)
    """
    spectrogram_max_value = eps + torch.max(spectrogram.abs().flatten(1), dim=1).values.detach().view(-1, 1, 1)
    spectrogram = spectrogram / spectrogram_max_value
    return spectrogram


class Encoder1d(nn.Module):
    def __init__(self, config:ModelArgs):
        super().__init__()
        self.stride = config.hop_length
        self.padding = config.kernel_size // 2
        self.component = config.component
        filters = compute_complex_sinc_kernel(
            kernel_size=config.kernel_size,
            fs=config.fs,
            n_bins=config.n_bins,
            scale=config.scale,
            causal=config.causal
        )
        self.register_buffer("filters", filters.unsqueeze(1))

    def forward(self, wav:torch.Tensor) -> torch.Tensor: 
        """(B, L) or (B, 1, L) -> (B,F,T)"""
        if self.component == "real":
            weights = torch.real(self.filters)
        else:
            weights = torch.imag(self.filters)

        if len(wav.shape) < 3:
            wav = wav.unsqueeze(1)
        wav = F.pad(wav, (self.padding, self.padding), mode="reflect")
        spectrogram = F.conv1d(wav, weight=weights, bias=None, stride=self.stride, padding=0)
        return spectrogram
    


class Decoder1d(nn.Module):
    def __init__(self, config:ModelArgs):
        super().__init__()
        self.config = config
        self.shifts = config.neighborhood
        self.size = len(self.shifts)
        self.Winv = nn.Parameter(torch.ones(self.size * config.n_bins, config.hop_length))
    
    def auto_resize(self, x:torch.Tensor) -> torch.Tensor:
        """Automatically pad or cut the frequency-axis to meet the dimensions of the inverter"""
        _, n_bins, _ = x.shape
        if n_bins > self.config.n_bins:
            x = x[:,:self.config.n_bins]
        elif n_bins < self.config.n_bins:
            pad = self.config.n_bins - n_bins
            x = F.pad(x, (0, 0, 0, pad))
        return x

    def forward(self, x:torch.Tensor, eps:float=1e-5) -> torch.Tensor:
        """(B,F,T) -> (B, 1, L)"""
        #resize the input
        x = rescale_spectrogram(x)
        x = self.auto_resize(x)

        #compensate for the missing component (real or imag) with
        #forward and backward lag frames
        x = x.transpose(1,2) #(B,T,F)
        x = [roll_and_mask(x=x, k=k) for k in self.shifts]
        x = torch.cat(x, dim=-1)
        x = x @ self.Winv
        x = x.flatten(1)
        
        y = x if self.training else x.abs() 
        xmax = eps + torch.max(y, dim=1).values.detach().view(-1, 1)
        x = x / xmax
        return x


class Tokenizer(nn.Module):
    def __init__(self, q_bits:int, use_mulaw_companding:bool=True):
        super().__init__()
        self.q_bits = q_bits
        self.use_mulaw_companding = use_mulaw_companding
        self.vocab_size = 2**q_bits

    def forward(self, x):
        x = rescale_spectrogram(x)
        if self.use_mulaw_companding:
            x = compute_forward_mu_law_companding(x, q_bits=self.q_bits)
        x = compute_forward_mu_law_quantize(x, q_bits=self.q_bits)
        return x
        
    def inverse(self, x):
        x = compute_backward_mu_law_quantize(x, q_bits=self.q_bits)
        if self.use_mulaw_companding:
            x = compute_backward_mu_law_companding(x, q_bits=self.q_bits)
        return x


class SincNet(nn.Module):
    """Custom mixed time and frequency trasnform """
    def __init__(self, config:ModelArgs=None, scale:str="lin"):
        super().__init__()
        self.config = config if config else ModelArgs(scale=scale)
        self.encoder = Encoder1d(self.config)
        self.decoder = Decoder1d(self.config)
        self.name = self.config.model_id

    def load_pretrained_weights(self, weights_folder:str, freeze:bool=True, device:str="cpu") -> None:
        """ Load pretrained weights for sincnet """
        weights_path = os.path.join(weights_folder, f"{self.name}.ckpt")
        print(f"Loading SincNet:{weights_path}...")
        checkpoint = torch.load(weights_path, map_location=torch.device(device))      
        self.load_state_dict(checkpoint["state_dict"], strict=True)
        for p in self.parameters():
            p.requires_grad = not freeze
        return self

    def encode(self, x:torch.Tensor) -> torch.Tensor:
        """Compute the sincNet spectrogram"""
        x = self.encoder(x)
        if self.training:
            assert torch.isfinite(x).all()
        return x

    def decode(self, x:torch.Tensor) -> torch.Tensor:
        """Reconstruct audio from sincNet spectrogram"""
        x = self.decoder(x)
        if self.training:
            assert torch.isfinite(x).all()
        return x
    
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

    sinc = SincNet()

    scalogram = sinc.encode(x.unsqueeze(0))
    print(sinc.decode(scalogram).shape)
    print(scalogram.shape, scalogram.min(), scalogram.max())


    plt.imshow(scalogram[0].detach().numpy())
    plt.savefig(f"spectral_representation.png")

    summary(sinc, input_data=x)
