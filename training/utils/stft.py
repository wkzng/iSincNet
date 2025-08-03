import torch
import numpy as np
from scipy.signal import get_window






class TorchSTFT(torch.nn.Module):
    """From :https://github.com/rishikksh20/iSTFTNet-pytorch/blob/ecbf0f635b36432bd3e432790326591bc86cadbc/stft.py#L181 """
    def __init__(self, sample_rate:int, n_fft:int,  win_length:int, window='hann', hop_length:int=None, **kwarsg):
        super().__init__()
        self.n_fft = n_fft
        self.sample_rate = sample_rate
        self.win_length = win_length
        self.hop_length = win_length // 4 if hop_length is None else hop_length
        self.window = torch.from_numpy(get_window(window, win_length, fftbins=True).astype(np.float32))


    def transform(self, waveforms:torch.Tensor, to_polar:bool=True) -> torch.Tensor:
        spectrum = torch.stft(
            waveforms,
            self.n_fft, 
            self.hop_length, 
            self.win_length,
            window=self.window.to(waveforms.device),
            return_complex=True
        )
        if to_polar:
            return torch.abs(spectrum), torch.angle(spectrum)
        return spectrum


    def inverse(self, spectrum:torch.Tensor) -> torch.Tensor:
        """perform inverse fourier transform from the complex-value spectrum"""
        signal = torch.istft(
            spectrum,
            self.n_fft, 
            self.hop_length, 
            self.win_length, 
            window=self.window.to(spectrum.device)
        )
        return signal.unsqueeze(-2) # unsqueeze to stay consistent with conv_transpose1d implementation


    def inverse_from_polar(self, magnitude:torch.Tensor, phase:torch.Tensor) -> torch.Tensor:
        spectrum = magnitude * torch.exp(phase * 1j)
        return self.inverse(spectrum)
    

    def compute_log1p_magnitude(self, x:torch.Tensor, n_bits:int=1) -> torch.Tensor:
        amplitude, _ = self.transform(x)
        mu = 2**n_bits - 1
        magnitude = torch.log(1 + mu * amplitude).unsqueeze(1)
        return magnitude



if __name__ == '__main__':

    sr = 22050
    n_fft = 1024
    win_length = 251
    stft = TorchSTFT(sample_rate = sr, n_fft = n_fft, win_length = win_length).cuda().eval()

    x = torch.rand(16, 110250).cuda()
    magntiude, phase = stft.transform(x)
    print(magntiude.shape)
