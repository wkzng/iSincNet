<div align="center">
<img src="docs/assets/logo.png" alt="isincnet" width="50%"/>
<h1>iSincNet / Fast Invertible Audio Frontend</h1>

<em>A drop-in, differentiable spectrogram layer for PyTorch that decodes back to waveforms.<br/>Linear, interpretable, CPU-fast. Handles spectrogram in various scales (linear, MEL, Bark, ERB)</em>
<br/><br/>

[![CI](https://github.com/wkzng/iSincNet/actions/workflows/ci.yml/badge.svg)](https://github.com/wkzng/iSincNet/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
</div>

---
iSincNet is as Fast and Lightweight Sincnet Spectrogram Vocoder neural network trained to reconstruct audio waveforms from their SincNet spectogram (real and signed 2d representation). We used the GTZAN dataset which is the most-used public dataset for evaluation in machine listening research for music genre recognition (MGR). The files were collected in 2000-2001 from a variety of sources including personal CDs, radio, microphone recordings, in order to represent a variety of recording conditions (http://marsyas.info/downloads/datasets.html).

<p align="center">
  <img src=docs/assets/SincNet-Filterbank.png alt="Fast and Lightweight Sincnet Spectrogram Vocoder" width="80%"/>
</p>

Datasets used during development:
- [GTZAN](https://github.com/chittalpatel/Music-Genre-Classification-GTZAN)
- [MUSDB-18](https://sigsep.github.io/datasets/musdb.html)

## Example Spectrogram 
The First 5s second of the Audio `audio/invertibility/15033000.mp3`

|  | Non-causal Encoder | Causal Encoder |
|:------:|:-------------------:|:--------------:|
| signed values | <img src="docs/assets/spec_noncausal_signed.jpeg" alt="non-causal 15033000" width="260"> | <img src="docs/assets/spec_causal_signed.jpeg" alt="causal 15033000" width="260"> |
| abs values | <img src="docs/assets/spec_noncausal_abs.jpeg" alt="non-causal 15033000" width="260"> | <img src="docs/assets/spec_causal_abs.jpeg" alt="causal 15033000" width="260"> |

## Effect of applying sincnet envelope 

As discussed in [Section 2.1](https://arxiv.org/pdf/1910.10400), SincNet can be recast as a standard wavelet transform with an envelopped defined by the sinc depending explicitly on the bandwidths as `envelope(x, B) = sinc(B x / 2)`. As a consequen the orignal cos and sine components of the filter are modulated (see example below, where we show causal filters).

| Kernel | index=10 | index=104 |
|:------:|:-------------------:|:--------------:|
| Without Sinc Envelope| <img src="docs/assets/kernels/nosinc/kernel_10.png" alt="non-causal 15033000" width="260"> | <img src="docs/assets/kernels/nosinc/kernel_104.png" alt="causal 15033000" width="260"> |
| With Sinc Envelope | <img src="docs/assets/kernels/sinc/kernel_10.png" alt="non-causal 15033000" width="260"> | <img src="docs/assets/kernels/sinc/kernel_104.png" alt="causal 15033000" width="260"> |

At lower freauencies (~low indices), the sinc envelope's effect are negligible unlike higher frequency where it forced the filter to be more localised.


### 🎧 Pretrained Models
The following table summarizes the key characteristics and access points for the available pretrained models.
All models are open-source and stored in the `pretrained/` folder.

| Sample Rate | FPS | #Bins | Weights | Corpus | Causal Encoder | Scale | Sinc Envelope | Open-Source |
|:------------:|:---:|:-----:|:--------|:--------|:----------------:|:-------:|:-------------:|:------------:|
| 16000 | 128 | 128 | [📦](pretrained/16000fs_128fps_128bins_lin_complex_ncausal.ckpt) | GTZAN | ✗ | Linear | ✗ | √ |
| 16000 | 128 | 128 | [📦](pretrained/16000fs_128fps_128bins_lin_real_causal.ckpt) | GTZAN | √ | Linear | ✗ | √ |
| 16000 | 128 | 128 | [📦](pretrained/16000fs_128fps_128bins_mel_real_causal.ckpt) | GTZAN | √ | Mel | ✗ | √ |
| 16000 | 128 | 256 | [📦](pretrained/16000fs_128fps_256bins_mel_complex_ncausal.ckpt) | GTZAN | ✗ | Mel | ✗ | √ |
| 16000 | 128 | 512 | [📦](pretrained/16000fs_128fps_512bins_mel_complex_ncausal.ckpt) | GTZAN | ✗ | Mel | ✗ | √ |
| 16000 | 128 | 128 | [📦](pretrained/16000fs_128fps_128bins_mel_complex_ncausal.ckpt) | GTZAN | ✗ | Mel | ✗ | √ |
| 16000 | 128 | 128 | [📦](pretrained/16000fs_128fps_128bins_mel_complex_ncausal_sinc.ckpt) | GTZAN | ✗ | Mel | √ | √ |
| 44100 | 350 | 128 | [📦](pretrained/44100fs_350fps_128bins_lin_complex_ncausal.ckpt) | GTZAN | ✗ | Linear | ✗ | √ |
| 44100 | 350 | 128 | [📦](pretrained/44100fs_350fps_128bins_mel_complex_ncausal.ckpt) | GTZAN | ✗ | Mel | ✗ | √ |
| 44100 | 350 | 256 | [📦](pretrained/44100fs_350fps_256bins_mel_complex_ncausal.ckpt) | GTZAN | ✗ | Mel | ✗ | √ |



## Quick Start 
```bash
pip install -r requirements.txt
```
Please refer to the [demo notebook](demo_sincnet.ipynb) which shows how to load and use the model


```python
import numpy as np
import librosa
import torch
from sincnet.model import SincNet
from datasets.utils.waveform import WaveformLoader 


SAMPLE_RATE = 16_000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
audio_loader = WaveformLoader(sample_rate=SAMPLE_RATE) 

# load the model
params = {
    "fs": SAMPLE_RATE,
    "fps": 128,
    "scale": "mel",
    "component": "complex",
    "causal": True,
    "q_bits": 8 
}

model : SincNet = (
    SincNet(**params)
    .load_pretrained_weights(weights_folder="pretrained", verbose=False)
    .eval()
    .to(device)
)

# encode and decode an audio waveform
duration = 5
offset = 0
audio_path = ... 
waveform = audio_loader.load_segment(audio_path, offset=0, duration=5, nchannels=1)
loudness = audio_loader.measure_loudness(waveform)
waveform = audio_loader.normalise_loudness(waveform, loudness, target_lufs=-23)

with torch.no_grad():
  audio_tensor = torch.from_numpy(waveform).to(device).float()

  #auto-encoding waveform -> spectrogram -> waveform
  spectrogram = model.encode(audio_tensor.unsqueeze(0))
  reconstructed_audio_tensor = model.decode(spectrogram)

  #auto-encoding waveform -> spectrogram -> quantized --> dequantized --> waveform
  spectrogram = model.encode(audio_tensor.unsqueeze(0))
  quantized_spectrogram, scale = model.mulaw.quantize(spectrogram)
  dequantized_spectrogam = model.mulaw.dequantize(quantized_spectrogram, scale)
  reconstructed_audio_tensor = model.decode(dequantized_spectrogam)
```


## Controllable STFT (exactly-invertible frontend)
`STFT` is a standard short-time Fourier transform that shares the same `encode` / `decode` / `mulaw` API as `SincNet`, but — unlike the decimated filterbank — it overlap-adds with exact COLA normalisation, so it is **alias-free and stripe-free**. You pick the two axes directly (ideally powers of two); the only coupling is `n_bins >= overlap * fs / (2 * fps)`.

Please refer to the [STFT demo notebook](demo_stft.ipynb).

```python
import torch
from sincnet import STFT, stft_params

# pick the two axes you control (ideally powers of two)
model = STFT(fs=16_000, fps=128, n_bins=128, overlap=2, q_bits=8).eval()
print(stft_params(16_000, 128, 128, 2))   # -> {'n_fft': 256, 'win_length': 250, 'hop_length': 125}

with torch.no_grad():
  #auto-encoding waveform -> spectrogram -> waveform
  spectrogram = model.encode(audio_tensor)                       # (B, 2, F, T)
  reconstructed_audio_tensor = model.decode(spectrogram, length=audio_tensor.shape[-1])

  #auto-encoding waveform -> spectrogram -> quantized --> dequantized --> waveform
  quantized_spectrogram, scale = model.mulaw.quantize(spectrogram)
  dequantized_spectrogram = model.mulaw.dequantize(quantized_spectrogram, scale)
  reconstructed_audio_tensor = model.decode(dequantized_spectrogram, length=audio_tensor.shape[-1])
```


## References Papers and Related Topics
- [1] Mirco Ravanelli, Yoshua Bengio, “Speaker Recognition from raw waveform with SincNet” [Arxiv](https://arxiv.org/abs/2109.08910)
- [2] MS-SincResNet: Joint Learning of 1D and 2D Kernels Using Multi-scale SincNet and ResNet for Music Genre Classification [Arxiv](https://arxiv.org/abs/2109.08910)
- [3] Curricular SincNet: Towards Robust Deep Speaker Recognition by Emphasizing Hard Samples in Latent Space
[Arxiv](https://arxiv.org/abs/2108.10714)
- [4] Interpretable SincNet-based Deep Learning for Emotion Recognition from EEG brain activity [Arxiv](https://arxiv.org/pdf/2107.10790)
- [5] Toward end-to-end interpretable convolutional neural networks for waveform signals [Arxiv](https://arxiv.org/pdf/2405.01815)
- [6] Filterband design for end-to-end speech separation [Arxiv](https://arxiv.org/pdf/1910.10400). This paper decomposes sinNet into a product sin * cos as implemented in this repo and bridgin the gap with Gabor filterbank

- [7] PF-Net: Personalized Filter for Speaker Recognition from Raw Waveform [Arxiv](https://arxiv.org/abs/2105.14826). This paper proposes to extend SincNet for more flexiblity by allowing alternative shapes to rectangle function in the spectral domain <img align="center"  src=docs/assets/PFnet.png width="300">

- [8] MelGAN: Generative Adversarial Networks for Conditional Waveform Synthesis [Arxiv](https://arxiv.org/pdf/1910.06711)
- [9] iSTFTNet: Fast and Lightweight Mel-Spectrogram Vocoder Incorporating Inverse Short-Time Fourier Transform [Arxiv](https://arxiv.org/abs/2203.02395)
- [10] iSTFTNet2: Faster and More Lightweight iSTFT-Based Neural Vocoder Using 1D-2D CNN [Arxiv](https://arxiv.org/pdf/2308.07117)
- [11] Deep Griffin-Lim Iteration [Arxiv](https://arxiv.org/abs/1903.03971)
- [12] Mel-Spectrogram Inversion via Alternating Direction Method of Multipliers [Arxiv](https://arxiv.org/pdf/2501.05557)
- [13] HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis [Arxiv](https://arxiv.org/abs/2010.05646)

Related discussion about SincNet vs STFT https://github.com/mravanelli/SincNet/issues/74

## Usages and Implementations around SincNet
- https://github.com/mravanelli/SincNet
- https://github.com/mravanelli/pytorch-kaldi
- https://github.com/PeiChunChang/MS-SincResNet
- https://github.com/ZaUt-bio/Exploring-Filters-in-SincNet-Access-and-Visualization/blob/main/SincNet_filters_visualization_initials.ipynb


## Roadmap and projects status
- [x] Host weights in Github and add auto-download
- [ ] Benchmark of inversion vs Griffin-Lim, iSTFTNet