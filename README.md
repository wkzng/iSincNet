# iSincNet (Lightweight Sincnet Spectrogram Vocoder)

[[Blog]](https://github.com/wkzng/iSincNet) [[SincNet Paper]](https://arxiv.org/abs/1808.00158)

iSincNet is as Fast and Lightweight Sincnet Spectrogram Vocoder neural network trained to reconstruct audio waveforms from their SincNet spectogram (real and signed 2d representation). We used the GTZAN dataset which is the most-used public dataset for evaluation in machine listening research for music genre recognition (MGR). The files were collected in 2000-2001 from a variety of sources including personal CDs, radio, microphone recordings, in order to represent a variety of recording conditions (http://marsyas.info/downloads/datasets.html).

<p align="center">
  <img src=illustrations/SincNet-Filterbank.png alt="Fast and Lightweight Sincnet Spectrogram Vocoder" width="80%"/>
</p>

### Invertibility samples

| Sample | Non-causal (image) | Causal (image) |
|:------:|:-------------------:|:--------------:|
| <audio controls src="audio/invertibility/15033000.mp3"></audio><br><audio controls src="audio/invertibility/p232_002.wav"></audio> | <img src="images/invertibility/noncausal_15033000.png" alt="non-causal 15033000" width="260"> | <img src="images/invertibility/causal_15033000.png" alt="causal 15033000" width="260"> |

<!-- Optional second row if you have separate images for p232_002 -->
<!--
| <audio controls src="audio/invertibility/p232_002.wav"></audio> | <img src="images/invertibility/noncausal_p232_002.png" alt="non-causal p232_002" width="260"> | <img src="images/invertibility/causal_p232_002.png" alt="causal p232_002" width="260"> |
-->
> Tip: Keep the audio & image paths **relative** (as above) so they work when viewing the repo on GitHub.


## Available models
The following table summarizes the key characteristics and access points for the available pretrained models.
| Sample rate | FPS | # Bins |                              Weights                             | Corpus | Causal Encoder | Open-Source |
| :---------: | :-: | :----: | :--------------------------------------------------------------: | :----: | :------------: | :---------: |
|    16 kHz   | 128 |   128  | [📦](pretrained/16000fs_128fps_128bins_lin_complex_ncausal.ckpt) |  GTZAN |        ×       |      √      |
|    16 kHz   | 128 |   128  |   [📦](pretrained/16000fs_128fps_128bins_lin_real_causal.ckpt)   |  GTZAN |        √       |      √      |
|    16 kHz   | 128 |   256  | [📦](pretrained/16000fs_128fps_256bins_mel_complex_ncausal.ckpt) |  GTZAN |        ×       |      √      |
|   44.1 kHz  | 210 |   256  | [📦](pretrained/44100fs_210fps_256bins_lin_complex_ncausal.ckpt) |  GTZAN |        ×       |      √      |
|   44.1 kHz  | 210 |   512  | [📦](pretrained/44100fs_210fps_512bins_mel_complex_ncausal.ckpt) |  GTZAN |        ×       |      √      |
|   44.1 kHz  | 350 |   128  |   [📦](pretrained/44100fs_350fps_128bins_lin_real_causal.ckpt)   |  GTZAN |        √       |      √      |
|   44.1 kHz  | 350 |   128  | [📦](pretrained/44100fs_350fps_128bins_lin_complex_ncausal.ckpt) |  GTZAN |        ×       |      √      |
|   44.1 kHz  | 350 |   256  | [📦](pretrained/44100fs_350fps_256bins_mel_complex_ncausal.ckpt) |  GTZAN |        ×       |      √      |
|   44.1 kHz  | 350 |   256  |   [📦](pretrained/44100fs_350fps_256bins_mel_real_causal.ckpt)   |  GTZAN |        √       |      √      |


# TODO: Benchmark
colums: architecture or method | dataset | MSE | MAE | SNR | checkpoint

datasets:
- [GTZAN](https://github.com/chittalpatel/Music-Genre-Classification-GTZAN)
- [MUSDB-18](https://sigsep.github.io/datasets/musdb.html)


## Quick Start 
```bash
pip install -r requirements.txt
```
Please refer to the [demo notebook](demo.ipynb) which shows how to load and use the model



```python
import numpy as np
import librosa
import torch
from sincnet.model import SincNet, Quantizer
from datasets.utils.waveform import WaveformLoader 

# load the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SincNet().load_pretrained_weights().eval().to(device)
processor = WaveformLoader(sample_rate=SAMPLE_RATE) 

# encode and decode an audio waveform
sample_rate = 16_000
duration = 5
offset = 0
audio_path = ... 
waveform = audio_loader.load_segment(audio_path, offset=0, duration=5, nchannels=1)
loudness = audio_loader.measure_loudness(waveform)
waveform = audio_loader.normalise_loudness(waveform, loudness, target_lufs=-23)

with torch.no_grad():
  audio_tensor = torch.from_numpy(waveform).to(device).float()
  spectrogram = model.encode(audio_tensor.unsqueeze(0), scale="mel")
  reconstructed_audio_tensor = model.decode(spectrogram, scale="mel")

#(optional) elementwise quantization into a vocabulary of size 2^{q_bits}
quantizer = Quantizer(q_bits=10).to(device)
indices = quantizer(spectrogram)
detokenized_spectrogram = tokenizer.inverse(indices)
detokenized_audio = model.decode(detokenized_spectrogram)
```


## References Papers and Related Topics
- [1] Mirco Ravanelli, Yoshua Bengio, “Speaker Recognition from raw waveform with SincNet” [Arxiv](https://arxiv.org/abs/2109.08910)
- [2] MS-SincResNet: Joint Learning of 1D and 2D Kernels Using Multi-scale SincNet and ResNet for Music Genre Classification [Arxiv](https://arxiv.org/abs/2109.08910)
- [3] Curricular SincNet: Towards Robust Deep Speaker Recognition by Emphasizing Hard Samples in Latent Space
[Arxiv](https://arxiv.org/abs/2108.10714)
- [4] Interpretable SincNet-based Deep Learning for Emotion Recognition from EEG brain activity [Arxiv](https://arxiv.org/pdf/2107.10790)
- [5] Toward end-to-end interpretable convolutional neural networks for waveform signals [Arxiv](https://arxiv.org/pdf/2405.01815)
- [6] Filterband design for end-to-end speech separation [Arxiv](https://arxiv.org/pdf/1910.10400). This paper decomposes sinNet into a product sin * cos as implemented in this repo and bridgin the gap with Gabor filterbank

- [7] PF-Net: Personalized Filter for Speaker Recognition from Raw Waveform [Arxiv](https://arxiv.org/abs/2105.14826). This paper proposes to extend SincNet for more flexiblity by allowing alternative shapes to rectangle function in the spectral domain <img align="center"  src=illustrations/PFnet.png width="300">

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
- [x] Added Automatic projection forward and backward to MEL scale
- [ ] Benchmark of inversion vs Griffin-Lim, iSTFTNet
- [ ] Host weights in cloud and add auto-download

## Contributions and acknowledgment
Show your appreciation to those who have contributed to the project.
