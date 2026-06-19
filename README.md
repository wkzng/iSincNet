<div align="center">
<img src="docs/assets/logo.png" alt="isincnet" width="50%"/>
<h1>iSincNet / Fast Invertible Audio Frontend</h1>

<em>A drop-in, differentiable spectrogram layer for PyTorch that decodes back to the waveform exactly, with no trained weights. Linear, interpretable, CPU-fast. Any scale: linear, mel, Bark, ERB.</em>
<br/><br/>

[![CI](https://github.com/wkzng/iSincNet/actions/workflows/ci.yml/badge.svg)](https://github.com/wkzng/iSincNet/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-yellow.svg)](LICENSE)
</div>

---

**iSincNet** turns a waveform into an interpretable 2-D spectrogram (a deterministic SincNet
filterbank) and back into the waveform. The decoder is a closed-form inverse of the encoder: no
vocoder, no training, no checkpoints. Pick any frequency scale (linear, mel, Bark, ERB);
reconstruction is length-exact and, for a well-conditioned bank, near-exact (about 100+ dB, up to
roughly 125 dB).

<p align="center">
  <img src="docs/assets/SincNet-Filterbank.png" alt="SincNet filterbank" width="80%"/>
</p>

Filterbank kernels, example spectrograms, and the optional sinc envelope: [docs/filters.md](docs/filters.md).

## Quick start

```bash
pip install .            # core (torch, numpy, librosa)
pip install .[demo]      # adds matplotlib + soundfile for the demo notebook
```

```python
import torch
from sincnet.model import SincNet

# nothing to download: the decoder is the analytical inverse of the encoder
model = SincNet(
  fs=16_000, fps=128, n_bins=256,
  scale="mel",            # any scale: "lin" | "mel" | "bark" | "erb"
  component="complex", causal=False, decoder_type="exact",
).eval()

wav = torch.randn(1, 16_000)                            # (B, T)
with torch.no_grad():
    spec  = model.encode(wav)                           # (B, 2, F, T) signed real/imag spectrogram
    recon = model.decode(spec, length=wav.shape[-1])    # (B, T) exact, training-free, length-preserving

    # optional mu-law quantization for a compact representation:
    q, scale = model.mulaw.quantize(spec)
    recon = model.decode(model.mulaw.dequantize(q, scale), length=wav.shape[-1])
```

See the [demo notebook](demo_sincnet.ipynb).

## Decoders

All share the same `encode` / `decode` API and are length-exact (pick via `decoder_type`):

| `decoder_type`   | reconstruction                                   | weights | differentiable | remark                                                                |
|:----------------:|:-------------------------------------------------|:-------:|:--------------:|:----------------------------------------------------------------------|
| `fast` (default) | ~37 dB, single-pass `conv_transpose` + equalizer | none    | yes            | fast path; fine for training losses                                   |
| `exact`          | ~100+ dB, conjugate-gradient pseudo-inverse      | none    | yes            | implicit backward, O(1) memory in iterations; for end-to-end training |

## Invertibility

Whether `decode(encode(x))` is exact is fixed, before any training, by `n_bins`, `fps`, and the
kernel length. The bank is per-frame invertible (a clean closed-form inverse exists) once:

```
2 * n_bins >= kernel_size = coverage * fs / fps        <=>        n_bins * fps >= (coverage / 2) * fs
```

At `fps = 128` that means `n_bins >= 256`. `SincNet` warns at construction when a config sits below
the line. Comparing `STFT(x)` with `STFT(decode(encode(x)))`:

**128 bins (below the line): visible residual.**
![128 bins](docs/assets/sincnet_128fps128bins.png)

**512 bins (well above the line): indistinguishable.**
![512 bins](docs/assets/sincnet_128fps512bins.png)

Full derivation and more figures: [docs/invertibility_constraint.md](docs/invertibility_constraint.md).

## References
- [1] Ravanelli & Bengio, *Speaker Recognition from raw waveform with SincNet* [arXiv](https://arxiv.org/abs/2109.08910)
- [2] *Filterbank design for end-to-end speech separation* [arXiv](https://arxiv.org/pdf/1910.10400), which decomposes SincNet into a `cos * sin` product (the formulation used here)
- [3] *Toward end-to-end interpretable convolutional neural networks for waveform signals* [arXiv](https://arxiv.org/pdf/2405.01815)
- [4] *PF-Net: Personalized Filter for Speaker Recognition from Raw Waveform* [arXiv](https://arxiv.org/abs/2105.14826)

Related discussion, SincNet vs STFT: https://github.com/mravanelli/SincNet/issues/74

## SincNet in the wild
- https://github.com/mravanelli/SincNet
- https://github.com/mravanelli/pytorch-kaldi
- https://github.com/PeiChunChang/MS-SincResNet
