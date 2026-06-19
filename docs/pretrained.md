# Pretrained weights

**You almost certainly don't need these.** iSincNet's default decoders are *analytical* — they invert
the encoder in closed form, with **no trained parameters**:

- `decoder_type="fast"`  — single-pass, ~37 dB, differentiable
- `decoder_type="exact"` — conjugate-gradient pseudo-inverse, ~120 dB, differentiable

The encoder filters are deterministic too (computed from the config), so they need no checkpoint
either. Weights are only required for `decoder_type="learnt"` — a small trained overlap-add conv
kept for legacy / comparison.

## Using the learnt decoder

```python
from sincnet.model import SincNet
model = (
    SincNet(fs=16000, fps=128, n_bins=128, scale="mel", component="complex",
            causal=False, decoder_type="learnt")
    .load_pretrained_weights("pretrained")
    .eval()
)
```

## Available checkpoints

All models are open-source and stored in `pretrained/` (trained on [GTZAN](https://github.com/chittalpatel/Music-Genre-Classification-GTZAN);
development also used [MUSDB-18](https://sigsep.github.io/datasets/musdb.html)).

| Sample Rate | FPS | #Bins | Weights | Causal | Scale | Sinc Envelope |
|:-----------:|:---:|:-----:|:--------|:------:|:-----:|:-------------:|
| 16000 | 128 | 128 | [📦](../pretrained/16000fs_128fps_128bins_lin_complex_ncausal.ckpt) | ✗ | Linear | ✗ |
| 16000 | 128 | 128 | [📦](../pretrained/16000fs_128fps_128bins_lin_real_causal.ckpt) | √ | Linear | ✗ |
| 16000 | 128 | 128 | [📦](../pretrained/16000fs_128fps_128bins_mel_real_causal.ckpt) | √ | Mel | ✗ |
| 16000 | 128 | 256 | [📦](../pretrained/16000fs_128fps_256bins_mel_complex_ncausal.ckpt) | ✗ | Mel | ✗ |
| 16000 | 128 | 512 | [📦](../pretrained/16000fs_128fps_512bins_mel_complex_ncausal.ckpt) | ✗ | Mel | ✗ |
| 16000 | 128 | 128 | [📦](../pretrained/16000fs_128fps_128bins_mel_complex_ncausal.ckpt) | ✗ | Mel | ✗ |
| 16000 | 128 | 128 | [📦](../pretrained/16000fs_128fps_128bins_mel_complex_ncausal_sinc.ckpt) | ✗ | Mel | √ |
| 44100 | 350 | 128 | [📦](../pretrained/44100fs_350fps_128bins_lin_complex_ncausal.ckpt) | ✗ | Linear | ✗ |
| 44100 | 350 | 128 | [📦](../pretrained/44100fs_350fps_128bins_mel_complex_ncausal.ckpt) | ✗ | Mel | ✗ |
| 44100 | 350 | 256 | [📦](../pretrained/44100fs_350fps_256bins_mel_complex_ncausal.ckpt) | ✗ | Mel | ✗ |

## Training a learnt decoder

```bash
python -m training.train      # builds SincNet(decoder_type="learnt") and trains the conv decoder
```
Not required for the analytical decoders.
