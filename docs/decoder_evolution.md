# Decoder evolution — inverting the SincNet spectrogram

> Engineering log for the SincNet **decoder** (spectrogram → waveform). It records why we touched
> the decoder, the principle that guided the work, every experiment with its numbers, the pivotal
> ablation, and what is still open. Code lives in [`sincnet/model.py`](../sincnet/model.py)
> (`Decoder1d`, `ISTFTDecoder`); the overfit harness is a local scratch script
> (`.work/overfit_decoder.py`, git-ignored) — its numbers are reproduced inline below.

## 1. Goal

The mel / bark / ERB ("warped") SincNet spectrogram is an excellent **representation** for our
downstream task (audio source separation) — far better than a linear STFT in practice. But the
warped SincNet transform is **not cleanly invertible**. We "got away with it" by training a learned
decoder (`Decoder1d`, a single `conv1d` that maps the spectrogram to raw samples), which did what it
could but hit a quality wall.

**Objective:** give the decoder the means to invert the warped representation cleanly, so we keep the
representation advantage upstream and gain a clean inversion downstream.

## 2. The guiding principle

> **A learned decoder can only recover what the forward transform preserved.**

The decoder's quality ceiling is set by the analysis, not the decoder's capacity. The warped SincNet
analysis loses information two ways (both diagnosed earlier in this project):

1. **Aliasing from decimation** — the decimated non-uniform filterbank is not a tight frame, so even
   the optimal diagonal-dual synthesis ceilings around ~15 dB for bark/ERB. That information is gone.
2. **Coverage gaps** — without the sinc envelope the warped bank had near-zero nulls between filter
   centres (the "horizontal stripes" saga).

So "help the decoder" means one of two things: **(a) preserve the information** (make the analysis a
better/oversampled frame), or **(b) stop asking the decoder to do the hard part of inversion** —
offload the phase-coherent overlap-add to an exact operator.

## 3. Directions considered

| # | Idea | Inversion | Effort | Verdict |
|---|------|-----------|--------|---------|
| ① | **iSTFTNet-style**: decoder predicts a *linear* STFT, exact `torch.istft` synthesizes | learned re-grid + exact OLA | low | **chosen** |
| ② | Oversampled warped frame (more redundancy + envelope) → analytic dual | mostly analytic | medium | complementary |
| ③ | Warped tight frame / NSGT (ERBlet) | exact, no learned decoder | high | future ideal |

We pursued **①** first: cheapest, reuses the exact STFT inverse we already hardened, and matches the
project's iSTFTNet references. Pipeline: `sinc-spec → (conv) STFT real/imag → torch.istft → waveform`.
The learned conv only re-grids warped bins → linear bins; the hard phase/overlap-add is exact.

## 4. Methodology — the overfit sanity

To answer "can a decoder invert the warped representation at all", we use a fast, falsifiable probe:

- Freeze the (mel, complex) SincNet **encoder** (it is fixed anyway).
- Train **only the decoder** to reconstruct **one** loudness-normalised clip (`Adam`, waveform MSE).
- Log **reconstruction SNR** vs training step.

A decoder that cannot push SNR up here is structurally broken; one that climbs fast/high *can* invert.
**Caveat baked in from the start:** this measures **memorization capacity, not generalization** — see §8.

## 5. Experiments & observations

### 5.1 First attempt — the iSTFT decoder barely learned

| decoder | params | @100 | @200 | @400 |
|---|---:|---:|---:|---:|
| conv (`Decoder1d`) | 96k | 2.2 | 5.7 | **11.6 dB** |
| istft (`n_fft=256`) | 198k | 0.2 | 0.4 | **0.8 dB** |

Surprising — the principled iSTFT decoder was *worse than the crude conv baseline*. Something was
wrong with **training**, not the idea.

### 5.2 Diagnosis — tiny inputs + wrong objective

Two causes, both confirmed:

- **Learning rate / steps.** With `lr=1e-2` and 2000 steps the iSTFT decoder reached **15.5 dB** — it
  *was* learning, just far too slowly at the original budget.
- **Loss.** Pure waveform MSE is phase-sensitive. A multi-resolution **STFT-magnitude** loss reached
  only ~4 dB *waveform* SNR (it is phase-free — good for perceptual magnitude, not waveform fidelity).
  So for the waveform-SNR metric, waveform loss is correct.

Root suspicion: the warped, **L1-normalised** SincNet spectrogram has **tiny ~1e-3 magnitudes**, which
starves the conv's gradients. (This is the same scale theme that ran through the whole project — the
SincNet coefficients are ~200× smaller than raw STFT coefficients because each filter is L1-normalised.)

### 5.3 The fix — normalize the decoder input

Add a normalization on the decoder input (per-channel). At the **original** budget (`lr=3e-3`, 400 steps):

| decoder | @50 | @100 | @200 | @400 |
|---|---:|---:|---:|---:|
| istft + input BatchNorm | 1.2 | 6.5 | 16.3 | **23.7 dB** |

From 0.8 → 23.7 dB. **Normalization was the dominant lever.** This immediately raised the question:
*is the iSTFT structure even necessary, or is normalization the whole story?*

### 5.4 The pivotal ablation — normalization vs structure

Test the **old conv flow** (conv → raw samples) but with a **freq-axis `GroupNorm(2 groups = real/imag)`**
on the input — same "normal flow", just normalized. Same budget (`lr=3e-3`, 400 steps):

| decoder | params | @100 | @200 | @400 |
|---|---:|---:|---:|---:|
| conv (no norm) | 96,000 | 2.2 | 5.7 | 11.6 |
| **gnconv** (conv + GroupNorm) | **96,512** | 22.0 | 27.1 | **32.3 dB** |
| istft + norm | 198,914 | 6.5 | 16.3 | 23.7 |

**The normalized conv won outright** — best reconstruction, ~conv's parameter count, beating the
iSTFT decoder at half the parameters. **Conclusion: on this probe, the win is normalization, not the
iSTFT machinery.** Normalizing the tiny ~1e-3 input is what mattered.

## 6. Formalization — three decoders

We kept all three as selectable options (encoded in `model_id`):

| `decoder=` | class | input norm | conv init | `model_id` suffix |
|---|---|---|---|---|
| `"conv"` *(default)* | `Decoder1d` | none | all-ones | *(none — legacy name)* |
| `"gnconv"` | `Decoder1d(normalize=True)` | `GroupNorm(2)` | default | `_gnconv` |
| `"istft"` | `ISTFTDecoder` | `GroupNorm(2)` | default | `_istft` |

Two implementation details that bit us:

- **Ones-init pathology.** `Decoder1d` force-inits the conv to all-ones (a reasonable "sum the bins"
  start for the *raw* spectrogram). After a GroupNorm (which zero-means the input), a ones-sum starts
  at ~0 and barely trains — the first formalized `gnconv` cratered to **−27 dB**. Fix: keep the
  ones-init only for the un-normalised conv; use default init when normalizing.
- **GroupNorm over BatchNorm.** We standardised both normalized decoders on `GroupNorm(2)` instead of
  BatchNorm: batch-size independent, no train/eval running-stat surprises (safer for varied batch /
  streaming), and it *improved* the iSTFT decoder (23.7 → **28.5 dB**).

### Final board (formalized, `lr=3e-3`, 400 steps, one clip)

| decoder | params | @400 |
|---|---:|---:|
| conv | 96,000 | 11.6 dB |
| **gnconv** | 96,512 | **32.6 dB** |
| istft | 198,914 | 28.5 dB |

## 7. Key findings

1. **Normalization is mandatory** for any SincNet decoder — the warped, L1-normalised spectrogram is
   ~1e-3 in scale and starves gradients otherwise. This is the same scale lesson that recurs across the
   project (it also explains why a *raw* STFT spectrogram, ~200× larger, wrecks a net tuned for SincNet).
2. **The dominant lever was normalization, not the iSTFT structure** — at least by the memorization probe.
3. **`GroupNorm(2 groups = real/imag)`** is a clean, batch-independent way to apply it.
4. **Init matters**: ones-init is good for the raw conv, pathological after a normalization layer.

## 8. Caveat — overfit ≠ generalization

The board ranks **memorization capacity on one clip**, and that is exactly where the architectures
differ in a way the probe cannot see:

- **`gnconv`** emits `T·hop` *independent* samples (no overlap-add). With normalized inputs it can fit
  one waveform almost as a linear map — so it tops the overfit board — but across a dataset it inherits
  `Decoder1d`'s weakness: **block-boundary artifacts** (the original wall).
- **`istft`** is *constrained* to emit a valid STFT then overlap-add. That inductive bias caps one-clip
  memorization but is exactly what tends to **generalize** cleanly (smooth, phase-coherent) across data.

So: "best at overfitting one clip" ≠ "best separation / vocoding". The real verdict is the separation
training, not this probe.

## 9. How to use

```python
from sincnet.model import SincNet

# start with the normalized conv (cheapest, best memorizer)
m = SincNet(fs=16000, fps=128, n_bins=128, scale="mel", component="complex",
            causal=False, decoder="gnconv")
print(m.name)   # 16000fs_128fps_128bins_mel_complex_ncausal_gnconv

spec = m.encode(wav)                  # (B, 2, F, T)  mel-sinc spectrogram
audio = m.decode(spec)                # gnconv/conv: length = T*hop
# audio = m.decode(spec, length=L)    # istft: pass length for exact sizing
```

`decoder="istft"` (and `n_fft=256→512→1024`) is one argument away for the A/B.

## 10. Open questions / next steps

- **Train all three on the real separation task** — the only verdict that counts. Hypothesis: `gnconv`
  converges fastest but plateaus on quality (boundary artifacts); `istft` trains slower but reconstructs
  cleaner; both need the normalization.
- **Perceptual losses** — multi-resolution STFT-magnitude (+ adversarial) for `istft`, since waveform
  MSE under-trains vocoder-style synthesis.
- **`n_fft` sweep** for `istft` (256 → 512 → 1024): fidelity vs params.
- **Directions ② / ③** — raise analysis redundancy (so the *analytic* dual inverts well) and/or build a
  warped tight frame (NSGT/ERBlet) for a fully-analytic, learning-free inversion. See
  [`docs/research.md`](research.md).
