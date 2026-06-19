# SincNet filterbank — spectrograms & kernels

The encoder is a bank of complex sinc kernels (a `cos` and a `sin` filter per bin); the spectrogram
is the **signed, real-valued** 2-D response. Filters are deterministic — computed from the config,
not learned.

## Example spectrogram

First 5 s of `audio/invertibility/15033000.mp3`:

|  | Non-causal Encoder | Causal Encoder |
|:------:|:-------------------:|:--------------:|
| signed values | <img src="assets/spec_noncausal_signed.jpeg" width="260"> | <img src="assets/spec_causal_signed.jpeg" width="260"> |
| abs values | <img src="assets/spec_noncausal_abs.jpeg" width="260"> | <img src="assets/spec_causal_abs.jpeg" width="260"> |

## Effect of the sinc envelope

As discussed in [Section 2.1 of the filterbank-design paper](https://arxiv.org/pdf/1910.10400),
SincNet can be recast as a wavelet transform with an envelope defined by the sinc and the bandwidth,
`envelope(x, B) = sinc(B x / 2)`. Enable it with `apply_sinc_envelope=True`; the cos/sin components
are then modulated (causal filters shown):

| Kernel | index = 10 | index = 104 |
|:------:|:-------------------:|:--------------:|
| without sinc envelope | <img src="assets/kernels/nosinc/kernel_10.png" width="260"> | <img src="assets/kernels/nosinc/kernel_104.png" width="260"> |
| with sinc envelope | <img src="assets/kernels/sinc/kernel_10.png" width="260"> | <img src="assets/kernels/sinc/kernel_104.png" width="260"> |

At low frequencies the envelope's effect is negligible; at high frequencies it localises the filter.
