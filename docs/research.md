# Research brief — invertibility & stripe-free reconstruction of the SincNet frontend

> **Purpose.** This document (a) formalizes the SincNet analysis/synthesis transform in
> `sincnet/model.py`, (b) proves *why* `decode(encode(x))` shows horizontal stripes and why
> the current waveform‑domain equalizer is a stop‑gap, and (c) poses a precise research task:
> derive a **scale‑independent, mathematically clean** construction that (1) removes the
> stripes for *any* frequency scale and (2) gives a principled inverse. It is written to be
> self‑contained for a fresh reader with no access to the prior debugging session.

---

## 0. TL;DR of the empirical finding

`y = decode(encode(x))` is the **frame operator** `y = Ψ*Ψ x` of a complex filterbank.
Its (linearized) transfer function `G(ν)/S` is **not flat** and is full of deep dips →
those dips print through `|STFT(y)|` as horizontal stripes. We currently:

1. forced the filters to have finite bandwidth (the *sinc envelope*, `apply_sinc_envelope=True`)
   so the bank covers the axis, and
2. divided the reconstruction by `G(ν)/S` in the Fourier domain (a regularized inverse).

This removes the stripes (round‑trip SNR went from **−53 dB → 15–35 dB**), **but it is not a
general solution**: it is a post‑hoc scalar correction of an operator that is *not actually
LTI* (it has aliasing), it needs a magic regularization floor `ε`, it is non‑causal and
length‑dependent, and the residual error is scale‑dependent (lin 35 dB, mel 20 dB, bark/erb
15 dB at equal redundancy `R≈2.05`). The smell the maintainer noticed is real. We want the
*right* object: the **canonical dual frame** / a **tight‑frame redesign**.

---

## 1. Notation and the exact forward transform

Sampling rate `f_s`, frame rate `fps`, **hop / decimation** `S = f_s / fps`, kernel length
`K = 4S + 1`. We build `F` complex kernels indexed `k = 0 … F−1` (`F = n_bins`).

**Frequency scale.** A monotone warping `φ : [0, f_s/2] → ℝ` (identity for `lin`; mel, bark,
Traunmüller, ERB‑rate for the others). Centers are uniform on the warped axis and bandwidths
are the warped step mapped back to Hz:

```
φ(f_k) = φ(0) + k·Δ,      Δ = (φ(f_s/2) − φ(0)) / (F−1)
f_k  = φ⁻¹(φ(0) + k·Δ)                 # center frequency  (Hz)   -> *_freqs(...)[0]
B_k  ≈ Δ / φ'(f_k)                     # bandwidth          (Hz)   -> *_freqs(...)[1]
```

(Code: `lin_freqs / mel_freqs / bark_freqs / erb_freqs` return `(centers f_k, bands B_k)`.)
Key structural fact we are "blessed" with: **on the warped axis `u = φ(ν)` the centers are
equispaced (`u_k = u_0 + kΔ`) and the bandwidths are ~constant (`B_k·φ'(f_k) ≈ Δ`).**

**Kernel (analysis filter).** With centered time `t_n = (n/(K−1) − ½)·K/f_s`,
`n = 0 … K−1`, and `x_n = 2π t_n`:

```
ψ_k[n] = exp( i·2π f_k t_n ) · w_k[n] · h[n]
         └ carrier ────────┘   └ env ┘  └ Hann ┘
w_k[n] = sinc(B_k · t_n)           # sinc envelope ⇒ ideal passband of width B_k about f_k
h[n]   = Hann(K)                   # smooth truncation
a_k = Re ψ_k ,   b_k = Im ψ_k      # the two real conv kernels actually used
```

(Code: `compute_complex_kernel`. `sinc` here is `torch.sinc`, i.e. `sin(πz)/(πz)`.)

**Encoder = strided cross‑correlation** (`F.conv1d`, stride `S`, "complex" mode stores the two
real channels separately):

```
c_k[m] = (x ⋆_S ψ_k)[m] = Σ_n x[mS + n] · ψ_k[n]            (stored as Re, Im channels)
       = Σ_n x[mS + n] a_k[n]  +  i · Σ_n x[mS + n] b_k[n]
```

So the analysis operator is `Ψ : x ↦ (c_k)_k`, complex, decimated by `S`. Redundancy
(real coefficients per input sample) `R = 2F/S` (≈ 2.05 in the demo).

**Decoder = matched synthesis** (`F.conv_transpose1d`, the *adjoint* of `conv1d`, same kernels,
summed over `k`) — code `Decoder1d.forward`:

```
x̂[n] = Σ_k ( Σ_m c_k^re[m] a_k[n − mS]  +  Σ_m c_k^im[m] b_k[n − mS] )
      = (Ψ* c)[n]
```

i.e. **synthesis = adjoint of analysis with the same atoms ⇒ `x̂ = Ψ*Ψ x`.**
`Ψ*Ψ` is the **frame operator** `S_op`.

---

## 2. Why the stripes appear (the math)

### 2.1 Undecimated limit (`S = 1`): a pure filter, gain `G(ν)`
Substituting `c` back, `x̂ = Σ_k (a_k ⋆ a_k + b_k ⋆ b_k) * x` is convolution by
`g = Σ_k (ã_k⋆a_k + b̃_k⋆b_k)`, with transfer function

```
G(ν) = Σ_k ( |â_k(ν)|² + |b̂_k(ν)|² ).
```

Using `a_k = Re ψ_k`, `b_k = Im ψ_k` and `Ψ_k(ν) := ψ̂_k(ν)` (one‑sided, bump near `+f_k`),
**the cross terms cancel exactly**:

```
|â_k(ν)|² + |b̂_k(ν)|²  =  ½ ( |Ψ_k(ν)|² + |Ψ_k(−ν)|² )
⇒  G(ν) = ½ Σ_k ( |Ψ_k(ν)|² + |Ψ_k(−ν)|² )  ≈  ½ Σ_k |Ψ_k(ν)|²   (ν > 0).
```

So **`G(ν)` is the sum of the filters' power responses** — the classic frame "synthesis gain".
Perfect (allpass) reconstruction ⇔ `G(ν) ≡ const` ⇔ the family `{ψ_k}` is a **tight frame**.

**It is not.** Measured `G` (envelope on, `n_bins=128`):

| scale | `min G / max G` | structure of `G(ν)` |
|------|------------------|---------------------|
| lin  | 0.39 | mild ripple |
| mel  | 0.020 | flat ≤1 kHz, growing ripple above |
| bark | 0.005 | deep comb at HF |
| erb  | 0.005 | deep comb at HF |

With the **envelope off** (old default) `min/max ≈ 1e‑6`: the carriers are pure tones, the bank
leaves **uncovered gaps** between centers, `G` is a comb of near‑zero nulls → information at
those frequencies is *destroyed* by the analysis, unrecoverable by any synthesis. That was the
dominant cause of the high‑frequency stripes; the envelope is what closes the gaps.

### 2.2 Decimated reality (`S > 1`): aliasing on top of `G`
With stride `S`, `S_op = Ψ*Ψ` is **not LTI** — it is `S`‑periodically shift‑varying. Standard
filterbank (alias‑component) analysis gives, in normalized frequency `ν` (cycles/sample):

```
x̂̂(ν) = (1/S) Σ_{l=0}^{S−1} T_l(ν) · x̂(ν − l/S),
T_0(ν) = Σ_k |Ψ̃_k(ν)|²            = G(ν)      (linear/desired term)
T_l(ν) = Σ_k Ψ̃_k(ν) Ψ̃_k(ν−l/S)*   (l ≠ 0)     (aliasing terms)
```

* `T_0 = G` ⇒ desired transfer `G(ν)/S`. (Empirically the needed rescale was exactly `α = S`.)
* `T_l, l≠0` ⇒ **aliasing**; non‑zero whenever a filter and its `l/S`‑shifted copy overlap,
  i.e. when bandwidth `B_k ≳ f_s/S = fps`. High‑frequency warped filters are wide (`B_k` up to
  hundreds of Hz ≫ `fps`), so they alias — and they alias *more* for bark/erb than mel,
  explaining the scale‑dependent SNR ceiling (35 / 20 / 15 dB) **at equal redundancy**.

### 2.3 What the current code does, and why it is insufficient
`Decoder1d._equalize` applies a regularized inverse of **only the `T_0` term**:

```
x̂_eq = 𝓕⁻¹ {  S / max(G(ν), ε·max G) · 𝓕{x̂}  },     ε = 1e‑2.
```

Limitations (the "smell"):
1. **Ignores aliasing** `T_l, l≠0`. A scalar per‑frequency gain cannot cancel aliasing; this is
   the residual‑error floor.
2. **Magic floor `ε`.** Where `G→0` (coverage gaps) the inverse must be capped or it amplifies
   aliasing/noise; no value of `ε` is right because the *frame itself* is deficient there.
3. **Non‑causal, length‑dependent** (full‑signal FFT). Breaks streaming / the `causal=True` path.
4. **Diagnostic‑grade, not a construction.** It patches the output instead of fixing the atoms.

---

## 3. The research task

We want to replace §2.3 with a principled construction. Two coupled deliverables.

### Goal A — kill the stripes for *every* scale (lin/mel/bark/erb), generally
Find the construction that makes the transfer flat **by design**, ideally exploiting the warped
structure (`f_k`, `B_k`, `φ`). Candidate framings to evaluate and compare:

1. **Tight‑frame / partition‑of‑unity (preferred if achievable).** Choose prototype window `P`
   and per‑filter gains `α_k` (and/or bandwidths `B_k`) such that
   `Σ_k α_k² |Ψ_k(ν)|² = const`. On the warped axis `u = φ(ν)` the filters are uniform, so this
   is the **COLA / partition‑of‑unity** condition `Σ_k P(u − u_k) = const`. Derive the prototype
   + spacing `Δ` (overlap factor) that satisfies COLA on the warped axis, and quantify the error
   from `φ` being nonlinear across a filter's support (first‑order `B_k φ'(f_k)=Δ` vs exact).
2. **Canonical dual frame.** Keep analysis atoms; define synthesis atoms
   `ψ̃_k = S_op⁻¹ ψ_k`. For a *diagonal‑in‑frequency* approximation this is
   `Ψ̃_k(ν) = Ψ_k(ν)/G(ν)` — i.e. bake the §2.3 EQ into the **filters** (per‑filter, not a global
   post‑EQ). Derive when this diagonal approximation is exact vs when off‑diagonal (aliasing)
   blocks matter, and give the exact dual via the **frame operator on the decimated lattice**.
3. **Nonstationary Gabor frames (NSGT).** The mel/bark/erb bank is a *nonuniform* filterbank;
   NSGT (Balazs–Dörfler–Holighaus–Jaillet) give **exact perfect reconstruction** with a closed
   form for the dual when the "painless" condition holds (atoms supported within one decimation
   period). Check whether our `(B_k, S)` satisfy painlessness; if not, what minimal change does.

Please state, for the chosen route, the **explicit aliasing‑cancellation / PR condition** in
terms of `(f_k, B_k, S, K)` and the prototype, and whether it needs more redundancy
(`R = 2F/S`) or oversampling in time (smaller `S`).

### Goal B — a clean inverse ("revert the transform")
Specify the synthesis that inverts the analysis, with its assumptions:

* If Goal A yields a **tight frame** (`G≡C`): `x = (1/C)·Ψ* c` — synthesis = scaled matched
  filter, **no EQ, exact** up to aliasing (which the PR condition removes). Give `C`.
* General frame: `x = S_op⁻¹ Ψ* c` (canonical dual / least‑squares pseudo‑inverse). Provide a
  **stable, streaming‑compatible** realization of `S_op⁻¹` (e.g. short FIR dual kernels, or a
  fixed per‑band gain) rather than a full‑signal FFT, and the regularization that is
  *principled* (not a hand‑tuned floor) — e.g. tied to redundancy / frame bounds `0 < A ≤ B`.
* Address the **`causal=True`** path: one‑sided (half‑Hann) atoms break the symmetric frame;
  say what inversion is achievable and at what cost.
* Optional: the magnitude‑only inversion (Griffin–Lim, `SincNet.griffin_lim`) should benefit
  automatically once `decode∘encode ≈ id`; note any interaction.

### Deliverables
1. Derivations for §2.1–§2.3 confirmed/corrected, and the PR condition for the chosen route.
2. A concrete recipe: prototype window, gains `α_k` and/or bandwidths `B_k`, spacing/overlap,
   and the synthesis filters `ψ̃_k` — **as functions of `(f_k, B_k, φ, S, K)`** so it works for
   all four scales without per‑scale tuning.
3. Predicted frame bounds `A, B` (hence conditioning `B/A`) and expected round‑trip SNR vs
   redundancy `R`, so we know what `R`/`S` to pick.
4. Minimal changes to `compute_complex_kernel` (atoms) and `Decoder1d` (synthesis) to realize it.

### Constraints / ground truth to respect
* The transform must stay a **fixed linear filterbank** (no learned weights); atoms are
  built from `(f_k, B_k)` and a prototype only.
* `n_bins` is a power of two; `component ∈ {real, complex}`; scales `lin/mel/bark/erb` all
  supported by the same construction.
* Validate on: (i) flatness of `G(ν)` (`min/max → 1`), (ii) per‑frequency transfer
  `|STFT(y)|/|STFT(x)|` (std → 0, the literal stripe test), (iii) round‑trip SNR on broadband
  audio + a 50 Hz→7 kHz chirp, across all four scales.
* Reference points already measured (envelope on, `n_bins=128`, `R≈2.05`, current diagonal EQ):
  SNR lin/mel/bark/erb ≈ 35 / 20 / 15 / 15 dB; `G min/max` ≈ 0.39 / 0.02 / 0.005 / 0.005.
  A correct construction should push `G min/max → 1` and lift the warped‑scale SNRs toward lin.

### Pointers (theory)
Frame theory & canonical dual (`S_op⁻¹`); Daubechies–Grossmann–Meyer *painless nonorthogonal
expansions* (1986); **Nonstationary Gabor frames** (Balazs, Dörfler, Holighaus, Jaillet —
the CQT/ERBlet/“invertible audion” line); filterbank **alias‑component / paraunitary PR**
conditions (Vaidyanathan); Pariente et al., *Filterbank design for end‑to‑end speech
separation*, arXiv:1910.10400 (the analytic‑filter parametrization this code follows).

---

## 4. Where to look in the code
* `compute_complex_kernel` — atom construction (carrier · sinc envelope · Hann). **Goal A** edits here.
* `lin_freqs / mel_freqs / bark_freqs / erb_freqs` — return `(f_k, B_k)`; the warping `φ` and its
  inverse (`hz_to_*`, `*_to_hz`).
* `Encoder1d.forward` — strided `conv1d` analysis `Ψ`.
* `Decoder1d.forward` + `Decoder1d._equalize` — matched synthesis `Ψ*` and the stop‑gap EQ.
  **Goal B** replaces `_equalize` with the dual.
* `tests/test_scales.py::test_filterbank_has_no_coverage_gaps` /
  `::test_round_trip_reconstruction_is_accurate` — the current regression guards; tighten their
  thresholds once the clean construction lands.
