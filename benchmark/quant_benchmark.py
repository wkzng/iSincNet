"""
quant_benchmark.py
------------------
Compares MuLaw quantizers across bit-depth configurations.

Metrics: SDR and SI-SDR (both in dB, higher = better).

Usage (drop into your notebook):
    from quant_benchmark import run_benchmark, plot_heatmaps
    results = run_benchmark(audio_paths, sinc, audio_loader, device)
    plot_heatmaps(results)

    # Or pass exactly the quantizers you want:
    specs = [QuantizerSpec("predictive", (6, 4), "Pred(6,4)", lambda: PredictivePolarQuant(q_mag=6, q_phi=4))]
    results = run_benchmark(audio_paths, sinc, audio_loader, device, quantizer_specs=specs)
"""

import torch
import numpy as np
from dataclasses import dataclass
from itertools import product
from typing import Callable

from sincnet.model import SincNet, scale_freqs
from sincnet.mulaw import DemodulatedPolarQuant, MuLawQuant, PolarMuLawQuant, PredictivePolarQuant


# ---------------------------------------------------------------------------
# Alias so both share the same interface: quantize() -> (tokens, scale)
# ---------------------------------------------------------------------------

class CartesianMuLawQuant(torch.nn.Module):
    """Thin wrapper around MuLawQuant exposing the same interface as PolarMuLawQuant."""

    def __init__(self, q_real: int, q_imag: int, eps: float = 1e-8, dither: bool = False):
        super().__init__()
        assert q_real == q_imag, "Cartesian wrapper assumes symmetric bits for now"
        #from mu_law_quant import MuLawQuant   # your original module
        self.q = MuLawQuant(q_bits=q_real, eps=eps, dither=dither, pre_scaling=True)
        self.label = f"Cart({q_real},{q_imag})"

    def quantize(self, x):
        tokens, scale = self.q.quantize(x)
        return tokens, scale   # tokens ~ (B, 2, F, T) int

    def dequantize(self, tokens, scale):
        return self.q.dequantize(tokens, scale)

    def forward(self, x):
        tokens, scale = self.quantize(x)
        return self.dequantize(tokens, scale)


@dataclass(frozen=True)
class QuantizerSpec:
    """Benchmark entry for one quantizer configuration."""
    family: str
    config: tuple[int, ...]
    label: str
    make_quantizer: Callable[[], torch.nn.Module]


def default_quantizer_specs(
    q_bits_cartesian: list[int] | None = None,
    q_mag_polar: list[int] | None = None,
    q_phi_polar: list[int] | None = None,
    q_mag_predictive: list[int] | None = None,
    q_phi_predictive: list[int] | None = None,
    q_mag_demodulated: list[int] | None = None,
    q_phi_demodulated: list[int] | None = None,
    center_frequencies_hz: np.ndarray | torch.Tensor | None = None,
    frame_rate: float | None = None,
) -> list[QuantizerSpec]:
    """Build the default Cartesian, Polar, PredictivePolar, and DemodulatedPolar sweep."""
    q_bits_cartesian = [4, 6, 8, 10] if q_bits_cartesian is None else q_bits_cartesian
    q_mag_polar = [4, 6, 8] if q_mag_polar is None else q_mag_polar
    q_phi_polar = [2, 4, 6, 8] if q_phi_polar is None else q_phi_polar
    q_mag_predictive = q_mag_polar if q_mag_predictive is None else q_mag_predictive
    q_phi_predictive = q_phi_polar if q_phi_predictive is None else q_phi_predictive
    q_mag_demodulated = q_mag_polar if q_mag_demodulated is None else q_mag_demodulated
    q_phi_demodulated = q_phi_polar if q_phi_demodulated is None else q_phi_demodulated

    specs: list[QuantizerSpec] = []
    for q in q_bits_cartesian:
        specs.append(
            QuantizerSpec(
                family="cartesian",
                config=(q, q),
                label=f"Cart({q},{q})",
                make_quantizer=lambda q=q: CartesianMuLawQuant(q_real=q, q_imag=q),
            )
        )

    for qm, qp in product(q_mag_polar, q_phi_polar):
        specs.append(
            QuantizerSpec(
                family="polar",
                config=(qm, qp),
                label=f"Polar({qm},{qp})",
                make_quantizer=lambda qm=qm, qp=qp: PolarMuLawQuant(q_mag=qm, q_phi=qp),
            )
        )

    for qm, qp in product(q_mag_predictive, q_phi_predictive):
        specs.append(
            QuantizerSpec(
                family="predictive",
                config=(qm, qp),
                label=f"Pred({qm},{qp})",
                make_quantizer=lambda qm=qm, qp=qp: PredictivePolarQuant(q_mag=qm, q_phi=qp),
            )
        )

    if center_frequencies_hz is not None and frame_rate is not None:
        for qm, qp in product(q_mag_demodulated, q_phi_demodulated):
            specs.append(
                QuantizerSpec(
                    family="demodulated",
                    config=(qm, qp),
                    label=f"Demod({qm},{qp})",
                    make_quantizer=lambda qm=qm, qp=qp: DemodulatedPolarQuant(
                        center_frequencies_hz=center_frequencies_hz,
                        frame_rate=frame_rate,
                        q_mag=qm,
                        q_phi=qp,
                    ),
                )
            )
    return specs


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def db_ratio(numerator: torch.Tensor, denominator: torch.Tensor, eps: float = 1e-8) -> float:
    """Return a finite dB ratio with eps guarding the denominator and log input."""
    ratio = numerator / (denominator + eps)
    return 10 * torch.log10(torch.clamp(ratio, min=eps)).item()


def sdr(ref: torch.Tensor, est: torch.Tensor, eps: float = 1e-8) -> float:
    """Signal-to-Distortion Ratio in dB.  ref, est ~ (samples,) or (C, samples)"""
    ref = ref.flatten().float()
    est = est.flatten().float()
    noise = est - ref
    return db_ratio((ref ** 2).sum(), (noise ** 2).sum(), eps=eps)


def si_sdr(ref: torch.Tensor, est: torch.Tensor, eps: float = 1e-8) -> float:
    """Scale-Invariant SDR in dB."""
    ref = ref.flatten().float()
    est = est.flatten().float()
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = (est * ref).sum() / ((ref ** 2).sum() + eps)
    proj = alpha * ref
    noise = est - proj
    return db_ratio((proj ** 2).sum(), (noise ** 2).sum(), eps=eps)


# ---------------------------------------------------------------------------
# Single audio evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_one(wav_np: np.ndarray, sinc, quantizer, device: str) -> dict:
    """
    wav_np  : numpy waveform (1, T) already loudness-normalised
    sinc    : encoder/decoder (encode returns (B,2,F,T), decode returns waveform)
    Returns dict with keys: sdr_db, si_sdr_db
    """
    wav = torch.from_numpy(wav_np).to(device).float()

    spec = sinc.encode(wav)                        # (B, 2, F, T)
    tokens, scale = quantizer.quantize(spec)
    spec_r = quantizer.dequantize(tokens, scale)

    wav_r = sinc.decode(spec_r)

    # align lengths (encoder/decoder may add/remove a few samples)
    n = min(wav.shape[-1], wav_r.shape[-1])
    return {
        "sdr":    sdr(wav[..., :n],   wav_r[..., :n]),
        "si_sdr": si_sdr(wav[..., :n], wav_r[..., :n]),
    }


# ---------------------------------------------------------------------------
# Full benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    audio_paths: dict,
    sinc,
    audio_loader,
    device: str,
    sample_rate: int = 22050,
    offset: float = 0.0,
    duration: float = 5.0,
    target_lufs: float = -23.0,
    q_bits_cartesian: list[int] | None = None,
    q_mag_polar: list[int] | None = None,
    q_phi_polar: list[int] | None = None,
    q_mag_predictive: list[int] | None = None,
    q_phi_predictive: list[int] | None = None,
    q_mag_demodulated: list[int] | None = None,
    q_phi_demodulated: list[int] | None = None,
    quantizer_specs: list[QuantizerSpec] | None = None,
) -> dict:
    """
    Returns a results dict with structure:
        results[family][config] = {"sdr": float, "si_sdr": float, "label": str}
    Values are means over all audio files.
    """
    if quantizer_specs is None:
        center_frequencies_hz, _ = scale_freqs(sinc.config.fs, sinc.config.n_bins, sinc.config.scale)
        quantizer_specs = default_quantizer_specs(
            q_bits_cartesian=q_bits_cartesian,
            q_mag_polar=q_mag_polar,
            q_phi_polar=q_phi_polar,
            q_mag_predictive=q_mag_predictive,
            q_phi_predictive=q_phi_predictive,
            q_mag_demodulated=q_mag_demodulated,
            q_phi_demodulated=q_phi_demodulated,
            center_frequencies_hz=center_frequencies_hz,
            frame_rate=sinc.config.fps,
        )

    # --- load all waveforms once ---
    waveforms = []
    for key, path in audio_paths.items():
        wav = audio_loader.load_segment(path, offset=offset, duration=duration, nchannels=1)
        loudness = audio_loader.measure_loudness(wav)
        wav = audio_loader.normalise_loudness(wav, loudness, target_lufs=target_lufs)
        waveforms.append(wav)
        print(f"  loaded [{key}] {path}")

    results = {spec.family: {} for spec in quantizer_specs}
    current_family = None
    for spec in quantizer_specs:
        if spec.family != current_family:
            current_family = spec.family
            print(f"\n=== {current_family.title()} sweep ===")

        quantizer = spec.make_quantizer().to(device)
        metrics = [eval_one(w, sinc, quantizer, device) for w in waveforms]
        results[spec.family][spec.config] = {
            "sdr":    np.mean([m["sdr"]    for m in metrics]),
            "si_sdr": np.mean([m["si_sdr"] for m in metrics]),
            "label": spec.label,
        }
        result = results[spec.family][spec.config]
        print(f"  {spec.label:14s}  SDR={result['sdr']:.2f} dB  SI-SDR={result['si_sdr']:.2f} dB")

    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_heatmaps(results: dict, n_bins: int, scale:str, metric: str = "si_sdr"):
    """
    Produces two figures:
      1. 2D heatmap of polar (q_mag x q_phi)
      2. Optional 2D heatmap of predictive polar (q_mag x q_phi)
      3. Bar chart comparing cartesian diagonal vs best polar/predictive at same total bits
    """
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    assert metric in ("sdr", "si_sdr")
    label = "SI-SDR (dB)" if metric == "si_sdr" else "SDR (dB)"

    polar = results.get("polar", {})
    predictive = results.get("predictive", {})
    demodulated = results.get("demodulated", {})
    cart  = results.get("cartesian", {})

    heatmap_specs = [
        ("Polar", polar),
        ("Predictive polar", predictive),
        ("Demodulated polar", demodulated),
    ]
    heatmaps = []
    for title, family_results in heatmap_specs:
        if not family_results:
            continue
        q_mags = sorted(set(k[0] for k in family_results))
        q_phis = sorted(set(k[1] for k in family_results))
        values = np.array([[family_results[(qm, qp)][metric] for qp in q_phis] for qm in q_mags])
        heatmaps.append((title, q_mags, q_phis, values))

    n_plots = len(heatmaps) + 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    axes = np.atleast_1d(axes)
    fig.suptitle(f"Quantizer comparison — {label} | scale ={scale} | bins={n_bins}", fontsize=13)

    for ax, (title, q_mags, q_phis, values) in zip(axes, heatmaps):
        im = ax.imshow(values, aspect="auto", origin="lower", cmap="RdYlGn")
        ax.set_xticks(range(len(q_phis)));  ax.set_xticklabels(q_phis)
        ax.set_yticks(range(len(q_mags)));  ax.set_yticklabels(q_mags)
        ax.set_xlabel("q_phi (phase bits)")
        ax.set_ylabel("q_mag (magnitude bits)")
        ax.set_title(f"{title} — heatmap")
        plt.colorbar(im, ax=ax, label=label)

        for i, qm in enumerate(q_mags):
            for j, qp in enumerate(q_phis):
                val = values[i, j]
                ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                        fontsize=8, color="black")

    # ---- figure 2: iso-budget comparison ----
    # for each total bit budget B = q_mag + q_phi, find best polar/predictive/demodulated config
    # and compare against cartesian with q = B/2.
    ax2 = axes[-1]

    budgets = sorted(
        set(k[0] + k[1] for k in polar)
        | set(k[0] + k[1] for k in predictive)
        | set(k[0] + k[1] for k in demodulated)
    )
    polar_best = []
    predictive_best = []
    demodulated_best = []
    cart_vals  = []
    x_labels   = []

    for B in budgets:
        configs = [(qm, qp) for (qm, qp) in polar if qm + qp == B]
        best_val = max((polar[(qm, qp)][metric] for (qm, qp) in configs), default=float("nan"))
        best_cfg = max(configs, key=lambda k: polar[k][metric]) if configs else None
        polar_best.append(best_val)

        pred_configs = [(qm, qp) for (qm, qp) in predictive if qm + qp == B]
        pred_val = max((predictive[(qm, qp)][metric] for (qm, qp) in pred_configs), default=float("nan"))
        predictive_best.append(pred_val)

        demod_configs = [(qm, qp) for (qm, qp) in demodulated if qm + qp == B]
        demod_val = max((demodulated[(qm, qp)][metric] for (qm, qp) in demod_configs), default=float("nan"))
        demodulated_best.append(demod_val)

        # cartesian diagonal at q = B//2
        q = B // 2
        c_val = cart.get((q, q), {}).get(metric, float("nan"))
        cart_vals.append(c_val)
        cfg_label = f"({best_cfg[0]},{best_cfg[1]})" if best_cfg else "n/a"
        x_labels.append(f"B={B}\nbest polar\n{cfg_label}\nvs cart q={q}")

    x = np.arange(len(budgets))
    has_demod = bool(demodulated)
    w = 0.2 if has_demod and predictive else 0.25 if predictive or has_demod else 0.35
    ax2.bar(x - w, polar_best, w, label="Polar (best config)", color="steelblue")
    if predictive:
        ax2.bar(x, predictive_best, w, label="Predictive (best config)", color="seagreen")
    if has_demod:
        demod_x = x + w if predictive else x
        ax2.bar(demod_x, demodulated_best, w, label="Demodulated (best config)", color="mediumpurple")
    cart_x = x + (2 * w if has_demod and predictive else w)
    ax2.bar(cart_x, cart_vals, w, label="Cartesian (q=B/2)", color="tomato")
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels, fontsize=7)
    ax2.set_ylabel(label)
    ax2.set_title("Polar best vs Cartesian — same bit budget")
    ax2.legend()
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))
    ax2.grid(axis="y", alpha=0.3)

    out_path = f"quant_benchmark_{scale}{n_bins}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"saved {out_path}")




if __name__ == "__main__":
    from datasets.utils.waveform import WaveformLoader

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SAMPLE_RATE = 16000
    audio_loader = WaveformLoader(sample_rate=SAMPLE_RATE)
    weights_folder = "pretrained"

    audio_paths = {
        1:"audio/invertibility/p232_001.wav",
        2:"audio/invertibility/p232_002.wav",
        3:"audio/invertibility/15033000.mp3",
        4:"audio/invertibility/16366200.mp3",
        5:"audio/invertibility/16129994.mp3",
        6:"audio/invertibility/16176213.mp3",
    }

    for n_bins in [128, 256, 512]:
        for scale in ["mel", "lin"]:
            params = {
                "fs": SAMPLE_RATE,
                "fps": 128,
                "n_bins": n_bins,
                "scale": scale,
                "component": "complex",
                "causal": False,
                "apply_sinc_envelope": False,
                "decoder_type": "fast",
            }

            sinc : SincNet = (
                SincNet(**params)
                .load_pretrained_weights(weights_folder=weights_folder, verbose=True)
                .eval()
                .to(device)
            )

            results = run_benchmark(
                audio_paths, sinc, audio_loader, device,
                q_bits_cartesian=[4, 6, 8, 10],
                q_mag_polar=[4, 6, 8],
                q_phi_polar=[2, 4, 6, 8],
            )
            plot_heatmaps(results, n_bins=n_bins, scale=scale, metric="si_sdr")
