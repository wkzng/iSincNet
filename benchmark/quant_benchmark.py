"""
quant_benchmark.py
------------------
Compares CartesianMuLawQuant vs PolarMuLawQuant across bit-depth configurations.

Metrics: SDR and SI-SDR (both in dB, higher = better).

Usage (drop into your notebook):
    from quant_benchmark import run_benchmark, plot_heatmaps
    results = run_benchmark(audio_paths, sinc, audio_loader, device)
    plot_heatmaps(results)
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from itertools import product

from sincnet.model import SincNet
from sincnet.mulaw import MuLawQuant, PolarMuLawQuant
from datasets.utils.waveform import WaveformLoader


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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def sdr(ref: torch.Tensor, est: torch.Tensor, eps: float = 1e-8) -> float:
    """Signal-to-Distortion Ratio in dB.  ref, est ~ (samples,) or (C, samples)"""
    ref = ref.flatten().float()
    est = est.flatten().float()
    noise = est - ref
    return 10 * torch.log10((ref ** 2).sum() / (noise ** 2).sum() + eps).item()


def si_sdr(ref: torch.Tensor, est: torch.Tensor, eps: float = 1e-8) -> float:
    """Scale-Invariant SDR in dB."""
    ref = ref.flatten().float()
    est = est.flatten().float()
    ref = ref - ref.mean()
    est = est - est.mean()
    alpha = (est * ref).sum() / ((ref ** 2).sum() + eps)
    proj = alpha * ref
    noise = est - proj
    return 10 * torch.log10((proj ** 2).sum() / (noise ** 2).sum() + eps).item()


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
    q_bits_cartesian: list[int] = [4, 6, 8, 10],
    q_mag_polar: list[int]      = [4, 6, 8],
    q_phi_polar: list[int]      = [2, 4, 6, 8],
) -> dict:
    """
    Returns a results dict with structure:
        results["cartesian"][(q,q)]  = {"sdr": float, "si_sdr": float}
        results["polar"][(qm, qp)]   = {"sdr": float, "si_sdr": float}
    Values are means over all audio files.
    """

    # --- load all waveforms once ---
    waveforms = []
    for key, path in audio_paths.items():
        wav = audio_loader.load_segment(path, offset=offset, duration=duration, nchannels=1)
        loudness = audio_loader.measure_loudness(wav)
        wav = audio_loader.normalise_loudness(wav, loudness, target_lufs=target_lufs)
        waveforms.append(wav)
        print(f"  loaded [{key}] {path}")

    results = {"cartesian": {}, "polar": {}}

    # --- cartesian sweep ---
    print("\n=== Cartesian sweep ===")
    for q in q_bits_cartesian:
        quantizer = CartesianMuLawQuant(q_real=q, q_imag=q).to(device)
        metrics = [eval_one(w, sinc, quantizer, device) for w in waveforms]
        results["cartesian"][(q, q)] = {
            "sdr":    np.mean([m["sdr"]    for m in metrics]),
            "si_sdr": np.mean([m["si_sdr"] for m in metrics]),
        }
        print(f"  q={q:2d}  SDR={results['cartesian'][(q,q)]['sdr']:.2f} dB"
              f"  SI-SDR={results['cartesian'][(q,q)]['si_sdr']:.2f} dB")

    # --- polar sweep ---
    print("\n=== Polar sweep ===")
    for qm, qp in product(q_mag_polar, q_phi_polar):
        quantizer = PolarMuLawQuant(q_mag=qm, q_phi=qp).to(device)
        metrics = [eval_one(w, sinc, quantizer, device) for w in waveforms]
        results["polar"][(qm, qp)] = {
            "sdr":    np.mean([m["sdr"]    for m in metrics]),
            "si_sdr": np.mean([m["si_sdr"] for m in metrics]),
        }
        print(f"  q_mag={qm}  q_phi={qp}  SDR={results['polar'][(qm,qp)]['sdr']:.2f} dB"
              f"  SI-SDR={results['polar'][(qm,qp)]['si_sdr']:.2f} dB")

    return results


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_heatmaps(results: dict, n_bins: int, scale:str, metric: str = "si_sdr"):
    """
    Produces two figures:
      1. 2D heatmap of polar (q_mag x q_phi)
      2. Bar chart comparing cartesian diagonal vs best polar at same total bits
    """
    assert metric in ("sdr", "si_sdr")
    label = "SI-SDR (dB)" if metric == "si_sdr" else "SDR (dB)"

    polar = results["polar"]
    cart  = results["cartesian"]

    q_mags = sorted(set(k[0] for k in polar))
    q_phis = sorted(set(k[1] for k in polar))

    # ---- figure 1: polar heatmap ----
    mat = np.array([[polar[(qm, qp)][metric] for qp in q_phis] for qm in q_mags])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Quantizer comparison — {label} | scale ={scale} | bins={n_bins}", fontsize=13)

    ax = axes[0]
    im = ax.imshow(mat, aspect="auto", origin="lower", cmap="RdYlGn")
    ax.set_xticks(range(len(q_phis)));  ax.set_xticklabels(q_phis)
    ax.set_yticks(range(len(q_mags)));  ax.set_yticklabels(q_mags)
    ax.set_xlabel("q_phi (phase bits)")
    ax.set_ylabel("q_mag (magnitude bits)")
    ax.set_title("Polar — heatmap")
    plt.colorbar(im, ax=ax, label=label)

    # annotate cells
    for i, qm in enumerate(q_mags):
        for j, qp in enumerate(q_phis):
            val = mat[i, j]
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=8, color="black")

    # ---- figure 2: iso-budget comparison ----
    # for each total bit budget B = q_mag + q_phi, find best polar config
    # and compare against cartesian with q = B/2 (if integer)
    ax2 = axes[1]

    budgets = sorted(set(k[0] + k[1] for k in polar))
    polar_best = []
    cart_vals  = []
    x_labels   = []

    for B in budgets:
        configs = [(qm, qp) for (qm, qp) in polar if qm + qp == B]
        if not configs:
            continue
        best_val = max(polar[(qm, qp)][metric] for (qm, qp) in configs)
        best_cfg = max(configs, key=lambda k: polar[k][metric])
        polar_best.append(best_val)

        # cartesian diagonal at q = B//2
        q = B // 2
        c_val = cart.get((q, q), {}).get(metric, float("nan"))
        cart_vals.append(c_val)
        x_labels.append(f"B={B}\nbest polar\n({best_cfg[0]},{best_cfg[1]})\nvs cart q={q}")

    x = np.arange(len(budgets))
    w = 0.35
    ax2.bar(x - w/2, polar_best, w, label="Polar (best config)", color="steelblue")
    ax2.bar(x + w/2, cart_vals,  w, label="Cartesian (q=B/2)",   color="tomato")
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
