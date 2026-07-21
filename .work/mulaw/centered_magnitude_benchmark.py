"""
Compare max-scaled vs mean-centered magnitude normalization for polar mu-law.

This is intentionally a scratch experiment under .work/ so production quantizers
can stay stable while we inspect whether centered magnitude normalization helps.

Run from the repository root:
    python .work/mulaw/centered_magnitude_benchmark.py --duration 2 --q-mag 4 6 8 --q-phi 4 6
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.quant_benchmark import sdr, si_sdr
from datasets.utils.waveform import WaveformLoader
from sincnet.model import SincNet
from sincnet.mulaw import PolarMuLawQuant, dequantize_unit, quantize_unit


DEFAULT_AUDIO_PATHS = [
    "audio/invertibility/p232_001.wav",
    "audio/invertibility/p232_002.wav",
    "audio/invertibility/15033000.mp3",
    "audio/invertibility/16366200.mp3",
]


@dataclass(frozen=True)
class BenchmarkRow:
    strategy: str
    q_mag: int
    q_phi: int
    total_bits: int
    sdr: float
    si_sdr: float
    mag_sdr: float
    mag_mae: float
    mag_rmse: float


class CenteredMagnitudePolarQuant(PolarMuLawQuant):
    """
    Experimental polar quantizer with centered magnitude normalization.

    Magnitude side information is (mean, scale), where:
        mean = mean(magnitude)
        scale = max(abs(magnitude - mean)) + eps
        z = (magnitude - mean) / scale

    z is signed mu-law companded and quantized in [-1, 1]. Phase uses the same
    uniform tokenization as PolarMuLawQuant.
    """

    def _compand_signed_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.log1p(self.mu * torch.abs(x)) / self.log_mu

    def _expand_signed_magnitude(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sign(x) * torch.expm1(torch.abs(x) * self.log_mu) / self.mu

    @staticmethod
    def _squeeze_stat(x: torch.Tensor | float) -> torch.Tensor | float:
        if isinstance(x, torch.Tensor) and x.ndim == 4:
            return x.squeeze(1)
        return x

    def compand(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        self._validate_complex_input(x)
        real = x[:, 0]
        imag = x[:, 1]

        magnitude = torch.sqrt(real**2 + imag**2)
        mean = magnitude.mean(dim=[-2, -1], keepdim=True)
        scale = torch.amax(torch.abs(magnitude - mean), dim=[-2, -1], keepdim=True) + self.eps
        magnitude = self._compand_signed_magnitude((magnitude - mean) / scale)

        phase = torch.atan2(imag, real)
        phase = (phase + torch.pi) / (2 * torch.pi)

        stats = (mean.unsqueeze(1), scale.unsqueeze(1))
        return torch.stack([magnitude, phase], dim=1), stats

    def expand(
        self,
        x: torch.Tensor,
        stats: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        magnitude, phase = self._split_channels(x, "centered polar tensor")
        mean, scale = stats

        mean = self._squeeze_stat(mean)
        scale = self._squeeze_stat(scale)
        magnitude = self._expand_signed_magnitude(magnitude) * scale + mean
        magnitude = torch.clamp(magnitude, min=0.0)

        phase = phase * (2 * torch.pi) - torch.pi
        real = magnitude * torch.cos(phase)
        imag = magnitude * torch.sin(phase)
        return torch.stack([real, imag], dim=1)

    def quantize(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        x, stats = self.compand(x)
        magnitude, phase = self._split_channels(x, "centered polar tensor")
        mag_tokens = quantize_unit((magnitude + 1.0) / 2.0, self.mag_vocab_size, add_noise=self.dither)
        phi_tokens = quantize_unit(phase, self.phase_levels, add_noise=self.dither)
        return torch.stack([mag_tokens, phi_tokens], dim=1), stats

    def dequantize(
        self,
        x: torch.Tensor,
        stats: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        mag_tokens, phi_tokens = self._split_channels(x, "token tensor")
        magnitude = 2.0 * dequantize_unit(mag_tokens, self.mag_vocab_size) - 1.0
        phase = dequantize_unit(phi_tokens, self.phase_levels)
        return self.expand(torch.stack([magnitude, phase], dim=1), stats=stats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens, stats = self.quantize(x)
        return self.dequantize(tokens, stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", nargs="+", default=DEFAULT_AUDIO_PATHS)
    parser.add_argument("--out-dir", default=".work/mulaw/results")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--fps", type=int, default=128)
    parser.add_argument("--n-bins", type=int, default=128)
    parser.add_argument("--scale", choices=["lin", "mel", "bark", "erb"], default="lin")
    parser.add_argument("--offset", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--target-lufs", type=float, default=-23.0)
    parser.add_argument("--q-mag", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--q-phi", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-hist-points", type=int, default=300_000)
    return parser.parse_args()


def load_waveforms(args: argparse.Namespace) -> list[tuple[str, np.ndarray]]:
    loader = WaveformLoader(sample_rate=args.sample_rate)
    waveforms = []
    for path in args.audio:
        wav = loader.load_segment(path, offset=args.offset, duration=args.duration, nchannels=1)
        if wav is None:
            print(f"skip {path}: failed to load")
            continue
        loudness = loader.measure_loudness(wav)
        wav = loader.normalise_loudness(wav, loudness, target_lufs=args.target_lufs)
        waveforms.append((path, wav))
        print(f"loaded {path} {wav.shape}")
    if not waveforms:
        raise RuntimeError("No audio files could be loaded.")
    return waveforms


def build_sinc(args: argparse.Namespace) -> SincNet:
    sinc = SincNet(
        fs=args.sample_rate,
        fps=args.fps,
        n_bins=args.n_bins,
        scale=args.scale,
        component="complex",
        causal=False,
        apply_sinc_envelope=False,
        decoder_type="fast",
    )
    return sinc.eval().to(args.device)


def magnitude(spec: torch.Tensor) -> torch.Tensor:
    real = spec[:, 0]
    imag = spec[:, 1]
    return torch.sqrt(real**2 + imag**2)


@torch.no_grad()
def encode_waveforms(
    waveforms: list[tuple[str, np.ndarray]],
    sinc: SincNet,
    device: str,
) -> list[tuple[str, torch.Tensor, torch.Tensor]]:
    encoded = []
    for path, wav_np in waveforms:
        wav = torch.from_numpy(wav_np).to(device).float()
        spec = sinc.encode(wav)
        encoded.append((path, wav, spec))
    return encoded


@torch.no_grad()
def eval_quantizer(
    encoded: list[tuple[str, torch.Tensor, torch.Tensor]],
    sinc: SincNet,
    quantizer: torch.nn.Module,
) -> dict[str, float]:
    metrics = []
    for _, wav, spec in encoded:
        tokens, stats = quantizer.quantize(spec)
        spec_r = quantizer.dequantize(tokens, stats)
        wav_r = sinc.decode(spec_r)

        n = min(wav.shape[-1], wav_r.shape[-1])
        mag = magnitude(spec)
        mag_r = magnitude(spec_r)
        mag_error = mag_r - mag

        metrics.append(
            {
                "sdr": sdr(wav[..., :n], wav_r[..., :n]),
                "si_sdr": si_sdr(wav[..., :n], wav_r[..., :n]),
                "mag_sdr": sdr(mag, mag_r),
                "mag_mae": torch.mean(torch.abs(mag_error)).item(),
                "mag_rmse": torch.sqrt(torch.mean(mag_error**2)).item(),
            }
        )

    return {key: float(np.mean([row[key] for row in metrics])) for key in metrics[0]}


def run_sweep(encoded: list[tuple[str, torch.Tensor, torch.Tensor]], sinc: SincNet, args: argparse.Namespace) -> list[BenchmarkRow]:
    rows = []
    for q_mag in args.q_mag:
        for q_phi in args.q_phi:
            specs = [
                (
                    "max",
                    PolarMuLawQuant(q_mag=q_mag, q_phi=q_phi),
                ),
                (
                    "centered",
                    CenteredMagnitudePolarQuant(q_mag=q_mag, q_phi=q_phi),
                ),
            ]
            for strategy, quantizer in specs:
                quantizer = quantizer.to(args.device)
                metrics = eval_quantizer(encoded, sinc, quantizer)
                row = BenchmarkRow(
                    strategy=strategy,
                    q_mag=q_mag,
                    q_phi=q_phi,
                    total_bits=q_mag + q_phi,
                    **metrics,
                )
                rows.append(row)
                print(
                    f"{strategy:8s} q=({q_mag},{q_phi}) "
                    f"SI-SDR={row.si_sdr:7.2f} dB  "
                    f"SDR={row.sdr:7.2f} dB  "
                    f"mag-SDR={row.mag_sdr:7.2f} dB"
                )
    return rows


def write_csv(rows: list[BenchmarkRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(BenchmarkRow.__dataclass_fields__.keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def sample_normalized_magnitudes(
    encoded: list[tuple[str, torch.Tensor, torch.Tensor]],
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    max_values = []
    centered_values = []
    for _, _, spec in encoded:
        mag = magnitude(spec)
        max_scale = torch.amax(mag, dim=[-2, -1], keepdim=True).clamp_min(1e-8)
        mean = mag.mean(dim=[-2, -1], keepdim=True)
        centered_scale = torch.amax(torch.abs(mag - mean), dim=[-2, -1], keepdim=True).clamp_min(1e-8)
        max_values.append((mag / max_scale).detach().flatten().cpu())
        centered_values.append(((mag - mean) / centered_scale).detach().flatten().cpu())

    max_values = torch.cat(max_values).numpy()
    centered_values = torch.cat(centered_values).numpy()
    if max_values.size > max_points:
        rng = np.random.default_rng(0)
        idx = rng.choice(max_values.size, size=max_points, replace=False)
        max_values = max_values[idx]
        centered_values = centered_values[idx]
    return max_values, centered_values


def row_grid(rows: list[BenchmarkRow], strategy: str, metric: str) -> tuple[list[int], list[int], np.ndarray]:
    q_mags = sorted({row.q_mag for row in rows if row.strategy == strategy})
    q_phis = sorted({row.q_phi for row in rows if row.strategy == strategy})
    lookup = {(row.q_mag, row.q_phi): getattr(row, metric) for row in rows if row.strategy == strategy}
    values = np.array([[lookup[(qm, qp)] for qp in q_phis] for qm in q_mags])
    return q_mags, q_phis, values


def plot_heatmap_pair(rows: list[BenchmarkRow], metric: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    q_mags, q_phis, max_values = row_grid(rows, "max", metric)
    _, _, centered_values = row_grid(rows, "centered", metric)
    delta = centered_values - max_values

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, title, values, cmap in [
        (axes[0], "max scaled", max_values, "RdYlGn"),
        (axes[1], "centered", centered_values, "RdYlGn"),
        (axes[2], "centered - max", delta, "coolwarm"),
    ]:
        im = ax.imshow(values, aspect="auto", origin="lower", cmap=cmap)
        ax.set_xticks(range(len(q_phis)))
        ax.set_xticklabels(q_phis)
        ax.set_yticks(range(len(q_mags)))
        ax.set_yticklabels(q_mags)
        ax.set_xlabel("q_phi")
        ax.set_ylabel("q_mag")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8)

    fig.suptitle(metric)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_histograms(max_values: np.ndarray, centered_values: np.ndarray, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(max_values, bins=100, color="steelblue", alpha=0.85, log=True)
    axes[0].set_title("magnitude / max")
    axes[0].set_xlabel("normalized magnitude")
    axes[0].set_ylabel("count, log scale")

    axes[1].hist(centered_values, bins=100, color="darkorange", alpha=0.85, log=True)
    axes[1].set_title("(magnitude - mean) / max_abs_dev")
    axes[1].set_xlabel("centered normalized magnitude")
    axes[1].set_ylabel("count, log scale")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary(rows: list[BenchmarkRow], args: argparse.Namespace, out_path: Path) -> None:
    best = {}
    for metric in ["si_sdr", "sdr", "mag_sdr"]:
        best[metric] = {}
        for strategy in ["max", "centered"]:
            candidates = [row for row in rows if row.strategy == strategy]
            winner = max(candidates, key=lambda row: getattr(row, metric))
            best[metric][strategy] = winner.__dict__

    payload = {
        "config": {
            "sample_rate": args.sample_rate,
            "fps": args.fps,
            "n_bins": args.n_bins,
            "scale": args.scale,
            "duration": args.duration,
            "offset": args.offset,
            "audio": args.audio,
            "q_mag": args.q_mag,
            "q_phi": args.q_phi,
        },
        "note": "centered uses one extra mean side-info scalar per sample in addition to scale",
        "best": best,
    }
    out_path.write_text(json.dumps(payload, indent=2))


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(out_dir / ".matplotlib"))

    sinc = build_sinc(args)
    waveforms = load_waveforms(args)
    encoded = encode_waveforms(waveforms, sinc, args.device)

    max_values, centered_values = sample_normalized_magnitudes(encoded, args.max_hist_points)
    plot_histograms(max_values, centered_values, out_dir / "normalized_magnitude_hist.png")

    rows = run_sweep(encoded, sinc, args)
    write_csv(rows, out_dir / "centered_magnitude_results.csv")
    write_summary(rows, args, out_dir / "summary.json")
    plot_heatmap_pair(rows, "si_sdr", out_dir / "si_sdr_heatmaps.png")
    plot_heatmap_pair(rows, "mag_sdr", out_dir / "magnitude_sdr_heatmaps.png")

    print(f"\nwrote results to {out_dir}")
    print(f"csv: {out_dir / 'centered_magnitude_results.csv'}")
    print(f"summary: {out_dir / 'summary.json'}")
    print(f"plots: {out_dir / 'normalized_magnitude_hist.png'}")
    print(f"       {out_dir / 'si_sdr_heatmaps.png'}")
    print(f"       {out_dir / 'magnitude_sdr_heatmaps.png'}")


if __name__ == "__main__":
    main()
