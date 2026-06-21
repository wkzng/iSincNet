"""
Soft-archived: learned lin->scale adapter for SincNet.

The Adapter + AdaptedSincNet approach showed limited gains in practice
(the trained lin2mel checkpoint produced spectra still close to linear).
Kept here for reference and potential future experimentation.

Usage:
    from sincnet.adapter import AdaptedSincNet
    m = AdaptedSincNet(fs=16000, fps=128, n_bins=128, component="complex", causal=False, scale="mel")
    m.load_pretrained_weights("pretrained/", verbose=True)
"""

import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import replace

from .model import SincNet, Encoder1d


class Adapter(nn.Module):
    """ Learned, invertible-on-the-data linear adapter between two (B,C,F,T) representations.

        forward(x) : representation A -> representation B   (e.g. linear spectrogram -> mel-like)
        inverse(y) : representation B -> representation A   (learned inverse of forward, on the data)

        A pair of Conv1d over time (kernel_size taps) on the C*F channels, centre-tap
        identity-initialised so forward/inverse start as identity (round-trip exact at step 0).
        Generic: pair it with any target (mel, bark, erb, a learned target, ...). Temporal context
        (kernel_size >= 3) matters when the two representations are time-convolutions of the signal
        (two SincNet banks): the map between them is multi-frame, which a per-frame (kernel=1)
        adapter cannot represent (it plateaus, leaving cross-frame stripes).
    """
    def __init__(self, channels:int, kernel_size:int=3):
        super().__init__()
        pad = kernel_size // 2
        self.forward_conv = nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False)
        self.inverse_conv = nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False)
        with torch.no_grad():
            for conv in (self.forward_conv, self.inverse_conv):
                conv.weight.zero_()
                conv.weight[:, :, pad] = torch.eye(channels)   # centre-tap identity

    def _mix(self, conv:nn.Conv1d, x:torch.Tensor) -> torch.Tensor:
        B, C, Fr, T = x.shape
        return conv(x.reshape(B, C * Fr, T)).reshape(B, C, Fr, T)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """A -> B"""
        return self._mix(self.forward_conv, x)

    def inverse(self, y:torch.Tensor) -> torch.Tensor:
        """B -> A"""
        return self._mix(self.inverse_conv, y)


class AdaptedSincNet(SincNet):
    """ Linear SincNet (frozen invertible autoencoder) + a learned in-model adapter that maps the
        linear spectrogram to a target-scale representation (mel by default; any scale works).

            encode(w)  -> target-like features  (linear spectrogram through adapter.forward)
            decode(x)  -> waveform              (adapter.inverse, then the frozen linear decoder)
            forward(w) -> reconstruction; stashes the relative match loss in self.match_loss
            target_features(w) -> the in-model fixed target bank (the match target)

        Load the pretrained LINEAR autoencoder, then freeze_autoencoder(): encoder+decoder freeze
        and only the adapter learns.
    """
    def __init__(self, fs:int=16000, fps:int=128, n_bins:int=128, component:str="complex",
                 q_bits:int=8, causal:bool=False, scale:str="mel", adapter_kernel:int=3, decoder_type:str="fast"):
        super().__init__(fs=fs, fps=fps, scale="lin", component=component, n_bins=n_bins, q_bits=q_bits, causal=causal, decoder_type=decoder_type)

        channels = (2 if component == "complex" else 1) * n_bins
        self.name = self.config.model_id.replace(f"_{self.config.scale}_", f"_lin2{scale}_")
        self.adapter = Adapter(channels=channels, kernel_size=adapter_kernel)

        self.target_bank = Encoder1d(replace(self.config, scale=scale))
        for p in self.target_bank.parameters():
            p.requires_grad = False

    def encode(self, x:torch.Tensor) -> torch.Tensor:
        """linear spectrogram -> target spectrogram"""
        return self.adapter(self.encoder(x))

    def decode(self, x:torch.Tensor) -> torch.Tensor:
        """target spectrogram -> waveform (adapter inverse, then the linear decoder)"""
        return super().decode(self.adapter.inverse(x))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """ roundtrip loss: w -> lin -> targ -> lin -> wav"""
        with torch.no_grad():
            spec_targ = self.target_bank(x)

        spec_pred = self.encode(x)
        wav_pred = self.decode(spec_pred)
        match_loss = F.mse_loss(spec_pred, spec_targ) / (spec_targ.pow(2).mean() + 1e-8)
        return {"spec_pred": spec_pred, "spec_targ": spec_targ, "wav_pred": wav_pred, "match_loss": match_loss}

    def load_pretrained_weights(self, weights_folder:str, freeze:bool=True, device:str="cpu", verbose:bool=False):
        """Load a full AdaptedSincNet checkpoint (encoder + adapter + target bank).

        Decoder keys are intentionally skipped: the fast/exact decoder buffers are derived
        from the encoder at init time (the checkpoint may have been saved with a different
        decoder_type)."""
        weights_path = os.path.join(weights_folder, f"{self.name}.ckpt")
        checkpoint = torch.load(weights_path, map_location=torch.device(device), weights_only=False)
        if verbose:
            print(f"Loading AdaptedSincNet: {weights_path}")
            print("EPOCH", checkpoint.get("epoch", "?"), "// NSTEP", checkpoint.get("n_steps", "?"))
        result = self.load_state_dict(checkpoint["state_dict"], strict=False)
        unexpected = [k for k in result.unexpected_keys if not k.startswith("decoder.")]
        missing    = [k for k in result.missing_keys    if not k.startswith("decoder.")]
        if unexpected:
            warnings.warn(f"load_pretrained_weights: unexpected keys: {unexpected}")
        if missing:
            warnings.warn(f"load_pretrained_weights: missing keys: {missing}")
        if verbose and not unexpected and not missing:
            print("All non-decoder keys loaded successfully.")
        for p in self.parameters():
            p.requires_grad = not freeze
        return self

    def load_linear_sincnet(self, weights_folder:str, device:str="cpu", verbose:bool=False):
        """ Bootstrap the FROZEN linear autoencoder from its pretrained checkpoint, then freeze it.

            Reads `{linear model_id}.ckpt` from `weights_folder` (the linear-scale SincNet, not this
            adapter's own checkpoint) into encoder+decoder (strict=False, since the adapter and
            target bank keys are new), and calls freeze_autoencoder() so only the adapter trains.
            The full trained adapter is saved/loaded separately under `self.name` via
            `load_pretrained_weights`.
        """
        weights_path = os.path.join(weights_folder, f"{self.config.model_id}.ckpt")
        checkpoint = torch.load(weights_path, map_location=torch.device(device))
        if verbose:
            print(f"Loading frozen linear SincNet: {weights_path}")
        self.load_state_dict(checkpoint["state_dict"], strict=False)
        self.freeze_autoencoder()
        return self
