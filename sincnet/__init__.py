from .cgdecoder import frame_pseudo_inverse
from .model import SincNet, Decoder1d, AnalyticDecoder1d, FastAnalyticDecoder1d
from .stft import STFT, stft_params


def create_spectrogram_transform(params: dict) -> STFT | SincNet:
    """Build a frontend whose encode output is real (B, 2, n_bins, T).

    Required spec_type is 'stft' or 'sincnet'. Shared options are fs, fps,
    n_bins, q_bits and mulaw_kwargs;
    options belonging to the other backend are ignored. Quantization defaults
    to 6-bit polar with pre-scaling disabled for either backend.
    Unknown options and incompatible component/layout choices are rejected.
    For mono (B, L) or (B, 1, L), both use T = ceil(L / hop), hop = fs // fps.
    STFT centers are shifted by floor(hop/2) samples relative to SincNet.
    STFT also requires overlap * (fs // fps) <= 2 * n_bins.
    The input dictionary is never modified.
    """
    options = dict(params)
    spec_type = options.pop("spec_type", None)
    if spec_type not in ("stft", "sincnet"):
        raise ValueError("spec_type must be 'stft' or 'sincnet'")

    shared = {"fs", "fps", "n_bins", "q_bits", "mulaw_kwargs"}
    stft_options = {"overlap", "layout"}
    sincnet_options = {
        "scale", "component", "causal", "apply_sinc_envelope",
        "decoder_type", "cg_iters",
    }
    unknown = options.keys() - (shared | stft_options | sincnet_options)
    if unknown:
        raise TypeError(f"Unknown spectrogram parameters: {', '.join(sorted(unknown))}")
    if options.get("component", "complex") != "complex":
        raise ValueError("The factory requires component='complex' for two output channels")
    if options.get("layout", "channel") != "channel":
        raise ValueError("The factory requires layout='channel' for (B, 2, F, T) output")
    fs, fps = options.get("fs", 16000), options.get("fps", 128)
    if fs <= 0 or fps <= 0 or fs % fps:
        raise ValueError("fs and fps must be positive and fps must divide fs")

    options.setdefault("q_bits", 6)
    quantizer_options = dict(options.get("mulaw_kwargs") or {})
    quantizer_options.setdefault("coordinate_system", "polar")
    quantizer_options.setdefault("pre_scaling", False)
    options["mulaw_kwargs"] = quantizer_options

    if spec_type == "stft":
        return STFT(**{key: value for key, value in options.items() if key in shared | stft_options})

    kwargs = {key: value for key, value in options.items() if key in shared | sincnet_options}
    kwargs["component"] = "complex"
    return SincNet(**kwargs)
