import math

import torch

from benchmark.quant_benchmark import sdr, si_sdr


def test_sdr_and_si_sdr_are_finite_for_silent_inputs():
    ref = torch.zeros(16)
    est = torch.zeros(16)

    assert math.isfinite(sdr(ref, est))
    assert math.isfinite(si_sdr(ref, est))


def test_sdr_and_si_sdr_are_finite_for_perfect_reconstruction():
    ref = torch.linspace(-1.0, 1.0, steps=16)
    est = ref.clone()

    assert math.isfinite(sdr(ref, est))
    assert math.isfinite(si_sdr(ref, est))
    assert sdr(ref, est) > 0
    assert si_sdr(ref, est) > 0
