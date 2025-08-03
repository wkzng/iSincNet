import os
import torch
import random


from torch_audiomentations import Compose, Gain, PolarityInversion, PitchShift, Shift, ApplyImpulseResponse, Identity, TimeInversion, AddColoredNoise
from torch_time_stretch import time_stretch


current_script_path = os.path.abspath(__file__)
current_script_dir = os.path.dirname(current_script_path)
REVERB_IR_FOLDER = os.path.join(current_script_dir, "audio_augmentations_kernels")



class TimeStrech(torch.nn.Module):

    def __init__(self, sample_rate:int, ratio:float):
        super(TimeStrech, self).__init__()
        self.sr = sample_rate
        self.ratio = ratio

    def forward(self, x:torch.Tensor):
        _, _, length = x.shape
        x = time_stretch(x, self.ratio, self.sr)

        if x.shape[2] < length:
            pad_length = (length - x.shape[2])
            x = torch.nn.functional.pad(x, (1, pad_length), mode='constant', value=0)
        return x[:,:,:length]



class AudioAugmenter:
    """ Generate random variations of the input waveform """

    def __init__(self, sr:int, p_mode:str="per_example", mode:str="per_example"):
        self.sr = sr
        T0 = Identity()

        T1 = Compose(
            transforms=[
                PolarityInversion(
                    p = 0.5,
                    mode = mode,
                    p_mode = p_mode
                ),
                PitchShift(
                    min_transpose_semitones = -3,
                    max_transpose_semitones = 3,
                    sample_rate = sr,
                    p = 1.0,
                    mode = mode
                ) if sr >= 16000 else Identity(),
                Gain(
                    min_gain_in_db = -15.0,
                    max_gain_in_db = 5.0,
                    p = 0.5,
                    mode = mode
                )
            ]
        )

        T2 = Compose(
            transforms=[
                PolarityInversion(
                    p = 0.5,
                    mode = mode,
                    p_mode = p_mode
                ),
                ApplyImpulseResponse(
                    ir_paths = REVERB_IR_FOLDER,
                    sample_rate = sr,
                    target_rate = sr,
                    p = 1.0,
                    p_mode = p_mode,
                    mode = mode
                ),
                Gain(
                    min_gain_in_db = -15.0,
                    max_gain_in_db = 5.0,
                    p = 0.5,
                    p_mode = p_mode,
                    mode = mode
                ),
            ]
        )

        # T3 = TimeInversion(
        #     p = 1.0,
        #     mode = mode,
        #     p_mode = p_mode,
        #     sample_rate = sr,
        #     target_rate = sr,
        # )

        T4 = Shift(
            p = 1.0,
            mode = mode,
            p_mode = p_mode,
            sample_rate = sr,
            target_rate = sr,
            min_shift = -0.2,
            max_shift = 0.2,
            shift_unit = "fraction",
            rollover = False,
        )

        T5 = TimeStrech(sample_rate=sr, ratio=1.05)
        T6 = TimeStrech(sample_rate=sr, ratio=0.95)

        T7 = AddColoredNoise(   
            sample_rate = sr,
            target_rate = sr,
            p = 1.0,
            mode = mode,
            p_mode = p_mode
        )

        self.transforms = [T0, T1, T2, T4, T5, T6, T7]
        #self.transforms = [T0, T1, T2, T4, T7]
        print("=========================================")
        print(f"Audio effects applicables:")
        print(self.transforms)
        print("=========================================")


    def get_stem_separation_transforms(self, mode:str="per_batch", p_mode:str="per_batch"):
        t1 = Compose(
            transforms=[
                PolarityInversion(
                    p = 0.5,
                    mode = mode,
                    p_mode = p_mode
                ),
                PitchShift(
                    min_transpose_semitones = -3,
                    max_transpose_semitones = 3,
                    sample_rate = self.sr,
                    p = 1.0,
                    mode = mode
                ) if self.sr >= 16000 else Identity()
            ]
        )
        t2 = TimeStrech(sample_rate=self.sr, ratio=1.05)
        t3 = TimeStrech(sample_rate=self.sr, ratio=0.95)
        return [t1, t2, t3]


    def get_random_transforms(self, k:int):
        return random.sample(self.transforms, k)
