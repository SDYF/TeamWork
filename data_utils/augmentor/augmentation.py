import numpy as np


def augmet(x, mode=None):
    if mode is None:
        pass
    elif mode == "volume":
        return VolumePerturbAugmentor(**params)
    elif mode == "shift":
        return ShiftPerturbAugmentor(**params)
    elif mode == "speed":
        return SpeedPerturbAugmentor(**params)
    elif mode == "resample":
        return ResampleAugmentor(**params)
    elif mode == "noise":
        return NoisePerturbAugmentor(min_snr_dB=10, max_snr_dB=50)
    elif mode == "specaug":
        return SpecAugmentor(**params)
    elif mode == "specsub":
        return SpecSubAugmentor(**params)
