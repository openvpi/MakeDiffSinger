import numpy as np
import parselmouth


def norm_f0(f0):
    f0 = np.log2(f0)
    return f0


def denorm_f0(f0, uv, pitch_padding=None):
    f0 = 2 ** f0
    if uv is not None:
        f0[uv > 0] = 0
    if pitch_padding is not None:
        f0[pitch_padding] = 0
    return f0


def interp_f0(f0, uv=None):
    if uv is None:
        uv = f0 == 0
    f0 = norm_f0(f0)
    if sum(uv) == len(f0):
        f0[uv] = -np.inf
    elif sum(uv) > 0:
        f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
    return denorm_f0(f0, uv=None), uv


def get_pitch_parselmouth(wav_data, hop_size, audio_sample_rate, interp_uv=True):
    time_step = hop_size / audio_sample_rate
    f0_min = 65
    f0_max = 800

    # noinspection PyArgumentList
    f0 = (
        parselmouth.Sound(wav_data, sampling_frequency=audio_sample_rate)
        .to_pitch_ac(
            time_step=time_step, voicing_threshold=0.6, pitch_floor=f0_min, pitch_ceiling=f0_max
        )
        .selected_array["frequency"]
    )
    uv = f0 == 0
    if interp_uv:
        f0, uv = interp_f0(f0, uv)
    return time_step, f0, uv

