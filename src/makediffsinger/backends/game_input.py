"""Prepare finite source audio for a declared note-inference frame domain."""

import math

import numpy as np

from ..annotations import TIME_EPS
from ..audio import audio_info, load_audio


def prepare_audio(annotation, syllables, *, rate, timestep, hop_size, fft_size, win_size):
    if syllables is not None and not syllables:
        raise ValueError("Note inference requires at least one syllable.")
    info = audio_info(annotation.sample.audio, require_mono=True)
    resolution = annotation.provenance.get("time_resolution_seconds", 0.0)
    if not math.isfinite(resolution) or resolution < 0:
        raise ValueError("Annotation time resolution must be finite and nonnegative.")
    tolerance = resolution + 1 / info["sample_rate"]
    if abs(annotation.duration - info["clip_seconds"]) > tolerance:
        raise ValueError("Annotation/audio mismatch exceeds the declared frame resolution; review the source mapping.")
    cursor = 0.0
    previous_frame = 0
    for syllable in syllables or ():
        if abs(syllable.start - cursor) > TIME_EPS or syllable.end <= syllable.start:
            raise ValueError("Syllables must cover one continuous positive time domain.")
        frame = round(syllable.end / timestep)
        if abs(frame * timestep - syllable.end) > TIME_EPS or frame <= previous_frame:
            raise ValueError("A syllable boundary cannot be represented on the GAME frame grid.")
        cursor, previous_frame = syllable.end, frame
    if syllables is None:
        # GAME's native unconstrained input uses the whole clip as its only domain.
        previous_frame = round(annotation.duration / timestep)
        if previous_frame < 1:
            raise ValueError("The clip is shorter than one GAME frame.")
    elif abs(cursor - annotation.duration) > TIME_EPS:
        raise ValueError("Syllables do not cover the annotation domain.")
    waveform, _ = load_audio(annotation.sample.audio, rate)
    # GAME's feature extractor pads by win_size - hop_size before the FFT.
    context = fft_size - win_size
    required = previous_frame * hop_size + context
    padding = max(0, required - len(waveform))
    allowance = timestep / 2 + 1 / info["sample_rate"] if syllables is None else tolerance
    maximum = math.ceil(allowance * rate) + max(0, context) + 1
    if padding > maximum:
        raise ValueError("Required model padding exceeds the declared frame-resolution allowance.")
    if padding:
        waveform = np.pad(waveform, (0, padding))
    return waveform, padding
