"""Prepare constrained intervals and waveform coverage on the note model's clock."""

from dataclasses import asdict, replace

import numpy as np

from ..audio import audio_info, load_audio
from ..timing import align_tiers, audio_warning, covered_frames
from ..types import Issue


def prepare_input(annotation, syllables, features):
    info = audio_info(annotation.sample.audio, require_mono=True)
    warning = audio_warning(annotation, info["clip_seconds"])
    issues = [warning] if warning else []
    rate, step = features.audio_sample_rate, features.timestep
    duration = annotation.duration if syllables is not None else info["clip_seconds"]
    frames = covered_frames(duration, step, info["clip_seconds"])
    provenance = dict(annotation.provenance, game_grid={
        "hop_size": features.hop_size, "sample_rate": rate,
    })
    annotation = replace(annotation, provenance=provenance)
    if syllables is not None:
        if not syllables:
            raise ValueError("Note inference requires at least one syllable.")
        required = {"phones"} if "imported_ph_num" in provenance else {"phones", "words"}
        tiers = {k: v for k, v in annotation.tiers.items() if k in required}
        def project(layers):
            return align_tiers(replace(annotation, tiers=layers), syllables, timestep=step, end_frame=frames)

        projected = project(tiers)
        # Admit auxiliary layers only when their geometry can coexist with the constraints.
        for name in ("texts", "words"):
            if name in required or name not in annotation.tiers:
                continue
            tier = annotation.tiers[name]
            try:
                if any(not i.label.strip() for i in tier):
                    raise ValueError("Empty label.")
                projected = project({**tiers, name: tier})
            except ValueError as exc:
                issues.append(Issue("unusable_auxiliary_tier", f"{name}: {exc}", "game", "warning"))
            else:
                tiers[name] = tier
        annotation = projected
        phones = annotation.tiers["phones"]
        syllables = [replace(s, start=phones[s.first_phone].start,
                             end=phones[s.first_phone + s.phone_count - 1].end) for s in syllables]
        if "imported_ph_num" in provenance:
            provenance.update(imported_phone_labels=[p.label for p in phones],
                              imported_ph_num=[s.phone_count for s in syllables])
        durations = [s.duration for s in syllables]
        provenance["syllable_input"] = dict(provenance.get("syllable_input", {}),
                                            syllables=[asdict(s) for s in syllables])
    else:
        annotation = replace(annotation, duration=frames * step)
        durations = [annotation.duration]
    waveform, _ = load_audio(annotation.sample.audio, rate)
    required = frames * features.hop_size + features.fft_size - features.win_size
    padding = max(0, required - len(waveform))
    if padding:
        waveform = np.pad(waveform, (0, padding))
    return annotation, durations, waveform, padding, issues
