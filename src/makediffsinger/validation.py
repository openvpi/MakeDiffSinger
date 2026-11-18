"""Independent structural checks; quality scores are not correctness probabilities."""

import math
import re

from .annotations import TIME_EPS
from .audio import audio_info
from .phonology import PhoneProfiles, derive_syllables
from .types import Annotation, Issue

NOTE = re.compile(r"(?:rest|[A-G](?:#|b)?-?\d+(?:[+-]\d+)?)\Z")


def note_slurs(annotation: Annotation, syllables) -> list[int]:
    notes = annotation.tiers.get("notes", [])
    slurs, cursor = [], 0
    for syllable in syllables:
        count = 0
        while cursor < len(notes) and notes[cursor].start < syllable.end - TIME_EPS:
            note = notes[cursor]
            if note.start < syllable.start - TIME_EPS or note.end > syllable.end + TIME_EPS:
                raise ValueError(f"A note crosses the syllable boundary at {syllable.end:.9f}.")
            slurs.append(int(count > 0))
            count += 1
            cursor += 1
        if not count:
            raise ValueError("A syllable has no corresponding note interval.")
    if cursor != len(notes):
        raise ValueError("Extra notes extend beyond the syllable sequence.")
    return slurs


def validate(annotation: Annotation, *, required=(), profiles: PhoneProfiles | None = None,
             check_audio=False, ignore_stale=False) -> list[Issue]:
    issues = []
    if not math.isfinite(annotation.duration) or annotation.duration <= 0:
        return [Issue("invalid_domain", "Annotation duration must be finite and positive.")]
    for name in required:
        if not annotation.tiers.get(name):
            issues.append(Issue("missing_layer", f"Missing {name} layer."))
    for name, intervals in annotation.tiers.items():
        cursor = 0.0
        if not intervals:
            issues.append(Issue("empty_layer", f"{name} contains no intervals."))
        for item in intervals:
            if not all(math.isfinite(x) for x in (item.start, item.end)) or item.end <= item.start:
                issues.append(Issue("invalid_interval", f"{name}: {item}"))
                continue
            if abs(item.start - cursor) > TIME_EPS:
                issues.append(Issue("layer_coverage", f"{name}: gap or overlap at {cursor:.9f}."))
            if not item.label.strip():
                issues.append(Issue("empty_label", f"{name}: empty label at {item.start:.9f}."))
            if name == "notes" and not NOTE.fullmatch(item.label):
                issues.append(Issue("invalid_note", f"Unsupported ASCII note label: {item.label!r}."))
            cursor = item.end
        if abs(cursor - annotation.duration) > TIME_EPS:
            issues.append(Issue("layer_endpoint", f"{name} ends at {cursor}, domain ends at {annotation.duration}."))
    phones, words = annotation.tiers.get("phones"), annotation.tiers.get("words")
    if phones and words:
        starts, ends = [i.start for i in phones], [i.end for i in phones]
        for word in words:
            if min(abs(word.start - t) for t in starts) > TIME_EPS or min(abs(word.end - t) for t in ends) > TIME_EPS:
                issues.append(Issue("group_alignment", "A pronunciation-group boundary is not a phone boundary."))
    if annotation.tiers.get("notes") and not (ignore_stale and "notes" in annotation.stale):
        if profiles is not None or "imported_ph_num" in annotation.provenance:
            try:
                note_slurs(annotation, derive_syllables(annotation, profiles))
            except ValueError as exc:
                issues.append(Issue("note_alignment", str(exc)))
    for key in ("confidence", "agreement", "determinacy", "monotonicity"):
        value = annotation.quality.get(key)
        if value is not None and not math.isfinite(value):
            issues.append(Issue("nonfinite_quality", f"{key} is not finite."))
    if annotation.stale and not ignore_stale:
        severity = "error" if set(annotation.stale) & {"texts", "words", "phones", "notes"} else "warning"
        issues.append(Issue("stale_result", f"Outdated results: {', '.join(annotation.stale)}.", severity=severity))
    if check_audio:
        try:
            info = audio_info(annotation.sample.audio, require_mono=True)
            tolerance = annotation.provenance.get("time_resolution_seconds", 0.0) + 1 / info["sample_rate"]
            if abs(annotation.duration - info["clip_seconds"]) > tolerance:
                issues.append(Issue("audio_endpoint", "Audio/annotation duration difference exceeds the recorded time resolution."))
        except (OSError, ValueError, ImportError, RuntimeError) as exc:
            issues.append(Issue("audio_unavailable", str(exc)))
    return issues


def require_valid(annotation: Annotation, **kwargs) -> None:
    errors = [i for i in validate(annotation, **kwargs) if i.severity == "error"]
    if errors:
        raise ValueError("; ".join(f"{i.code}: {i.message}" for i in errors))
