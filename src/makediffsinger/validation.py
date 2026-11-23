"""Independent structural checks; quality scores are not correctness probabilities."""

import math
import re

from .timing import JOIN_TOLERANCE_MS, audio_warning, near, match_boundaries, phone_ownership, tier_edges, tolerance
from .audio import audio_info
from .phonology import PhoneProfiles, derive_syllables
from .types import Annotation, Issue

NOTE = re.compile(r"(?:rest|[A-G](?:#|b)?-?\d+(?:[+-]\d+)?)\Z")


def note_slurs(annotation: Annotation, syllables) -> list[int]:
    notes = annotation.tiers.get("notes", [])
    boundaries = [s.start for s in syllables] + [syllables[-1].end]
    ms = tolerance(annotation)
    indices = match_boundaries(boundaries, tier_edges(notes, annotation.duration, ms), ms)
    onsets = set(indices[:-1])
    return [int(i not in onsets) for i in range(len(notes))]


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
            if not near(item.start, cursor, JOIN_TOLERANCE_MS):
                issues.append(Issue("layer_coverage", f"{name}: gap or overlap at {cursor:.9f}."))
            if not item.label.strip():
                issues.append(Issue("empty_label", f"{name}: empty label at {item.start:.9f}."))
            if name == "notes" and not NOTE.fullmatch(item.label):
                issues.append(Issue("invalid_note", f"Unsupported ASCII note label: {item.label!r}."))
            cursor = item.end
        if not near(cursor, annotation.duration, tolerance(annotation)):
            issues.append(Issue("layer_endpoint", f"{name} ends at {cursor}, domain ends at {annotation.duration}."))
    phones, words = annotation.tiers.get("phones"), annotation.tiers.get("words")
    if (phones and words and annotation.provenance.get("groups_verified") is not False
            and "imported_ph_num" not in annotation.provenance):
        try:
            ms = tolerance(annotation)
            phone_ownership(tier_edges(words, annotation.duration, ms), tier_edges(phones, annotation.duration, ms))
        except ValueError as exc:
            issues.append(Issue("group_alignment", str(exc)))
    if annotation.tiers.get("notes") and not (ignore_stale and "notes" in annotation.stale):
        if profiles is not None or "imported_ph_num" in annotation.provenance:
            try:
                note_slurs(annotation, derive_syllables(annotation, profiles))
            except ValueError as exc:
                issues.append(Issue("note_alignment", str(exc)))
    for key in ("confidence", "agreement", "determinacy", "monotonicity"):
        value = annotation.quality.get(key)
        if value is not None and (not isinstance(value, (float, int)) or not math.isfinite(value)):
            issues.append(Issue("nonfinite_quality", f"{key} is unavailable.", severity="warning"))
    if annotation.stale and not ignore_stale:
        severity = "error" if set(annotation.stale) & {"texts", "words", "phones", "notes"} else "warning"
        issues.append(Issue("stale_result", f"Outdated results: {', '.join(annotation.stale)}.", severity=severity))
    if check_audio:
        try:
            info = audio_info(annotation.sample.audio, require_mono=True)
            if warning := audio_warning(annotation, info["clip_seconds"]):
                issues.append(warning)
        except (OSError, ValueError, ImportError, RuntimeError) as exc:
            issues.append(Issue("audio_unavailable", str(exc)))
    return issues


def require_valid(annotation: Annotation, **kwargs) -> None:
    errors = [i for i in validate(annotation, **kwargs) if i.severity == "error"]
    if errors:
        raise ValueError("; ".join(f"{i.code}: {i.message}" for i in errors))
