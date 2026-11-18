"""Explicit source mappings; importing labels never starts inference."""

import csv
from pathlib import Path

from .annotations import TIERS, tier_signatures
from .catalog import input_dataset
from .phonology import syllables_from_counts
from .storage import read_annotations
from .types import Annotation, AudioSource, BatchResult, Interval, Issue, Outcome, Sample


def import_intervals(sample: Sample, duration: float, tiers: dict, *, origin: str,
                     groups_verified=False, time_resolution=0.0) -> Annotation:
    """Adapters supply named interval lists after checking source semantics and units."""
    if not tiers or set(tiers) - set(TIERS):
        raise ValueError(f"Map source layers explicitly to {TIERS}; got {tuple(tiers)}.")
    layers = {name: [i if isinstance(i, Interval) else Interval(*i) for i in items]
              for name, items in tiers.items()}
    return Annotation(sample, duration, layers, provenance={
        "source_annotation": origin, "groups_verified": groups_verified,
        "time_resolution_seconds": time_resolution,
        "layer_origins": {name: origin for name in layers},
    })


def _intervals(labels, durations):
    if len(labels) != len(durations) or not labels:
        raise ValueError("Labels and durations must be nonempty sequences of the same length.")
    result, cursor = [], 0.0
    for label, duration in zip(labels, durations):
        if duration <= 0:
            raise ValueError("Nonpositive source durations need review; they are not fabricated into intervals.")
        result.append(Interval(cursor, cursor + duration, label))
        cursor += duration
    return result


def import_transcriptions(path, *, audio_root=None, language=None, speaker=None, subset="",
                          time_resolution=0.0) -> BatchResult:
    path = Path(path).resolve()
    dataset = str(input_dataset(path))
    audio_root = Path(audio_root).resolve() if audio_root else path.parent / "wavs"
    outcomes = []
    keys = set()
    with path.open(encoding="utf-8-sig", newline="") as stream:
        for line, row in enumerate(csv.DictReader(stream), start=2):
            name = row["name"]
            audio = next((audio_root / (name + ext) for ext in (".wav", ".flac")
                          if (audio_root / (name + ext)).is_file()), None)
            sample = Sample(name, AudioSource(str(audio)) if audio else AudioSource(),
                            language=language, speaker=speaker, subset=subset,
                            metadata={"dataset_root": dataset, "source_csv": str(path), "source_row": line})
            if sample.key in keys:
                raise ValueError(f"Duplicate source sample: {sample.key}")
            keys.add(sample.key)
            try:
                phones = _intervals(row["ph_seq"].split(), [float(v) for v in row["ph_dur"].split()])
                tiers = {"phones": phones}
                if row.get("note_seq") or row.get("note_dur"):
                    tiers["notes"] = _intervals(row["note_seq"].split(), [float(v) for v in row["note_dur"].split()])
                annotation = import_intervals(sample, phones[-1].end, tiers, origin=str(path),
                                              time_resolution=time_resolution)
                if row.get("ph_num"):
                    counts = [int(v) for v in row["ph_num"].split()]
                    syllables_from_counts(annotation, counts)
                    annotation.provenance.update(imported_ph_num=counts,
                                                 imported_phones_signature=tier_signatures({"phones": phones})["phones"])
                mapped = {"name", "ph_seq", "ph_dur", "ph_num", "note_seq", "note_dur", "note_slur"}
                annotation.provenance["unmapped_source_fields"] = sorted(set(row) - mapped)
                outcomes.append(Outcome(sample, "imported", annotation))
            except (KeyError, ValueError) as exc:
                outcomes.append(Outcome(sample, "failed", issues=[Issue("invalid_source_annotation", str(exc), "import")]))
    return BatchResult(outcomes)


def import_annotations(source, *, sample=None, audio_root=None, language=None, speaker=None,
                       subset="", tier_map=None, groups_verified=False, time_resolution=0.0):
    source = Path(source).resolve()
    if source.suffix.lower() == ".csv":
        return import_transcriptions(source, audio_root=audio_root, language=language,
                                     speaker=speaker, subset=subset, time_resolution=time_resolution)
    paths = [source] if source.is_file() else sorted(source.rglob("*.TextGrid"))
    outcomes = []
    base = source.parent if source.is_file() else source
    dataset = str(input_dataset(source))
    for path in paths:
        relative = path.relative_to(base)
        group = "" if str(relative.parent) == "." else relative.parent.as_posix()
        current = sample
        if current is None:
            directory = Path(audio_root) / relative.parent if audio_root else path.parent
            audio = next((directory / (path.stem + ext) for ext in (".wav", ".flac")
                          if (directory / (path.stem + ext)).is_file()), None)
            current = Sample(path.stem, AudioSource(str(audio)) if audio else AudioSource(),
                             language=language, speaker=speaker, subset="/".join(p for p in (subset, group) if p),
                             metadata={"dataset_root": dataset})
        try:
            annotation = read_annotations(path, sample=current, tier_map=tier_map)
            if groups_verified:
                annotation.provenance["groups_verified"] = True
            annotation.provenance.setdefault("time_resolution_seconds", time_resolution)
            outcomes.append(Outcome(current, "imported", annotation))
        except (OSError, ValueError) as exc:
            outcomes.append(Outcome(current, "failed", issues=[Issue("invalid_source_annotation", str(exc), "import")]))
    if not outcomes:
        raise ValueError("No annotations found; use a source-specific adapter for other input formats.")
    return BatchResult(outcomes)
