"""DiffSinger CSV and audio layout compiled from current annotation assets."""

from collections import defaultdict
from dataclasses import replace
import csv
from pathlib import Path

from ..audio import audio_info, replay_audio
from ..catalog import sample_dataset
from ..phonology import current_phone_counts, derive_syllables
from ..review import quality_review_directory, sync_quality_review
from ..selection import Selected, select
from ..types import check_identity
from ..validation import note_slurs, require_valid
from ..timing import DEFAULT_GAME_GRID, align_tiers, audio_warning, covered_frames


def transcription(annotation, profiles=None, *, game_grid=None, audio_seconds=None):
    used = {"phones", "notes"}
    if current_phone_counts(annotation) is None:
        used.add("words")
    view = replace(annotation, tiers={k: v for k, v in annotation.tiers.items() if k in used},
                   stale=[name for name in annotation.stale if name in used])
    require_valid(view, required=("phones", "notes"), profiles=profiles)
    syllables = derive_syllables(view, profiles)
    grid = game_grid or annotation.provenance.get("game_grid") or DEFAULT_GAME_GRID
    step = grid["hop_size"] / grid["sample_rate"]
    annotation = align_tiers(view, syllables, timestep=step,
                             end_frame=covered_frames(view.duration, step, audio_seconds))
    phones = annotation.tiers["phones"]
    if "#" in annotation.sample.id or ":" in annotation.sample.id:
        raise ValueError("DiffSinger reserves '#' and ':' in names; supply an explicit output ID mapping.")
    syllables = [replace(s, start=phones[s.first_phone].start,
                         end=phones[s.first_phone + s.phone_count - 1].end) for s in syllables]
    notes = annotation.tiers["notes"]
    if all(note.label == "rest" for note in notes):
        raise ValueError(f"DiffSinger pitch data needs at least one pitched note: {annotation.key}. "
                         "Exclude this item from the training export and keep its editable annotation.")
    return {
        "name": annotation.sample.id,
        "ph_seq": " ".join(i.label for i in phones),
        "ph_dur": " ".join(format(i.duration, ".17g") for i in phones),
        "ph_num": " ".join(str(s.phone_count) for s in syllables),
        "note_seq": " ".join(i.label for i in notes),
        "note_dur": " ".join(format(i.duration, ".17g") for i in notes),
        "note_slur": " ".join(map(str, note_slurs(annotation, syllables))),
    }


def _subset_path(sample, speakers):
    language = (sample.language or "und").lower()
    speaker = speakers.get((sample.speaker or "").casefold(), "unknown") if len(speakers) > 1 else None
    parts = (speaker, language) if speaker else (language,)
    for part in parts:
        check_identity(part)
        if part != part.strip() or part.startswith(".") or part.endswith(".") or any(c in '<>:"|?*' for c in part):
            raise ValueError(f"Invalid DiffSinger subset component: {part!r}")
    return Path(*parts)


def _unique_row_names(items):
    reserved = {row["name"].casefold() for _, row in items}
    used = set()
    for _, row in items:
        stem = name = row["name"]
        if name.casefold() in used:
            serial = 2
            name = f"{stem}_{serial}"
            while name.casefold() in reserved or name.casefold() in used:
                serial += 1
                name = f"{stem}_{serial}"
        row["name"] = name
        used.add(name.casefold())


def export_diffsinger(annotations, output, *, profiles=None, copy_audio=True, criteria=None, review_output=None,
                      game_grid=None):
    if isinstance(annotations, Selected):
        if criteria is not None:
            raise ValueError("Pass selection criteria with the source annotations, or pass a Selected result.")
        source, selected = annotations.annotations, annotations
        sample_dataset(a.sample for a in source)
    else:
        source = list(annotations)
        selected = select(source, criteria)
    speakers = {}
    for annotation in source:
        if annotation.sample.speaker and annotation.sample.speaker.casefold() != "unknown":
            name = annotation.sample.speaker
            speakers.setdefault(name.casefold(), name)
    if not selected.annotations and not selected.quality_removed:
        raise ValueError("The export selection is empty.")
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty export directory: {output}")
    rows = defaultdict(list)
    failed = []
    warnings = []
    seen = set()
    for annotation in selected.annotations:
        if annotation.key in seen:
            raise ValueError(f"Duplicate output identity: {annotation.key}")
        seen.add(annotation.key)
        try:
            seconds = None
            audio = annotation.sample.audio
            if copy_audio or audio.waveform is not None or (audio.path and Path(audio.path).is_file()):
                # CSV-only conversion can still use the source duration when it is readable.
                try:
                    seconds = audio_info(audio, require_mono=copy_audio)["clip_seconds"]
                except (ValueError, OSError, RuntimeError):
                    if copy_audio:
                        raise
            if seconds is not None:
                if issue := audio_warning(annotation, seconds):
                    warnings.append({"key": annotation.key, "message": issue.message})
            row = transcription(annotation, profiles, game_grid=game_grid, audio_seconds=seconds)
            rows[_subset_path(annotation.sample, speakers)].append((annotation, row))
        except (ValueError, OSError, RuntimeError) as exc:
            failed.append({"key": annotation.key, "reason": str(exc)})
    if selected.quality_filter_applied:
        review = Path(review_output).resolve() if review_output else quality_review_directory(output)
        if review.is_relative_to(output):
            raise ValueError("Choose a quality review directory outside the training export.")
        sync_quality_review(selected, review)
    output.mkdir(parents=True, exist_ok=True)
    entries = []
    for subset, items in sorted(rows.items()):
        _unique_row_names(items)
        leaf = output / subset
        leaf.mkdir(parents=True, exist_ok=True)
        if copy_audio:
            (leaf / "wavs").mkdir(exist_ok=True)
        written_rows = []
        for annotation, row in items:
            if copy_audio:
                destination = leaf / "wavs" / (row["name"] + ".wav")
                try:
                    replay_audio(annotation.sample.audio, destination)
                except (ValueError, FileNotFoundError) as exc:
                    destination.unlink(missing_ok=True)
                    failed.append({"key": annotation.key, "reason": str(exc)})
                    continue
            written_rows.append(row)
            entries.append({"key": annotation.key, "name": row["name"],
                            "csv": (subset / "transcriptions.csv").as_posix(),
                            "audio": (subset / "wavs" / (row["name"] + ".wav")).as_posix() if copy_audio else None,
                            "speaker": annotation.sample.speaker, "language": annotation.sample.language,
                            "source": annotation.sample.audio.path})
        with (leaf / "transcriptions.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(items[0][1]))
            writer.writeheader()
            writer.writerows(written_rows)
    manifest = {"selected": entries, "failed": failed, "warnings": warnings, "excluded": selected.excluded, "criteria": selected.criteria,
                "selection_details": selected.details}
    return manifest
