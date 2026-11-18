"""Discover clip and transcript pairs for forced alignment and inspect unfamiliar sources."""

from collections import Counter
import csv
from dataclasses import replace
import json
from pathlib import Path
from xml.etree import ElementTree

from .annotations import read_textgrid
from .audio import audio_info
from .catalog import association_for, collection_root, dataset_root as bound_dataset
from .storage import read_slice_origin
from .types import AudioSource, Sample, Transcript

DEFAULT_AUDIO_EXTENSIONS = {".wav", ".flac", ".opus", ".mp3", ".aac", ".ogg"}
TRANSCRIPT_EXTENSIONS = (".pfml", ".txt", ".lab")


def working_pfml_path(audio: str | Path) -> Path | None:
    audio = Path(audio)
    for parent in audio.parents:
        if parent.name == "01-clips":
            return parent.parent / ".work" / "paired-lyrics" / audio.relative_to(parent).with_suffix(".pfml")
    return None


def pfml_written_text(content: str) -> str:
    """Extract written text without including pronunciation markup or readings."""
    def text_scope(element):
        parts = [element.text or ""]
        for child in element:
            if child.tag == "word":
                parts.append(child.get("text", child.text or ""))
            elif child.tag == "scope":
                parts.append(text_scope(child))
            parts.append(child.tail or "")
        return "".join(parts)

    try:
        root = ElementTree.fromstring(f"<fragment>{content}</fragment>")
    except ElementTree.ParseError as error:
        raise ValueError(f"Invalid PFML: {error}") from error
    return text_scope(root).strip()


def paired_transcript(audio: str | Path) -> Transcript | None:
    audio = Path(audio)
    for suffix in TRANSCRIPT_EXTENSIONS:
        path = audio.with_suffix(suffix)
        if path.is_file():
            content = path.read_text(encoding="utf-8").strip()
            if suffix == ".pfml":
                return Transcript(content, "pfml", str(path.resolve()))
            work = working_pfml_path(audio)
            if work is not None and work.is_file():
                override = work.read_text(encoding="utf-8").strip()
                if pfml_written_text(override) == content:
                    return Transcript(override, "pfml", str(work.resolve()))
            return Transcript(content, "text", str(path.resolve()))
    return None


def with_candidate_transcript(sample, path):
    """Stage one complete candidate on the original audio without changing its selected sidecar."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in TRANSCRIPT_EXTENSIONS:
        raise ValueError("The candidate must have a .pfml, .txt, or .lab extension.")
    content = path.read_text(encoding="utf-8-sig")
    metadata = {k: v for k, v in sample.metadata.items() if k not in {"transcript_selection", "transcript_read_error"}}
    return replace(sample, transcript=Transcript(content, "pfml" if suffix == ".pfml" else "text"), metadata=metadata)


def refresh_transcript(sample: Sample) -> Sample:
    selection = sample.metadata.get("transcript_selection")
    if selection in {"paired", "paired_with_candidate"} and sample.audio.path:
        from copy import deepcopy
        sample = deepcopy(sample)
        previous = sample.transcript
        paired = paired_transcript(sample.audio.path)
        if selection == "paired_with_candidate" and not (paired and paired.content.strip()):
            return sample
        sample.transcript = paired
        sample.metadata["transcript_selection"] = "paired"
        if previous is not None and previous != sample.transcript:
            sample.metadata["transcript_change"] = {"previous_content": previous.content,
                                                    "previous_format": previous.format, "previous_path": previous.path,
                                                    "reason": "paired transcript changed on disk"}
    elif sample.transcript and sample.transcript.path:
        from copy import deepcopy
        sample = deepcopy(sample)
        path = Path(sample.transcript.path)
        sample.transcript.content = path.read_text(encoding="utf-8").strip()
    return sample


def list_audio_files(root, extensions=None):
    root = Path(root).resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file()
                                                               and p.suffix.lower() in (extensions or DEFAULT_AUDIO_EXTENSIONS))
    if not files:
        raise ValueError(f"No input audio found under {root}.")
    return files


def discover_pairs(root: str | Path, *, language=None, speaker=None, subset="",
                   extended_languages=(), extensions=None, dataset_root=None) -> list[Sample]:
    root = Path(root).resolve()
    base = root.parent if root.is_file() else root
    dataset = Path(dataset_root).resolve() if dataset_root else bound_dataset(collection_root(base)) or base
    files = list_audio_files(root, extensions)
    samples, keys = [], set()
    for path in files:
        relative = path.relative_to(base)
        parent = "" if str(relative.parent) == "." else relative.parent.as_posix()
        _, association = association_for(path)
        sample = Sample(path.stem, AudioSource(str(path)), None, language or association.get("language"),
                        list(extended_languages or association.get("extended_languages", [])),
                        speaker or association.get("speaker"), "/".join(p for p in (subset, parent) if p),
                        {"dataset_root": str(dataset), "source_root": str(base), "source_relative_path": relative.as_posix(),
                         "transcript_selection": "paired"})
        try:
            sample.transcript = paired_transcript(path)
        except (OSError, UnicodeError, ValueError) as exc:
            sample.metadata["transcript_read_error"] = str(exc)
        if source := read_slice_origin(path):
            sample.metadata.update(origin_recording=source["origin_recording"], origin_slice=source["origin_slice"])
        if sample.key in keys:
            raise ValueError(f"Multiple audio files produce the same sample identity: {sample.key}")
        keys.add(sample.key)
        samples.append(sample)
    return samples


def inspect_dataset(root: str | Path, *, examples_per_format=3) -> dict:
    root = Path(root).resolve()
    base = root.parent if root.is_file() else root
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
    counts = Counter(p.suffix.lower() or "(none)" for p in files)
    examples, seen = [], Counter()
    for path in files:
        extension = path.suffix.lower()
        if seen[extension] >= examples_per_format:
            continue
        seen[extension] += 1
        entry = {"path": path.relative_to(base).as_posix(), "bytes": path.stat().st_size}
        try:
            if extension in DEFAULT_AUDIO_EXTENSIONS:
                entry["audio"] = dict(audio_info(AudioSource(str(path))), path=entry["path"])
                entry["transcript_candidates"] = [s for s in TRANSCRIPT_EXTENSIONS if path.with_suffix(s).is_file()]
            elif path.stat().st_size > 2_000_000:
                entry["detail"] = "Large source file; inspect a bounded sample with a source-specific adapter."
            elif extension in {".csv", ".tsv"}:
                with path.open(encoding="utf-8-sig", newline="") as stream:
                    reader = csv.DictReader(stream, delimiter="\t" if extension == ".tsv" else ",")
                    entry["columns"] = reader.fieldnames
                    entry["example"] = next(reader, None)
            elif extension in {".json", ".ds"}:
                data = json.loads(path.read_text(encoding="utf-8"))
                item = data[0] if isinstance(data, list) and data else data
                entry["json_type"] = type(data).__name__
                entry["keys"] = list(item) if isinstance(item, dict) else None
            elif extension == ".textgrid":
                import textgrid
                grid = textgrid.TextGrid()
                grid.read(str(path), round_digits=15)
                entry["tiers"] = [{"name": tier.name, "intervals": len(tier)} for tier in grid]
                entry["domain"] = [grid.minTime, grid.maxTime]
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            entry["problem"] = str(exc)
        examples.append(entry)
    return {"root": ".", "file_count": len(files), "extensions": dict(counts), "examples": examples}
