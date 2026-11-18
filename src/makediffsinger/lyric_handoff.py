"""Keep generated clip lyrics available between matching, alignment and review."""

from dataclasses import replace
from pathlib import Path

from .catalog import (bind_dataset, collection_root, dataset_root, edit_tables, path_reference,
                      portable_message, sample_dataset, table)
from .slicing import write_lyric_table
from .types import Transcript


def _audio_key(sample, root, source):
    audio = sample.audio
    if not audio.path:
        return None
    path = path_reference(audio.path, root, source)
    if audio.start_sample == 0 and audio.end_sample is None:
        return path
    end = "" if audio.end_sample is None else audio.end_sample
    return f"range:{audio.start_sample}:{end}:{path}"


def _file_version(path):
    try:
        stat = Path(path).stat()
    except OSError:
        return None
    return [stat.st_size, stat.st_mtime_ns]


def write_candidate_table(samples, records, output):
    """Write a working lyric table from match or ASR clip records, retaining the FA inputs."""
    samples = list(samples)
    root = collection_root(Path(output).parent)
    source = sample_dataset(samples)
    if source:
        bind_dataset(root, source)
    else:
        source = dataset_root(root)
    suggestions, evidence = {}, {}
    for sample in samples:
        record = records.get(sample.key, {})
        asr = record.get("asr_text", record.get("text", "")).strip()
        selected = record.get("transcript") or {}
        if selected.get("content", "").strip():
            candidate = Transcript(selected["content"], selected.get("format", "text"))
        else:
            candidate = Transcript((record.get("suggestion") or "").strip() or asr)
        if candidate.content.strip() and sample.transcript is None and not sample.metadata.get("transcript_read_error"):
            suggestions[sample.key] = candidate
        key = _audio_key(sample, root, source)
        if key is not None:
            row = {"version": _file_version(sample.audio.path), "asr_text": asr}
            if error := record.get("asr_error") or record.get("error"):
                row["asr_error"] = portable_message(error, root, source)
            if candidate.content.strip():
                row["candidate"] = {"content": candidate.content, "format": candidate.format}
            evidence[key] = row
    with edit_tables(root) as batch:
        write_lyric_table(samples, output, suggestions)
        path = root / ".work/lyric-candidates.json"
        for key, row in evidence.items():
            # A retry has its own table. It cannot replace the original before FA compares them.
            previous = batch.get(path, key)
            if not previous.get("candidate") or previous.get("version") != row["version"]:
                batch.set(path, key, row)
    return suggestions


def lyric_evidence(sample, root):
    rows = table(root / ".work/lyric-candidates.json")
    if not rows or not sample.audio.path:
        return {}
    source = dataset_root(root)
    original = sample.metadata.get("dataset_root")
    if source and original and Path(original).resolve() != source:
        return {}
    row = rows.get(_audio_key(sample, root, source), {})
    return row if row and row["version"] and row["version"] == _file_version(sample.audio.path) else {}


def alignment_inputs(samples, delivery):
    """Use a generated candidate for an unpaired clip without overwriting a sidecar."""
    root = collection_root(delivery)
    inputs = []
    for sample in samples:
        recovered = sample.metadata.get("transcript_selection") == "paired_with_candidate"
        empty = sample.transcript is None or not sample.transcript.content.strip()
        if (recovered or empty) and not sample.metadata.get("transcript_read_error"):
            candidate = lyric_evidence(sample, root).get("candidate")
            if candidate:
                metadata = sample.metadata.copy()
                if metadata.get("transcript_selection") in {"paired", "paired_with_candidate"}:
                    if sample.audio.start_sample == 0 and sample.audio.end_sample is None:
                        metadata["transcript_selection"] = "paired_with_candidate"
                    else:
                        metadata.pop("transcript_selection")
                sample = replace(sample, transcript=Transcript(**candidate), metadata=metadata)
            elif recovered:
                sample = replace(sample, transcript=None, metadata=dict(sample.metadata, transcript_selection="paired"))
        inputs.append(sample)
    return inputs
