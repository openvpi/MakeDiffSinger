"""Export an explicitly selected set of independent audio/TextGrid pairs."""

from hashlib import file_digest
from html import escape
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import quote

from .annotations import write_textgrid
from .audio import audio_info, replay_audio
from .catalog import (collection_root, dataset_locator, dataset_root, edit_tables, portable_message,
                      resolve_locator, sample_dataset)
from .inputs import pfml_written_text
from .lyric_handoff import lyric_evidence
from .selection import Selected
from .storage import load_problems, write_json


def _destination(output, list_name, paths):
    output = Path(output).resolve()
    if output.parent.name == "05-review-and-decisions" and output.name in {"pending", "selected", "flagged"}:
        output = output.parent
    for path in (list_name, *paths):
        if (output / path).exists():
            raise FileExistsError(output / path)
    return output


def make_review_pack(selected: Selected, output):
    if not isinstance(selected, Selected):
        raise TypeError("Pass an explicit selection from select() or select_quality().")
    if not selected.annotations:
        raise ValueError("No samples match the requested review selection.")
    output = _destination(output, "review-samples.json",
                          [a.key + ext for a in selected.annotations for ext in (".wav", ".TextGrid")])
    if len({a.key for a in selected.annotations}) != len(selected.annotations):
        raise ValueError("Review sample keys must be unique.")
    for annotation in selected.annotations:
        audio_info(annotation.sample.audio, require_mono=True)
    for annotation in selected.annotations:
        directory = output / annotation.sample.subset
        directory.mkdir(parents=True, exist_ok=True)
        replay_audio(annotation.sample.audio, directory / (annotation.sample.id + ".wav"))
        write_textgrid(annotation, directory / (annotation.sample.id + ".TextGrid"))
    manifest = {"samples": [a.key for a in selected.annotations], "criteria": selected.criteria,
                "selection_details": selected.details, "excluded": selected.excluded}
    write_json(output / "review-samples.json", {"samples": manifest["samples"], "criteria": selected.criteria})
    return manifest


_ATTACHMENTS = {"audio": ("Audio", ".wav"), "attempted_transcript": ("Lyrics", ".lab"),
                "candidate": ("TextGrid", ".TextGrid"), "annotation": ("TextGrid", ".TextGrid")}


def _pending_item(item, staging, delivery, *, route_alignment):
    sample = item.sample
    if item.ok or not item.issues:
        raise ValueError(f"Provide a pending or failed sample with a reason: {sample.key}.")
    base = staging / sample.subset
    base.mkdir(parents=True, exist_ok=True)
    row = {"key": sample.key, "status": item.status, "reasons": [issue.message for issue in item.issues],
           "stages": list(dict.fromkeys(issue.stage for issue in item.issues))}
    evidence = lyric_evidence(sample, delivery)
    if evidence:
        row["asr_text"] = evidence["asr_text"]
        if "asr_error" in evidence:
            row["reasons"].append("ASR: " + evidence["asr_error"])
        if candidate := evidence.get("candidate"):
            row["lyric_candidate"] = candidate
    has_text = row.get("lyric_candidate") or (sample.transcript and sample.transcript.content.strip())
    errors = [issue for issue in item.issues if issue.severity == "error"]
    if (route_alignment and has_text and item.annotation is None and not sample.metadata.get("transcript_read_error") and errors
            and all(i.code == "missing_transcript" or i.stage in {"lyrics", "lyric_matching"} for i in errors)):
        row["needs_alignment"] = True
        if "lyric_candidate" not in row:
            row["lyric_candidate"] = {"content": sample.transcript.content, "format": sample.transcript.format}
        return row
    try:
        wav = replay_audio(sample.audio, base / (sample.id + ".wav"))
        row["audio"] = wav.relative_to(staging).as_posix()
    except (OSError, ValueError, RuntimeError, ImportError) as exc:
        row["reasons"].append(f"Review audio unavailable: {exc}")
    if sample.transcript and sample.transcript.content.strip():
        text = sample.transcript.content
        if sample.transcript.format == "pfml":
            try:
                text = pfml_written_text(text)
            except ValueError:
                text = ""
        if text:
            transcript = base / (sample.id + ".lab")
            transcript.write_text(text, encoding="utf-8")
            row["attempted_transcript"] = transcript.relative_to(staging).as_posix()
    if item.annotation is not None:
        textgrid = base / (sample.id + ".TextGrid")
        try:
            write_textgrid(item.annotation, textgrid)
            row["candidate"] = textgrid.relative_to(staging).as_posix()
        except ValueError as exc:
            row["reasons"].append(f"Candidate TextGrid unavailable: {exc}")
    return row


def _digest(path):
    with path.open("rb") as stream:
        return file_digest(stream, "sha256").hexdigest()


def _review_bullet(row):
    reasons = " ".join(" ".join(reason.split()) for reason in row["reasons"])
    links = " ".join(f"[{label}]({quote(row[role])})" for role, (label, _) in _ATTACHMENTS.items() if role in row)
    parts = [f"- `{row['key']}` ({', '.join(row['stages'])}).", reasons, links]
    if "asr_text" in row:
        parts.append("ASR: <code>" + escape(row["asr_text"] or "(empty)") + "</code>")
    if candidate := row.get("lyric_candidate"):
        text = candidate["content"]
        if candidate["format"] == "pfml":
            try:
                text = pfml_written_text(text)
            except ValueError:
                pass  # Show the attempted markup when it cannot be parsed.
        parts.append("Lyric candidate: <code>" + escape(text) + "</code>")
    return " ".join(part for part in parts if part)


def _replace_attachments(output, staging, rows, previous_rows, kind):
    # A reviewer can edit a delivered copy. Refresh only files still identical to
    # our last publication, and leave unrelated or edited files in place.
    owned, listed = set(), {}
    for row in previous_rows:
        for role, (_, suffix) in _ATTACHMENTS.items():
            if role not in row:
                continue
            path = output / row[role]
            expected = output / (row["key"] + suffix)
            numbered = path.stem.removeprefix(f"{expected.stem}_{kind}")
            if (path.parent != expected.parent or path.suffix != suffix or
                    (path != expected and not (numbered != path.stem and numbered.isdecimal())) or
                    path.is_symlink() or not path.resolve().is_relative_to(output) or not path.is_file()):
                continue
            listed[row["key"], role] = path
            digest = row.get("attachment_digests", {}).get(role)
            if digest is not None and _digest(path) == digest:
                owned.add(path)
    used = set()
    for row in rows:
        digests = {}
        for role in _ATTACHMENTS.keys() & row.keys():
            source = staging / row[role]
            digests[role] = _digest(source)
            old = listed.get((row["key"], role))
            target = output / row[role]
            if old and (old in owned or _digest(old) == digests[role]):
                target = old
                owned.add(old)
            if not target.resolve().is_relative_to(output):
                raise ValueError(f"Review destination outside output: {target}")
            serial = 1
            while target in used or (target.exists() and target not in owned):
                serial += 1
                target = (output / row[role]).with_name(f"{source.stem}_{kind}{serial}{source.suffix}")
            target.parent.mkdir(parents=True, exist_ok=True)
            source.replace(target)
            used.add(target)
            row[role] = target.relative_to(output).as_posix()
        if digests:
            row["attachment_digests"] = digests
    return owned - used


def _publish_pending(outcomes, output, previous, delivery, *, route_alignment):
    if len({item.sample.key for item in outcomes}) != len(outcomes):
        raise ValueError("Review sample keys must be unique.")
    if not outcomes and not previous:
        return {"pending": []}
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=output.parent) as tmp:
        staging = Path(tmp)
        rows, align = [], []
        for item in outcomes:
            row = _pending_item(item, staging, delivery, route_alignment=route_alignment)
            source = item.sample.metadata.get("dataset_root")
            source = Path(source) if source else dataset_root(delivery)
            row["reasons"] = [portable_message(reason, delivery, source) for reason in row["reasons"]]
            if row.pop("needs_alignment", False):
                align.append(row)
            else:
                rows.append(row)
        obsolete = _replace_attachments(output, staging, rows, (previous or {}).get("pending", []), "pending")
        manifest = {"pending": rows}
        if align:
            manifest["return_to_alignment"] = align
        if rows or align:
            note = staging / "review.md"
            sections = ["# Pending review", *(_review_bullet(row) for row in rows)]
            if align:
                sections += ["## Return to forced alignment", *(_review_bullet(row) for row in align)]
            note.write_text("\n\n".join(sections) + "\n", encoding="utf-8")
            note.replace(output / "review.md")
            write_json(output / "pending.json", manifest)
        else:
            (output / "pending.json").unlink(missing_ok=True)
            (output / "review.md").unlink(missing_ok=True)
        for path in obsolete:
            path.unlink(missing_ok=True)
    return manifest


def make_pending_review_pack(outcomes, output, *, delivery=None):
    """Publish an explicit selection into a fresh review pack."""
    output = _destination(output, "pending.json", ["review.md"])
    return _publish_pending(list(outcomes), output, None, collection_root(delivery or output), route_alignment=False)


def quality_review_directory(anchor, *, standalone=None):
    anchor = Path(anchor).resolve()
    root = collection_root(anchor)
    if root == anchor and not (root / "03-annotations").is_dir():
        root = Path(standalone or anchor).resolve().parent
    return root / "05-review-and-decisions"


def sync_quality_review(selected: Selected, output):
    """Publish the actual quality exclusions and reconcile earlier copies within this selection."""
    if not selected.quality_filter_applied:
        return
    scope = selected.annotations + selected.quality_removed
    keys = {a.key for a in scope}
    if len(keys) != len(scope):
        raise ValueError("Quality review requires unique sample keys.")
    source = sample_dataset(a.sample for a in scope)
    output = Path(output).resolve()
    index, note = output / "quality-review.json", output / "quality-review.md"
    reasons = {row["key"]: row for row in selected.excluded}
    root = collection_root(output)
    with edit_tables(root):
        previous = json.loads(index.read_text(encoding="utf-8")) if index.is_file() else {}
        if not previous and not selected.quality_removed:
            return
        if not previous and note.exists():
            raise FileExistsError(note)
        binding = previous.get("dataset")
        bound_source = dataset_root(root)
        if binding:
            bound_source = resolve_locator(output, binding, local_key=index)
        if bound_source and source and bound_source != source:
            raise ValueError("Choose a separate quality review directory for each input dataset.")
        output.mkdir(parents=True, exist_ok=True)
        untouched, replaced = [], []
        for row in previous.get("excluded", []):
            (replaced if row["key"] in keys else untouched).append(row)
        with TemporaryDirectory(dir=output.parent) as tmp:
            staging = Path(tmp)
            rows = []
            for item in selected.quality_removed:
                entry = reasons[item.key]
                reason = entry.get("detail") or f"Bottom {100 * selected.criteria['drop_bottom_fraction']:g}% by alignment quality."
                base = staging / item.sample.subset
                base.mkdir(parents=True, exist_ok=True)
                wav = replay_audio(item.sample.audio, base / (item.sample.id + ".wav"))
                grid = base / (item.sample.id + ".TextGrid")
                write_textgrid(item, grid)
                rows.append({"key": item.key, "reason": entry["reason"], "reasons": [reason], "stages": ["quality"],
                             "audio": wav.relative_to(staging).as_posix(), "annotation": grid.relative_to(staging).as_posix()})
            obsolete = _replace_attachments(output, staging, rows, replaced, "quality")
            rows = untouched + rows
            if rows:
                manifest = {"excluded": rows}
                if binding or (source and not bound_source):
                    manifest["dataset"] = binding or dataset_locator(output, source, local_key=index)
                page = staging / note.name
                page.write_text("# Excluded by annotation quality\n\n" + "\n\n".join(map(_review_bullet, rows)) + "\n",
                                encoding="utf-8")
                page.replace(note)
                write_json(index, manifest)
            else:
                index.unlink(missing_ok=True)
                note.unlink(missing_ok=True)
            for path in obsolete:
                path.unlink(missing_ok=True)


def sync_pending_review(annotations, output):
    """Publish the complete current unresolved set after annotation or retry."""
    output = Path(output).resolve()
    annotations = Path(annotations).resolve()
    if not annotations.is_dir():
        raise FileNotFoundError(annotations)
    root = collection_root(annotations)
    if output == root / "05-review-and-decisions" and annotations.is_relative_to(root / "03-annotations"):
        annotations = root / "03-annotations"
    with edit_tables(root):
        pending = load_problems(annotations).problems
        index = output / "pending.json"
        previous = json.loads(index.read_text(encoding="utf-8")) if index.is_file() else None
        if pending and not index.exists() and (output / "review.md").exists():
            raise FileExistsError(output / "review.md")
        return _publish_pending(pending, output, previous, root, route_alignment=True)
