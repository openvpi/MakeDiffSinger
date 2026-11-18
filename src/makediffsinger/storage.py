"""Editable TextGrid assets, per-item state, and optional JSON checkpoints."""

from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path

from .annotations import read_textgrid, tier_signatures, write_textgrid
from .audio import replay_audio
from .catalog import (bind_dataset, collection_root, dataset_locator, dataset_root, edit_tables, path_reference,
                      portable_message, provenance_paths, resolve_locator, resolve_reference, sample_dataset,
                      source_for, table, write_json)
from .types import Annotation, AudioSource, BatchResult, Interval, Issue, Outcome, Sample, Transcript


METADATA_DIR = ".metadata"


def read_slice_origin(path):
    path = Path(path)
    current = source_for(path)
    if current is not None:
        return current
    legacy = path.with_suffix(".source.json")
    sidecar = path.parent / METADATA_DIR / legacy.name
    if not sidecar.is_file():
        sidecar = legacy
    if sidecar.is_file():
        record = json.loads(sidecar.read_text(encoding="utf-8"))
        if record.get("format") == "makediffsinger.slice":
            return record
    return None


def _annotation_sidecar(path):
    sidecar = path.parent / METADATA_DIR / path.with_suffix(".annotation.json").name
    legacy = path.with_suffix(".json")
    return legacy if legacy.exists() and not sidecar.exists() else sidecar


def sample_to_dict(sample: Sample) -> dict:
    if sample.audio.waveform is not None:
        raise ValueError("Persist in-memory audio with replay_audio() before writing a JSON checkpoint.")
    return asdict(sample)


def sample_from_dict(data: dict) -> Sample:
    data = dict(data)
    data["audio"] = AudioSource(**data["audio"])
    if data.get("transcript") is not None:
        data["transcript"] = Transcript(**data["transcript"])
    return Sample(**data)


def _persist_buffer(sample: Sample, root: Path) -> Sample:
    if sample.audio.waveform is None:
        return sample
    path = collection_root(root) / ".work" / "audio" / sample.subset / (sample.id + ".wav")
    index = 0
    while path.exists():
        index += 1
        path = path.with_name(f"{sample.id}.{index}.wav")
    replay_audio(sample.audio, path)
    stored = deepcopy(sample)
    stored.audio = AudioSource(path=str(path))
    stored.metadata["provided_as_buffer"] = True
    stored.metadata["buffer_sample_range"] = {
        "start": sample.audio.start_sample, "end": sample.audio.end_sample,
        "sample_rate": sample.audio.sample_rate,
    }
    return stored


def annotation_to_dict(annotation: Annotation, *, include_tiers=True) -> dict:
    result = {"format": "makediffsinger.annotation", "version": 1,
              "sample": sample_to_dict(annotation.sample), "duration": annotation.duration,
              "quality": annotation.quality,
              "provenance": {k: v for k, v in annotation.provenance.items() if not k.startswith("_")},
              "status": annotation.status, "stale": annotation.stale,
              "edited_layers": annotation.edited_layers,
              "tier_signatures": tier_signatures(annotation.tiers)}
    if include_tiers:
        result["tiers"] = {name: [asdict(i) for i in items] for name, items in annotation.tiers.items()}
    return result


def annotation_from_dict(data: dict) -> Annotation:
    if data.get("format") != "makediffsinger.annotation" or data.get("version") != 1:
        raise ValueError("Unsupported annotation checkpoint format.")
    return Annotation(sample_from_dict(data["sample"]), data["duration"],
                      {name: [Interval(**i) for i in items] for name, items in data.get("tiers", {}).items()},
                      data.get("quality", {}), data.get("provenance", {}), data.get("status", "imported"),
                      data.get("stale", []), data.get("edited_layers", []))


def save_checkpoint(annotation: Annotation, path: str | Path) -> None:
    path = Path(path).resolve()
    sample = annotation.sample
    source = sample_dataset([sample]) or (Path(sample.audio.path).parent if sample.audio.path else path.parent)
    data = annotation_to_dict(annotation)
    data["dataset"] = dataset_locator(path.parent, source, local_key=path)
    data["sample"] = _portable_sample(sample, path.parent, source, standalone=True)
    data["provenance"] = provenance_paths(data["provenance"], path.parent, source=source)
    write_json(path, data)


def load_checkpoint(path: str | Path) -> Annotation:
    path = Path(path).resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if "dataset" in data:
        source = resolve_locator(path.parent, data["dataset"], local_key=path)
        data["sample"] = _restore_sample(data["sample"], path.parent, source)
        data["provenance"] = provenance_paths(data["provenance"], path.parent, source=source, restore=True)
    return annotation_from_dict(data)


def _metric_key(key):
    if key.startswith(".work/fa/"):
        return "03-annotations/" + key.removeprefix(".work/fa/"), "alignment"
    return key, "annotation"


def _store_metrics(batch, key, quality):
    path = batch.root / "04-quality/metrics.json"
    sample, stage = _metric_key(key)
    row = dict(batch.get(path, sample))
    if stage == "alignment" and "annotation" not in row and batch.get(batch.root / ".metadata/annotations.json", sample):
        row["annotation"] = row.get("alignment", {})
    row[stage] = quality
    if row.get("annotation") == row.get("alignment"):
        row.pop("annotation", None)
    batch.set(path, sample, row if any(row.values()) else None)


def _sample_paths(data, convert):
    meta = data["metadata"]
    for obj, key in ((data["audio"], "path"), (data.get("transcript"), "path"), (meta, "source_root"),
                     (meta, "source_csv"), (meta.get("audio"), "path"), (meta.get("origin_recording"), "path"),
                     (meta.get("transcript_change"), "previous_path")):
        if obj and obj.get(key):
            obj[key] = convert(obj[key])


def _portable_sample(sample, root, source=None, *, standalone=False):
    data = sample_to_dict(sample)
    source = source or dataset_root(root)
    _sample_paths(data, lambda v: path_reference(str(Path(v).resolve()), root, source))
    meta = data["metadata"]
    meta.pop("dataset_root", None)
    if meta.get("transcript_read_error"):
        meta["transcript_read_error"] = portable_message(meta["transcript_read_error"], root, source)
    if not standalone and sample.audio.path and Path(sample.audio.path).is_relative_to(root) and source_for(sample.audio.path):
        meta.pop("origin_recording", None)
        meta.pop("origin_slice", None)
    return data


def _restore_sample(data, root, source=None):
    result = deepcopy(data)
    source = source or dataset_root(root)
    _sample_paths(result, lambda v: resolve_reference(v, root, source))
    if source:
        result["metadata"]["dataset_root"] = str(source)
    if result["audio"].get("path") and "origin_recording" not in result["metadata"]:
        source = source_for(result["audio"]["path"])
        if source:
            result["metadata"].update(origin_recording=source["origin_recording"], origin_slice=source["origin_slice"])
    return result


def _associate_sample(batch, sample):
    if sample.audio.path is None:
        return
    path = Path(sample.audio.path)
    key = path_reference(str(path), batch.root, dataset_root(batch.root))
    fields = {"language": sample.language, "extended_languages": sample.extended_languages, "speaker": sample.speaker}
    batch.update(batch.root / ".metadata/sources.json", key, fields)


def store_asset(batch, path, annotation, signatures):
    root = batch.root
    key = path.relative_to(root).as_posix()
    if not key.startswith(".work/"):
        _associate_sample(batch, annotation.sample)
    provenance = {k: v for k, v in annotation.provenance.items() if k in (
        "groups_verified", "imported_ph_num", "imported_phones_signature", "source_annotation",
        "time_resolution_seconds", "layer_origins", "unmapped_source_fields")}
    if "syllable_input" in annotation.provenance:
        provenance["syllable_input"] = {"method": annotation.provenance["syllable_input"]["method"]}
        if provenance["syllable_input"]["method"] == "unconstrained":
            provenance["syllable_input"]["syllables"] = []
    provenance = provenance_paths(provenance, root)
    data = {"sample": _portable_sample(annotation.sample, root), "provenance": provenance,
            "status": annotation.status, "stale": annotation.stale, "edited_layers": annotation.edited_layers,
            "tier_signatures": signatures}
    batch.set(root / ".metadata/annotations.json", key, data)
    _store_metrics(batch, key, annotation.quality)


def _problems_path(root):
    return root / "05-review-and-decisions/issues.json"


def _bind_samples(root, samples):
    source = sample_dataset(samples)
    if source is None and dataset_root(root) is None:
        # Direct API callers can bind a larger original tree before their first batch.
        for sample in samples:
            origin = sample.metadata.get("source_root") or sample.metadata.get("source_csv")
            origin = origin or sample.audio.path
            if origin:
                path = Path(origin).resolve()
                source = path if path.is_dir() else path.parent
                break
    if source is not None:
        bind_dataset(root, source)


def _clear_problem(batch, key):
    for path in (_problems_path(batch.root), batch.root / ".metadata/pending-inputs.json",
                 batch.root / ".work/retry-candidates.json"):
        batch.set(path, key, None)


def read_annotations(path: str | Path, *, sample: Sample | None = None, tier_map=None) -> Annotation:
    path = Path(path).resolve()
    if path.suffix.lower() == ".json":
        return load_checkpoint(path)
    duration, tiers = read_textgrid(path, tier_map)
    sidecar = _annotation_sidecar(path)
    root = collection_root(path.parent)
    key = path.relative_to(root).as_posix()
    data = table(root / ".metadata/annotations.json").get(key)
    if data is not None:
        metric_key, stage = _metric_key(key)
        row = table(root / "04-quality/metrics.json").get(metric_key, {})
        quality = row.get(stage, row.get("alignment", {}))
        data = dict(data, sample=_restore_sample(data["sample"], root), duration=duration,
                    stale=list(data["stale"]), edited_layers=list(data["edited_layers"]),
                    provenance=provenance_paths(deepcopy(data["provenance"]), root, restore=True), quality=deepcopy(quality),
                    format="makediffsinger.annotation", version=1)
    else:
        data = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.is_file() else {}
    if isinstance(data, dict) and data.get("format") == "makediffsinger.annotation":
        annotation = annotation_from_dict(data)
        if sample is not None:
            annotation.sample = sample
        previous = data.get("tier_signatures", {})
    else:
        if sample is None:
            audio = next((path.with_suffix(ext) for ext in (".wav", ".flac") if path.with_suffix(ext).is_file()), None)
            sample = Sample(path.stem, AudioSource(path=str(audio)) if audio else AudioSource())
        annotation = Annotation(sample, duration, {}, provenance={"source_annotation": str(path), "groups_verified": False})
        previous = {}
    current = tier_signatures(tiers)
    changed = [name for name in previous.keys() | current.keys() if previous.get(name) != current.get(name)] if previous else []
    annotation.tiers, annotation.duration = tiers, duration
    annotation.edited_layers = sorted(set(annotation.edited_layers) | set(changed))
    if set(changed) & {"phones", "words"}:
        annotation.stale = sorted(set(annotation.stale) | {"quality"} | ({"notes"} if "notes" in tiers else set()))
    annotation.provenance["_loaded_signatures"] = current
    annotation.provenance["_loaded_path"] = str(path)
    return annotation


def write_annotations(annotation: Annotation, root: str | Path, *, overwrite=False, replace_edited=False,
                      archive_fa=False) -> Path:
    root = Path(root).resolve()
    delivery = collection_root(root)
    _bind_samples(delivery, [annotation.sample])
    with edit_tables(delivery) as batch:
        return _write_asset(annotation, root, batch, overwrite, replace_edited, archive_fa)


def _write_asset(annotation, root, batch, overwrite, replace_edited, archive_fa):
    path = root / annotation.sample.subset / (annotation.sample.id + ".TextGrid")
    path.parent.mkdir(parents=True, exist_ok=True)
    sidecar = _annotation_sidecar(path)
    pfml = annotation.provenance.get("_aligned_pfml") if archive_fa else None
    pfml_path = path.with_suffix(".pfml")
    if pfml and pfml_path.exists() and not (path.exists() and overwrite):
        raise FileExistsError(pfml_path)
    if sidecar.exists():
        existing_metadata = json.loads(sidecar.read_text(encoding="utf-8"))
        if not isinstance(existing_metadata, dict) or existing_metadata.get("format") != "makediffsinger.annotation":
            raise FileExistsError(f"Unrelated source JSON would be overwritten: {sidecar}")
    if path.exists():
        if not overwrite:
            raise FileExistsError(path)
        existing = read_annotations(path)
        expected = annotation.provenance.get("_loaded_signatures")
        if annotation.provenance.get("_loaded_path") == str(path) and expected != tier_signatures(existing.tiers):
            raise RuntimeError(f"Annotation changed since it was read: {path}")
        incoming = tier_signatures(annotation.tiers)
        replaced_edits = [layer for layer in existing.edited_layers
                          if incoming.get(layer) != tier_signatures(existing.tiers).get(layer)]
        if replaced_edits and not replace_edited:
            raise RuntimeError(f"Refusing to replace edited layers {replaced_edits} in {path}.")
    if annotation.sample.audio.waveform is not None:
        annotation.sample = _persist_buffer(annotation.sample, root)
    temporary = path.with_name(path.name + ".tmp")
    temporary_pfml = pfml_path.with_suffix(".pfml.tmp") if pfml else None
    metadata = annotation_to_dict(annotation, include_tiers=False)
    json.dumps(metadata, ensure_ascii=False, allow_nan=False)
    write_textgrid(annotation, temporary)
    _, serialized_tiers = read_textgrid(temporary)
    metadata["tier_signatures"] = tier_signatures(serialized_tiers)
    if temporary_pfml:
        temporary_pfml.write_text(pfml.rstrip("\n") + "\n", encoding="utf-8")
    temporary.replace(path)
    store_asset(batch, path, annotation, metadata["tier_signatures"])
    if sidecar.is_file():
        sidecar.unlink()
    if temporary_pfml:
        temporary_pfml.replace(pfml_path)
    annotation.provenance["_loaded_path"] = str(path)
    annotation.provenance["_loaded_signatures"] = metadata["tier_signatures"]
    problem = path.with_suffix(".problem.json")
    problem.unlink(missing_ok=True)
    (path.parent / METADATA_DIR / problem.name).unlink(missing_ok=True)
    _clear_problem(batch, path.relative_to(batch.root).as_posix())
    return path


def load_annotations(root: str | Path) -> list[Annotation]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)
    if root.is_file():
        return [read_annotations(root)]
    if (root / "03-annotations").is_dir():
        root = root / "03-annotations"
    elif any(p.is_dir() for p in root.glob("*/03-annotations")):
        raise ValueError("Read each input dataset's 03-annotations separately.")
    return [read_annotations(path) for path in sorted(root.rglob("*.TextGrid"))
            if not any(part.startswith(".") for part in path.relative_to(root).parts)]


def save_result(result: BatchResult, root: str | Path, *, overwrite=False, replace_edited=False,
                archive_fa=False) -> dict:
    root = Path(root).resolve()
    delivery = collection_root(root)
    _bind_samples(delivery, [outcome.sample for outcome in result.outcomes])
    root.mkdir(parents=True, exist_ok=True)
    with edit_tables(delivery) as batch:
        return _save_batch(result, root, batch, overwrite, replace_edited, archive_fa)


def _save_batch(result, root, batch, overwrite, replace_edited, archive_fa):
    counts = {"saved": 0, "pending": 0, "failed": 0, "conflict": 0}
    for outcome in result.outcomes:
        if outcome.sample.audio.waveform is not None:
            outcome.sample = _persist_buffer(outcome.sample, root)
            if outcome.annotation is not None:
                outcome.annotation.sample = outcome.sample
        try:
            if outcome.ok and outcome.annotation is not None:
                _write_asset(outcome.annotation, root, batch, overwrite, replace_edited, archive_fa)
                counts["saved"] += 1
                continue
        except (ValueError, OSError, RuntimeError) as exc:
            outcome.status = "conflict"
            outcome.issues.append(Issue("save_conflict", str(exc), "storage"))
        if outcome.status == "ready":
            continue
        path = root / outcome.sample.subset / (outcome.sample.id + ".TextGrid")
        _associate_sample(batch, outcome.sample)
        key = path.relative_to(batch.root).as_posix()
        batch.set(batch.root / ".metadata/pending-inputs.json", key, _portable_sample(outcome.sample, batch.root))
        payload = {"status": outcome.status, "issues": [
            dict(asdict(i), message=portable_message(i.message, batch.root, dataset_root(batch.root))) for i in outcome.issues]}
        candidate = None
        if outcome.annotation is not None:
            try:
                candidate = annotation_to_dict(outcome.annotation)
                candidate["sample"] = _portable_sample(outcome.annotation.sample, batch.root)
                candidate["provenance"] = provenance_paths(candidate["provenance"], batch.root)
                json.dumps(candidate, allow_nan=False)
            except (ValueError, TypeError) as exc:
                candidate = None
                payload["candidate_serialization_error"] = str(exc)
        batch.set(batch.root / ".work/retry-candidates.json", key, candidate)
        batch.set(_problems_path(batch.root), key, payload)
        counts[outcome.status if outcome.status in counts else "failed"] += 1
    return counts


def load_problems(root: str | Path) -> BatchResult:
    root = Path(root).resolve()
    base = collection_root(root)
    records = {}
    paths = sorted(root.rglob("*.problem.json"))
    for path in paths:
        if ".work" in path.relative_to(root).parts:
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        records[path.parent.parent / path.name if path.parent.name == METADATA_DIR else path] = data
    for key, row in table(_problems_path(base)).items():
        path = base / key
        if path.is_relative_to(root) and ".work" not in path.relative_to(root).parts:
            sample = table(base / ".metadata/pending-inputs.json").get(key) or row["sample"]
            data = dict(row, sample=_restore_sample(sample, base))
            candidate = table(base / ".work/retry-candidates.json").get(key)
            if candidate:
                data["candidate"] = dict(candidate, sample=_restore_sample(candidate["sample"], base),
                                         provenance=provenance_paths(candidate["provenance"], base, restore=True))
            records[path.with_suffix(".problem.json")] = data
    outcomes = []
    for data in records.values():
        outcomes.append(Outcome(sample_from_dict(data["sample"]), data["status"],
                                annotation=annotation_from_dict(data["candidate"]) if "candidate" in data else None,
                                issues=[Issue(**i) for i in data.get("issues", [])]))
    return BatchResult(outcomes)
