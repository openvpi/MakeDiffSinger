"""Optional source-sample slice planning and explicit replay."""

from dataclasses import asdict, dataclass, field
import csv
import importlib.util
import json
import math
from pathlib import Path

from .audio import audio_info, load_audio, replay_audio
from .config import repository_root
from .cut_planning import MAX_PADDING_SECONDS, WEAK_THRESHOLD_OFFSET_DB, recording_cuts
from .catalog import (bind_dataset, collection_root, dataset_locator, dataset_root, input_dataset,
                      path_reference, resolve_locator, resolve_reference, store_origins)
from .storage import read_slice_origin, write_json
from .types import AudioSource, check_identity


@dataclass
class Slice:
    id: str
    start_sample: int
    end_sample: int
    review: list[str] = field(default_factory=list)
    cut_evidence: list[dict] = field(default_factory=list)


@dataclass
class SlicePlan:
    source: dict
    slices: list[Slice]
    settings: dict = field(default_factory=dict)
    dataset: Path | None = field(default=None, compare=False)

    def write(self, path):
        directory = Path(path).resolve().parent
        delivery = collection_root(directory)
        bound = dataset_root(delivery)
        source = self.dataset or bound or input_dataset(self.source.get("origin_recording", self.source)["path"])
        data = {"format": "makediffsinger.slices", "version": 2, "source": dict(self.source),
                "slices": [asdict(s) for s in self.slices], "settings": self.settings}
        if bound is None and ".work" not in directory.relative_to(delivery).parts:
            delivery = directory
            data["dataset"] = dataset_locator(directory, source, local_key=Path(path).resolve())
        else:
            bind_dataset(delivery, source)
        info = data["source"]
        info["path"] = path_reference(info["path"], delivery, source)
        if "origin_recording" in info:
            origin = info["origin_recording"]
            info["origin_recording"] = dict(origin, path=path_reference(origin["path"], delivery, source))
        write_json(path, data)

    @classmethod
    def read(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format") != "makediffsinger.slices" or data.get("version") not in (1, 2):
            raise ValueError("Unsupported slice plan.")
        info, source = data["source"], None
        if data["version"] == 2:
            directory = Path(path).resolve().parent
            delivery = directory if "dataset" in data else collection_root(directory)
            source = resolve_locator(directory, data["dataset"], local_key=Path(path).resolve()) if "dataset" in data else dataset_root(delivery)
            for record in (info, info.get("origin_recording")):
                if record:
                    record["path"] = resolve_reference(record["path"], delivery, source)
        return cls(info, [Slice(**item) for item in data["slices"]], data.get("settings", {}), source)


def _slicer_module(game_repo=None):
    path = (Path(game_repo) if game_repo else repository_root() / "external" / "GAME") / "inference" / "slicer2.py"
    spec = importlib.util.spec_from_file_location("_mds_slicer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plan_slices(source: str | Path, *, recording_id=None, keep_whole=False, target_seconds=10.0,
                minimum_seconds=5.0, maximum_target_seconds=15.0, threshold_db=-40.0,
                weak_threshold_db=None, phone_hints=(), game_repo=None, source_root=None) -> SlicePlan:
    if not 0 < minimum_seconds <= target_seconds <= maximum_target_seconds:
        raise ValueError("Require 0 < minimum_seconds <= target_seconds <= maximum_target_seconds.")
    if weak_threshold_db is None:
        weak_threshold_db = threshold_db + WEAK_THRESHOLD_OFFSET_DB
    if not math.isfinite(threshold_db) or not math.isfinite(weak_threshold_db) or weak_threshold_db <= threshold_db:
        raise ValueError("The weak RMS threshold must be finite and higher than the finite silence threshold.")
    dataset = Path(source_root).resolve() if source_root else dataset_root(collection_root(Path(source).resolve().parent))
    source = AudioSource(str(source))
    info = audio_info(source, require_mono=True)
    if origin := read_slice_origin(source.path):
        info.update(origin_recording=origin["origin_recording"], origin_slice=origin["origin_slice"])
    recording_id = recording_id or Path(source.path).stem
    check_identity(recording_id)
    settings = {"keep_whole": keep_whole, "target_seconds": target_seconds,
                "minimum_seconds": minimum_seconds, "maximum_target_seconds": maximum_target_seconds,
                "threshold_db": threshold_db, "weak_threshold_db": weak_threshold_db,
                "max_padding_seconds": MAX_PADDING_SECONDS}
    if keep_whole:
        return SlicePlan(info, [Slice(recording_id, 0, info["frames"])], settings, dataset)
    audio, rate = load_audio(source)
    module = _slicer_module(game_repo)
    slicer = module.Slicer(sr=rate, threshold=threshold_db, min_length=round(minimum_seconds * 1000))
    limits = tuple(max(1, round(seconds * rate))
                   for seconds in (minimum_seconds, target_seconds, maximum_target_seconds))
    ranges, evidence = recording_cuts(audio, slicer, limits, tuple(phone_hints),
                                     weak_threshold=10 ** (weak_threshold_db / 20))
    segments = []
    for index, (left, right) in enumerate(ranges):
        adjacent = [evidence[cut] for cut in (left, right) if cut in evidence]
        reasons = {reason for cut in adjacent for reason in cut["reasons"]}
        if right - left > limits[-1]:
            reasons.add("above_maximum_target")
        clip_id = recording_id if len(ranges) == 1 else f"{recording_id}_{index:04d}"
        segments.append(Slice(clip_id, left, right, sorted(reasons), adjacent))
    return SlicePlan(info, segments, settings, dataset)


def replay_slices(plan: SlicePlan, output: str | Path):
    output = Path(output).resolve()
    actual = audio_info(AudioSource(plan.source["path"]), require_mono=True)
    if any(actual[key] != plan.source[key] for key in ("frames", "sample_rate", "channels")):
        raise ValueError("The source recording no longer matches the slice plan.")
    seen = set()
    for item in plan.slices:
        check_identity(item.id)
        if item.id in seen:
            raise ValueError(f"Duplicate slice ID: {item.id}")
        seen.add(item.id)
        if not 0 <= item.start_sample < item.end_sample <= actual["frames"]:
            raise ValueError(f"Invalid source sample range: {item.id}")
        if (output / (item.id + ".wav")).exists():
            raise FileExistsError(output / (item.id + ".wav"))
    origin = plan.source.get("origin_recording", plan.source)
    delivery = collection_root(output)
    dataset = plan.dataset or dataset_root(delivery) or Path(origin["path"]).parent
    if not Path(origin["path"]).is_relative_to(dataset):
        raise ValueError("The recording is outside this input dataset.")
    bind_dataset(delivery, dataset)
    parent_slice = plan.source.get("origin_slice", {})
    offset = parent_slice.get("start_sample", 0)
    origins = []
    for item in plan.slices:
        replay_audio(AudioSource(plan.source["path"], item.start_sample, item.end_sample, plan.source["sample_rate"]),
                     output / (item.id + ".wav"))
        archived = asdict(item)
        archived.update(start_sample=item.start_sample + offset, end_sample=item.end_sample + offset)
        if "channel" in parent_slice:
            archived["channel"] = parent_slice["channel"]
        if offset:
            for evidence in archived["cut_evidence"]:
                if "sample" in evidence:
                    evidence["sample"] += offset
        origins.append((output / (item.id + ".wav"), {"origin_recording": origin, "origin_slice": archived}))
    store_origins(output, origins, source_root=dataset)
    return {"slices": len(plan.slices), "source_seconds": actual["source_seconds"],
            "slice_seconds": sum(i.end_sample - i.start_sample for i in plan.slices) / actual["sample_rate"],
            "needs_review": [i.id for i in plan.slices if i.review]}


def write_audition_markers(plans, output, *, recording=None):
    """Export sample-based Audition markers in a tab-separated .csv file."""
    output = Path(output).resolve()
    origins, rows, source = [], [], None
    for plan in plans:
        source = plan.dataset or source
        origin = plan.source.get("origin_recording", plan.source)
        parent_slice = plan.source.get("origin_slice", {})
        offset = parent_slice.get("start_sample", 0)
        if origin["sample_rate"] != plan.source["sample_rate"]:
            raise ValueError("Slice and original recording sample rates differ.")
        origins.append(origin)
        for item in plan.slices:
            if not 0 <= item.start_sample < item.end_sample <= plan.source["frames"] or item.end_sample + offset > origin["frames"]:
                raise ValueError(f"Invalid source sample range: {item.id}")
            rows.append((item.start_sample + offset, item.end_sample + offset, item.id, parent_slice.get("channel", "")))
    if not rows:
        raise ValueError("No slice ranges to export.")
    original = origins[0]
    if any(origin["path"] != original["path"] for origin in origins):
        raise ValueError("Write one marker file per original recording.")
    if recording is not None:
        linked = audio_info(AudioSource(str(recording)))
        if any(linked[key] != original[key] for key in ("frames", "sample_rate")):
            raise ValueError("The linked recording must use the original sample grid.")
    linked_path = Path(recording if recording is not None else original["path"]).resolve()
    delivery = collection_root(output.parent)
    source = dataset_root(delivery) or source or Path(original["path"]).parent
    if recording is not None and linked_path.is_relative_to(delivery):
        reference = path_reference(str(linked_path), output.parent, None)
    elif linked_path.is_relative_to(source):
        reference = linked_path.relative_to(source).as_posix()
    else:
        reference = path_reference(str(linked_path), output.parent, source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(("Name", "Start", "Duration", "Time Format", "Type", "Description"))
        for start, end, name, channel in sorted(rows):
            description = f"{reference} ({channel})" if channel else reference
            writer.writerow((name, start, end - start, f"{original['sample_rate']} Hz", "Cue", description))
    return {"markers": str(output), "recording": str(linked_path), "slices": len(rows)}


def write_lyric_table(samples, output, suggestions=None):
    """Write the shared review/apply format, optionally with proposed transcripts."""
    rows = []
    for sample in samples:
        source_info = sample.metadata.get("origin_slice", {})
        transcript = (sample.transcript if suggestions is None else suggestions.get(sample.key))
        extension = ".pfml" if transcript and transcript.format == "pfml" else ".lab"
        if transcript and transcript.path:
            extension = Path(transcript.path).suffix
        rows.append({"key": sample.key, "audio": sample.metadata["source_relative_path"],
                     "start_sample": source_info.get("start_sample", ""),
                     "end_sample": source_info.get("end_sample", ""),
                     "format": transcript.format if transcript else "text",
                     "extension": extension,
                     "content": transcript.content if transcript else ""})
    path = Path(output)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def prepare_lyrics(audio_root, output):
    from .inputs import discover_pairs
    return write_lyric_table(discover_pairs(audio_root), output)


def apply_lyrics(table, audio_root, *, overwrite=False):
    from .inputs import TRANSCRIPT_EXTENSIONS, pfml_written_text, working_pfml_path
    root = Path(audio_root).resolve()
    writes, pending, retired = [], [], []
    written = 0
    with Path(table).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            audio = (root / row["audio"]).resolve()
            if not audio.is_relative_to(root) or not audio.is_file():
                raise ValueError(f"Audio path is missing or outside the input root: {row['audio']}")
            extension = row.get("extension") or (".pfml" if row["format"] == "pfml" else ".lab")
            if extension not in TRANSCRIPT_EXTENSIONS or (extension == ".pfml") != (row["format"] == "pfml"):
                raise ValueError(f"Transcript extension/format mismatch for {row['audio']}.")
            if not row["content"].strip():
                pending.append(row["key"])
                continue
            content = row["content"]
            work = working_pfml_path(audio)
            if work is not None and extension == ".pfml":
                if work.exists() and not overwrite:
                    raise FileExistsError(work)
                writes.append((work, content))
                content = pfml_written_text(content)
                extension = ".txt" if audio.with_suffix(".txt").is_file() else ".lab"
            elif work is not None and work.is_file() and overwrite:
                retired.append(work)
            path = audio.with_suffix(extension)
            if path.exists() and not overwrite:
                raise FileExistsError(path)
            for higher in TRANSCRIPT_EXTENSIONS[:TRANSCRIPT_EXTENSIONS.index(extension)]:
                if audio.with_suffix(higher).exists():
                    raise ValueError(f"{path.name} would be shadowed by {audio.with_suffix(higher).name}.")
            writes.append((path, content))
            written += 1
    if len({path for path, _ in writes}) != len(writes):
        raise ValueError("The lyric table contains duplicate destinations.")
    for path, content in writes:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content + "\n", encoding="utf-8")
    for path in retired:
        path.unlink(missing_ok=True)
    return {"written": written, "pending": pending}
