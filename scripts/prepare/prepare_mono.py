"""Check audio channels and optionally write a mono working copy."""

import argparse
from collections import Counter
import math
import os
from pathlib import Path
import shutil

from makediffsinger.audio import (
    MIN_STEREO_DIFFERENCE_SECONDS, audio_info, compare_stereo, load_audio, mono_track, write_mono_track,
)
from makediffsinger.inputs import TRANSCRIPT_EXTENSIONS, list_audio_files
from makediffsinger.processing_notes import append_processing_notes
from makediffsinger.script_utils import run
from makediffsinger.catalog import bind_dataset, collection_root, input_dataset, store_origins
from makediffsinger.storage import read_slice_origin
from makediffsinger.types import AudioSource


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Audio file or directory of recordings.")
    parser.add_argument("--output", type=Path, help="Separate empty directory for prepared audio. Omit to inspect only.")
    parser.add_argument("--dataset-root", type=Path, help="Original dataset directory when preparing a subset.")
    parser.add_argument("--copy-transcripts", action="store_true", help="Copy authorized paired PFML, TXT, and LAB sidecars.")
    parser.add_argument("--cancellation-db", type=float, default=12.0,
                        help="Level loss at which to keep the first channel instead of the average (default: 12 dB).")
    parser.add_argument("--min-difference-seconds", type=float, default=MIN_STEREO_DIFFERENCE_SECONDS,
                        help="Continuous channel difference required for separation (default: %(default)s s).")
    parser.add_argument("--processing-notes", type=Path, help="Append actions to the shared Markdown processing notes.")
    return parser


def stem_key(path):
    return os.path.normcase(str(path.parent / path.stem))


def output_names(paths):
    reserved = {stem_key(path) for path in paths}
    used, names = set(), {}
    for path in sorted(paths, key=lambda path: (path.suffix.lower() != ".wav", path)):
        candidate = path
        if stem_key(path) in used:
            stem = f"{path.stem}_{path.suffix[1:].lower()}"
            candidate = path.with_stem(stem)
            serial = 2
            while stem_key(candidate) in reserved or stem_key(candidate) in used:
                candidate = path.with_stem(f"{stem}_{serial}")
                serial += 1
        used.add(stem_key(candidate))
        names[path] = candidate
    return names


def reserve_name(path, suffix, reserved):
    if not suffix:
        return path
    stem = path.stem + suffix
    candidate = path.with_stem(stem)
    serial = 2
    while stem_key(candidate) in reserved:
        candidate = path.with_stem(f"{stem}_{serial}")
        serial += 1
    reserved.add(stem_key(candidate))
    return candidate


def prepare_file(path, relative, output, output_name, reserved, args):
    source = AudioSource(str(path))
    info = audio_info(source)
    channels = info["channels"]
    result = {"source": relative.as_posix(), "channels": channels, "sample_rate": info["sample_rate"]}
    if channels not in (1, 2):
        raise ValueError(f"Found {channels} channels. Supply an explicit channel selection or mono source.")
    if channels == 2:
        waveform, rate = load_audio(source, mono=False)
        result.update(compare_stereo(waveform, rate, cancellation_db=args.cancellation_db,
                                     min_difference_seconds=args.min_difference_seconds))
    else:
        result["method"] = "already_mono"
    if output is None:
        return result

    if channels == 2 and output_name.suffix.lower() not in {".wav", ".flac"}:
        output_name = output_name.with_suffix(".wav")
    regions = result.get("regions", [{"start_sample": 0, "end_sample": info["frames"],
                                     "method": "already_mono", "channels": ["mono"]}])
    origin = read_slice_origin(path)
    recording = origin["origin_recording"] if origin else info
    original_slice = origin["origin_slice"] if origin else {}
    offset = original_slice.get("start_sample", 0)
    segmented = len(regions) > 1
    distinct_methods = {"split_channels", "single_active_channel"}
    transformed = segmented or any(region["method"] in distinct_methods for region in regions)
    sidecars = [path.with_suffix(suffix) for suffix in TRANSCRIPT_EXTENSIONS
                if path.with_suffix(suffix).is_file()] if args.copy_transcripts else []
    created, names, notices = [], [], []
    origins = []
    subject = f"`{result['source']}`"
    try:
        for index, region in enumerate(regions, 1):
            start, end = region["start_sample"], region["end_sample"]
            portion = f"samples [{start}, {end})" if segmented else "the full recording"
            distinct = region["method"] in distinct_methods
            local_outputs = []
            for channel in region["channels"]:
                suffix = f"_r{index:03d}" if segmented else ""
                if distinct:
                    suffix += {"left": "_L", "right": "_R"}[channel]
                name = reserve_name(output_name, suffix, reserved)
                target = output / name
                if target.exists():
                    raise FileExistsError(target)
                if channels == 1:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    created.append(target)
                    shutil.copy2(path, target)
                else:
                    write_mono_track(target, mono_track(waveform[start:end], channel), rate, info["subtype"])
                    created.append(target)
                origins.append((target, {"origin_recording": recording,
                                   "origin_slice": {"id": target.stem, "start_sample": offset + start,
                                                    "end_sample": offset + end,
                                                    "channel": original_slice.get("channel", channel)}}))
                for sidecar in (() if transformed else sidecars):
                    transcript = target.with_suffix(sidecar.suffix)
                    if transcript.exists():
                        raise FileExistsError(transcript)
                    created.append(transcript)
                    shutil.copy2(sidecar, transcript)
                local_outputs.append(name.as_posix())
            region["outputs"] = local_outputs
            names.extend(local_outputs)
            targets = ", ".join(f"`{name}`" for name in local_outputs)
            if distinct:
                notices.append(f"{subject} had distinct channel content in {portion}. Prepared {targets} separately.")
            elif region["method"] == "first_channel":
                notices.append(f"{subject} showed severe cancellation in {portion}. Kept the left channel in {targets}.")
        store_origins(output, origins)
    except (OSError, ValueError, RuntimeError):
        for target in created:
            target.unlink(missing_ok=True)
        raise

    result["outputs"] = names
    if len(names) == 1:
        result["output"] = names[0]
    if sidecars and transformed:
        notices.append(f"The original transcript for {subject} was left unassigned pending range and channel matching.")
    if output_name.stem != relative.stem:
        notices.append(f"Used the stem `{output_name.stem}` for {subject} to avoid a name collision.")
    if notices:
        result["notices"] = notices
    return result


def execute(args):
    if not math.isfinite(args.cancellation_db) or args.cancellation_db <= 0:
        raise ValueError("--cancellation-db must be positive and finite.")
    if not math.isfinite(args.min_difference_seconds) or args.min_difference_seconds <= 0:
        raise ValueError("--min-difference-seconds must be positive and finite.")
    root = args.input.resolve(strict=True)
    base = root.parent if root.is_file() else root
    files = list_audio_files(root)
    output = args.output.resolve() if args.output is not None else None
    if output is not None:
        if output.is_relative_to(root):
            raise ValueError("Choose a working directory outside the source.")
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(f"Use an empty working directory: {output}")
        bind_dataset(collection_root(output), args.dataset_root or input_dataset(root))
    names = output_names([path.relative_to(base) for path in files]) if output is not None else {}
    reserved = {stem_key(name) for name in names.values()}

    items = []
    for path in files:
        relative = path.relative_to(base)
        try:
            items.append(prepare_file(path, relative, output, names.get(relative), reserved, args))
        except (OSError, ValueError, RuntimeError, ImportError) as exc:
            items.append({"source": relative.as_posix(), "method": "failed", "error": str(exc)})
    counts = Counter(item["method"] for item in items)
    messages = [notice for item in items for notice in item.get("notices", [])]
    failures = [item for item in items if item["method"] == "failed"]
    if output is not None:
        messages.extend(f"Skipped `{item['source']}`. {item['error']}" for item in failures)
        append_processing_notes(args.processing_notes, "Mono preparation", messages)
    return {"counts": dict(counts), "cancellation_db": args.cancellation_db,
            "min_difference_seconds": args.min_difference_seconds, "files": items}, 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(run(build_parser(), execute))
