"""Compile maintained annotations into a DiffSinger dataset."""

import argparse
from dataclasses import replace
import json
import yaml
from pathlib import Path

from makediffsinger.exporters.diffsinger import export_diffsinger
from makediffsinger.review import quality_review_directory
from makediffsinger.script_utils import add_resources, add_selection, criteria, get_profiles, get_resources, run
from makediffsinger.storage import load_annotations
from makediffsinger.timing import DEFAULT_GAME_GRID


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    add_selection(parser)
    add_resources(parser)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-audio", action="store_true", help="Write the full CSV without copying WAV files.")
    parser.add_argument("--speaker-map", type=Path, help="JSON object mapping sample keys to speaker labels or null.")
    return parser


def execute(args):
    unreadable = []
    annotations = load_annotations(args.input, failures=unreadable)
    if not annotations and unreadable:
        return {"samples": 0, "failed": unreadable}, 2
    if args.speaker_map:
        mapping = json.loads(args.speaker_map.read_text(encoding="utf-8-sig"))
        if not isinstance(mapping, dict) or any(label is not None and
                                               (not isinstance(label, str) or not label.strip())
                                               for label in mapping.values()):
            raise ValueError("The speaker map must map sample keys to nonempty labels or null.")
        unknown = mapping.keys() - {a.key for a in annotations} - {r["key"] for r in unreadable}
        if unknown:
            raise ValueError(f"Unknown sample keys in speaker map: {', '.join(sorted(unknown)[:5])}")
        annotations = [replace(a, sample=replace(a.sample, speaker=mapping[a.key])) if a.key in mapping else a
                       for a in annotations]
    review = args.review_output or quality_review_directory(args.input, standalone=args.output)
    resources = get_resources(args)
    grid = None
    if resources.game_model:
        config = Path(resources.game_model).with_name("config.yaml")
        features = yaml.safe_load(config.read_text(encoding="utf-8"))["inference"].get("features", {})
        grid = {"sample_rate": features.get("audio_sample_rate", DEFAULT_GAME_GRID["sample_rate"]),
                "hop_size": features.get("hop_size", DEFAULT_GAME_GRID["hop_size"])}
    result = export_diffsinger(annotations, args.output, profiles=get_profiles(resources), game_grid=grid,
                              copy_audio=not args.no_audio, criteria=criteria(args), review_output=review)
    summary = {"output": str(args.output), "samples": len(result["selected"]), "excluded": len(result["excluded"])}
    if args.drop_bottom_fraction is not None or args.exclude_quality:
        info = {"fraction": args.drop_bottom_fraction or 0.0, "group_by": args.percentile_group,
                "excluded": [row for row in result["excluded"] if row["reason"] in {"bottom_quality_fraction", "annotation_defect"}]}
        if args.drop_bottom_fraction:
            step, = result["selection_details"]["quality_steps"]
            info.update(eligible=step["eligible"], groups=step["groups"])
        if info["excluded"]:
            info["review"] = str(review / "quality-review.md")
        summary["quality_filter"] = info
    summary["failed"] = unreadable + result["failed"]
    summary["warnings"] = result["warnings"]
    return summary, 2 if summary["failed"] else 0


def main(argv=None):
    return run(build_parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())
