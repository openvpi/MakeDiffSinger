"""Import verified source mappings without running inference."""

import argparse
from pathlib import Path

from makediffsinger.importers import import_annotations
from makediffsinger.catalog import input_dataset
from makediffsinger.script_utils import add_identity, add_output, mapping, matches_sample, run, save
from makediffsinger.types import BatchResult


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    add_identity(parser)
    add_output(parser)
    parser.add_argument("--audio-root", type=Path)
    parser.add_argument("--tier-map", action="append", default=[], metavar="SOURCE=TARGET")
    parser.add_argument("--groups", choices=("unknown", "pronunciation"), default="unknown")
    parser.add_argument("--time-resolution", type=float, default=0.0)
    return parser


def execute(args):
    result = import_annotations(args.input, audio_root=args.audio_root, language=args.language,
                                speaker=args.speaker, subset=args.subset,
                                tier_map=mapping(args.tier_map) or None,
                                groups_verified=args.groups == "pronunciation", time_resolution=args.time_resolution)
    result = BatchResult([o for o in result.outcomes if matches_sample(o.sample, args.sample)])
    dataset = str(args.dataset_root.resolve() if args.dataset_root else input_dataset(args.input))
    for outcome in result.outcomes:
        outcome.sample.metadata["dataset_root"] = dataset
    return save(result, args)


def main(argv=None):
    return run(build_parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())
