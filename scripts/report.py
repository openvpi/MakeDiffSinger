"""Summarize annotation quality and optionally save the aligner's native figures."""

import argparse
from pathlib import Path

from makediffsinger.quality import write_overview
from makediffsinger.catalog import collection_root
from makediffsinger.script_utils import (add_resources, add_selection, annotation_report, criteria,
                                         get_resources, publish_quality_exclusions, run)
from makediffsinger.selection import select
from makediffsinger.storage import load_annotations


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    add_selection(parser)
    add_resources(parser)
    parser.add_argument("--output", type=Path, help="Write the aligner's native statistics and diagnosis.")
    parser.add_argument("--plots", action="store_true", help=argparse.SUPPRESS)
    return parser


def execute(args):
    if args.plots and not args.output:
        raise ValueError("--plots requires an explicit --output directory.")
    annotations = load_annotations(args.input)
    selected = select(annotations, criteria(args))
    if args.output and args.output.resolve() == collection_root(args.input) / "04-quality":
        catalog = collection_root(args.input) / "03-annotations"
        if catalog.exists() and (args.input.resolve() != catalog or len(selected.annotations) != len(annotations)):
            raise ValueError("Use a separate statistics directory for a partial selection.")
    report = annotation_report(selected, args.input, keep_input_order=bool(args.output))
    if args.output:
        write_overview(report, args.output, tifa_repo=get_resources(args).repo("tifa"))
    publish_quality_exclusions(selected, args)
    return report, 0


def main(argv=None):
    return run(build_parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())
