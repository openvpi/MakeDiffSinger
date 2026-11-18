"""Inspect the ranked quality tail and preview exclusions without writing files."""

import argparse
from pathlib import Path

from makediffsinger.quality import quality_tail_summary
from makediffsinger.script_utils import add_quality_exclusions, add_selection, criteria, run
from makediffsinger.storage import load_annotations


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Saved annotation directory.")
    add_selection(parser, quality=False)
    add_quality_exclusions(parser)
    parser.add_argument("--percentile-group", choices=("speaker", "subset", "language"),
                        help="Rank and summarize each existing group separately.")
    parser.add_argument("--bands", type=int, default=10, help="Number of rank bands per group for display.")
    parser.add_argument("--preview-fraction", type=float, help="Preview this exclusion fraction without applying it.")
    return parser


def execute(args):
    report = quality_tail_summary(load_annotations(args.input), criteria(args), bands=args.bands,
                                  preview_fraction=args.preview_fraction)
    return report, 0


def main(argv=None):
    return run(build_parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())
