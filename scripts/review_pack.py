"""Copy only explicitly selected samples for manual review."""

import argparse
from pathlib import Path

from makediffsinger.review import make_review_pack
from makediffsinger.script_utils import add_selection, criteria, run
from makediffsinger.selection import QualityRule, select, select_quality
from makediffsinger.storage import load_annotations


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    add_selection(parser, quality=False)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bottom-fraction", type=float, help="Select the worst fraction by alignment quality rank.")
    parser.add_argument("--group-by", choices=("speaker", "subset", "language"),
                        help="Apply the fraction separately to each group.")
    return parser


def execute(args):
    selected = select(load_annotations(args.input), criteria(args))
    if args.bottom_fraction is not None:
        selected = select_quality(selected, QualityRule(args.bottom_fraction, group_by=args.group_by))
    elif args.group_by or not args.sample:
        raise ValueError("Select --sample IDs/patterns or specify --bottom-fraction.")
    return make_review_pack(selected, args.output), 0


def main(argv=None):
    return run(build_parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())
