"""Deliver all unresolved annotation problems, or package an explicit selection."""

import argparse
from pathlib import Path

from makediffsinger.review import make_pending_review_pack, sync_pending_review
from makediffsinger.script_utils import matches_sample, run
from makediffsinger.storage import load_problems


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Directory with saved problem records.")
    parser.add_argument("--sample", action="append", help="Limit a separate review pack to matching IDs or patterns.")
    parser.add_argument("--output", type=Path, required=True)
    return parser


def execute(args):
    if args.sample:
        outcomes = [item for item in load_problems(args.input).problems if matches_sample(item.sample, args.sample)]
        if not outcomes:
            raise ValueError("No pending review samples match the selection.")
        report = make_pending_review_pack(outcomes, args.output, delivery=args.input)
    else:
        report = sync_pending_review(args.input, args.output)
    return report, 2 if report.get("return_to_alignment") else 0


def main(argv=None):
    return run(build_parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())
