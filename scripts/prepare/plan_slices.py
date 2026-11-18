"""Prepare a reviewable slice plan from a recording."""

import argparse
from pathlib import Path

from makediffsinger.script_utils import run
from makediffsinger.slicing import plan_slices


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recording-id")
    parser.add_argument("--dataset-root", type=Path, help="Original input directory when planning one file from a larger dataset.")
    parser.add_argument("--keep-whole", action="store_true")
    parser.add_argument("--target-seconds", type=float, default=10.0)
    parser.add_argument("--minimum-seconds", type=float, default=5.0)
    parser.add_argument("--maximum-target-seconds", type=float, default=15.0)
    parser.add_argument("--threshold-db", type=float, default=-40.0)
    parser.add_argument("--weak-threshold-db", type=float,
                        help="Secondary RMS threshold. Defaults to 10 dB above --threshold-db.")
    return parser


def execute(args):
    plan = plan_slices(args.input, recording_id=args.recording_id, keep_whole=args.keep_whole,
                       target_seconds=args.target_seconds, minimum_seconds=args.minimum_seconds,
                       maximum_target_seconds=args.maximum_target_seconds, threshold_db=args.threshold_db,
                       weak_threshold_db=args.weak_threshold_db, source_root=args.dataset_root)
    plan.write(args.output)
    return {"plan": str(args.output), "slices": len(plan.slices),
            "needs_review": [item.id for item in plan.slices if item.review]}, 0


def main(argv=None):
    return run(build_parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())
