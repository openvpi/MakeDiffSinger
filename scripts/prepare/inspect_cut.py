"""Inspect a local cut and optionally prepare a short probe for replay."""

import argparse
from pathlib import Path

from makediffsinger.cut_revision import inspect_cut
from makediffsinger.script_utils import run
from makediffsinger.slicing import Slice, SlicePlan


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Existing slice plan.")
    parser.add_argument("left", help="Left slice ID.")
    parser.add_argument("right", help="Right slice ID.")
    parser.add_argument("--window", nargs=2, type=int, metavar=("START", "END"), required=True,
                        help="Original sample coordinates straddling the cut.")
    parser.add_argument("--probe-plan", type=Path, help="New replay plan for the inspection window.")
    return parser


def execute(args):
    plan = SlicePlan.read(args.input)
    start, end = args.window
    report = inspect_cut(plan, args.left, args.right, start, end)
    if args.probe_plan is not None:
        if args.probe_plan.exists():
            raise FileExistsError(args.probe_plan)
        SlicePlan(plan.source, [Slice("cut_probe", start, end)], plan.settings, plan.dataset).write(args.probe_plan)
        report["probe_plan"] = str(args.probe_plan)
    return report, 0


if __name__ == "__main__":
    raise SystemExit(run(build_parser(), execute))
