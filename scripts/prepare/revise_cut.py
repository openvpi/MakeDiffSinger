"""Stage an explicit cut move or merge as a plan for replacement clips."""

import argparse
from pathlib import Path

from makediffsinger.cut_revision import revise_cut
from makediffsinger.script_utils import run
from makediffsinger.slicing import SlicePlan


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Existing slice plan.")
    parser.add_argument("left", help="Left slice ID.")
    parser.add_argument("right", help="Right slice ID.")
    operation = parser.add_mutually_exclusive_group(required=True)
    operation.add_argument("--move-to", type=int, metavar="SAMPLE", help="New cut in original sample coordinates.")
    operation.add_argument("--merge-as", metavar="ID", help="Remove the boundary and give the merged clip a new ID.")
    parser.add_argument("--output", type=Path, required=True, help="New replay plan for the replacement clips.")
    return parser


def execute(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    plan = revise_cut(SlicePlan.read(args.input), args.left, args.right,
                      move_to=args.move_to, merge_as=args.merge_as)
    plan.write(args.output)
    return {"plan": str(args.output), **plan.settings["cut_revision"],
            "replacement_ids": [item.id for item in plan.slices]}, 0


if __name__ == "__main__":
    raise SystemExit(run(build_parser(), execute))
