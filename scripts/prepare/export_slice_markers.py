"""Export accepted slice plans as tab-separated Audition .csv markers."""

import argparse
from pathlib import Path

from makediffsinger.script_utils import run
from makediffsinger.slicing import SlicePlan, write_audition_markers


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plans", nargs="+", type=Path, help="Slice plans for one original recording.")
    parser.add_argument("--output", required=True, type=Path, help="Marker .csv path beside the corresponding clips.")
    parser.add_argument("--recording", type=Path, help="Link a kept copy with the original sample grid.")
    return parser


def execute(args):
    plans = (SlicePlan.read(path) for path in args.plans)
    return write_audition_markers(plans, args.output, recording=args.recording), 0


if __name__ == "__main__":
    raise SystemExit(run(build_parser(), execute))
