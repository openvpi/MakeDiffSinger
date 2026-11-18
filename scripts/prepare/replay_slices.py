"""Replay an accepted slice plan into independent clips."""

import argparse
from pathlib import Path

from makediffsinger.script_utils import run
from makediffsinger.slicing import SlicePlan, replay_slices


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def execute(args):
    return replay_slices(SlicePlan.read(args.input), args.output), 0


def main(argv=None):
    return run(build_parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())
