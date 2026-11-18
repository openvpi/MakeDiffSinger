"""Inspect an unfamiliar dataset without running models."""

import argparse
from pathlib import Path

from makediffsinger.inputs import inspect_dataset
from makediffsinger.script_utils import run
from makediffsinger.storage import write_json


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--examples-per-format", type=int, default=3)
    return parser


def execute(args):
    result = inspect_dataset(args.input, examples_per_format=args.examples_per_format)
    if args.output:
        write_json(args.output, result)
    return result, 0


def main(argv=None):
    return run(build_parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())
