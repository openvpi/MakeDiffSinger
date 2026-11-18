"""Extract independent note-only annotations without guaranteed phoneme alignment."""

import argparse
from pathlib import Path

from makediffsinger.script_utils import add_identity, add_output, add_runtime, input_samples, pipeline, run, save


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    add_identity(parser)
    add_output(parser)
    add_runtime(parser, notes=True)
    return parser


def execute(args):
    with pipeline(args) as session:
        return save(session.extract_notes(input_samples(args)), args)


def main(argv=None):
    return run(build_parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())
