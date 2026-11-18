"""Prepare a lyric-matching table for sliced clips."""

import argparse
from pathlib import Path

from makediffsinger.lyric_asr import read_hypotheses
from makediffsinger.lyric_handoff import write_candidate_table
from makediffsinger.script_utils import add_identity, input_samples, run
from makediffsinger.slicing import write_lyric_table


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--asr", type=Path, help="Use independent clip ASR as FA candidates when no song reference is available.")
    add_identity(parser)
    return parser


def execute(args):
    samples = input_samples(args)
    if args.asr:
        report = read_hypotheses(args.asr, samples)
        filled = write_candidate_table(samples, report["clips"], args.output)
        return {"rows": len(samples), "prefilled": len(filled)}, 0
    return {"rows": write_lyric_table(samples, args.output)}, 0


def main(argv=None):
    return run(build_parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())
