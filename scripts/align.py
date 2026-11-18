"""Run forced alignment on paired clips and save editable annotations."""

import argparse
from pathlib import Path

from makediffsinger.inputs import with_candidate_transcript
from makediffsinger.lyric_handoff import alignment_inputs
from makediffsinger.script_utils import add_identity, add_output, add_runtime, input_samples, pipeline, run, save


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    add_identity(parser)
    add_output(parser)
    add_runtime(parser, alignment=True)
    parser.add_argument("--pfml", action="store_true", help="Include aligned pronunciation PFML beside the TextGrids.")
    parser.add_argument("--candidate", type=Path, help="Align a separate TXT, LAB or PFML transcript for one selected clip.")
    return parser


def execute(args):
    samples = input_samples(args)
    if args.candidate is not None:
        if len(samples) != 1:
            raise ValueError("--candidate requires one clip. Select an audio file or use --sample.")
        samples = [with_candidate_transcript(samples[0], args.candidate)]
    else:
        samples = alignment_inputs(samples, args.output)
    with pipeline(args) as session:
        return save(session.align(samples), args, archive_fa=args.pfml)


def main(argv=None):
    return run(build_parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())
