"""Apply a lyric table. Under 01-clips, keep PFML in .work/paired-lyrics."""

import argparse
from pathlib import Path

from makediffsinger.script_utils import run
from makediffsinger.slicing import apply_lyrics


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--audio-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def execute(args):
    return apply_lyrics(args.input, args.audio_root, overwrite=args.overwrite), 0


def main(argv=None):
    return run(build_parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())
