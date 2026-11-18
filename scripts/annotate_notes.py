"""Extract notes using the verified syllable boundaries in existing annotations."""

import argparse
from pathlib import Path

from makediffsinger.script_utils import (add_output, add_runtime, add_selection, criteria, pipeline,
                                         publish_quality_exclusions, run, save)
from makediffsinger.selection import select
from makediffsinger.storage import load_annotations


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    add_output(parser)
    add_runtime(parser, notes=True)
    add_selection(parser)
    return parser


def execute(args):
    selected = select(load_annotations(args.input), criteria(args))
    publish_quality_exclusions(selected, args)
    if not selected.annotations:
        if selected.quality_removed:
            return {"processed": 0, "quality_excluded": len(selected.quality_removed)}, 0
        raise ValueError("The annotation selection is empty.")
    with pipeline(args) as session:
        return save(session.annotate_notes(selected.annotations), args)


def main(argv=None):
    return run(build_parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())
