"""Retry selected pending or failed samples using current inputs."""

import argparse
from fnmatch import fnmatchcase
from pathlib import Path

from makediffsinger.lyric_handoff import alignment_inputs
from makediffsinger.script_utils import add_output, add_runtime, add_selection, matches_sample, pipeline, run, save
from makediffsinger.storage import load_annotations, load_problems


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    add_output(parser)
    add_runtime(parser, alignment=True, notes=True)
    add_selection(parser, quality=False)
    parser.add_argument("--stage", choices=("fa", "notes", "all"), default="all")
    return parser


def execute(args):
    pending = load_problems(args.input)
    selected = [o for o in pending.outcomes if
                matches_sample(o.sample, args.sample)
                and (not args.status or o.status in args.status)
                and (not args.language or o.sample.language in args.language)
                and (not args.speaker or o.sample.speaker in args.speaker)
                and (not args.subset or any(fnmatchcase(o.sample.subset, p) for p in args.subset))]
    if not selected:
        raise ValueError("No pending or failed samples match the selection.")
    with pipeline(args) as session:
        if args.stage == "notes":
            wanted = {o.sample.key for o in selected}
            existing = {a.key: a for a in load_annotations(args.input) if a.key in wanted}
            for outcome in selected:
                if outcome.sample.key not in existing and outcome.annotation is not None:
                    existing[outcome.sample.key] = outcome.annotation
            if set(existing) != wanted:
                raise ValueError("Notes retry needs prior FA assets or saved FA candidates.")
            result = session.reannotate(list(existing.values()), layers=("notes",), replace_edited=args.replace_edited)
        else:
            function = session.align if args.stage == "fa" else session.run
            result = function(alignment_inputs([o.sample for o in selected], args.input))
        return save(result, args)


def main(argv=None):
    return run(build_parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())
