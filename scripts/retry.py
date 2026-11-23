"""Retry selected pending or failed samples using current inputs."""

import argparse
from fnmatch import fnmatchcase
from pathlib import Path

from makediffsinger.lyric_handoff import alignment_inputs
from makediffsinger.catalog import collection_root
from makediffsinger.script_utils import add_output, add_runtime, add_selection, matches_sample, pipeline, run, save
from makediffsinger.storage import load_annotations, load_problems, read_annotations
from makediffsinger.types import Outcome, Issue


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
            unreadable = []
            existing = {a.key: a for a in load_annotations(args.input, failures=unreadable) if a.key in wanted}
            damaged = {r["key"]: r["reason"] for r in unreadable}
            fa_root = collection_root(args.input) / ".work/fa"
            for outcome in selected:
                key = outcome.sample.key
                if key in existing or key in damaged:
                    continue
                if outcome.annotation is not None:
                    existing[key] = outcome.annotation
                elif (archive := fa_root / (key + ".TextGrid")).is_file():
                    try:
                        existing[key] = read_annotations(archive)
                    except (ValueError, OSError) as exc:
                        damaged[key] = str(exc)
            result = session.reannotate(list(existing.values()), layers=("notes",), replace_edited=args.replace_edited)
            for outcome in selected:
                if outcome.sample.key not in existing:
                    reason = damaged.get(outcome.sample.key, "Notes retry needs a prior FA asset or saved FA candidate.")
                    result.outcomes.append(Outcome(outcome.sample, "pending", issues=[Issue("missing_fa", reason, "game")]))
        else:
            function = session.align if args.stage == "fa" else session.run
            result = function(alignment_inputs([o.sample for o in selected], args.input))
        return save(result, args)


def main(argv=None):
    return run(build_parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())
