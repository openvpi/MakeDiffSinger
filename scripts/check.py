"""Validate stored annotations and optionally their source audio."""

import argparse
from pathlib import Path

from dataclasses import asdict

from makediffsinger.script_utils import (add_resources, add_selection, criteria, get_profiles, get_resources,
                                         publish_quality_exclusions, run)
from makediffsinger.selection import select
from makediffsinger.storage import load_annotations
from makediffsinger.validation import validate


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    add_selection(parser)
    add_resources(parser)
    parser.add_argument("--complete", action="store_true")
    parser.add_argument("--audio", action="store_true")
    return parser


def execute(args):
    selected = select(load_annotations(args.input), criteria(args))
    profiles = get_profiles(get_resources(args))
    reports = [{"key": a.key, "issues": [asdict(i) for i in validate(
        a, required=("texts", "words", "phones", "notes") if args.complete else (),
        profiles=profiles, check_audio=args.audio)]} for a in selected.annotations]
    failed = any(i["severity"] == "error" for report in reports for i in report["issues"])
    publish_quality_exclusions(selected, args)
    return {"samples": reports}, 2 if failed else 0


def main(argv=None):
    return run(build_parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())
