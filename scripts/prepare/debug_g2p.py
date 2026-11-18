"""Inspect a clip's paired transcript and test a candidate without changing either file."""

import argparse
from dataclasses import asdict
from pathlib import Path

from makediffsinger.inputs import discover_pairs, with_candidate_transcript
from makediffsinger.script_utils import add_runtime, pipeline, run


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio", type=Path, help="One clip with a paired transcript.")
    parser.add_argument("--language", help="Primary language used for annotation.")
    parser.add_argument("--extended-language", action="append", default=[])
    parser.add_argument("--candidate", type=Path, help="A complete UTF-8 .pfml, .txt, or .lab candidate.")
    parser.add_argument("--probe", action="append", default=[], metavar="TEXT",
                        help="Show encodable phone paths for a plain-text word or short phrase. Repeatable.")
    add_runtime(parser, alignment=True)
    return parser


def describe(outcome):
    return {"status": outcome.status,
            "transcript": asdict(outcome.sample.transcript) if outcome.sample.transcript is not None else None,
            "issues": [asdict(issue) for issue in outcome.issues]}


def execute(args):
    if not args.audio.is_file():
        raise ValueError("Select one audio file.")
    sample, = discover_pairs(args.audio, language=args.language, extended_languages=args.extended_language)
    candidate = with_candidate_transcript(sample, args.candidate) if args.candidate is not None else None

    with pipeline(args) as session:
        current, = session.preflight([sample]).outcomes
        report = {"audio": sample.audio.path, "languages": sample.languages, "current": describe(current)}
        checked = current
        if candidate is not None:
            checked, = session.preflight([candidate]).outcomes
            report["candidate"] = describe(checked)
            report["candidate"]["transcript"]["path"] = str(args.candidate.resolve())
        if args.probe:
            probes = session.tifa.pronunciations(args.probe, languages=sample.languages)
            report["probes"] = [{"text": text, **result} for text, result in zip(args.probe, probes)]
    return report, 0 if checked.status == "ready" else 2


def main(argv=None):
    return run(build_parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())
