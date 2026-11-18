"""Match pre-cut clips to a song's full lyrics using the aligner's pronunciation scorer."""

import argparse
from collections import Counter
from pathlib import Path
from time import perf_counter

from makediffsinger.lyric_asr import read_hypotheses
from makediffsinger.lyric_handoff import write_candidate_table
from makediffsinger.lyric_match import match_lyrics
from makediffsinger.catalog import bind_dataset, collection_root, edit_tables, path_reference, provenance_paths, sample_dataset
from makediffsinger.processing_notes import append_processing_notes
from makediffsinger.script_utils import add_identity, add_runtime, input_samples, pipeline, run
from makediffsinger.storage import write_json


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Clips from one song.")
    parser.add_argument("--lyrics", type=Path, required=True, help="Full reference lyrics as UTF-8 text.")
    parser.add_argument("--asr", type=Path, required=True, help="ASR JSON or a separate LAB/TXT directory.")
    parser.add_argument("--output", type=Path, required=True, help="Destination for the matched lyric table and report.")
    parser.add_argument("--evidence", type=Path, help="Optional full candidate and pronunciation trace.")
    parser.add_argument("--g2p-only", action="store_true", help="Rank by phone edit distance without acoustic inference.")
    parser.add_argument("--candidate-count", type=int, default=8,
                        help="Coarse shortlist size for choosing a reference excerpt.")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--min-similarity", type=float, help="Flag low phone similarity in the report.")
    parser.add_argument("--min-margin", type=float,
                        help="Flag a small pronunciation score gap within the shared template.")
    parser.add_argument("--min-score", type=float, default=0.0,
                        help="Minimum independent pronunciation score per source and aligned block (default: 0).")
    parser.add_argument("--top-k", type=int, default=3,
                        help="Maximum paths per source in each aligned block (default: 3).")
    parser.add_argument("--exclude-line", type=int, action="append", default=[])
    parser.add_argument("--keep-line", type=int, action="append", default=[])
    parser.add_argument("--processing-notes", type=Path, help="Append matching fallbacks to the shared processing notes.")
    add_identity(parser)
    add_runtime(parser, alignment=True)
    return parser


def execute(args):
    table = args.output / "lyrics.csv"
    for path in (table, args.evidence):
        if path is not None and path.exists():
            raise FileExistsError(path)
    samples = input_samples(args)
    root = collection_root(args.output)
    source = sample_dataset(samples)
    if source:
        bind_dataset(root, source)
    if not args.language:
        raise ValueError("Choose --language for the forced aligner's G2P and pronunciation scoring.")
    hypotheses = read_hypotheses(args.asr, samples)
    verify = not args.g2p_only
    start = perf_counter()
    with pipeline(args) as worker:
        report = match_lyrics(
            samples, args.lyrics.read_text(encoding="utf-8-sig"), hypotheses,
            languages=samples[0].languages, pipeline=worker, verify=verify,
            candidate_count=args.candidate_count, chunk_size=args.chunk_size,
            min_similarity=args.min_similarity, min_margin=args.min_margin,
            min_score=args.min_score, top_k=args.top_k,
            exclude_lines=args.exclude_line, keep_lines=args.keep_line,
        )
        report["tifa"] = provenance_paths(worker.tifa.info, root, source=source)
    report.update(lyrics_source=path_reference(str(args.lyrics.resolve()), root, source),
                  asr_source=path_reference(str(args.asr.resolve()), root, source),
                  asr={k: v for k, v in hypotheses.items() if k not in {"clips", "dataset"}}, languages=samples[0].languages,
                  device=report["tifa"]["runtime"]["device"] if verify else None,
                  elapsed_seconds=perf_counter() - start)
    for sample in samples:
        item = report["clips"][sample.key]
        if sample.transcript is not None or sample.metadata.get("transcript_read_error"):
            item["existing_transcript"] = path_reference(sample.transcript.path, root, source) if sample.transcript else None
            item["transcript_preserved"] = True
    suggestions = write_candidate_table(samples, report["clips"], table)
    if args.evidence is not None:
        write_json(args.evidence, report)
    review = root / "05-review-and-decisions/lyric-matching.json"
    with edit_tables(root) as batch:
        for sample in samples:
            item = report["clips"][sample.key]
            key = path_reference(sample.audio.path, root, source) or sample.key
            decision = None
            if item["diagnostics"] or item["status"] == "unmatched":
                decision = {k: item[k] for k in ("status", "reason", "diagnostics", "asr_text", "suggestion") if k in item}
            batch.set(review, key, decision)
    counts = Counter(item["status"] for item in report["clips"].values())
    flagged = sum(bool(item["diagnostics"]) for item in report["clips"].values())
    fallbacks = sum("tifa_could_not_verify" in item["diagnostics"] for item in report["clips"].values())
    messages = []
    if fallbacks:
        messages.append(f"Acoustic verification was unavailable for {fallbacks} clips. Kept the phone-based lyric matches.")
    if counts["unmatched"]:
        messages.append(f"No reference match for {counts['unmatched']} clips. Available ASR lyrics go to forced alignment.")
    append_processing_notes(args.processing_notes, "Lyric matching", messages)
    return {"counts": dict(counts), "prefilled": len(suggestions), "flagged": flagged,
            "table": str(table), "evidence": str(args.evidence) if args.evidence else None}, 0


def main(argv=None):
    return run(build_parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())
