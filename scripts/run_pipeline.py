"""Run preflight, alignment, notes, and validation with recoverable stage outputs."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

from makediffsinger.catalog import bind_dataset, dataset_root, input_dataset
from makediffsinger.lyric_handoff import alignment_inputs
from makediffsinger.processing_notes import append_processing_notes, problem_counts, problem_messages
from makediffsinger.quality import write_overview
from makediffsinger.review import sync_pending_review
from makediffsinger.script_utils import add_identity, add_output, add_runtime, annotation_report, get_resources, input_samples, pipeline, run
from makediffsinger.selection import Selection, select
from makediffsinger.storage import load_annotations, save_result
from makediffsinger.types import BatchResult
from makediffsinger.validation import validate


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="+", type=Path, help="One or more independent input datasets with paired clips.")
    add_identity(parser)
    add_output(parser)
    add_runtime(parser, alignment=True, notes=True)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--report-output", type=Path, help="Native alignment statistics destination (default: 04-quality).")
    parser.add_argument("--plots", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pfml", action="store_true", help="Include aligned pronunciation PFML beside the final TextGrids.")
    return parser


def _run_dataset(args, session):
    samples = alignment_inputs(input_samples(args), args.output)
    work_root, annotation_root = args.output / ".work", args.output / "03-annotations"
    if not annotation_root.exists() and any((args.output / name).is_dir() for name in ("fa", "annotations")):
        work_root, annotation_root = args.output, args.output / "annotations"
    fa_root = work_root / "fa"
    counts, fa_counts = Counter(), Counter()
    problems = Counter()
    summary = {"total": len(samples), "fa_output": str(fa_root), "annotation_output": str(annotation_root)}
    storage_options = {"overwrite": args.overwrite, "replace_edited": args.replace_edited}
    for offset in range(0, len(samples), args.chunk_size):
        chunk = samples[offset:offset + args.chunk_size]
        checked = session.preflight(chunk)
        aligned = session.align(checked.ready)
        fa_counts.update(save_result(aligned, fa_root, archive_fa=True, **storage_options))
        noted = session.annotate_notes(aligned.annotations)
        for outcome in noted.outcomes:
            if outcome.ok and outcome.annotation is not None:
                issues = validate(outcome.annotation, required=("texts", "words", "phones", "notes"),
                                  profiles=session.profiles, check_audio=True)
                outcome.issues.extend(i for i in issues if i not in outcome.issues)
                if any(i.severity == "error" for i in issues):
                    outcome.status = "failed"
        result = BatchResult(checked.problems + aligned.problems + noted.outcomes)
        for outcome in noted.outcomes:
            for issue in outcome.issues:
                if issue.severity == "warning":
                    summary.setdefault("warnings", []).append({"key": outcome.sample.key, "message": issue.message})
        counts.update(save_result(result, annotation_root, archive_fa=args.pfml, **storage_options))
        problems.update(problem_counts(result.problems))
        summary.update(processed=offset + len(chunk), counts=dict(counts), fa_counts=dict(fa_counts))
        print(json.dumps({"processed": summary["processed"], "total": len(samples), "counts": dict(counts)}),
              file=sys.stderr, flush=True)
    current_review = sync_pending_review(annotation_root, args.output / "05-review-and-decisions")
    summary["pending_review"] = len(current_review["pending"])
    if current_review.get("return_to_alignment"):
        summary["return_to_alignment"] = [row["key"] for row in current_review["return_to_alignment"]]
    report_output = args.report_output or args.output / "04-quality"
    report = annotation_report(select(load_annotations(annotation_root), Selection()), annotation_root, keep_input_order=True)
    try:
        measured = write_overview(report, report_output, tifa_repo=get_resources(args).repo("tifa"))
    except (OSError, RuntimeError) as exc:
        summary["statistics_error"] = str(exc)
        measured = 0
    summary.update(quality_output=str(report_output), diagnosed=measured)
    if args.processing_notes:
        append_processing_notes(args.processing_notes, "Annotation pipeline", problem_messages(problems))
    incomplete = any(counts[name] for name in ("pending", "failed", "conflict")) or summary.get("return_to_alignment")
    return summary, 2 if incomplete or "statistics_error" in summary else 0


def _destination(parent, name, source, reserved):
    serial = 1
    while True:
        path = parent / (name if serial == 1 else f"{name}_{serial}")
        if path not in reserved:
            if not path.exists():
                return path
            if path.is_dir() and (not any(path.iterdir()) or dataset_root(path.resolve()) == source):
                return path
        serial += 1


def execute(args):
    if args.chunk_size < 1:
        raise ValueError("--chunk-size must be positive.")
    inputs = list(dict.fromkeys(p.resolve(strict=True) for p in args.input))
    if len(inputs) > 1 and args.dataset_root:
        raise ValueError("Each input in a multi-dataset run supplies its own dataset root.")
    deliveries = []
    for path in inputs:
        job = argparse.Namespace(**vars(args))
        job.input = path
        job.dataset_root = (args.dataset_root or input_dataset(path)).resolve()
        if len(inputs) > 1:
            name = path.name if path.is_dir() else path.stem
            job.output = _destination(args.output, name, job.dataset_root, {other.output for other in deliveries})
            if args.report_output:
                job.report_output = args.report_output / job.output.name
            if args.processing_notes:
                job.processing_notes = args.processing_notes.parent / job.output.name / args.processing_notes.name
        bind_dataset(job.output, job.dataset_root)
        deliveries.append(job)
    reports, codes = [], []
    with pipeline(args) as session:
        if len(deliveries) == 1:
            return _run_dataset(deliveries[0], session)
        for job in deliveries:
            try:
                report, code = _run_dataset(job, session)
            except (ValueError, OSError, RuntimeError, ImportError) as exc:
                report, code = {"error": type(exc).__name__, "message": str(exc)}, 1
            reports.append({"input": str(job.input), "output": str(job.output), **report})
            codes.append(code)
    return {"datasets": reports}, 1 if 1 in codes else max(codes)


def main(argv=None):
    return run(build_parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())
