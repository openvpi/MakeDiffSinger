"""Shared argument groups, resource loading, and reporting for Python scripts."""

from dataclasses import asdict
from fnmatch import fnmatchcase
import json
from pathlib import Path
import sys

from .api import Pipeline
from .catalog import collection_root, dataset_root
from .config import AlignmentOptions, NoteOptions, Resources, Runtime
from .inputs import discover_pairs
from .phonology import PhoneProfiles
from .processing_notes import append_processing_notes, problem_counts, problem_messages
from .quality import quality_summary
from .review import quality_review_directory, sync_quality_review
from .selection import Selection, summarize
from .storage import load_problems, save_result
from .types import ResourceError


def add_resources(parser):
    parser.add_argument("--resources", type=Path, help="JSON resource references; paths are relative to this file.")
    for name in ("tifa-model", "ved-model", "game-model", "g2p-config", "tifa-repo", "game-repo"):
        parser.add_argument("--" + name)
    parser.add_argument("--dictionary", action="append", default=[], metavar="LANG=FILE")
    parser.add_argument("--phone-classes", action="append", default=[], metavar="LANG=FILE")


def add_runtime(parser, *, alignment=False, notes=False):
    add_resources(parser)
    parser.add_argument("--device", help="cpu, cuda, or cuda:N. Inference otherwise uses the saved device.")
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-tf32", action="store_true", help="Enable TF32 for CUDA matmul and cuDNN.")
    if alignment:
        parser.add_argument("--ved-events", default="AP")
        parser.add_argument("--no-ved", action="store_true")
        parser.add_argument("--oov-handling", choices=("raise", "discard", "force"), default="raise")
        parser.add_argument("--score-unit", choices=("levenshtein", "word", "none"), default="levenshtein")
        parser.add_argument("--skip-penalty", type=float, default=0.5)
    if notes:
        parser.add_argument("--game-language", help="Advanced GAME language conditioning. Omit for standard note extraction.")
        parser.add_argument("--seg-threshold", type=float, default=0.2)
        parser.add_argument("--seg-radius-ms", type=int, default=20)
        parser.add_argument("--est-threshold", type=float, default=0.2)
        parser.add_argument("--steps", type=int, default=8)
        parser.add_argument("--t0", type=float, default=0.0)
        parser.add_argument("--timesteps", type=float, nargs="+")


def add_identity(parser):
    parser.add_argument("--dataset-root", type=Path, help="Original input dataset when processing one subdirectory.")
    parser.add_argument("--language")
    parser.add_argument("--extended-language", action="append", default=[])
    parser.add_argument("--speaker", help="Speaker label supplied for these inputs.")
    parser.add_argument("--subset", default="")
    parser.add_argument("--sample", action="append", default=[])


def add_output(parser):
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--replace-edited", action="store_true")
    parser.add_argument("--processing-notes", type=Path, help="Append problems and actions to the shared processing notes.")


def add_selection(parser, *, quality=True):
    for name in ("sample", "speaker", "subset", "language", "status"):
        parser.add_argument("--" + name, action="append", default=[])
    if quality:
        add_quality_exclusions(parser)
        parser.add_argument("--review-output", type=Path, help="Override the stage 05 directory for quality exclusions.")
        parser.add_argument("--drop-bottom-fraction", type=float,
                            help="Exclude the worst fraction of samples with usable alignment diagnostics.")
        parser.add_argument("--percentile-group", choices=("speaker", "subset", "language"),
                            help="Apply the fraction separately to each group.")


def add_quality_exclusions(parser):
    parser.add_argument("--exclude-quality", action="append", nargs=2, default=[], metavar=("KEY", "REASON"),
                        help="Exclude an exact sample key for a documented annotation defect. Repeatable.")


def mapping(values):
    return dict(value.split("=", 1) for value in values)


def get_resources(args):
    resources = Resources.read(args.resources) if args.resources else Resources()
    for name in ("tifa_model", "ved_model", "game_model", "g2p_config", "tifa_repo", "game_repo"):
        if getattr(args, name, None):
            setattr(resources, name, str(Path(getattr(args, name)).resolve()))
    resources.dictionaries.update({key: str(Path(value).resolve()) for key, value in mapping(args.dictionary).items()})
    resources.phone_classes.update({key: str(Path(value).resolve()) for key, value in mapping(args.phone_classes).items()})
    if getattr(args, "no_ved", False):
        resources.ved_model = None
    return resources


def get_profiles(resources):
    if resources.tifa_model or resources.g2p_config or resources.dictionaries or resources.phone_classes:
        return PhoneProfiles.from_resources(resources)
    return None


def pipeline(args):
    runtime = Runtime(device=args.device, precision=args.precision, batch_size=args.batch_size,
                      num_workers=args.num_workers, seed=args.seed, allow_tf32=args.allow_tf32)
    alignment = AlignmentOptions()
    if hasattr(args, "ved_events"):
        alignment = AlignmentOptions(tuple(filter(None, args.ved_events.split(","))),
                                     args.oov_handling, args.score_unit, args.skip_penalty)
    notes = NoteOptions()
    if hasattr(args, "game_language"):
        notes = NoteOptions(args.game_language, args.seg_threshold, args.seg_radius_ms, args.est_threshold,
                            args.t0, args.steps, args.timesteps)
    return Pipeline(get_resources(args), runtime, alignment, notes)


def criteria(args):
    return Selection(args.sample, args.speaker, args.subset, args.language, args.status,
                     drop_bottom_fraction=getattr(args, "drop_bottom_fraction", None),
                     percentile_group=getattr(args, "percentile_group", None),
                     quality_exclusions=dict(getattr(args, "exclude_quality", [])))


def publish_quality_exclusions(selected, args):
    if selected.quality_filter_applied:
        destination = args.review_output or quality_review_directory(args.input)
        sync_quality_review(selected, destination)


def matches_sample(sample, patterns):
    return not patterns or any(fnmatchcase(sample.key, p) or fnmatchcase(sample.id, p) for p in patterns)


def input_samples(args):
    source = getattr(args, "dataset_root", None)
    if source is None and getattr(args, "output", None):
        bound = dataset_root(collection_root(args.output))
        if bound and Path(args.input).resolve().is_relative_to(bound):
            source = bound
    samples = discover_pairs(args.input, language=args.language, speaker=args.speaker, subset=args.subset,
                             extended_languages=args.extended_language, dataset_root=source)
    selected = [sample for sample in samples if matches_sample(sample, args.sample)]
    if not selected:
        raise ValueError("The sample selection is empty.")
    return selected


def save(result, args, *, archive_fa=False):
    storage = save_result(result, args.output, overwrite=args.overwrite, replace_edited=args.replace_edited,
                          archive_fa=archive_fa)
    if args.output.name == "03-annotations":
        from .review import sync_pending_review
        sync_pending_review(args.output, args.output.parent / "05-review-and-decisions")
    if args.processing_notes:
        append_processing_notes(args.processing_notes, "Annotation processing", problem_messages(problem_counts(result.problems)))
    return {**summarize(result), "storage": storage}, 2 if result.problems else 0


def annotation_report(selected, root, *, keep_input_order=False):
    return {"assets": summarize(selected.annotations), "quality": quality_summary(selected.annotations, keep_input_order=keep_input_order),
            "criteria": selected.criteria, "excluded": selected.excluded,
            "last_attempt_problems": [{"key": o.sample.key, "status": o.status,
                                       "issues": [asdict(i) for i in o.issues]}
                                      for o in load_problems(root).outcomes]}


def run(parser, operation, argv=None):
    args = parser.parse_args(argv)
    try:
        result, code = operation(args)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return code
    except (ValueError, OSError, RuntimeError, ImportError, ResourceError) as exc:
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
