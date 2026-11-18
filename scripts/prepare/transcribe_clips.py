"""Run local Qwen3-ASR on pre-cut clips and save independent hypotheses."""

import argparse
import contextlib
import json
from pathlib import Path
import sys

from makediffsinger.lyric_asr import transcribe_clips
from makediffsinger.catalog import bind_dataset, collection_root, dataset_locator, dataset_root, sample_dataset
from makediffsinger.script_utils import add_identity, input_samples, run


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--model", required=True, type=Path, help="Local Qwen3-ASR model directory.")
    parser.add_argument("--device", help="cpu, cuda, or cuda:N. Otherwise uses the saved device.")
    parser.add_argument("--threads", default=2, type=int)
    parser.add_argument("--asr-language", help="Override the Qwen3-ASR recognition language.")
    parser.add_argument("--output", required=True, type=Path)
    add_identity(parser)
    return parser


def execute(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    samples = input_samples(args)
    source = sample_dataset(samples)
    directory = args.output.resolve().parent
    delivery = collection_root(directory)
    shared = dataset_root(delivery) is not None or ".work" in directory.relative_to(delivery).parts
    if source and shared:
        bind_dataset(delivery, source)
    with contextlib.redirect_stdout(sys.stderr):
        report = transcribe_clips(samples, model=args.model, device=args.device,
                                  threads=args.threads, language=args.asr_language or args.language)
    if source and not shared:
        report["dataset"] = dataset_locator(directory, source, local_key=args.output.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
    missing = sum(not r["text"].strip() for r in report["clips"].values())
    return {"clips": len(samples), "empty_or_failed": missing, "output": str(args.output),
            "device": report["device"], "device_fallback": report["device_fallback"]}, 0


def main(argv=None):
    return run(build_parser(), execute, argv)


if __name__ == "__main__":
    raise SystemExit(main())
