"""Extract one channel or the stereo average over an explicit source interval."""

import argparse
from pathlib import Path

from makediffsinger.audio import audio_info, load_audio, mono_track, write_mono_track
from makediffsinger.script_utils import run
from makediffsinger.types import AudioSource


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Stereo recording.")
    parser.add_argument("--range-samples", nargs=2, type=int, metavar=("START", "END"), required=True,
                        help="Original sample interval with an exclusive end.")
    parser.add_argument("--channel", choices=("left", "right", "average"), required=True)
    parser.add_argument("--output", type=Path, required=True, help="New mono WAV or FLAC file.")
    return parser


def execute(args):
    start, end = args.range_samples
    source = AudioSource(str(args.input.resolve(strict=True)), start_sample=start, end_sample=end)
    info = audio_info(source)
    if info["channels"] != 2:
        raise ValueError("Channel extraction requires stereo audio.")
    waveform, rate = load_audio(source, mono=False)
    write_mono_track(args.output, mono_track(waveform, args.channel), rate, info["subtype"])
    return {"source": info["path"], "output": str(args.output), "channel": args.channel, "start_sample": start,
            "end_sample": end, "sample_rate": rate}, 0


if __name__ == "__main__":
    raise SystemExit(run(build_parser(), execute))
