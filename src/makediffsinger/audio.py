"""Read model audio without modifying sources; replay source sample intervals."""

from itertools import groupby
from pathlib import Path
import shutil
import wave

import numpy as np
import soundfile as sf

from .types import AudioSource


MIN_STEREO_DIFFERENCE_SECONDS = 5.0


def _decode(path: str) -> tuple[np.ndarray, int]:
    if Path(path).suffix.lower() == ".opus":
        try:
            import opuscodec
        except ImportError as exc:
            raise ImportError("Opus decoding needs the optional 'opuscodec' dependency.") from exc
        decoder = opuscodec.OpusBufferedDecoder()
        audio = decoder.decode(Path(path).read_bytes()).astype(np.float32) / 32768.0
        return audio, 48000
    try:
        return sf.read(path, dtype="float32", always_2d=False)
    except sf.LibsndfileError:
        from audioread.exceptions import DecodeError
        import librosa
        try:
            waveform, rate = librosa.load(path, sr=None, mono=False)
        except DecodeError as exc:
            raise ValueError(f"No available decoder could read audio: {path}") from exc
        return waveform.T if waveform.ndim == 2 else waveform, rate


def audio_info(source: AudioSource, *, require_mono=False) -> dict:
    if source.path is None and source.waveform is None:
        raise FileNotFoundError("No source audio was supplied for this annotation.")
    if source.waveform is not None:
        frames, rate = len(source.waveform), source.sample_rate
        channels = source.waveform.shape[1] if source.waveform.ndim == 2 else 1
        subtype = str(source.waveform.dtype)
    else:
        try:
            info = sf.info(source.path)
            frames, rate, channels, subtype = info.frames, info.samplerate, info.channels, info.subtype
        except sf.LibsndfileError:
            audio, rate = _decode(source.path)
            frames = len(audio)
            channels = audio.shape[1] if audio.ndim == 2 else 1
            subtype = "decoded"
    if require_mono and channels != 1:
        raise ValueError(f"Expected mono audio, found {channels} channels. Use scripts/prepare/prepare_mono.py first.")
    end = source.end_sample if source.end_sample is not None else frames
    if not 0 <= source.start_sample < end <= frames:
        raise ValueError(f"Invalid original sample range [{source.start_sample}, {end}) for {frames} frames.")
    if source.sample_rate is not None and source.sample_rate != rate:
        raise ValueError(f"Source sample-rate mismatch: expected {source.sample_rate}, found {rate}.")
    return {"path": source.path, "sample_rate": rate, "channels": channels, "frames": frames,
            "subtype": subtype, "start_sample": source.start_sample, "end_sample": end,
            "source_seconds": frames / rate, "clip_seconds": (end - source.start_sample) / rate}


def load_audio(source: AudioSource, sample_rate: int | None = None, *, mono=True) -> tuple[np.ndarray, int]:
    info = audio_info(source, require_mono=mono)
    if source.waveform is not None:
        data = source.waveform
    else:
        data, _ = _decode(source.path)
    data = np.array(data[info["start_sample"]:info["end_sample"]], dtype=np.float32, copy=True)
    if not np.isfinite(data).all():
        raise ValueError("Audio contains non-finite samples.")
    if mono and data.ndim == 2:
        data = data[:, 0]
    rate = info["sample_rate"]
    if sample_rate is not None and sample_rate != rate:
        import librosa
        data = librosa.resample(data, orig_sr=rate, target_sr=sample_rate, axis=0)
        rate = sample_rate
    return data, rate


def _stereo_windows(frames, rate):
    window = min(frames, max(1, round(rate * .1)))
    last = frames - window
    starts = list(range(0, last + 1, window))
    if starts[-1] != last:
        starts.append(last)
    return window, starts


def mix_stereo(waveform: np.ndarray, rate: int, *, cancellation_db=12.0):
    """Try the average and check for strong cancellation over active 100 ms windows."""
    if not np.isfinite(cancellation_db) or cancellation_db <= 0:
        raise ValueError("The cancellation threshold must be positive and finite.")
    left, right = waveform.T
    average = left * .5 + right * .5
    window, starts = _stereo_windows(len(average), rate)
    powers = []
    for start in starts:
        end = start + window
        frame = waveform[start:end]
        mixed = average[start:end]
        powers.append((np.einsum("ij,ij->", frame, frame, dtype=np.float64) / 2,
                       np.einsum("i,i->", mixed, mixed, dtype=np.float64)))
    peak = max(reference for reference, _ in powers)
    active = [(reference, mixed) for reference, mixed in powers if reference > 0 and reference >= peak * 1e-4]
    ratios = [mixed / reference for reference, mixed in active]
    worst = min(ratios, default=1.0)
    loss = max(0.0, float(-10 * np.log10(worst))) if worst > 0 else None
    fallback = worst == 0 or loss >= cancellation_db
    return (left if fallback else average), {
        "method": "first_channel" if fallback else "average",
        "worst_mix_loss_db": loss,
        "active_windows": len(active),
    }


def _different_stereo_content(frame, rate):
    taper = np.hanning(len(frame) + 2)[1:-1, None]
    centered = (frame - frame.mean(axis=0)) * taper
    energy = np.linalg.norm(centered, axis=0).prod()
    if not energy or abs(centered[:, 0] @ centered[:, 1]) >= .55 * energy:
        return False
    spectrum = np.fft.rfft(centered, n=2 * len(frame), axis=0)
    magnitude = np.abs(spectrum)
    if magnitude[:, 0] @ magnitude[:, 1] >= .8 * np.linalg.norm(magnitude, axis=0).prod():
        return False
    correlation = np.fft.irfft(spectrum[:, 0].conj() * spectrum[:, 1])
    lag = min(len(frame) - 1, round(rate * .02))
    nearby_peak = np.abs(correlation[:lag + 1]).max()
    if lag:
        nearby_peak = max(nearby_peak, np.abs(correlation[-lag:]).max())
    return nearby_peak < .55 * energy


def compare_stereo(waveform: np.ndarray, rate: int, *, cancellation_db=12.0,
                   min_difference_seconds=MIN_STEREO_DIFFERENCE_SECONDS):
    """Describe shared and distinct intervals in original stereo sample coordinates."""
    if waveform.ndim != 2 or waveform.shape[1] != 2 or not len(waveform) or rate <= 0:
        raise ValueError("Stereo comparison requires nonempty two-channel audio and a positive sample rate.")
    if not np.isfinite(cancellation_db) or cancellation_db <= 0:
        raise ValueError("The cancellation threshold must be positive and finite.")
    if not np.isfinite(min_difference_seconds) or min_difference_seconds <= 0:
        raise ValueError("The minimum difference duration must be positive and finite.")
    window = max(1, round(rate * .1))
    starts = range(0, len(waveform), window)
    powers = np.array([waveform[start:start + window].var(axis=0, dtype=np.float64) for start in starts])
    active = (powers > 0) & (powers >= powers.max() * 1e-4)
    shared = ("average",)
    decisions = [None] * len(starts)
    if active.any(axis=0).all():
        for index, start in enumerate(starts):
            channels = tuple(name for name, audible in zip(("left", "right"), active[index]) if audible)
            if channels:
                frame = waveform[start:start + window]
                decisions[index] = channels if len(channels) == 1 or _different_stereo_content(frame, rate) else shared

    classified = []
    start = 0
    for distinct, run in groupby(decisions, key=lambda decision: decision not in (None, shared)):
        choices = list(run)
        end = min(start + len(choices) * window, len(waveform))
        if distinct and end - start >= min_difference_seconds * rate:
            classified.extend((start, channels) for channels in choices)
        else:
            classified.extend(None if channels is None else (-1, shared) for channels in choices)
        start = end

    # Attach quiet windows after qualifying the distinct runs, without joining separate runs.
    previous = next((decision for decision in classified if decision is not None), (-1, shared))
    for index, decision in enumerate(classified):
        previous = decision or previous
        classified[index] = previous

    regions = []
    start = 0
    for (_, channels), run in groupby(classified):
        end = min(start + sum(1 for _ in run) * window, len(waveform))
        region = {"start_sample": start, "end_sample": end, "channels": list(channels)}
        if channels == shared:
            _, mix = mix_stereo(waveform[start:end], rate, cancellation_db=cancellation_db)
            region.update(mix)
            if mix["method"] == "first_channel":
                region["channels"] = ["left"]
        else:
            region["method"] = "split_channels" if len(channels) == 2 else "single_active_channel"
        regions.append(region)
        start = end
    return {"method": regions[0]["method"] if len(regions) == 1 else "regional_channels",
            "comparison_window_samples": window, "regions": regions}


def mono_track(waveform, channel):
    left, right = waveform.T
    if channel == "average":
        return left * .5 + right * .5
    if channel == "left":
        return left
    if channel == "right":
        return right
    raise ValueError("Choose left, right, or average.")


def write_mono_track(destination, samples, rate, subtype):
    kind = {".wav": "WAV", ".flac": "FLAC"}.get(destination.suffix.lower())
    if kind is None:
        raise ValueError("Choose a WAV or FLAC destination.")
    if destination.exists():
        raise FileExistsError(destination)
    lossless = {"PCM_S8", "PCM_U8", "PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}
    if subtype not in lossless or not sf.check_format(kind, subtype):
        subtype = "PCM_24" if kind == "FLAC" else "FLOAT"
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        try:
            sf.write(stream, samples, rate, format=kind, subtype=subtype)
        except (OSError, ValueError, RuntimeError):
            stream.close()
            destination.unlink(missing_ok=True)
            raise


def replay_audio(source: AudioSource, destination: str | Path) -> Path:
    """Write an independent WAV, preserving PCM bytes for PCM WAV sources."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    info = audio_info(source)
    if source.path is not None and Path(source.path).suffix.lower() == ".wav":
        if info["start_sample"] == 0 and info["end_sample"] == info["frames"]:
            shutil.copy2(source.path, destination)
            return destination
        try:
            with wave.open(source.path, "rb") as stream:
                params = stream.getparams()
                stream.setpos(info["start_sample"])
                count = info["end_sample"] - info["start_sample"]
                pcm = stream.readframes(count)
            if len(pcm) != count * params.nchannels * params.sampwidth:
                raise ValueError("Source recording ended before the requested slice.")
            with wave.open(str(destination), "wb") as stream:
                stream.setparams(params)
                stream.writeframes(pcm)
            return destination
        except wave.Error:
            pass
    data, rate = load_audio(source, mono=False)
    subtype = info["subtype"] if info["subtype"] in {"PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"} else "FLOAT"
    sf.write(destination, data, rate, subtype=subtype)
    return destination
