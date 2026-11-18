"""Produce independent ASR evidence for sliced audio."""

import json
from importlib.metadata import version
from pathlib import Path
import sys
from time import perf_counter

from .audio import load_audio
from .catalog import collection_root, dataset_root, path_reference, portable_message, resolve_locator, resolve_reference
from .config import is_device_unsupported, resolve_device


# Qwen3-ASR takes language names, while the workflow uses language codes.
_QWEN_LANGUAGE_NAMES = {
    "ar": "Arabic",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fil": "Filipino",
    "fr": "French",
    "hi": "Hindi",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "mk": "Macedonian",
    "ms": "Malay",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sv": "Swedish",
    "th": "Thai",
    "tr": "Turkish",
    "vi": "Vietnamese",
    "yue": "Cantonese",
    "zh": "Chinese",
}


def _load_qwen(model, device, threads, language):
    import torch
    from qwen_asr import Qwen3ASRModel

    if language:
        language = _QWEN_LANGUAGE_NAMES.get(language.strip().lower(), language)
    if device == "cpu":
        torch.set_num_threads(threads)
    recognizer = Qwen3ASRModel.from_pretrained(
        str(model), dtype=torch.float32 if device == "cpu" else torch.bfloat16,
        device_map={"": device}, local_files_only=True, max_inference_batch_size=1,
    )

    def decode(audio):
        result = recognizer.transcribe((audio, 16000), language=language, return_time_stamps=False)
        return result[0].text.strip() if result else ""

    return decode


def transcribe_clips(samples, *, model, device=None, threads=2, language=None):
    """Transcribe clips independently with a local Qwen3-ASR model."""
    if threads < 1:
        raise ValueError("ASR threads must be positive.")
    device = requested_device = resolve_device(device)
    if device.startswith("cuda"):
        import torch
        index = int(device.partition(":")[2] or "0")
        if index >= torch.cuda.device_count():
            raise ValueError(f"The requested ASR device is unavailable: {device}")
    model = Path(model).resolve()
    if not model.is_dir():
        raise ValueError(f"ASR model directory does not exist: {model}")
    fallback_reason = None

    def load_cpu(error):
        nonlocal device, fallback_reason
        device, fallback_reason = "cpu", str(error)
        print(f"Qwen3-ASR cannot run on {requested_device}: {error}. Retrying on CPU.", file=sys.stderr)
        return _load_qwen(model, device, threads, language)

    start = perf_counter()
    try:
        decode = _load_qwen(model, device, threads, language)
    except (RuntimeError, ValueError) as exc:
        if not is_device_unsupported(device, exc):
            raise
        decode = load_cpu(exc)
    load_seconds = perf_counter() - start
    clips = {}
    for sample in samples:
        if sample.key in clips:
            raise ValueError(f"Duplicate ASR sample key: {sample.key}")
        start = perf_counter()
        record = {"audio_path": None}
        if sample.audio.path:
            clip = Path(sample.audio.path)
            delivery = collection_root(clip.parent)
            bound = dataset_root(delivery)
            source = Path(sample.metadata["dataset_root"]) if sample.metadata.get("dataset_root") else bound
            if source and clip.is_relative_to(source) and not bound:
                record.update(audio_base="dataset", audio_path=clip.relative_to(source).as_posix())
            else:
                record.update(audio_base="delivery", audio_path=path_reference(str(clip), delivery, source))
        try:
            audio, rate = load_audio(sample.audio, 16000)
        except (OSError, ValueError, RuntimeError) as exc:
            record.update(error=str(exc), text="")
        else:
            for _ in range(2):
                try:
                    record.update(audio_seconds=len(audio) / rate, text=decode(audio), device=device)
                    break
                except (OSError, RuntimeError, ValueError) as exc:
                    if not is_device_unsupported(device, exc):
                        record.update(error=str(exc), text="")
                        break
                    decode = load_cpu(exc)
        record["elapsed_seconds"] = perf_counter() - start
        if record.get("error") and sample.audio.path:
            record["error"] = portable_message(record["error"], delivery, source)
        clips[sample.key] = record
    return {"engine": "qwen", "version": version("qwen-asr"), "model": model.name,
            "device": device, "requested_device": requested_device, "device_fallback": fallback_reason,
            "threads": threads, "language": language,
            "load_seconds": load_seconds, "clips": clips}


def read_hypotheses(source, samples):
    """Read a transcription report or a separate directory of clip LAB/TXT files."""
    source = Path(source).resolve()
    if source.is_dir():
        clips = {}
        for sample in samples:
            audio = sample.metadata.get("source_relative_path", sample.key)
            for extension in (".lab", ".txt"):
                path = (source / audio).with_suffix(extension).resolve()
                if not path.is_relative_to(source):
                    raise ValueError(f"ASR path outside its root: {audio}")
                if path.is_file():
                    clips[sample.key] = {"text": path.read_text(encoding="utf-8-sig").strip()}
                    break
        return {"engine": "external", "clips": clips}
    report = json.loads(source.read_text(encoding="utf-8-sig"))
    if not isinstance(report, dict) or not isinstance(report.get("clips"), dict):
        raise ValueError("ASR JSON needs a clips object keyed by sample key.")
    for record in report["clips"].values():
        if not isinstance(record, dict) or not isinstance(record.get("text"), str):
            raise ValueError("Each ASR clip record needs a text string.")
    report_dataset = (resolve_locator(source.parent, report["dataset"], local_key=source)
                      if "dataset" in report else dataset_root(collection_root(source.parent)))
    for sample in samples:
        original = sample.metadata.get("dataset_root")
        if report_dataset and original and report_dataset != Path(original):
            raise ValueError("ASR hypotheses belong to a different input dataset.")
        row = report["clips"].get(sample.key, {})
        recorded = row.get("audio_path")
        if recorded and sample.audio.path:
            clip = Path(sample.audio.path)
            delivery = collection_root(clip.parent)
            if row.get("audio_base") == "dataset":
                path = Path(original or delivery) / recorded
            elif row.get("audio_base") == "delivery":
                path = Path(resolve_reference(recorded, delivery, Path(original) if original else None))
            else:
                path = Path(recorded)
            if path.resolve() != clip:
                raise ValueError(f"ASR report points to a different clip for {sample.key}.")
    return report
