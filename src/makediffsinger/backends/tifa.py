"""TIFA adapter. Imported only inside the TIFA worker."""

from dataclasses import asdict, replace
from itertools import groupby
import json
import math
from pathlib import Path

import lightning.pytorch as pl
import torch
from g2pflow import G2PConversionError, G2PGroup, G2PReading, G2PWord, PFMLError, to_pfml

from inference.api import load_config_for_inference, load_g2p_config, load_inference_model, load_voice_event_detector
from inference.callbacks import StatisticsCallback
from inference.data import AudioTextDataset
from inference.module import ForcedAlignmentInferenceModule, VoiceEventInferenceModule
from lib.config.schema import ConfigurationScope
from lib.g2p import build_pipeline_from_config
from lib.g2p_encoding import G2PEncodingError, encode_paths, resolve_phoneme
from lib.vocabulary import NUM_RESERVED_TOKENS, Vocabulary

from ..annotations import aggregate_owners, fill_gaps
from ..audio import audio_info, load_audio
from ..config import is_device_unsupported
from ..inputs import refresh_transcript
from ..types import Annotation, BatchResult, Interval, Issue, Outcome
from .common import MetadataTransferMixin, ResultCollector, configure_precision, model_record
from .tifa_statistics import pronunciation_records


def _alignment_pfml(texts, scripts, word_ids, group_ids, labels, keep):
    # Encode the retained pronunciation path with the same qualified phones as TIFA's PFML export.
    tokens = ((word_ids[i], group_ids[i], labels[i]) for i in keep)
    words = []
    for word_id, word in groupby(tokens, key=lambda token: token[0]):
        path = [G2PGroup(script=scripts[group_id - 1], phonemes=[phone for _, _, phone in group])
                for group_id, group in groupby(word, key=lambda token: token[1])]
        words.append(G2PWord(text=texts[word_id - 1], readings=[G2PReading(paths=[path])]))
    return to_pfml(words)


class DatasetInferenceModule(MetadataTransferMixin, ForcedAlignmentInferenceModule):
    pass


class TifaInputs(torch.utils.data.Dataset):
    def __init__(self, samples, config, root, vocabulary, rate, ved_rate, stop_symbols, options, pipeline=None):
        self.samples = samples
        self.config, self.root, self.vocabulary = config, root, vocabulary
        self.rate, self.ved_rate, self.stop_symbols = rate, ved_rate, stop_symbols
        self.options, self.pipeline = options, pipeline

    def __len__(self):
        return len(self.samples)

    def prepare(self, index, with_audio=True):
        sample = self.samples[index]
        try:
            sample = refresh_transcript(sample)
            info = audio_info(sample.audio, require_mono=True)
        except (OSError, ValueError, RuntimeError, UnicodeError) as exc:
            return {"skip": True, "sample": sample, "issues": [Issue("invalid_input", str(exc), "preflight")]}
        if sample.transcript is None or not sample.transcript.content.strip():
            reason = sample.metadata.get("transcript_read_error", "Missing or empty selected transcript.")
            return {"skip": True, "sample": sample, "issues": [Issue("missing_transcript", reason, "preflight")]}
        if self.pipeline is None:
            self.pipeline = build_pipeline_from_config(self.config, root_path=self.root)
        try:
            convert = self.pipeline.convert_pfml if sample.transcript.format == "pfml" else self.pipeline.convert
            words = convert(sample.transcript.content.strip(), languages=sample.languages)
        except KeyError as exc:
            message = exc.args[0] if len(exc.args) == 1 else None
            if not isinstance(message, str) or not (
                message.startswith("Script token ") and message.endswith(" not found in script-to-phoneme dict.")
            ):
                raise
            # Some converters expose a dictionary miss before wrapping it as a
            # conversion error. Keep this input failure local; do not guess a reading.
            return {"skip": True, "sample": sample,
                    "issues": [Issue("unsupported_script_token", message, "g2p")]}
        except (G2PConversionError, PFMLError) as exc:
            return {"skip": True, "sample": sample,
                    "issues": [Issue(type(exc).__name__, str(exc), "g2p")]}
        try:
            encoded, lexicon, texts = encode_paths(
                words, self.vocabulary, self.options.oov_handling,
                languages=sample.languages, stop_symbols=self.stop_symbols,
            )
        except G2PEncodingError as exc:
            return {"skip": True, "sample": sample,
                    "issues": [Issue(type(exc).__name__, str(exc), "g2p")]}
        if not encoded["paths"].any() and self.ved_rate is None:
            return {"skip": True, "sample": sample,
                    "issues": [Issue("empty_token_sequence", "No valid token sequence.", "g2p")]}
        dropped = sum(len(r.paths) for w in words for r in w.readings) - int(encoded["candidates"].sum())
        issues = [Issue("dropped_pronunciations", f"Dropped {dropped} OOV pronunciation paths.", "g2p", "warning")] if dropped else []
        sample.metadata["audio"] = info
        item = {"skip": False, "sample": sample, "issues": issues, "warning": "",
                "identifier": sample.key, "lexicon": lexicon, "texts": texts,
                **{key: torch.from_numpy(value) for key, value in encoded.items()}}
        if with_audio:
            try:
                waveform, _ = load_audio(sample.audio, self.rate)
                item["waveform"] = torch.from_numpy(waveform).float()
                item["duration"] = len(waveform) / self.rate
                if self.ved_rate:
                    ved_audio = waveform if self.ved_rate == self.rate else load_audio(sample.audio, self.ved_rate)[0]
                    item["ved_waveform"] = torch.from_numpy(ved_audio).float()
            except (OSError, ValueError, RuntimeError) as exc:
                return {"skip": True, "sample": sample, "issues": [Issue("audio_decode", str(exc), "preflight")]}
        return item

    def __getitem__(self, index):
        return self.prepare(index)

    @staticmethod
    def collate(items):
        batch = AudioTextDataset.collate(items)
        batch["records"] = [{"sample": i["sample"], "issues": i["issues"], "skip": i["skip"]} for i in items]
        return batch


class TifaAdapter:
    def __init__(self, resources, runtime, options):
        if not resources.tifa_model:
            raise ValueError("A TIFA model checkpoint is required.")
        self.resources, self.runtime, self.options = resources, runtime, options
        configure_precision(runtime)
        self.info = model_record("tifa", resources, runtime, resources.tifa_model, options=asdict(options))
        model_dir = Path(resources.tifa_model).parent
        _, config = load_config_for_inference(model_dir / "config.yaml", scope=ConfigurationScope.FA)
        self.config = config
        self.vocabulary = Vocabulary.from_file(model_dir / "vocabulary.json")
        self.g2p_config = load_g2p_config(Path(resources.g2p_config)) if resources.g2p_config else config.g2p
        if self.g2p_config is None:
            raise ValueError("The TIFA model has no G2P configuration; supply g2p_config.")
        self.g2p_root = model_dir
        if resources.g2p_config:
            path = Path(resources.g2p_config)
            self.g2p_root = path.parent if path.is_relative_to(model_dir) else Path.cwd()
        self.g2p = build_pipeline_from_config(self.g2p_config, root_path=self.g2p_root)
        self.ved_rate, self.events, self.event_tokens = None, [], {}
        if resources.ved_model:
            path = Path(resources.ved_model)
            if not path.is_file():
                raise FileNotFoundError(path)
            _, ved_config = load_config_for_inference(path.parent / "config.yaml", scope=ConfigurationScope.VED)
            self.ved_rate = ved_config.features.audio_sample_rate
            self.events = json.loads((path.parent / "vocabulary.json").read_text(encoding="utf-8"))["events"]
            for event in options.events:
                token = self.vocabulary.encode(event, None)
                if event not in self.events or token is None or token < NUM_RESERVED_TOKENS:
                    raise ValueError(f"Event {event!r} is not supported by both the FA and VED vocabularies.")
                self.event_tokens[event] = token
        self.info.update(time_resolution_seconds=config.features.timestep,
                         sample_rate=config.features.audio_sample_rate,
                         ved_checkpoint=resources.ved_model, g2p_root=str(self.g2p_root),
                         g2p_config=self.g2p_config.model_dump())
        self.module = self.trainer = None
        self._acoustic_backend = self._fusion_module = self._fusion_trainer = None
        self.collector = ResultCollector(self._convert)

    def _dataset(self, samples):
        return TifaInputs(samples, self.g2p_config, self.g2p_root, self.vocabulary,
                          self.config.features.audio_sample_rate, self.ved_rate, set(self.events),
                          self.options, self.g2p if self.runtime.num_workers == 0 else None)

    def preflight(self, samples):
        dataset = self._dataset(samples)
        outcomes = []
        for i in range(len(dataset)):
            item = dataset.prepare(i, with_audio=False)
            outcomes.append(Outcome(item["sample"], "pending" if item["skip"] else "ready", issues=item["issues"]))
        return BatchResult(outcomes)

    def pronunciations(self, texts, languages=None):
        """Return complete encodable G2P paths without loading acoustic weights."""
        converted = []
        stop_symbols = frozenset(self.events)
        for text in texts:
            try:
                words = self.g2p.convert(text, languages=languages)
            except (G2PConversionError, PFMLError) as exc:
                converted.append({"words": [], "error": str(exc)})
                continue
            except KeyError as exc:
                message = exc.args[0] if len(exc.args) == 1 else None
                if not isinstance(message, str) or not (
                    message.startswith("Script token ") and message.endswith(" not found in script-to-phoneme dict.")
                ):
                    raise
                converted.append({"words": [], "error": message})
                continue
            output = []
            for word in words:
                paths = {}
                for reading in word.readings:
                    for path in reading.paths:
                        phones = []
                        for token, symbol in (
                                resolve_phoneme(phone, word.language, self.vocabulary, languages=languages,
                                                stop_symbols=stop_symbols)
                                for group in path for phone in group.phonemes):
                            if symbol in stop_symbols:
                                continue
                            if token is None or token < NUM_RESERVED_TOKENS:
                                break
                            phones.append(symbol)
                        else:
                            if phones:
                                paths[tuple(phones)] = None
                output.append({"text": word.text, "paths": [list(path) for path in paths]})
            converted.append({"words": output})
        return converted

    def _backend(self):
        if self._acoustic_backend is None:
            self._acoustic_backend, _, _ = load_inference_model(Path(self.resources.tifa_model), scope=ConfigurationScope.FA)
        return self._acoustic_backend

    def fuse_lyrics(self, requests, languages=None, *, min_score=0.0, top_k=3):
        """Score clip-local pronunciation alternatives without aligning them."""
        if not requests:
            return []
        if self.options.score_unit == "none":
            return [{"status": "unavailable", "reason": "TIFA pronunciation scoring is disabled."} for _ in requests]
        if self._fusion_module is None:
            from .tifa_fusion import LyricFusion

            pl.seed_everything(self.runtime.seed, workers=True)
            self._fusion_module = LyricFusion(self._backend(), self)
            self._fusion_trainer = pl.Trainer(**self.runtime.trainer_options())
        dataset = [(*request, languages, min_score, top_k) for request in requests]
        loader = torch.utils.data.DataLoader(dataset, batch_size=None, num_workers=self.runtime.num_workers)
        return self._fusion_trainer.predict(self._fusion_module, dataloaders=loader)

    def _load(self):
        if self.module is not None:
            return
        pl.seed_everything(self.runtime.seed, workers=True)
        backend = self._backend()
        ved = None
        if self.resources.ved_model:
            model, events, config = load_voice_event_detector(Path(self.resources.ved_model))
            ved = VoiceEventInferenceModule(model, events, config.features)
        self.module = DatasetInferenceModule(
            backend, score_unit=self.options.score_unit, skip_penalty=self.options.skip_penalty,
            ved=ved, ved_event_tokens=self.event_tokens,
        )
        self.trainer = pl.Trainer(**self.runtime.trainer_options(), callbacks=[self.collector])

    def _predict(self, samples):
        self.collector.results = []
        dataset = self._dataset(samples)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.runtime.batch_size,
                                            num_workers=self.runtime.num_workers, shuffle=False,
                                            collate_fn=TifaInputs.collate)
        self.trainer.predict(self.module, dataloaders=loader, return_predictions=False)
        return self.collector.results

    def align(self, samples):
        self._load()
        pl.seed_everything(self.runtime.seed, workers=True)
        try:
            results = self._predict(samples)
        except (RuntimeError, ValueError) as exc:
            if is_device_unsupported(self.runtime.device, exc):
                raise
            results = list(self.collector.results)
            finished = {o.sample.key for o in results}
            for sample in samples:
                if sample.key in finished:
                    continue
                if self.runtime.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                self.trainer = pl.Trainer(**self.runtime.trainer_options(), callbacks=[self.collector])
                try:
                    results.extend(self._predict([sample]))
                except (RuntimeError, ValueError) as item_error:
                    if is_device_unsupported(self.runtime.device, item_error):
                        raise
                    results.append(Outcome(sample, "failed", issues=[Issue("inference_failed", str(item_error), "tifa")]))
        by_key = {o.sample.key: o for o in results}
        return BatchResult([by_key[s.key] for s in samples])

    def _convert(self, outputs, batch):
        records = {r["sample"].key: r for r in batch["records"]}
        converted = [Outcome(r["sample"], "pending", issues=r["issues"])
                     for r in batch["records"] if r["skip"]]
        for result in outputs:
            record = records[result["identifier"]]
            sample, timestep = record["sample"], self.config.features.timestep
            raw_spans = result["spans"].detach().cpu().tolist()
            keep = [i for i, (start, end) in enumerate(raw_spans) if end > start]
            if not keep:
                converted.append(Outcome(sample, "failed", issues=[Issue("empty_alignment", "No nonzero phone intervals.", "tifa")]))
                continue
            labels = result["phonemes"]
            prefix = f"{sample.language}/" if sample.language else None
            phones = [Interval(raw_spans[i][0] * timestep, raw_spans[i][1] * timestep,
                               labels[i][len(prefix):] if prefix and labels[i].startswith(prefix) else labels[i])
                      for i in keep]
            duration = result["spectrogram"].shape[0] * timestep
            tiers = {"phones": phones}
            word_ids = result["words"].detach().cpu().tolist()
            group_ids = result["groups"].detach().cpu().tolist()
            tiers["texts"] = aggregate_owners(phones, [word_ids[i] for i in keep], result["texts"])
            tiers["words"] = aggregate_owners(phones, [group_ids[i] for i in keep], result["group_scripts"])
            tiers = {name: fill_gaps(items, duration) for name, items in tiers.items()}
            stats = StatisticsCallback(save_dir=Path("."))
            stats.on_predict_batch_end(None, None, [result], batch)
            quality = dict(stats._metric_records[0])
            quality.pop("identifier")
            spans = result["spans"]
            n_mask = result["tokens"] != 0
            skipped = ((spans[:, 0] == spans[:, 1]) & n_mask).nonzero().flatten().tolist()
            quality["skipped_tokens"] = [{"index": i, "phone": labels[i], "span_frames": raw_spans[i]} for i in skipped]
            if stats._score_records:
                quality["pronunciations"] = pronunciation_records(stats._score_records[0]["words"])
            scores = None
            if result["scores"] is not None:
                scores = [[v if math.isfinite(v) else str(v) for v in row[:len(candidates)]]
                          for row, candidates in zip(result["scores"].detach().cpu().tolist(), result["lexicon"])]
            provenance = {"tifa": self.info, "time_resolution_seconds": timestep, "groups_verified": True,
                          "_aligned_pfml": _alignment_pfml(result["texts"], result["group_scripts"],
                                                           word_ids, group_ids, labels, keep),
                          "raw_phone_spans_frames": raw_spans,
                          "pronunciation_choices": result["choices"].detach().cpu().tolist(),
                          "pronunciation_candidates": result["lexicon"],
                          "pronunciation_scores": scores,
                          "layer_origins": {name: "tifa" for name in tiers}}
            annotation = Annotation(sample, duration, tiers, quality, provenance, "aligned")
            converted.append(Outcome(sample, "aligned", annotation, record["issues"]))
        return converted
